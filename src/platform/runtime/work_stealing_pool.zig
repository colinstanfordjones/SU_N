//! Persistent worker pool with per-worker queues and basic work stealing.

const std = @import("std");
const builtin = @import("builtin");
const work_queue_mod = @import("work_queue.zig");
const task_arena_mod = @import("task_arena.zig");

const Inner = struct {
    const Self = @This();

    const Worker = struct {
        queue: work_queue_mod.WorkQueue,
    };

    allocator: std.mem.Allocator,
    workers: []Worker,
    threads: []std.Thread,
    next_queue: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pending_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    wake_mutex: std.Thread.Mutex = .{},
    wake_cond: std.Thread.Condition = .{},

    pub const Task = work_queue_mod.WorkQueue.Task;

    pub fn init(self: *Self, allocator: std.mem.Allocator, n_threads: ?usize) !void {
        if (builtin.single_threaded) {
            self.* = .{
                .allocator = allocator,
                .workers = &[_]Worker{},
                .threads = &[_]std.Thread{},
            };
            return;
        }

        const thread_count = n_threads orelse (std.Thread.getCpuCount() catch 4);
        const workers = try allocator.alloc(Worker, thread_count);
        errdefer allocator.free(workers);

        for (workers) |*worker| {
            worker.* = .{ .queue = work_queue_mod.WorkQueue.init(allocator) };
        }

        const threads = try allocator.alloc(std.Thread, thread_count);
        errdefer allocator.free(threads);

        self.* = .{
            .allocator = allocator,
            .workers = workers,
            .threads = threads,
        };

        for (threads, 0..) |*thread, idx| {
            thread.* = try std.Thread.spawn(.{ .allocator = allocator }, workerMain, .{ self, idx });
        }
    }

    pub fn deinit(self: *Self) void {
        if (builtin.single_threaded) return;

        self.stop.store(true, .release);
        self.wake_mutex.lock();
        self.wake_cond.broadcast();
        self.wake_mutex.unlock();

        for (self.threads) |thread| {
            thread.join();
        }

        for (self.workers) |*worker| {
            worker.queue.deinit();
        }

        self.allocator.free(self.threads);
        self.allocator.free(self.workers);
    }

    pub fn submit(self: *Self, task: Task) void {
        if (builtin.single_threaded or self.workers.len == 0) {
            task.run(task.ctx);
            return;
        }

        const idx = self.next_queue.fetchAdd(1, .monotonic) % self.workers.len;
        self.workers[idx].queue.push(task) catch {
            task.run(task.ctx);
            return;
        };

        _ = self.pending_tasks.fetchAdd(1, .release);
        self.signalWork();
    }

    fn signalWork(self: *Self) void {
        self.wake_mutex.lock();
        defer self.wake_mutex.unlock();
        self.wake_cond.signal();
    }

    fn workerMain(self: *Self, index: usize) void {
        while (true) {
            if (self.stop.load(.acquire)) return;

            if (self.workers[index].queue.pop()) |task| {
                _ = self.pending_tasks.fetchSub(1, .acq_rel);
                task.run(task.ctx);
                continue;
            }

            if (self.stealWork(index)) continue;

            self.waitForWork();
        }
    }

    fn stealWork(self: *Self, thief: usize) bool {
        const count = self.workers.len;
        var offset: usize = 1;
        while (offset < count) : (offset += 1) {
            const victim = (thief + offset) % count;
            if (self.workers[victim].queue.pop()) |task| {
                _ = self.pending_tasks.fetchSub(1, .acq_rel);
                task.run(task.ctx);
                return true;
            }
        }
        return false;
    }

    fn waitForWork(self: *Self) void {
        self.wake_mutex.lock();
        defer self.wake_mutex.unlock();
        while (self.pending_tasks.load(.acquire) == 0 and !self.stop.load(.acquire)) {
            self.wake_cond.wait(&self.wake_mutex);
        }
    }
};

pub const WorkStealingPool = struct {
    const Self = @This();

    pub const Task = Inner.Task;

    allocator: std.mem.Allocator,
    inner: *Inner,

    pub fn init(allocator: std.mem.Allocator, n_threads: ?usize) !Self {
        const inner = try allocator.create(Inner);
        errdefer allocator.destroy(inner);
        try inner.init(allocator, n_threads);
        return Self{
            .allocator = allocator,
            .inner = inner,
        };
    }

    pub fn deinit(self: *Self) void {
        self.inner.deinit();
        self.allocator.destroy(self.inner);
    }

    pub fn submit(self: *Self, task: Task) void {
        self.inner.submit(task);
    }

    pub fn workerCount(self: *const Self) usize {
        if (builtin.single_threaded or self.inner.workers.len == 0) return 1;
        return self.inner.workers.len;
    }
};

pub const TaskGroup = struct {
    const Self = @This();

    arena: task_arena_mod.TaskArena,
    wait_group: std.Thread.WaitGroup = .{},
    waited: bool = false,

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .arena = try task_arena_mod.TaskArena.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.waited) {
            self.wait_group.wait();
            self.waited = true;
        }
        self.arena.deinit();
    }

    pub fn wait(self: *Self) void {
        if (!self.waited) {
            self.wait_group.wait();
            self.waited = true;
        }
    }

    pub fn reset(self: *Self) void {
        self.wait_group.reset();
        self.arena.reset();
        self.waited = false;
    }

    pub fn submit(self: *Self, pool: *WorkStealingPool, task: WorkStealingPool.Task) void {
        self.wait_group.start();

        const wrapper = self.arena.allocatorHandle().create(TaskWrapper) catch {
            task.run(task.ctx);
            self.wait_group.finish();
            return;
        };

        wrapper.* = .{
            .group = self,
            .task = task,
        };

        pool.submit(.{
            .run = TaskWrapper.run,
            .ctx = wrapper,
        });
    }

    const TaskWrapper = struct {
        group: *Self,
        task: WorkStealingPool.Task,

        fn run(ctx: *anyopaque) void {
            const wrapper: *TaskWrapper = @ptrCast(@alignCast(ctx));
            wrapper.task.run(wrapper.task.ctx);
            wrapper.group.wait_group.finish();
        }
    };
};

//! Work queue with a simple mutex guard.
//!
//! This is a minimal building block for future work-stealing schedulers.

const std = @import("std");

pub const WorkQueue = struct {
    const Self = @This();

    pub const Task = struct {
        run: *const fn (ctx: *anyopaque) void,
        ctx: *anyopaque,
    };

    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    items: std.ArrayListUnmanaged(Task) = .{},
    head: usize = 0,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }

    pub fn initWithCapacity(allocator: std.mem.Allocator, capacity: usize) !Self {
        var queue = Self{
            .allocator = allocator,
        };
        try queue.items.ensureTotalCapacityPrecise(allocator, capacity);
        return queue;
    }

    pub fn deinit(self: *Self) void {
        self.items.deinit(self.allocator);
    }

    pub fn reset(self: *Self) void {
        self.items.clearRetainingCapacity();
        self.head = 0;
    }

    pub fn push(self: *Self, task: Task) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.items.append(self.allocator, task);
    }

    pub fn pop(self: *Self) ?Task {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.head >= self.items.items.len) return null;

        const task = self.items.items[self.head];
        self.head += 1;

        if (self.head == self.items.items.len) {
            self.items.clearRetainingCapacity();
            self.head = 0;
        } else if (self.head > 64 and self.head * 2 >= self.items.items.len) {
            const remaining = self.items.items[self.head..];
            std.mem.copyForwards(Task, self.items.items[0..remaining.len], remaining);
            self.items.items.len = remaining.len;
            self.head = 0;
        }

        return task;
    }

    pub fn len(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.items.items.len - self.head;
    }
};

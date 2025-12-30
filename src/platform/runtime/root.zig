//! Runtime-facing utilities (threading and scheduling).

pub const work_queue = @import("work_queue.zig");
pub const task_arena = @import("task_arena.zig");
pub const work_stealing_pool = @import("work_stealing_pool.zig");

pub const WorkQueue = work_queue.WorkQueue;
pub const TaskArena = task_arena.TaskArena;
pub const WorkStealingPool = work_stealing_pool.WorkStealingPool;
pub const TaskGroup = work_stealing_pool.TaskGroup;

test {
    _ = work_queue;
    _ = task_arena;
    _ = work_stealing_pool;
}

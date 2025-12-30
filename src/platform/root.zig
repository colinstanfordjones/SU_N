//! Platform utilities for runtime and IO integration.
//!
//! This module centralizes OS-facing helpers like work-stealing pools/work queues,
//! MPI wrappers, and file operations or checkpointing to keep them out of the domain modules.

pub const runtime = @import("runtime/root.zig");
pub const io = @import("io/root.zig");
pub const checkpoint = @import("checkpoint/root.zig");
pub const mpi = @import("mpi/root.zig");

pub const WorkQueue = runtime.WorkQueue;
pub const TaskArena = runtime.TaskArena;
pub const WorkStealingPool = runtime.WorkStealingPool;
pub const TaskGroup = runtime.TaskGroup;

pub const FileOps = io.FileOps;
pub const Checkpoint = checkpoint;
pub const Mpi = mpi;

test {
    _ = runtime;
    _ = io;
    _ = checkpoint;
    _ = mpi;
}

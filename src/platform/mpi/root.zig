//! MPI wrapper (MPICH) with a disabled stub when MPI is not enabled.

const build_options = @import("build_options");

pub const enabled = build_options.enable_mpi;

pub const types = @import("types.zig");

const Impl = if (enabled) @import("mpich.zig") else @import("disabled.zig");

pub const Comm = Impl.Comm;
pub const Request = Impl.Request;
pub const ThreadLevel = types.ThreadLevel;
pub const preferred_thread_level = Impl.preferred_thread_level;
pub const any_source = Impl.any_source;
pub const any_tag = Impl.any_tag;

pub const initThreaded = Impl.initThreaded;
pub const initSerialized = Impl.initSerialized;
pub const finalize = Impl.finalize;
pub const commWorld = Impl.commWorld;
pub const rank = Impl.rank;
pub const size = Impl.size;
pub const barrier = Impl.barrier;
pub const bcastBytes = Impl.bcastBytes;
pub const isend = Impl.isend;
pub const irecv = Impl.irecv;
pub const irecvAny = Impl.irecvAny;
pub const waitAll = Impl.waitAll;
pub const allreduceSum = Impl.allreduceSum;
pub const allgatherI32 = Impl.allgatherI32;
pub const allgatherBytes = Impl.allgatherBytes;
pub const allgatherVBytes = Impl.allgatherVBytes;

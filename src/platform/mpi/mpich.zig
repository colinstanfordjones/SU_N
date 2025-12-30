const std = @import("std");
const types = @import("types.zig");

const c = @cImport({
    @cInclude("mpi.h");
});

pub const enabled = true;

pub const Comm = struct {
    handle: c.MPI_Comm,
};

pub const Request = struct {
    handle: c.MPI_Request,
};

pub const ThreadLevel = types.ThreadLevel;

pub const preferred_thread_level: ThreadLevel = .serialized;
pub const any_source: i32 = c.MPI_ANY_SOURCE;
pub const any_tag: i32 = c.MPI_ANY_TAG;

fn toMpiThreadLevel(level: ThreadLevel) c_int {
    return switch (level) {
        .single => c.MPI_THREAD_SINGLE,
        .funneled => c.MPI_THREAD_FUNNELED,
        .serialized => c.MPI_THREAD_SERIALIZED,
        .multiple => c.MPI_THREAD_MULTIPLE,
    };
}

fn fromMpiThreadLevel(level: c_int) ThreadLevel {
    return switch (level) {
        c.MPI_THREAD_SINGLE => .single,
        c.MPI_THREAD_FUNNELED => .funneled,
        c.MPI_THREAD_SERIALIZED => .serialized,
        c.MPI_THREAD_MULTIPLE => .multiple,
        else => .single,
    };
}

fn check(rc: c_int) !void {
    if (rc != c.MPI_SUCCESS) return types.Error.MpiFailure;
}

pub fn initThreaded(required: ThreadLevel) !ThreadLevel {
    const required_level = toMpiThreadLevel(required);

    var initialized: c_int = 0;
    _ = c.MPI_Initialized(&initialized);
    if (initialized != 0) {
        var provided: c_int = 0;
        _ = c.MPI_Query_thread(&provided);
        if (provided < required_level) return types.Error.MpiThreadLevelUnsupported;
        return fromMpiThreadLevel(provided);
    }
    var provided_level: c_int = 0;

    try check(c.MPI_Init_thread(null, null, required_level, &provided_level));
    if (provided_level < required_level) {
        return types.Error.MpiThreadLevelUnsupported;
    }
    return fromMpiThreadLevel(provided_level);
}

pub fn initSerialized() !ThreadLevel {
    return initThreaded(.serialized);
}

pub fn finalize() void {
    var finalized: c_int = 0;
    _ = c.MPI_Finalized(&finalized);
    if (finalized != 0) return;

    var initialized: c_int = 0;
    _ = c.MPI_Initialized(&initialized);
    if (initialized == 0) return;

    _ = c.MPI_Finalize();
}

pub fn commWorld() Comm {
    return .{ .handle = c.MPI_COMM_WORLD };
}

pub fn rank(comm: Comm) !i32 {
    var out: c_int = 0;
    try check(c.MPI_Comm_rank(comm.handle, &out));
    return @intCast(out);
}

pub fn size(comm: Comm) !i32 {
    var out: c_int = 0;
    try check(c.MPI_Comm_size(comm.handle, &out));
    return @intCast(out);
}

pub fn barrier(comm: Comm) !void {
    try check(c.MPI_Barrier(comm.handle));
}

pub fn bcastBytes(comm: Comm, buf: []u8, root: i32) !void {
    try check(c.MPI_Bcast(
        buf.ptr,
        @intCast(buf.len),
        c.MPI_BYTE,
        @intCast(root),
        comm.handle,
    ));
}

pub fn isend(comm: Comm, buf: []const u8, dest: i32, tag: types.Tag) !Request {
    var req: c.MPI_Request = undefined;
    try check(c.MPI_Isend(
        @constCast(buf.ptr),
        @intCast(buf.len),
        c.MPI_BYTE,
        @intCast(dest),
        @intCast(tag),
        comm.handle,
        &req,
    ));
    return .{ .handle = req };
}

pub fn irecv(comm: Comm, buf: []u8, src: i32, tag: types.Tag) !Request {
    var req: c.MPI_Request = undefined;
    try check(c.MPI_Irecv(
        buf.ptr,
        @intCast(buf.len),
        c.MPI_BYTE,
        @intCast(src),
        @intCast(tag),
        comm.handle,
        &req,
    ));
    return .{ .handle = req };
}

pub fn irecvAny(comm: Comm, buf: []u8, tag: types.Tag) !Request {
    var req: c.MPI_Request = undefined;
    try check(c.MPI_Irecv(
        buf.ptr,
        @intCast(buf.len),
        c.MPI_BYTE,
        c.MPI_ANY_SOURCE,
        @intCast(tag),
        comm.handle,
        &req,
    ));
    return .{ .handle = req };
}

pub fn waitAll(reqs: []Request) !void {
    if (reqs.len == 0) return;

    const statuses = try std.heap.page_allocator.alloc(c.MPI_Status, reqs.len);
    defer std.heap.page_allocator.free(statuses);

    var raw_reqs = try std.heap.page_allocator.alloc(c.MPI_Request, reqs.len);
    defer std.heap.page_allocator.free(raw_reqs);

    for (reqs, 0..) |req, idx| {
        raw_reqs[idx] = req.handle;
    }

    try check(c.MPI_Waitall(@intCast(reqs.len), raw_reqs.ptr, statuses.ptr));
}

pub fn allreduceSum(comm: Comm, value: f64) !f64 {
    var out: f64 = 0.0;
    try check(c.MPI_Allreduce(&value, &out, 1, c.MPI_DOUBLE, c.MPI_SUM, comm.handle));
    return out;
}

pub fn allgatherI32(comm: Comm, value: i32, recv: []i32) !void {
    if (recv.len == 0) return;
    try check(c.MPI_Allgather(
        &value,
        1,
        c.MPI_INT,
        recv.ptr,
        1,
        c.MPI_INT,
        comm.handle,
    ));
}

pub fn allgatherBytes(comm: Comm, send: []const u8, recv: []u8) !void {
    if (send.len == 0 or recv.len == 0) return;
    const count: c_int = @intCast(send.len);
    try check(c.MPI_Allgather(
        @constCast(send.ptr),
        count,
        c.MPI_BYTE,
        recv.ptr,
        count,
        c.MPI_BYTE,
        comm.handle,
    ));
}

pub fn allgatherVBytes(
    comm: Comm,
    send: []const u8,
    recv: []u8,
    recv_counts: []const i32,
    recv_displs: []const i32,
) !void {
    if (recv_counts.len == 0) return;
    const send_count: c_int = @intCast(send.len);
    try check(c.MPI_Allgatherv(
        @constCast(send.ptr),
        send_count,
        c.MPI_BYTE,
        recv.ptr,
        @ptrCast(recv_counts.ptr),
        @ptrCast(recv_displs.ptr),
        c.MPI_BYTE,
        comm.handle,
    ));
}

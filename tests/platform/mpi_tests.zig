const std = @import("std");
const su_n = @import("su_n");
const constants = su_n.constants;
const mpi = su_n.platform.mpi;

var init_level: ?mpi.ThreadLevel = null;

fn ensureMpi() !mpi.ThreadLevel {
    if (!mpi.enabled) return error.SkipZigTest;
    if (init_level == null) {
        init_level = try mpi.initSerialized();
    }
    return init_level.?;
}

fn isAtLeastSerialized(level: mpi.ThreadLevel) bool {
    return switch (level) {
        .single, .funneled => false,
        .serialized, .multiple => true,
    };
}

test "mpi disabled returns error" {
    if (mpi.enabled) return error.SkipZigTest;
    try std.testing.expectError(mpi.types.Error.MpiDisabled, mpi.initSerialized());
}

test "mpi init and world info" {
    const level = try ensureMpi();
    try std.testing.expect(isAtLeastSerialized(level));

    const comm = mpi.commWorld();
    const rank = try mpi.rank(comm);
    const size = try mpi.size(comm);

    try std.testing.expect(size > 0);
    try std.testing.expect(rank >= 0);
    try std.testing.expect(rank < size);
}

test "mpi allreduce sum" {
    _ = try ensureMpi();

    const comm = mpi.commWorld();
    const rank = try mpi.rank(comm);
    const size = try mpi.size(comm);

    const local = @as(f64, @floatFromInt(rank + 1));
    const total = try mpi.allreduceSum(comm, local);

    const n = @as(f64, @floatFromInt(size));
    const expected = n * (n + 1.0) / 2.0;
    try std.testing.expectApproxEqAbs(expected, total, constants.test_epsilon);
}

test "mpi send recv ping pong" {
    _ = try ensureMpi();

    const comm = mpi.commWorld();
    const rank = try mpi.rank(comm);
    const size = try mpi.size(comm);

    if (size < 2) return error.SkipZigTest;

    try mpi.barrier(comm);

    const tag: mpi.types.Tag = 7;
    var send_buf: [16]u8 = undefined;
    var recv_buf: [16]u8 = undefined;
    var expected_buf: [16]u8 = undefined;
    const base: u8 = if (rank == 0) 0x10 else 0x20;
    const expected_base: u8 = if (rank == 0) 0x20 else 0x10;

    for (&send_buf, 0..) |*byte, idx| {
        byte.* = base + @as(u8, @intCast(idx));
    }
    for (&expected_buf, 0..) |*byte, idx| {
        byte.* = expected_base + @as(u8, @intCast(idx));
    }

    if (rank == 0 or rank == 1) {
        const recv_req = try mpi.irecv(comm, recv_buf[0..], if (rank == 0) 1 else 0, tag);
        const send_req = try mpi.isend(comm, send_buf[0..], if (rank == 0) 1 else 0, tag);
        var reqs = [_]mpi.Request{ recv_req, send_req };
        try mpi.waitAll(reqs[0..]);
        try std.testing.expect(std.mem.eql(u8, recv_buf[0..], expected_buf[0..]));
    }

    try mpi.barrier(comm);
}

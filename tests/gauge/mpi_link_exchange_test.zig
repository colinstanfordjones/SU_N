const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const su_n = @import("su_n");
const constants = su_n.constants;
const platform = su_n.platform;

test "mpi gauge link exchange uses full face payload" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);
    const Nd = 2;
    const block_size = 4;
    const domain_extent = @as(f64, @floatFromInt(block_size * 2));
    const Topology = amr.topology.OpenTopology(Nd, .{ domain_extent, 4.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, Nd, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();
    var field = try GaugeField.init(std.testing.allocator, &tree);
    defer field.deinit();

    const origin = .{ @as(usize, @intCast(rank)) * block_size, 0 };
    const block_idx = try tree.insertBlock(origin, 0);
    try field.syncWithTree(&tree);

    const link_value = @as(f64, @floatFromInt(rank + 2));
    if (field.getBlockLinksMut(block_idx)) |links| {
        var link = Link.identity();
        link.matrix.data[0][0] = Complex.init(link_value, 0);
        for (links) |*l| l.* = link;
    }

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);
    try field.fillGhosts(&tree);

    const ghost = field.ghosts.get(block_idx).?;
    const face_idx: usize = if (rank == 0) 0 else 1;
    const expected = @as(f64, @floatFromInt((1 - rank) + 2));
    inline for (0..Nd) |link_dim| {
        const ghost_slice = ghost.get(face_idx, link_dim);
        try std.testing.expect(ghost_slice.len > 0);

        const first = ghost_slice[0].matrix.data[0][0].re;
        const last = ghost_slice[ghost_slice.len - 1].matrix.data[0][0].re;
        try std.testing.expectApproxEqAbs(expected, first, constants.test_epsilon);
        try std.testing.expectApproxEqAbs(expected, last, constants.test_epsilon);
    }
}

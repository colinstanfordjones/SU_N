const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const su_n = @import("su_n");
const constants = su_n.constants;
const platform = su_n.platform;

test "mpi gauge repartition migrates links" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(2, .{ 64.0, 4.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, block_size, Topology);
    const GaugeTree = gauge.GaugeTree(Frontend);
    const Arena = GaugeTree.FieldArena;
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    var tree = try GaugeTree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const origins = [_][2]usize{ .{ 0, 0 }, .{ 4, 0 }, .{ 8, 0 }, .{ 12, 0 } };

    if (rank == 0) {
        for (origins[0..3]) |origin| {
            _ = try tree.insertBlockWithField(origin, 0, &arena);
        }
    } else if (rank == 1) {
        _ = try tree.insertBlockWithField(origins[3], 0, &arena);
    }

    for (tree.tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = tree.tree.getFieldSlot(idx);
        const field_slice = arena.getSlot(slot);
        for (field_slice) |*value| {
            value.*[0] = Complex.init(1.0, 0.0);
        }

        const link_value = @as(f64, @floatFromInt(block.origin[0] / block_size + 2));
        var link = Link.identity();
        link.matrix.data[0][0] = Complex.init(link_value, 0.0);
        if (tree.getBlockLinksMut(idx)) |links| {
            for (links) |*l| l.* = link;
        }
    }

    var shard = try amr.ShardContext(GaugeTree.TreeType).initFromTree(
        std.testing.allocator,
        &tree.tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);

    try gauge.repartition.repartitionEntropyWeighted(GaugeTree, &tree, &arena, &shard, .{});

    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());
    try std.testing.expectEqual(@as(usize, 2), shard.localBlockIndices().len);

    for (shard.localBlockIndices()) |block_idx| {
        const block = tree.tree.getBlock(block_idx).?;
        const expected = @as(f64, @floatFromInt(block.origin[0] / block_size + 2));
        const links = tree.getBlockLinksConst(block_idx).?;
        try std.testing.expectApproxEqAbs(
            expected,
            links[0].matrix.data[0][0].re,
            constants.test_epsilon,
        );
    }
}

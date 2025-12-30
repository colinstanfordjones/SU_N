const std = @import("std");
const amr = @import("amr");
const su_n = @import("su_n");
const platform = su_n.platform;

test "mpi repartition entropy weighted balances blocks" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(1, .{64.0});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const origins = [_][1]usize{ .{0}, .{4}, .{8}, .{12} };

    const setBlockValues = struct {
        fn apply(origin: [1]usize, field_slice: []f64) void {
            const block_id = origin[0] / block_size;
            for (field_slice, 0..) |*value, idx| {
                const bit = (block_id >> @intCast(idx)) & 1;
                value.* = if (bit == 1) -1.0 else 1.0;
            }
        }
    }.apply;

    if (rank == 0) {
        for (origins[0..3]) |origin| {
            const block_idx = try tree.insertBlockWithField(origin, 0, &arena);
            const slot = tree.getFieldSlot(block_idx);
            const field_slice = arena.getSlot(slot);
            setBlockValues(origin, field_slice);
        }
    } else if (rank == 1) {
        const origin = origins[3];
        const block_idx = try tree.insertBlockWithField(origin, 0, &arena);
        const slot = tree.getFieldSlot(block_idx);
        const field_slice = arena.getSlot(slot);
        setBlockValues(origin, field_slice);
    }

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);

    try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, .{});

    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());
    try std.testing.expectEqual(@as(usize, 2), shard.localBlockIndices().len);

    const expected = if (rank == 0) [_]usize{ 0, 4 } else [_]usize{ 8, 12 };
    var found = [_]bool{ false, false };

    for (shard.localBlockIndices()) |block_idx| {
        const block = tree.getBlock(block_idx).?;
        const origin = block.origin[0];

        var matched = false;
        for (expected, 0..) |exp, idx| {
            if (origin == exp) {
                found[idx] = true;
                matched = true;
                break;
            }
        }
        try std.testing.expect(matched);

        const slot = tree.getFieldSlot(block_idx);
        const field_slice = arena.getSlotConst(slot);
        const block_id = origin / block_size;
        for (field_slice, 0..) |value, idx| {
            const bit = (block_id >> @intCast(idx)) & 1;
            const expected_value: f64 = if (bit == 1) -1.0 else 1.0;
            try std.testing.expectEqual(expected_value, value);
        }
    }

    try std.testing.expect(found[0] and found[1]);
}

test "mpi repartition entropy weighted handles empty blocks" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(1, .{64.0});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const origins = [_][1]usize{ .{0}, .{4}, .{8}, .{12} };

    if (rank == 0) {
        for (origins[0..3]) |origin| {
            _ = try tree.insertBlock(origin, 0);
        }
    } else if (rank == 1) {
        _ = try tree.insertBlock(origins[3], 0);
    }

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);

    try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, .{ .min_weight = 1.0 });

    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());
    try std.testing.expectEqual(@as(usize, 2), shard.localBlockIndices().len);

    for (shard.localBlockIndices()) |block_idx| {
        try std.testing.expect(!tree.hasFieldSlot(block_idx));
    }
}

test "mpi repartition adaptive skips when balanced" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(1, .{64.0});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const origins = [_][1]usize{ .{0}, .{4}, .{8}, .{12} };

    if (rank == 0) {
        for (origins[0..2]) |origin| {
            _ = try tree.insertBlockWithField(origin, 0, &arena);
        }
    } else if (rank == 1) {
        for (origins[2..4]) |origin| {
            _ = try tree.insertBlockWithField(origin, 0, &arena);
        }
    }

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (!tree.hasFieldSlot(idx)) continue;
        const slot = tree.getFieldSlot(idx);
        const field_slice = arena.getSlot(slot);
        for (field_slice) |*value| value.* = 1.0;
    }

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);

    const did_repartition = try amr.repartition.repartitionAdaptiveEntropyWeighted(
        Tree,
        &tree,
        &arena,
        &shard,
        .{},
        .{},
    );

    try std.testing.expect(!did_repartition);
    try std.testing.expectEqual(@as(usize, 2), shard.localBlockIndices().len);
}

test "mpi repartition adaptive triggers when imbalanced" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(1, .{64.0});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const origins = [_][1]usize{ .{0}, .{4}, .{8}, .{12} };

    if (rank == 0) {
        for (origins[0..3]) |origin| {
            _ = try tree.insertBlockWithField(origin, 0, &arena);
        }
    } else if (rank == 1) {
        _ = try tree.insertBlockWithField(origins[3], 0, &arena);
    }

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (!tree.hasFieldSlot(idx)) continue;
        const slot = tree.getFieldSlot(idx);
        const field_slice = arena.getSlot(slot);
        for (field_slice) |*value| value.* = 1.0;
    }

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);

    const did_repartition = try amr.repartition.repartitionAdaptiveEntropyWeighted(
        Tree,
        &tree,
        &arena,
        &shard,
        .{},
        .{ .weight_imbalance_threshold = 0.1, .block_imbalance_threshold = 0.1 },
    );

    try std.testing.expect(did_repartition);
    try std.testing.expectEqual(@as(usize, 2), shard.localBlockIndices().len);
}

test "mpi repartition entropy weighted balances 3 ranks" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size != 3) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(1, .{96.0});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 16);
    defer arena.deinit();

    const origins = [_][1]usize{ .{0}, .{4}, .{8}, .{12}, .{16}, .{20} };

    if (rank == 0) {
        for (origins[0..4]) |origin| {
            _ = try tree.insertBlockWithField(origin, 0, &arena);
        }
    } else if (rank == 1) {
        _ = try tree.insertBlockWithField(origins[4], 0, &arena);
    } else if (rank == 2) {
        _ = try tree.insertBlockWithField(origins[5], 0, &arena);
    }

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (!tree.hasFieldSlot(idx)) continue;
        const slot = tree.getFieldSlot(idx);
        const field_slice = arena.getSlot(slot);
        for (field_slice) |*value| value.* = 1.0;
    }

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    tree.attachShard(&shard);

    try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, .{});

    try std.testing.expectEqual(@as(usize, 2), shard.localBlockIndices().len);
}

const std = @import("std");
const su_n = @import("su_n");

const amr = su_n.amr;
const gauge = su_n.gauge;
const checkpoint = su_n.platform.checkpoint;

fn expectBytesEqual(a: anytype, b: anytype) !void {
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b)));
}

test "checkpoint roundtrip tree and arena" {
    const Frontend = amr.ScalarFrontend(2, 4, amr.OpenTopology(2, .{ 8.0, 8.0 }));
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const Checkpoint = checkpoint.TreeCheckpoint(Tree);

    var tree = try Tree.init(std.testing.allocator, 0.5, 4, 8);
    defer tree.deinit();
    tree.max_level = 6;

    var arena = try Arena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const idx0 = try tree.insertBlockWithField(.{ 0, 0 }, 0, &arena);
    const idx1 = try tree.insertBlockWithField(.{ 4, 0 }, 1, &arena);

    for ([_]usize{ idx0, idx1 }) |block_idx| {
        const slot = tree.getFieldSlot(block_idx);
        const data = arena.getSlot(slot);
        for (data, 0..) |*val, i| {
            val.* = @as(f64, @floatFromInt(block_idx * 1000 + i));
        }
    }

    if (arena.allocSlot()) |slot_extra| {
        const data = arena.getSlot(slot_extra);
        for (data, 0..) |*val, i| {
            val.* = @as(f64, @floatFromInt(i)) * 0.25;
        }
        arena.freeSlot(slot_extra);
    }

    tree.invalidateBlock(idx1);

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(std.testing.allocator);
    const writer = buffer.writer(std.testing.allocator);
    try Checkpoint.write(&tree, &arena, writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    var restored = try Checkpoint.read(std.testing.allocator, stream.reader());
    defer restored.deinit();

    try std.testing.expectEqual(tree.base_spacing, restored.tree.base_spacing);
    try std.testing.expectEqual(tree.bits_per_dim, restored.tree.bits_per_dim);
    try std.testing.expectEqual(tree.max_level, restored.tree.max_level);

    try expectBytesEqual(tree.blocks.items, restored.tree.blocks.items);
    try expectBytesEqual(tree.field_slots.items, restored.tree.field_slots.items);
    try expectBytesEqual(arena.storage, restored.arena.storage);
    try expectBytesEqual(arena.free_slots, restored.arena.free_slots);
    try std.testing.expectEqual(arena.free_count, restored.arena.free_count);

    const invalid = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == invalid) continue;
        const key = tree.blockKeyFromOrigin(block.origin, block.level);
        try std.testing.expectEqual(@as(?usize, idx), restored.tree.findBlockByKey(key));
    }
}

test "checkpoint restart rebuilds ghost buffers deterministically" {
    const Frontend = amr.ScalarFrontend(2, 4, amr.OpenTopology(2, .{ 8.0, 8.0 }));
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const Ghosts = amr.GhostBuffer(Frontend);
    const Checkpoint = checkpoint.TreeCheckpoint(Tree);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const idx0 = try tree.insertBlockWithField(.{ 0, 0 }, 0, &arena);
    const idx1 = try tree.insertBlockWithField(.{ 4, 0 }, 0, &arena);

    for ([_]usize{ idx0, idx1 }) |block_idx| {
        const slot = tree.getFieldSlot(block_idx);
        const data = arena.getSlot(slot);
        for (data, 0..) |*val, i| {
            val.* = @as(f64, @floatFromInt(block_idx * 1000 + i));
        }
    }

    var ghosts_before = try Ghosts.init(std.testing.allocator, 4);
    defer ghosts_before.deinit();
    try ghosts_before.ensureForTree(&tree);
    amr.ghost.fillGhostLayers(Tree, &tree, &arena, ghosts_before.slice(tree.blocks.items.len));

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(std.testing.allocator);
    const writer = buffer.writer(std.testing.allocator);
    try Checkpoint.write(&tree, &arena, writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    var restored = try Checkpoint.read(std.testing.allocator, stream.reader());
    defer restored.deinit();

    var ghosts_after = try Ghosts.init(std.testing.allocator, 4);
    defer ghosts_after.deinit();
    try ghosts_after.ensureForTree(&restored.tree);
    amr.ghost.fillGhostLayers(Tree, &restored.tree, &restored.arena, ghosts_after.slice(restored.tree.blocks.items.len));

    const invalid = std.math.maxInt(usize);
    const count = @min(tree.blocks.items.len, restored.tree.blocks.items.len);
    for (0..count) |block_idx| {
        if (tree.blocks.items[block_idx].block_index == invalid) continue;
        if (restored.tree.blocks.items[block_idx].block_index == invalid) continue;
        const before_faces = ghosts_before.get(block_idx) orelse continue;
        const after_faces = ghosts_after.get(block_idx) orelse continue;
        try std.testing.expect(std.mem.eql(u8, std.mem.asBytes(&before_faces.*), std.mem.asBytes(&after_faces.*)));
    }
}



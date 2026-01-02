const std = @import("std");
const su_n = @import("su_n");

const amr = su_n.amr;
const gauge = su_n.gauge;
const checkpoint = su_n.platform.checkpoint;

fn expectBytesEqual(a: anytype, b: anytype) !void {
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b)));
}

test "GaugeField - checkpoint roundtrip with links" {
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, amr.OpenTopology(2, .{ 8.0, 8.0 }));
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const Complex = std.math.Complex(f64);
    const TreeCheckpoint = checkpoint.TreeCheckpoint(Tree);

    var tree = try Tree.init(std.testing.allocator, 0.25, 4, 8);
    defer tree.deinit();
    var field = try GaugeField.init(std.testing.allocator, &tree);
    defer field.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const idx = try tree.insertBlockWithField(.{ 0, 0 }, 0, &arena);
    try field.syncWithTree(&tree);
    const slot = tree.getFieldSlot(idx);
    const data = arena.getSlot(slot);
    for (data, 0..) |*val, i| {
        val.* = .{Complex.init(@as(f64, @floatFromInt(i)), -0.5)};
    }

    // Set random links
    const HaarSampler = gauge.haar.HaarSampler(Frontend.LinkType.dim);
    var sampler = HaarSampler.init(1234);
    
    for (0..tree.blockCount()) |block_idx| {
        const link_slice = field.getBlockLinksMut(block_idx).?;
        for (link_slice) |*link| {
            link.* = sampler.sample();
        }
    }

    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    const writer = buffer.writer(std.testing.allocator);
    
    try TreeCheckpoint.write(&tree, &arena, writer);
    try field.writeCheckpoint(writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    
    const state = try TreeCheckpoint.read(std.testing.allocator, stream.reader());
    var restored_tree = state.tree;
    var restored_arena = state.arena;
    var restored_field = try GaugeField.readCheckpoint(std.testing.allocator, &restored_tree, stream.reader());
    defer restored_tree.deinit();
    defer restored_arena.deinit();
    defer restored_field.deinit();

    try std.testing.expectEqual(tree.base_spacing, restored_tree.base_spacing);
    try std.testing.expectEqual(tree.bits_per_dim, restored_tree.bits_per_dim);
    try std.testing.expectEqual(tree.max_level, restored_tree.max_level);

    try expectBytesEqual(tree.blocks.items, restored_tree.blocks.items);
    try expectBytesEqual(arena.storage, restored_arena.storage);

    try std.testing.expectEqual(tree.blockCount(), restored_tree.blockCount());
    for (0..tree.blockCount()) |block_idx| {
        const lhs = field.getBlockLinks(block_idx).?;
        const rhs = restored_field.getBlockLinks(block_idx).?;
        try expectBytesEqual(lhs, rhs);
    }
}

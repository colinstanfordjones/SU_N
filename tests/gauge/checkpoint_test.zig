const std = @import("std");
const su_n = @import("su_n");

const amr = su_n.amr;
const gauge = su_n.gauge;
const checkpoint = su_n.platform.checkpoint;

fn expectBytesEqual(a: anytype, b: anytype) !void {
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b)));
}

test "GaugeTree - checkpoint roundtrip with links" {
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, amr.OpenTopology(2, .{ 8.0, 8.0 }));
    const GaugeTree = gauge.GaugeTree(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const Complex = std.math.Complex(f64);
    const Link = Frontend.LinkType;

    var tree = try GaugeTree.init(std.testing.allocator, 0.25, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const idx = try tree.insertBlockWithField(.{ 0, 0 }, 0, &arena);
    const slot = tree.tree.getFieldSlot(idx);
    const data = arena.getSlot(slot);
    for (data, 0..) |*val, i| {
        val.* = .{Complex.init(@as(f64, @floatFromInt(i)), -0.5)};
    }

    for (tree.links.items, 0..) |link_slice, block_idx| {
        for (link_slice, 0..) |*link, link_idx| {
            var out = Link.identity();
            out.matrix.data[0][0] = Complex.init(
                @as(f64, @floatFromInt(block_idx * 100 + link_idx)),
                @as(f64, @floatFromInt(link_idx)) * 0.01,
            );
            link.* = out;
        }
    }

    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    const writer = buffer.writer(std.testing.allocator);
    
    // Use new API
    try tree.writeCheckpoint(&arena, writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    
    // Use new API
    const state = try GaugeTree.readCheckpoint(std.testing.allocator, stream.reader());
    var restored_tree = state.tree;
    var restored_arena = state.arena;
    defer restored_tree.deinit();
    defer restored_arena.deinit();

    try std.testing.expectEqual(tree.tree.base_spacing, restored_tree.tree.base_spacing);
    try std.testing.expectEqual(tree.tree.bits_per_dim, restored_tree.tree.bits_per_dim);
    try std.testing.expectEqual(tree.tree.max_level, restored_tree.tree.max_level);
    try std.testing.expect(!restored_tree.ghosts_valid);

    try expectBytesEqual(tree.tree.blocks.items, restored_tree.tree.blocks.items);
    try expectBytesEqual(arena.storage, restored_arena.storage);

    try std.testing.expectEqual(tree.links.items.len, restored_tree.links.items.len);
    for (tree.links.items, restored_tree.links.items) |lhs, rhs| {
        try expectBytesEqual(lhs, rhs);
    }
}

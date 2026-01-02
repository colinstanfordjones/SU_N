const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const su_n = @import("su_n");
const constants = su_n.constants;
const ghost_policy = gauge.ghost_policy;

test "gauge ghost normal link matches neighbor" {
    const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, Topology);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Block = Tree.BlockType;
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 4);
    defer tree.deinit();
    var field = try GaugeField.init(std.testing.allocator, &tree);
    defer field.deinit();

    const left_idx = try tree.insertBlock(.{ 0, 0 }, 0);
    const right_idx = try tree.insertBlock(.{ Block.size, 0 }, 0);
    try field.syncWithTree(&tree);

    var link = Link.identity();
    link.matrix.data[0][0] = Complex.init(3.0, 0);
    if (field.getBlockLinksMut(right_idx)) |links| {
        for (links) |*l| l.* = link;
    }

    try field.fillGhosts(&tree);

    const face_idx: usize = 0; // +x face
    const ghost_idx: usize = 0;
    const ghost = field.ghosts.get(left_idx).?;
    const ghost_slice = ghost.get(face_idx, 0);
    try std.testing.expect(ghost_slice.len > ghost_idx);
    const ghost_link = ghost_slice[ghost_idx];
    try std.testing.expectApproxEqAbs(3.0, ghost_link.matrix.data[0][0].re, constants.test_epsilon);
}

test "gauge ghost fine-to-coarse does not accumulate" {
    const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, Topology);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 4);
    defer tree.deinit();
    var field = try GaugeField.init(std.testing.allocator, &tree);
    defer field.deinit();

    const coarse_idx = try tree.insertBlock(.{ 0, 0 }, 0);
    const coarse_block = tree.getBlock(coarse_idx).?;

    var neighbor_physical = tree.getPhysicalOrigin(coarse_block);
    neighbor_physical[0] += tree.getBlockPhysicalExtent(coarse_block.level);
    const fine_origin_base = tree.physicalToBlockOrigin(neighbor_physical, coarse_block.level + 1);

    const face_dim: usize = 0;
    const fine_count = @as(usize, 1) << @intCast(Frontend.Nd - 1);
    for (0..fine_count) |combo| {
        var origin = fine_origin_base;
        var combo_idx: usize = 0;
        inline for (0..Frontend.Nd) |d| {
            if (d != face_dim) {
                const half = (combo >> @intCast(combo_idx)) & 1;
                origin[d] += half * Frontend.block_size;
                combo_idx += 1;
            }
        }
        _ = try tree.insertBlock(origin, coarse_block.level + 1);
    }

    try field.syncWithTree(&tree);
    for (0..tree.blockCount()) |idx| {
        if (idx == coarse_idx) continue;
        const links = field.getBlockLinksMut(idx).?;
        var link = Link.identity();
        link.matrix.data[0][0] = Complex.init(@as(f64, @floatFromInt(idx + 1)), 0);
        for (links) |*l| l.* = link;
    }

    try field.fillGhosts(&tree);

    const ghost = field.ghosts.get(coarse_idx).?;
    const face_idx: usize = 0; // +x face
    const link_dim: usize = 1; // tangential direction
    const ghost_slice = ghost.get(face_idx, link_dim);
    try std.testing.expect(ghost_slice.len > 0);

    const snapshot = try std.testing.allocator.alloc(Link, ghost_slice.len);
    defer std.testing.allocator.free(snapshot);
    @memcpy(snapshot, ghost_slice);

    if (field.getBlockLinksMut(coarse_idx)) |links| {
        links[0] = links[0];
    }
    try field.fillGhosts(&tree);

    const refreshed = ghost.get(face_idx, link_dim);
    for (refreshed, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(
            snapshot[i].matrix.data[0][0].re,
            val.matrix.data[0][0].re,
            constants.test_epsilon,
        );
    }
}

test "gauge ghost exchange uses custom spec for local fills" {
    const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, Topology);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Policy = ghost_policy.LinkGhostPolicy(GaugeField);
    const Context = Policy.Context;
    const Payload = Policy.Payload;
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    const packSame = struct {
        fn run(_: Context, _: usize, _: usize, dest: []Payload, _: std.mem.Allocator) void {
            var link = Link.identity();
            link.matrix.data[0][0] = Complex.init(7.0, 0.0);
            for (dest) |*v| v.* = link;
        }
    }.run;

    var spec = Policy.exchangeSpec();
    spec.pack_same_level = packSame;

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 4);
    defer tree.deinit();
    var field = try GaugeField.initWithOptions(std.testing.allocator, &tree, spec);
    defer field.deinit();

    const left_idx = try tree.insertBlock(.{ 0, 0 }, 0);
    _ = try tree.insertBlock(.{ Frontend.block_size, 0 }, 0);

    try field.syncWithTree(&tree);
    try field.fillGhosts(&tree);

    const face_idx: usize = 0; // +x face
    inline for (0..Frontend.Nd) |link_dim| {
        const ghost = field.ghosts.get(left_idx).?;
        const ghost_slice = ghost.get(face_idx, link_dim);
        try std.testing.expect(ghost_slice.len > 0);
        const ghost_link = ghost_slice[0];
        try std.testing.expectApproxEqAbs(7.0, ghost_link.matrix.data[0][0].re, constants.test_epsilon);
    }
}

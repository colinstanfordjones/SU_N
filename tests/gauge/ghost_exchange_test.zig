const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const su_n = @import("su_n");
const constants = su_n.constants;
const ghost_policy = gauge.ghost_policy;

test "gauge ghost normal link matches neighbor" {
    const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, Topology);
    const GaugeTree = gauge.GaugeTree(Frontend);
    const Block = GaugeTree.BlockType;
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    var tree = try GaugeTree.init(std.testing.allocator, 1.0, 4, 4);
    defer tree.deinit();

    const left_idx = try tree.insertBlock(.{ 0, 0 }, 0);
    const right_idx = try tree.insertBlock(.{ Block.size, 0 }, 0);

    var link = Link.identity();
    link.matrix.data[0][0] = Complex.init(3.0, 0);
    if (tree.getBlockLinksMut(right_idx)) |links| {
        for (links) |*l| l.* = link;
    }

    try tree.fillGhosts();

    const face_idx: usize = 0; // +x face
    const ghost_idx: usize = 0;
    const ghost_link = tree.getGhostLink(left_idx, face_idx, 0, ghost_idx);
    try std.testing.expectApproxEqAbs(3.0, ghost_link.matrix.data[0][0].re, constants.test_epsilon);
}

test "gauge ghost fine-to-coarse does not accumulate" {
    const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 4, Topology);
    const GaugeTree = gauge.GaugeTree(Frontend);
    const Link = Frontend.LinkType;
    const Complex = std.math.Complex(f64);

    var tree = try GaugeTree.init(std.testing.allocator, 1.0, 4, 4);
    defer tree.deinit();

    const coarse_idx = try tree.insertBlock(.{ 0, 0 }, 0);
    const coarse_block = tree.tree.getBlock(coarse_idx).?;

    var neighbor_physical = tree.tree.getPhysicalOrigin(coarse_block);
    neighbor_physical[0] += tree.tree.getBlockPhysicalExtent(coarse_block.level);
    const fine_origin_base = tree.tree.physicalToBlockOrigin(neighbor_physical, coarse_block.level + 1);

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

    for (tree.links.items, 0..) |links, idx| {
        if (idx == coarse_idx) continue;
        var link = Link.identity();
        link.matrix.data[0][0] = Complex.init(@as(f64, @floatFromInt(idx + 1)), 0);
        for (links) |*l| l.* = link;
    }

    try tree.fillGhosts();

    const ghost = &tree.ghosts.items[coarse_idx];
    const face_idx: usize = 0; // +x face
    const link_dim: usize = 1; // tangential direction
    const ghost_slice = ghost.get(face_idx, link_dim);
    try std.testing.expect(ghost_slice.len > 0);

    const snapshot = try std.testing.allocator.alloc(Link, ghost_slice.len);
    defer std.testing.allocator.free(snapshot);
    @memcpy(snapshot, ghost_slice);

    if (tree.getBlockLinksMut(coarse_idx)) |links| {
        links[0] = links[0];
    }
    try tree.fillGhosts();

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
    const GaugeTree = gauge.GaugeTree(Frontend);
    const Policy = ghost_policy.LinkGhostPolicy(GaugeTree);
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

    var tree = try GaugeTree.initWithOptions(
        std.testing.allocator,
        1.0,
        4,
        4,
        .{ .link_exchange_spec = spec },
    );
    defer tree.deinit();

    const left_idx = try tree.insertBlock(.{ 0, 0 }, 0);
    _ = try tree.insertBlock(.{ Frontend.block_size, 0 }, 0);

    try tree.fillGhosts();

    const face_idx: usize = 0; // +x face
    inline for (0..Frontend.Nd) |link_dim| {
        const ghost_link = tree.getGhostLink(left_idx, face_idx, link_dim, 0);
        try std.testing.expectApproxEqAbs(7.0, ghost_link.matrix.data[0][0].re, constants.test_epsilon);
    }
}

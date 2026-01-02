const std = @import("std");
const su_n = @import("su_n");
const amr = @import("amr");
const constants = su_n.constants;

test "ghost exchange begin/finish matches bulk fill" {
    const Topology = amr.topology.OpenTopology(2, .{ 64.0, 64.0 });
    const Frontend = amr.ScalarFrontend(2, 8, Topology);
    const Tree = amr.AMRTree(Frontend);
    const FieldArena = amr.FieldArena(Frontend);
    const GhostBuffer = amr.GhostBuffer(Frontend);
    const Block = amr.AMRBlock(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try FieldArena.init(std.testing.allocator, 16);
    defer arena.deinit();

    // Two adjacent coarse blocks.
    const left_idx = try tree.insertBlockWithField(.{ 0, 0 }, 0, &arena);
    const right_idx = try tree.insertBlockWithField(.{ Block.size, 0 }, 0, &arena);

    // Refine the right block to introduce fine->coarse push.
    try tree.refineBlock(right_idx);

    // Assign field slots to new refined blocks.
    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (!tree.hasFieldSlot(idx)) {
            const slot = arena.allocSlot() orelse unreachable;
            tree.assignFieldSlot(idx, slot);
        }
    }

    // Initialize field data with deterministic values per block/site.
    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = tree.getFieldSlot(idx);
        if (slot == std.math.maxInt(usize)) continue;

        const data = arena.getSlot(slot);
        for (data, 0..) |*v, i| {
            v.* = @as(f64, @floatFromInt(idx)) + @as(f64, @floatFromInt(i)) * 0.01;
        }
    }

    var ghosts_bulk = try GhostBuffer.init(std.testing.allocator, 16);
    defer ghosts_bulk.deinit();

    var ghosts_split = try GhostBuffer.init(std.testing.allocator, 16);
    defer ghosts_split.deinit();

    try ghosts_bulk.ensureForTree(&tree);
    try ghosts_split.ensureForTree(&tree);

    const ghost_len = tree.blocks.items.len;
    try tree.fillGhostLayers(&arena, ghosts_bulk.slice(ghost_len));
    var split_state = try tree.beginGhostExchange(&arena, ghosts_split.slice(ghost_len));
    try tree.finishGhostExchange(&split_state);

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;

        const bulk_faces = ghosts_bulk.get(idx) orelse continue;
        const split_faces = ghosts_split.get(idx) orelse continue;

        for (bulk_faces, 0..) |face_bulk, face_idx| {
            const face_split = split_faces[face_idx];
            for (face_bulk, 0..) |bulk_val, j| {
                try std.testing.expectApproxEqAbs(
                    bulk_val,
                    face_split[j],
                    constants.test_epsilon,
                );
            }
        }
    }

    _ = left_idx;
}

test "ghost exchange uses custom exchange spec for local fills" {
    const Topology = amr.topology.OpenTopology(1, .{ 8.0 });
    const Frontend = amr.ScalarFrontend(1, 4, Topology);
    const Tree = amr.AMRTree(Frontend);
    const FieldArena = amr.FieldArena(Frontend);
    const GhostBuffer = amr.GhostBuffer(Frontend);
    const FieldPolicy = amr.ghost_policy.FieldGhostPolicy(Tree);
    const Context = FieldPolicy.Context;
    const Payload = FieldPolicy.Payload;

    const sentinel: f64 = 42.0;

    const packSame = struct {
        fn run(_: Context, _: usize, _: usize, dest: []Payload, _: std.mem.Allocator) void {
            for (dest) |*v| v.* = sentinel;
        }
    }.run;

    const unpackSame = struct {
        fn run(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost_faces = ctx.ghosts[block_idx] orelse return;
            if (face >= ghost_faces.len) return;
            std.mem.copyForwards(Payload, ghost_faces[face][0..], src);
        }
    }.run;

    var spec = FieldPolicy.exchangeSpec();
    spec.pack_same_level = packSame;
    spec.unpack_same_level = unpackSame;

    var tree = try Tree.initWithOptions(
        std.testing.allocator,
        1.0,
        2,
        2,
        .{ .field_exchange_spec = spec },
    );
    defer tree.deinit();

    var arena = try FieldArena.init(std.testing.allocator, 4);
    defer arena.deinit();

    _ = try tree.insertBlockWithField(.{ 0 }, 0, &arena);
    _ = try tree.insertBlockWithField(.{ 4 }, 0, &arena);

    var ghosts = try GhostBuffer.init(std.testing.allocator, 4);
    defer ghosts.deinit();
    try ghosts.ensureForTree(&tree);

    const ghost_len = tree.blocks.items.len;
    try tree.fillGhostLayers(&arena, ghosts.slice(ghost_len));

    const left_faces = ghosts.get(0) orelse return error.TestExpectedEqual;
    const right_faces = ghosts.get(1) orelse return error.TestExpectedEqual;

    // Right neighbor fills left +X face with sentinel.
    try std.testing.expectApproxEqAbs(sentinel, left_faces[0][0], constants.test_epsilon);
    // Left neighbor fills right -X face with sentinel.
    try std.testing.expectApproxEqAbs(sentinel, right_faces[1][0], constants.test_epsilon);
    // Boundary faces remain zero-initialized.
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), left_faces[1][0], constants.test_epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), right_faces[0][0], constants.test_epsilon);
}

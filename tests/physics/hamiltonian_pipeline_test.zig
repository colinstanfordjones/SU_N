const std = @import("std");
const su_n = @import("su_n");
const amr = @import("amr");
const gauge = @import("gauge");
const physics = @import("physics");
const constants = su_n.constants;

const Complex = std.math.Complex(f64);

fn zeroPotential(pos: [2]f64, spacing: f64) f64 {
    _ = pos;
    _ = spacing;
    return 0.0;
}

test "HamiltonianAMR pipeline matches bulk sync" {
    const Topology = amr.topology.OpenTopology(2, .{ 32.0, 32.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, 8, Topology);
    const GT = gauge.GaugeTree(Frontend);
    const FieldArena = GT.FieldArena;
    const GhostBuffer = amr.GhostBuffer(Frontend);
    const HAMR = physics.hamiltonian_amr.HamiltonianAMR(Frontend);
    const ApplyContext = amr.ApplyContext(Frontend);
    const Block = GT.BlockType;

    var psi_arena = try FieldArena.init(std.testing.allocator, 16);
    defer psi_arena.deinit();

    var out_bulk = try FieldArena.init(std.testing.allocator, 16);
    defer out_bulk.deinit();

    var out_pipe = try FieldArena.init(std.testing.allocator, 16);
    defer out_pipe.deinit();

    var gauge_tree = try GT.init(std.testing.allocator, 1.0, 4, 8);
    defer gauge_tree.deinit();

    // Two adjacent blocks to exercise same-level ghosts.
    const left_idx = try gauge_tree.insertBlockWithField(.{ 0, 0 }, 0, &psi_arena);
    const right_idx = try gauge_tree.insertBlockWithField(.{ Block.size, 0 }, 0, &psi_arena);
    _ = right_idx;

    // Keep output arenas in lockstep with input slots.
    _ = out_bulk.allocSlot() orelse unreachable;
    _ = out_bulk.allocSlot() orelse unreachable;
    _ = out_pipe.allocSlot() orelse unreachable;
    _ = out_pipe.allocSlot() orelse unreachable;

    var ghosts = try GhostBuffer.init(std.testing.allocator, 16);
    defer ghosts.deinit();

    const mass = 1.0;
    var H = HAMR.init(&gauge_tree, mass, zeroPotential);

    // Initialize psi with block- and site-dependent values.
    for (gauge_tree.tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = gauge_tree.tree.getFieldSlot(idx);
        if (slot == std.math.maxInt(usize)) continue;
        const psi = psi_arena.getSlot(slot);
        for (psi, 0..) |*v, i| {
            const re = @as(f64, @floatFromInt(idx)) + @as(f64, @floatFromInt(i)) * 0.01;
            v.*[0] = Complex.init(re, 0.0);
        }
    }

    try ghosts.ensureForTree(&gauge_tree.tree);
    try gauge_tree.tree.fillGhostLayers(&psi_arena, ghosts.slice(gauge_tree.tree.blocks.items.len));
    try gauge_tree.fillGhosts();

    // Manual bulk sync: iterate blocks and call execute() with ApplyContext
    var ctx_bulk = ApplyContext.init(&gauge_tree.tree);
    ctx_bulk.field_in = &psi_arena;
    ctx_bulk.field_out = &out_bulk;
    ctx_bulk.field_ghosts = &ghosts;

    for (gauge_tree.tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = gauge_tree.tree.getFieldSlot(idx);
        if (slot == std.math.maxInt(usize)) continue;

        H.execute(idx, block, &ctx_bulk);
    }

    // Pipeline path: use apply() convenience wrapper
    try H.apply(&psi_arena, &out_pipe, &ghosts, null);

    for (gauge_tree.tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = gauge_tree.tree.getFieldSlot(idx);
        if (slot == std.math.maxInt(usize)) continue;

        const bulk = out_bulk.getSlot(slot);
        const pipe = out_pipe.getSlot(slot);
        for (bulk, 0..) |bulk_val, i| {
            try std.testing.expectApproxEqAbs(
                bulk_val[0].re,
                pipe[i][0].re,
                constants.test_epsilon,
            );
            try std.testing.expectApproxEqAbs(
                bulk_val[0].im,
                pipe[i][0].im,
                constants.test_epsilon,
            );
        }
    }

    _ = left_idx;
}

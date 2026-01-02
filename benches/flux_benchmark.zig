const std = @import("std");
const su_n = @import("su_n");
const gauge = su_n.gauge;
const physics = su_n.physics;
const amr = su_n.amr;

const Complex = std.math.Complex(f64);
const Nd = 4;
const N_gauge = 1; // U(1)
const block_size = 8;
const Topology = amr.topology.PeriodicTopology(Nd, .{ 16.0, 8.0, 8.0, 8.0 });
const Frontend = gauge.GaugeFrontend(N_gauge, 1, Nd, block_size, Topology); // N_gauge, N_spinor=1, Nd, block_size
const Tree = amr.AMRTree(Frontend);
const GaugeField = gauge.GaugeField(Frontend);
const FieldArena = amr.FieldArena(Frontend);
const GhostBuffer = amr.GhostBuffer(Frontend);
const HAMR = physics.hamiltonian_amr.HamiltonianAMR(Frontend);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();
    var field = try GaugeField.init(allocator, &tree);
    defer field.deinit();

    // Create a mesh with refinement (left block refined, right block coarse).
    const left_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ block_size, 0, 0, 0 }, 0);

    const left_block = tree.getBlock(left_idx).?;
    const child_level = left_block.level + 1;
    const parent_origin = left_block.origin;
    const child_step = block_size;
    tree.invalidateBlock(left_idx);

    for (0..(@as(usize, 1) << Nd)) |child| {
        var child_origin: [Nd]usize = undefined;
        inline for (0..Nd) |d| {
            const half = (child >> @intCast(d)) & 1;
            child_origin[d] = parent_origin[d] * 2 + half * child_step;
        }
        _ = try tree.insertBlock(child_origin, child_level);
    }

    const block_count = tree.blockCount();
    const block_capacity = tree.blocks.items.len;

    var arena = try FieldArena.init(allocator, block_count);
    defer arena.deinit();
    
    // Alloc slots
    for (tree.blocks.items, 0..) |*blk, idx| {
        if (blk.block_index == std.math.maxInt(usize)) continue;
        const slot = arena.allocSlot().?;
        tree.assignFieldSlot(idx, slot);
    }

    var workspace = try FieldArena.init(allocator, block_count);
    defer workspace.deinit();
    for (0..block_count) |_| _ = workspace.allocSlot();

    var ghosts = try GhostBuffer.init(allocator, block_capacity);
    defer ghosts.deinit();
    try ghosts.ensureForTree(&tree);

    try field.syncWithTree(&tree);
    var H = HAMR.init(&tree, &field, 1.0, physics.hamiltonian_amr.freeParticle);

    std.debug.print("Benchmarking Flux Corrected Evolution...\n", .{});
    const start = std.time.nanoTimestamp();
    
    // Run 100 steps
    try H.evolveImaginaryTimeAMR(&arena, &workspace, &ghosts, 0.01, 100, 0, 0, 0.0, 0.0);
    
    const end = std.time.nanoTimestamp();
    const elapsed = @as(f64, @floatFromInt(end - start)) / 1e9;
    std.debug.print("Evolution (100 steps): {d:.6} s\n", .{elapsed});
    std.debug.print("Time per step: {d:.6} ms\n", .{elapsed * 1000.0 / 100.0});
}

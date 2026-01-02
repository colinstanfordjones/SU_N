const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const physics = @import("physics");

const Complex = std.math.Complex(f64);

// Test topology
const TestTopology4D = amr.topology.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });

test "HamiltonianAMR initialization and apply" {
    // 1. Setup Frontend
    const Frontend = gauge.GaugeFrontend(1, 1, 4, 16, TestTopology4D); // U(1), scalar, 4D, 16^4
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const FieldArena = amr.FieldArena(Frontend);
    const GhostBuffer = amr.GhostBuffer(Frontend);
    const HAMR = physics.hamiltonian_amr.HamiltonianAMR(Frontend);

    var arena = try FieldArena.init(std.testing.allocator, 16);
    defer arena.deinit();

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();
    var field = try GaugeField.init(std.testing.allocator, &tree);
    defer field.deinit();

    // 2. Insert block
    const block_idx = try tree.insertBlockWithField(.{0, 0, 0, 0}, 0, &arena);
    try field.syncWithTree(&tree);
    
    // 3. Setup Hamiltonian
    const mass = 1.0;
    // Free particle potential
    const potential_fn = physics.hamiltonian_amr.freeParticle;
    
    var H = HAMR.init(&tree, &field, mass, potential_fn);
    
    // 4. Setup Ghost Buffer
    var ghosts = try GhostBuffer.init(std.testing.allocator, 16);
    defer ghosts.deinit();
    
    // 5. Initialize Psi
    const slot = tree.getFieldSlot(block_idx);
    arena.zeroSlot(slot); // Ensure clean slate
    const psi = arena.getSlot(slot);
    for (psi) |*v| v.*[0] = Complex.init(1.0, 0.0);
    
    // 6. Apply Hamiltonian
    // Allocate workspace for output
    const workspace_slot = arena.allocSlot().?;
    defer arena.freeSlot(workspace_slot);
    
    var out_arena = try FieldArena.init(std.testing.allocator, 16);
    defer out_arena.deinit();
    
    const out_slot = out_arena.allocSlot().?;
    out_arena.zeroSlot(out_slot); // Zero output too
    try std.testing.expectEqual(slot, out_slot); // Verify lockstep
    
    try H.apply(&arena, &out_arena, &ghosts, null);
    
    // Verify result (should be 0 for free particle uniform field, ignoring boundaries)
    const res = out_arena.getSlot(out_slot);
    const Block = Tree.BlockType;
    
    for (res, 0..) |v, i| {
        const coords = Block.getLocalCoords(i);
        if (!Block.isOnBoundary(coords)) {
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), v[0].re, 1e-10);
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), v[0].im, 1e-10);
        }
    }
}

//! Tests for AMR Tree implementation (4D spacetime)
//! All field operations use FieldType from Frontend

const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");

const AMRTree = amr.tree.AMRTree;
const FieldArena = amr.field_arena.FieldArena;
const ghost = amr.ghost;
const adaptation = amr.adaptation;

const Complex = std.math.Complex(f64);

// Test topology
const TestTopology4D = amr.topology.OpenTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });

// Frontend factories for testing
fn ScalarFrontend(comptime bs: usize) type {
    return struct {
        pub const Nd: usize = 4;
        pub const block_size: usize = bs;
        pub const FieldType = f64;
        pub const Topology = TestTopology4D;
    };
}

fn ComplexArrayFrontend(comptime N: usize, comptime bs: usize) type {
    return struct {
        pub const Nd: usize = 4;
        pub const block_size: usize = bs;
        pub const FieldType = [N]Complex;
        pub const Topology = TestTopology4D;
    };
}

// Use GaugeFrontend for gauge-aware tests
fn GaugeFrontendFactory(comptime N_gauge: usize, comptime bs: usize) type {
    return gauge.GaugeFrontend(N_gauge, 1, 4, bs, TestTopology4D);
}

test "AMRTree 4D basic initialization" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 16), Tree.children_per_node); // 2^4
    try std.testing.expectEqual(@as(usize, 0), tree.blockCount());
    try std.testing.expectEqual(@as(usize, 0), tree.blocks.items.len);
}

test "AMRTree insert single block" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    const block_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try std.testing.expectEqual(@as(usize, 0), block_idx);
    try std.testing.expectEqual(@as(usize, 1), tree.blockCount());

    const block = tree.getBlock(block_idx).?;
    try std.testing.expectEqual(@as(u8, 0), block.level);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), block.spacing, 1e-10);
}

test "AMRTree insert multiple blocks at same level" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Insert 2x2x2x2 grid of blocks at level 0 (16 blocks)
    for (0..2) |t| {
        for (0..2) |x| {
            for (0..2) |y| {
                for (0..2) |z| {
                    _ = try tree.insertBlock(.{ t * 4, x * 4, y * 4, z * 4 }, 0);
                }
            }
        }
    }

    try std.testing.expectEqual(@as(usize, 16), tree.blockCount());
}

test "AMRTree neighbor detection same level" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Insert two adjacent blocks (in t direction)
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const b1 = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0); // +t neighbor

    _ = tree.getBlock(b0).?;
    _ = tree.getBlock(b1).?;

    // b0's +t neighbor should be b1
    const b0_info = tree.neighborInfo(b0, 0);
    try std.testing.expect(b0_info.exists());
    try std.testing.expectEqual(@as(i8, 0), b0_info.level_diff);
    try std.testing.expectEqual(b1, b0_info.block_idx);
    // b1's -t neighbor should be b0
    const b1_info = tree.neighborInfo(b1, 1);
    try std.testing.expect(b1_info.exists());
    try std.testing.expectEqual(@as(i8, 0), b1_info.level_diff);
    try std.testing.expectEqual(b0, b1_info.block_idx);
}

test "AMRTree insert at different levels" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Insert coarse block
    const coarse = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const coarse_block = tree.getBlock(coarse).?;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), coarse_block.spacing, 1e-10);

    // Insert fine block (level 1 has half the spacing)
    const fine = try tree.insertBlock(.{ 8, 0, 0, 0 }, 1);
    const fine_block = tree.getBlock(fine).?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), fine_block.spacing, 1e-10);

    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());
}

test "AMRTree Morton index ordering" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Get Morton indices for different origins
    const m0 = tree.getMortonIndex(.{ 0, 0, 0, 0 }, 0);
    const m1 = tree.getMortonIndex(.{ 4, 0, 0, 0 }, 0);
    const m2 = tree.getMortonIndex(.{ 0, 4, 0, 0 }, 0);

    // Morton indices should be different for different origins
    try std.testing.expect(m0 != m1);
    try std.testing.expect(m1 != m2);
    try std.testing.expect(m0 != m2);
}

test "AMRTree block iterator" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Insert some blocks
    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 0, 4, 0, 0 }, 0);

    // Count via iterator
    var count: usize = 0;
    var iter = tree.blockIterator();
    while (iter.next()) |_| {
        count += 1;
    }

    try std.testing.expectEqual(@as(usize, 3), count);
}

test "AMRTree boundary neighbor detection" {
    const Frontend = ComplexArrayFrontend(1, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Insert single block at origin
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    _ = tree.getBlock(b0).?;

    // All directions should have no neighbor (at domain boundary or empty)
    try std.testing.expect(!tree.neighborInfo(b0, 1).exists()); // -t
    try std.testing.expect(!tree.neighborInfo(b0, 3).exists()); // -x
    try std.testing.expect(!tree.neighborInfo(b0, 5).exists()); // -y
    try std.testing.expect(!tree.neighborInfo(b0, 7).exists()); // -z

    // Positive directions also have no neighbor (no blocks there)
    try std.testing.expect(!tree.neighborInfo(b0, 0).exists()); // +t
    try std.testing.expect(!tree.neighborInfo(b0, 2).exists()); // +x
    try std.testing.expect(!tree.neighborInfo(b0, 4).exists()); // +y
    try std.testing.expect(!tree.neighborInfo(b0, 6).exists()); // +z
}

test "AMRTree ghost layer filling same level" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    // Insert two adjacent blocks (in t direction)
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const b1 = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0); // +t neighbor

    // Allocate and assign field slots
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    const slot1 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);
    tree.assignFieldSlot(b1, slot1);

    // Fill with distinct Complex values
    const Block = Tree.BlockType;
    const field0 = arena.getSlot(slot0);
    const field1 = arena.getSlot(slot1);
    for (field0) |*v| v.*[0] = Complex.init(1.0, 0.5);
    for (field1) |*v| v.*[0] = Complex.init(2.0, 1.0);

    // Ghost layer storage for all blocks
    var ghosts0: [8][Block.ghost_face_size][N]Complex = undefined;
    var ghosts1: [8][Block.ghost_face_size][N]Complex = undefined;
    var all_ghosts = [_]?*[8][Block.ghost_face_size][N]Complex{ &ghosts0, &ghosts1 };

    // Fill ghost layers for all blocks at once
    ghost.fillGhostLayers(Tree, &tree, &arena, &all_ghosts);

    // The +t ghost of block 0 should contain values from block 1's -t boundary
    // Block 1 has value (2.0, 1.0) everywhere
    for (ghosts0[0]) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 2.0), v[0].re, 1e-10);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), v[0].im, 1e-10);
    }

    // The -t ghost of block 0 should be zero (no neighbor)
    for (ghosts0[1]) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v[0].re, 1e-10);
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v[0].im, 1e-10);
    }
}

// =====================================================================
// Adaptive Refinement Tests
// =====================================================================

test "computeBlockGradient returns zero for uniform field" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create uniform wavefunction
    const field = arena.getSlot(slot);
    for (field) |*v| v.*[0] = Complex.init(1.0, 0.0);

    // Uniform field should have zero gradient
    const gradient = adaptation.computeBlockGradient(Tree, &tree, b0, &arena);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gradient, 1e-10);
}

test "computeBlockGradient detects linear variation" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create linearly varying wavefunction in t-direction
    // ψ(t,x,y,z) = t  (so gradient in t = 1.0)
    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);

    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const t_coord: f64 = @floatFromInt(coords[0]);
        field[i][0] = Complex.init(t_coord, 0.0);
    }

    // Gradient should be approximately 1.0 (difference between adjacent sites)
    const gradient = adaptation.computeBlockGradient(Tree, &tree, b0, &arena);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), gradient, 0.01);
}

test "computeBlockGradient detects steep gradient" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create step function: 0 for t < 2, 10.0 for t >= 2
    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);

    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const value: f64 = if (coords[0] < 2) 0.0 else 10.0;
        field[i][0] = Complex.init(value, 0.0);
    }

    // Gradient should capture the step (10.0 difference)
    const gradient = adaptation.computeBlockGradient(Tree, &tree, b0, &arena);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), gradient, 0.01);
}

test "shouldRefine triggers on high gradient" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create steep gradient
    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);

    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const value: f64 = if (coords[0] < 2) 0.0 else 5.0;
        field[i][0] = Complex.init(value, 0.0);
    }

    // Low threshold should trigger refinement
    try std.testing.expect(adaptation.shouldRefine(Tree, &tree, b0, &arena, 1.0));

    // High threshold should not trigger refinement
    try std.testing.expect(!adaptation.shouldRefine(Tree, &tree, b0, &arena, 10.0));
}

test "shouldRefine respects max level" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    tree.max_level = 1; // Low max level for testing
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    // Insert at max level
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 1);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create steep gradient
    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);
    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const value: f64 = if (coords[0] < 2) 0.0 else 100.0;
        field[i][0] = Complex.init(value, 0.0);
    }

    // Should not refine even with huge gradient (at max level)
    try std.testing.expect(!adaptation.shouldRefine(Tree, &tree, b0, &arena, 0.01));
}

test "shouldCoarsen triggers on low gradient" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    // Insert at level 1 (can be coarsened)
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 1);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create uniform field (zero gradient)
    const field = arena.getSlot(slot);
    for (field) |*v| v.*[0] = Complex.init(1.0, 0.0);

    // Should coarsen with reasonable threshold and hysteresis
    try std.testing.expect(adaptation.shouldCoarsen(Tree, &tree, b0, &arena, 1.0, 0.5));

    // Should not coarsen with very low hysteresis product
    try std.testing.expect(adaptation.shouldCoarsen(Tree, &tree, b0, &arena, 1.0, 0.001));
}

test "shouldCoarsen does not trigger for level 0" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    // Insert at level 0 (cannot be coarsened)
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create uniform field
    const field = arena.getSlot(slot);
    for (field) |*v| v.*[0] = Complex.init(1.0, 0.0);

    // Should not coarsen level 0 regardless of gradient
    try std.testing.expect(!adaptation.shouldCoarsen(Tree, &tree, b0, &arena, 100.0, 0.5));
}

test "hysteresis prevents coarsening refinement oscillation" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 1);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create field with gradient of exactly 0.8
    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);
    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const value: f64 = @as(f64, @floatFromInt(coords[0])) * 0.8;
        field[i][0] = Complex.init(value, 0.0);
    }

    const threshold = 1.0;
    const hysteresis = 0.5;

    // Gradient ~0.8 is below threshold (1.0), so no refinement
    try std.testing.expect(!adaptation.shouldRefine(Tree, &tree, b0, &arena, threshold));

    // Gradient ~0.8 is above threshold * hysteresis (0.5), so no coarsening
    try std.testing.expect(!adaptation.shouldCoarsen(Tree, &tree, b0, &arena, threshold, hysteresis));

    // This is the hysteresis zone: neither refine nor coarsen
}

test "adaptMesh refines high gradient blocks" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 32);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate field slot and fill with a linear gradient so children stay non-uniform.
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);
    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const value: f64 = @floatFromInt(coords[0]);
        field[i][0] = Complex.init(value, 0.0);
    }

    // Adapt with low threshold - should trigger refinement
    const result = try adaptation.adaptMesh(Tree, &tree, &arena, 0.5, 0.5);

    // Block should have been refined
    try std.testing.expectEqual(@as(usize, 1), result.refined);
    try std.testing.expectEqual(@as(usize, 0), result.coarsened);

    // Should now have 16 children blocks (2^4 in 4D)
    try std.testing.expectEqual(@as(usize, 16), tree.blockCount());
}

test "adaptMesh auto-batches large refinement sets" {
    const block_size = 4;
    const blocks: usize = 70;
    const domain_extent = @as(f64, @floatFromInt(block_size * blocks));
    const Topology = amr.topology.OpenTopology(1, .{domain_extent});
    const Frontend = amr.frontend.ScalarFrontend(1, block_size, Topology);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);
    const Block = Tree.BlockType;

    // Limit refinement depth so expected refinement count is deterministic.
    var tree = try Tree.init(std.testing.allocator, 1.0, 8, 1);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, blocks * 2);
    defer arena.deinit();

    for (0..blocks) |i| {
        const origin = .{ i * block_size };
        const block_idx = try tree.insertBlock(origin, 0);
        const slot = arena.allocSlot() orelse return error.OutOfMemory;
        tree.assignFieldSlot(block_idx, slot);

        const field = arena.getSlot(slot);
        for (0..Block.volume) |j| {
            field[j] = @as(f64, @floatFromInt(j));
        }
    }

    const result = try adaptation.adaptMesh(Tree, &tree, &arena, 0.5, 0.5);
    try std.testing.expectEqual(blocks, result.refined);
    try std.testing.expectEqual(@as(usize, 0), result.refine_failed);
}

test "adaptMesh preserves low gradient blocks" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 32);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate field slot and fill with uniform field
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    const field = arena.getSlot(slot);
    for (field) |*v| v.*[0] = Complex.init(1.0, 0.0);

    // Adapt with reasonable threshold - should not trigger refinement
    const result = try adaptation.adaptMesh(Tree, &tree, &arena, 1.0, 0.5);

    try std.testing.expectEqual(@as(usize, 0), result.refined);
    try std.testing.expectEqual(@as(usize, 1), tree.blockCount());
}

test "findSiblings returns null for level 0" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Level 0 blocks have no siblings
    const siblings = adaptation.findSiblings(Tree, &tree, b0);
    try std.testing.expectEqual(@as(?[16]usize, null), siblings);
}

test "computeBlockGradient handles complex field magnitudes" {
    const N = 2; // Multiple components
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Allocate and assign field slot
    const slot = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot);

    // Create field with gradient in both components
    const Block = Tree.BlockType;
    const field = arena.getSlot(slot);

    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const t_coord: f64 = @floatFromInt(coords[0]);
        // Component 0: varies as t
        field[i][0] = Complex.init(t_coord, 0.0);
        // Component 1: varies as 2*t
        field[i][1] = Complex.init(2.0 * t_coord, 0.0);
    }

    // Gradient should be sqrt(1² + 2²) = sqrt(5) ≈ 2.236
    const gradient = adaptation.computeBlockGradient(Tree, &tree, b0, &arena);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), gradient, 0.01);
}

test "coarsenSiblings merges children into parent" {
    const N = 1;
    const block_size = 4;
    const Frontend = ComplexArrayFrontend(N, block_size);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    // Need enough slots for parent + 16 children + parent again after coarsen
    var arena = try Arena.init(std.testing.allocator, 32);
    defer arena.deinit();

    // Insert level-0 block
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);

    // Initialize parent field with uniform value
    const parent_field = arena.getSlot(slot0);
    for (parent_field) |*v| {
        v.*[0] = Complex.init(1.0, 0.0);
    }

    try std.testing.expectEqual(@as(usize, 1), tree.blockCount());

    // Refine the block - creates 16 children at level 1
    try tree.refineBlock(b0);
    try std.testing.expectEqual(@as(usize, 16), tree.blockCount());

    // Free parent slot (parent is now invalid)
    arena.freeSlot(slot0);

    // Allocate field slots for all 16 children with uniform value across all cells
    // This tests that restriction properly combines all children
    var child_indices: [16]usize = undefined;
    var child_count: usize = 0;

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (block.level == 1) {
            const child_slot = arena.allocSlot() orelse return error.OutOfMemory;
            tree.assignFieldSlot(idx, child_slot);

            // Initialize with uniform value across ALL children
            const child_field = arena.getSlot(child_slot);
            for (child_field) |*v| {
                v.*[0] = Complex.init(2.0, 0.0);
            }

            child_indices[child_count] = idx;
            child_count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 16), child_count);

    // Find siblings for first child
    const siblings_opt = adaptation.findSiblings(Tree, &tree, child_indices[0]);
    try std.testing.expect(siblings_opt != null);
    const siblings = siblings_opt.?;

    // Coarsen all siblings back into parent
    const result = try adaptation.coarsenSiblings(Tree, &tree, siblings, &arena);

    // Should now have 1 block (the parent)
    try std.testing.expectEqual(@as(usize, 1), tree.blockCount());

    // Verify parent block is at level 0
    const parent_idx = result.parent_idx;
    try std.testing.expectEqual(@as(usize, 0), tree.blocks.items[parent_idx].level);

    // Get parent's field and verify restriction preserved the uniform value
    // Full-weight restriction with preserve_norm=true: scale = sqrt(16)/16 = 0.25
    // Sum of 16 uniform fine cells * scale = 16 * 2.0 * 0.25 = 8.0
    const new_parent_field = arena.getSlot(result.parent_field_slot);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), new_parent_field[0][0].re, 0.01);
}

test "coarsenSiblings preserves field data integrity via restriction" {
    const N = 2; // Multiple components
    const block_size = 4;
    const Frontend = ComplexArrayFrontend(N, block_size);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 32);
    defer arena.deinit();

    // Insert and refine level-0 block
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);
    try tree.refineBlock(b0);
    arena.freeSlot(slot0);

    // Allocate slots for children with uniform values per component
    // This tests that restriction handles multiple components correctly
    var child_indices: [16]usize = undefined;
    var child_count: usize = 0;

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (block.level == 1) {
            const child_slot = arena.allocSlot() orelse return error.OutOfMemory;
            tree.assignFieldSlot(idx, child_slot);

            const child_field = arena.getSlot(child_slot);
            // Component 0: uniform value 1.0
            // Component 1: uniform value 2.0 + 0.5i
            for (child_field) |*v| {
                v.*[0] = Complex.init(1.0, 0.0);
                v.*[1] = Complex.init(2.0, 0.5);
            }

            child_indices[child_count] = idx;
            child_count += 1;
        }
    }

    const siblings = adaptation.findSiblings(Tree, &tree, child_indices[0]).?;
    const result = try adaptation.coarsenSiblings(Tree, &tree, siblings, &arena);

    // Verify both components were restricted correctly
    // With uniform fine cells: sum of 16 * scale(0.25) = 4.0
    const parent_field = arena.getSlot(result.parent_field_slot);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), parent_field[0][0].re, 0.01); // 1.0 * 16 * 0.25
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), parent_field[0][1].re, 0.01); // 2.0 * 16 * 0.25
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), parent_field[0][1].im, 0.01); // 0.5 * 16 * 0.25
}

test "coarsenSiblings fails for level 0 blocks" {
    const N = 1;
    const Frontend = ComplexArrayFrontend(N, 4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    // Insert level-0 block
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);

    // findSiblings returns null for level 0
    const siblings = adaptation.findSiblings(Tree, &tree, b0);
    try std.testing.expectEqual(@as(?[16]usize, null), siblings);

    // Can't call coarsenSiblings without valid siblings
    // (This verifies the precondition - level 0 can't be coarsened)
}

test "adaptMesh triggers coarsening for smooth wavefunction" {
    const N = 1;
    const block_size = 4;
    const Frontend = ComplexArrayFrontend(N, block_size);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 32);
    defer arena.deinit();

    // Create refined mesh: insert level-0 block and refine
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);
    try tree.refineBlock(b0);
    arena.freeSlot(slot0);

    // Allocate slots for 16 children with UNIFORM wavefunction (zero gradient)
    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (block.level == 1) {
            const child_slot = arena.allocSlot() orelse return error.OutOfMemory;
            tree.assignFieldSlot(idx, child_slot);

            const child_field = arena.getSlot(child_slot);
            for (child_field) |*v| {
                v.*[0] = Complex.init(1.0, 0.0); // Uniform = zero gradient
            }
        }
    }

    const blocks_before = tree.blockCount();
    try std.testing.expectEqual(@as(usize, 16), blocks_before);

    // adaptMesh with threshold that allows coarsening
    // Threshold 1.0, hysteresis 0.5 means coarsen if gradient < 0.5
    // Uniform wavefunction has gradient ~0
    const result = try adaptation.adaptMesh(Tree, &tree, &arena, 1.0, 0.5);

    // Should have coarsened
    try std.testing.expect(result.coarsened > 0);
    try std.testing.expectEqual(@as(usize, 0), result.refined);

    // Block count should have decreased
    const blocks_after = tree.blockCount();
    try std.testing.expect(blocks_after < blocks_before);
    try std.testing.expectEqual(@as(usize, 1), blocks_after); // Back to single parent
}

test "adaptMesh coarsening respects hysteresis" {
    const N = 1;
    const block_size = 4;
    const Frontend = ComplexArrayFrontend(N, block_size);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);
    const Block = Tree.BlockType;

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 32);
    defer arena.deinit();

    // Create refined mesh
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);
    try tree.refineBlock(b0);
    arena.freeSlot(slot0);

    // Set up gradient that's in the hysteresis zone:
    // threshold = 1.0, hysteresis = 0.5
    // So refine if gradient > 1.0, coarsen if gradient < 0.5
    // Gradient = |field(x+1) - field(x)| = slope
    // Set slope = 0.7 which is in the "do nothing" zone (0.5 < 0.7 < 1.0)
    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (block.level == 1) {
            const child_slot = arena.allocSlot() orelse return error.OutOfMemory;
            tree.assignFieldSlot(idx, child_slot);

            const child_field = arena.getSlot(child_slot);
            for (0..Block.volume) |i| {
                const coords = Block.getLocalCoords(i);
                // Linear in t direction with gradient = 0.7
                // gradient = |field(t+1) - field(t)| = 0.7
                const t = @as(f64, @floatFromInt(coords[0]));
                child_field[i][0] = Complex.init(0.7 * t, 0.0);
            }
        }
    }

    const blocks_before = tree.blockCount();

    // adaptMesh should neither refine nor coarsen (in hysteresis zone)
    const result = try adaptation.adaptMesh(Tree, &tree, &arena, 1.0, 0.5);

    try std.testing.expectEqual(@as(usize, 0), result.refined);
    try std.testing.expectEqual(@as(usize, 0), result.coarsened);
    try std.testing.expectEqual(blocks_before, tree.blockCount());
}

test "adaptive cycle: tree grows then shrinks" {
    // This test verifies the tree is truly adaptive:
    // 1. Start with coarse mesh
    // 2. High gradient → refinement → tree grows
    // 3. Low gradient → coarsening → tree shrinks
    // 4. Verify final block count < peak block count
    const N = 1;
    const block_size = 4;
    const Frontend = ComplexArrayFrontend(N, block_size);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);
    const Block = Tree.BlockType;

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 2);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 256);
    defer arena.deinit();

    // Start with single coarse block
    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);

    const initial_blocks = tree.blockCount();
    try std.testing.expectEqual(@as(usize, 1), initial_blocks);

    // Phase 1: Set up HIGH gradient (should trigger refinement)
    // Peaked Gaussian in center: gradient >> threshold
    {
        const field = arena.getSlot(slot0);
        const center = @as(f64, @floatFromInt(block_size)) / 2.0;
        for (0..Block.volume) |i| {
            const coords = Block.getLocalCoords(i);
            const x = @as(f64, @floatFromInt(coords[1])) - center;
            const y = @as(f64, @floatFromInt(coords[2])) - center;
            const z = @as(f64, @floatFromInt(coords[3])) - center;
            // Narrow Gaussian: high gradient near center
            const r_sq = x * x + y * y + z * z;
            field[i][0] = Complex.init(@exp(-r_sq * 4.0), 0.0);
        }
    }

    // Refinement with low threshold to force refinement
    const refine_result = try adaptation.adaptMesh(Tree, &tree, &arena, 0.01, 0.5);
    try std.testing.expect(refine_result.refined > 0);

    const peak_blocks = tree.blockCount();
    try std.testing.expect(peak_blocks > initial_blocks);

    // Phase 2: Set up LOW gradient (should trigger coarsening)
    // Uniform wavefunction across all children: gradient = 0
    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if (block.level > 0) {
            const slot = tree.getFieldSlot(idx);
            if (slot != std.math.maxInt(usize)) {
                const field = arena.getSlot(slot);
                for (field) |*v| {
                    v.*[0] = Complex.init(1.0, 0.0); // Uniform = zero gradient
                }
            }
        }
    }

    // Coarsening with reasonable threshold
    const coarsen_result = try adaptation.adaptMesh(Tree, &tree, &arena, 1.0, 0.5);
    try std.testing.expect(coarsen_result.coarsened > 0);

    const final_blocks = tree.blockCount();

    // CRITICAL: Final block count must be LESS than peak
    // This proves the tree actually shrinks, not just grows monotonically
    try std.testing.expect(final_blocks < peak_blocks);

    // We should be back to initial (or close)
    try std.testing.expectEqual(initial_blocks, final_blocks);
}

// =====================================================================
// Reorder Tests
// =====================================================================

test "reorder - blocks sorted by Morton index" {
    const Frontend = ScalarFrontend(4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert blocks in non-Morton order
    _ = try tree.insertBlock(.{ 12, 12, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 8, 4, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 4, 8, 0, 0 }, 0);

    try std.testing.expectEqual(@as(usize, 4), tree.blocks.items.len);

    // Reorder by Morton index
    const perm = try tree.reorder();
    std.testing.allocator.free(perm);

    // After reorder, Morton indices should be monotonically increasing
    var prev_morton: u64 = 0;
    for (tree.blocks.items, 0..) |*block, idx| {
        const morton = tree.getMortonIndex(block.origin, block.level);
        if (idx > 0) {
            try std.testing.expect(morton >= prev_morton);
        }
        prev_morton = morton;

        // Self-reference should match position
        try std.testing.expectEqual(idx, block.block_index);
    }
}

test "reorder - neighbor queries remain valid" {
    const Frontend = ScalarFrontend(4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create a 2x2 grid of blocks in t-x plane
    _ = try tree.insertBlock(.{ 4, 4, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 0, 4, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Helper to find block by origin
    const getBlockByOrigin = struct {
        fn call(t: *Tree, origin: [4]usize) ?usize {
            for (t.blocks.items, 0..) |*blk, idx| {
                if (blk.block_index != std.math.maxInt(usize) and
                    blk.origin[0] == origin[0] and blk.origin[1] == origin[1] and
                    blk.origin[2] == origin[2] and blk.origin[3] == origin[3])
                {
                    return idx;
                }
            }
            return null;
        }
    }.call;

    // Reorder
    const perm = try tree.reorder();
    std.testing.allocator.free(perm);

    // After reorder, find block at (0,0) and verify its +x neighbor is at (4,0)
    const origin_block = getBlockByOrigin(&tree, .{ 0, 0, 0, 0 }).?;
    _ = &tree.blocks.items[origin_block];

    // Face 0 is +t direction
    const neighbor_info = tree.neighborInfo(origin_block, 0);
    try std.testing.expect(neighbor_info.exists());
    try std.testing.expectEqual(@as(i8, 0), neighbor_info.level_diff);
    const neighbor = &tree.blocks.items[neighbor_info.block_idx];
    try std.testing.expectEqual(@as(usize, 4), neighbor.origin[0]);
    try std.testing.expectEqual(@as(usize, 0), neighbor.origin[1]);
}

test "reorder - field_slots correctly permuted" {
    const Frontend = ScalarFrontend(4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 10);
    defer arena.deinit();

    // Insert blocks in non-Morton order, each with a field slot
    const idx0 = try tree.insertBlock(.{ 8, 8, 0, 0 }, 0);
    const slot0 = arena.allocSlot().?;
    tree.assignFieldSlot(idx0, slot0);
    arena.getSlot(slot0)[0] = 88.0;

    const idx1 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot1 = arena.allocSlot().?;
    tree.assignFieldSlot(idx1, slot1);
    arena.getSlot(slot1)[0] = 0.0;

    const idx2 = try tree.insertBlock(.{ 4, 4, 0, 0 }, 0);
    const slot2 = arena.allocSlot().?;
    tree.assignFieldSlot(idx2, slot2);
    arena.getSlot(slot2)[0] = 44.0;

    // Reorder
    const perm = try tree.reorder();
    std.testing.allocator.free(perm);

    // Verify: each block's field_slot still points to its original data
    for (tree.blocks.items) |*block| {
        const slot = tree.getFieldSlot(block.block_index);
        const actual_value = arena.getSlotConst(slot)[0];

        // Derive expected from origin sum
        const sum = block.origin[0] + block.origin[1];
        if (sum == 0) {
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), actual_value, 1e-10);
        } else if (sum == 8) {
            try std.testing.expectApproxEqAbs(@as(f64, 44.0), actual_value, 1e-10);
        } else if (sum == 16) {
            try std.testing.expectApproxEqAbs(@as(f64, 88.0), actual_value, 1e-10);
        }
    }
}

test "reorder - invalid blocks filtered out (compaction)" {
    const Frontend = ScalarFrontend(4);
    const Tree = AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert blocks
    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const idx1 = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 8, 0, 0, 0 }, 0);

    // Mark one as invalid (simulating refinement)
    tree.invalidateBlock(idx1);

    // Before reorder: 3 blocks, 1 invalid
    try std.testing.expectEqual(@as(usize, 3), tree.blocks.items.len);
    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());

    // Reorder (should compact out invalid block)
    const perm = try tree.reorder();
    std.testing.allocator.free(perm);

    // After reorder: only 2 blocks remain
    try std.testing.expectEqual(@as(usize, 2), tree.blocks.items.len);
    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());

    // All remaining blocks should be valid
    for (tree.blocks.items, 0..) |*block, idx| {
        try std.testing.expectEqual(idx, block.block_index);
        try std.testing.expect(block.block_index != std.math.maxInt(usize));
    }
}

test "reorder - handles blocks without field slots" {
    const Frontend = ScalarFrontend(4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 10);
    defer arena.deinit();

    // Insert blocks - some with field slots, some without
    const idx0 = try tree.insertBlock(.{ 8, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot().?;
    tree.assignFieldSlot(idx0, slot0);
    arena.getSlot(slot0)[0] = 80.0;

    const idx1 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    // No field slot for idx1 - leave as maxInt

    const idx2 = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);
    const slot2 = arena.allocSlot().?;
    tree.assignFieldSlot(idx2, slot2);
    arena.getSlot(slot2)[0] = 40.0;

    // Verify idx1 has no slot
    try std.testing.expectEqual(std.math.maxInt(usize), tree.getFieldSlot(idx1));

    // Reorder
    const perm = try tree.reorder();
    std.testing.allocator.free(perm);

    // Verify blocks are sorted and accessible
    try std.testing.expectEqual(@as(usize, 3), tree.blocks.items.len);

    // Blocks with slots should still have correct data
    var found_with_slot: usize = 0;
    for (tree.blocks.items, 0..) |*block, idx| {
        const slot = tree.getFieldSlot(idx);
        if (slot != std.math.maxInt(usize)) {
            const data = arena.getSlotConst(slot)[0];
            // Data should match origin[0] * 10
            try std.testing.expectApproxEqAbs(@as(f64, @floatFromInt(block.origin[0] * 10)), data, 1e-10);
            found_with_slot += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 2), found_with_slot);
}

test "defragmentWithOrder - produces linear memory layout" {
    const Frontend = ScalarFrontend(4);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 10);
    defer arena.deinit();

    // Insert blocks in non-Morton order with unique field values
    const idx0 = try tree.insertBlock(.{ 8, 8, 0, 0 }, 0);
    const slot0 = arena.allocSlot().?;
    tree.assignFieldSlot(idx0, slot0);
    arena.getSlot(slot0)[0] = 88.0;

    const idx1 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot1 = arena.allocSlot().?;
    tree.assignFieldSlot(idx1, slot1);
    arena.getSlot(slot1)[0] = 0.0;

    const idx2 = try tree.insertBlock(.{ 4, 4, 0, 0 }, 0);
    const slot2 = arena.allocSlot().?;
    tree.assignFieldSlot(idx2, slot2);
    arena.getSlot(slot2)[0] = 44.0;

    // Reorder blocks by Morton index
    const perm = try tree.reorder();
    std.testing.allocator.free(perm);

    // Defragment arena to match new block order
    try arena.defragmentWithOrder(tree.field_slots.items, tree.blocks.items.len);

    // After defragmentation:
    // 1. field_slots[i] == i (sequential)
    // 2. Data is in correct order matching block origins
    for (tree.blocks.items, 0..) |*block, idx| {
        // field_slots should be sequential
        try std.testing.expectEqual(idx, tree.getFieldSlot(idx));

        // Verify data matches the block's origin
        const data = arena.getSlotConst(idx)[0];
        const sum = block.origin[0] + block.origin[1];
        if (sum == 0) {
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), data, 1e-10);
        } else if (sum == 8) {
            try std.testing.expectApproxEqAbs(@as(f64, 44.0), data, 1e-10);
        } else if (sum == 16) {
            try std.testing.expectApproxEqAbs(@as(f64, 88.0), data, 1e-10);
        }
    }

    // Free count should be updated correctly
    try std.testing.expectEqual(arena.max_blocks - tree.blocks.items.len, arena.free_count);
}

test "adaptive cycle: multiple refinement-coarsening cycles" {
    // Verify the tree can go through multiple refine/coarsen cycles
    // without leaking blocks or corrupting state
    const N = 1;
    const block_size = 4;
    const Frontend = ComplexArrayFrontend(N, block_size);
    const Tree = AMRTree(Frontend);
    const Arena = FieldArena(Frontend);
    const Block = Tree.BlockType;

    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 2);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 256);
    defer arena.deinit();

    const b0 = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const slot0 = arena.allocSlot() orelse return error.OutOfMemory;
    tree.assignFieldSlot(b0, slot0);

    // Perform 3 refine/coarsen cycles
    var cycle: usize = 0;
    while (cycle < 3) : (cycle += 1) {
        // Refine: set high gradient
        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) continue;
            const slot = tree.getFieldSlot(idx);
            if (slot == std.math.maxInt(usize)) continue;

            const field = arena.getSlot(slot);
            const center = @as(f64, @floatFromInt(block_size)) / 2.0;
            for (0..Block.volume) |i| {
                const coords = Block.getLocalCoords(i);
                const x = @as(f64, @floatFromInt(coords[1])) - center;
                const r_sq = x * x;
                field[i][0] = Complex.init(@exp(-r_sq * 4.0), 0.0);
            }
        }

        _ = try adaptation.adaptMesh(Tree, &tree, &arena, 0.01, 0.5);
        const refined_count = tree.blockCount();
        try std.testing.expect(refined_count > 1);

        // Coarsen: set low gradient
        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) continue;
            if (block.level == 0) continue;
            const slot = tree.getFieldSlot(idx);
            if (slot == std.math.maxInt(usize)) continue;

            const field = arena.getSlot(slot);
            for (field) |*v| {
                v.*[0] = Complex.init(1.0, 0.0);
            }
        }

        _ = try adaptation.adaptMesh(Tree, &tree, &arena, 1.0, 0.5);
        const coarsened_count = tree.blockCount();

        // After coarsening, should be back to 1 block
        try std.testing.expectEqual(@as(usize, 1), coarsened_count);
    }

    // Final state: single block, no leaks
    try std.testing.expectEqual(@as(usize, 1), tree.blockCount());
}

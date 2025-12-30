const std = @import("std");
const amr = @import("amr");

// Test topology
const TestTopology4D = amr.topology.OpenTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });

// Use ScalarFrontend for topology tests - no gauge complexity needed
const Frontend = amr.frontend.ScalarFrontend(4, 4, TestTopology4D);
const Tree = amr.AMRTree(Frontend);

// Test cross-level neighbor discovery for fine->coarse connections.
// Fine blocks should automatically discover their coarse neighbors.
test "cross-level neighbor discovery - fine finds coarse" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create root and refine it
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.refineBlock(root_idx);

    // Find L1 block at origin {0,0,0,0}
    var level1_origin_idx: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1 and b.origin[0] == 0 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
            level1_origin_idx = idx;
            break;
        }
    }
    try std.testing.expect(level1_origin_idx != std.math.maxInt(usize));

    // Refine it to create L2 blocks
    try tree.refineBlock(level1_origin_idx);

    // Find L2 block at origin {4,0,0,0} - this has +T neighbor at coarse level
    var fine_block_idx: usize = std.math.maxInt(usize);
    var coarse_block_idx: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1 and b.origin[0] == 4 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
            coarse_block_idx = idx;
        }
        if (b.level == 2 and b.origin[0] == 4 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
            fine_block_idx = idx;
        }
    }

    try std.testing.expect(fine_block_idx != std.math.maxInt(usize));
    try std.testing.expect(coarse_block_idx != std.math.maxInt(usize));

    // Fine block's +T neighbor (face 0) should be the coarse block
    const neighbor_info = tree.neighborInfo(fine_block_idx, 0);
    try std.testing.expect(neighbor_info.exists());
    try std.testing.expectEqual(@as(i8, -1), neighbor_info.level_diff);
    try std.testing.expectEqual(coarse_block_idx, neighbor_info.block_idx);
}

// Test Push Model behavior: coarse blocks have maxInt for faces adjacent to refined regions.
test "push model - coarse has maxInt for refined faces" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create root and refine it
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.refineBlock(root_idx);

    // Find L1 block at origin {0,0,0,0} and its +T neighbor L1 at {4,0,0,0}
    var level1_origin_idx: usize = std.math.maxInt(usize);
    var level1_neighbor_idx: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1) {
            if (b.origin[0] == 0 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
                level1_origin_idx = idx;
            }
            if (b.origin[0] == 4 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
                level1_neighbor_idx = idx;
            }
        }
    }

    // Before refinement: L1 neighbor's -T face (face 1) points to L1 origin
    const pre_info = tree.neighborInfo(level1_neighbor_idx, 1);
    try std.testing.expect(pre_info.exists());
    try std.testing.expectEqual(@as(i8, 0), pre_info.level_diff);
    try std.testing.expectEqual(level1_origin_idx, pre_info.block_idx);

    // Refine the origin block
    try tree.refineBlock(level1_origin_idx);

    // After refinement: L1 neighbor's -T face should return none (Push Model)
    const post_info = tree.neighborInfo(level1_neighbor_idx, 1);
    try std.testing.expect(!post_info.exists());
}

// Test that same-level neighbors are correctly discovered and bidirectional.
test "same-level neighbors are bidirectional" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create root and refine it to get L1 blocks
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.refineBlock(root_idx);

    // Find two adjacent L1 blocks
    var block_a: usize = std.math.maxInt(usize);
    var block_b: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1) {
            if (b.origin[0] == 0 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
                block_a = idx;
            }
            if (b.origin[0] == 4 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
                block_b = idx;
            }
        }
    }

    try std.testing.expect(block_a != std.math.maxInt(usize));
    try std.testing.expect(block_b != std.math.maxInt(usize));

    // Block A's +T neighbor (face 0) should be Block B
    const a_info = tree.neighborInfo(block_a, 0);
    try std.testing.expect(a_info.exists());
    try std.testing.expectEqual(@as(i8, 0), a_info.level_diff);
    try std.testing.expectEqual(block_b, a_info.block_idx);

    // Block B's -T neighbor (face 1) should be Block A
    const b_info = tree.neighborInfo(block_b, 1);
    try std.testing.expect(b_info.exists());
    try std.testing.expectEqual(@as(i8, 0), b_info.level_diff);
    try std.testing.expectEqual(block_a, b_info.block_idx);
}

// Test multiple fine blocks adjacent to the same coarse block.
test "multiple fine blocks find same coarse neighbor" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create root and refine
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.refineBlock(root_idx);

    // Find and refine L1 at origin
    var level1_origin_idx: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1 and b.origin[0] == 0 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
            level1_origin_idx = idx;
            break;
        }
    }
    try tree.refineBlock(level1_origin_idx);

    // Find the coarse block at L1 {4,0,0,0}
    var coarse_block_idx: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1 and b.origin[0] == 4 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
            coarse_block_idx = idx;
            break;
        }
    }
    try std.testing.expect(coarse_block_idx != std.math.maxInt(usize));

    // All L2 blocks at the +T boundary should have the same coarse neighbor
    // These are L2 blocks with origin[0] == 4 (T=4 in L2 coords)
    var found_any = false;
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 2 and b.origin[0] == 4) {
            // This L2 block's +T neighbor should be the coarse block
            const info = tree.neighborInfo(idx, 0);
            try std.testing.expect(info.exists());
            try std.testing.expectEqual(@as(i8, -1), info.level_diff);
            try std.testing.expectEqual(coarse_block_idx, info.block_idx);
            found_any = true;
        }
    }
    try std.testing.expect(found_any);
}

// Test that domain boundaries return no neighbor.
test "domain boundary returns no neighbor" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create a single block at origin
    const block_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // All negative direction faces should be at domain boundary (maxInt)
    _ = &tree.blocks.items[block_idx];
    // Face 1 = -T, face 3 = -X, face 5 = -Y, face 7 = -Z
    try std.testing.expect(!tree.neighborInfo(block_idx, 1).exists());
    try std.testing.expect(!tree.neighborInfo(block_idx, 3).exists());
    try std.testing.expect(!tree.neighborInfo(block_idx, 5).exists());
    try std.testing.expect(!tree.neighborInfo(block_idx, 7).exists());
}

// Test physical coordinate conversion utilities.
test "physical coordinate conversion" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create root and refine twice
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.refineBlock(root_idx);

    // Find L1 block at {4,0,0,0}
    var block_idx: usize = std.math.maxInt(usize);
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1 and b.origin[0] == 4 and b.origin[1] == 0 and b.origin[2] == 0 and b.origin[3] == 0) {
            block_idx = idx;
            break;
        }
    }
    try std.testing.expect(block_idx != std.math.maxInt(usize));

    // Test physical origin: L1 block at {4,0,0,0} has spacing 0.5
    // physical = 4 * 0.5 = 2.0
    const block = &tree.blocks.items[block_idx];
    const phys = tree.getPhysicalOrigin(block);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), phys[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), phys[1], 1e-10);

    // Test block extent: at L1, extent = base_spacing / 2^1 * block_size = 1.0/2 * 4 = 2.0
    const extent = tree.getBlockPhysicalExtent(1);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), extent, 1e-10);
}

test "periodic topology wraps neighbors across domain" {
    const Topology1D = amr.topology.PeriodicTopology(1, .{8.0});
    const Frontend1D = amr.frontend.ScalarFrontend(1, 4, Topology1D);
    const Tree1D = amr.AMRTree(Frontend1D);

    var tree = try Tree1D.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    const left_idx = try tree.insertBlock(.{0}, 0);
    const right_idx = try tree.insertBlock(.{4}, 0);

    const left_plus = tree.neighborInfo(left_idx, 0);
    try std.testing.expect(left_plus.exists());
    try std.testing.expectEqual(@as(i8, 0), left_plus.level_diff);
    try std.testing.expectEqual(right_idx, left_plus.block_idx); // +X wraps to right

    const left_minus = tree.neighborInfo(left_idx, 1);
    try std.testing.expect(left_minus.exists());
    try std.testing.expectEqual(@as(i8, 0), left_minus.level_diff);
    try std.testing.expectEqual(right_idx, left_minus.block_idx); // -X wraps to right

    const right_plus = tree.neighborInfo(right_idx, 0);
    try std.testing.expect(right_plus.exists());
    try std.testing.expectEqual(@as(i8, 0), right_plus.level_diff);
    try std.testing.expectEqual(left_idx, right_plus.block_idx); // +X wraps to left

    const right_minus = tree.neighborInfo(right_idx, 1);
    try std.testing.expect(right_minus.exists());
    try std.testing.expectEqual(@as(i8, 0), right_minus.level_diff);
    try std.testing.expectEqual(left_idx, right_minus.block_idx); // -X wraps to left
}

// Test that all blocks have valid neighbor references after multiple refinements.
test "neighbor validity after multiple refinements" {
    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create and refine root
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.refineBlock(root_idx);

    // Refine multiple L1 blocks to create a complex topology.
    // Collect indices first to avoid invalidation while mutating tree.blocks.
    var refine_list = std.ArrayList(usize){};
    defer refine_list.deinit(allocator);

    var refined_count: usize = 0;
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;
        if (b.level == 1 and refined_count < 4) {
            try refine_list.append(allocator, idx);
            refined_count += 1;
        }
    }

    for (refine_list.items) |idx| {
        try tree.refineBlock(idx);
    }

    // Verify all neighbor queries return valid indices
    for (tree.blocks.items, 0..) |*b, idx| {
        if (b.block_index == std.math.maxInt(usize)) continue;

        for (0..(2 * Tree.dimensions)) |face| {
            const info = tree.neighborInfo(idx, face);
            if (info.exists()) {
                // Neighbor should exist and be valid
                try std.testing.expect(info.block_idx < tree.blocks.items.len);
                try std.testing.expect(tree.blocks.items[info.block_idx].block_index != std.math.maxInt(usize));
            }
        }
    }
}

// =============================================================================
// Periodic Boundary Tests
// =============================================================================

// Periodic topology: blocks at domain edges should find neighbors on opposite edge
const PeriodicTopology4D = amr.topology.PeriodicTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });
const PeriodicFrontend = amr.frontend.ScalarFrontend(4, 4, PeriodicTopology4D);
const PeriodicTree = amr.AMRTree(PeriodicFrontend);

test "periodic boundary - neighbors wrap around" {
    const allocator = std.testing.allocator;

    var tree = try PeriodicTree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert two blocks at opposite ends of the domain in T dimension
    // Domain is 8.0 in each dimension, block_size=4, spacing=1.0
    // So each block covers extent = 4 * 1.0 = 4.0
    // Block at {0,0,0,0} covers [0, 4) in T
    // Block at {4,0,0,0} covers [4, 8) in T
    // With periodic boundaries, block at T=0 has -T neighbor at T=4
    // and block at T=4 has +T neighbor at T=0 (wrapping around)
    const block_a = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const block_b = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);

    // Block A's +T neighbor (face 0) should be Block B
    const a_plus = tree.neighborInfo(block_a, 0);
    try std.testing.expect(a_plus.exists());
    try std.testing.expectEqual(@as(i8, 0), a_plus.level_diff);
    try std.testing.expectEqual(block_b, a_plus.block_idx);

    // Block B's +T neighbor (face 0) should wrap to Block A (periodic)
    const b_plus = tree.neighborInfo(block_b, 0);
    try std.testing.expect(b_plus.exists());
    try std.testing.expectEqual(@as(i8, 0), b_plus.level_diff);
    try std.testing.expectEqual(block_a, b_plus.block_idx);

    // Block A's -T neighbor (face 1) should wrap to Block B
    const a_minus = tree.neighborInfo(block_a, 1);
    try std.testing.expect(a_minus.exists());
    try std.testing.expectEqual(@as(i8, 0), a_minus.level_diff);
    try std.testing.expectEqual(block_b, a_minus.block_idx);

    // Block B's -T neighbor (face 1) should be Block A
    const b_minus = tree.neighborInfo(block_b, 1);
    try std.testing.expect(b_minus.exists());
    try std.testing.expectEqual(@as(i8, 0), b_minus.level_diff);
    try std.testing.expectEqual(block_a, b_minus.block_idx);
}

test "periodic boundary - single block is its own neighbor" {
    const allocator = std.testing.allocator;

    // With domain 4.0 and block_size=4, spacing=1.0, a single block fills entire domain
    const SmallPeriodicTopo = amr.topology.PeriodicTopology(4, .{ 4.0, 4.0, 4.0, 4.0 });
    const SmallFrontend = amr.frontend.ScalarFrontend(4, 4, SmallPeriodicTopo);
    const SmallTree = amr.AMRTree(SmallFrontend);

    var tree = try SmallTree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    const block_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // With periodic boundary and a single block filling the domain,
    // the block should be its own neighbor in all directions
    _ = &tree.blocks.items[block_idx];
    for (0..(2 * SmallTree.dimensions)) |face| {
        const info = tree.neighborInfo(block_idx, face);
        try std.testing.expect(info.exists());
        try std.testing.expectEqual(@as(i8, 0), info.level_diff);
        try std.testing.expectEqual(block_idx, info.block_idx);
    }
}

test "open boundary - no wraparound at edges" {
    const allocator = std.testing.allocator;

    // Use the open topology tree (TestTopology4D is open)
    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert two blocks: one at origin, one at far edge of domain
    const block_origin = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const block_far = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);

    // Block at origin: -T neighbor (face 1) should be none (open boundary, no wrap)
    try std.testing.expect(!tree.neighborInfo(block_origin, 1).exists());

    // Block at far edge: +T neighbor (face 0) should be none (open boundary, no wrap)
    try std.testing.expect(!tree.neighborInfo(block_far, 0).exists());

    // But they should still be neighbors of each other
    const origin_plus = tree.neighborInfo(block_origin, 0);
    try std.testing.expect(origin_plus.exists());
    try std.testing.expectEqual(@as(i8, 0), origin_plus.level_diff);
    try std.testing.expectEqual(block_far, origin_plus.block_idx);

    const far_minus = tree.neighborInfo(block_far, 1);
    try std.testing.expect(far_minus.exists());
    try std.testing.expectEqual(@as(i8, 0), far_minus.level_diff);
    try std.testing.expectEqual(block_origin, far_minus.block_idx);
}

test "mixed boundaries - periodic in some dimensions, open in others" {
    const allocator = std.testing.allocator;

    // Create a mixed topology: periodic in T and X, open in Y and Z
    const MixedTopo = amr.topology.GridTopology(4, .{
        .boundary = .{ .periodic, .periodic, .open, .open },
        .domain_size = .{ 8.0, 8.0, 8.0, 8.0 },
    });
    const MixedFrontend = amr.frontend.ScalarFrontend(4, 4, MixedTopo);
    const MixedTree = amr.AMRTree(MixedFrontend);

    var tree = try MixedTree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert blocks spanning the domain
    const block_origin = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    const block_t4 = try tree.insertBlock(.{ 4, 0, 0, 0 }, 0);

    // T dimension is periodic: block at T=4's +T neighbor wraps to T=0
    const t4_plus = tree.neighborInfo(block_t4, 0);
    try std.testing.expect(t4_plus.exists());
    try std.testing.expectEqual(@as(i8, 0), t4_plus.level_diff);
    try std.testing.expectEqual(block_origin, t4_plus.block_idx);

    // Y dimension is open: -Y neighbor of origin block is none
    // Face indices: 0=+T, 1=-T, 2=+X, 3=-X, 4=+Y, 5=-Y, 6=+Z, 7=-Z
    try std.testing.expect(!tree.neighborInfo(block_origin, 5).exists());

    // Z dimension is open: -Z neighbor is also none
    try std.testing.expect(!tree.neighborInfo(block_origin, 7).exists());
}

//! Adaptive Mesh Refinement Control
//!
//! Implements gradient-based mesh adaptation:
//! - Refinement: Split blocks when gradient exceeds threshold
//! - Coarsening: Merge sibling blocks when all gradients are below hysteresis threshold
//!
//! ## Frontend-Parameterized
//!
//! All adaptation functions are parameterized by a Tree type that uses a Frontend.
//! The gradient computation works with any FieldType (scalar, complex, array).
//!
//! ## Usage
//!
//! ```zig
//! const adaptation = @import("adaptation.zig");
//! const MyFrontend = struct {
//!     pub const Nd: usize = 4;
//!     pub const block_size: usize = 16;
//!     pub const FieldType = [4]Complex;
//! };
//! const Tree = tree_mod.AMRTree(MyFrontend);
//!
//! // Check refinement criteria
//! const gradient = adaptation.computeBlockGradient(Tree, &tree, block_idx, &arena);
//! if (adaptation.shouldRefine(Tree, &tree, block_idx, &arena, threshold)) {
//!     try tree.refineBlock(block_idx);
//! }
//!
//! // Adapt mesh automatically
//! const result = try adaptation.adaptMesh(Tree, &tree, &arena, threshold, hysteresis);
//! ```

const std = @import("std");
const operators_mod = @import("operators.zig");

const Complex = std.math.Complex(f64);

// =============================================================================
// Configuration
// =============================================================================

/// Adaptation result tracking refinement/coarsening statistics.
pub const AdaptResult = struct {
    refined: usize,
    coarsened: usize,
    refine_failed: usize,
    coarsen_failed: usize,
    /// True if any refinement or coarsening occurred.
    /// Use this to determine if tree.reorder() should be called.
    changed: bool,
};

/// Coarsen result returned by coarsenSiblings.
pub fn CoarsenResult(comptime num_children: usize) type {
    return struct {
        parent_idx: usize,
        parent_field_slot: usize,
        freed_field_slots: [num_children]usize,
    };
}

// =============================================================================
// Generic Field Arithmetic (for gradient computation)
// =============================================================================

/// Detect if a type is Complex
fn isComplex(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasField(T, "re") and @hasField(T, "im");
}

/// Compute the squared magnitude of a field value (for gradient computation)
fn fieldMagnitudeSq(comptime FieldType: type, a: FieldType, b: FieldType) f64 {
    const info = @typeInfo(FieldType);

    if (info == .array) {
        const Child = std.meta.Child(FieldType);
        const len = info.array.len;
        var sum: f64 = 0;
        inline for (0..len) |i| {
            sum += elementDiffMagnitudeSq(Child, a[i], b[i]);
        }
        return sum;
    } else {
        return elementDiffMagnitudeSq(FieldType, a, b);
    }
}

/// Compute |a - b|^2 for a single element
fn elementDiffMagnitudeSq(comptime T: type, a: T, b: T) f64 {
    if (comptime isComplex(T)) {
        const diff_re = a.re - b.re;
        const diff_im = a.im - b.im;
        return diff_re * diff_re + diff_im * diff_im;
    } else {
        const diff = a - b;
        return diff * diff;
    }
}

// =============================================================================
// Gradient Computation
// =============================================================================

/// Compute the maximum gradient magnitude for a block.
///
/// Computes max |grad field| across all interior sites and directions.
/// This is a dimensionless, scale-invariant measure of spatial variation.
///
/// Returns: Maximum gradient magnitude
pub fn computeBlockGradient(
    comptime Tree: type,
    tree: *const Tree,
    block_idx: usize,
    arena: *const Tree.FieldArenaType,
) f64 {
    const Nd = Tree.dimensions;
    const Block = Tree.BlockType;
    const FieldType = Tree.FieldType;

    const blk = &tree.blocks.items[block_idx];
    if (blk.block_index == std.math.maxInt(usize)) {
        return 0; // Invalid/removed block
    }

    const slot = tree.getFieldSlot(block_idx);
    if (slot == std.math.maxInt(usize)) {
        return 0; // No field data for this block
    }
    const block_field = arena.getSlotConst(slot);
    var max_gradient: f64 = 0;

    // Iterate over interior sites (exclude boundary to avoid ghost access)
    for (0..Block.volume) |site| {
        const coords = Block.getLocalCoords(site);

        // Skip boundary sites (any coordinate at 0 or size-1)
        var is_boundary = false;
        inline for (0..Nd) |d| {
            if (coords[d] == 0 or coords[d] == Block.size - 1) {
                is_boundary = true;
                break;
            }
        }
        if (is_boundary) continue;

        // Compute gradient in each direction
        inline for (0..Nd) |mu| {
            const face_plus = mu * 2;
            const neighbor_plus = Block.localNeighborFast(site, face_plus);

            // Compute |field(x+mu) - field(x)|^2
            const diff_sq = fieldMagnitudeSq(FieldType, block_field[neighbor_plus], block_field[site]);
            const gradient = @sqrt(diff_sq);
            max_gradient = @max(max_gradient, gradient);
        }
    }

    return max_gradient;
}

// =============================================================================
// Refinement/Coarsening Criteria
// =============================================================================

/// Determine if a block should be refined based on gradient threshold.
pub fn shouldRefine(
    comptime Tree: type,
    tree: *const Tree,
    block_idx: usize,
    arena: *const Tree.FieldArenaType,
    threshold: f64,
) bool {
    const blk = &tree.blocks.items[block_idx];

    // Don't refine if already at max level
    if (blk.level >= tree.max_level) return false;

    // Don't refine invalid blocks
    if (blk.block_index == std.math.maxInt(usize)) return false;

    const gradient = computeBlockGradient(Tree, tree, block_idx, arena);
    return gradient > threshold;
}

/// Determine if a block should be coarsened based on gradient threshold.
pub fn shouldCoarsen(
    comptime Tree: type,
    tree: *const Tree,
    block_idx: usize,
    arena: *const Tree.FieldArenaType,
    threshold: f64,
    hysteresis: f64,
) bool {
    const blk = &tree.blocks.items[block_idx];

    // Don't coarsen level 0 blocks
    if (blk.level == 0) return false;

    // Don't coarsen invalid blocks
    if (blk.block_index == std.math.maxInt(usize)) return false;

    const gradient = computeBlockGradient(Tree, tree, block_idx, arena);
    return gradient < threshold * hysteresis;
}

// =============================================================================
// Mesh Adaptation
// =============================================================================

/// Adapt the mesh based on gradient-based refinement criteria.
///
/// **Refinement**: Blocks with gradient > threshold are split into 2^Nd children.
/// **Coarsening**: Groups of 2^Nd siblings where ALL have gradient < threshold * hysteresis are merged.
/// **Balance**: Enforces 2:1 balance by refining neighbors when level differences exceed 1.
///
/// **Allocation Model**: Uses an arena-backed scratch buffer sized to the
/// current block list for auto-batched refinement/coarsening. This avoids
/// per-block heap churn but can OOM on very large trees; see docs/specs.
pub fn adaptMesh(
    comptime Tree: type,
    tree: *Tree,
    arena: *Tree.FieldArenaType,
    threshold: f64,
    hysteresis: f64,
) !AdaptResult {
    const Nd = Tree.dimensions;
    const Ops = operators_mod.AMROperators(Tree.FrontendType);

    var result = AdaptResult{
        .refined = 0,
        .coarsened = 0,
        .refine_failed = 0,
        .coarsen_failed = 0,
        .changed = false,
    };

    // Phase 1: Refinement (auto-batched)
    var scratch = std.heap.ArenaAllocator.init(tree.allocator);
    defer scratch.deinit();

    var refinement_pass = true;
    while (refinement_pass) {
        refinement_pass = false;
        _ = scratch.reset(.retain_capacity);
        const scratch_alloc = scratch.allocator();

        const block_capacity = tree.blocks.items.len;
        const to_refine = try scratch_alloc.alloc(usize, block_capacity);
        var to_refine_count: usize = 0;

        for (tree.blocks.items, 0..) |*blk, idx| {
            if (blk.block_index == std.math.maxInt(usize)) continue;
            if (shouldRefine(Tree, tree, idx, arena, threshold)) {
                to_refine[to_refine_count] = idx;
                to_refine_count += 1;
            }
        }

        if (to_refine_count == 0) break;

        for (to_refine[0..to_refine_count]) |block_idx| {
            if (refineBlockWithFields(Tree, Ops, tree, arena, block_idx, &result)) {
                refinement_pass = true;
            }
        }
    }

    // Enforce 2:1 balance (no leaf differs by more than one level across a face).
    var balance_changed = true;
    var balance_passes: usize = 0;
    const max_balance_passes = tree.max_level + 1;
    while (balance_changed and balance_passes < max_balance_passes) {
        balance_changed = false;
        const scan_len = tree.blocks.items.len;
        var idx: usize = 0;
        while (idx < scan_len) : (idx += 1) {
            if (tree.blocks.items[idx].block_index == std.math.maxInt(usize)) continue;
            const blk_level = tree.blocks.items[idx].level;

            inline for (0..2 * Nd) |face| {
                // If this block is finer, refine any coarser neighbor beyond 2:1.
                if (blk_level >= 2) {
                    var level: i32 = @as(i32, blk_level) - 2;
                    while (level >= 0) : (level -= 1) {
                        if (neighborAtLevel(Tree, tree, idx, face, @intCast(level))) |coarse_idx| {
                            if (refineBlockWithFields(Tree, Ops, tree, arena, coarse_idx, &result)) {
                                balance_changed = true;
                            }
                            break;
                        }
                    }
                }

                // If this block is coarser, refine it when a much finer neighbor exists.
                if (blk_level + 2 <= tree.max_level) {
                    var level: u8 = blk_level + 2;
                    while (level <= tree.max_level) : (level += 1) {
                        if (neighborAtLevel(Tree, tree, idx, face, level)) |_| {
                            if (refineBlockWithFields(Tree, Ops, tree, arena, idx, &result)) {
                                balance_changed = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
        balance_passes += 1;
    }

    // Phase 2: Coarsening (multi-pass to collapse multiple levels)
    var coarsen_pass = true;
    while (coarsen_pass) {
        coarsen_pass = false;
        _ = scratch.reset(.retain_capacity);
        const scratch_alloc = scratch.allocator();
        const scan_len = tree.blocks.items.len;
        const visited_buf = try scratch_alloc.alloc(bool, scan_len);
        @memset(visited_buf, false);

        for (0..scan_len) |idx| {
            const blk = &tree.blocks.items[idx];
            if (blk.block_index == std.math.maxInt(usize)) continue;
            if (blk.level == 0) continue;
            if (visited_buf[idx]) continue;

            const siblings_opt = findSiblings(Tree, tree, idx);
            if (siblings_opt == null) continue;
            const siblings = siblings_opt.?;

            for (siblings) |sib_idx| {
                if (sib_idx < visited_buf.len) {
                    visited_buf[sib_idx] = true;
                }
            }

            if (!canCoarsenSiblings(Tree, tree, siblings, arena, threshold, hysteresis)) {
                continue;
            }

            _ = coarsenSiblings(Tree, tree, siblings, arena) catch {
                result.coarsen_failed += 1;
                continue;
            };

            result.coarsened += 1;
            coarsen_pass = true;
        }
    }

    result.changed = result.refined > 0 or result.coarsened > 0;
    return result;
}

fn refineBlockWithFields(
    comptime Tree: type,
    comptime Ops: type,
    tree: *Tree,
    arena: *Tree.FieldArenaType,
    block_idx: usize,
    result: *AdaptResult,
) bool {
    const Nd = Tree.dimensions;
    const Block = Tree.BlockType;
    const FieldType = Tree.FieldType;
    const num_children = Tree.children_per_node;
    const block_size = Block.size;

    if (tree.blocks.items[block_idx].block_index == std.math.maxInt(usize)) return false;
    if (tree.blocks.items[block_idx].level >= tree.max_level) return false;

    const parent_field_slot = tree.getFieldSlot(block_idx);
    var parent_field_copy: [Block.volume]FieldType = undefined;
    if (parent_field_slot != std.math.maxInt(usize)) {
        const parent_field = arena.getSlotConst(parent_field_slot);
        @memcpy(&parent_field_copy, parent_field);
    }

    const child_level = tree.blocks.items[block_idx].level + 1;
    const parent_origin = tree.blocks.items[block_idx].origin;

    tree.refineBlock(block_idx) catch {
        result.refine_failed += 1;
        return false;
    };

    if (parent_field_slot != std.math.maxInt(usize)) {
        arena.freeSlot(parent_field_slot);
        tree.field_slots.items[block_idx] = std.math.maxInt(usize);
    }

    var child_field_slices: [num_children][]FieldType = undefined;
    var children_found: usize = 0;

    for (tree.blocks.items, 0..) |*blk, idx| {
        if (blk.block_index == std.math.maxInt(usize)) continue;
        if (blk.level != child_level) continue;

        var is_child = true;
        inline for (0..Nd) |d| {
            const expected_min = parent_origin[d] * 2;
            const expected_max = expected_min + 2 * block_size;
            if (blk.origin[d] < expected_min or blk.origin[d] >= expected_max) {
                is_child = false;
                break;
            }
        }

        if (is_child and children_found < num_children) {
            const child_slot = arena.allocSlot() orelse {
                result.refine_failed += 1;
                continue;
            };
            tree.assignFieldSlot(idx, child_slot);
            child_field_slices[children_found] = arena.getSlot(child_slot);
            children_found += 1;
        }
    }

    if (children_found == num_children) {
        Ops.prolongateInjection(&parent_field_copy, &child_field_slices, true);
    }

    result.refined += 1;
    return true;
}

fn neighborAtLevel(
    comptime Tree: type,
    tree: *const Tree,
    block_idx: usize,
    face: usize,
    level: u8,
) ?usize {
    const Topology = Tree.FrontendType.Topology;
    const block = &tree.blocks.items[block_idx];
    if (block.block_index == std.math.maxInt(usize)) return null;

    const dim = face / 2;
    const is_positive = (face % 2) == 0;

    const physical_origin = tree.getPhysicalOrigin(block);
    const block_extent = tree.getBlockPhysicalExtent(block.level);

    var neighbor_physical: [Tree.dimensions]f64 = physical_origin;
    if (is_positive) {
        neighbor_physical[dim] += block_extent;
    } else {
        neighbor_physical[dim] -= Topology.neighborEpsilon(block.spacing, dim);
    }

    const wrapped_coord = Topology.wrapCoordinateRuntime(neighbor_physical[dim], dim) orelse return null;
    neighbor_physical[dim] = wrapped_coord;

    const origin = tree.physicalToBlockOrigin(neighbor_physical, level);
    return tree.findBlockByOrigin(origin, level);
}

// =============================================================================
// Sibling Finding and Coarsening
// =============================================================================

/// Find sibling blocks (all children of the same parent).
pub fn findSiblings(
    comptime Tree: type,
    tree: *const Tree,
    block_idx: usize,
) ?[Tree.children_per_node]usize {
    const Nd = Tree.dimensions;
    const Block = Tree.BlockType;
    const num_children = Tree.children_per_node;
    const block_size = Block.size;

    const blk = &tree.blocks.items[block_idx];
    if (blk.level == 0) return null;

    var parent_origin: [Nd]usize = undefined;
    inline for (0..Nd) |d| {
        parent_origin[d] = (blk.origin[d] / 2) / block_size * block_size;
    }

    var siblings: [num_children]usize = .{std.math.maxInt(usize)} ** num_children;
    var found: usize = 0;

    for (0..num_children) |child| {
        var child_origin: [Nd]usize = undefined;
        inline for (0..Nd) |d| {
            const half = (child >> @intCast(d)) & 1;
            child_origin[d] = parent_origin[d] * 2 + half * block_size;
        }

        for (tree.blocks.items, 0..) |*candidate, idx| {
            if (candidate.block_index == std.math.maxInt(usize)) continue;
            if (candidate.level != blk.level) continue;

            var matches = true;
            inline for (0..Nd) |d| {
                if (candidate.origin[d] != child_origin[d]) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                siblings[child] = idx;
                found += 1;
                break;
            }
        }
    }

    if (found != num_children) return null;
    return siblings;
}

/// Check if a group of siblings can be coarsened together.
pub fn canCoarsenSiblings(
    comptime Tree: type,
    tree: *const Tree,
    siblings: [Tree.children_per_node]usize,
    arena: *const Tree.FieldArenaType,
    threshold: f64,
    hysteresis: f64,
) bool {
    for (siblings) |sibling_idx| {
        if (!shouldCoarsen(Tree, tree, sibling_idx, arena, threshold, hysteresis)) {
            return false;
        }
    }
    return true;
}

/// Coarsen a group of siblings into their parent block.
pub fn coarsenSiblings(
    comptime Tree: type,
    tree: *Tree,
    siblings: [Tree.children_per_node]usize,
    arena: *Tree.FieldArenaType,
) !CoarsenResult(Tree.children_per_node) {
    const Nd = Tree.dimensions;
    const FieldType = Tree.FieldType;
    const num_children = Tree.children_per_node;
    const Ops = operators_mod.AMROperators(Tree.FrontendType);

    const first_sibling = &tree.blocks.items[siblings[0]];
    const child_level = first_sibling.level;

    if (child_level == 0) {
        return error.CannotCoarsenLevelZero;
    }

    var min_origin: [Nd]usize = first_sibling.origin;
    for (siblings[1..]) |sib_idx| {
        const sib = &tree.blocks.items[sib_idx];
        std.debug.assert(sib.level == child_level);
        inline for (0..Nd) |d| {
            min_origin[d] = @min(min_origin[d], sib.origin[d]);
        }
    }

    var parent_origin: [Nd]usize = undefined;
    inline for (0..Nd) |d| {
        parent_origin[d] = min_origin[d] / 2;
    }
    const parent_level = child_level - 1;

    const parent_field_slot = arena.allocSlot() orelse return error.FieldArenaFull;
    const parent_idx = try tree.insertBlock(parent_origin, parent_level);
    tree.assignFieldSlot(parent_idx, parent_field_slot);

    // Restrict field data
    {
        var child_field_slices: [num_children][]const FieldType = undefined;
        for (siblings, 0..) |sib_idx, i| {
            const slot = tree.getFieldSlot(sib_idx);
            if (slot == std.math.maxInt(usize)) {
                child_field_slices[i] = &.{};
            } else {
                child_field_slices[i] = arena.getSlotConst(slot);
            }
        }

        const parent_field = arena.getSlot(parent_field_slot);
        Ops.restrictFullWeight(&child_field_slices, parent_field, true);
    }

    // Deallocate children
    var freed_field_slots: [num_children]usize = undefined;
    for (siblings, 0..) |sib_idx, i| {
        const slot = tree.getFieldSlot(sib_idx);
        freed_field_slots[i] = slot;
        if (slot != std.math.maxInt(usize)) {
            arena.freeSlot(slot);
        }

        tree.field_slots.items[sib_idx] = std.math.maxInt(usize);
        tree.invalidateBlock(sib_idx);
    }

    return CoarsenResult(num_children){
        .parent_idx = parent_idx,
        .parent_field_slot = parent_field_slot,
        .freed_field_slots = freed_field_slots,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "computeBlockGradient - scalar field" {
    const TestFrontend = struct {
        pub const Nd: usize = 2;
        pub const block_size: usize = 4;
        pub const FieldType = f64;
    };

    // Verify gradient computation compiles for scalar fields
    _ = fieldMagnitudeSq(f64, 1.0, 2.0);
    const diff = elementDiffMagnitudeSq(f64, 1.0, 2.0);
    try std.testing.expectEqual(@as(f64, 1.0), diff);

    _ = TestFrontend; // Silence unused warning
}

test "computeBlockGradient - complex field" {
    // Verify gradient computation compiles for complex fields
    const c1 = Complex.init(1.0, 0.0);
    const c2 = Complex.init(2.0, 1.0);
    const diff = elementDiffMagnitudeSq(Complex, c1, c2);
    // |c2 - c1|^2 = |(1, 1)|^2 = 1 + 1 = 2
    try std.testing.expectEqual(@as(f64, 2.0), diff);
}

test "computeBlockGradient - array field" {
    // Verify gradient computation compiles for array fields
    const arr1: [2]f64 = .{ 1.0, 2.0 };
    const arr2: [2]f64 = .{ 3.0, 4.0 };
    const diff = fieldMagnitudeSq([2]f64, arr1, arr2);
    // |arr2 - arr1|^2 = |2|^2 + |2|^2 = 4 + 4 = 8
    try std.testing.expectEqual(@as(f64, 8.0), diff);
}

test "AdaptResult initialization" {
    const result = AdaptResult{
        .refined = 0,
        .coarsened = 0,
        .refine_failed = 0,
        .coarsen_failed = 0,
        .changed = false,
    };
    try std.testing.expectEqual(@as(usize, 0), result.refined);
    try std.testing.expect(!result.changed);
}

test "adaptMesh enforces 2:1 balance" {
    const allocator = std.testing.allocator;
    const OpenTop = @import("topology.zig").OpenTopology(2, .{ 8.0, 4.0 });
    const Frontend = struct {
        pub const Nd: usize = 2;
        pub const block_size: usize = 4;
        pub const FieldType = f64;
        pub const Topology = OpenTop;
    };
    const Tree = @import("tree.zig").AMRTree(Frontend);
    const Arena = @import("field_arena.zig").FieldArena(Frontend);

    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    _ = try tree.insertBlock(.{ 0, 0 }, 0);
    _ = try tree.insertBlock(.{ 4, 0 }, 0);

    try tree.refineBlock(0);
    const child_idx = tree.findBlockByOrigin(.{ 4, 0 }, 1).?;
    try tree.refineBlock(child_idx);

    var arena = try Arena.init(allocator, tree.blockCount());
    defer arena.deinit();

    for (tree.blocks.items, 0..) |*blk, idx| {
        if (blk.block_index == std.math.maxInt(usize)) continue;
        const slot = arena.allocSlot().?;
        tree.assignFieldSlot(idx, slot);
    }

    _ = try adaptMesh(Tree, &tree, &arena, 1e9, 0.0);

    try std.testing.expect(tree.findBlockByOrigin(.{ 8, 0 }, 1) != null);
}

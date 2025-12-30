//! Geometric Multigrid (GMG) Infrastructure for AMR
//!
//! Provides restriction, prolongation, and smoothing operators for
//! solving linear systems (like Poisson) on the AMR hierarchy.

const std = @import("std");
const root = @import("root.zig");
const amr_block = @import("block.zig");

/// Geometric Multigrid solver for AMR hierarchies.
pub fn Multigrid(comptime Tree: type) type {
    const Frontend = Tree.FrontendType;
    const Nd = Frontend.Nd;
    const block_size = Frontend.block_size;
    const FieldType = Frontend.FieldType;
    const FieldArena = @import("field_arena.zig").FieldArena(Frontend);
    const Block = amr_block.AMRBlock(Frontend);

    return struct {
        const Self = @This();

        /// Restrict field from fine level to coarse level.
        ///
        /// Performed by averaging fine cells into coarse cells.
        /// If coarse_level is provided, only processes blocks at that level.
        pub fn restrict(
            tree: *const Tree,
            fine_arena: *const FieldArena,
            coarse_arena: *FieldArena,
            coarse_level_opt: ?u8,
        ) void {
            // Restriction is local to each coarse block that has fine children.
            // But in our linear octree, we can iterate over all blocks and
            // if a block has children, we restrict from them.
            
            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                
                if (coarse_level_opt) |lvl| {
                    if (block.level != lvl) continue;
                }

                const coarse_slot = tree.getFieldSlot(idx);
                if (coarse_slot == std.math.maxInt(usize)) continue;
                const coarse_data = coarse_arena.getSlot(coarse_slot);

                // For each child (up to 2^Nd)
                const child_step = block_size;
                const child_level = block.level + 1;
                
                for (0..(@as(usize, 1) << Nd)) |child_idx| {
                    var child_origin: [Nd]usize = undefined;
                    inline for (0..Nd) |d| {
                        const half = (child_idx >> @intCast(d)) & 1;
                        child_origin[d] = block.origin[d] * 2 + half * child_step;
                    }

                    if (tree.findBlockByOrigin(child_origin, child_level)) |child_block_idx| {
                        const fine_slot = tree.getFieldSlot(child_block_idx);
                        if (fine_slot == std.math.maxInt(usize)) continue;
                        const fine_data = fine_arena.getSlotConst(fine_slot);

                        // Restrict fine_data to coarse_data sub-region
                        restrictBlock(fine_data, coarse_data, child_idx);
                    }
                }
            }
        }

        fn restrictBlock(fine: []const FieldType, coarse: []FieldType, child_idx: usize) void {
            const child_offset = getChildOffset(child_idx);

            var coarse_coords: [Nd]usize = undefined;
            // Iterate over coarse sites in this quadrant
            var c_idx: usize = 0;
            const sub_size = block_size / 2;
            var sites_to_do: usize = 1;
            inline for (0..Nd) |_| sites_to_do *= sub_size;

            while (c_idx < sites_to_do) : (c_idx += 1) {
                // Map c_idx to quadrant-local coords
                var rem = c_idx;
                inline for (0..Nd) |d| {
                    coarse_coords[d] = (rem % sub_size) + child_offset[d];
                    rem /= sub_size;
                }

                const coarse_site = Block.getLocalIndex(coarse_coords);
                
                // Fine sites corresponding to this coarse site: 2^Nd sites
                var sum = Frontend.zeroField();
                
                var f_idx: usize = 0;
                while (f_idx < (@as(usize, 1) << Nd)) : (f_idx += 1) {
                    var fine_coords: [Nd]usize = undefined;
                    var f_rem = f_idx;
                    inline for (0..Nd) |d| {
                        fine_coords[d] = (coarse_coords[d] - child_offset[d]) * 2 + (f_rem % 2);
                        f_rem /= 2;
                    }
                    const fine_site = Block.getLocalIndex(fine_coords);
                    sum = addFields(sum, fine[fine_site]);
                }

                const factor = 1.0 / @as(f64, @floatFromInt(@as(usize, 1) << Nd));
                coarse[coarse_site] = scaleField(sum, factor);
            }
        }

        /// Prolongate (interpolate) field from coarse level to fine level.
        /// Overwrites fine data.
        pub fn prolongate(
            tree: *const Tree,
            coarse_arena: *const FieldArena,
            fine_arena: *FieldArena,
            coarse_level_opt: ?u8,
        ) void {
            prolongateInternal(tree, coarse_arena, fine_arena, coarse_level_opt, false);
        }

        /// Prolongate correction: fine += P(coarse).
        pub fn prolongateAdd(
            tree: *const Tree,
            coarse_arena: *const FieldArena,
            fine_arena: *FieldArena,
            coarse_level_opt: ?u8,
        ) void {
            prolongateInternal(tree, coarse_arena, fine_arena, coarse_level_opt, true);
        }

        fn prolongateInternal(
            tree: *const Tree,
            coarse_arena: *const FieldArena,
            fine_arena: *FieldArena,
            coarse_level_opt: ?u8,
            comptime add: bool,
        ) void {
            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                
                if (coarse_level_opt) |lvl| {
                    if (block.level != lvl) continue;
                }
                
                const coarse_slot = tree.getFieldSlot(idx);
                if (coarse_slot == std.math.maxInt(usize)) continue;
                const coarse_data = coarse_arena.getSlotConst(coarse_slot);

                const child_step = block_size;
                const child_level = block.level + 1;

                for (0..(@as(usize, 1) << Nd)) |child_idx| {
                    var child_origin: [Nd]usize = undefined;
                    inline for (0..Nd) |d| {
                        const half = (child_idx >> @intCast(d)) & 1;
                        child_origin[d] = block.origin[d] * 2 + half * child_step;
                    }

                    if (tree.findBlockByOrigin(child_origin, child_level)) |child_block_idx| {
                        const fine_slot = tree.getFieldSlot(child_block_idx);
                        if (fine_slot == std.math.maxInt(usize)) continue;
                        const fine_data = fine_arena.getSlot(fine_slot);

                        prolongateBlock(coarse_data, fine_data, child_idx, add);
                    }
                }
            }
        }

        fn prolongateBlock(coarse: []const FieldType, fine: []FieldType, child_idx: usize, comptime add: bool) void {
            const child_offset = getChildOffset(child_idx);

            for (0..Block.volume) |f_site| {
                const fine_coords = Block.getLocalCoords(f_site);
                var coarse_coords: [Nd]usize = undefined;
                inline for (0..Nd) |d| {
                    coarse_coords[d] = fine_coords[d] / 2 + child_offset[d];
                }
                const c_site = Block.getLocalIndex(coarse_coords);
                if (add) {
                    fine[f_site] = addFields(fine[f_site], coarse[c_site]);
                } else {
                    fine[f_site] = coarse[c_site];
                }
            }
        }

        /// Zero out fields at a specific level.
        pub fn zeroLevel(tree: *const Tree, level: u8, arena: *FieldArena) void {
            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                if (block.level != level) continue;
                
                const slot = tree.getFieldSlot(idx);
                if (slot == std.math.maxInt(usize)) continue;
                arena.zeroSlot(slot);
            }
        }

        /// Execute a V-Cycle.
        ///
        /// `smoother` must implement:
        /// - solveCoarsest(tree, u, f)
        /// - smooth(tree, level, u, f)
        /// - residual(tree, level, u, f, r)
        pub fn vCycle(
            tree: *const Tree,
            arena_u: *FieldArena, // Solution (Initial guess -> Result)
            arena_f: *FieldArena, // RHS
            arena_r: *FieldArena, // Residual / Coarse RHS (Scratch)
            level: u8,
            smoother: anytype,
        ) void {
            if (level == 0) {
                smoother.solveCoarsest(tree, arena_u, arena_f);
                return;
            }

            // 1. Pre-smooth
            smoother.smooth(tree, level, arena_u, arena_f);

            // 2. Residual: r = f - A u
            // Result stored in arena_r at `level`
            smoother.residual(tree, level, arena_u, arena_f, arena_r);

            // 3. Restrict: f_coarse = R r
            // We restrict from `level` to `level-1`. 
            // Source is `arena_r` (fine residual).
            // Dest is `arena_f` (coarse RHS).
            zeroLevel(tree, level - 1, arena_f);
            restrict(tree, arena_r, arena_f, level - 1);

            // 4. Zero initial guess for error equation on coarse grid
            zeroLevel(tree, level - 1, arena_u);

            // 5. Recursion
            // Use arena_r as the residual scratch for the next level too?
            // Yes, as long as we don't overwrite data needed for THIS level's post-correction.
            // At this point, `arena_r` at `level` holds the residual R_L.
            // We restricted it to F_{L-1} (in arena_f).
            // Recursive call will use arena_r at L-1. 
            // It won't touch R_L.
            vCycle(tree, arena_u, arena_f, arena_r, level - 1, smoother);

            // 6. Prolongate correction: u_fine += P(u_coarse)
            // Coarse solution (error) is in arena_u at level-1.
            // Fine solution is arena_u at level.
            prolongateAdd(tree, arena_u, arena_u, level - 1);

            // 7. Post-smooth
            smoother.smooth(tree, level, arena_u, arena_f);
        }

        fn getChildOffset(child_idx: usize) [Nd]usize {
            var offset: [Nd]usize = undefined;
            var rem = child_idx;
            const sub_size = block_size / 2;
            inline for (0..Nd) |d| {
                offset[d] = (rem % 2) * sub_size;
                rem /= 2;
            }
            return offset;
        }

        fn addFields(a: FieldType, b: FieldType) FieldType {
            var res: FieldType = undefined;
            const N_field = Frontend.field_dim;
            inline for (0..N_field) |i| {
                res[i] = a[i].add(b[i]);
            }
            return res;
        }

        fn scaleField(f: FieldType, s: f64) FieldType {
            var res: FieldType = undefined;
            const N_field = Frontend.field_dim;
            const factor = std.math.Complex(f64).init(s, 0);
            inline for (0..N_field) |i| {
                res[i] = f[i].mul(factor);
            }
            return res;
        }
    };
}

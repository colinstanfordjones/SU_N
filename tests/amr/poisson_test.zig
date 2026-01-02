//! Poisson Solver Test using GMG and Flux Registers
//!
//! Verifies that:
//! 1. Laplacian kernel works on AMR hierarchy.
//! 2. Restrict/Prolongate operators maintain consistency.
//! 3. Flux registers successfully identify conservation mismatches.

const std = @import("std");
const amr = @import("amr");
const math = @import("math");
const gauge = @import("gauge");
const builtin = @import("builtin");

const Complex = std.math.Complex(f64);

const Nd = 4;
const block_size = 8;
const topology = amr.topology;
const TestTopology = topology.PeriodicTopology(Nd, .{ 1.0, 1.0, 1.0, 1.0 });
const Frontend = gauge.GaugeFrontend(1, 1, Nd, block_size, TestTopology);
const Tree = amr.AMRTree(Frontend);
const FieldArena = amr.FieldArena(Frontend);
const GhostBuffer = amr.GhostBuffer(Frontend);
const ApplyContext = amr.ApplyContext(Frontend);
const MG = amr.multigrid.Multigrid(Tree);

const CountingAllocator = struct {
    allocator: std.mem.Allocator,
    alloc_calls: usize = 0,
    resize_calls: usize = 0,
    remap_calls: usize = 0,
    free_calls: usize = 0,
    grow_events: usize = 0,

    pub fn init(allocator: std.mem.Allocator) CountingAllocator {
        return .{ .allocator = allocator };
    }

    pub fn allocatorHandle(self: *CountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.alloc_calls += 1;
        self.grow_events += 1;
        return std.mem.Allocator.rawAlloc(self.allocator, len, alignment, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.resize_calls += 1;
        if (new_len > memory.len) {
            self.grow_events += 1;
        }
        return std.mem.Allocator.rawResize(self.allocator, memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.remap_calls += 1;
        if (new_len > memory.len) {
            self.grow_events += 1;
        }
        return std.mem.Allocator.rawRemap(self.allocator, memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.free_calls += 1;
        std.mem.Allocator.rawFree(self.allocator, memory, alignment, ret_addr);
    }
};

/// Standard Laplacian Kernel for Scalar Field
const PoissonKernel = struct {
    tree: *const Tree,
    const Block = Tree.BlockType;

    pub fn execute(
        self: *const PoissonKernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        ctx: *ApplyContext,
    ) void {
        const slot = self.tree.getFieldSlot(block_idx);
        const psi_in = ctx.field_in orelse return;
        const psi_out = ctx.field_out orelse return;

        const in = psi_in.getSlotConst(slot);
        const out = psi_out.getSlot(slot);

        const h = block.spacing;
        const h_inv = 1.0 / h;
        const h_inv_sq = h_inv * h_inv;

        // Check for C-F boundaries and add flux
        inline for (0..2 * Tree.dimensions) |face| {
            const neighbor = self.tree.neighborInfo(block_idx, face);

            if (neighbor.exists()) {
                const is_coarse_boundary = (neighbor.level_diff == -1);

                if (is_coarse_boundary) {
                    // We are Fine. Neighbor is Coarse.
                    if (ctx.flux_reg) |fr| {
                        self.computeAndAddFlux(fr, block_idx, block, in, face, h, true);
                    }
                }
            } else {
                // Could be Fine neighbor.
                var fine_neighbors: [Tree.max_fine_neighbors]usize = undefined;
                const count = self.tree.collectFineNeighbors(block_idx, face, &fine_neighbors);
                if (count > 0) {
                    // We are Coarse. Neighbor is Fine.
                    if (ctx.flux_reg) |fr| {
                        self.computeAndAddFlux(fr, block_idx, block, in, face, h, false);
                    }
                }
            }
        }

        // Compute Laplacian for all cells
        for (0..Tree.BlockType.volume) |site| {
            const coords = Tree.BlockType.getLocalCoords(site);
            var sum = Complex.init(0, 0);

            inline for (0..Tree.dimensions) |d| {
                var val_plus: Complex = undefined;
                var val_minus: Complex = undefined;

                // Handle boundary cells with ghost data
                if (coords[d] == Tree.BlockType.size - 1) {
                    if (ctx.field_ghosts) |ghosts| {
                        if (ghosts.get(block_idx)) |gp| {
                            const idx = Tree.BlockType.getGhostIndexRuntime(coords, d * 2);
                            if (idx < gp[d * 2].len) {
                                val_plus = gp[d * 2][idx][0];
                            } else {
                                val_plus = Complex.init(0, 0);
                            }
                        } else {
                            val_plus = Complex.init(0, 0);
                        }
                    } else {
                        val_plus = Complex.init(0, 0);
                    }
                } else {
                    const neighbor_plus = Tree.BlockType.localNeighborFast(site, d * 2);
                    val_plus = in[neighbor_plus][0];
                }

                if (coords[d] == 0) {
                    if (ctx.field_ghosts) |ghosts| {
                        if (ghosts.get(block_idx)) |gp| {
                            const idx = Tree.BlockType.getGhostIndexRuntime(coords, d * 2 + 1);
                            if (idx < gp[d * 2 + 1].len) {
                                val_minus = gp[d * 2 + 1][idx][0];
                            } else {
                                val_minus = Complex.init(0, 0);
                            }
                        } else {
                            val_minus = Complex.init(0, 0);
                        }
                    } else {
                        val_minus = Complex.init(0, 0);
                    }
                } else {
                    const neighbor_minus = Tree.BlockType.localNeighborFast(site, d * 2 + 1);
                    val_minus = in[neighbor_minus][0];
                }

                sum = sum.add(val_plus).add(val_minus);
            }

            const val_center = in[site][0];
            sum = sum.sub(val_center.mul(Complex.init(2.0 * @as(f64, @floatFromInt(Tree.dimensions)), 0)));
            out[site][0] = sum.mul(Complex.init(h_inv_sq, 0));
        }
    }

    fn computeAndAddFlux(
        self: *const PoissonKernel,
        fr: *Tree.FluxRegister,
        block_idx: usize,
        _: *const Tree.BlockType,
        in: []const [1]Complex,
        face: usize,
        h: f64,
        is_fine_block: bool,
    ) void {
        const dim = face / 2;
        const is_upper = (face % 2) == 0;

        const face_cells = Tree.BlockType.ghost_face_size;
        var flux_face: [face_cells][1]Complex = undefined;

        var area: f64 = 1.0;
        inline for (0..Tree.dimensions - 1) |_| area *= h;
        const area_factor = Complex.init(area, 0);

        const coord_fixed = if (is_upper) Tree.BlockType.size - 1 else 0;

        for (0..face_cells) |face_idx| {
            var coords: [Tree.dimensions]usize = undefined;
            var rem = face_idx;
            inline for (0..Tree.dimensions) |d| {
                if (d == dim) {
                    coords[d] = coord_fixed;
                } else {
                    coords[d] = rem % Tree.BlockType.size;
                    rem /= Tree.BlockType.size;
                }
            }

            const idx = Tree.BlockType.getLocalIndex(coords);
            const val = in[idx][0];

            const inner_idx = if (is_upper)
                idx - Tree.BlockType.strides[dim]
            else
                idx + Tree.BlockType.strides[dim];

            const val_inner = in[inner_idx][0];
            const grad = val.sub(val_inner).mul(Complex.init(1.0 / h, 0));
            flux_face[face_idx][0] = grad.mul(area_factor);
        }

        if (is_fine_block) {
            fr.addFine(self.tree, block_idx, face, flux_face, 1.0) catch {};
        } else {
            fr.addCoarse(self.tree, block_idx, face, flux_face, 1.0) catch {};
        }
    }
};

test "Multigrid - Restrict and Prolongate" {
    const allocator = std.testing.allocator;
    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create a 2-level hierarchy: one root and its children.
    // Multigrid operators expect a full hierarchy (parent + children), so we
    // intentionally keep the parent block here.
    const root_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    // try tree.refineBlock(root_idx); // Invalidates parent; not used for this MG-only setup.

    // Manually insert children
    // Copy origin to avoid invalidation during insertion reallocs
    const root_origin = tree.getBlock(root_idx).?.origin;
    const root_level = tree.getBlock(root_idx).?.level;
    const child_step = block_size;
    const child_level = root_level + 1;
    for (0..(@as(usize, 1) << Nd)) |child_idx| {
        var child_origin: [Nd]usize = undefined;
        inline for (0..Nd) |d| {
            const half = (child_idx >> @intCast(d)) & 1;
            child_origin[d] = root_origin[d] * 2 + half * child_step;
        }
        _ = try tree.insertBlock(child_origin, child_level);
    }

    var arena = try FieldArena.init(allocator, 32);
    defer arena.deinit();

    // Allocate fields for all blocks
    for (0..tree.blocks.items.len) |i| {
        const slot = arena.allocSlot() orelse return error.OutOfSlots;
        tree.assignFieldSlot(i, slot);
    }

    // Set a constant value on fine level
    const fine_val = Complex.init(1.0, 0);
    // root_block, child_step, child_level already defined above

    for (0..(@as(usize, 1) << Nd)) |child_rel_idx| {
        var child_origin: [Nd]usize = undefined;
        inline for (0..Nd) |d| {
            const half = (child_rel_idx >> @intCast(d)) & 1;
            child_origin[d] = root_origin[d] * 2 + half * child_step;
        }

        if (tree.findBlockByOrigin(child_origin, child_level)) |child_idx| {
            const slot = tree.getFieldSlot(child_idx);
            const data = arena.getSlot(slot);
            for (data) |*v| v.*[0] = fine_val;
        }
    }

    // Restrict to coarse level
    MG.restrict(&tree, &arena, &arena, null);

    // Coarse level should now have average value (which is same constant)
    const coarse_slot = tree.getFieldSlot(root_idx);
    const coarse_data = arena.getSlot(coarse_slot);
    for (coarse_data) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), v[0].re, 1e-10);
    }

    // Set coarse to something else and prolongate
    for (coarse_data) |*v| v.*[0] = Complex.init(2.0, 0);
    MG.prolongate(&tree, &arena, &arena, null);

    // Fine level should now have 2.0
    for (0..(@as(usize, 1) << Nd)) |child_rel_idx| {
        var child_origin: [Nd]usize = undefined;
        inline for (0..Nd) |d| {
            const half = (child_rel_idx >> @intCast(d)) & 1;
            child_origin[d] = root_origin[d] * 2 + half * child_step;
        }

        if (tree.findBlockByOrigin(child_origin, child_level)) |child_idx| {
            const slot = tree.getFieldSlot(child_idx);
            const data = arena.getSlot(slot);
            for (data) |v| {
                try std.testing.expectApproxEqAbs(@as(f64, 2.0), v[0].re, 1e-10);
            }
        }
    }
}

test "Poisson - Apply Laplacian with wiring" {
    const allocator = std.testing.allocator;
    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    var arena_in = try FieldArena.init(allocator, 4);
    defer arena_in.deinit();
    var arena_out = try FieldArena.init(allocator, 4);
    defer arena_out.deinit();

    const slot_in = arena_in.allocSlot().?;
    _ = arena_out.allocSlot().?;
    tree.assignFieldSlot(0, slot_in);

    var ghosts = try GhostBuffer.init(allocator, 4);
    defer ghosts.deinit();

    var flux_reg = Tree.FluxRegister.init(allocator);
    defer flux_reg.deinit();

    const kernel = PoissonKernel{ .tree = &tree };
    var ctx = ApplyContext.init(&tree);
    ctx.field_in = &arena_in;
    ctx.field_out = &arena_out;
    ctx.field_ghosts = &ghosts;
    ctx.flux_reg = &flux_reg;
    try tree.apply(&kernel, &ctx);
}

test "FluxRegister - Accumulation at C-F interface" {
    const allocator = std.testing.allocator;
    // Use non-periodic topology for simpler boundary logic
    const OpenTop = amr.topology.OpenTopology(Nd, .{ 16.0, 8.0, 8.0, 8.0 });
    const OpenFrontend = gauge.GaugeFrontend(1, 1, Nd, block_size, OpenTop);
    const OpenTree = amr.AMRTree(OpenFrontend);
    const OpenArena = amr.FieldArena(OpenFrontend);
    const OpenGhost = amr.GhostBuffer(OpenFrontend);
    const OpenApplyContext = amr.ApplyContext(OpenFrontend);
    const OpenKernel = struct {
        tree: *const OpenTree,
        const Block = OpenTree.BlockType;

        pub fn execute(
            self: *const @This(),
            block_idx: usize,
            block: *const Block,
            ctx: *OpenApplyContext,
        ) void {
            const psi_in = ctx.field_in orelse return;
            const slot = self.tree.getFieldSlot(block_idx);
            const in = psi_in.getSlotConst(slot);
            const h = block.spacing;

            inline for (0..2 * OpenTree.dimensions) |face| {
                var is_cf = false;
                var is_fc = false;

                const neighbor = self.tree.neighborInfo(block_idx, face);
                if (neighbor.exists() and neighbor.level_diff == -1) {
                    is_fc = true;
                } else if (!neighbor.exists()) {
                    var fine_neighbors: [OpenTree.max_fine_neighbors]usize = undefined;
                    if (self.tree.collectFineNeighbors(block_idx, face, &fine_neighbors) > 0) {
                        is_cf = true;
                    }
                }

                if (is_fc) {
                    if (ctx.flux_reg) |fr| computeAndAddFlux(self.tree, fr, block_idx, in, face, h, true);
                }
                if (is_cf) {
                    if (ctx.flux_reg) |fr| computeAndAddFlux(self.tree, fr, block_idx, in, face, h, false);
                }
            }
        }

        fn computeAndAddFlux(
            tree_ptr: *const OpenTree,
            fr: *OpenTree.FluxRegister,
            block_idx: usize,
            in: []const [1]Complex,
            face: usize,
            h: f64,
            is_fine: bool,
        ) void {
            _ = in;

            const face_cells = OpenTree.BlockType.ghost_face_size;
            var flux_face: [face_cells][1]Complex = undefined;

            var area: f64 = 1.0;
            inline for (0..OpenTree.dimensions - 1) |_| area *= h;
            const area_factor = Complex.init(area, 0);

            const base_flux: f64 = if (is_fine) 1.0 else 0.5;
            const flux_val = Complex.init(base_flux, 0).mul(area_factor);

            for (0..face_cells) |idx| {
                flux_face[idx][0] = flux_val;
            }

            if (is_fine) {
                fr.addFine(tree_ptr, block_idx, face, flux_face, 1.0) catch {};
            } else {
                fr.addCoarse(tree_ptr, block_idx, face, flux_face, 1.0) catch {};
            }
        }
    };

    var tree = try OpenTree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Two coarse blocks, refine the left one to create a C-F interface.
    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ block_size, 0, 0, 0 }, 0);
    try tree.refineBlock(0);

    const coarse_idx = tree.findBlockByOrigin(.{ block_size, 0, 0, 0 }, 0).?;
    const fine_idx = tree.findBlockByOrigin(.{ block_size, 0, 0, 0 }, 1).?;

    var arena = try OpenArena.init(allocator, tree.blockCount());
    defer arena.deinit();

    for (tree.blocks.items, 0..) |*blk, idx| {
        if (blk.block_index == std.math.maxInt(usize)) continue;
        const slot = arena.allocSlot().?;
        tree.assignFieldSlot(idx, slot);
    }

    var ghosts = try OpenGhost.init(allocator, tree.blocks.items.len);
    defer ghosts.deinit();

    var flux_reg = OpenTree.FluxRegister.init(allocator);
    defer flux_reg.deinit();

    const kernel = OpenKernel{ .tree = &tree };
    var ctx = OpenApplyContext.init(&tree);
    ctx.field_in = &arena;
    ctx.field_out = &arena;
    ctx.field_ghosts = &ghosts;
    ctx.flux_reg = &flux_reg;
    try tree.apply(&kernel, &ctx);

    var reduce_arena = std.heap.ArenaAllocator.init(allocator);
    defer reduce_arena.deinit();

    var reduced = try flux_reg.reduce(reduce_arena.allocator());
    defer reduced.deinit();
    try std.testing.expect(reduced.count() > 0);

    const coarse_block = tree.getBlock(coarse_idx).?;
    const coarse_key = tree.blockKeyForBlock(coarse_block);
    const coarse_face: u8 = 1; // -X face on the coarse block

    const fine_block = tree.getBlock(fine_idx).?;
    var fine_area: f64 = 1.0;
    var coarse_area: f64 = 1.0;
    inline for (0..Nd - 1) |_| {
        fine_area *= fine_block.spacing;
        coarse_area *= coarse_block.spacing;
    }

    const fine_factor = @as(f64, @floatFromInt(@as(usize, 1) << (Nd - 1)));
    const expected = fine_factor * fine_area * 1.0 - coarse_area * 0.5;

    var found = false;
    var iter = reduced.iterator();
    while (iter.next()) |entry| {
        if (entry.key_ptr.face != coarse_face) continue;
        if (!std.meta.eql(entry.key_ptr.block_key, coarse_key)) continue;
        found = true;
        for (entry.value_ptr.*) |cell| {
            try std.testing.expectApproxEqAbs(expected, cell[0].re, 1e-12);
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), cell[0].im, 1e-12);
        }
    }

    try std.testing.expect(found);
}

test "FluxRegister - no allocations during accumulation" {
    const allocator = std.testing.allocator;
    const OpenTop = amr.topology.OpenTopology(Nd, .{ 16.0, 8.0, 8.0, 8.0 });
    const OpenFrontend = gauge.GaugeFrontend(1, 1, Nd, block_size, OpenTop);
    const OpenTree = amr.AMRTree(OpenFrontend);

    var tree = try OpenTree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    _ = try tree.insertBlock(.{ block_size, 0, 0, 0 }, 0);
    try tree.refineBlock(0);

    var counting = CountingAllocator.init(allocator);
    var flux_reg = OpenTree.FluxRegister.init(counting.allocatorHandle());
    defer flux_reg.deinit();

    const reserve_hint = tree.blockCount() * 2 * OpenTree.dimensions;
    const face_cells = OpenTree.BlockType.ghost_face_size;
    var fine_neighbors: [OpenTree.max_fine_neighbors]usize = undefined;

    const warm_idx = tree.findBlockByOrigin(.{ block_size, 0, 0, 0 }, 0) orelse unreachable;
    var warm_flux: [face_cells][1]Complex = undefined;
    for (&warm_flux) |*cell| {
        cell.*[0] = Complex.init(0.0, 0.0);
    }
    try flux_reg.addCoarse(&tree, warm_idx, 0, warm_flux, 1.0);

    try flux_reg.clearAndReserve(reserve_hint);
    flux_reg.setNoAlloc(true);

    const before = counting.grow_events;

    for (tree.blocks.items, 0..) |*blk, idx| {
        if (blk.block_index == std.math.maxInt(usize)) continue;

        inline for (0..2 * OpenTree.dimensions) |face| {
            const neighbor = tree.neighborInfo(idx, face);
            var flux_face: [face_cells][1]Complex = undefined;
            for (&flux_face) |*cell| {
                cell.*[0] = Complex.init(1.0, 0.0);
            }

            if (neighbor.exists() and neighbor.level_diff == -1) {
                try flux_reg.addFine(&tree, idx, face, flux_face, 1.0);
            } else if (!neighbor.exists()) {
                const fine_count = tree.collectFineNeighbors(idx, face, &fine_neighbors);
                if (fine_count > 0) {
                    try flux_reg.addCoarse(&tree, idx, face, flux_face, 1.0);
                }
            }
        }
    }

    try std.testing.expectEqual(before, counting.grow_events);
}

test "FluxRegister - reduce merges per-thread registers" {
    if (builtin.single_threaded) return;

    const allocator = std.testing.allocator;
    const OpenTop = amr.topology.OpenTopology(Nd, .{ 16.0, 8.0, 8.0, 8.0 });
    const OpenFrontend = gauge.GaugeFrontend(1, 1, Nd, block_size, OpenTop);
    const OpenTree = amr.AMRTree(OpenFrontend);

    var tree = try OpenTree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();

    const block_idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    var flux_reg = OpenTree.FluxRegister.init(allocator);
    defer flux_reg.deinit();

    const total_faces = tree.blockCount() * 2 * OpenTree.dimensions;
    const workers = tree.threadCount();
    const reserve_hint = (total_faces + workers - 1) / workers;
    try flux_reg.clearAndReserve(reserve_hint);

    const face_cells = OpenTree.BlockType.ghost_face_size;
    const iterations: usize = 64;
    const face: usize = 0;

    const WorkerCtx = struct {
        fr: *OpenTree.FluxRegister,
        tree: *const OpenTree,
        block_idx: usize,
        face: usize,
        iterations: usize,
        value: f64,
    };

    const Worker = struct {
        fn run(wctx: WorkerCtx) void {
            var flux_face: [face_cells][1]Complex = undefined;
            for (&flux_face) |*cell| {
                cell.*[0] = Complex.init(wctx.value, 0.0);
            }

            for (0..wctx.iterations) |_| {
                wctx.fr.addCoarse(wctx.tree, wctx.block_idx, wctx.face, flux_face, 1.0) catch unreachable;
            }
        }
    };

    const values = [2]f64{ 1.0, 2.0 };
    var threads: [2]std.Thread = undefined;

    for (&threads, 0..) |*thread, i| {
        const wctx = WorkerCtx{
            .fr = &flux_reg,
            .tree = &tree,
            .block_idx = block_idx,
            .face = face,
            .iterations = iterations,
            .value = values[i],
        };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{wctx});
    }

    for (threads) |thread| thread.join();

    var reduce_arena = std.heap.ArenaAllocator.init(allocator);
    defer reduce_arena.deinit();

    var reduced = try flux_reg.reduce(reduce_arena.allocator());
    defer reduced.deinit();

    const block = tree.getBlock(block_idx).?;
    const key = tree.blockKeyForBlock(block);
    const expected = -@as(f64, @floatFromInt(iterations)) * (values[0] + values[1]);

    var found = false;
    var iter = reduced.iterator();
    while (iter.next()) |entry| {
        if (entry.key_ptr.face != face) continue;
        if (!std.meta.eql(entry.key_ptr.block_key, key)) continue;
        found = true;
        for (entry.value_ptr.*) |cell| {
            try std.testing.expectApproxEqAbs(expected, cell[0].re, 1e-12);
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), cell[0].im, 1e-12);
        }
    }

    try std.testing.expect(found);
}

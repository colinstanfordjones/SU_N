const std = @import("std");
const builtin = @import("builtin");
const su_n = @import("su_n");
const amr = @import("amr");
const constants = su_n.constants;

const StressTopology = amr.topology.OpenTopology(2, .{ 32.0, 32.0 });
const Frontend = amr.ScalarFrontend(2, 8, StressTopology);
const Tree = amr.AMRTree(Frontend);
const Block = amr.AMRBlock(Frontend);
const Arena = amr.FieldArena(Frontend);
const GhostBuffer = amr.GhostBuffer(Frontend);
const ApplyContext = amr.ApplyContext(Frontend);

const TimingTopology = amr.topology.OpenTopology(2, .{ 4.0, 4.0 });
const TimingFrontend = amr.ScalarFrontend(2, 4, TimingTopology);
const TimingTree = amr.AMRTree(TimingFrontend);
const TimingBlock = amr.AMRBlock(TimingFrontend);
const TimingGhostBuffer = amr.GhostBuffer(TimingFrontend);
const TimingApplyContext = amr.ApplyContext(TimingFrontend);

/// Diffusion kernel for stress testing.
/// Implements the heat equation: du/dt = alpha * laplacian(u)
const StressKernel = struct {
    tree: *const Tree,
    alpha: f64,
    dt: f64,

    /// Execute kernel on a single block using ApplyContext.
    pub fn execute(
        self: *const StressKernel,
        block_idx: usize,
        block: *const Block,
        ctx: *ApplyContext,
    ) void {
        const slot = self.tree.getFieldSlot(block_idx);
        if (slot == std.math.maxInt(usize)) return;

        const inputs = ctx.field_in orelse return;
        const outputs = ctx.field_out orelse return;

        const u = inputs.getSlotConst(slot);
        const u_new = outputs.getSlot(slot);

        // Get ghost faces
        var ghost_faces: ?*GhostBuffer.GhostFaces = null;
        if (ctx.field_ghosts) |ghosts| {
            ghost_faces = ghosts.get(block_idx);
        }

        const spacing = block.spacing;
        const inv_dx2 = 1.0 / (spacing * spacing);
        const factor = self.alpha * self.dt * inv_dx2;

        for (0..Block.volume) |i| {
            const coords = Block.getLocalCoords(i);
            const val = u[i];
            var laplacian: f64 = 0.0;

            var val_plus: f64 = 0.0;
            var val_minus: f64 = 0.0;

            // +X
            if (coords[0] == Frontend.block_size - 1) {
                if (ghost_faces) |gf| {
                    const ghost_idx = Block.getGhostIndex(coords, 0);
                    if (ghost_idx < gf[0].len) val_plus = gf[0][ghost_idx];
                }
            } else {
                val_plus = u[Block.localNeighborFast(i, 0)];
            }

            // -X
            if (coords[0] == 0) {
                if (ghost_faces) |gf| {
                    const ghost_idx = Block.getGhostIndex(coords, 1);
                    if (ghost_idx < gf[1].len) val_minus = gf[1][ghost_idx];
                }
            } else {
                val_minus = u[Block.localNeighborFast(i, 1)];
            }
            laplacian += (val_plus - 2.0 * val + val_minus);

            // +Y
            if (coords[1] == Frontend.block_size - 1) {
                if (ghost_faces) |gf| {
                    const ghost_idx = Block.getGhostIndex(coords, 2);
                    if (ghost_idx < gf[2].len) val_plus = gf[2][ghost_idx];
                }
            } else {
                val_plus = u[Block.localNeighborFast(i, 2)];
            }

            // -Y
            if (coords[1] == 0) {
                if (ghost_faces) |gf| {
                    const ghost_idx = Block.getGhostIndex(coords, 3);
                    if (ghost_idx < gf[3].len) val_minus = gf[3][ghost_idx];
                }
            } else {
                val_minus = u[Block.localNeighborFast(i, 3)];
            }
            laplacian += (val_plus - 2.0 * val + val_minus);

            u_new[i] = val + factor * laplacian;
        }
    }
};

/// Timing kernel for measuring execution patterns.
const TimingKernel = struct {
    exec_sleep_ns: u64,
    exec_count: *std.atomic.Value(usize),

    pub fn execute(
        self: *const TimingKernel,
        _: usize,
        _: *const TimingBlock,
        _: *TimingApplyContext,
    ) void {
        _ = self.exec_count.fetchAdd(1, .acq_rel);
        std.Thread.sleep(self.exec_sleep_ns);
    }
};

test "AMR pipeline stress matches reference" {
    var tree = try Tree.init(std.testing.allocator, 1.0, 2, 8);
    defer tree.deinit();

    const grid_dim = 4;
    const block_count = grid_dim * grid_dim;

    var arena_in = try Arena.init(std.testing.allocator, block_count);
    defer arena_in.deinit();
    var arena_out = try Arena.init(std.testing.allocator, block_count);
    defer arena_out.deinit();
    var arena_ref = try Arena.init(std.testing.allocator, block_count);
    defer arena_ref.deinit();

    for (0..grid_dim) |x| {
        for (0..grid_dim) |y| {
            const idx = try tree.insertBlock(.{ x * Frontend.block_size, y * Frontend.block_size }, 0);
            const slot = arena_in.allocSlot() orelse return error.OutOfMemory;
            _ = arena_out.allocSlot() orelse return error.OutOfMemory;
            _ = arena_ref.allocSlot() orelse return error.OutOfMemory;
            tree.assignFieldSlot(idx, slot);
        }
    }

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = tree.getFieldSlot(idx);
        if (slot == std.math.maxInt(usize)) continue;

        const data = arena_in.getSlot(slot);
        for (data, 0..) |*v, i| {
            v.* = @as(f64, @floatFromInt(idx)) + @as(f64, @floatFromInt(i)) * 0.001;
        }
    }

    var ghosts_pipeline = try GhostBuffer.init(std.testing.allocator, block_count);
    defer ghosts_pipeline.deinit();
    var ghosts_ref = try GhostBuffer.init(std.testing.allocator, block_count);
    defer ghosts_ref.deinit();

    try ghosts_pipeline.ensureForTree(&tree);
    try ghosts_ref.ensureForTree(&tree);

    var kernel = StressKernel{ .tree = &tree, .alpha = 0.25, .dt = 0.1 };
    var ref_kernel = StressKernel{ .tree = &tree, .alpha = 0.25, .dt = 0.1 };

    var iter: usize = 0;
    while (iter < 3) : (iter += 1) {
        // Pipeline execution
        var ctx = ApplyContext.init(&tree);
        ctx.field_in = &arena_in;
        ctx.field_out = &arena_out;
        ctx.field_ghosts = &ghosts_pipeline;
        try tree.apply(&kernel, &ctx);

        // Reference execution (manual ghost fill + sequential)
        const ghost_len = tree.blocks.items.len;
        try tree.fillGhostLayers(&arena_in, ghosts_ref.slice(ghost_len));

        var ref_ctx = ApplyContext.init(&tree);
        ref_ctx.field_in = &arena_in;
        ref_ctx.field_out = &arena_ref;
        ref_ctx.field_ghosts = &ghosts_ref;

        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) continue;
            const slot = tree.getFieldSlot(idx);
            if (slot == std.math.maxInt(usize)) continue;

            ref_kernel.execute(idx, block, &ref_ctx);
        }

        // Compare results
        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) continue;
            const slot = tree.getFieldSlot(idx);
            if (slot == std.math.maxInt(usize)) continue;

            const out = arena_out.getSlotConst(slot);
            const ref = arena_ref.getSlotConst(slot);
            for (out, 0..) |val, i| {
                try std.testing.expectApproxEqAbs(val, ref[i], constants.test_epsilon);
            }
        }

        std.mem.swap(Arena, &arena_in, &arena_out);
    }
}

test "AMR kernel execution count" {
    var tree = try TimingTree.init(std.testing.allocator, 1.0, 1, 8);
    defer tree.deinit();
    _ = try tree.insertBlock(.{ 0, 0 }, 0);

    var exec_count = std.atomic.Value(usize).init(0);

    const sleep_ns = 10 * std.time.ns_per_ms;
    var kernel = TimingKernel{
        .exec_sleep_ns = sleep_ns,
        .exec_count = &exec_count,
    };

    var ctx = TimingApplyContext.init(&tree);
    try tree.apply(&kernel, &ctx);

    // Single block should be executed once
    try std.testing.expectEqual(@as(usize, 1), exec_count.load(.acquire));
}

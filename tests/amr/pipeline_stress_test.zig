const std = @import("std");
const builtin = @import("builtin");
const su_n = @import("su_n");
const amr = @import("amr");
const constants = su_n.constants;

const StressKernel = struct {
    tree: *const Tree,
    alpha: f64,
    dt: f64,

    pub fn executeInterior(
        self: *StressKernel,
        block_idx: usize,
        block: *const Block,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*GhostBuffer,
        flux_reg: ?*Tree.FluxRegister,
    ) void {
        _ = flux_reg;
        self.executeRegion(.interior, block_idx, block, inputs, outputs, ghosts);
    }

    pub fn executeBoundary(
        self: *StressKernel,
        block_idx: usize,
        block: *const Block,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*GhostBuffer,
        flux_reg: ?*Tree.FluxRegister,
    ) void {
        _ = flux_reg;
        self.executeRegion(.boundary, block_idx, block, inputs, outputs, ghosts);
    }

    const SiteRegion = enum {
        interior,
        boundary,
    };

    fn executeRegion(
        self: *StressKernel,
        region: SiteRegion,
        block_idx: usize,
        block: *const Block,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*GhostBuffer,
    ) void {
        const slot = self.tree.getFieldSlot(block_idx);
        const u = inputs.getSlotConst(slot);
        const u_new = outputs.getSlot(slot);

        const ghost_faces = if (ghosts) |g| g.get(block_idx) orelse return else return;

        const spacing = block.spacing;
        const inv_dx2 = 1.0 / (spacing * spacing);
        const factor = self.alpha * self.dt * inv_dx2;

        for (0..Block.volume) |i| {
            const coords = Block.getLocalCoords(i);
            const on_boundary = Block.isOnBoundary(coords);
            switch (region) {
                .interior => if (on_boundary) continue,
                .boundary => if (!on_boundary) continue,
            }

            const val = u[i];
            var laplacian: f64 = 0.0;

            var val_plus: f64 = 0.0;
            var val_minus: f64 = 0.0;

            // +X
            if (coords[0] == Frontend.block_size - 1) {
                const ghost_idx = Block.getGhostIndex(coords, 0);
                if (ghost_idx < ghost_faces[0].len) val_plus = ghost_faces[0][ghost_idx];
            } else {
                val_plus = u[Block.localNeighborFast(i, 0)];
            }

            // -X
            if (coords[0] == 0) {
                const ghost_idx = Block.getGhostIndex(coords, 1);
                if (ghost_idx < ghost_faces[1].len) val_minus = ghost_faces[1][ghost_idx];
            } else {
                val_minus = u[Block.localNeighborFast(i, 1)];
            }
            laplacian += (val_plus - 2.0 * val + val_minus);

            // +Y
            if (coords[1] == Frontend.block_size - 1) {
                const ghost_idx = Block.getGhostIndex(coords, 2);
                if (ghost_idx < ghost_faces[2].len) val_plus = ghost_faces[2][ghost_idx];
            } else {
                val_plus = u[Block.localNeighborFast(i, 2)];
            }

            // -Y
            if (coords[1] == 0) {
                const ghost_idx = Block.getGhostIndex(coords, 3);
                if (ghost_idx < ghost_faces[3].len) val_minus = ghost_faces[3][ghost_idx];
            } else {
                val_minus = u[Block.localNeighborFast(i, 3)];
            }
            laplacian += (val_plus - 2.0 * val + val_minus);

            u_new[i] = val + factor * laplacian;
        }
    }
};

const TimingKernel = struct {
    pull_sleep_ns: u64,
    interior_sleep_ns: u64,
    interior_count: *std.atomic.Value(usize),
    boundary_count: *std.atomic.Value(usize),

    pub fn ghostPrepare(self: *TimingKernel) !bool {
        _ = self;
        return true;
    }

    pub fn ghostPull(self: *TimingKernel, block_idx: usize) void {
        _ = block_idx;
        std.Thread.sleep(self.pull_sleep_ns);
    }

    pub fn ghostPush(self: *TimingKernel, block_idx: usize) void {
        _ = self;
        _ = block_idx;
    }

    pub fn ghostFinalize(self: *TimingKernel) void {
        _ = self;
    }

    pub fn executeInterior(
        self: *const @This(),
        blk_idx: usize,
        block: anytype,
        psi_in: anytype,
        psi_out: anytype,
        ghosts: anytype,
        flux_reg: anytype,
    ) void {
        _ = blk_idx;
        _ = block;
        _ = psi_in;
        _ = psi_out;
        _ = ghosts;
        _ = flux_reg;
        _ = self.interior_count.fetchAdd(1, .acq_rel);
        std.Thread.sleep(self.interior_sleep_ns);
    }

    pub fn executeBoundary(
        self: *const @This(),
        blk_idx: usize,
        block: anytype,
        psi_in: anytype,
        psi_out: anytype,
        ghosts: anytype,
        flux_reg: anytype,
    ) void {
        _ = blk_idx;
        _ = block;
        _ = psi_in;
        _ = psi_out;
        _ = ghosts;
        _ = flux_reg;
        _ = self.boundary_count.fetchAdd(1, .acq_rel);
    }
};

const StressTopology = amr.topology.OpenTopology(2, .{ 32.0, 32.0 });
const Frontend = amr.ScalarFrontend(2, 8, StressTopology);
const Tree = amr.AMRTree(Frontend);
const Block = amr.AMRBlock(Frontend);
const Arena = amr.FieldArena(Frontend);
const GhostBuffer = amr.GhostBuffer(Frontend);

const TimingTopology = amr.topology.OpenTopology(2, .{ 4.0, 4.0 });
const TimingFrontend = amr.ScalarFrontend(2, 4, TimingTopology);
const TimingTree = amr.AMRTree(TimingFrontend);
const TimingBlock = amr.AMRBlock(TimingFrontend);
const TimingGhostBuffer = amr.GhostBuffer(TimingFrontend);

test "AMR pipeline stress matches sequential" {
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

    var iter: usize = 0;
    while (iter < 3) : (iter += 1) {
        try tree.apply(&kernel, &arena_in, &arena_out, &ghosts_pipeline, null);

        const ghost_len = tree.blocks.items.len;
        amr.ghost.fillGhostLayers(Tree, &tree, &arena_in, ghosts_ref.slice(ghost_len));

        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) continue;
            const slot = tree.getFieldSlot(idx);
            if (slot == std.math.maxInt(usize)) continue;

            kernel.executeInterior(idx, block, &arena_in, &arena_ref, &ghosts_ref, null);
            kernel.executeBoundary(idx, block, &arena_in, &arena_ref, &ghosts_ref, null);
        }

        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) continue;
            const slot = tree.getFieldSlot(idx);
            if (slot == std.math.maxInt(usize)) continue;

            const out = arena_out.getSlot(slot);
            const ref = arena_ref.getSlot(slot);
            for (out, 0..) |val, i| {
                try std.testing.expectApproxEqAbs(val, ref[i], constants.test_epsilon);
            }
        }

        std.mem.swap(Arena, &arena_in, &arena_out);
    }
}

test "AMR pipeline overlaps interior and ghost pull" {
    var tree = try TimingTree.init(std.testing.allocator, 1.0, 1, 8);
    defer tree.deinit();
    _ = try tree.insertBlock(.{ 0, 0 }, 0);

    var interior_count = std.atomic.Value(usize).init(0);
    var boundary_count = std.atomic.Value(usize).init(0);

    const sleep_ns = 50 * std.time.ns_per_ms;
    var kernel = TimingKernel{
        .pull_sleep_ns = sleep_ns,
        .interior_sleep_ns = sleep_ns,
        .interior_count = &interior_count,
        .boundary_count = &boundary_count,
    };

    var timer = try std.time.Timer.start();
    try tree.apply(&kernel, @as(void, {}), @as(void, {}), null, null);
    const elapsed = timer.read();

    try std.testing.expectEqual(@as(usize, 1), interior_count.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), boundary_count.load(.acquire));

    if (!builtin.single_threaded) {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        if (cpu_count > 1) {
            const threshold = sleep_ns + (sleep_ns * 8 / 10);
            try std.testing.expect(elapsed < threshold);
        }
    }
}

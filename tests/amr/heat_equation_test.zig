const std = @import("std");
const amr = @import("amr");

// 1. Define Frontend (Scalar, 2D, Open Boundary)
const HeatFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 16;
    pub const FieldType = f64;
    pub const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
};

const Tree = amr.AMRTree(HeatFrontend);
const Block = amr.AMRBlock(HeatFrontend);
const Arena = amr.FieldArena(HeatFrontend);
const GhostBuffer = amr.GhostBuffer(HeatFrontend);
const ApplyContext = amr.ApplyContext(HeatFrontend);

const Complex = std.math.Complex(f64);

const ComplexFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 16;
    pub const FieldType = Complex;
    pub const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
};

const TreeComplex = amr.AMRTree(ComplexFrontend);

const Nd = 2;
const ComplexTree = TreeComplex;
const ComplexBlock = ComplexTree.BlockType;
const ComplexArena = ComplexTree.FieldArenaType;
const ComplexGhostBuffer = amr.GhostBuffer(ComplexFrontend);
const ComplexApplyContext = amr.ApplyContext(ComplexFrontend);

const RealHeatKernel = struct {
    tree: *const Tree,
    alpha: f64,
    dt: f64,

    pub fn execute(
        self: *const RealHeatKernel,
        block_idx: usize,
        _: *const Block,
        ctx: *ApplyContext,
    ) void {
        const slot = self.tree.getFieldSlot(block_idx);
        const psi_in = ctx.field_in orelse return;
        const psi_out = ctx.field_out orelse return;

        const in = psi_in.getSlotConst(slot);
        const out = psi_out.getSlot(slot);

        var ghost_slices: [2 * Nd][]const f64 = undefined;
        for (0..2 * Nd) |f| ghost_slices[f] = &.{};
        if (ctx.field_ghosts) |ghosts| {
            if (ghosts.get(block_idx)) |gp| {
                for (0..2 * Nd) |f| ghost_slices[f] = &gp[f];
            }
        }

        for (0..Block.volume) |i| {
            const coords = Block.getLocalCoords(i);
            var sum: f64 = 0.0;

            inline for (0..Nd) |dim| {
                // Forward neighbor
                var val_plus: f64 = undefined;
                if (coords[dim] == Block.size - 1) {
                    // Boundary
                    const face_idx = dim * 2;
                    const ghost_idx = Block.getGhostIndexRuntime(coords, face_idx);
                    if (ghost_slices[face_idx].len > ghost_idx) {
                        val_plus = ghost_slices[face_idx][ghost_idx];
                    } else {
                        val_plus = 0.0; // Dirichlet BC
                    }
                } else {
                    const neighbor = Block.localNeighborFast(i, dim * 2);
                    val_plus = in[neighbor];
                }

                // Backward neighbor
                var val_minus: f64 = undefined;
                if (coords[dim] == 0) {
                    const face_idx = dim * 2 + 1;
                    const ghost_idx = Block.getGhostIndexRuntime(coords, face_idx);
                    if (ghost_slices[face_idx].len > ghost_idx) {
                        val_minus = ghost_slices[face_idx][ghost_idx];
                    } else {
                        val_minus = 0.0; // Dirichlet BC
                    }
                } else {
                    const neighbor = Block.localNeighborFast(i, dim * 2 + 1);
                    val_minus = in[neighbor];
                }

                sum += val_plus + val_minus - 2.0 * in[i];
            }

            out[i] = in[i] + self.alpha * self.dt * sum;
        }
    }
};

const ComplexHeatKernel = struct {
    tree: *const ComplexTree,
    alpha: f64,

    pub fn execute(
        self: *const ComplexHeatKernel,
        block_idx: usize,
        _: *const ComplexBlock,
        ctx: *ComplexApplyContext,
    ) void {
        const slot = self.tree.getFieldSlot(block_idx);
        const psi_in = ctx.field_in orelse return;
        const psi_out = ctx.field_out orelse return;

        const in = psi_in.getSlotConst(slot);
        const out = psi_out.getSlot(slot);

        var ghost_slices: [2 * Nd][]const Complex = undefined;
        for (0..2 * Nd) |f| ghost_slices[f] = &.{};
        if (ctx.field_ghosts) |ghosts| {
            if (ghosts.get(block_idx)) |gp| {
                for (0..2 * Nd) |f| ghost_slices[f] = gp[f];
            }
        }

        for (0..ComplexBlock.volume) |i| {
            const coords = ComplexBlock.getLocalCoords(i);
            var sum = Complex.init(0, 0);

            inline for (0..Nd) |dim| {
                // Forward neighbor
                var val_plus: Complex = undefined;
                if (coords[dim] == ComplexBlock.size - 1) {
                    const face_idx = dim * 2;
                    const ghost_idx = ComplexBlock.getGhostIndexRuntime(coords, face_idx);
                    if (ghost_slices[face_idx].len > ghost_idx) {
                        val_plus = ghost_slices[face_idx][ghost_idx];
                    } else {
                        val_plus = Complex.init(0, 0);
                    }
                } else {
                    const neighbor = ComplexBlock.localNeighborFast(i, dim * 2);
                    val_plus = in[neighbor];
                }

                // Backward neighbor
                var val_minus: Complex = undefined;
                if (coords[dim] == 0) {
                    const face_idx = dim * 2 + 1;
                    const ghost_idx = ComplexBlock.getGhostIndexRuntime(coords, face_idx);
                    if (ghost_slices[face_idx].len > ghost_idx) {
                        val_minus = ghost_slices[face_idx][ghost_idx];
                    } else {
                        val_minus = Complex.init(0, 0);
                    }
                } else {
                    const neighbor = ComplexBlock.localNeighborFast(i, dim * 2 + 1);
                    val_minus = in[neighbor];
                }

                sum = sum.add(val_plus).add(val_minus).sub(in[i].mul(Complex.init(2.0, 0)));
            }

            out[i] = in[i].add(sum.mul(Complex.init(self.alpha, 0)));
        }
    }
};

test "Heat Equation Diffusion" {
    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 16);
    defer arena.deinit();
    var arena_out = try Arena.init(std.testing.allocator, 16);
    defer arena_out.deinit();
    var ghosts = try GhostBuffer.init(std.testing.allocator, 16);
    defer ghosts.deinit();

    // Insert block
    const idx = try tree.insertBlockWithField(.{ 0, 0 }, 0, &arena);
    _ = arena_out.allocSlot(); // Sync slots

    // Initialize Gaussian pulse
    const slot = tree.getFieldSlot(idx);
    const u = arena.getSlot(slot);
    const center = 8.0;
    for (0..Block.volume) |i| {
        const coords = Block.getLocalCoords(i);
        const dx = @as(f64, @floatFromInt(coords[0])) - center;
        const dy = @as(f64, @floatFromInt(coords[1])) - center;
        u[i] = @exp(-(dx * dx + dy * dy) / 4.0);
    }

    // Step
    var kernel = RealHeatKernel{ .tree = &tree, .alpha = 1.0, .dt = 0.1 };
    var ctx = ApplyContext.init(&tree);
    ctx.field_in = &arena;
    ctx.field_out = &arena_out;
    ctx.field_ghosts = &ghosts;
    try tree.apply(&kernel, &ctx);

    // Verify diffusion (peak should decrease)
    const u_new = arena_out.getSlot(slot);
    const peak_old = 1.0; // exp(0)
    const center_idx = Block.getLocalIndex(.{ 8, 8 });
    const peak_new = u_new[center_idx];

    try std.testing.expect(peak_new < peak_old);
    try std.testing.expect(peak_new > 0.9); // Shouldn't decay too fast
}

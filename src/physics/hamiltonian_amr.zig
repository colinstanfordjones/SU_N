//! AMR-Aware Hamiltonian Module with Gauge-Covariant Laplacian
//!
//! Implements a Hamiltonian that operates on block-structured AMR wavefunctions
//! using gauge-covariant derivatives from GaugeTree.
//!
//! ## Usage
//!
//! ```zig
//! const gauge = @import("gauge");
//! const physics = @import("physics");
//! const amr = @import("amr");
//!
//! const Frontend = gauge.GaugeFrontend(3, 4, 4, 16);  // SU(3), Dirac, 4D, 16^4 blocks
//! const GaugeTree = gauge.GaugeTree(Frontend);
//! const FieldArena = amr.FieldArena(Frontend);
//!
//! var gauge_tree = try GaugeTree.init(allocator, 1.0, 4, 8);
//! defer gauge_tree.deinit();
//!
//! var psi_arena = try FieldArena.init(allocator, 256);
//! defer psi_arena.deinit();
//!
//! const HAMR = physics.hamiltonian_amr.HamiltonianAMR(Frontend);
//! var H = HAMR.init(&gauge_tree, mass, potential_fn);
//!
//! var ghosts = try amr.GhostBuffer(Frontend).init(allocator, 256);
//! defer ghosts.deinit();
//!
//! try H.apply(&psi_arena, &workspace, &ghosts);
//! ```
//!
//! ## Gauge-Covariant Laplacian
//!
//! The kinetic energy uses the gauge-covariant Laplacian from GaugeTree:
//!   ∇²ψ(x) = Σ_μ [U_μ(x)ψ(x+μ̂) + U†_μ(x-μ̂)ψ(x-μ̂) - 2ψ(x)] / a²

const std = @import("std");
const amr_mod = @import("amr");
const gauge_mod = @import("gauge");
const adaptation_mod = @import("amr").adaptation;
const constants = @import("constants");

const Complex = std.math.Complex(f64);

/// AMR Hamiltonian for multi-scale lattice computations.
///
/// Uses GaugeTree for gauge-covariant operations (Laplacian, link access).
/// Manages only psi field ghost layers internally.
pub fn HamiltonianAMR(comptime Frontend: type) type {
    if (!@hasDecl(Frontend, "gauge_group_dim")) {
        @compileError("HamiltonianAMR requires a GaugeFrontend with gauge_group_dim");
    }
    if (!@hasDecl(Frontend, "LinkType")) {
        @compileError("HamiltonianAMR requires a GaugeFrontend with LinkType");
    }

    const Nd = Frontend.Nd;
    const N_field = Frontend.field_dim;
    const FieldType = Frontend.FieldType;

    const GaugeTree = gauge_mod.GaugeTree(Frontend);
    const Block = amr_mod.AMRBlock(Frontend);
    const FieldArena = amr_mod.FieldArena(Frontend);

    const num_faces = 2 * Nd;

    return struct {
        const Self = @This();

        pub const GaugeTreeType = GaugeTree;
        pub const BlockType = Block;
        pub const FrontendType = Frontend;
        pub const FieldArenaType = FieldArena;
        pub const block_volume = Block.volume;
        pub const GhostBuffer = amr_mod.GhostBuffer(Frontend);

        /// Reference to the GaugeTree (provides links and covariant Laplacian)
        gauge_tree: *GaugeTree,

        /// Particle mass
        mass: f64,

        /// Potential function
        potential_fn: *const fn ([Nd]f64, f64) f64,

        /// Persistent MPI exchange state for overlapping
        dist_state: ?GaugeTree.LinkExchange.ExchangeState = null,

        /// Current time step (for flux register)
        current_dt: f64 = 0.0,

        /// Initialize the AMR Hamiltonian.
        pub fn init(
            gauge_tree: *GaugeTree,
            mass: f64,
            potential_fn: *const fn ([Nd]f64, f64) f64,
        ) Self {
            return Self{
                .gauge_tree = gauge_tree,
                .mass = mass,
                .potential_fn = potential_fn,
                .dist_state = null,
                .current_dt = 0.0,
            };
        }

        pub fn deinit(self: *Self) void {
            _ = self;
        }

        /// Get the underlying tree reference
        pub fn getTree(self: *const Self) *const GaugeTree.TreeType {
            return &self.gauge_tree.tree;
        }

        pub fn ghostPrepare(self: *Self) !bool {
            _ = try self.gauge_tree.prepareLinkGhostExchange();
            if (self.gauge_tree.shardContext()) |s| {
                self.dist_state = try self.gauge_tree.beginExchange(s);
            }
            return true;
        }

        pub fn ghostPull(self: *Self, block_idx: usize) void {
            self.gauge_tree.fillGhostsPull(block_idx);
        }

        pub fn ghostPush(self: *Self, block_idx: usize) void {
            self.gauge_tree.fillGhostsPush(block_idx);
        }

        pub fn ghostFinalize(self: *Self) void {
            if (self.dist_state) |*s| {
                self.gauge_tree.finishExchange(s) catch {};
                self.dist_state = null;
            }
            self.gauge_tree.finalizeLinkGhostExchange();
        }

        pub fn executeInterior(
            self: *const Self,
            block_idx: usize,
            block: *const Block,
            psi_in: *const FieldArena,
            psi_out: *FieldArena,
            ghosts: ?*GhostBuffer,
            flux_reg: ?*GaugeTree.TreeType.FluxRegister,
        ) void {
            _ = flux_reg;
            @constCast(self).executeRegion(.interior, block_idx, block, psi_in, psi_out, ghosts);
        }

        pub fn executeBoundary(
            self: *const Self,
            block_idx: usize,
            block: *const Block,
            psi_in: *const FieldArena,
            psi_out: *FieldArena,
            ghosts: ?*GhostBuffer,
            flux_reg: ?*GaugeTree.TreeType.FluxRegister,
        ) void {
            @constCast(self).executeRegion(.boundary, block_idx, block, psi_in, psi_out, ghosts);

            if (flux_reg) |fr| {
                const tree = &self.gauge_tree.tree;
                const slot = tree.getFieldSlot(block_idx);
                const in = psi_in.getSlotConst(slot);
                const h = block.spacing;

                // Get ghost slices
                var psi_ghost_slices: [num_faces][]const FieldType = undefined;
                for (0..num_faces) |f| {
                    psi_ghost_slices[f] = &.{};
                }
                if (ghosts) |g| {
                    if (g.get(block_idx)) |gp| {
                        inline for (0..num_faces) |f| psi_ghost_slices[f] = &gp[f];
                    }
                }

                inline for (0..num_faces) |face| {
                    const neighbor = tree.neighborInfo(block_idx, face);
                    var is_cf = false;
                    var is_fc = false;

                    if (neighbor.exists() and neighbor.level_diff == -1) {
                        is_fc = true; // We are Fine, Neighbor Coarse
                    } else if (!neighbor.exists()) {
                        var fine_neighbors: [GaugeTree.TreeType.max_fine_neighbors]usize = undefined;
                        if (tree.collectFineNeighbors(block_idx, face, &fine_neighbors) > 0) {
                            is_cf = true; // We are Coarse, Neighbor Fine
                        }
                    }

                    if (is_fc) {
                        self.computeAndAddFlux(fr, block_idx, in, psi_ghost_slices, face, h, true);
                    }
                    if (is_cf) {
                        self.computeAndAddFlux(fr, block_idx, in, psi_ghost_slices, face, h, false);
                    }
                }
            }
        }

        fn computeAndAddFlux(
            self: *const Self,
            fr: *GaugeTree.TreeType.FluxRegister,
            block_idx: usize,
            in: []const FieldType,
            ghosts: [num_faces][]const FieldType,
            face: usize,
            h: f64,
            is_fine: bool,
        ) void {
            const LinkType = Frontend.LinkType;
            const dim = face / 2;
            const is_upper = (face % 2) == 0;
            const tree = &self.gauge_tree.tree;

            const face_cells = Block.ghost_face_size;
            var flux_face: [face_cells]FieldType = undefined;

            var area: f64 = 1.0;
            inline for (0..Nd - 1) |_| area *= h;
            const area_factor = Complex.init(area, 0);

            const coord_fixed = if (is_upper) Block.size - 1 else 0;

            for (0..face_cells) |face_idx| {
                var coords: [Nd]usize = undefined;
                var rem = face_idx;
                inline for (0..Nd) |d| {
                    if (d == dim) {
                        coords[d] = coord_fixed;
                    } else {
                        coords[d] = rem % Block.size;
                        rem /= Block.size;
                    }
                }

                const idx = Block.getLocalIndex(coords);
                const psi_val = in[idx];

                var grad: FieldType = undefined;

                if (is_upper) {
                    // Flux F+ = D+ psi = (U psi(x+mu) - psi(x))/h
                    const link = self.gauge_tree.getLink(block_idx, idx, dim);

                    var psi_outer: FieldType = undefined;
                    const g_idx = Block.getGhostIndexRuntime(coords, face);
                    if (g_idx < ghosts[face].len) {
                        psi_outer = ghosts[face][g_idx];
                    } else {
                        psi_outer = FrontendType.zeroField();
                    }

                    const transported = FrontendType.applyLinkToField(link, psi_outer);
                    inline for (0..N_field) |a| {
                        grad[a] = transported[a].sub(psi_val[a]).mul(Complex.init(1.0 / h, 0));
                    }
                } else {
                    // Flux F- = D- psi = (psi(x) - Udagger psi(x-mu))/h
                    var link_dag: LinkType = LinkType.identity();

                    const neighbor_info = tree.neighborInfo(block_idx, face);
                    if (neighbor_info.exists()) {
                        const n_idx = neighbor_info.block_idx;
                        var n_coords = coords;
                        n_coords[dim] = Block.size - 1;
                        const n_site = Block.getLocalIndex(n_coords);
                        const link = self.gauge_tree.getLink(n_idx, n_site, dim);
                        link_dag = link.adjoint();
                    }

                    var psi_outer: FieldType = undefined;
                    const g_idx = Block.getGhostIndexRuntime(coords, face);
                    if (g_idx < ghosts[face].len) {
                        psi_outer = ghosts[face][g_idx];
                    } else {
                        psi_outer = FrontendType.zeroField();
                    }

                    const transported = FrontendType.applyLinkToField(link_dag, psi_outer);
                    inline for (0..N_field) |a| {
                        grad[a] = psi_val[a].sub(transported[a]).mul(Complex.init(1.0 / h, 0));
                    }
                }

                inline for (0..N_field) |a| {
                    flux_face[face_idx][a] = grad[a].mul(area_factor);
                }
            }

            const scale = self.current_dt;

            if (is_fine) {
                fr.addFine(tree, block_idx, face, flux_face, scale) catch {};
            } else {
                fr.addCoarse(tree, block_idx, face, flux_face, scale) catch {};
            }
        }

        /// Apply the Hamiltonian: H·ψ = (-ℏ²/2m)∇²ψ + V·ψ.
        /// Wrapper around AMRTree.apply(self, ...).
        pub fn apply(
            self: *Self,
            psi_in: *const FieldArena,
            psi_out: *FieldArena,
            ghosts: *GhostBuffer,
            flux_reg: ?*GaugeTree.TreeType.FluxRegister,
        ) !void {
            try self.gauge_tree.tree.apply(self, psi_in, psi_out, ghosts, flux_reg);
        }

        fn applyToBlock(
            self: *Self,
            block_idx: usize,
            block: *const Block,
            psi_in: []const FieldType,
            psi_out: []FieldType,
            psi_ghost_slices: [num_faces][]const FieldType,
        ) void {
            self.applyToBlockRegion(.all, block_idx, block, psi_in, psi_out, psi_ghost_slices);
        }

        const SiteRegion = enum {
            all,
            interior,
            boundary,
        };

        fn executeRegion(
            self: *Self,
            region: SiteRegion,
            block_idx: usize,
            block: *const Block,
            inputs: anytype,
            outputs: anytype,
            ghosts: ?*GhostBuffer,
        ) void {
            const psi_in_arena = inputs;
            const psi_out_arena = outputs;

            const slot = self.gauge_tree.tree.getFieldSlot(block_idx);
            const psi = psi_in_arena.getSlotConst(slot);
            const out = psi_out_arena.getSlot(slot);

            var psi_ghost_slices: [num_faces][]const FieldType = undefined;
            for (0..num_faces) |f| {
                psi_ghost_slices[f] = &.{};
            }

            if (ghosts) |g| {
                if (g.get(block_idx)) |gp| {
                    for (0..num_faces) |f| {
                        psi_ghost_slices[f] = &gp[f];
                    }
                }
            }

            self.applyToBlockRegion(region, block_idx, block, psi, out, psi_ghost_slices);
        }

        fn applyToBlockRegion(
            self: *Self,
            region: SiteRegion,
            block_idx: usize,
            block: *const Block,
            psi_in: []const FieldType,
            psi_out: []FieldType,
            psi_ghost_slices: [num_faces][]const FieldType,
        ) void {
            const spacing = block.spacing;
            const kinetic_pf = -1.0 / (2.0 * self.mass);

            for (0..block_volume) |i| {
                const coords = Block.getLocalCoords(i);
                const on_boundary = Block.isOnBoundary(coords);
                switch (region) {
                    .all => {},
                    .interior => if (on_boundary) continue,
                    .boundary => if (!on_boundary) continue,
                }

                // Use GaugeTree's covariant Laplacian (already includes 1/a²)
                const laplacian = self.gauge_tree.covariantLaplacianSite(
                    block_idx,
                    i,
                    psi_in,
                    psi_ghost_slices,
                    spacing,
                );

                // Apply potential
                const pos = block.getPhysicalPosition(i);
                const potential = self.potential_fn(pos, spacing);

                // H·ψ = kinetic + potential
                inline for (0..N_field) |a| {
                    const kinetic = laplacian[a].mul(Complex.init(kinetic_pf, 0));
                    const pot_term = psi_in[i][a].mul(Complex.init(potential, 0));
                    psi_out[i][a] = kinetic.add(pot_term);
                }
            }
        }

        // =====================================================================
        // Energy Measurement
        // =====================================================================

        /// Compute total energy: E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
        pub fn measureEnergy(
            self: *Self,
            psi_arena: *const FieldArena,
            workspace: *FieldArena,
            ghosts: *GhostBuffer,
        ) !f64 {
            try self.apply(psi_arena, workspace, ghosts, null);

            var expectation: f64 = 0.0;
            var norm_sq: f64 = 0.0;

            for (self.gauge_tree.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    const slot = self.gauge_tree.tree.getFieldSlot(idx);
                    if (slot != std.math.maxInt(usize)) {
                        const psi = psi_arena.getSlotConst(slot);
                        const hpsi = workspace.getSlotConst(slot);
                        var a_n: f64 = 1.0;
                        for (0..Nd) |_| {
                            a_n *= block.spacing;
                        }
                        for (0..block_volume) |i| {
                            inline for (0..N_field) |a| {
                                const psi_conj = psi[i][a].conjugate();
                                expectation += psi_conj.mul(hpsi[i][a]).re * a_n;
                                const c = psi[i][a];
                                norm_sq += (c.re * c.re + c.im * c.im) * a_n;
                            }
                        }
                    }
                }
            }

            // Guard against divide-by-zero (empty arena or all-zero fields)
            if (norm_sq <= 0.0 or !std.math.isFinite(norm_sq)) return 0.0;
            return expectation / norm_sq;
        }

        /// Compute norm squared with proper volume weighting
        pub fn normSquaredAMR(self: *const Self, psi_arena: *const FieldArena) f64 {
            var norm_sq: f64 = 0.0;

            for (self.gauge_tree.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    const slot = self.gauge_tree.tree.getFieldSlot(idx);
                    if (slot != std.math.maxInt(usize)) {
                        const psi = psi_arena.getSlotConst(slot);
                        var a_n: f64 = 1.0;
                        for (0..Nd) |_| {
                            a_n *= block.spacing;
                        }
                        for (0..block_volume) |i| {
                            inline for (0..N_field) |a| {
                                const c = psi[i][a];
                                norm_sq += (c.re * c.re + c.im * c.im) * a_n;
                            }
                        }
                    }
                }
            }

            return norm_sq;
        }

        /// Normalize the wavefunction
        pub fn normalizeAMR(self: *const Self, psi_arena: *FieldArena) void {
            const norm_sq = self.normSquaredAMR(psi_arena);
            if (norm_sq <= 0.0 or !std.math.isFinite(norm_sq)) return;

            const inv_norm = 1.0 / @sqrt(norm_sq);

            for (self.gauge_tree.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    const slot = self.gauge_tree.tree.getFieldSlot(idx);
                    if (slot != std.math.maxInt(usize)) {
                        const psi = psi_arena.getSlot(slot);
                        for (0..block_volume) |i| {
                            inline for (0..N_field) |a| {
                                psi[i][a] = psi[i][a].mul(Complex.init(inv_norm, 0));
                            }
                        }
                    }
                }
            }
        }

        // =====================================================================
        // Imaginary Time Evolution
        // =====================================================================

        /// Evolve in imaginary time
        pub fn evolveImaginaryTimeAMR(
            self: *Self,
            psi_arena: *FieldArena,
            workspace: *FieldArena,
            ghosts: *GhostBuffer,
            delta_tau: f64,
            num_steps: usize,
            normalize_interval: usize,
            adapt_interval: usize,
            adapt_threshold: f64,
            adapt_hysteresis: f64,
        ) !void {
            const dt = Complex.init(delta_tau, 0);
            self.current_dt = delta_tau; // Set DT for flux register

            // Init FluxRegister
            var flux_reg = GaugeTree.TreeType.FluxRegister.init(self.gauge_tree.allocator);
            defer flux_reg.deinit();

            for (0..num_steps) |step| {
                const total_faces = self.gauge_tree.tree.blockCount() * num_faces;
                const workers = self.gauge_tree.tree.threadCount();
                const reserve_hint = (total_faces + workers - 1) / workers;
                try flux_reg.clearAndReserve(reserve_hint);

                try self.apply(psi_arena, workspace, ghosts, &flux_reg);

                for (self.gauge_tree.tree.blocks.items, 0..) |*block, idx| {
                    if (block.block_index != std.math.maxInt(usize)) {
                        const slot = self.gauge_tree.tree.getFieldSlot(idx);
                        if (slot != std.math.maxInt(usize)) {
                            const psi = psi_arena.getSlot(slot);
                            const hpsi = workspace.getSlotConst(slot);
                            for (0..block_volume) |i| {
                                inline for (0..N_field) |a| {
                                    psi[i][a] = psi[i][a].sub(dt.mul(hpsi[i][a]));
                                }
                            }
                        }
                    }
                }

                // Reflux correction
                flux_reg.reflux(&self.gauge_tree.tree, psi_arena);

                if (normalize_interval > 0 and (step + 1) % normalize_interval == 0) {
                    self.normalizeAMR(psi_arena);
                }

                if (adapt_interval > 0 and (step + 1) % adapt_interval == 0) {
                    _ = try adaptation_mod.adaptMesh(GaugeTree.TreeType, &self.gauge_tree.tree, psi_arena, adapt_threshold, adapt_hysteresis);
                    try ghosts.ensureForTree(&self.gauge_tree.tree);
                }
            }

            self.normalizeAMR(psi_arena);
        }

        /// Evolve until energy converges
        pub fn evolveUntilConvergedAMR(
            self: *Self,
            psi_arena: *FieldArena,
            workspace: *FieldArena,
            ghosts: *GhostBuffer,
            delta_tau: f64,
            max_steps: usize,
            energy_tolerance: f64,
            normalize_interval: usize,
            adapt_interval: usize,
            adapt_threshold: f64,
            adapt_hysteresis: f64,
        ) !struct { steps: usize, energy: f64, converged: bool } {
            const convergence_check_interval = constants.evolution_convergence_check_interval;
            var prev_energy = try self.measureEnergy(psi_arena, workspace, ghosts);
            const dt = Complex.init(delta_tau, 0);
            self.current_dt = delta_tau; // Set DT for flux register

            // Init FluxRegister
            var flux_reg = GaugeTree.TreeType.FluxRegister.init(self.gauge_tree.allocator);
            defer flux_reg.deinit();

            for (0..max_steps) |step| {
                const total_faces = self.gauge_tree.tree.blockCount() * num_faces;
                const workers = self.gauge_tree.tree.threadCount();
                const reserve_hint = (total_faces + workers - 1) / workers;
                try flux_reg.clearAndReserve(reserve_hint);

                try self.apply(psi_arena, workspace, ghosts, &flux_reg);

                for (self.gauge_tree.tree.blocks.items, 0..) |*block, idx| {
                    if (block.block_index != std.math.maxInt(usize)) {
                        const slot = self.gauge_tree.tree.getFieldSlot(idx);
                        if (slot != std.math.maxInt(usize)) {
                            const psi = psi_arena.getSlot(slot);
                            const hpsi = workspace.getSlotConst(slot);
                            for (0..block_volume) |i| {
                                inline for (0..N_field) |a| {
                                    psi[i][a] = psi[i][a].sub(dt.mul(hpsi[i][a]));
                                }
                            }
                        }
                    }
                }

                // Reflux correction
                flux_reg.reflux(&self.gauge_tree.tree, psi_arena);

                if (normalize_interval > 0 and (step + 1) % normalize_interval == 0) {
                    self.normalizeAMR(psi_arena);
                }

                if (adapt_interval > 0 and (step + 1) % adapt_interval == 0) {
                    _ = try adaptation_mod.adaptMesh(GaugeTree.TreeType, &self.gauge_tree.tree, psi_arena, adapt_threshold, adapt_hysteresis);
                    try ghosts.ensureForTree(&self.gauge_tree.tree);
                }

                if ((step + 1) % convergence_check_interval == 0) {
                    self.normalizeAMR(psi_arena);
                    const current_energy = try self.measureEnergy(psi_arena, workspace, ghosts);

                    if (@abs(current_energy - prev_energy) < energy_tolerance) {
                        return .{ .steps = step + 1, .energy = current_energy, .converged = true };
                    }
                    prev_energy = current_energy;
                }
            }

            self.normalizeAMR(psi_arena);
            return .{
                .steps = max_steps,
                .energy = try self.measureEnergy(psi_arena, workspace, ghosts),
                .converged = false,
            };
        }
    };
}

// =====================================================================
// Standard Potential Functions
// =====================================================================

/// Coulomb potential: V(r) = -alpha/r (natural units)
pub fn coulombPotential(pos: [4]f64, spacing: f64) f64 {
    const x = pos[1];
    const y = pos[2];
    const z = pos[3];
    const r_sq = x * x + y * y + z * z;
    const delta_sq = spacing * spacing;
    const r = @sqrt(r_sq + delta_sq);
    return -constants.fine_structure_constant / r;
}

/// Yukawa potential factory
pub fn yukawaFactory(comptime g_sq_over_4pi: f64, comptime mu: f64) fn ([4]f64, f64) f64 {
    return struct {
        fn potential(pos: [4]f64, spacing: f64) f64 {
            const x = pos[1];
            const y = pos[2];
            const z = pos[3];
            const r_sq = x * x + y * y + z * z;
            const delta_sq = spacing * spacing;
            const r = @sqrt(r_sq + delta_sq);
            return -g_sq_over_4pi * @exp(-mu * r) / r;
        }
    }.potential;
}

/// Free particle: V(r) = 0
pub fn freeParticle(pos: [4]f64, spacing: f64) f64 {
    _ = pos;
    _ = spacing;
    return 0.0;
}

// Tests are owned by integration suites.

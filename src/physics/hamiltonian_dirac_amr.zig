//! Dirac Hamiltonian on Block-Structured AMR
//!
//! Implements the relativistic Dirac Hamiltonian on an adaptive mesh refinement
//! (AMR) grid with gauge-covariant derivatives. This extends HamiltonianDirac to
//! work with block-structured AMR wavefunctions stored in FieldArena.
//!
//! ## Physical Model
//!
//! The Dirac Hamiltonian in natural units (hbar = c = 1):
//!   H = alpha . (-iD) + beta*m + V
//!
//! Where:
//! - alpha^i = gamma^0 gamma^i are the Dirac alpha matrices (i = 1, 2, 3 for spatial)
//! - beta = gamma^0 is the Dirac beta matrix
//! - D_mu = d_mu - ieA_mu is the gauge-covariant derivative
//! - m is the fermion mass
//! - V is an external potential (e.g., Coulomb for hydrogen)
//!
//! ## AMR Integration
//!
//! Uses GaugeFrontend from the gauge module which defines:
//! - N_field = 4 * N_gauge (4 spinor components x N_gauge gauge indices)
//! - Field layout: [spinor_idx * N_gauge + gauge_idx]
//!
//! Gauge links are owned by GaugeField; AMR ghost layers handle inter-block
//! communication for matter fields via the Push Model.
//!
//! ## Units
//!
//! Natural units: hbar = c = 1, m_e = 1 (for atomic)
//! Energy in m_e, length in 1/m_e.

const std = @import("std");
const amr_mod = @import("amr");
const gauge = @import("gauge");
const constants = @import("constants");

const Complex = std.math.Complex(f64);

/// Potential types for the Dirac Hamiltonian on AMR.
pub const PotentialType = enum {
    /// Free particle: V = 0
    none,
    /// Coulomb potential: V = -Z*alpha/r with soft-core regularization
    coulomb,
    /// Custom user-provided potential function
    custom,
};

/// Dirac Hamiltonian on AMR blocks with gauge-covariant derivatives.
///
/// Parameters:
/// - N_gauge: Gauge group dimension (1 for U(1), 2 for SU(2), 3 for SU(3))
/// - block_size: Sites per dimension in each block
/// - Topology: Grid topology type defining boundary conditions
///
/// Note: Topology is the 3rd parameter here, unlike GaugeFrontend where it's 5th.
/// This is because HamiltonianDiracAMR fixes Nd=4 and N_spinor=4 internally.
///
/// Uses GaugeFrontend internally for Frontend interface compatibility.
/// The matter field dimension is automatically N_field = 4 * N_gauge for Dirac spinors.
pub fn HamiltonianDiracAMR(comptime N_gauge: usize, comptime block_size: usize, comptime Topology: type) type {
    // Use GaugeFrontend: 4D spacetime, 4 spinor components
    const Frontend = gauge.GaugeFrontend(N_gauge, 4, 4, block_size, Topology);
    const Nd = Frontend.Nd;
    const N_field = Frontend.field_dim;

    const Tree = amr_mod.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Block = Tree.BlockType;
    const Arena = amr_mod.FieldArena(Frontend);
    const GhostBuffer = amr_mod.GhostBuffer(Frontend);
    const ApplyContext = amr_mod.ApplyContext(Frontend);
    const Link = Frontend.LinkType;

    return struct {
        const Self = @This();

        // Type exports for external use
        pub const FrontendType = Frontend;
        pub const TreeType = Tree;
        pub const GaugeFieldType = GaugeField;
        pub const BlockType = Block;
        pub const LinkType = Link;
        pub const FieldArena = Arena;
        pub const GhostBufferType = GhostBuffer;

        pub const block_volume = Frontend.volume;
        pub const num_faces = 2 * Nd;
        pub const ghost_face_size = Frontend.face_size;
        pub const spinor_components = 4;
        pub const gauge_dim = N_gauge;
        pub const field_dim = N_field;

        /// Reference to the AMR tree (not owned)
        tree: *Tree,

        /// Reference to gauge link storage
        field: *GaugeField,

        /// Particle mass
        mass: f64,

        /// Nuclear charge for Coulomb potential
        Z: f64,

        /// Potential type
        potential_type: PotentialType,

        /// Custom potential function (if potential_type == .custom)
        custom_potential: ?*const fn ([Nd]f64, f64) f64,

        /// Center of the potential (for Coulomb)
        center: [Nd]f64,

        /// Soft-core regularization parameter (for Coulomb)
        soft_core: f64,

        /// Initialize the Dirac AMR Hamiltonian.
        pub fn init(
            tree: *Tree,
            field: *GaugeField,
            mass: f64,
            Z: f64,
            potential_type: PotentialType,
        ) Self {
            // Center of potential at domain center
            const block_extent = @as(f64, @floatFromInt(block_size));
            const center = [Nd]f64{
                block_extent / 2.0,
                block_extent / 2.0,
                block_extent / 2.0,
                block_extent / 2.0,
            };

            return Self{
                .tree = tree,
                .field = field,
                .mass = mass,
                .Z = Z,
                .potential_type = potential_type,
                .custom_potential = null,
                .center = center,
                .soft_core = 0.1, // Default soft-core for Coulomb regularization
            };
        }

        /// Initialize with custom potential function.
        pub fn initWithCustomPotential(
            tree: *Tree,
            field: *GaugeField,
            mass: f64,
            potential_fn: *const fn ([Nd]f64, f64) f64,
        ) Self {
            var self = init(tree, field, mass, 1.0, .custom);
            self.custom_potential = potential_fn;
            return self;
        }

        pub fn deinit(self: *Self) void {
            _ = self;
        }

        /// Set the center of the Coulomb potential.
        pub fn setCenter(self: *Self, center: [Nd]f64) void {
            self.center = center;
        }

        /// Set the soft-core regularization parameter.
        pub fn setSoftCore(self: *Self, soft_core: f64) void {
            self.soft_core = soft_core;
        }

        /// Execute the Dirac Hamiltonian on a single block.
        /// This is the kernel interface for AMRTree.apply().
        /// H = alpha . (-iD) + beta*m + V
        pub fn execute(
            self: *const Self,
            block_idx: usize,
            block: *const Block,
            ctx: *ApplyContext,
        ) void {
            const tree = ctx.tree;
            const slot = tree.getFieldSlot(block_idx);
            if (slot == std.math.maxInt(usize)) return;

            const psi_in_arena = ctx.field_in orelse return;
            const psi_out_arena = ctx.field_out orelse return;

            const psi = psi_in_arena.getSlotConst(slot);
            const out = psi_out_arena.getSlot(slot);

            // Get ghost slices
            var psi_ghost_slices: [num_faces][]const [N_field]Complex = undefined;
            for (0..num_faces) |f| {
                psi_ghost_slices[f] = &.{};
            }

            if (ctx.field_ghosts) |ghosts| {
                if (ghosts.get(block_idx)) |gp| {
                    for (0..num_faces) |f| {
                        psi_ghost_slices[f] = &gp[f];
                    }
                }
            }

            // Apply Hamiltonian to all sites
            self.applyToBlock(block_idx, block, psi, out, psi_ghost_slices);
        }

        /// Apply Dirac Hamiltonian: H psi = [alpha . (-iD) + beta*m + V] psi.
        /// Convenience wrapper that creates ApplyContext and calls tree.apply().
        pub fn apply(
            self: *Self,
            psi_in: *const FieldArena,
            psi_out: *FieldArena,
            ghosts: *GhostBuffer,
            flux_reg: ?*Tree.FluxRegister,
        ) !void {
            // Build ApplyContext
            var ctx = ApplyContext.init(self.tree);
            ctx.field_in = psi_in;
            ctx.field_out = psi_out;
            ctx.field_ghosts = ghosts;
            ctx.flux_reg = flux_reg;

            try self.field.fillGhosts(self.tree);
            ctx.setEdges(&self.field.arena, &self.field.ghosts);
            ctx.edge_ghosts_dirty = false;
            try self.tree.apply(self, &ctx);
        }

        /// Apply Dirac Hamiltonian to a single block (internal).
        fn applyToBlock(
            self: *const Self,
            block_idx: usize,
            block: *const Block,
            psi_in: []const [N_field]Complex,
            psi_out: [][N_field]Complex,
            psi_ghosts: [num_faces][]const [N_field]Complex,
        ) void {
            self.applyToBlockImpl(block_idx, block, psi_in, psi_out, psi_ghosts);
        }

        fn applyToBlockImpl(
            self: *const Self,
            block_idx: usize,
            block: *const Block,
            psi_in: []const [N_field]Complex,
            psi_out: [][N_field]Complex,
            psi_ghosts: [num_faces][]const [N_field]Complex,
        ) void {
            const spacing = block.spacing;
            const inv_2a = 0.5 / spacing;

            for (0..block_volume) |site| {
                const coords = Block.getLocalCoords(site);
                const pos = block.getPhysicalPosition(site);

                // Get spinor components at current site
                // Layout: psi[spinor_idx * N_gauge + gauge_idx]
                var psi_local: [4][N_gauge]Complex = undefined;
                inline for (0..4) |s| {
                    inline for (0..N_gauge) |g| {
                        psi_local[s][g] = psi_in[site][s * N_gauge + g];
                    }
                }

                // Compute covariant derivatives in spatial directions (x, y, z = 1, 2, 3)
                // D_mu psi = [U_mu(x) psi(x+mu) - U_dag_mu(x-mu) psi(x-mu)] / (2a)
                var dx: [4][N_gauge]Complex = undefined;
                var dy: [4][N_gauge]Complex = undefined;
                var dz: [4][N_gauge]Complex = undefined;

                // X-direction (mu = 1)
                self.computeCovariantDerivative(block_idx, block, psi_in, psi_ghosts, site, coords, 1, inv_2a, &dx);

                // Y-direction (mu = 2)
                self.computeCovariantDerivative(block_idx, block, psi_in, psi_ghosts, site, coords, 2, inv_2a, &dy);

                // Z-direction (mu = 3)
                self.computeCovariantDerivative(block_idx, block, psi_in, psi_ghosts, site, coords, 3, inv_2a, &dz);

                // Compute -i*alpha.D psi using Dirac alpha matrix structure
                // In Dirac representation: alpha_i = sigma_1 x sigma_i = [[0, sigma_i], [sigma_i, 0]]
                //
                // alpha1(D_x) gives: (dx_3, dx_2, dx_1, dx_0)
                // alpha2(D_y) gives: (-i dy_3, i dy_2, -i dy_1, i dy_0)
                // alpha3(D_z) gives: (dz_2, -dz_3, dz_0, -dz_1)
                //
                // Then multiply by -i:
                // -i*alpha1(Dx): -i(dx_3, dx_2, dx_1, dx_0)
                // -i*alpha2(Dy): (-dy_3, dy_2, -dy_1, dy_0)
                // -i*alpha3(Dz): -i(dz_2, -dz_3, dz_0, -dz_1)

                var kinetic: [4][N_gauge]Complex = undefined;

                inline for (0..N_gauge) |g| {
                    // Component 0: -i*dx[3] + (-dy[3]) + (-i)*dz[2]
                    kinetic[0][g] = Complex.init(
                        dx[3][g].im - dy[3][g].re + dz[2][g].im,
                        -dx[3][g].re - dy[3][g].im - dz[2][g].re,
                    );

                    // Component 1: -i*dx[2] + dy[2] + i*dz[3]
                    kinetic[1][g] = Complex.init(
                        dx[2][g].im + dy[2][g].re - dz[3][g].im,
                        -dx[2][g].re + dy[2][g].im + dz[3][g].re,
                    );

                    // Component 2: -i*dx[1] + (-dy[1]) + (-i)*dz[0]
                    kinetic[2][g] = Complex.init(
                        dx[1][g].im - dy[1][g].re + dz[0][g].im,
                        -dx[1][g].re - dy[1][g].im - dz[0][g].re,
                    );

                    // Component 3: -i*dx[0] + dy[0] + i*dz[1]
                    kinetic[3][g] = Complex.init(
                        dx[0][g].im + dy[0][g].re - dz[1][g].im,
                        -dx[0][g].re + dy[0][g].im + dz[1][g].re,
                    );
                }

                // Mass term: beta m psi where beta = diag(1, 1, -1, -1)
                var mass_term: [4][N_gauge]Complex = undefined;
                inline for (0..N_gauge) |g| {
                    mass_term[0][g] = Complex.init(psi_local[0][g].re * self.mass, psi_local[0][g].im * self.mass);
                    mass_term[1][g] = Complex.init(psi_local[1][g].re * self.mass, psi_local[1][g].im * self.mass);
                    mass_term[2][g] = Complex.init(-psi_local[2][g].re * self.mass, -psi_local[2][g].im * self.mass);
                    mass_term[3][g] = Complex.init(-psi_local[3][g].re * self.mass, -psi_local[3][g].im * self.mass);
                }

                // Potential term: V(x) psi
                const V = self.computePotential(pos, spacing);
                var pot_term: [4][N_gauge]Complex = undefined;
                inline for (0..4) |s| {
                    inline for (0..N_gauge) |g| {
                        pot_term[s][g] = Complex.init(psi_local[s][g].re * V, psi_local[s][g].im * V);
                    }
                }

                // Sum all contributions and write to output
                inline for (0..4) |s| {
                    inline for (0..N_gauge) |g| {
                        psi_out[site][s * N_gauge + g] = Complex.init(
                            kinetic[s][g].re + mass_term[s][g].re + pot_term[s][g].re,
                            kinetic[s][g].im + mass_term[s][g].im + pot_term[s][g].im,
                        );
                    }
                }
            }
        }

        /// Compute gauge-covariant derivative in direction mu.
        /// D_mu psi = [U_mu(x) psi(x+mu) - U_dag_mu(x-mu) psi(x-mu)] / (2a)
        fn computeCovariantDerivative(
            self: *const Self,
            block_idx: usize,
            block: *const Block,
            psi_in: []const [N_field]Complex,
            psi_ghosts: [num_faces][]const [N_field]Complex,
            site: usize,
            coords: [Nd]usize,
            comptime mu: usize,
            inv_2a: f64,
            out: *[4][N_gauge]Complex,
        ) void {
            _ = block;
            const face_plus = mu * 2;
            const face_minus = mu * 2 + 1;

            // Forward neighbor
            var psi_plus: [N_field]Complex = undefined;
            const link_fwd = self.field.getLink(block_idx, site, mu);

            if (coords[mu] == block_size - 1) {
                const ghost_idx = Block.getGhostIndex(coords, face_plus);
                if (ghost_idx < psi_ghosts[face_plus].len) {
                    psi_plus = psi_ghosts[face_plus][ghost_idx];
                } else {
                    psi_plus = Frontend.zeroField();
                }
            } else {
                const neighbor_plus = Block.localNeighborFast(site, face_plus);
                psi_plus = psi_in[neighbor_plus];
            }

            // Backward neighbor
            var psi_minus: [N_field]Complex = undefined;
            var link_bwd: Link = undefined;

            if (coords[mu] == 0) {
                const ghost_idx = Block.getGhostIndex(coords, face_minus);
                if (ghost_idx < psi_ghosts[face_minus].len) {
                    psi_minus = psi_ghosts[face_minus][ghost_idx];
                } else {
                    psi_minus = Frontend.zeroField();
                }

                link_bwd = Link.identity();
                const neighbor_info = self.tree.neighborInfo(block_idx, face_minus);
                if (neighbor_info.exists() and neighbor_info.level_diff == 0) {
                    const neighbor_idx = neighbor_info.block_idx;
                    if (neighbor_idx < self.tree.blocks.items.len) {
                        var neighbor_coords = coords;
                        neighbor_coords[mu] = block_size - 1;
                        const neighbor_site = Block.getLocalIndex(neighbor_coords);
                        link_bwd = self.field.getLink(neighbor_idx, neighbor_site, mu).adjoint();
                    }
                }
            } else {
                const neighbor_minus = Block.localNeighborFast(site, face_minus);
                psi_minus = psi_in[neighbor_minus];
                // Backward link: U_dag(x-mu, mu) = adjoint of link at neighbor pointing forward
                link_bwd = self.field.getLink(block_idx, neighbor_minus, mu).adjoint();
            }

            // Apply links and compute derivative for each spinor component
            inline for (0..4) |s| {
                // Extract gauge part for this spinor component
                var gauge_plus: [N_gauge]Complex = undefined;
                var gauge_minus: [N_gauge]Complex = undefined;
                inline for (0..N_gauge) |g| {
                    gauge_plus[g] = psi_plus[s * N_gauge + g];
                    gauge_minus[g] = psi_minus[s * N_gauge + g];
                }

                // Apply links
                const transported_plus = link_fwd.actOnVector(gauge_plus);
                const transported_minus = link_bwd.actOnVector(gauge_minus);

                // Compute (fwd - bwd) / 2a
                inline for (0..N_gauge) |g| {
                    out[s][g] = Complex.init(
                        (transported_plus[g].re - transported_minus[g].re) * inv_2a,
                        (transported_plus[g].im - transported_minus[g].im) * inv_2a,
                    );
                }
            }
        }

        /// Compute potential at physical position.
        fn computePotential(self: *const Self, pos: [Nd]f64, spacing: f64) f64 {
            return switch (self.potential_type) {
                .none => 0.0,
                .coulomb => blk: {
                    // V = -Z*alpha/r with soft-core regularization
                    // r is spatial distance from center
                    const dx = pos[1] - self.center[1];
                    const dy = pos[2] - self.center[2];
                    const dz = pos[3] - self.center[3];
                    const r = @sqrt(dx * dx + dy * dy + dz * dz + self.soft_core * self.soft_core);

                    // Fine structure constant alpha ~ 1/137
                    const alpha = constants.fine_structure_constant;
                    break :blk -self.Z * alpha / r;
                },
                .custom => if (self.custom_potential) |fn_ptr| fn_ptr(pos, spacing) else 0.0,
            };
        }

        /// Measure energy expectation value: <psi|H|psi>
        pub fn measureEnergy(
            self: *Self,
            psi: *const FieldArena,
            workspace: *FieldArena,
            ghosts: *GhostBuffer,
        ) !f64 {
            // Apply H to psi
            try self.apply(psi, workspace, ghosts, null);

            // Compute <psi|H psi> with volume weighting
            var energy: f64 = 0.0;

            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                const slot = self.tree.getFieldSlot(idx);
                if (slot == std.math.maxInt(usize)) continue;

                const psi_data = psi.getSlotConst(slot);
                const hpsi_data = workspace.getSlotConst(slot);

                // Volume element: a^4 where a = block spacing
                const volume_element = std.math.pow(f64, block.spacing, 4);

                for (0..block_volume) |i| {
                    var site_energy: f64 = 0.0;
                    inline for (0..N_field) |c| {
                        // Re(psi* . H psi) = Re(psi)*Re(H psi) + Im(psi)*Im(H psi)
                        site_energy += psi_data[i][c].re * hpsi_data[i][c].re +
                            psi_data[i][c].im * hpsi_data[i][c].im;
                    }
                    energy += site_energy * volume_element;
                }
            }

            return energy;
        }

        /// Compute norm squared with volume weighting: integral |psi|^2 d^4x
        pub fn normSquared(self: *const Self, psi: *const FieldArena) f64 {
            var norm_sq: f64 = 0.0;

            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                const slot = self.tree.getFieldSlot(idx);
                if (slot == std.math.maxInt(usize)) continue;

                const psi_data = psi.getSlotConst(slot);
                const volume_element = std.math.pow(f64, block.spacing, 4);

                for (0..block_volume) |i| {
                    var site_norm: f64 = 0.0;
                    inline for (0..N_field) |c| {
                        site_norm += psi_data[i][c].re * psi_data[i][c].re +
                            psi_data[i][c].im * psi_data[i][c].im;
                    }
                    norm_sq += site_norm * volume_element;
                }
            }

            return norm_sq;
        }

        /// Normalize wavefunction with volume weighting.
        pub fn normalize(self: *const Self, psi: *FieldArena) void {
            const norm_sq = self.normSquared(psi);
            if (norm_sq <= 0.0) return;

            const inv_norm = 1.0 / @sqrt(norm_sq);

            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                const slot = self.tree.getFieldSlot(idx);
                if (slot == std.math.maxInt(usize)) continue;

                const psi_data = psi.getSlot(slot);
                for (0..block_volume) |i| {
                    inline for (0..N_field) |c| {
                        psi_data[i][c] = Complex.init(
                            psi_data[i][c].re * inv_norm,
                            psi_data[i][c].im * inv_norm,
                        );
                    }
                }
            }
        }

        /// Evolve in imaginary time: psi(tau+dtau) = psi(tau) - dtau * (H-m)^2 psi(tau)
        /// Projects onto positive energy states. Requires two workspace arenas.
        pub fn evolveImaginaryTimeStep(
            self: *Self,
            psi: *FieldArena,
            workspace1: *FieldArena,
            workspace2: *FieldArena,
            ghosts: *GhostBuffer,
            delta_tau: f64,
        ) !void {
            // 1. Compute H psi -> workspace1
            try self.apply(psi, workspace1, ghosts, null);

            // 2. Compute phi = (H-m) psi -> workspace1
            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                const slot = self.tree.getFieldSlot(idx);
                if (slot == std.math.maxInt(usize)) continue;

                const psi_data = psi.getSlotConst(slot);
                const w1_data = workspace1.getSlot(slot);

                for (0..block_volume) |i| {
                    inline for (0..N_field) |c| {
                        w1_data[i][c] = Complex.init(
                            w1_data[i][c].re - self.mass * psi_data[i][c].re,
                            w1_data[i][c].im - self.mass * psi_data[i][c].im,
                        );
                    }
                }
            }

            // 3. Compute H phi -> workspace2
            try self.apply(workspace1, workspace2, ghosts, null);

            // 4. Compute (H-m) phi and update psi
            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                const slot = self.tree.getFieldSlot(idx);
                if (slot == std.math.maxInt(usize)) continue;

                const psi_data = psi.getSlot(slot);
                const w1_data = workspace1.getSlotConst(slot);
                const w2_data = workspace2.getSlotConst(slot);

                for (0..block_volume) |i| {
                    inline for (0..N_field) |c| {
                        // (H-m) phi = H phi - m phi
                        const op_sq_re = w2_data[i][c].re - self.mass * w1_data[i][c].re;
                        const op_sq_im = w2_data[i][c].im - self.mass * w1_data[i][c].im;

                        // Update psi
                        psi_data[i][c] = Complex.init(
                            psi_data[i][c].re - delta_tau * op_sq_re,
                            psi_data[i][c].im - delta_tau * op_sq_im,
                        );
                    }
                }
            }
        }

        /// Evolve in imaginary time for multiple steps with periodic normalization.
        pub fn evolveImaginaryTime(
            self: *Self,
            psi: *FieldArena,
            workspace1: *FieldArena,
            workspace2: *FieldArena,
            ghosts: *GhostBuffer,
            delta_tau: f64,
            num_steps: usize,
            normalize_interval: usize,
        ) !void {
            for (0..num_steps) |step| {
                try self.evolveImaginaryTimeStep(psi, workspace1, workspace2, ghosts, delta_tau);

                if (normalize_interval > 0 and (step + 1) % normalize_interval == 0) {
                    self.normalize(psi);
                }
            }
        }

        /// Result of convergence evolution
        pub const EvolutionResult = struct {
            steps: usize,
            energy: f64,
            converged: bool,
        };

        /// Evolve in imaginary time until energy converges.
        pub fn evolveUntilConvergedAMR(
            self: *Self,
            psi: *FieldArena,
            workspace1: *FieldArena,
            workspace2: *FieldArena,
            ghosts: *GhostBuffer,
            delta_tau: f64,
            max_steps: usize,
            tolerance: f64,
            check_interval: usize,
        ) !EvolutionResult {
            var current_energy: f64 = 0.0;
            var prev_energy: f64 = 0.0;
            var converged = false;

            // Initial normalization
            self.normalize(psi);
            prev_energy = try self.measureEnergy(psi, workspace1, ghosts);

            var step: usize = 0;
            while (step < max_steps) {
                // Evolve for check_interval steps
                const steps_to_run = @min(check_interval, max_steps - step);
                try self.evolveImaginaryTime(psi, workspace1, workspace2, ghosts, delta_tau, steps_to_run, 0);
                step += steps_to_run;

                // Normalize and measure
                self.normalize(psi);
                current_energy = try self.measureEnergy(psi, workspace1, ghosts);

                // Check convergence
                const diff = @abs(current_energy - prev_energy);
                if (diff < tolerance) {
                    converged = true;
                    break;
                }
                prev_energy = current_energy;
            }

            return EvolutionResult{
                .steps = step,
                .energy = current_energy,
                .converged = converged,
            };
        }
    };
}

/// Free particle potential (V = 0)
pub fn freeParticlePotential(pos: [4]f64, spacing: f64) f64 {
    _ = pos;
    _ = spacing;
    return 0.0;
}

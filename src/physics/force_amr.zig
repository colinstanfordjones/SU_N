//! AMR Force Module for Gauge Dynamics
//!
//! Implements force calculation for Hybrid Monte Carlo (HMC) on AMR grids.
//! Uses AMRTree + GaugeField for all link and ghost operations.
//!
//! ## Usage
//!
//! ```zig
//! const gauge = @import("gauge");
//! const physics = @import("physics");
//!
//! const Frontend = gauge.GaugeFrontend(3, 1, 4, 16);  // SU(3), scalar, 4D, 16^4
//! const Tree = amr.AMRTree(Frontend);
//! const GaugeField = gauge.GaugeField(Frontend);
//! const Force = physics.force_amr.AMRForce(Frontend);
//!
//! var tree = try Tree.init(allocator, 1.0, 4);
//! defer tree.deinit();
//! var field = try GaugeField.init(allocator, &tree);
//! defer field.deinit();
//!
//! var forces = try Force.AlgebraBuffer.init(allocator, 64);
//! defer forces.deinit();
//!
//! Force.computeTreeForces(&tree, &field, &forces, beta);
//! ```
//!
//! ## The Hanging Node Problem
//!
//! At a refinement boundary, a single coarse link U_coarse (length 2a) borders
//! two fine links U_f1, U_f2 (length a). In the constrained-links approach:
//! - Fine boundary links are functions of coarse links: U_fi = prolongate(U_coarse)[i]
//! - The Wilson action S at fine level depends on these constrained links
//! - Force on coarse link = local force + transmitted force via chain rule
//!
//! ## Differentiable Prolongation
//!
//! Uses algebra-based prolongation for all gauge groups:
//! - Extract anti-Hermitian algebra element A from U = exp(A) via matrix logarithm
//! - Split: U^{1/2} = exp(A/2)
//! - Derivative: dA_half/dA = 0.5 (scalar in algebra coordinates)

const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const math = @import("math");

const Complex = std.math.Complex(f64);

/// AMR Force calculation for gauge dynamics.
///
/// Parameterized by a GaugeFrontend which defines:
/// - Nd: Spacetime dimensions
/// - block_size: Sites per dimension
/// - gauge_group_dim: Gauge group dimension
/// - LinkType: Gauge link type
pub fn AMRForce(comptime Frontend: type) type {
    // Validate that this is a gauge frontend
    if (!@hasDecl(Frontend, "gauge_group_dim")) {
        @compileError("AMRForce requires a GaugeFrontend with gauge_group_dim");
    }
    if (!@hasDecl(Frontend, "LinkType")) {
        @compileError("AMRForce requires a GaugeFrontend with LinkType");
    }

    const Nd = Frontend.Nd;
    const block_size = Frontend.block_size;
    const N_gauge = Frontend.gauge_group_dim;
    const Link = Frontend.LinkType;

    comptime {
        if (Nd != 4) @compileError("AMRForce requires exactly 4 dimensions");
    }

    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const LinkOps = Frontend.LinkOperators;
    const Block = Tree.BlockType;
    const Matrix = math.Matrix(Complex, N_gauge, N_gauge);

    return struct {
        const Self = @This();

        pub const FrontendType = Frontend;
        pub const LinkType = Link;
        pub const TreeType = Tree;
        pub const GaugeFieldType = GaugeField;
        pub const MatrixType = Matrix;

        /// Number of links per block
        pub const links_per_block = Block.volume * Nd;

        /// Force value: element of the Lie algebra su(N).
        pub const Force = struct {
            algebra: Matrix,

            pub fn zero() Force {
                return .{ .algebra = Matrix.zero() };
            }

            pub fn fromMatrix(m: Matrix) Force {
                return .{ .algebra = m };
            }

            pub fn add(self: Force, other: Force) Force {
                return .{ .algebra = self.algebra.add(other.algebra) };
            }

            pub fn sub(self: Force, other: Force) Force {
                var result = Matrix.zero();
                for (0..N_gauge) |i| {
                    for (0..N_gauge) |j| {
                        result.data[i][j] = self.algebra.data[i][j].sub(other.algebra.data[i][j]);
                    }
                }
                return .{ .algebra = result };
            }

            pub fn scale(self: Force, s: f64) Force {
                return .{ .algebra = self.algebra.scale(Complex.init(s, 0)) };
            }

            pub fn project(self: Force) Force {
                return projectToAlgebra(self.algebra);
            }

            pub fn normSquared(self: Force) f64 {
                var sum: f64 = 0;
                for (0..N_gauge) |i| {
                    for (0..N_gauge) |j| {
                        const z = self.algebra.data[i][j];
                        sum += z.re * z.re + z.im * z.im;
                    }
                }
                return sum;
            }

            pub fn norm(self: Force) f64 {
                return @sqrt(self.normSquared());
            }
        };

        pub const Momentum = Force;

        // =====================================================================
        // AMR Buffer Types for HMC
        // =====================================================================

        /// Buffer storage for forces or momenta across all blocks in a tree.
        pub const AlgebraBuffer = struct {
            allocator: std.mem.Allocator,
            slices: std.ArrayList([]Force),
            max_blocks: usize,

            pub fn init(allocator: std.mem.Allocator, max_blocks: usize) AlgebraBuffer {
                return .{
                    .allocator = allocator,
                    .slices = std.ArrayList([]Force){},
                    .max_blocks = max_blocks,
                };
            }

            pub fn deinit(self: *AlgebraBuffer) void {
                for (self.slices.items) |slice| {
                    self.allocator.free(slice);
                }
                self.slices.deinit(self.allocator);
            }

            pub fn allocateBlock(self: *AlgebraBuffer) ![]Force {
                if (self.slices.items.len >= self.max_blocks) {
                    return error.BufferFull;
                }
                const slice = try self.allocator.alloc(Force, Block.volume * Nd);
                for (slice) |*f| f.* = Force.zero();
                try self.slices.append(self.allocator, slice);
                return slice;
            }

            pub fn ensureBlocks(self: *AlgebraBuffer, n: usize) !void {
                while (self.slices.items.len < n) {
                    _ = try self.allocateBlock();
                }
            }

            pub fn get(self: *const AlgebraBuffer, block_idx: usize) []Force {
                return self.slices.items[block_idx];
            }

            pub fn getMut(self: *AlgebraBuffer, block_idx: usize) []Force {
                return self.slices.items[block_idx];
            }

            pub fn setZero(self: *AlgebraBuffer) void {
                for (self.slices.items) |slice| {
                    for (slice) |*f| f.* = Force.zero();
                }
            }

            pub fn kineticEnergy(self: *const AlgebraBuffer) f64 {
                var total: f64 = 0;
                for (self.slices.items) |slice| {
                    for (slice) |p| {
                        total += p.normSquared();
                    }
                }
                return 0.5 * total;
            }

            pub fn sampleGaussian(self: *AlgebraBuffer, rng: std.Random) void {
                for (self.slices.items) |slice| {
                    for (slice) |*p| {
                        p.* = sampleGaussianMomentum(rng);
                    }
                }
            }
        };

        fn sampleGaussianMomentum(rng: std.Random) Force {
            var result = Matrix.zero();

            if (N_gauge == 1) {
                const z = rng.floatNorm(f64);
                result.data[0][0] = Complex.init(0, z);
            } else {
                for (0..N_gauge) |i| {
                    for ((i + 1)..N_gauge) |j| {
                        const a = rng.floatNorm(f64) * std.math.sqrt1_2;
                        const b = rng.floatNorm(f64) * std.math.sqrt1_2;
                        result.data[i][j] = Complex.init(a, b);
                        result.data[j][i] = Complex.init(-a, b);
                    }
                }

                var diag_sum: f64 = 0;
                for (0..(N_gauge - 1)) |i| {
                    const z = rng.floatNorm(f64);
                    result.data[i][i] = Complex.init(0, z);
                    diag_sum += z;
                }
                result.data[N_gauge - 1][N_gauge - 1] = Complex.init(0, -diag_sum);
            }

            return Force{ .algebra = result };
        }

        // =====================================================================
        // Algebra Projection
        // =====================================================================

        pub fn projectToAlgebra(m: Matrix) Force {
            var result = Matrix.zero();

            for (0..N_gauge) |i| {
                for (0..N_gauge) |j| {
                    const m_ij = m.data[i][j];
                    const m_ji_conj = Complex.init(m.data[j][i].re, -m.data[j][i].im);
                    const diff = m_ij.sub(m_ji_conj);
                    result.data[i][j] = Complex.init(diff.re * 0.5, diff.im * 0.5);
                }
            }

            if (N_gauge > 1) {
                var trace = Complex.init(0, 0);
                for (0..N_gauge) |i| {
                    trace = trace.add(result.data[i][i]);
                }
                const correction = Complex.init(trace.re / @as(f64, N_gauge), trace.im / @as(f64, N_gauge));
                for (0..N_gauge) |i| {
                    result.data[i][i] = result.data[i][i].sub(correction);
                }
            }

            return .{ .algebra = result };
        }

        // =====================================================================
        // Force Computation
        // =====================================================================

        /// Compute force on a single link using the gauge staple.
        pub fn computeLocalForce(
            tree: *const Tree,
            field: *const GaugeField,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
            beta: f64,
        ) Force {
            const u = field.getLink(block_idx, site_idx, mu);
            const staple = LinkOps.computeStaple(tree, field, block_idx, site_idx, mu);
            const u_staple = u.mul(staple);
            const force = projectToAlgebra(u_staple.matrix);
            return force.scale(-beta / @as(f64, N_gauge));
        }

        /// Compute force with runtime mu parameter
        fn computeLocalForceRuntime(
            tree: *const Tree,
            field: *const GaugeField,
            block_idx: usize,
            site_idx: usize,
            mu: usize,
            beta: f64,
        ) Force {
            var result = Force.zero();
            inline for (0..Nd) |mu_idx| {
                if (mu == mu_idx) {
                    result = computeLocalForce(tree, field, block_idx, site_idx, mu_idx, beta);
                }
            }
            return result;
        }

        /// Kernel for parallel force computation
        const ForceKernel = struct {
            tree: *Tree,
            field: *GaugeField,
            forces: *AlgebraBuffer,
            beta: f64,

            /// Execute kernel on a block using ApplyContext
            pub fn execute(
                self: *const ForceKernel,
                block_idx: usize,
                _: *const Block,
                _: *amr.ApplyContext(Frontend),
            ) void {
                if (block_idx >= self.forces.slices.items.len) return;

                const force_slice = self.forces.slices.items[block_idx];
                for (0..Block.volume) |site| {
                    for (0..Nd) |mu| {
                        const force_idx = site * Nd + mu;
                        if (force_idx < force_slice.len) {
                            force_slice[force_idx] = computeLocalForceRuntime(self.tree, self.field, block_idx, site, mu, self.beta);
                        }
                    }
                }
            }
        };

        /// Compute forces on all links in the tree.
        pub fn computeTreeForces(
            tree: *Tree,
            field: *GaugeField,
            forces: *AlgebraBuffer,
            beta: f64,
        ) !usize {
            // Ensure force buffer has enough blocks
            try forces.ensureBlocks(tree.blocks.items.len);

            var kernel = ForceKernel{ .tree = tree, .field = field, .forces = forces, .beta = beta };

            // Build ApplyContext (no field data, just tree reference)
            var ctx = amr.ApplyContext(Frontend).init(tree);
            try field.fillGhosts(tree);
            try tree.apply(&kernel, &ctx);

            // Accumulate transmitted forces at refinement boundaries
            accumulateTransmittedForces(tree, field, forces, beta);

            // Return count (approximate, since we don't sum in parallel easily)
            // Just return total links?
            return tree.blockCount() * links_per_block;
        }

        fn accumulateTransmittedForces(
            tree: *const Tree,
            field: *const GaugeField,
            forces: *AlgebraBuffer,
            beta: f64,
        ) void {
            const deriv_factor: f64 = 0.5; // Derivative of prolongation

            for (tree.blocks.items, 0..) |*fine_block, fine_idx| {
                if (fine_block.block_index == std.math.maxInt(usize)) continue;

                for (0..(2 * Nd)) |face| {
                    if ((face % 2) != 0) continue; // Only positive faces

                    const neighbor_info = tree.neighborInfo(fine_idx, face);
                    if (!neighbor_info.exists() or neighbor_info.level_diff >= 0) continue;
                    const neighbor_idx = neighbor_info.block_idx;
                    if (neighbor_idx >= tree.blocks.items.len) continue;
                    const neighbor = &tree.blocks.items[neighbor_idx];

                    if (neighbor_idx >= forces.slices.items.len) continue;
                    const coarse_forces = forces.slices.items[neighbor_idx];

                    const face_dim = face / 2;

                    // For each tangential direction
                    for (0..(Nd - 1)) |t_idx| {
                        const link_dim = getTangentialDim(face_dim, t_idx);

                        for (0..Block.ghost_face_size) |ghost_idx| {
                            const ghost_link = getGhostLink(field, fine_idx, face, link_dim, ghost_idx);

                            // Compute simplified ghost staple
                            const ghost_staple = computeGhostStaple(field, fine_idx, face, ghost_idx, link_dim);
                            const u_staple = ghost_link.mul(ghost_staple);
                            const ghost_force = projectToAlgebra(u_staple.matrix).scale(-beta / @as(f64, N_gauge));

                            // Map ghost index to coarse site
                            const face_coords = indexToFaceCoords(ghost_idx);
                            var fine_global: [Nd]usize = undefined;

                            var fc_idx: usize = 0;
                            for (0..Nd) |d| {
                                if (d == face_dim) {
                                    fine_global[d] = fine_block.origin[d] + block_size;
                                } else {
                                    fine_global[d] = fine_block.origin[d] + face_coords[fc_idx];
                                    fc_idx += 1;
                                }
                            }

                            var coarse_coords: [Nd]usize = undefined;
                            var in_bounds = true;
                            for (0..Nd) |d| {
                                const coarse_global = fine_global[d] / 2;
                                if (coarse_global < neighbor.origin[d]) {
                                    in_bounds = false;
                                    break;
                                }
                                coarse_coords[d] = coarse_global - neighbor.origin[d];
                                if (coarse_coords[d] >= block_size) {
                                    in_bounds = false;
                                    break;
                                }
                            }

                            if (!in_bounds) continue;

                            const coarse_site = Block.getLocalIndex(coarse_coords);
                            const coarse_force_idx = coarse_site * Nd + link_dim;

                            if (coarse_force_idx < coarse_forces.len) {
                                coarse_forces[coarse_force_idx] = coarse_forces[coarse_force_idx].add(
                                    ghost_force.scale(deriv_factor),
                                );
                            }
                        }
                    }
                }
            }
        }

        fn getGhostLink(
            field: *const GaugeField,
            block_idx: usize,
            face_idx: usize,
            link_dim: usize,
            ghost_idx: usize,
        ) Link {
            const ghost = field.ghosts.get(block_idx) orelse return Link.identity();
            const slice = ghost.get(face_idx, link_dim);
            if (ghost_idx < slice.len) return slice[ghost_idx];
            return Link.identity();
        }

        fn computeGhostStaple(
            field: *const GaugeField,
            block_idx: usize,
            face: usize,
            ghost_idx: usize,
            link_dim: usize,
        ) Link {
            var staple = Link.zero();
            const face_dim = face / 2;
            const is_positive = (face % 2) == 0;

            const face_coords = indexToFaceCoords(ghost_idx);

            var boundary_coords: [Nd]usize = undefined;
            var fc_idx: usize = 0;
            for (0..Nd) |d| {
                if (d == face_dim) {
                    boundary_coords[d] = if (is_positive) block_size - 1 else 0;
                } else {
                    boundary_coords[d] = face_coords[fc_idx];
                    fc_idx += 1;
                }
            }

            if (boundary_coords[link_dim] < block_size - 1) {
                var shifted_coords = boundary_coords;
                shifted_coords[link_dim] += 1;

                const boundary_site = Block.getLocalIndex(boundary_coords);
                const shifted_site = Block.getLocalIndex(shifted_coords);

                const u_mu_at_x = field.getLink(block_idx, boundary_site, face_dim);
                const u_t_at_x = field.getLink(block_idx, boundary_site, link_dim);
                const u_mu_at_shifted = field.getLink(block_idx, shifted_site, face_dim);

                staple = staple.add(u_mu_at_shifted.adjoint().mul(u_t_at_x.adjoint()).mul(u_mu_at_x));
            }

            return staple;
        }

        fn indexToFaceCoords(idx: usize) [Nd - 1]usize {
            var coords: [Nd - 1]usize = undefined;
            var remaining = idx;
            for (0..(Nd - 1)) |d| {
                coords[d] = remaining % block_size;
                remaining /= block_size;
            }
            return coords;
        }

        fn getTangentialDim(face_dim: usize, tan_idx: usize) usize {
            var count: usize = 0;
            for (0..Nd) |d| {
                if (d == face_dim) continue;
                if (count == tan_idx) return d;
                count += 1;
            }
            return 0;
        }

        // =====================================================================
        // Leapfrog Integration
        // =====================================================================

        pub fn updateMomentumHalfStep(
            momenta: *AlgebraBuffer,
            forces: *const AlgebraBuffer,
            dt: f64,
        ) void {
            const half_dt = 0.5 * dt;
            for (momenta.slices.items, 0..) |mom_slice, idx| {
                if (idx >= forces.slices.items.len) break;
                const force_slice = forces.slices.items[idx];
                for (mom_slice, force_slice) |*p, f| {
                    p.* = p.sub(f.scale(half_dt));
                }
            }
        }

        pub fn updateLinks(
            tree: *Tree,
            field: *GaugeField,
            momenta: *const AlgebraBuffer,
            dt: f64,
        ) void {
            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                if (idx >= momenta.slices.items.len) continue;
                const mom_slice = momenta.slices.items[idx];
                const links = field.getBlockLinksMut(idx) orelse continue;

                for (0..Block.volume) |site| {
                    inline for (0..Nd) |mu| {
                        const force_idx = site * Nd + mu;
                        const p = mom_slice[force_idx];
                        const link_idx = site * Nd + mu;
                        const u_old = links[link_idx];

                        const scaled_algebra = scaleMatrix(p.algebra, dt);
                        const exp_p = expAlgebra(scaled_algebra);

                        links[link_idx] = exp_p.mul(u_old);
                    }
                }
            }
            field.ghosts.invalidateAll();
        }

        pub fn leapfrogIntegrate(
            tree: *Tree,
            field: *GaugeField,
            momenta: *AlgebraBuffer,
            forces: *AlgebraBuffer,
            dt: f64,
            n_steps: usize,
            beta: f64,
        ) !usize {
            var total_force_count: usize = 0;

            forces.setZero();
            total_force_count += try computeTreeForces(tree, field, forces, beta);

            for (0..n_steps) |_| {
                updateMomentumHalfStep(momenta, forces, dt);
                updateLinks(tree, field, momenta, dt);

                forces.setZero();
                total_force_count += try computeTreeForces(tree, field, forces, beta);
                updateMomentumHalfStep(momenta, forces, dt);
            }

            return total_force_count;
        }

        pub fn computeHamiltonian(
            tree: *Tree,
            field: *GaugeField,
            momenta: *const AlgebraBuffer,
            beta: f64,
        ) !f64 {
            try field.fillGhosts(tree);
            const kinetic = momenta.kineticEnergy();
            const potential = LinkOps.wilsonAction(tree, field, beta);
            return kinetic + potential;
        }

        // =====================================================================
        // HMC Metropolis
        // =====================================================================

        pub const HMCResult = struct {
            accepted: bool,
            delta_H: f64,
            final_H: f64,
            initial_H: f64,
        };

        pub fn metropolisAccept(delta_H: f64, rng: std.Random) bool {
            if (delta_H <= 0) {
                return true;
            }
            const prob = @exp(-delta_H);
            const u = rng.float(f64);
            return u < prob;
        }

        pub fn hmcStep(
            tree: *Tree,
            field: *GaugeField,
            momenta: *AlgebraBuffer,
            forces: *AlgebraBuffer,
            dt: f64,
            n_steps: usize,
            beta: f64,
            rng: std.Random,
        ) !HMCResult {
            // Save link configuration
            const backup = try field.saveLinks(tree.allocator);
            defer GaugeField.freeBackup(tree.allocator, backup);

            // Sample fresh momenta
            momenta.sampleGaussian(rng);

            const H_initial = try computeHamiltonian(tree, field, momenta, beta);

            _ = try leapfrogIntegrate(tree, field, momenta, forces, dt, n_steps, beta);

            const H_final = try computeHamiltonian(tree, field, momenta, beta);
            const delta_H = H_final - H_initial;

            const accepted = metropolisAccept(delta_H, rng);

            if (!accepted) {
                field.restoreLinks(backup);
            }

            return .{
                .accepted = accepted,
                .delta_H = delta_H,
                .final_H = H_final,
                .initial_H = H_initial,
            };
        }

        // =====================================================================
        // Differentiable Link Prolongation
        // =====================================================================

        pub const ProlongationResult = struct {
            fine_links: [2]Link,
            derivative_factor: f64,
        };

        pub fn prolongateLinkDifferentiable(coarse: Link) ProlongationResult {
            const algebra = extractAlgebra(coarse);
            const half_algebra = scaleMatrix(algebra, 0.5);
            const half_link = expAlgebra(half_algebra);

            return .{
                .fine_links = .{ half_link, half_link },
                .derivative_factor = 0.5,
            };
        }

        pub fn extractAlgebra(u: Link) Matrix {
            if (N_gauge == 1) {
                return extractAlgebraU1(u);
            } else if (N_gauge == 2) {
                return extractAlgebraSU2(u);
            } else {
                return extractAlgebraSUN(u);
            }
        }

        fn extractAlgebraU1(u: Link) Matrix {
            const phase = u.matrix.data[0][0];
            const angle = std.math.atan2(phase.im, phase.re);
            var a = Matrix.zero();
            a.data[0][0] = Complex.init(0, angle);
            return a;
        }

        fn extractAlgebraSU2(u: Link) Matrix {
            const elem00 = u.matrix.data[0][0];
            const elem01 = u.matrix.data[0][1];

            const a = elem00.re;
            const d = elem00.im;
            const b = elem01.re;
            const c = elem01.im;

            const sin_half = @sqrt(b * b + c * c + d * d);

            var result = Matrix.zero();

            if (sin_half < 1e-10) {
                return result;
            }

            const a_clamped = std.math.clamp(a, -1.0 + 1e-10, 1.0 - 1e-10);
            const half_theta = std.math.acos(a_clamped);

            const inv_sin_half = 1.0 / sin_half;
            const nx = b * inv_sin_half;
            const ny = c * inv_sin_half;
            const nz = d * inv_sin_half;

            result.data[0][0] = Complex.init(0, half_theta * nz);
            result.data[0][1] = Complex.init(half_theta * ny, half_theta * nx);
            result.data[1][0] = Complex.init(-half_theta * ny, half_theta * nx);
            result.data[1][1] = Complex.init(0, -half_theta * nz);

            return result;
        }

        fn extractAlgebraSUN(u: Link) Matrix {
            const id = Matrix.identity();

            var delta = Matrix.zero();
            for (0..N_gauge) |i| {
                for (0..N_gauge) |j| {
                    delta.data[i][j] = u.matrix.data[i][j].sub(id.data[i][j]);
                }
            }

            const delta2 = delta.mul(delta);
            const delta3 = delta2.mul(delta);
            const delta4 = delta3.mul(delta);

            var log_u = Matrix.zero();
            for (0..N_gauge) |i| {
                for (0..N_gauge) |j| {
                    const t1 = delta.data[i][j];
                    const t2 = delta2.data[i][j].mul(Complex.init(-0.5, 0));
                    const t3 = delta3.data[i][j].mul(Complex.init(1.0 / 3.0, 0));
                    const t4 = delta4.data[i][j].mul(Complex.init(-0.25, 0));
                    log_u.data[i][j] = t1.add(t2).add(t3).add(t4);
                }
            }

            return log_u;
        }

        pub fn expAlgebra(algebra: Matrix) Link {
            return Link.fromMatrix(algebra.exp());
        }

        fn scaleMatrix(m: Matrix, s: f64) Matrix {
            var result = Matrix.zero();
            for (0..N_gauge) |i| {
                for (0..N_gauge) |j| {
                    result.data[i][j] = Complex.init(m.data[i][j].re * s, m.data[i][j].im * s);
                }
            }
            return result;
        }

        // =====================================================================
        // Gradient Check Utility
        // =====================================================================

        pub fn gradientCheck(
            tree: *Tree,
            field: *GaugeField,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
            beta: f64,
            epsilon: f64,
        ) !f64 {
            try field.fillGhosts(tree);

            const analytic = computeLocalForce(tree, field, block_idx, site_idx, mu, beta);
            const analytic_norm = analytic.norm();

            if (analytic_norm < 1e-15) {
                return 0.0;
            }

            const original = field.getLink(block_idx, site_idx, mu);

            var numerical = Force.zero();

            for (0..N_gauge) |a| {
                for (0..N_gauge) |b| {
                    if (a > b) continue;

                    var perturb = Matrix.zero();
                    if (a == b) {
                        perturb.data[a][a] = Complex.init(0, epsilon);
                    } else {
                        perturb.data[a][b] = Complex.init(epsilon, 0);
                        perturb.data[b][a] = Complex.init(-epsilon, 0);
                    }

                    const exp_plus = expAlgebra(perturb);
                    field.setLink(block_idx, site_idx, mu, original.mul(exp_plus));
                    try field.fillGhosts(tree);
                    const action_plus = LinkOps.wilsonAction(tree, field, beta);

                    var neg_perturb = Matrix.zero();
                    for (0..N_gauge) |i| {
                        for (0..N_gauge) |j| {
                            neg_perturb.data[i][j] = Complex.init(-perturb.data[i][j].re, -perturb.data[i][j].im);
                        }
                    }
                    const exp_minus = expAlgebra(neg_perturb);
                    field.setLink(block_idx, site_idx, mu, original.mul(exp_minus));
                    try field.fillGhosts(tree);
                    const action_minus = LinkOps.wilsonAction(tree, field, beta);

                    const deriv = (action_plus - action_minus) / (2.0 * epsilon);

                    if (a == b) {
                        numerical.algebra.data[a][a] = numerical.algebra.data[a][a].add(Complex.init(0, -deriv));
                    } else {
                        numerical.algebra.data[a][b] = numerical.algebra.data[a][b].add(Complex.init(-deriv, 0));
                        numerical.algebra.data[b][a] = numerical.algebra.data[b][a].add(Complex.init(deriv, 0));
                    }
                }
            }

            field.setLink(block_idx, site_idx, mu, original);
            field.ghosts.invalidateAll();

            const diff = analytic.sub(numerical);
            return diff.norm() / analytic_norm;
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const amr_topology = @import("amr").topology;
const TestTopology4D = amr_topology.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });

test "AMRForce - basic initialization" {
    const Frontend = gauge.frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const Force = AMRForce(Frontend);

    const f = Force.Force.zero();
    try std.testing.expectEqual(@as(f64, 0.0), f.norm());
}

test "AMRForce - algebra projection" {
    const Frontend = gauge.frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const Force = AMRForce(Frontend);

    const m = Force.MatrixType.identity();
    const projected = Force.projectToAlgebra(m);

    // Identity is Hermitian, so anti-Hermitian part should be zero
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), projected.norm(), 1e-10);
}

test "AMRForce - momentum sampling" {
    const Frontend = gauge.frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const ForceModule = AMRForce(Frontend);

    const allocator = std.testing.allocator;

    var buffer = ForceModule.AlgebraBuffer.init(allocator, 1);
    defer buffer.deinit();

    _ = try buffer.allocateBlock();

    var prng = std.Random.DefaultPrng.init(42);
    buffer.sampleGaussian(prng.random());

    // Momenta should be non-zero after sampling
    const ke = buffer.kineticEnergy();
    try std.testing.expect(ke > 0.0);
}

test "AMRForce - compute forces on single block" {
    const Frontend = gauge.frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const ForceModule = AMRForce(Frontend);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);

    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4);
    defer tree.deinit();
    var field = try GaugeField.init(allocator, &tree);
    defer field.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try field.syncWithTree(&tree);

    var forces = ForceModule.AlgebraBuffer.init(allocator, 4);
    defer forces.deinit();

    const count = try ForceModule.computeTreeForces(&tree, &field, &forces, 1.0);
    try std.testing.expect(count > 0);

    // For identity links, forces should be zero (derivative of action at minimum)
    const first_force = forces.get(0)[0];
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), first_force.norm(), 1e-10);
}

test "AMRForce - HMC step accepts identity" {
    const Frontend = gauge.frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const ForceModule = AMRForce(Frontend);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);

    const allocator = std.testing.allocator;

    var tree = try Tree.init(allocator, 1.0, 4);
    defer tree.deinit();
    var field = try GaugeField.init(allocator, &tree);
    defer field.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try field.syncWithTree(&tree);

    var momenta = ForceModule.AlgebraBuffer.init(allocator, 4);
    defer momenta.deinit();
    _ = try momenta.allocateBlock();

    var forces = ForceModule.AlgebraBuffer.init(allocator, 4);
    defer forces.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const result = try ForceModule.hmcStep(&tree, &field, &momenta, &forces, 0.01, 5, 1.0, prng.random());

    // delta_H should be small for small step size
    try std.testing.expect(@abs(result.delta_H) < 1.0);
}

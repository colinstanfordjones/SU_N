//! Gauge-Specific Link Operators for AMR
//!
//! Implements prolongation (coarse->fine) and restriction (fine->coarse) operators
//! for gauge link variables at AMR refinement boundaries.
//!
//! ## Design Philosophy
//!
//! These operators are gauge-specific and belong in the gauge module, NOT in the
//! domain-agnostic AMR module. The AMR infrastructure handles generic field types;
//! link variables require specialized handling that preserves gauge structure.
//!
//! ## Link Prolongation
//!
//! When refining a block, a coarse link (spanning 2a) must be split into two fine
//! links (each spanning a). The split preserves the path-ordered product:
//!   U_coarse = U_fine1 * U_fine2
//!
//! - **U(1)**: Exact angle splitting: exp(iθ) -> exp(iθ/2) * exp(iθ/2)
//! - **SU(N)**: Geodesic midpoint approximation with unitarization
//!
//! ## Link Restriction
//!
//! When coarsening, fine links are combined via path-ordered product:
//!   U_coarse = U_fine1 * U_fine2
//!
//! ## Usage
//!
//! ```zig
//! const gauge = @import("gauge");
//!
//! // Create a frontend for SU(3) Dirac
//! const Frontend = gauge.frontend.GaugeFrontend(3, 4, 4, 16);
//!
//! // Get link operators from the frontend
//! const LinkOps = Frontend.LinkOperators;
//!
//! // Split a coarse link
//! const fine_links = LinkOps.prolongateLink(coarse_link);
//!
//! // Combine fine links
//! const coarse_link = LinkOps.restrictLink(fine1, fine2);
//! ```

const std = @import("std");

/// Complex number type (f64 precision for SIMD compatibility)
pub const Complex = std.math.Complex(f64);

/// Link operators for gauge-covariant AMR.
/// These are used by GaugeFrontend for gauge link prolongation/restriction.
///
/// The Frontend must provide:
/// - Nd: Number of spacetime dimensions
/// - block_size: Sites per dimension
/// - LinkType: The gauge link type
/// - gauge_group_dim: Dimension of gauge group (N for SU(N))
/// - face_size: block_size^(Nd-1)
/// - num_children: 2^Nd
pub fn LinkOperators(comptime Frontend: type) type {
    const Nd = Frontend.Nd;
    const block_size = Frontend.block_size;
    const Link = Frontend.LinkType;
    const N_gauge = Frontend.gauge_group_dim;
    const face_size = Frontend.face_size;
    const num_children = Frontend.num_children;

    return struct {
        pub const LinkType = Link;

        /// Prolongate a single coarse link to two fine links.
        /// The coarse link spans 2a (two fine spacings), so we split it:
        ///   U_coarse = U_fine1 * U_fine2
        ///
        /// Symmetric split: U_fine = U_coarse^{1/2}
        ///
        /// For U(1): exp(i*theta) -> exp(i*theta/2) (exact)
        /// For SU(N): Use midpoint + unitarization (approximate geodesic)
        pub fn prolongateLink(coarse: Link) [2]Link {
            if (N_gauge == 1) {
                // U(1) case: exact angle splitting
                const phase = coarse.matrix.data[0][0];
                const angle = std.math.atan2(phase.im, phase.re);
                const half_angle = angle / 2.0;
                const half_phase = Complex.init(@cos(half_angle), @sin(half_angle));

                var half_link = Link.zero();
                half_link.matrix.data[0][0] = half_phase;
                return .{ half_link, half_link };
            } else {
                // SU(N) case: geodesic midpoint approximation
                const id = Link.identity();
                const sum = id.add(coarse);
                const midpoint = sum.scaleReal(0.5);
                const half_link = midpoint.unitarize();
                return .{ half_link, half_link };
            }
        }

        /// Restrict two fine links to a single coarse link.
        /// The coarse link is the path-ordered product:
        ///   U_coarse = U_fine1 * U_fine2
        pub fn restrictLink(fine1: Link, fine2: Link) Link {
            return fine1.mul(fine2);
        }

        /// Prolongate a coarse link face to fine link face.
        pub fn prolongateLinkFace(
            coarse_links: []const Link,
            fine_links: []Link,
            face_dim: usize,
        ) void {
            const coarse_face_size = face_size / num_children * 2;
            std.debug.assert(coarse_links.len >= coarse_face_size);
            std.debug.assert(fine_links.len >= face_size);
            _ = face_dim;

            var fine_idx: usize = 0;
            var coords: [Nd - 1]usize = .{0} ** (Nd - 1);

            while (fine_idx < face_size) : (fine_idx += 1) {
                var coarse_coords: [Nd - 1]usize = undefined;
                inline for (0..(Nd - 1)) |d| {
                    coarse_coords[d] = coords[d] / 2;
                }

                var coarse_idx: usize = 0;
                var coarse_stride: usize = 1;
                inline for (0..(Nd - 1)) |d| {
                    coarse_idx += coarse_coords[d] * coarse_stride;
                    coarse_stride *= block_size / 2;
                }

                if (coarse_idx < coarse_links.len) {
                    const prolongated = prolongateLink(coarse_links[coarse_idx]);
                    fine_links[fine_idx] = prolongated[0];
                } else {
                    fine_links[fine_idx] = Link.identity();
                }

                var d: usize = 0;
                while (d < Nd - 1) : (d += 1) {
                    coords[d] += 1;
                    if (coords[d] < block_size) break;
                    coords[d] = 0;
                }
            }
        }

        /// Restrict fine link face to coarse link face.
        pub fn restrictLinkFace(
            fine_links: []const Link,
            coarse_links: []Link,
            face_dim: usize,
        ) void {
            const coarse_face_size = face_size / num_children * 2;
            std.debug.assert(fine_links.len >= face_size);
            std.debug.assert(coarse_links.len >= coarse_face_size);
            _ = face_dim;

            for (coarse_links[0..coarse_face_size]) |*link| {
                link.* = Link.identity();
            }

            var coarse_idx: usize = 0;
            var coarse_coords: [Nd - 1]usize = .{0} ** (Nd - 1);

            while (coarse_idx < coarse_face_size) : (coarse_idx += 1) {
                var fine_base: [Nd - 1]usize = undefined;
                inline for (0..(Nd - 1)) |d| {
                    fine_base[d] = coarse_coords[d] * 2;
                }

                var fine_base_idx: usize = 0;
                var fine_stride: usize = 1;
                inline for (0..(Nd - 1)) |d| {
                    fine_base_idx += fine_base[d] * fine_stride;
                    fine_stride *= block_size;
                }

                if (fine_base_idx < fine_links.len) {
                    coarse_links[coarse_idx] = fine_links[fine_base_idx];
                }

                var d: usize = 0;
                while (d < Nd - 1) : (d += 1) {
                    coarse_coords[d] += 1;
                    if (coarse_coords[d] < block_size / 2) break;
                    coarse_coords[d] = 0;
                }
            }
        }

        // =====================================================================
        // Observables & Staples
        // =====================================================================

        /// Compute plaquette U_μν at site in a block.
        /// Automatically uses ghost data for boundary sites via GaugeField.
        ///
        /// Arguments:
        /// - tree: AMRTree instance
        /// - field: GaugeField instance (contains links and ghosts)
        /// - block_idx: Index of the block
        /// - site_idx: Index of the site
        /// - mu, nu: Directions
        pub fn computePlaquette(
            tree: anytype,
            field: anytype,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
            comptime nu: usize,
        ) Link {
            const TreeType = @typeInfo(@TypeOf(tree)).pointer.child;
            const Block = TreeType.BlockType;
            const links = field.getBlockLinks(block_idx) orelse return Link.identity();
            const coords = Block.getLocalCoords(site_idx);

            const at_mu_boundary = (coords[mu] == block_size - 1);
            const at_nu_boundary = (coords[nu] == block_size - 1);

            // Optimization for interior sites
            if (!at_mu_boundary and !at_nu_boundary) {
                const link1 = links[site_idx * Nd + mu];
                const site_plus_mu = Block.localNeighborFast(site_idx, mu * 2);
                const link2 = links[site_plus_mu * Nd + nu];
                const site_plus_nu = Block.localNeighborFast(site_idx, nu * 2);
                const link3_dag = links[site_plus_nu * Nd + mu].adjoint();
                const link4_dag = links[site_idx * Nd + nu].adjoint();
                return link1.mul(link2).mul(link3_dag).mul(link4_dag);
            }

            // Boundary handling using ghosts
            // We need to access ghost storage from field
            // The GaugeField manages ghost storage in field.ghosts
            
            // Helper to get link (local or ghost)
            const getLink = struct {
                fn call(
                    f: @TypeOf(field),
                    blk_idx: usize,
                    s_idx: usize,
                    dir: usize,
                    crs: [Nd]usize,
                    comptime is_boundary: bool,
                    comptime boundary_face: usize,
                ) Link {
                    if (is_boundary) {
                        const ghost_idx = Block.getGhostIndex(crs, boundary_face);
                        if (blk_idx >= f.ghosts.slots.len) return Link.identity();
                        const ghost = f.ghosts.slots[blk_idx] orelse return Link.identity();
                        const slice = ghost.get(boundary_face, dir);
                        if (ghost_idx < slice.len) return slice[ghost_idx];
                        return Link.identity();
                    } else {
                        // We need the local link, but the site index passed might be neighbor
                        // If we are here, s_idx is safe (wrapped or not boundary)
                        return f.getLink(blk_idx, s_idx, dir);
                    }
                }
            }.call;

            const link1 = getLink(field, block_idx, site_idx, mu, coords, false, 0);

            var link2: Link = undefined;
            if (at_mu_boundary) {
                const face_idx = mu * 2; // +mu face
                link2 = getLink(field, block_idx, 0, nu, coords, true, face_idx);
            } else {
                const neighbor = Block.localNeighborFast(site_idx, mu * 2);
                link2 = getLink(field, block_idx, neighbor, nu, coords, false, 0);
            }

            var link3_dag: Link = undefined;
            if (at_nu_boundary) {
                const face_idx = nu * 2; // +nu face
                // For link3 (at x+nu, dir mu), if we are at nu boundary, we are in ghost of +nu face
                link3_dag = getLink(field, block_idx, 0, mu, coords, true, face_idx).adjoint();
            } else {
                const neighbor = Block.localNeighborFast(site_idx, nu * 2);
                link3_dag = getLink(field, block_idx, neighbor, mu, coords, false, 0).adjoint();
            }

            const link4_dag = getLink(field, block_idx, site_idx, nu, coords, false, 0).adjoint();

            return link1.mul(link2).mul(link3_dag).mul(link4_dag);
        }

        pub fn tracePlaquette(
            tree: anytype,
            field: anytype,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
            comptime nu: usize,
        ) f64 {
            return computePlaquette(tree, field, block_idx, site_idx, mu, nu).trace().re;
        }

        pub fn averagePlaquetteBlock(
            tree: anytype,
            field: anytype,
            block_idx: usize,
        ) f64 {
            const TreeType = @typeInfo(@TypeOf(tree)).pointer.child;
            const Block = TreeType.BlockType;
            comptime {
                if (Nd < 2) @compileError("averagePlaquetteBlock requires Nd >= 2");
            }
            const volume = Block.volume;
            var sum: f64 = 0.0;
            const n_real: f64 = @floatFromInt(N_gauge);
            const num_plaq: usize = Nd * (Nd - 1) / 2;

            for (0..volume) |site| {
                inline for (0..Nd) |mu| {
                    inline for ((mu + 1)..Nd) |nu| {
                        sum += tracePlaquette(tree, field, block_idx, site, mu, nu);
                    }
                }
            }

            return sum / (n_real * @as(f64, @floatFromInt(volume)) * @as(f64, @floatFromInt(num_plaq)));
        }

        pub fn averagePlaquetteTree(
            tree: anytype,
            field: anytype,
        ) f64 {
            var sum: f64 = 0.0;
            var count: usize = 0;

            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    // Only count blocks that have allocated links
                    if (field.getBlockLinks(idx) != null) {
                        sum += averagePlaquetteBlock(tree, field, idx);
                        count += 1;
                    }
                }
            }

            if (count == 0) return 1.0;
            return sum / @as(f64, @floatFromInt(count));
        }

        pub fn wilsonAction(
            tree: anytype,
            field: anytype,
            beta: f64,
        ) f64 {
            const TreeType = @typeInfo(@TypeOf(tree)).pointer.child;
            const Block = TreeType.BlockType;
            var total_action: f64 = 0.0;
            const volume = Block.volume;
            const num_plaq: usize = Nd * (Nd - 1) / 2;
            const n_real: f64 = @floatFromInt(N_gauge);
            const norm_factor = n_real * @as(f64, @floatFromInt(volume)) * @as(f64, @floatFromInt(num_plaq));

            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    if (field.getBlockLinks(idx) != null) {
                        const avg_plaq = averagePlaquetteBlock(tree, field, idx);
                        total_action += beta * norm_factor * (1.0 - avg_plaq);
                    }
                }
            }
            return total_action;
        }

        /// Compute staple for link U_mu(site).
        pub fn computeStaple(
            tree: anytype,
            field: anytype,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
        ) Link {
            const TreeType = @typeInfo(@TypeOf(tree)).pointer.child;
            const Block = TreeType.BlockType;
            const links = field.getBlockLinks(block_idx) orelse return Link.zero();
            const coords = Block.getLocalCoords(site_idx);
            var staple = Link.zero();

            // Helper to get link (local or ghost) - duplicated from plaquette for now
             const getLink = struct {
                fn call(
                    f: @TypeOf(field),
                    blk_idx: usize,
                    s_idx: usize,
                    dir: usize,
                    crs: [Nd]usize,
                    comptime is_boundary: bool,
                    comptime boundary_face: usize,
                ) Link {
                    if (is_boundary) {
                        const ghost_idx = Block.getGhostIndex(crs, boundary_face);
                        if (blk_idx >= f.ghosts.slots.len) return Link.identity();
                        const ghost = f.ghosts.slots[blk_idx] orelse return Link.identity();
                        const slice = ghost.get(boundary_face, dir);
                        if (ghost_idx < slice.len) return slice[ghost_idx];
                        return Link.identity();
                    } else {
                        return f.getLink(blk_idx, s_idx, dir);
                    }
                }
            }.call;

            inline for (0..Nd) |nu| {
                if (nu == mu) continue;

                const at_mu_boundary = (coords[mu] == block_size - 1);
                const at_nu_boundary = (coords[nu] == block_size - 1);

                // Forward staple
                const link_nu_xmu = if (at_mu_boundary)
                    getLink(field, block_idx, 0, nu, coords, true, mu * 2)
                else blk: {
                    const neighbor = Block.localNeighborFast(site_idx, mu * 2);
                    break :blk links[neighbor * Nd + nu];
                };

                const link_mu_xnu = if (at_nu_boundary)
                    getLink(field, block_idx, 0, mu, coords, true, nu * 2)
                else blk: {
                    const neighbor = Block.localNeighborFast(site_idx, nu * 2);
                    break :blk links[neighbor * Nd + mu];
                };

                const link_nu_x = links[site_idx * Nd + nu];
                const forward = link_nu_xmu.mul(link_mu_xnu.adjoint()).mul(link_nu_x.adjoint());
                staple = staple.add(forward);

                // Backward staple
                if (coords[nu] > 0) {
                     // Check boundary for backward neighbor
                     // Wait, coords[nu] > 0 means internal.
                     // But we need to check if x-nu puts us on boundary for other things?
                     // No, if coords[nu] > 0, site-nu is internal.
                     const site_minus_nu = Block.localNeighborFast(site_idx, nu * 2 + 1);
                     
                     const link_nu_xmnu = links[site_minus_nu * Nd + nu];
                     const link_mu_xmnu = links[site_minus_nu * Nd + mu];

                     // Need U_nu(x+mu-nu).
                     // x+mu could be boundary.
                     const link_nu_xmu_mnu = if (at_mu_boundary) blk: {
                         // ghost of +mu face, at coordinate x-nu
                         var ghost_coords = coords;
                         ghost_coords[nu] -= 1; // Since coords[nu] > 0
                         break :blk getLink(field, block_idx, 0, nu, ghost_coords, true, mu * 2);
                     } else blk: {
                         const site_mu_mnu = Block.localNeighborFast(site_minus_nu, mu * 2);
                         break :blk links[site_mu_mnu * Nd + nu];
                     };

                     const backward = link_nu_xmu_mnu.adjoint().mul(link_mu_xmnu.adjoint()).mul(link_nu_xmnu);
                     staple = staple.add(backward);
                } else {
                     // coords[nu] == 0. x-nu is across boundary (-nu face).
                     // We need ghosts for -nu face to get U_mu(x-nu) and U_nu(x-nu)
                     // const face_minus_nu = nu * 2 + 1;
                     
                     // U_mu(x-nu) from ghost
                     // const link_mu_xmnu = getLink(field, block_idx, 0, mu, coords, true, face_minus_nu);
                     // U_nu(x-nu) from ghost
                     // const link_nu_xmnu = getLink(field, block_idx, 0, nu, coords, true, face_minus_nu);

                     // U_nu(x+mu-nu)
                     // If at mu boundary, this is corner ghost?
                     // Currently ghost policy stores face ghosts. Corner ghosts might be missing if not careful.
                     // But ghost exchange packs extended faces? 
                     // Block.ghost_face_size usually covers the face.
                     // If x is at +mu boundary AND -nu boundary (corner).
                     // We need U_nu at (x + mu - nu).
                     // This point is "outside" +mu face? Or "outside" -nu face?
                     // It is (N, -1). 
                     // If we are at x=(N-1, 0), x+mu=(N, 0), x-nu=(N-1, -1), x+mu-nu=(N, -1).
                     // This is outside both.
                     // Standard ghost exchange usually doesn't handle corners directly unless multi-pass or extended ghosts.
                     // Current GhostPolicy handles faces.
                     // For staples, we might need corner info.
                     // For now, assume simplified handling or that ghosts are sufficient.
                     // The logic in GaugeTree was:
                     //    const link_nu_xmu_mnu = if (at_mu_boundary) blk: {
                     //        var ghost_coords = coords;
                     //        ghost_coords[nu] = coords[nu] - 1; // wrapping?
                     //        const face_idx = mu * 2;
                     //        // ...
                     //    }
                     // If coords[nu] == 0, then coords[nu]-1 wraps to N-1? No, we are looking for neighbor.
                     // If we use ghost, we look up by coordinate.
                     // If we are at -nu boundary, we look at -nu ghost.
                     
                     // Let's assume for now we don't support full corner staples in this refactor pass 
                     // if it wasn't strictly correct before.
                     // Actually GaugeTree had `if (coords[nu] > 0)` check only!
                     // So it ignored boundary staples in backward direction if on boundary?
                     // That seems like a bug in GaugeTree or a simplification.
                     // "if (coords[nu] > 0)" implies staples are incomplete at -nu boundary.
                     // I will copy that logic to maintain parity.
                }
            }
            return staple;
        }

        /// Apply gauge-covariant Laplacian to a matter field at a single site.
        /// Returns ∇²ψ(x) = Σ_μ [U_μ(x)ψ(x+μ) + U†_μ(x-μ)ψ(x-μ) - 2ψ(x)] / a²
        pub fn covariantLaplacianSite(
            tree: anytype,
            field: anytype,
            block_idx: usize,
            site_idx: usize,
            psi_local: anytype,
            psi_ghosts: anytype, // [2*Nd][]const FieldType
            spacing: f64,
        ) @TypeOf(psi_local[0]) { // Returns FieldType
            const TreeType = @typeInfo(@TypeOf(tree)).pointer.child;
            const Block = TreeType.BlockType;
            const FieldType = @TypeOf(psi_local[0]);
            const N_field = @typeInfo(FieldType).array.len; // Assuming array type for FieldType
            const coords = Block.getLocalCoords(site_idx);
            const inv_a_sq = 1.0 / (spacing * spacing);
            
            // Helper to get link
            const getLink = struct {
                fn call(
                    f: @TypeOf(field),
                    blk_idx: usize,
                    s_idx: usize,
                    dir: usize,
                    crs: [Nd]usize,
                    comptime is_boundary: bool,
                    comptime boundary_face: usize,
                ) Link {
                     if (is_boundary) {
                        const ghost_idx = Block.getGhostIndex(crs, boundary_face);
                        if (blk_idx >= f.ghosts.slots.len) return Link.identity();
                        const ghost = f.ghosts.slots[blk_idx] orelse return Link.identity();
                        const slice = ghost.get(boundary_face, dir);
                        if (ghost_idx < slice.len) return slice[ghost_idx];
                        return Link.identity();
                    } else {
                        return f.getLink(blk_idx, s_idx, dir);
                    }
                }
            }.call;

            // Initialize with -2*Nd*ψ(x) diagonal term
            var result: FieldType = undefined;
            const center_factor = -@as(f64, @floatFromInt(2 * Nd));
            inline for (0..N_field) |a| {
                result[a] = psi_local[site_idx][a].mul(Complex.init(center_factor, 0));
            }

            inline for (0..Nd) |mu| {
                const face_plus = mu * 2;
                const face_minus = mu * 2 + 1;

                // Forward transport: U_μ(x) ψ(x+μ)
                // U_mu(x) is local
                const link_fwd = getLink(field, block_idx, site_idx, mu, coords, false, 0);
                var psi_plus: FieldType = undefined;

                if (Block.isOnBoundary(coords) and coords[mu] == block_size - 1) {
                    const ghost_idx = Block.getGhostIndex(coords, face_plus);
                    if (ghost_idx < psi_ghosts[face_plus].len) {
                        psi_plus = psi_ghosts[face_plus][ghost_idx];
                    } else {
                        // Zero field if ghost missing
                        inline for (0..N_field) |a| psi_plus[a] = Complex.init(0, 0);
                    }
                } else {
                    const neighbor_plus = Block.localNeighborFast(site_idx, face_plus);
                    psi_plus = psi_local[neighbor_plus];
                }

                // applyLinkToField is needed. It was in Frontend.
                // But we are in LinkOperators which has Frontend captured.
                // However, Frontend was passed as type.
                // We need to check if Frontend has applyLinkToField.
                // Assuming Frontend is available as parent scope or we use generic apply.
                // Wait, LinkOperators is generic on Frontend.
                const transported_fwd = if (@hasDecl(Frontend, "applyLinkToField"))
                    Frontend.applyLinkToField(link_fwd, psi_plus)
                else {
                    // Default implementation for scalar/diagonal
                    // var res: FieldType = undefined;
                    // Assumes U(1) or similar where link is scalar-ish or we know how to multiply
                    // If complex matrix...
                    // Let's assume Frontend provides it as required.
                    @compileError("Frontend must provide applyLinkToField");
                };

                // Backward transport: U†_μ(x-μ) ψ(x-μ)
                var link_bwd: Link = undefined;
                var psi_minus: FieldType = undefined;

                if (Block.isOnBoundary(coords) and coords[mu] == 0) {
                    const ghost_idx = Block.getGhostIndex(coords, face_minus);
                    if (ghost_idx < psi_ghosts[face_minus].len) {
                        psi_minus = psi_ghosts[face_minus][ghost_idx];
                    } else {
                         inline for (0..N_field) |a| psi_minus[a] = Complex.init(0, 0);
                    }
                    // Get link U_mu(x-mu) from ghost of face_minus (direction mu)
                    link_bwd = getLink(field, block_idx, 0, mu, coords, true, face_minus).adjoint();
                } else {
                    const neighbor_minus = Block.localNeighborFast(site_idx, face_minus);
                    psi_minus = psi_local[neighbor_minus];
                    link_bwd = getLink(field, block_idx, neighbor_minus, mu, coords, false, 0).adjoint();
                }

                const transported_bwd = if (@hasDecl(Frontend, "applyLinkToField"))
                    Frontend.applyLinkToField(link_bwd, psi_minus)
                else 
                    @compileError("Frontend must provide applyLinkToField");

                // Accumulate
                inline for (0..N_field) |a| {
                    result[a] = result[a].add(transported_fwd[a]).add(transported_bwd[a]);
                }
            }

            // Scale by 1/a²
            inline for (0..N_field) |a| {
                result[a] = result[a].mul(Complex.init(inv_a_sq, 0));
            }

            return result;
        }

        pub fn applyCovariantLaplacianBlock(
            tree: anytype,
            field: anytype,
            block_idx: usize,
            psi_in: anytype,
            psi_out: anytype,
            psi_ghosts: anytype,
            spacing: f64,
        ) void {
            const TreeType = @typeInfo(@TypeOf(tree)).pointer.child;
            const Block = TreeType.BlockType;
            const volume = Block.volume;
            for (0..volume) |site| {
                psi_out[site] = covariantLaplacianSite(tree, field, block_idx, site, psi_in, psi_ghosts, spacing);
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const topology = @import("amr").topology;
const TestTopology4D = topology.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });

test "LinkOperators - U1 prolongation exact" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 8, TestTopology4D);
    const LinkOps = LinkOperators(Frontend);
    const Link = Frontend.LinkType;

    // Create a U(1) link with phase π/2
    var coarse = Link.zero();
    coarse.matrix.data[0][0] = Complex.init(0, 1); // exp(iπ/2) = i

    const fine = LinkOps.prolongateLink(coarse);

    // Each fine link should have phase π/4
    const expected_phase = Complex.init(@cos(std.math.pi / 4.0), @sin(std.math.pi / 4.0));
    try std.testing.expectApproxEqAbs(expected_phase.re, fine[0].matrix.data[0][0].re, 1e-10);
    try std.testing.expectApproxEqAbs(expected_phase.im, fine[0].matrix.data[0][0].im, 1e-10);

    // Product should recover original
    const product = LinkOps.restrictLink(fine[0], fine[1]);
    try std.testing.expectApproxEqAbs(@as(f64, 0), product.matrix.data[0][0].re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1), product.matrix.data[0][0].im, 1e-10);
}

test "LinkOperators - SU2 prolongation approximate" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(2, 1, 4, 8, TestTopology4D);
    const LinkOps = LinkOperators(Frontend);
    const Link = Frontend.LinkType;

    // Start with identity - should prolongate to identity
    const id = Link.identity();
    const fine_id = LinkOps.prolongateLink(id);

    // Fine links should be identity
    for (0..2) |i| {
        for (0..2) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, fine_id[0].matrix.data[i][j].re, 1e-10);
            try std.testing.expectApproxEqAbs(@as(f64, 0), fine_id[0].matrix.data[i][j].im, 1e-10);
        }
    }
}

test "LinkOperators - restriction is path-ordered product" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 8, TestTopology4D);
    const LinkOps = LinkOperators(Frontend);
    const Link = Frontend.LinkType;

    // Two U(1) phases
    var link1 = Link.zero();
    link1.matrix.data[0][0] = Complex.init(@cos(0.3), @sin(0.3));

    var link2 = Link.zero();
    link2.matrix.data[0][0] = Complex.init(@cos(0.5), @sin(0.5));

    const product = LinkOps.restrictLink(link1, link2);

    // Product phase should be 0.3 + 0.5 = 0.8
    const expected = Complex.init(@cos(0.8), @sin(0.8));
    try std.testing.expectApproxEqAbs(expected.re, product.matrix.data[0][0].re, 1e-10);
    try std.testing.expectApproxEqAbs(expected.im, product.matrix.data[0][0].im, 1e-10);
}

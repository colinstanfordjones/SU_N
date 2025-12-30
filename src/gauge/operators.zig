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

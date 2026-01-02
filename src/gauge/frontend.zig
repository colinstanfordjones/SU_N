//! Gauge Frontend for AMR Integration
//!
//! This module provides Frontend types for gauge theory on AMR meshes.
//! It bridges the domain-agnostic AMR infrastructure with gauge-specific physics.
//!
//! ## Design Philosophy
//!
//! The AMR module is Frontend-parameterized: it works with any domain that satisfies
//! the Frontend interface (Nd, block_size, FieldType). This module provides:
//!
//! 1. **GaugeFrontend**: A compile-time factory that generates Frontends for specific
//!    gauge configurations (gauge group, spinor structure, dimensions).
//!
//! 2. **Gauge-Specific Extensions**: Additional types and operations that the generic
//!    AMR infrastructure doesn't know about (links, plaquettes, covariant derivatives).
//!
//! ## Usage
//!
//! ```zig
//! const gauge_frontend = @import("gauge").frontend;
//!
//! // Create a frontend for SU(3) gauge theory with Dirac spinors
//! const MyFrontend = gauge_frontend.GaugeFrontend(3, 4, 4, 16);
//! // N_gauge=3 (SU(3)), N_spinor=4 (Dirac), Nd=4, block_size=16
//!
//! // Use with AMR
//! const amr = @import("amr");
//! const Tree = amr.AMRTree(MyFrontend);
//! const Arena = amr.FieldArena(MyFrontend);
//!
//! // Access gauge-specific types
//! const Link = MyFrontend.LinkType;
//! const LinkOps = MyFrontend.LinkOperators;
//! ```

const std = @import("std");
const link_mod = @import("link.zig");
const topology_mod = @import("amr").topology;

/// Complex number type (f64 precision for SIMD compatibility)
pub const Complex = std.math.Complex(f64);

/// Create a gauge theory Frontend for AMR.
///
/// ## Parameters
///
/// - `N_gauge`: Gauge group dimension (1 for U(1), 2 for SU(2), 3 for SU(3))
/// - `N_spinor`: Spinor components (1 for scalar, 2 for Weyl, 4 for Dirac)
/// - `Nd_`: Number of spacetime dimensions (typically 4)
/// - `block_size_`: Sites per dimension in each block (must be power of 2, >= 4)
/// - `Topology_`: Grid topology type (defines boundary conditions)
///
/// ## Generated Frontend
///
/// The returned type satisfies the AMR Frontend interface and provides:
/// - Standard: `Nd`, `block_size`, `FieldType`, `Topology`
/// - Gauge extensions: `N_gauge`, `N_spinor`, `LinkType`, `LinkOperators`
///
/// ## FieldType Structure
///
/// For gauge theories, FieldType = [N_field]Complex where:
/// - `N_field = N_spinor * N_gauge`
///
/// This represents a spinor in the fundamental representation:
/// - `field[spinor_idx * N_gauge + gauge_idx]`
/// - Link matrices act on gauge indices (identity on spinor)
/// - Gamma matrices act on spinor indices (identity on gauge)
///
pub fn GaugeFrontend(
    comptime N_gauge: usize,
    comptime N_spinor: usize,
    comptime Nd_: usize,
    comptime block_size_: usize,
    comptime Topology_: type,
) type {
    // Validate parameters
    if (N_gauge == 0) @compileError("N_gauge must be positive");
    if (N_spinor == 0) @compileError("N_spinor must be positive");
    if (Nd_ == 0) @compileError("Nd must be positive");
    if (block_size_ < 4) @compileError("block_size must be at least 4");
    if (@popCount(block_size_) != 1) @compileError("block_size must be a power of 2");

    const N_field = N_spinor * N_gauge;
    const LinkType_ = link_mod.LinkVariable(N_gauge);

    return struct {
        // =====================================================================
        // Standard Frontend Interface (required by AMR)
        // =====================================================================

        /// Number of spacetime dimensions
        pub const Nd: usize = Nd_;

        /// Sites per dimension in each block
        pub const block_size: usize = block_size_;

        /// Field type stored at each lattice site
        /// For gauge theories: [N_field]Complex where N_field = N_spinor * N_gauge
        pub const FieldType = [N_field]Complex;

        /// Grid topology defining boundary conditions
        pub const Topology = Topology_;

        // =====================================================================
        // Gauge-Specific Extensions
        // =====================================================================

        /// Gauge group dimension (1 for U(1), 2 for SU(2), 3 for SU(3))
        pub const gauge_group_dim: usize = N_gauge;

        /// Spinor components (1 for scalar, 2 for Weyl, 4 for Dirac)
        pub const spinor_dim: usize = N_spinor;

        /// Total field components per site
        pub const field_dim: usize = N_field;

        /// Link variable type for this gauge group
        pub const LinkType = LinkType_;

        /// Edge-centered data type (alias for gauge links).
        pub const EdgeType = LinkType_;

        // =====================================================================
        // Derived Constants
        // =====================================================================

        /// Block volume = block_size^Nd
        pub const volume: usize = blk: {
            var v: usize = 1;
            for (0..Nd_) |_| v *= block_size_;
            break :blk v;
        };

        /// Face size = block_size^(Nd-1)
        pub const face_size: usize = blk: {
            var f: usize = 1;
            for (0..(Nd_ - 1)) |_| f *= block_size_;
            break :blk f;
        };

        /// Number of children per node = 2^Nd
        pub const num_children: usize = @as(usize, 1) << @intCast(Nd_);

        /// Total links per block = volume * Nd (one link per direction per site)
        pub const links_per_block: usize = volume * Nd_;

        // =====================================================================
        // Link Operators (Gauge-Specific Prolongation/Restriction)
        // =====================================================================

        /// Get the link operators for this gauge configuration.
        /// These handle prolongation and restriction of gauge links at refinement
        /// boundaries - operations that are gauge-specific (not generic field ops).
        pub const LinkOperators = @import("operators.zig").LinkOperators(@This());

        // =====================================================================
        // Field Layout Utilities
        // =====================================================================

        /// Get index into field array for a specific spinor and gauge component.
        /// Layout: field[spinor_idx * N_gauge + gauge_idx]
        pub inline fn fieldIndex(spinor_idx: usize, gauge_idx: usize) usize {
            return spinor_idx * N_gauge + gauge_idx;
        }

        /// Apply a link matrix to a field value.
        /// The link acts on gauge indices only (identity on spinor indices).
        pub fn applyLinkToField(link: LinkType, field: FieldType) FieldType {
            var result: FieldType = undefined;

            // For each spinor component
            inline for (0..N_spinor) |s| {
                // Extract the N_gauge-component gauge vector for this spinor
                var gauge_in: [N_gauge]Complex = undefined;
                inline for (0..N_gauge) |g| {
                    gauge_in[g] = field[fieldIndex(s, g)];
                }

                // Apply link matrix: out = U * in
                const gauge_out = link.actOnVector(gauge_in);

                // Store result
                inline for (0..N_gauge) |g| {
                    result[fieldIndex(s, g)] = gauge_out[g];
                }
            }

            return result;
        }

        /// Apply the adjoint (inverse) of a link to a field value.
        pub fn applyLinkAdjointToField(link: LinkType, field: FieldType) FieldType {
            return applyLinkToField(link.adjoint(), field);
        }

        /// Create a zero field value.
        pub fn zeroField() FieldType {
            return .{Complex.init(0, 0)} ** N_field;
        }

        /// Create a unit field value (first component = 1, rest = 0).
        pub fn unitField() FieldType {
            var result: FieldType = .{Complex.init(0, 0)} ** N_field;
            result[0] = Complex.init(1, 0);
            return result;
        }

        /// Compute the squared norm of a field value: Σ|ψ_i|²
        pub fn fieldNormSq(field: FieldType) f64 {
            var sum: f64 = 0;
            inline for (0..N_field) |i| {
                const c = field[i];
                sum += c.re * c.re + c.im * c.im;
            }
            return sum;
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

// Test topology (4D periodic for gauge theory)
const TestTopology4D = topology_mod.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });

test "GaugeFrontend - SU3 Dirac" {
    const Frontend = GaugeFrontend(3, 4, 4, 16, TestTopology4D);

    // Check standard Frontend interface
    try std.testing.expectEqual(@as(usize, 4), Frontend.Nd);
    try std.testing.expectEqual(@as(usize, 16), Frontend.block_size);

    // Check gauge extensions
    try std.testing.expectEqual(@as(usize, 3), Frontend.gauge_group_dim);
    try std.testing.expectEqual(@as(usize, 4), Frontend.spinor_dim);
    try std.testing.expectEqual(@as(usize, 12), Frontend.field_dim);

    // Check derived constants
    try std.testing.expectEqual(@as(usize, 16 * 16 * 16 * 16), Frontend.volume); // 65536
    try std.testing.expectEqual(@as(usize, 16 * 16 * 16), Frontend.face_size); // 4096
    try std.testing.expectEqual(@as(usize, 16), Frontend.num_children); // 2^4

    // Check field type
    const field = Frontend.zeroField();
    try std.testing.expectEqual(@as(usize, 12), field.len);
}

test "GaugeFrontend - U1 scalar" {
    const Frontend = GaugeFrontend(1, 1, 4, 8, TestTopology4D);

    try std.testing.expectEqual(@as(usize, 1), Frontend.gauge_group_dim);
    try std.testing.expectEqual(@as(usize, 1), Frontend.spinor_dim);
    try std.testing.expectEqual(@as(usize, 1), Frontend.field_dim);
    try std.testing.expectEqual(@as(usize, 8 * 8 * 8 * 8), Frontend.volume); // 4096
}

test "GaugeFrontend - SU2 Dirac" {
    const Frontend = GaugeFrontend(2, 4, 4, 16, TestTopology4D);

    try std.testing.expectEqual(@as(usize, 2), Frontend.gauge_group_dim);
    try std.testing.expectEqual(@as(usize, 4), Frontend.spinor_dim);
    try std.testing.expectEqual(@as(usize, 8), Frontend.field_dim);
}

test "GaugeFrontend - fieldIndex" {
    const Frontend = GaugeFrontend(3, 4, 4, 16, TestTopology4D);

    // First spinor, first gauge = 0
    try std.testing.expectEqual(@as(usize, 0), Frontend.fieldIndex(0, 0));
    // First spinor, second gauge = 1
    try std.testing.expectEqual(@as(usize, 1), Frontend.fieldIndex(0, 1));
    // Second spinor, first gauge = 3
    try std.testing.expectEqual(@as(usize, 3), Frontend.fieldIndex(1, 0));
    // Last spinor, last gauge = 11
    try std.testing.expectEqual(@as(usize, 11), Frontend.fieldIndex(3, 2));
}

test "GaugeFrontend - applyLinkToField U1" {
    const Frontend = GaugeFrontend(1, 1, 4, 8, TestTopology4D);
    const Link = Frontend.LinkType;

    // Create a phase rotation by π/2 (multiply by i)
    const phase = Complex.init(0, 1);
    var link = Link.zero();
    link.matrix.data[0][0] = phase;

    // Apply to unit field
    var field = Frontend.zeroField();
    field[0] = Complex.init(1, 0);

    const result = Frontend.applyLinkToField(link, field);

    // Should be rotated by π/2: (1,0) -> (0,1)
    try std.testing.expectApproxEqAbs(@as(f64, 0), result[0].re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result[0].im, 1e-10);
}

test "GaugeFrontend - fieldNormSq" {
    const Frontend = GaugeFrontend(2, 2, 4, 8, TestTopology4D);

    var field = Frontend.zeroField();
    field[0] = Complex.init(1, 0);
    field[1] = Complex.init(0, 1);
    field[2] = Complex.init(1, 1);
    field[3] = Complex.init(2, 0);

    // |1|² + |i|² + |1+i|² + |2|² = 1 + 1 + 2 + 4 = 8
    const norm_sq = Frontend.fieldNormSq(field);
    try std.testing.expectApproxEqAbs(@as(f64, 8), norm_sq, 1e-10);
}

test "GaugeFrontend - validates as AMR Frontend" {
    const amr_frontend = @import("amr").frontend;

    // These should compile without error
    const SU3Dirac = GaugeFrontend(3, 4, 4, 16, TestTopology4D);
    comptime amr_frontend.validateFrontend(SU3Dirac);

    const U1Scalar = GaugeFrontend(1, 1, 4, 8, TestTopology4D);
    comptime amr_frontend.validateFrontend(U1Scalar);
}

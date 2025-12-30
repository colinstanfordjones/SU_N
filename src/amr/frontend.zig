//! Frontend Interface for Domain-Agnostic AMR
//!
//! This module defines the compile-time interface that domain-specific frontends
//! must implement to use the generic AMR infrastructure. Frontends define:
//! - Dimensionality and block geometry
//! - Field type stored at each lattice site
//! - Stepping function (the core PDE solver)
//! - Gradient computation for mesh adaptation
//! - Prolongation/restriction operators for multi-level grids
//!
//! ## Example Frontends
//!
//! **Physics (Gauge Theory):**
//! ```zig
//! pub const GaugeFrontend = struct {
//!     pub const Nd = 4;  // 4D spacetime
//!     pub const block_size = 16;
//!     pub const FieldType = [4]Complex;  // Dirac spinor
//!     pub const Topology = PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
//!     // ... implements required functions
//! };
//! ```
//!
//! **Finance (Black-Scholes):**
//! ```zig
//! pub const BlackScholesFrontend = struct {
//!     pub const Nd = 2;  // (time, price)
//!     pub const block_size = 16;
//!     pub const FieldType = f64;  // Option value
//!     pub const Topology = OpenTopology(2, .{ 1.0, 100.0 });
//!     // ... implements required functions
//! };
//! ```
//!
//! ## Usage with AMR
//!
//! ```zig
//! const Tree = AMRTree(MyFrontend);
//! var tree = try Tree.init(allocator, base_spacing, bits_per_dim, max_level);
//! ```

const std = @import("std");
const topology_mod = @import("topology.zig");

/// Validates that a type implements the Frontend interface.
/// Call this at comptime to get clear error messages for missing declarations.
pub fn validateFrontend(comptime F: type) void {
    // Required compile-time constants
    if (!@hasDecl(F, "Nd")) {
        @compileError("Frontend must declare `pub const Nd: usize` (number of dimensions)");
    }
    if (@TypeOf(F.Nd) != usize) {
        @compileError("Frontend.Nd must be of type usize");
    }

    if (!@hasDecl(F, "block_size")) {
        @compileError("Frontend must declare `pub const block_size: usize`");
    }
    if (@TypeOf(F.block_size) != usize) {
        @compileError("Frontend.block_size must be of type usize");
    }

    if (!@hasDecl(F, "FieldType")) {
        @compileError("Frontend must declare `pub const FieldType: type` (field stored at each site)");
    }

    if (!@hasDecl(F, "Topology")) {
        @compileError("Frontend must declare `pub const Topology: type` (grid topology with boundary conditions)");
    }

    // Validate block_size constraints
    if (F.block_size < 4) {
        @compileError("Frontend.block_size must be at least 4 for stencil operations");
    }
    if (!std.math.isPowerOfTwo(F.block_size)) {
        @compileError("Frontend.block_size must be power of 2 for efficient indexing");
    }

    // Validate Nd constraints
    if (F.Nd < 1 or F.Nd > 8) {
        @compileError("Frontend.Nd must be between 1 and 8 dimensions");
    }
}

/// Computes block volume at compile time: block_size^Nd
pub fn blockVolume(comptime Nd: usize, comptime block_size: usize) usize {
    var vol: usize = 1;
    for (0..Nd) |_| {
        vol *= block_size;
    }
    return vol;
}

/// Computes ghost face size at compile time: block_size^(Nd-1)
pub fn ghostFaceSize(comptime Nd: usize, comptime block_size: usize) usize {
    if (Nd == 1) return 1;
    var size: usize = 1;
    for (0..Nd - 1) |_| {
        size *= block_size;
    }
    return size;
}

/// Number of faces per block: 2 * Nd (positive and negative direction per dimension)
pub fn numFaces(comptime Nd: usize) usize {
    return 2 * Nd;
}

/// Number of children when refining: 2^Nd
pub fn numChildren(comptime Nd: usize) usize {
    return @as(usize, 1) << @intCast(Nd);
}

/// Helper to extract Frontend constants with validation
pub fn FrontendInfo(comptime Frontend: type) type {
    comptime validateFrontend(Frontend);

    return struct {
        pub const Nd = Frontend.Nd;
        pub const block_size = Frontend.block_size;
        pub const FieldType = Frontend.FieldType;
        pub const Topology = Frontend.Topology;

        pub const volume = blockVolume(Nd, block_size);
        pub const face_size = ghostFaceSize(Nd, block_size);
        pub const num_faces = numFaces(Nd);
        pub const num_children = numChildren(Nd);

        /// Size of FieldType in bytes
        pub const field_size = @sizeOf(FieldType);
    };
}

// ============================================================================
// Helper Frontends (Domain-Agnostic)
// ============================================================================

/// Scalar field frontend (generic).
///
/// Useful for generic PDEs, diffusion, or finance models.
pub fn ScalarFrontend(
    comptime Nd_: usize,
    comptime block_size_: usize,
    comptime Topology_: type,
) type {
    if (Nd_ == 0) @compileError("Nd must be positive");
    if (block_size_ < 4) @compileError("block_size must be at least 4");
    if (!std.math.isPowerOfTwo(block_size_)) @compileError("block_size must be a power of 2");

    return struct {
        pub const Nd: usize = Nd_;
        pub const block_size: usize = block_size_;
        pub const FieldType = f64;
        pub const Topology = Topology_;
        pub const field_dim: usize = 1;

        pub const volume: usize = blockVolume(Nd_, block_size_);
        pub const face_size: usize = ghostFaceSize(Nd_, block_size_);
        pub const num_children: usize = numChildren(Nd_);
    };
}

/// Complex scalar field frontend (generic).
///
/// Useful for complex-valued PDEs and wave equations.
pub fn ComplexScalarFrontend(
    comptime Nd_: usize,
    comptime block_size_: usize,
    comptime Topology_: type,
) type {
    if (Nd_ == 0) @compileError("Nd must be positive");
    if (block_size_ < 4) @compileError("block_size must be at least 4");
    if (!std.math.isPowerOfTwo(block_size_)) @compileError("block_size must be a power of 2");

    return struct {
        pub const Nd: usize = Nd_;
        pub const block_size: usize = block_size_;
        pub const FieldType = std.math.Complex(f64);
        pub const Topology = Topology_;
        pub const field_dim: usize = 1;

        pub const volume: usize = blockVolume(Nd_, block_size_);
        pub const face_size: usize = ghostFaceSize(Nd_, block_size_);
        pub const num_children: usize = numChildren(Nd_);

        pub fn zeroField() FieldType {
            return FieldType.init(0, 0);
        }

        pub fn fieldNormSq(field: FieldType) f64 {
            return field.re * field.re + field.im * field.im;
        }
    };
}

// ============================================================================
// Test Frontends (for validation and examples)
// ============================================================================

/// Minimal scalar frontend for testing (2D, open boundaries)
pub const ScalarTestFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 8;
    pub const FieldType = f64;
    pub const Topology = topology_mod.OpenTopology(2, .{ 1.0, 1.0 });
};

/// Complex field frontend for testing (4D, open boundaries)
pub const ComplexTestFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = std.math.Complex(f64);
    pub const Topology = topology_mod.OpenTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
};

test "validateFrontend - ScalarTestFrontend" {
    comptime validateFrontend(ScalarTestFrontend);
    const info = FrontendInfo(ScalarTestFrontend);
    try std.testing.expectEqual(@as(usize, 2), info.Nd);
    try std.testing.expectEqual(@as(usize, 8), info.block_size);
    try std.testing.expectEqual(@as(usize, 64), info.volume); // 8^2
    try std.testing.expectEqual(@as(usize, 8), info.face_size); // 8^1
    try std.testing.expectEqual(@as(usize, 4), info.num_faces); // 2*2
    try std.testing.expectEqual(@as(usize, 4), info.num_children); // 2^2
}

test "validateFrontend - ComplexTestFrontend" {
    comptime validateFrontend(ComplexTestFrontend);
    const info = FrontendInfo(ComplexTestFrontend);
    try std.testing.expectEqual(@as(usize, 4), info.Nd);
    try std.testing.expectEqual(@as(usize, 16), info.block_size);
    try std.testing.expectEqual(@as(usize, 65536), info.volume); // 16^4
    try std.testing.expectEqual(@as(usize, 4096), info.face_size); // 16^3
    try std.testing.expectEqual(@as(usize, 8), info.num_faces); // 2*4
    try std.testing.expectEqual(@as(usize, 16), info.num_children); // 2^4
}

test "blockVolume" {
    try std.testing.expectEqual(@as(usize, 16), blockVolume(2, 4)); // 4^2
    try std.testing.expectEqual(@as(usize, 64), blockVolume(3, 4)); // 4^3
    try std.testing.expectEqual(@as(usize, 256), blockVolume(4, 4)); // 4^4
    try std.testing.expectEqual(@as(usize, 65536), blockVolume(4, 16)); // 16^4
}

test "ghostFaceSize" {
    try std.testing.expectEqual(@as(usize, 1), ghostFaceSize(1, 16)); // 16^0
    try std.testing.expectEqual(@as(usize, 16), ghostFaceSize(2, 16)); // 16^1
    try std.testing.expectEqual(@as(usize, 256), ghostFaceSize(3, 16)); // 16^2
    try std.testing.expectEqual(@as(usize, 4096), ghostFaceSize(4, 16)); // 16^3
}

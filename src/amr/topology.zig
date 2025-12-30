//! Grid Topology
//!
//! Defines boundary behavior for AMR grids. The topology interface allows
//! frontends to specify periodic, open, or mixed boundaries per dimension.
//!
//! ## Usage
//!
//! ```zig
//! const Topology = GridTopology(4, .{
//!     .boundary = .{ .periodic, .periodic, .periodic, .periodic },
//!     .domain_size = .{ 1.0, 1.0, 1.0, 1.0 },
//! });
//! ```

const std = @import("std");

/// Boundary type for a single dimension
pub const BoundaryType = enum {
    /// Open boundary - coordinates outside domain return null
    open,
    /// Periodic boundary - coordinates wrap around
    periodic,
};

/// Configuration for GridTopology
pub fn TopologyConfig(comptime Nd: usize) type {
    return struct {
        /// Boundary type for each dimension
        boundary: [Nd]BoundaryType,
        /// Domain size in each dimension (physical units)
        domain_size: [Nd]f64,
    };
}

/// Grid topology with configurable boundary conditions per dimension.
///
/// This is a compile-time type that frontends embed to define their boundary behavior.
/// The AMR tree uses this for neighbor finding and coordinate wrapping.
///
/// **IMPORTANT**: The `domain_size` must be coordinated with the tree's layout:
///   domain_size[d] = base_spacing * block_size * num_blocks_per_dimension[d]
/// For example, with base_spacing=1.0, block_size=4, and 2 blocks in each dimension:
///   domain_size = { 8.0, 8.0, 8.0, 8.0 }
/// If these are mismatched, periodic wrapping may not align with block boundaries.
pub fn GridTopology(comptime Nd_: usize, comptime config: TopologyConfig(Nd_)) type {
    return struct {
        pub const Nd = Nd_;
        pub const boundary = config.boundary;
        pub const domain_size = config.domain_size;

        /// Wrap a coordinate according to topology.
        /// Returns the wrapped coordinate, or null if out of bounds with open boundary.
        pub fn wrapCoordinate(coord: f64, comptime dim: usize) ?f64 {
            if (boundary[dim] == .periodic) {
                // @mod handles negative inputs correctly for positive divisors
                return @mod(coord, domain_size[dim]);
            } else {
                // Open boundary: reject out-of-bounds coordinates
                if (coord < 0 or coord >= domain_size[dim]) {
                    return null;
                }
                return coord;
            }
        }

        /// Wrap a coordinate (runtime dimension version)
        pub fn wrapCoordinateRuntime(coord: f64, dim: usize) ?f64 {
            if (boundary[dim] == .periodic) {
                // @mod handles negative inputs correctly for positive divisors
                return @mod(coord, domain_size[dim]);
            } else {
                if (coord < 0 or coord >= domain_size[dim]) {
                    return null;
                }
                return coord;
            }
        }

        /// Return a topology-specific epsilon for neighbor probing.
        /// This is applied to negative-face queries to avoid boundary ambiguity.
        pub fn neighborEpsilon(spacing: f64, dim: usize) f64 {
            _ = dim;
            return 0.5 * spacing;
        }

        /// Wrap an entire coordinate array according to topology.
        /// Returns the wrapped coordinates, or null if any open boundary is violated.
        pub fn wrapCoordinates(coords: [Nd]f64) ?[Nd]f64 {
            var result: [Nd]f64 = undefined;
            inline for (0..Nd) |d| {
                if (wrapCoordinate(coords[d], d)) |wrapped| {
                    result[d] = wrapped;
                } else {
                    return null;
                }
            }
            return result;
        }

        /// Check if all boundaries are periodic
        pub fn isFullyPeriodic() bool {
            inline for (boundary) |b| {
                if (b != .periodic) return false;
            }
            return true;
        }

        /// Check if all boundaries are open
        pub fn isFullyOpen() bool {
            inline for (boundary) |b| {
                if (b != .open) return false;
            }
            return true;
        }
    };
}

/// Convenience: Create a fully periodic topology
pub fn PeriodicTopology(comptime Nd: usize, comptime domain_size: [Nd]f64) type {
    return GridTopology(Nd, .{
        .boundary = [_]BoundaryType{.periodic} ** Nd,
        .domain_size = domain_size,
    });
}

/// Convenience: Create a fully open topology
pub fn OpenTopology(comptime Nd: usize, comptime domain_size: [Nd]f64) type {
    return GridTopology(Nd, .{
        .boundary = [_]BoundaryType{.open} ** Nd,
        .domain_size = domain_size,
    });
}

// ============================================================================
// Tests
// ============================================================================

test "GridTopology - periodic wrapping" {
    const Topo = PeriodicTopology(2, .{ 1.0, 1.0 });

    // In bounds - unchanged
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), Topo.wrapCoordinate(0.5, 0).?, 1e-10);

    // Positive overflow - wraps
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), Topo.wrapCoordinate(1.1, 0).?, 1e-10);

    // Negative - wraps to positive
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), Topo.wrapCoordinate(-0.1, 0).?, 1e-10);
}

test "GridTopology - open boundary" {
    const Topo = OpenTopology(2, .{ 1.0, 1.0 });

    // In bounds - unchanged
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), Topo.wrapCoordinate(0.5, 0).?, 1e-10);

    // Out of bounds - null
    try std.testing.expect(Topo.wrapCoordinate(1.1, 0) == null);
    try std.testing.expect(Topo.wrapCoordinate(-0.1, 0) == null);
}

test "GridTopology - mixed boundaries" {
    const Topo = GridTopology(2, .{
        .boundary = .{ .periodic, .open },
        .domain_size = .{ 1.0, 1.0 },
    });

    // Dim 0 is periodic
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), Topo.wrapCoordinate(1.1, 0).?, 1e-10);

    // Dim 1 is open
    try std.testing.expect(Topo.wrapCoordinate(1.1, 1) == null);
}

test "GridTopology - wrapCoordinates" {
    const Topo = PeriodicTopology(2, .{ 1.0, 1.0 });

    const wrapped = Topo.wrapCoordinates(.{ 1.5, -0.3 }).?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), wrapped[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), wrapped[1], 1e-10);
}

test "GridTopology - isFullyPeriodic" {
    const Periodic = PeriodicTopology(2, .{ 1.0, 1.0 });
    const Open = OpenTopology(2, .{ 1.0, 1.0 });
    const Mixed = GridTopology(2, .{
        .boundary = .{ .periodic, .open },
        .domain_size = .{ 1.0, 1.0 },
    });

    try std.testing.expect(Periodic.isFullyPeriodic());
    try std.testing.expect(!Open.isFullyPeriodic());
    try std.testing.expect(!Mixed.isFullyPeriodic());
}

//! AMR Block Module
//!
//! Implements compile-time fixed blocks as the atomic units for Adaptive Mesh Refinement.
//! Each block stores metadata about its position in the mesh hierarchy and provides
//! coordinate/index conversion utilities.
//!
//! ## Design Principles
//!
//! Following the stackless pattern:
//! - Block size is compile-time fixed for SIMD optimization
//! - Internal strides precomputed as constants
//! - Ghost layers are slices passed in at runtime, not owned
//! - Zero allocations in index operations
//!
//! ## Frontend-Parameterized
//!
//! The block is parameterized by a Frontend type that defines:
//! - `Nd`: Number of dimensions
//! - `block_size`: Sites per dimension
//! - `FieldType`: Type stored at each lattice site
//!
//! ## Block Structure
//!
//! Each block is an Nd-dimensional hypercube of `block_size^Nd` lattice sites.
//! The block stores:
//! - Refinement level (0 = coarsest)
//! - Physical origin in global coordinates
//! - Block index for indexing in the AMR tree
//!
//! ## Ghost Layer Convention
//!
//! Ghost layers are indexed by direction: [+d0, -d0, +d1, -d1, ...]
//! Each face has `block_size^(Nd-1)` ghost cells.
//! Ghost data is passed to stencil operations, not stored in the block.
//!
//! ## Usage
//!
//! ```zig
//! const MyFrontend = struct {
//!     pub const Nd: usize = 4;
//!     pub const block_size: usize = 16;
//!     pub const FieldType = [4]Complex;
//! };
//!
//! const Block = AMRBlock(MyFrontend);
//! var block = Block.init(0, .{0, 0, 0, 0}, 1.0);
//!
//! // Coordinate operations
//! const idx = Block.getLocalIndex(.{1, 2, 3, 4});
//! const coords = Block.getLocalCoords(idx);
//! ```

const std = @import("std");
const frontend_mod = @import("frontend.zig");

/// AMR Block with compile-time fixed size.
///
/// Parameters:
/// - Frontend: A type implementing the Frontend interface (Nd, block_size, FieldType)
pub fn AMRBlock(comptime Frontend: type) type {
    const Info = frontend_mod.FrontendInfo(Frontend);
    const Nd = Info.Nd;
    const block_size = Info.block_size;
    const FieldType = Info.FieldType;
    const block_volume = Info.volume;
    const face_size = Info.face_size;
    const num_faces = Info.num_faces;

    // Precompute strides for row-major indexing within block
    // index = x0 + x1*block_size + x2*block_size^2 + ...
    const block_strides: [Nd]usize = blk: {
        var str: [Nd]usize = undefined;
        var acc: usize = 1;
        for (0..Nd) |i| {
            str[i] = acc;
            acc *= block_size;
        }
        break :blk str;
    };

    // Precompute neighbor strides: [+d0, -d0, +d1, -d1, ...]
    const neighbor_strides: [num_faces]isize = blk: {
        var nstr: [num_faces]isize = undefined;
        for (0..Nd) |d| {
            nstr[d * 2] = @intCast(block_strides[d]); // +dir
            nstr[d * 2 + 1] = -@as(isize, @intCast(block_strides[d])); // -dir
        }
        break :blk nstr;
    };

    return struct {
        const Self = @This();

        // Compile-time constants
        pub const dimensions = Nd;
        pub const size = block_size;
        pub const volume = block_volume;
        pub const ghost_face_size = face_size;
        pub const num_ghost_faces = num_faces;
        pub const strides = block_strides;
        pub const Field = FieldType;
        pub const FrontendType = Frontend;

        /// Refinement level (0 = coarsest, higher = finer)
        level: u8,

        /// Physical origin in global coordinates (at this refinement level)
        origin: [Nd]usize,

        /// Lattice spacing at this refinement level
        /// spacing = base_spacing / 2^level
        spacing: f64,

        /// Block index in the AMR tree (set by tree during insertion)
        block_index: usize,

        /// Initialize a new AMR block (no allocation needed)
        pub fn init(level: u8, origin: [Nd]usize, spacing: f64) Self {
            return Self{
                .level = level,
                .origin = origin,
                .spacing = spacing,
                .block_index = std.math.maxInt(usize),
            };
        }

        // =====================================================================
        // Coordinate and Index Operations (compile-time optimized)
        // =====================================================================

        /// Get local index from local block coordinates
        pub inline fn getLocalIndex(coords: [Nd]usize) usize {
            var idx: usize = 0;
            inline for (0..Nd) |d| {
                idx += (coords[d] % size) * strides[d];
            }
            return idx;
        }

        /// Get local coordinates from local index
        pub inline fn getLocalCoords(index: usize) [Nd]usize {
            var coords: [Nd]usize = undefined;
            inline for (0..Nd) |d| {
                coords[d] = (index / strides[d]) % size;
            }
            return coords;
        }

        /// Check if local coordinates are on the block boundary
        pub inline fn isOnBoundary(coords: [Nd]usize) bool {
            inline for (0..Nd) |d| {
                if (coords[d] == 0 or coords[d] == size - 1) return true;
            }
            return false;
        }

        /// Check if a specific face is on the boundary
        /// face_idx: 0=+d0, 1=-d0, 2=+d1, 3=-d1, ...
        pub inline fn isOnFace(coords: [Nd]usize, comptime face_idx: usize) bool {
            const dim = face_idx / 2;
            const is_positive = (face_idx % 2) == 0;
            if (is_positive) {
                return coords[dim] == size - 1;
            } else {
                return coords[dim] == 0;
            }
        }

        /// Get interior neighbor index (no boundary check, wraps within block)
        /// stride_idx: 0=+d0, 1=-d0, 2=+d1, 3=-d1, ...
        pub inline fn localNeighborFast(index: usize, comptime stride_idx: usize) usize {
            const dim = stride_idx / 2;
            const is_positive = (stride_idx % 2) == 0;
            const coord = (index / strides[dim]) % size;

            // Check boundary and wrap
            if (is_positive) {
                if (coord == size - 1) {
                    return @intCast(@as(isize, @intCast(index)) - @as(isize, @intCast((size - 1) * strides[dim])));
                }
            } else {
                if (coord == 0) {
                    return index + (size - 1) * strides[dim];
                }
            }
            return @intCast(@as(isize, @intCast(index)) + neighbor_strides[stride_idx]);
        }

        /// Get global coordinates from local index
        pub fn getGlobalCoords(self: *const Self, local_index: usize) [Nd]usize {
            const local = getLocalCoords(local_index);
            var global: [Nd]usize = undefined;
            inline for (0..Nd) |d| {
                global[d] = self.origin[d] + local[d];
            }
            return global;
        }

        /// Runtime version of localNeighborFast (stride_idx not comptime)
        pub fn localNeighborRuntime(index: usize, stride_idx: usize) usize {
            const dim = stride_idx / 2;
            const is_positive = (stride_idx % 2) == 0;

            const stride: isize = if (is_positive)
                @intCast(strides[dim])
            else
                -@as(isize, @intCast(strides[dim]));

            const coord = (index / strides[dim]) % size;

            if (is_positive) {
                if (coord == size - 1) {
                    return @intCast(@as(isize, @intCast(index)) - @as(isize, @intCast((size - 1) * strides[dim])));
                }
            } else {
                if (coord == 0) {
                    return index + (size - 1) * strides[dim];
                }
            }
            return @intCast(@as(isize, @intCast(index)) + stride);
        }

        /// Runtime version of getGhostIndex (face_idx not comptime)
        pub fn getGhostIndexRuntime(coords: [Nd]usize, face_idx: usize) usize {
            const dim = face_idx / 2;
            var ghost_idx: usize = 0;
            var ghost_stride: usize = 1;

            for (0..Nd) |d| {
                if (d != dim) {
                    ghost_idx += coords[d] * ghost_stride;
                    ghost_stride *= size;
                }
            }
            return ghost_idx;
        }

        // =====================================================================
        // Ghost Layer Indexing
        // =====================================================================

        /// Ghost face index from local boundary coordinates.
        /// For a face in dimension d, the ghost index uses coordinates in all other dimensions.
        /// Returns index into the ghost face array.
        pub inline fn getGhostIndex(coords: [Nd]usize, comptime face_idx: usize) usize {
            const dim = face_idx / 2;
            var ghost_idx: usize = 0;
            var ghost_stride: usize = 1;

            inline for (0..Nd) |d| {
                if (d != dim) {
                    ghost_idx += coords[d] * ghost_stride;
                    ghost_stride *= size;
                }
            }
            return ghost_idx;
        }

        // =====================================================================
        // Field Ghost Layer Extraction (generic, works with any FieldType)
        // =====================================================================

        /// Extract boundary face data for fields.
        /// face_idx: 0=+d0, 1=-d0, ...
        /// out: buffer of size face_size to receive boundary field values
        pub fn extractBoundaryFace(
            field: []const FieldType,
            comptime face_idx: usize,
            out: []FieldType,
        ) void {
            std.debug.assert(out.len >= face_size);

            const dim = face_idx / 2;
            const is_positive = (face_idx % 2) == 0;
            const boundary_coord = if (is_positive) size - 1 else 0;

            var ghost_idx: usize = 0;
            var coords: [Nd]usize = .{0} ** Nd;
            coords[dim] = boundary_coord;

            extractBoundaryFaceRecursive(field, &coords, dim, 0, out, &ghost_idx);
        }

        fn extractBoundaryFaceRecursive(
            field: []const FieldType,
            coords: *[Nd]usize,
            comptime skip_dim: usize,
            comptime current_dim: usize,
            out: []FieldType,
            ghost_idx: *usize,
        ) void {
            if (current_dim == Nd) {
                const local_idx = getLocalIndex(coords.*);
                out[ghost_idx.*] = field[local_idx];
                ghost_idx.* += 1;
                return;
            }

            if (current_dim == skip_dim) {
                extractBoundaryFaceRecursive(field, coords, skip_dim, current_dim + 1, out, ghost_idx);
            } else {
                for (0..size) |c| {
                    coords[current_dim] = c;
                    extractBoundaryFaceRecursive(field, coords, skip_dim, current_dim + 1, out, ghost_idx);
                }
            }
        }

        /// Runtime version of extractBoundaryFace.
        pub fn extractBoundaryFaceRuntime(
            field: []const FieldType,
            face_idx: usize,
            out: []FieldType,
        ) void {
            std.debug.assert(out.len >= face_size);
            std.debug.assert(face_idx < num_faces);

            const dim = face_idx / 2;
            const is_positive = (face_idx % 2) == 0;
            const boundary_coord = if (is_positive) size - 1 else 0;

            var ghost_idx: usize = 0;

            for (0..face_size) |face_cell| {
                var coords: [Nd]usize = undefined;
                var remaining = face_cell;

                for (0..Nd) |d| {
                    if (d == dim) {
                        coords[d] = boundary_coord;
                    } else {
                        coords[d] = remaining % size;
                        remaining /= size;
                    }
                }

                const local_idx = getLocalIndex(coords);
                out[ghost_idx] = field[local_idx];
                ghost_idx += 1;
            }
        }

        // =====================================================================
        // Block Metadata
        // =====================================================================

        /// Get the physical extent of this block
        pub fn getExtent(self: *const Self) [Nd]f64 {
            var extent: [Nd]f64 = undefined;
            inline for (0..Nd) |d| {
                extent[d] = @as(f64, @floatFromInt(size)) * self.spacing;
            }
            return extent;
        }

        /// Get physical position of a local site
        pub fn getPhysicalPosition(self: *const Self, local_index: usize) [Nd]f64 {
            const coords = getLocalCoords(local_index);
            var pos: [Nd]f64 = undefined;
            inline for (0..Nd) |d| {
                pos[d] = (@as(f64, @floatFromInt(self.origin[d])) + @as(f64, @floatFromInt(coords[d]))) * self.spacing;
            }
            return pos;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const TestFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
};

const TestFrontend4D = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = std.math.Complex(f64);
};

test "AMRBlock - dimensions" {
    const Block = AMRBlock(TestFrontend);
    try std.testing.expectEqual(@as(usize, 2), Block.dimensions);
    try std.testing.expectEqual(@as(usize, 4), Block.size);
    try std.testing.expectEqual(@as(usize, 16), Block.volume); // 4^2
    try std.testing.expectEqual(@as(usize, 4), Block.ghost_face_size); // 4^1
    try std.testing.expectEqual(@as(usize, 4), Block.num_ghost_faces); // 2*2
}

test "AMRBlock - 4D dimensions" {
    const Block = AMRBlock(TestFrontend4D);
    try std.testing.expectEqual(@as(usize, 4), Block.dimensions);
    try std.testing.expectEqual(@as(usize, 16), Block.size);
    try std.testing.expectEqual(@as(usize, 65536), Block.volume); // 16^4
    try std.testing.expectEqual(@as(usize, 4096), Block.ghost_face_size); // 16^3
    try std.testing.expectEqual(@as(usize, 8), Block.num_ghost_faces); // 2*4
}

test "AMRBlock - coordinate conversion" {
    const Block = AMRBlock(TestFrontend);

    // Test round-trip conversion
    const coords: [2]usize = .{ 2, 3 };
    const idx = Block.getLocalIndex(coords);
    const recovered = Block.getLocalCoords(idx);

    try std.testing.expectEqual(coords[0], recovered[0]);
    try std.testing.expectEqual(coords[1], recovered[1]);
}

test "AMRBlock - boundary detection" {
    const Block = AMRBlock(TestFrontend);

    // Interior point
    try std.testing.expect(!Block.isOnBoundary(.{ 1, 1 }));

    // Boundary points
    try std.testing.expect(Block.isOnBoundary(.{ 0, 1 })); // -x
    try std.testing.expect(Block.isOnBoundary(.{ 3, 1 })); // +x
    try std.testing.expect(Block.isOnBoundary(.{ 1, 0 })); // -y
    try std.testing.expect(Block.isOnBoundary(.{ 1, 3 })); // +y
}

test "AMRBlock - init" {
    const Block = AMRBlock(TestFrontend);
    const block = Block.init(2, .{ 16, 32 }, 0.25);

    try std.testing.expectEqual(@as(u8, 2), block.level);
    try std.testing.expectEqual(@as(usize, 16), block.origin[0]);
    try std.testing.expectEqual(@as(usize, 32), block.origin[1]);
    try std.testing.expectEqual(@as(f64, 0.25), block.spacing);
    try std.testing.expectEqual(std.math.maxInt(usize), block.block_index);
}

//! AMR Operators Module
//!
//! Implements prolongation (coarse->fine) and restriction (fine->coarse) operators
//! for Adaptive Mesh Refinement. These operators handle inter-level data transfer.
//!
//! ## Prolongation (Coarse -> Fine)
//!
//! When refining a block, we interpolate coarse values to create fine values:
//! - **Injection**: Copy coarse value to all children (simple, first-order)
//!
//! For wavefunctions, we preserve the L2 norm by dividing by sqrt(2^Nd).
//!
//! ## Restriction (Fine -> Coarse)
//!
//! When coarsening or filling ghost layers:
//! - **Full-weighting**: Average all 2^Nd children (smooth, second-order)
//!
//! For wavefunctions, we preserve the L2 norm by multiplying by sqrt(2^Nd).
//!
//! ## Usage
//!
//! ```zig
//! const MyFrontend = struct {
//!     pub const Nd: usize = 4;
//!     pub const block_size: usize = 16;
//!     pub const FieldType = f64;
//! };
//! const Ops = AMROperators(MyFrontend);
//!
//! // Prolongate coarse block to 16 fine blocks
//! Ops.prolongateInjection(coarse_field, &fine_field_blocks, true);
//!
//! // Restrict fine blocks to coarse
//! Ops.restrictFullWeight(&fine_field_blocks, coarse_field, true);
//! ```

const std = @import("std");
const frontend_mod = @import("frontend.zig");

const Complex = std.math.Complex(f64);

/// AMR Operators for prolongation and restriction.
///
/// Parameterized by Frontend which defines Nd, block_size, and FieldType.
pub fn AMROperators(comptime Frontend: type) type {
    comptime frontend_mod.validateFrontend(Frontend);

    const Info = frontend_mod.FrontendInfo(Frontend);
    const Nd = Info.Nd;
    const block_size = Info.block_size;
    const FieldType = Info.FieldType;

    // Number of children when refining = 2^Nd
    const num_children = Info.num_children;

    // Block volume
    const block_volume = Info.volume;

    // Face size for ghost layers
    const face_size = Info.face_size;

    // Half block size (for child block indexing)
    const half_size = block_size / 2;

    // Precompute strides for row-major indexing
    const strides: [Nd]usize = blk: {
        var str: [Nd]usize = undefined;
        var acc: usize = 1;
        for (0..Nd) |i| {
            str[i] = acc;
            acc *= block_size;
        }
        break :blk str;
    };

    // Strides for half-size block
    const half_strides: [Nd]usize = blk: {
        var str: [Nd]usize = undefined;
        var acc: usize = 1;
        for (0..Nd) |i| {
            str[i] = acc;
            acc *= half_size;
        }
        break :blk str;
    };

    // Normalization factor for wavefunction: sqrt(2^Nd)
    const norm_factor = @sqrt(@as(f64, @floatFromInt(num_children)));

    return struct {
        pub const dimensions = Nd;
        pub const children_count = num_children;
        pub const volume = block_volume;
        pub const Field = FieldType;

        // =====================================================================
        // Compile-Time Field Type Detection
        // =====================================================================

        /// Detect if FieldType is a Complex number
        const is_complex = @typeInfo(FieldType) == .@"struct" and @hasField(FieldType, "re") and @hasField(FieldType, "im");

        /// Detect if FieldType is an array
        const is_array = @typeInfo(FieldType) == .array;

        /// Get the element type if array, otherwise FieldType itself
        const ElementType = if (is_array) std.meta.Child(FieldType) else FieldType;

        /// Detect if element type is Complex
        const element_is_complex = @typeInfo(ElementType) == .@"struct" and @hasField(ElementType, "re") and @hasField(ElementType, "im");

        /// Number of elements per field (1 for scalar, N for [N]T)
        const field_len = if (is_array) @typeInfo(FieldType).array.len else 1;

        // =====================================================================
        // Generic Field Arithmetic
        // =====================================================================

        /// Scale a field value by a real factor
        inline fn scaleField(val: FieldType, scale: f64) FieldType {
            if (is_array) {
                var result: FieldType = undefined;
                inline for (0..field_len) |i| {
                    result[i] = scaleElement(val[i], scale);
                }
                return result;
            } else {
                return scaleElement(val, scale);
            }
        }

        /// Scale a single element by a real factor
        inline fn scaleElement(val: ElementType, scale: f64) ElementType {
            if (element_is_complex) {
                return val.mul(Complex.init(scale, 0));
            } else {
                return val * scale;
            }
        }

        /// Add two field values
        inline fn addField(a: FieldType, b: FieldType) FieldType {
            if (is_array) {
                var result: FieldType = undefined;
                inline for (0..field_len) |i| {
                    result[i] = addElement(a[i], b[i]);
                }
                return result;
            } else {
                return addElement(a, b);
            }
        }

        /// Add two elements
        inline fn addElement(a: ElementType, b: ElementType) ElementType {
            if (element_is_complex) {
                return a.add(b);
            } else {
                return a + b;
            }
        }

        /// Zero value for field type
        fn zeroField() FieldType {
            if (is_array) {
                var result: FieldType = undefined;
                inline for (0..field_len) |i| {
                    result[i] = zeroElement();
                }
                return result;
            } else {
                return zeroElement();
            }
        }

        /// Zero value for element type
        inline fn zeroElement() ElementType {
            if (element_is_complex) {
                return Complex.init(0, 0);
            } else {
                return 0;
            }
        }

        // =====================================================================
        // Coordinate Utilities
        // =====================================================================

        /// Get index from coordinates in a block
        pub inline fn getIndex(coords: [Nd]usize) usize {
            var idx: usize = 0;
            inline for (0..Nd) |d| {
                idx += coords[d] * strides[d];
            }
            return idx;
        }

        /// Get coordinates from index in a block
        pub inline fn getCoords(index: usize) [Nd]usize {
            var coords: [Nd]usize = undefined;
            inline for (0..Nd) |d| {
                coords[d] = (index / strides[d]) % block_size;
            }
            return coords;
        }

        /// Get index from coordinates in a half-size block
        inline fn getHalfIndex(coords: [Nd]usize) usize {
            var idx: usize = 0;
            inline for (0..Nd) |d| {
                idx += coords[d] * half_strides[d];
            }
            return idx;
        }

        /// Get coordinates from index in a half-size block
        inline fn getHalfCoords(index: usize) [Nd]usize {
            var coords: [Nd]usize = undefined;
            inline for (0..Nd) |d| {
                coords[d] = (index / half_strides[d]) % half_size;
            }
            return coords;
        }

        // =====================================================================
        // Field Prolongation/Restriction (Generic FieldType)
        // =====================================================================

        /// Prolongate field from coarse to fine using injection.
        /// Each coarse value maps to 2^Nd fine values.
        /// Divides by sqrt(2^Nd) to preserve L2 norm if preserve_norm is true.
        pub fn prolongateInjection(
            coarse: []const FieldType,
            fine_children: *[num_children][]FieldType,
            preserve_norm: bool,
        ) void {
            std.debug.assert(coarse.len >= block_volume);
            for (fine_children) |child| {
                std.debug.assert(child.len >= block_volume);
            }

            const scale = if (preserve_norm) 1.0 / norm_factor else 1.0;

            for (0..block_volume) |coarse_idx| {
                const coarse_coords = getCoords(coarse_idx);

                // Scale the field value
                const scaled_val = scaleField(coarse[coarse_idx], scale);

                // Determine which child block and fine cell mapping
                var child_idx: usize = 0;
                var fine_base_coords: [Nd]usize = undefined;

                inline for (0..Nd) |d| {
                    const half = coarse_coords[d] / half_size;
                    child_idx |= half << @intCast(d);
                    fine_base_coords[d] = (coarse_coords[d] % half_size) * 2;
                }

                // Write to all 2^Nd fine cells
                for (0..num_children) |offset| {
                    var fine_coords: [Nd]usize = undefined;
                    inline for (0..Nd) |d| {
                        fine_coords[d] = fine_base_coords[d] + ((offset >> @intCast(d)) & 1);
                    }
                    const fine_idx = getIndex(fine_coords);
                    fine_children[child_idx][fine_idx] = scaled_val;
                }
            }
        }

        /// Restrict field from fine to coarse using full-weighting.
        /// Each coarse value is the average of 2^Nd fine values.
        /// Multiplies by sqrt(2^Nd) to preserve L2 norm if preserve_norm is true.
        pub fn restrictFullWeight(
            fine_children: *const [num_children][]const FieldType,
            coarse: []FieldType,
            preserve_norm: bool,
        ) void {
            for (fine_children) |child| {
                std.debug.assert(child.len >= block_volume);
            }
            std.debug.assert(coarse.len >= block_volume);

            const avg_factor = 1.0 / @as(f64, @floatFromInt(num_children));
            const scale = if (preserve_norm) norm_factor * avg_factor else avg_factor;

            for (0..block_volume) |coarse_idx| {
                const coarse_coords = getCoords(coarse_idx);

                var child_idx: usize = 0;
                var fine_base_coords: [Nd]usize = undefined;

                inline for (0..Nd) |d| {
                    const half = coarse_coords[d] / half_size;
                    child_idx |= half << @intCast(d);
                    fine_base_coords[d] = (coarse_coords[d] % half_size) * 2;
                }

                // Sum over all 2^Nd fine cells
                var sum: FieldType = zeroField();
                for (0..num_children) |offset| {
                    var fine_coords: [Nd]usize = undefined;
                    inline for (0..Nd) |d| {
                        fine_coords[d] = fine_base_coords[d] + ((offset >> @intCast(d)) & 1);
                    }
                    const fine_idx = getIndex(fine_coords);
                    sum = addField(sum, fine_children[child_idx][fine_idx]);
                }

                // Apply scale
                coarse[coarse_idx] = scaleField(sum, scale);
            }
        }

        /// Prolongate coarse ghost face to fine ghost face.
        pub fn prolongateFace(
            coarse_face: []const FieldType,
            fine_face: []FieldType,
            comptime face_dim: usize,
        ) void {
            const coarse_face_size = face_size / num_children * 2;
            std.debug.assert(coarse_face.len >= coarse_face_size);
            std.debug.assert(fine_face.len >= face_size);

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
                    _ = face_dim;
                    coarse_idx += coarse_coords[d] * coarse_stride;
                    coarse_stride *= block_size / 2;
                }

                if (coarse_idx < coarse_face.len) {
                    fine_face[fine_idx] = coarse_face[coarse_idx];
                }

                var d: usize = 0;
                while (d < Nd - 1) : (d += 1) {
                    coords[d] += 1;
                    if (coords[d] < block_size) break;
                    coords[d] = 0;
                }
            }
        }

        /// Restrict fine ghost face to coarse ghost face.
        pub fn restrictFace(
            fine_face: []const FieldType,
            coarse_face: []FieldType,
            comptime face_dim: usize,
        ) void {
            const coarse_face_size = face_size / num_children * 2;
            std.debug.assert(fine_face.len >= face_size);
            std.debug.assert(coarse_face.len >= coarse_face_size);

            // Initialize coarse face to zero
            for (coarse_face) |*cf| {
                cf.* = zeroField();
            }

            var fine_idx: usize = 0;
            var coords: [Nd - 1]usize = .{0} ** (Nd - 1);
            const weight = 1.0 / @as(f64, @floatFromInt(num_children / 2));

            while (fine_idx < face_size) : (fine_idx += 1) {
                var coarse_coords: [Nd - 1]usize = undefined;
                inline for (0..(Nd - 1)) |d| {
                    coarse_coords[d] = coords[d] / 2;
                }

                var coarse_idx: usize = 0;
                var coarse_stride: usize = 1;
                inline for (0..(Nd - 1)) |d| {
                    _ = face_dim;
                    coarse_idx += coarse_coords[d] * coarse_stride;
                    coarse_stride *= block_size / 2;
                }

                if (coarse_idx < coarse_face_size) {
                    const weighted = scaleField(fine_face[fine_idx], weight);
                    coarse_face[coarse_idx] = addField(coarse_face[coarse_idx], weighted);
                }

                var d: usize = 0;
                while (d < Nd - 1) : (d += 1) {
                    coords[d] += 1;
                    if (coords[d] < block_size) break;
                    coords[d] = 0;
                }
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "AMROperators - scalar field prolongation/restriction" {
    const TestFrontend = struct {
        pub const Nd: usize = 2;
        pub const block_size: usize = 4;
        pub const FieldType = f64;
    };

    const Ops = AMROperators(TestFrontend);
    const vol = Ops.volume; // 4^2 = 16
    const n_children = Ops.children_count; // 2^2 = 4

    // Create coarse field with value 4.0 everywhere
    var coarse: [vol]f64 = .{4.0} ** vol;

    // Create fine children storage
    var fine_storage: [n_children][vol]f64 = undefined;
    var fine_ptrs: [n_children][]f64 = undefined;
    for (0..n_children) |i| {
        fine_storage[i] = .{0.0} ** vol;
        fine_ptrs[i] = &fine_storage[i];
    }

    // Prolongate without norm preservation
    Ops.prolongateInjection(&coarse, &fine_ptrs, false);

    // All fine cells should have value 4.0
    for (0..n_children) |c| {
        for (fine_ptrs[c]) |v| {
            try std.testing.expectApproxEqRel(4.0, v, 1e-10);
        }
    }

    // Restrict back
    var coarse_out: [vol]f64 = .{0.0} ** vol;
    const fine_const: [n_children][]const f64 = blk: {
        var arr: [n_children][]const f64 = undefined;
        for (0..n_children) |i| arr[i] = &fine_storage[i];
        break :blk arr;
    };

    Ops.restrictFullWeight(&fine_const, &coarse_out, false);

    // Should recover original value (averaging 4.0s gives 4.0)
    for (coarse_out) |v| {
        try std.testing.expectApproxEqRel(4.0, v, 1e-10);
    }
}

test "AMROperators - complex field prolongation/restriction" {
    const TestFrontend = struct {
        pub const Nd: usize = 2;
        pub const block_size: usize = 4;
        pub const FieldType = Complex;
    };

    const Ops = AMROperators(TestFrontend);
    const vol = Ops.volume;
    const n_children = Ops.children_count;

    // Create coarse field
    var coarse: [vol]Complex = .{Complex.init(2.0, 1.0)} ** vol;

    // Create fine children storage
    var fine_storage: [n_children][vol]Complex = undefined;
    var fine_ptrs: [n_children][]Complex = undefined;
    for (0..n_children) |i| {
        fine_storage[i] = .{Complex.init(0, 0)} ** vol;
        fine_ptrs[i] = &fine_storage[i];
    }

    // Prolongate with norm preservation
    Ops.prolongateInjection(&coarse, &fine_ptrs, true);

    // Check values are scaled by 1/sqrt(4) = 0.5
    const expected = Complex.init(1.0, 0.5);
    for (0..n_children) |c| {
        for (fine_ptrs[c]) |v| {
            try std.testing.expectApproxEqRel(expected.re, v.re, 1e-10);
            try std.testing.expectApproxEqRel(expected.im, v.im, 1e-10);
        }
    }
}

test "AMROperators - array field type" {
    const TestFrontend = struct {
        pub const Nd: usize = 2;
        pub const block_size: usize = 4;
        pub const FieldType = [2]f64;
    };

    const Ops = AMROperators(TestFrontend);
    const vol = Ops.volume;
    const n_children = Ops.children_count;

    // Create coarse field
    var coarse: [vol][2]f64 = .{.{ 1.0, 2.0 }} ** vol;

    // Create fine children storage
    var fine_storage: [n_children][vol][2]f64 = undefined;
    var fine_ptrs: [n_children][][2]f64 = undefined;
    for (0..n_children) |i| {
        fine_storage[i] = .{.{ 0.0, 0.0 }} ** vol;
        fine_ptrs[i] = &fine_storage[i];
    }

    // Prolongate without norm preservation
    Ops.prolongateInjection(&coarse, &fine_ptrs, false);

    // Check values
    for (0..n_children) |c| {
        for (fine_ptrs[c]) |v| {
            try std.testing.expectApproxEqRel(1.0, v[0], 1e-10);
            try std.testing.expectApproxEqRel(2.0, v[1], 1e-10);
        }
    }
}

test "AMROperators - complex array field type" {
    const TestFrontend = struct {
        pub const Nd: usize = 2;
        pub const block_size: usize = 4;
        pub const FieldType = [2]Complex;
    };

    const Ops = AMROperators(TestFrontend);
    const vol = Ops.volume;
    const n_children = Ops.children_count;

    // Create coarse field
    var coarse: [vol][2]Complex = .{.{ Complex.init(1.0, 0.5), Complex.init(2.0, -0.5) }} ** vol;

    // Create fine children storage
    var fine_storage: [n_children][vol][2]Complex = undefined;
    var fine_ptrs: [n_children][][2]Complex = undefined;
    for (0..n_children) |i| {
        fine_storage[i] = .{.{ Complex.init(0, 0), Complex.init(0, 0) }} ** vol;
        fine_ptrs[i] = &fine_storage[i];
    }

    // Prolongate without norm preservation
    Ops.prolongateInjection(&coarse, &fine_ptrs, false);

    // Check values
    for (0..n_children) |c| {
        for (fine_ptrs[c]) |v| {
            try std.testing.expectApproxEqRel(1.0, v[0].re, 1e-10);
            try std.testing.expectApproxEqRel(0.5, v[0].im, 1e-10);
            try std.testing.expectApproxEqRel(2.0, v[1].re, 1e-10);
            try std.testing.expectApproxEqRel(-0.5, v[1].im, 1e-10);
        }
    }
}

test "AMROperators - 4D frontend" {
    const TestFrontend = struct {
        pub const Nd: usize = 4;
        pub const block_size: usize = 4;
        pub const FieldType = f64;
    };

    const Ops = AMROperators(TestFrontend);

    try std.testing.expectEqual(@as(usize, 4), Ops.dimensions);
    try std.testing.expectEqual(@as(usize, 16), Ops.children_count); // 2^4
    try std.testing.expectEqual(@as(usize, 256), Ops.volume); // 4^4
}

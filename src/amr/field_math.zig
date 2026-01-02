//! Field Math Utilities
//!
//! Generic comptime math helpers for field operations on AMR grids.
//! These functions work with scalar types, complex types, and arrays thereof.
//!
//! All functions are pure and stateless.

const std = @import("std");

/// Check if a type is a complex number (has .re and .im fields).
pub fn isComplex(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasField(T, "re") and @hasField(T, "im");
}

/// Return a zero-initialized field value.
/// Works with scalars, complex numbers, and arrays of either.
pub fn zeroField(comptime Field: type) Field {
    const info = @typeInfo(Field);

    if (info == .array) {
        const Child = std.meta.Child(Field);
        const len = info.array.len;
        var result: Field = undefined;
        for (0..len) |i| {
            result[i] = zeroField(Child);
        }
        return result;
    }

    return zeroElement(Field);
}

/// Return a zero value for a scalar or complex element.
pub fn zeroElement(comptime T: type) T {
    if (comptime isComplex(T)) {
        return T.init(0, 0);
    }
    return @as(T, 0);
}

/// Add two field values element-wise.
/// Works with scalars, complex numbers, and arrays of either.
pub fn addField(comptime Field: type, a: Field, b: Field) Field {
    const info = @typeInfo(Field);

    if (info == .array) {
        const Child = std.meta.Child(Field);
        const len = info.array.len;
        var result: Field = undefined;
        for (0..len) |i| {
            result[i] = addField(Child, a[i], b[i]);
        }
        return result;
    }

    return addElement(Field, a, b);
}

/// Add two scalar or complex elements.
pub fn addElement(comptime T: type, a: T, b: T) T {
    if (comptime isComplex(T)) {
        return a.add(b);
    }
    return a + b;
}

/// Subtract two field values element-wise: a - b.
/// Works with scalars, complex numbers, and arrays of either.
pub fn subField(comptime Field: type, a: Field, b: Field) Field {
    const info = @typeInfo(Field);

    if (info == .array) {
        const Child = std.meta.Child(Field);
        const len = info.array.len;
        var result: Field = undefined;
        for (0..len) |i| {
            result[i] = subField(Child, a[i], b[i]);
        }
        return result;
    }

    return subElement(Field, a, b);
}

/// Subtract two scalar or complex elements: a - b.
pub fn subElement(comptime T: type, a: T, b: T) T {
    if (comptime isComplex(T)) {
        return a.sub(b);
    }
    return a - b;
}

/// Scale a field value by a real scalar.
/// Works with scalars, complex numbers, and arrays of either.
pub fn scaleField(comptime Field: type, value: Field, scale: f64) Field {
    const info = @typeInfo(Field);

    if (info == .array) {
        const Child = std.meta.Child(Field);
        const len = info.array.len;
        var result: Field = undefined;
        for (0..len) |i| {
            result[i] = scaleField(Child, value[i], scale);
        }
        return result;
    }

    return scaleElement(Field, value, scale);
}

/// Scale a scalar or complex element by a real scalar.
pub fn scaleElement(comptime T: type, value: T, scale: f64) T {
    if (comptime isComplex(T)) {
        return value.mul(T.init(scale, 0));
    }
    return value * @as(T, @floatCast(scale));
}

/// Add a scaled field to a destination: dest += src * scale.
/// Works with scalars, complex numbers, and arrays of either.
pub fn addScaledField(comptime Field: type, dest: *Field, src: Field, scale: f64) void {
    const info = @typeInfo(Field);

    if (info == .array) {
        const Child = std.meta.Child(Field);
        const len = info.array.len;
        for (0..len) |i| {
            addScaledField(Child, &dest.*[i], src[i], scale);
        }
        return;
    }

    if (comptime isComplex(Field)) {
        dest.* = dest.*.add(src.mul(Field.init(scale, 0)));
    } else {
        dest.* += src * @as(Field, @floatCast(scale));
    }
}

/// Compute the squared magnitude of a field value.
/// For complex arrays, returns sum of |z|^2 for each component.
pub fn normSquared(comptime Field: type, value: Field) f64 {
    const info = @typeInfo(Field);

    if (info == .array) {
        const Child = std.meta.Child(Field);
        const len = info.array.len;
        var sum: f64 = 0;
        for (0..len) |i| {
            sum += normSquared(Child, value[i]);
        }
        return sum;
    }

    if (comptime isComplex(Field)) {
        return value.re * value.re + value.im * value.im;
    }
    return @as(f64, @floatCast(value * value));
}

// =============================================================================
// Tests
// =============================================================================

const Complex = std.math.Complex(f64);

test "isComplex" {
    try std.testing.expect(isComplex(Complex));
    try std.testing.expect(!isComplex(f64));
    try std.testing.expect(!isComplex(f32));
    try std.testing.expect(!isComplex([4]f64));
    try std.testing.expect(!isComplex([4]Complex));
}

test "zeroField - scalar" {
    const z: f64 = zeroField(f64);
    try std.testing.expectEqual(@as(f64, 0), z);
}

test "zeroField - complex" {
    const z: Complex = zeroField(Complex);
    try std.testing.expectEqual(@as(f64, 0), z.re);
    try std.testing.expectEqual(@as(f64, 0), z.im);
}

test "zeroField - array of complex" {
    const z: [4]Complex = zeroField([4]Complex);
    for (z) |c| {
        try std.testing.expectEqual(@as(f64, 0), c.re);
        try std.testing.expectEqual(@as(f64, 0), c.im);
    }
}

test "addField - scalar" {
    const result = addField(f64, 1.5, 2.5);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result, 1e-10);
}

test "addField - complex" {
    const a = Complex.init(1, 2);
    const b = Complex.init(3, 4);
    const result = addField(Complex, a, b);
    try std.testing.expectApproxEqAbs(@as(f64, 4), result.re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6), result.im, 1e-10);
}

test "addField - array" {
    const a = [2]f64{ 1, 2 };
    const b = [2]f64{ 3, 4 };
    const result = addField([2]f64, a, b);
    try std.testing.expectApproxEqAbs(@as(f64, 4), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6), result[1], 1e-10);
}

test "scaleField - scalar" {
    const result = scaleField(f64, 2.0, 3.0);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result, 1e-10);
}

test "scaleField - complex" {
    const c = Complex.init(2, 3);
    const result = scaleField(Complex, c, 2.0);
    try std.testing.expectApproxEqAbs(@as(f64, 4), result.re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6), result.im, 1e-10);
}

test "addScaledField - complex array" {
    var dest = [2]Complex{ Complex.init(1, 0), Complex.init(0, 1) };
    const src = [2]Complex{ Complex.init(2, 0), Complex.init(0, 2) };
    addScaledField([2]Complex, &dest, src, 0.5);
    try std.testing.expectApproxEqAbs(@as(f64, 2), dest[0].re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), dest[0].im, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), dest[1].re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2), dest[1].im, 1e-10);
}

test "normSquared - complex" {
    const c = Complex.init(3, 4);
    const result = normSquared(Complex, c);
    try std.testing.expectApproxEqAbs(@as(f64, 25), result, 1e-10);
}

test "normSquared - complex array" {
    const arr = [2]Complex{ Complex.init(3, 4), Complex.init(0, 1) };
    const result = normSquared([2]Complex, arr);
    try std.testing.expectApproxEqAbs(@as(f64, 26), result, 1e-10);
}

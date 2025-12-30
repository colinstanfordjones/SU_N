const std = @import("std");
pub const Complex = std.math.Complex(f64);

/// Returns the squared magnitude of a complex number
pub fn magSq(c: Complex) f64 {
    return c.re * c.re + c.im * c.im;
}

/// Returns the magnitude of a complex number
pub fn mag(c: Complex) f64 {
    return std.math.sqrt(magSq(c));
}

/// Returns the phase angle of a complex number in [-π, π]
pub fn arg(c: Complex) f64 {
    return std.math.atan2(c.im, c.re);
}

// =============================================================================
// Tests
// =============================================================================

test "atan2 quadrant I" {
    const result = std.math.atan2(@as(f64, 1.0), @as(f64, 1.0));
    const expected = std.math.pi / 4.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

test "atan2 quadrant II" {
    const result = std.math.atan2(@as(f64, 1.0), @as(f64, -1.0));
    const expected = 3.0 * std.math.pi / 4.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

test "atan2 quadrant III" {
    const result = std.math.atan2(@as(f64, -1.0), @as(f64, -1.0));
    const expected = -3.0 * std.math.pi / 4.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

test "atan2 quadrant IV" {
    const result = std.math.atan2(@as(f64, -1.0), @as(f64, 1.0));
    const expected = -std.math.pi / 4.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

test "atan2 positive y axis" {
    const result = std.math.atan2(@as(f64, 1.0), @as(f64, 0.0));
    const expected = std.math.pi / 2.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

test "atan2 negative y axis" {
    const result = std.math.atan2(@as(f64, -1.0), @as(f64, 0.0));
    const expected = -std.math.pi / 2.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

test "arg phase angle" {
    const c = Complex.init(1.0, 1.0);
    const result = arg(c);
    const expected = std.math.pi / 4.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-14);
}

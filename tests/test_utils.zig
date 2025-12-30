const std = @import("std");
const Complex = std.math.Complex(f64);
const constants = @import("su_n").constants;

pub fn expectApproxEq(a: anytype, b: anytype) !void {
    const a_64: f64 = @floatCast(a);
    const b_64: f64 = @floatCast(b);
    try std.testing.expect(std.math.approxEqAbs(f64, a_64, b_64, constants.test_epsilon));
}

pub fn expectComplexApproxEq(a: anytype, b: anytype) !void {
    try expectApproxEq(a.re, b.re);
    try expectApproxEq(a.im, b.im);
}

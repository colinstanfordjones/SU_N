const std = @import("std");
const su_n = @import("su_n");
const U1 = su_n.gauge.gauge_u1.U1;
const Complex = std.math.Complex(f64);
const constants = su_n.constants;
const utils = @import("../test_utils.zig");

test "U(1) Unitarity" {
    // U = e^(i*theta)
    // U^dag = e^(-i*theta)
    // U * U^dag = 1
    const theta = constants.pi / 3.0;
    const u = U1.rotation(theta);
    const adj = u.adjoint();
    
    const prod = u.mul(adj);
    const id = U1.identity();
    
    try utils.expectComplexApproxEq(prod.data[0][0], id.data[0][0]);
}

test "U(1) Composition" {
    // e^(i*a) * e^(i*b) = e^(i*(a+b))
    const a = 0.5;
    const b = 0.25;
    
    const ua = U1.rotation(a);
    const ub = U1.rotation(b);
    const uab = ua.mul(ub);
    
    const expected = U1.rotation(a + b);
    
    try utils.expectComplexApproxEq(uab.data[0][0], expected.data[0][0]);
}

test "U(1) Abelian" {
    // U(1) is Abelian: [A, B] = 0
    const ua = U1.rotation(0.5);
    const ub = U1.rotation(1.2);
    
    // ua * ub
    const ab = ua.mul(ub);
    // ub * ua
    const ba = ub.mul(ua);
    
    try utils.expectComplexApproxEq(ab.data[0][0], ba.data[0][0]);
    
    // Commutator should be 0
    const comm = ab.sub(ba);
    try utils.expectComplexApproxEq(comm.data[0][0], Complex.init(0, 0));
}

test "U(1) Generator" {
    // Check local expansion: U(eps) ~ I + i*eps*T
    const eps = constants.small_epsilon;
    const u = U1.rotation(eps);
    
    // approx = I + i*eps*T
    // i*eps*T = i*eps*1 = (0, eps)
    // I + ... = (1, eps)
    
    // Actual u = cos(eps) + i*sin(eps)
    // cos(eps) ~ 1 - eps^2/2
    // sin(eps) ~ eps
    
    try utils.expectApproxEq(u.data[0][0].re, 1.0); // Close to 1
    try utils.expectApproxEq(u.data[0][0].im, eps); // Close to eps
    
    // Let's check difference
    const diff_re = @abs(u.data[0][0].re - 1.0);
    const diff_im = @abs(u.data[0][0].im - eps);
    
    try std.testing.expect(diff_re < constants.test_epsilon); // Taylor series error is O(eps^2) ~ 1e-10
    try std.testing.expect(diff_im < constants.test_epsilon); // Taylor series error O(eps^3)
}
const std = @import("std");
const su_n = @import("su_n");
const Matrix = su_n.math.Matrix;
const Complex = std.math.Complex(f64);
const utils = @import("../test_utils.zig");
const pi = std.math.pi;

// =============================================================================
// exp(0) = I tests
// =============================================================================

test "exp(zero matrix) equals identity - scalar 2x2" {
    const M2 = Matrix(f64, 2, 2);
    const zero = M2.zero();
    const result = zero.exp();
    const identity = M2.identity();

    for (0..2) |i| {
        for (0..2) |j| {
            try utils.expectApproxEq(result.data[i][j], identity.data[i][j]);
        }
    }
}

test "exp(zero matrix) equals identity - complex 2x2" {
    const M2 = Matrix(Complex, 2, 2);
    const zero = M2.zero();
    const result = zero.exp();
    const identity = M2.identity();

    for (0..2) |i| {
        for (0..2) |j| {
            try utils.expectComplexApproxEq(result.data[i][j], identity.data[i][j]);
        }
    }
}

test "exp(zero matrix) equals identity - scalar 3x3" {
    const M3 = Matrix(f64, 3, 3);
    const zero = M3.zero();
    const result = zero.exp();
    const identity = M3.identity();

    for (0..3) |i| {
        for (0..3) |j| {
            try utils.expectApproxEq(result.data[i][j], identity.data[i][j]);
        }
    }
}

// =============================================================================
// exp(i*theta*sigma_z) rotation tests
// =============================================================================

test "exp(i*theta*sigma_z) produces correct rotation - theta=0" {
    // exp(i*0*sigma_z) = I
    const M2 = Matrix(Complex, 2, 2);
    const zero = M2.zero();
    const result = zero.exp();
    const identity = M2.identity();

    for (0..2) |i| {
        for (0..2) |j| {
            try utils.expectComplexApproxEq(result.data[i][j], identity.data[i][j]);
        }
    }
}

test "exp(i*theta*sigma_z) produces correct rotation - theta=pi/4" {
    // sigma_z = [1 0; 0 -1]
    // i*theta*sigma_z = [i*theta 0; 0 -i*theta]
    // exp(i*theta*sigma_z) = [exp(i*theta) 0; 0 exp(-i*theta)]
    //                      = [cos(theta)+i*sin(theta) 0; 0 cos(theta)-i*sin(theta)]

    const M2 = Matrix(Complex, 2, 2);
    const theta: f64 = pi / 4.0;

    // Construct i*theta*sigma_z
    const i_theta_sigma_z = M2.init([_][2]Complex{
        .{ Complex.init(0, theta), Complex.init(0, 0) },
        .{ Complex.init(0, 0), Complex.init(0, -theta) },
    });

    const result = i_theta_sigma_z.exp();

    // Expected: [cos(theta)+i*sin(theta), 0; 0, cos(theta)-i*sin(theta)]
    const cos_t = @cos(theta);
    const sin_t = @sin(theta);

    try utils.expectComplexApproxEq(result.data[0][0], Complex.init(cos_t, sin_t));
    try utils.expectComplexApproxEq(result.data[0][1], Complex.init(0, 0));
    try utils.expectComplexApproxEq(result.data[1][0], Complex.init(0, 0));
    try utils.expectComplexApproxEq(result.data[1][1], Complex.init(cos_t, -sin_t));
}

test "exp(i*theta*sigma_z) produces correct rotation - theta=pi/2" {
    const M2 = Matrix(Complex, 2, 2);
    const theta: f64 = pi / 2.0;

    const i_theta_sigma_z = M2.init([_][2]Complex{
        .{ Complex.init(0, theta), Complex.init(0, 0) },
        .{ Complex.init(0, 0), Complex.init(0, -theta) },
    });

    const result = i_theta_sigma_z.exp();

    // At theta=pi/2: cos=0, sin=1
    // Expected: [i, 0; 0, -i]
    try utils.expectComplexApproxEq(result.data[0][0], Complex.init(0, 1));
    try utils.expectComplexApproxEq(result.data[0][1], Complex.init(0, 0));
    try utils.expectComplexApproxEq(result.data[1][0], Complex.init(0, 0));
    try utils.expectComplexApproxEq(result.data[1][1], Complex.init(0, -1));
}

// =============================================================================
// Pauli matrix exponential tests
// =============================================================================

test "exp(i*theta*sigma_x) analytical formula" {
    // For any Pauli matrix sigma: exp(i*theta*sigma) = cos(theta)*I + i*sin(theta)*sigma
    // sigma_x = [0 1; 1 0]
    // exp(i*theta*sigma_x) = cos(theta)*I + i*sin(theta)*sigma_x
    //                      = [cos(theta) i*sin(theta); i*sin(theta) cos(theta)]

    const M2 = Matrix(Complex, 2, 2);
    const theta: f64 = pi / 3.0;
    const i = Complex.init(0, 1);

    // Construct i*theta*sigma_x
    const i_theta_sigma_x = M2.init([_][2]Complex{
        .{ Complex.init(0, 0), i.mul(Complex.init(theta, 0)) },
        .{ i.mul(Complex.init(theta, 0)), Complex.init(0, 0) },
    });

    const result = i_theta_sigma_x.exp();

    const cos_t = @cos(theta);
    const sin_t = @sin(theta);
    const i_sin_t = Complex.init(0, sin_t);

    try utils.expectComplexApproxEq(result.data[0][0], Complex.init(cos_t, 0));
    try utils.expectComplexApproxEq(result.data[0][1], i_sin_t);
    try utils.expectComplexApproxEq(result.data[1][0], i_sin_t);
    try utils.expectComplexApproxEq(result.data[1][1], Complex.init(cos_t, 0));
}

test "exp(i*theta*sigma_y) analytical formula" {
    // sigma_y = [0 -i; i 0]
    // exp(i*theta*sigma_y) = cos(theta)*I + i*sin(theta)*sigma_y
    //                      = [cos(theta) sin(theta); -sin(theta) cos(theta)]

    const M2 = Matrix(Complex, 2, 2);
    const theta: f64 = pi / 6.0;

    // i*sigma_y = [0 1; -1 0]
    // i*theta*sigma_y = [0 theta; -theta 0]
    const i_theta_sigma_y = M2.init([_][2]Complex{
        .{ Complex.init(0, 0), Complex.init(theta, 0) },
        .{ Complex.init(-theta, 0), Complex.init(0, 0) },
    });

    const result = i_theta_sigma_y.exp();

    const cos_t = @cos(theta);
    const sin_t = @sin(theta);

    // Expected: [cos(theta) sin(theta); -sin(theta) cos(theta)]
    try utils.expectComplexApproxEq(result.data[0][0], Complex.init(cos_t, 0));
    try utils.expectComplexApproxEq(result.data[0][1], Complex.init(sin_t, 0));
    try utils.expectComplexApproxEq(result.data[1][0], Complex.init(-sin_t, 0));
    try utils.expectComplexApproxEq(result.data[1][1], Complex.init(cos_t, 0));
}

// =============================================================================
// Frobenius norm tests
// =============================================================================

test "frobeniusNorm of identity matrix" {
    const M2 = Matrix(f64, 2, 2);
    const identity = M2.identity();
    // ||I|| = sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2)
    try utils.expectApproxEq(identity.frobeniusNorm(), std.math.sqrt(2.0));
}

test "frobeniusNorm of zero matrix" {
    const M2 = Matrix(f64, 2, 2);
    const zero = M2.zero();
    try utils.expectApproxEq(zero.frobeniusNorm(), 0.0);
}

test "frobeniusNorm of complex matrix" {
    const M2 = Matrix(Complex, 2, 2);
    // [1+i  0]
    // [0    1-i]
    const m = M2.init([_][2]Complex{
        .{ Complex.init(1, 1), Complex.init(0, 0) },
        .{ Complex.init(0, 0), Complex.init(1, -1) },
    });
    // |1+i|^2 + |0|^2 + |0|^2 + |1-i|^2 = 2 + 0 + 0 + 2 = 4
    // ||m|| = sqrt(4) = 2
    try utils.expectApproxEq(m.frobeniusNorm(), 2.0);
}

// =============================================================================
// Scalar matrix exponential tests
// =============================================================================

test "exp of scalar diagonal matrix" {
    const M2 = Matrix(f64, 2, 2);
    // [a 0; 0 b] -> exp = [exp(a) 0; 0 exp(b)]
    const m = M2.init([_][2]f64{
        .{ 1.0, 0.0 },
        .{ 0.0, 2.0 },
    });

    const result = m.exp();

    try utils.expectApproxEq(result.data[0][0], std.math.e);
    try utils.expectApproxEq(result.data[0][1], 0.0);
    try utils.expectApproxEq(result.data[1][0], 0.0);
    try utils.expectApproxEq(result.data[1][1], std.math.e * std.math.e);
}

test "exp of nilpotent matrix" {
    const M2 = Matrix(f64, 2, 2);
    // [0 1; 0 0] is nilpotent (N^2 = 0)
    // exp(N) = I + N = [1 1; 0 1]
    const n = M2.init([_][2]f64{
        .{ 0.0, 1.0 },
        .{ 0.0, 0.0 },
    });

    const result = n.exp();

    try utils.expectApproxEq(result.data[0][0], 1.0);
    try utils.expectApproxEq(result.data[0][1], 1.0);
    try utils.expectApproxEq(result.data[1][0], 0.0);
    try utils.expectApproxEq(result.data[1][1], 1.0);
}

// =============================================================================
// Edge case tests for larger matrices
// =============================================================================

test "exp(zero matrix) equals identity - scalar 4x4" {
    const M4 = Matrix(f64, 4, 4);
    const zero = M4.zero();
    const result = zero.exp();
    const identity = M4.identity();

    for (0..4) |i| {
        for (0..4) |j| {
            try utils.expectApproxEq(result.data[i][j], identity.data[i][j]);
        }
    }
}

test "exp(zero matrix) equals identity - complex 4x4" {
    const M4 = Matrix(Complex, 4, 4);
    const zero = M4.zero();
    const result = zero.exp();
    const identity = M4.identity();

    for (0..4) |i| {
        for (0..4) |j| {
            try utils.expectComplexApproxEq(result.data[i][j], identity.data[i][j]);
        }
    }
}

test "exp of large norm matrix triggers scaling-and-squaring" {
    const M2 = Matrix(f64, 2, 2);
    // Matrix with large norm (> 0.5 threshold) to test scaling-and-squaring
    const large = M2.init([_][2]f64{
        .{ 5.0, 0.0 },
        .{ 0.0, 3.0 },
    });

    const result = large.exp();

    // exp([5 0; 0 3]) = [e^5 0; 0 e^3]
    try utils.expectApproxEq(result.data[0][0], @exp(5.0));
    try utils.expectApproxEq(result.data[0][1], 0.0);
    try utils.expectApproxEq(result.data[1][0], 0.0);
    try utils.expectApproxEq(result.data[1][1], @exp(3.0));
}

test "exp of very small matrix elements" {
    const M2 = Matrix(f64, 2, 2);
    // Very small values should still work correctly
    const tiny = M2.init([_][2]f64{
        .{ 1e-10, 0.0 },
        .{ 0.0, 1e-10 },
    });

    const result = tiny.exp();

    // exp(tiny) ≈ I + tiny for very small values
    try utils.expectApproxEq(result.data[0][0], @exp(1e-10));
    try utils.expectApproxEq(result.data[0][1], 0.0);
    try utils.expectApproxEq(result.data[1][0], 0.0);
    try utils.expectApproxEq(result.data[1][1], @exp(1e-10));
}

const std = @import("std");
const su_n = @import("su_n");
const Su2 = su_n.gauge.su2.Su2;
const Complex = std.math.Complex(f64);
const utils = @import("../test_utils.zig");

test "SU(2) Generators Traceless" {
    const s1 = Su2.sigma1();
    const s2 = Su2.sigma2();
    const s3 = Su2.sigma3();

    try utils.expectComplexApproxEq(s1.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(s2.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(s3.trace(), Complex.init(0, 0));
}

test "SU(2) Commutation Relations" {
    // [Si, Sj] = 2i * epsilon_ijk * Sk
    // Actually Pauli matrices satisfy [sigma_a, sigma_b] = 2i * epsilon_abc * sigma_c
    // Let's check [s1, s2] = 2i * s3
    
    const s1 = Su2.sigma1();
    const s2 = Su2.sigma2();
    const s3 = Su2.sigma3();
    
    // s1*s2 - s2*s1
    const comm12 = s1.mul(s2).sub(s2.mul(s1));
    
    // 2i * s3
    // 2i = 0 + 2i
    const two_i = Complex.init(0, 2);
    
    // Construct expected 2i*s3 manually or add scalar mul support later
    // s3 is diag(1, -1)
    // 2i*s3 = diag(2i, -2i)
    var expected = Su2.Matrix2x2.zero();
    expected.data[0][0] = s3.data[0][0].mul(two_i);
    expected.data[1][1] = s3.data[1][1].mul(two_i);
    
    for (0..2) |r| {
        for (0..2) |c| {
            try utils.expectComplexApproxEq(comm12.data[r][c], expected.data[r][c]);
        }
    }
}

test "SU(2) Unitarity" {
    // Sigma matrices are unitary: U * U^dag = I
    // And Hermitian: U = U^dag
    // So U * U = I
    const s2 = Su2.sigma2();
    const adj = s2.adjoint();
    
    // Check Hermitian
    for (0..2) |r| {
        for (0..2) |c| {
            try utils.expectComplexApproxEq(s2.data[r][c], adj.data[r][c]);
        }
    }
    
    // Check Unitary
    const prod = s2.mul(adj);
    const id = Su2.identity();
    
    for (0..2) |r| {
        for (0..2) |c| {
            try utils.expectComplexApproxEq(prod.data[r][c], id.data[r][c]);
        }
    }
}

test "SU(2) Determinant" {
    // Det(sigma) = -1
    // Wait, Pauli matrices have det -1.
    // SU(2) GROUP elements have det 1. Generators are NOT in the group, they are in the algebra su(2).
    // Generators are traceless Hermitian. Group elements are Unitary det=1.
    // Exponentials of generators are in the group. exp(i * theta * sigma)
    
    const s1 = Su2.sigma1();
    try utils.expectComplexApproxEq(s1.det(), Complex.init(-1, 0));
    
    const s2 = Su2.sigma2();
    try utils.expectComplexApproxEq(s2.det(), Complex.init(-1, 0));
}
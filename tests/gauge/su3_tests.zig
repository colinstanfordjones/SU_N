const std = @import("std");
const su_n = @import("su_n");
const Su3 = su_n.gauge.su3.Su3;
const Complex = std.math.Complex(f64);
const utils = @import("../test_utils.zig");

test "SU(3) Generators Traceless" {
    const l1 = Su3.lambda1();
    const l2 = Su3.lambda2();
    const l3 = Su3.lambda3();
    const l4 = Su3.lambda4();
    const l5 = Su3.lambda5();
    const l6 = Su3.lambda6();
    const l7 = Su3.lambda7();
    const l8 = Su3.lambda8();

    try utils.expectComplexApproxEq(l1.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l2.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l3.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l4.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l5.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l6.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l7.trace(), Complex.init(0, 0));
    try utils.expectComplexApproxEq(l8.trace(), Complex.init(0, 0));
}

test "SU(3) Generators Hermitian" {
    // lambda^dag = lambda
    const l2 = Su3.lambda2();
    const adj2 = l2.adjoint();
    
    for (0..3) |r| {
        for (0..3) |c| {
            try utils.expectComplexApproxEq(l2.data[r][c], adj2.data[r][c]);
        }
    }

    const l5 = Su3.lambda5();
    const adj5 = l5.adjoint();
    for (0..3) |r| {
        for (0..3) |c| {
            try utils.expectComplexApproxEq(l5.data[r][c], adj5.data[r][c]);
        }
    }
}

test "SU(3) Orthogonality" {
    // Tr(lambda_a * lambda_b) = 2 * delta_ab
    const l1 = Su3.lambda1();
    const l2 = Su3.lambda2();
    const l3 = Su3.lambda3();
    const l8 = Su3.lambda8();

    // Tr(l1 * l1) = 2
    const sq1 = l1.mul(l1);
    try utils.expectComplexApproxEq(sq1.trace(), Complex.init(2, 0));

    // Tr(l1 * l2) = 0
    const prod12 = l1.mul(l2);
    try utils.expectComplexApproxEq(prod12.trace(), Complex.init(0, 0));

    // Tr(l8 * l8) = 2
    const sq8 = l8.mul(l8);
    try utils.expectComplexApproxEq(sq8.trace(), Complex.init(2, 0));
    
    // Tr(l3 * l8) = 0
    const prod38 = l3.mul(l8);
    try utils.expectComplexApproxEq(prod38.trace(), Complex.init(0, 0));
}

test "SU(3) Structure Constant f123" {
    // [lambda_1, lambda_2] = 2i * f_123 * lambda_3
    // f_123 = 1
    // So [l1, l2] = 2i * l3
    const l1 = Su3.lambda1();
    const l2 = Su3.lambda2();
    const l3 = Su3.lambda3();

    const comm = l1.mul(l2).sub(l2.mul(l1));
    
    const two_i = Complex.init(0, 2);
    
    // Construct 2i * l3
    var expected = Su3.Matrix3x3.zero();
    for(0..3) |r| {
        for(0..3) |c| {
            expected.data[r][c] = l3.data[r][c].mul(two_i);
        }
    }

    for (0..3) |r| {
        for (0..3) |c| {
            try utils.expectComplexApproxEq(comm.data[r][c], expected.data[r][c]);
        }
    }
}

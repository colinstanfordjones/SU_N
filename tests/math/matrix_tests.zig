const std = @import("std");
const su_n = @import("su_n");
const Matrix = su_n.math.Matrix;
const Complex = std.math.Complex(f64);

test "matrix scalar add/sub/mul" {
    const M2 = Matrix(f64, 2, 2);
    const m1 = M2.init([_][2]f64{.{1, 2}, .{3, 4}});
    const m2 = M2.init([_][2]f64{.{5, 6}, .{7, 8}});
    
    // Add
    const sum = m1.add(m2);
    try std.testing.expectEqual(sum.data[0][0], 6);
    try std.testing.expectEqual(sum.data[1][1], 12);

    // Sub
    const diff = m2.sub(m1);
    try std.testing.expectEqual(diff.data[0][0], 4);
    
    // Mul
    // [1 2] * [5 6] = [1*5+2*7 1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7 3*6+4*8]   [43 50]
    const prod = m1.mul(m2);
    try std.testing.expectEqual(prod.data[0][0], 19);
    try std.testing.expectEqual(prod.data[1][1], 50);
}

test "matrix complex operations" {
    const M2 = Matrix(Complex, 2, 2);
    
    // Pauli Sigma Y
    // [0 -i]
    // [i  0]
    const sy = M2.init([_][2]Complex{
        .{ Complex.init(0, 0), Complex.init(0, -1) },
        .{ Complex.init(0, 1), Complex.init(0, 0) },
    });
    
    // sy * sy = I
    const sysy = sy.mul(sy);
    try std.testing.expectEqual(sysy.data[0][0].re, 1);
    try std.testing.expectEqual(sysy.data[0][1].re, 0);
    try std.testing.expectEqual(sysy.data[0][1].im, 0);
}

test "matrix determinant" {
    const M2 = Matrix(f64, 2, 2);
    // [1 2]
    // [3 4] det = 4 - 6 = -2
    const m = M2.init([_][2]f64{.{1, 2}, .{3, 4}});
    try std.testing.expectEqual(m.det(), -2);
    
    // Identity
    const id = M2.identity();
    try std.testing.expectEqual(id.det(), 1);
}

test "matrix trace" {
    const M2 = Matrix(f64, 2, 2);
    const m = M2.init([_][2]f64{.{1, 2}, .{3, 4}});
    // Trace = 1 + 4 = 5
    try std.testing.expectEqual(m.trace(), 5);
}

test "matrix adjoint" {
    const M2 = Matrix(Complex, 2, 2);
    // [1  i]
    // [-i 2]
    const m = M2.init([_][2]Complex{
        .{ Complex.init(1, 0), Complex.init(0, 1) },
        .{ Complex.init(0, -1), Complex.init(2, 0) },
    });
    
    // Adjoint should be transpose + conjugate
    // Transpose:
    // [1 -i]
    // [i  2]
    // Conjugate:
    // [1  i]
    // [-i 2]
    // So this matrix is Hermitian (Self-Adjoint)
    const adj = m.adjoint();
    
    try std.testing.expectEqual(adj.data[0][1].im, 1);
    try std.testing.expectEqual(adj.data[1][0].im, -1);
}

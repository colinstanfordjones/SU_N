const std = @import("std");
const math = @import("math");
const Complex = std.math.Complex(f64);

pub const Su2 = struct {
    pub const Matrix2x2 = math.Matrix(Complex, 2, 2);

    pub fn identity() Matrix2x2 {
        return Matrix2x2.init([2][2]Complex{
            .{ Complex.init(1, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(1, 0) },
        });
    }

    pub fn sigma1() Matrix2x2 {
        return Matrix2x2.init([2][2]Complex{
            .{ Complex.init(0, 0), Complex.init(1, 0) },
            .{ Complex.init(1, 0), Complex.init(0, 0) },
        });
    }

    pub fn sigma2() Matrix2x2 {
        return Matrix2x2.init([2][2]Complex{
            .{ Complex.init(0, 0), Complex.init(0, -1) },
            .{ Complex.init(0, 1), Complex.init(0, 0) },
        });
    }

    pub fn sigma3() Matrix2x2 {
        return Matrix2x2.init([2][2]Complex{
            .{ Complex.init(1, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(-1, 0) },
        });
    }
};

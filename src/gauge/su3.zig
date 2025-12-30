const std = @import("std");
const math = @import("math");
const constants = @import("constants");
const Complex = std.math.Complex(f64);

pub const Su3 = struct {
    pub const Matrix3x3 = math.Matrix(Complex, 3, 3);

    pub fn identity() Matrix3x3 {
        return Matrix3x3.identity();
    }

    pub fn lambda1() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(0, 0), Complex.init(1, 0), Complex.init(0, 0) },
            .{ Complex.init(1, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
        });
    }

    pub fn lambda2() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(0, 0), Complex.init(0, -1), Complex.init(0, 0) },
            .{ Complex.init(0, 1), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
        });
    }

    pub fn lambda3() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(1, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(-1, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
        });
    }

    pub fn lambda4() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(1, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(1, 0), Complex.init(0, 0), Complex.init(0, 0) },
        });
    }

    pub fn lambda5() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, -1) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 1), Complex.init(0, 0), Complex.init(0, 0) },
        });
    }

    pub fn lambda6() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(1, 0) },
            .{ Complex.init(0, 0), Complex.init(1, 0), Complex.init(0, 0) },
        });
    }

    pub fn lambda7() Matrix3x3 {
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, -1) },
            .{ Complex.init(0, 0), Complex.init(0, 1), Complex.init(0, 0) },
        });
    }

    pub fn lambda8() Matrix3x3 {
        const inv_sqrt3 = constants.inv_sqrt3;
        return Matrix3x3.init([3][3]Complex{
            .{ Complex.init(inv_sqrt3, 0), Complex.init(0, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(inv_sqrt3, 0), Complex.init(0, 0) },
            .{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(-2.0 * inv_sqrt3, 0) },
        });
    }
};

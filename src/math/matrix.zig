const std = @import("std");

pub fn Matrix(comptime T: type, comptime rows_in: usize, comptime cols_in: usize) type {
    const type_info = @typeInfo(T);
    const is_struct_type = (std.meta.activeTag(type_info) == .@"struct");
    
    // Define properties at type level
    const is_complex_type = (is_struct_type and @hasDecl(T, "re") and @hasDecl(T, "im"));
    const is_scalar_type = (std.meta.activeTag(type_info) == .float or std.meta.activeTag(type_info) == .int);

    return struct {
        pub const rows = rows_in;
        pub const cols = cols_in;
        data: [rows][cols]T,

        const Self = @This();

        pub fn init(data: [rows][cols]T) Self {
            return .{ .data = data };
        }

        pub fn identity() Self {
            var m: Self = undefined;
            const zero_val = std.mem.zeroes(T);
            
            var one_val: T = undefined;
            if (is_struct_type) {
                 if (@hasDecl(T, "init")) {
                    one_val = T.init(1, 0);
                 } else {
                    one_val = zero_val;
                 }
            } else {
                 one_val = 1;
            }

            for (0..rows) |i| {
                for (0..cols) |j| {
                    m.data[i][j] = if (i == j) one_val else zero_val;
                }
            }
            return m;
        }

        pub fn zero() Self {
            var m: Self = undefined;
            const z = std.mem.zeroes(T);
            for (0..rows) |i| {
                for (0..cols) |j| {
                    m.data[i][j] = z;
                }
            }
            return m;
        }

        fn applySimdBinaryOp(comptime op: enum { add, sub }, self: Self, other: Self) ?Self {
            if (comptime is_scalar_type) {
                 const vec_len = rows * cols;
                 const VecType = @Vector(vec_len, T);
                 const v_self: VecType = @bitCast(self.data);
                 const v_other: VecType = @bitCast(other.data);
                 
                 const v_res = switch (op) {
                     .add => v_self + v_other,
                     .sub => v_self - v_other,
                 };
                 return .{ .data = @bitCast(v_res) };
            } 
            
            if (comptime is_complex_type) {
                 const fields = std.meta.fields(T);
                 const BaseFloat = fields[0].type; 
                 
                 const vec_len = rows * cols * 2;
                 const VecType = @Vector(vec_len, BaseFloat);
                 const v_self: VecType = @bitCast(self.data);
                 const v_other: VecType = @bitCast(other.data);
                 
                 const v_res = switch (op) {
                     .add => v_self + v_other,
                     .sub => v_self - v_other,
                 };
                 return .{ .data = @bitCast(v_res) };
            }
            return null;
        }

        pub fn add(self: Self, other: Self) Self {
            if (applySimdBinaryOp(.add, self, other)) |res| return res;

            // Fallback
            var result: Self = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                     if (is_struct_type) {
                         result.data[i][j] = self.data[i][j].add(other.data[i][j]);
                     } else {
                         result.data[i][j] = self.data[i][j] + other.data[i][j];
                     }
                }
            }
            return result;
        }

        pub fn sub(self: Self, other: Self) Self {
            if (applySimdBinaryOp(.sub, self, other)) |res| return res;

            var result: Self = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                     if (is_struct_type) {
                         result.data[i][j] = self.data[i][j].sub(other.data[i][j]);
                     } else {
                         result.data[i][j] = self.data[i][j] - other.data[i][j];
                     }
                }
            }
            return result;
        }
        
        /// Matrix multiplication with SIMD optimization for scalar types.
        /// For scalar matrices with power-of-2 inner dimension, uses vectorized
        /// dot products with @reduce. For complex matrices, uses cache-optimized
        /// transpose access pattern. Falls back to scalar loop for other cases.
        pub fn mul(self: Self, other: anytype) Matrix(T, rows, @TypeOf(other).cols) {
            const OtherT = @TypeOf(other);
            const other_cols = OtherT.cols;
            // Ensure inner dimensions match
            if (cols != OtherT.rows) {
                @compileError("Matrix dimension mismatch in multiplication");
            }

            var result: Matrix(T, rows, other_cols) = undefined;

            // SIMD optimization for scalar types
            if (comptime is_scalar_type) {
                // Transpose other matrix for cache-friendly row-wise access
                const other_t = other.transpose();

                // Use SIMD when inner dimension is power of 2
                const is_power_of_2 = cols > 0 and (cols & (cols - 1)) == 0;

                if (comptime is_power_of_2) {
                    // Vectorized dot product using SIMD
                    const VecType = @Vector(cols, T);

                    for (0..rows) |i| {
                        const a_vec: VecType = self.data[i];
                        for (0..other_cols) |j| {
                            const b_vec: VecType = other_t.data[j];
                            result.data[i][j] = @reduce(.Add, a_vec * b_vec);
                        }
                    }
                } else {
                    // Non-power-of-2: scalar loop with transposed access for cache efficiency
                    for (0..rows) |i| {
                        for (0..other_cols) |j| {
                            var sum: T = 0;
                            for (0..cols) |k| {
                                sum += self.data[i][k] * other_t.data[j][k];
                            }
                            result.data[i][j] = sum;
                        }
                    }
                }
                return result;
            }

            // Complex type optimization: use transpose for cache-friendly access
            // Complex multiplication doesn't vectorize as cleanly, but transposed
            // access pattern still improves cache utilization
            if (comptime is_complex_type) {
                const other_t = other.transpose();
                const fields = std.meta.fields(T);
                const BaseFloat = fields[0].type;

                // For power-of-2 dimensions, vectorize the underlying float operations
                const is_power_of_2 = cols > 0 and (cols & (cols - 1)) == 0;

                if (comptime is_power_of_2 and cols >= 2) {
                    const VecType = @Vector(cols, BaseFloat);

                    for (0..rows) |i| {
                        // Extract real and imaginary parts as vectors
                        var a_re: [cols]BaseFloat = undefined;
                        var a_im: [cols]BaseFloat = undefined;
                        for (0..cols) |k| {
                            a_re[k] = self.data[i][k].re;
                            a_im[k] = self.data[i][k].im;
                        }
                        const a_re_vec: VecType = a_re;
                        const a_im_vec: VecType = a_im;

                        for (0..other_cols) |j| {
                            var b_re: [cols]BaseFloat = undefined;
                            var b_im: [cols]BaseFloat = undefined;
                            for (0..cols) |k| {
                                b_re[k] = other_t.data[j][k].re;
                                b_im[k] = other_t.data[j][k].im;
                            }
                            const b_re_vec: VecType = b_re;
                            const b_im_vec: VecType = b_im;

                            // Complex dot product: (a.re*b.re - a.im*b.im) + i(a.re*b.im + a.im*b.re)
                            const re_re = @reduce(.Add, a_re_vec * b_re_vec);
                            const im_im = @reduce(.Add, a_im_vec * b_im_vec);
                            const re_im = @reduce(.Add, a_re_vec * b_im_vec);
                            const im_re = @reduce(.Add, a_im_vec * b_re_vec);

                            result.data[i][j] = T.init(re_re - im_im, re_im + im_re);
                        }
                    }
                } else {
                    // Scalar fallback with transposed access
                    for (0..rows) |i| {
                        for (0..other_cols) |j| {
                            var sum: T = std.mem.zeroes(T);
                            for (0..cols) |k| {
                                sum = sum.add(self.data[i][k].mul(other_t.data[j][k]));
                            }
                            result.data[i][j] = sum;
                        }
                    }
                }
                return result;
            }

            // Fallback for other struct types (non-complex)
            for (0..rows) |i| {
                for (0..other_cols) |j| {
                    var sum: T = std.mem.zeroes(T);
                    for (0..cols) |k| {
                        const a = self.data[i][k];
                        const b = other.data[k][j];
                        sum = sum.add(a.mul(b));
                    }
                    result.data[i][j] = sum;
                }
            }
            return result;
        }

        pub fn transpose(self: Self) Matrix(T, cols, rows) {
            var result: Matrix(T, cols, rows) = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    result.data[j][i] = self.data[i][j];
                }
            }
            return result;
        }

        pub fn conjugate(self: Self) Self {
            var result: Self = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    if (is_struct_type and @hasDecl(T, "conjugate")) {
                        result.data[i][j] = self.data[i][j].conjugate();
                    } else {
                        // For real scalars, conjugate is identity
                        result.data[i][j] = self.data[i][j];
                    }
                }
            }
            return result;
        }

        pub fn adjoint(self: Self) Matrix(T, cols, rows) {
            var result: Matrix(T, cols, rows) = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    const val = self.data[i][j];
                    if (is_struct_type and @hasDecl(T, "conjugate")) {
                        result.data[j][i] = val.conjugate();
                    } else {
                        result.data[j][i] = val;
                    }
                }
            }
            return result;
        }

        pub fn trace(self: Self) T {
            if (rows != cols) @compileError("Trace only defined for square matrices");
            var sum: T = std.mem.zeroes(T);
            for (0..rows) |i| {
                if (is_struct_type) {
                     sum = sum.add(self.data[i][i]);
                } else {
                     sum += self.data[i][i];
                }
            }
            return sum;
        }

        pub fn det(self: Self) T {
            if (rows != cols) @compileError("Determinant only defined for square matrices");

            if (rows == 1) {
                // 1x1: det = single element
                return self.data[0][0];
            } else if (rows == 2) {
                const a = self.data[0][0];
                const b = self.data[0][1];
                const c = self.data[1][0];
                const d = self.data[1][1];

                if (is_struct_type) {
                    // ad - bc
                    return a.mul(d).sub(b.mul(c));
                } else {
                    return a * d - b * c;
                }
            } else if (rows == 3) {
                // Rule of Sarrus or expansion
                const a = self.data[0][0];
                const b = self.data[0][1];
                const c = self.data[0][2];
                const d = self.data[1][0];
                const e = self.data[1][1];
                const f = self.data[1][2];
                const g = self.data[2][0];
                const h = self.data[2][1];
                const i = self.data[2][2];

                if (is_struct_type) {
                    // a(ei - fh) - b(di - fg) + c(dh - eg)
                    const ei_fh = e.mul(i).sub(f.mul(h));
                    const di_fg = d.mul(i).sub(f.mul(g));
                    const dh_eg = d.mul(h).sub(e.mul(g));

                    return a.mul(ei_fh).sub(b.mul(di_fg)).add(c.mul(dh_eg));
                } else {
                    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
                }
            } else {
                @compileError("Determinant for >3x3 not implemented yet");
            }
        }


        pub fn scale(self: Self, scalar: anytype) Self {
            var result: Self = undefined;

            const ScaleMode = enum { scalar, struct_direct, struct_convert, invalid };
            const mode: ScaleMode = if (!is_struct_type) .scalar
                         else if (@TypeOf(scalar) == T) .struct_direct
                         else if (@hasDecl(T, "mul") and @hasDecl(T, "init")) .struct_convert
                         else .invalid;

            for (0..rows) |i| {
                for (0..cols) |j| {
                    switch (mode) {
                        .scalar => result.data[i][j] = self.data[i][j] * scalar,
                        .struct_direct => result.data[i][j] = self.data[i][j].mul(scalar),
                        .struct_convert => {
                            // Assumes complex-like structure with init(re, im)
                            const s = T.init(scalar, 0); 
                            result.data[i][j] = self.data[i][j].mul(s);
                        },
                        .invalid => @compileError("Cannot scale struct type without mul/init"),
                    }
                }
            }
            return result;
        }

        /// Compute the Frobenius norm: sqrt(sum of |a_ij|^2)
        pub fn frobeniusNorm(self: Self) f64 {
            var sum: f64 = 0;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    const val = self.data[i][j];
                    if (is_complex_type) {
                        sum += val.re * val.re + val.im * val.im;
                    } else if (is_struct_type) {
                        // For other struct types, try to get magnitude
                        if (@hasDecl(T, "magnitude")) {
                            const mag = val.magnitude();
                            sum += mag * mag;
                        } else {
                            @compileError("Struct type needs magnitude() for frobeniusNorm");
                        }
                    } else {
                        // Scalar type
                        const f: f64 = @floatCast(val);
                        sum += f * f;
                    }
                }
            }
            return std.math.sqrt(sum);
        }

        /// Matrix exponential via Taylor series with scaling-and-squaring.
        /// exp(A) = I + A + A²/2! + A³/3! + ...
        ///
        /// Uses scaling-and-squaring for numerical stability with large norms:
        /// 1. Find k such that ||A/2^k|| < threshold
        /// 2. Compute exp(A/2^k) via Taylor series
        /// 3. Square the result k times: exp(A) = (exp(A/2^k))^(2^k)
        ///
        /// Uses f64 precision with careful numerical bounds.
        ///
        /// Only defined for square matrices.
        ///
        /// ## Limitations
        ///
        /// - **Ill-conditioned matrices**: For matrices with condition number > 1e12
        ///   (eigenvalue ratio), consider eigendecomposition methods instead.
        ///   The squaring phase can amplify rounding errors for such matrices.
        ///
        /// - **Norm bounds**: Assumes ||A|| < 1e18 (f64 overflow threshold).
        ///   Matrices exceeding this will hit the 60-squaring safety limit.
        ///
        /// - **Structured matrices**: For diagonal, block-diagonal, or other
        ///   structured matrices with known closed-form exponentials, specialized
        ///   methods may be more efficient and accurate.
        ///
        /// - **Defective matrices**: Matrices that are not diagonalizable may
        ///   require Jordan decomposition for reliable results at extreme scales.
        ///
        /// ## Future Improvements
        ///
        /// For production use at scale, consider implementing:
        /// - Padé approximants (better convergence than Taylor for same order)
        /// - Eigendecomposition path for normal matrices
        /// - Schur decomposition for general matrices (LAPACK's matrix_exp approach)
        pub fn exp(self: Self) Self {
            if (rows != cols) @compileError("exp() only defined for square matrices");

            // Algorithm parameters for matrix exponential (inline to avoid cross-module deps)
            const matrix_exp_scaling_threshold: f64 = 0.5;
            const matrix_exp_max_terms: usize = 100;
            const matrix_exp_epsilon: f64 = 1e-14;

            // Scaling phase: find k such that ||A/2^k|| < threshold
            var scaled_matrix = self;
            var num_squarings: usize = 0;
            const threshold = matrix_exp_scaling_threshold;

            while (scaled_matrix.frobeniusNorm() > threshold) {
                scaled_matrix = scaled_matrix.scale(0.5);
                num_squarings += 1;

                // Safety limit to prevent infinite loop
                if (num_squarings > 60) break;
            }

            // Taylor series
            var result = Self.identity();
            var term = Self.identity();
            var inv_factorial: f64 = 1.0;

            const max_terms = matrix_exp_max_terms;
            const epsilon = matrix_exp_epsilon;

            for (1..max_terms) |n| {
                // term = term * A (gives A^n for scaled matrix)
                term = term.mul(scaled_matrix);
                inv_factorial /= @as(f64, @floatFromInt(n));

                // scaled_term = A^n / n!
                const scaled_term = term.scale(inv_factorial);

                result = result.add(scaled_term);

                // Check for convergence
                if (scaled_term.frobeniusNorm() < epsilon) break;
            }

            // Squaring phase: square result num_squarings times
            // This recovers exp(A) from exp(A/2^k)
            for (0..num_squarings) |_| {
                result = result.mul(result);
            }

            return result;
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.writeAll("Matrix{\n");
            for (self.data) |row| {
                try writer.writeAll("  ");
                try writer.print("{any}\n", .{row});
            }
            try writer.writeAll("}");
        }
    };
}

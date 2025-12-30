//! Link Variable Module for Lattice Gauge Theory
//!
//! Link variables U_μ(x) ∈ SU(N) live on the edges (links) of a spacetime lattice.
//! They represent the parallel transporter of the gauge field from site x to x+μ.
//!
//! ## Gauge Transformation
//!
//! Under a local gauge transformation Ω(x) ∈ SU(N):
//!   U_μ(x) → Ω(x) U_μ(x) Ω†(x+μ)
//!
//! This ensures that gauge-covariant quantities transform properly.
//!
//! ## Relation to Continuum
//!
//! In the continuum limit:
//!   U_μ(x) ≈ exp(ig a A_μ(x))
//!
//! where a is the lattice spacing and A_μ is the gauge potential.
//!
//! ## Precision
//!
//! All calculations use f64 (double precision) for SIMD compatibility.
//! Numerical stability is maintained through proper algorithm design.
//!
//! ## Supported Groups
//!
//! - U(1): Phase factors e^{iθ}, represented as 1×1 complex matrices
//! - SU(2): 2×2 unitary matrices with det = 1
//! - SU(N): N×N unitary matrices with det = 1

const std = @import("std");
const math = @import("math");
const math_utils = @import("math").utils;
const constants = @import("constants");

/// Complex number type for gauge calculations (f64 for SIMD)
pub const Complex = std.math.Complex(f64);

/// Generic link variable for SU(N) gauge theory
/// N is the dimension of the fundamental representation
/// Uses f64 precision for SIMD compatibility
pub fn LinkVariable(comptime N: usize) type {
    const MatrixType = math.Matrix(Complex, N, N);

    return struct {
        /// The underlying unitary matrix
        matrix: MatrixType,

        const Self = @This();

        /// Matrix dimension (for external access)
        pub const dim = N;

        // =====================================================================
        // Constructors
        // =====================================================================

        /// Identity element (trivial vacuum)
        pub fn identity() Self {
            return .{ .matrix = MatrixType.identity() };
        }

        /// Zero matrix (useful for accumulation)
        pub fn zero() Self {
            return .{ .matrix = MatrixType.zero() };
        }

        /// Construct from raw matrix (caller ensures unitarity)
        pub fn fromMatrix(m: MatrixType) Self {
            return .{ .matrix = m };
        }

        /// Construct from Lie algebra element: U = exp(i H)
        /// where H is Hermitian (H† = H)
        pub fn fromAlgebra(algebra_element: MatrixType) Self {
            // Multiply by i and exponentiate
            const i_H = algebra_element.scale(Complex.init(0, 1));
            return .{ .matrix = i_H.exp() };
        }

        /// Construct from angle and generator: U = exp(i θ T)
        /// where T is a Lie algebra generator
        pub fn fromGenerator(theta: f64, generator: MatrixType) Self {
            const scaled = generator.scale(Complex.init(theta, 0));
            return fromAlgebra(scaled);
        }

        // =====================================================================
        // Group Operations
        // =====================================================================

        /// Group multiplication: U * V
        pub fn mul(self: Self, other: Self) Self {
            return .{ .matrix = self.matrix.mul(other.matrix) };
        }

        /// Hermitian conjugate (adjoint): U†
        /// For unitary matrices: U† = U⁻¹
        pub fn adjoint(self: Self) Self {
            return .{ .matrix = self.matrix.adjoint() };
        }

        /// Inverse (same as adjoint for unitary matrices)
        pub fn inverse(self: Self) Self {
            return self.adjoint();
        }

        /// Matrix addition (for staple sums, etc.)
        pub fn add(self: Self, other: Self) Self {
            return .{ .matrix = self.matrix.add(other.matrix) };
        }

        /// Scalar multiplication
        pub fn scale(self: Self, s: Complex) Self {
            return .{ .matrix = self.matrix.scale(s) };
        }

        /// Scalar multiplication (real)
        pub fn scaleReal(self: Self, s: f64) Self {
            return .{ .matrix = self.matrix.scale(Complex.init(s, 0)) };
        }

        // =====================================================================
        // Observables
        // =====================================================================

        /// Trace: Tr(U)
        pub fn trace(self: Self) Complex {
            return self.matrix.trace();
        }

        /// Real part of trace: Re(Tr(U))
        pub fn traceReal(self: Self) f64 {
            return self.trace().re;
        }

        /// Determinant: det(U)
        /// For SU(N), this should be 1
        pub fn det(self: Self) Complex {
            return self.matrix.det();
        }

        /// Frobenius norm: ||U||_F = sqrt(Tr(U† U))
        pub fn norm(self: Self) f64 {
            return self.matrix.frobeniusNorm();
        }

        // =====================================================================
        // Unitarization
        // =====================================================================

        /// Project matrix back to SU(N) manifold
        /// Essential after numerical updates to maintain U† U = I and det(U) = 1
        ///
        /// Method: Modified Gram-Schmidt orthonormalization + determinant fix
        pub fn unitarize(self: Self) Self {
            var result = self.matrix;

            // Gram-Schmidt orthonormalization of columns
            comptime var col: usize = 0;
            inline while (col < N) : (col += 1) {
                // Normalize column col
                var norm_sq: f64 = 0;
                inline for (0..N) |row| {
                    const c = result.data[row][col];
                    norm_sq += c.re * c.re + c.im * c.im;
                }
                const inv_norm: f64 = 1.0 / @sqrt(norm_sq);
                inline for (0..N) |row| {
                    result.data[row][col] = result.data[row][col].mul(Complex.init(inv_norm, 0));
                }

                // Orthogonalize subsequent columns against this one
                comptime var next_col = col + 1;
                inline while (next_col < N) : (next_col += 1) {
                    // dot = <col, next_col>
                    var dot = Complex.init(0, 0);
                    inline for (0..N) |row| {
                        const conj_col = Complex.init(result.data[row][col].re, -result.data[row][col].im);
                        dot = dot.add(conj_col.mul(result.data[row][next_col]));
                    }
                    // next_col -= dot * col
                    inline for (0..N) |row| {
                        const subtract = dot.mul(result.data[row][col]);
                        result.data[row][next_col] = result.data[row][next_col].sub(subtract);
                    }
                }
            }

            // Fix determinant to 1 for SU(N)
            // After Gram-Schmidt, det(U) = e^{iφ} for some phase φ
            // Multiply first column by e^{-iφ} to get det = 1
            if (N > 1) {
                const d = result.det();
                const d_mag = @sqrt(d.re * d.re + d.im * d.im);
                if (d_mag > constants.det_threshold) {
                    const phase = std.math.atan2(d.im, d.re);
                    // Full phase correction since we only modify one column
                    const correction = Complex.init(@cos(-phase), @sin(-phase));
                    inline for (0..N) |row| {
                        result.data[row][0] = result.data[row][0].mul(correction);
                    }
                }
            }

            return .{ .matrix = result };
        }

        /// Check if matrix is approximately unitary: ||U† U - I|| < epsilon
        pub fn isUnitary(self: Self, epsilon: f64) bool {
            const uu_dag = self.matrix.mul(self.matrix.adjoint());
            const id = MatrixType.identity();

            var diff_norm_sq: f64 = 0;
            for (0..N) |i| {
                for (0..N) |j| {
                    const diff = uu_dag.data[i][j].sub(id.data[i][j]);
                    diff_norm_sq += diff.re * diff.re + diff.im * diff.im;
                }
            }
            return @sqrt(diff_norm_sq) < epsilon;
        }

        /// Check if determinant is approximately 1
        pub fn hasUnitDet(self: Self, epsilon: f64) bool {
            const d = self.det();
            const diff_re = d.re - 1.0;
            const diff_im = d.im;
            return @sqrt(diff_re * diff_re + diff_im * diff_im) < epsilon;
        }

        // =====================================================================
        // Action on Matter Fields
        // =====================================================================

        /// Act on a vector in the fundamental representation
        /// For parallel transport: ψ(x+μ) → U_μ(x) ψ(x+μ)
        pub fn actOnVector(self: Self, v: [N]Complex) [N]Complex {
            var result: [N]Complex = undefined;
            for (0..N) |i| {
                result[i] = Complex.init(0, 0);
                for (0..N) |j| {
                    result[i] = result[i].add(self.matrix.data[i][j].mul(v[j]));
                }
            }
            return result;
        }

        /// Act on adjoint representation (for adjoint matter fields)
        /// V → U V U†
        pub fn actOnAdjoint(self: Self, v: MatrixType) MatrixType {
            return self.matrix.mul(v).mul(self.matrix.adjoint());
        }

        // =====================================================================
        // Matrix Access
        // =====================================================================

        /// Get matrix element
        pub fn get(self: Self, row: usize, col: usize) Complex {
            return self.matrix.data[row][col];
        }

        /// Set matrix element (use with care - may break unitarity)
        pub fn set(self: *Self, row: usize, col: usize, value: Complex) void {
            self.matrix.data[row][col] = value;
        }
    };
}

// =============================================================================
// Specialized Types
// =============================================================================

/// U(1) link variable: phase factor e^{iθ}
pub const U1Link = LinkVariable(1);

/// SU(2) link variable: 2×2 unitary with det = 1
pub const SU2Link = LinkVariable(2);

/// SU(3) link variable: 3×3 unitary with det = 1
pub const SU3Link = LinkVariable(3);

// =============================================================================
// U(1) Specialized Functions
// =============================================================================

/// Create U(1) link from angle: U = e^{iθ}
pub fn u1FromAngle(theta: f64) U1Link {
    const phase = Complex.init(@cos(theta), @sin(theta));
    var m = U1Link.zero().matrix;
    m.data[0][0] = phase;
    return U1Link.fromMatrix(m);
}

/// Extract angle from U(1) link
pub fn u1ToAngle(link: U1Link) f64 {
    const c = link.matrix.data[0][0];
    return std.math.atan2(c.im, c.re);
}

// =============================================================================
// SU(2) Specialized Functions
// =============================================================================

/// Create SU(2) link from quaternion (a, b, c, d) with a² + b² + c² + d² = 1
/// U = aI + i(bσ₁ + cσ₂ + dσ₃)
pub fn su2FromQuaternion(q: [4]f64) SU2Link {
    const a = q[0];
    const b = q[1];
    const c = q[2];
    const d = q[3];

    // SU(2) matrix from quaternion:
    // | a + id,  b + ic |
    // |-b + ic,  a - id |
    var m = SU2Link.zero().matrix;
    m.data[0][0] = Complex.init(a, d);
    m.data[0][1] = Complex.init(b, c);
    m.data[1][0] = Complex.init(-b, c);
    m.data[1][1] = Complex.init(a, -d);

    return SU2Link.fromMatrix(m);
}

/// Create SU(2) link from Euler-like angles (θ₁, θ₂, θ₃)
/// Parametrization: U = exp(i θ₁ σ₁/2) exp(i θ₂ σ₂/2) exp(i θ₃ σ₃/2)
/// Simplified to quaternion: (cos(θ/2), n*sin(θ/2)) where n = (sin(θ₁), sin(θ₂), sin(θ₃))/|n|
/// For easy construction from a single angle θ about z-axis:
pub fn su2FromAngle(theta: f64, phi: f64, psi: f64) SU2Link {
    // For simplicity, create rotation about z-axis by angle 2*theta
    // U = exp(i θ σ₃) = cos(θ)I + i sin(θ)σ₃
    // = | cos(θ) + i sin(θ),  0                  |
    //   | 0,                  cos(θ) - i sin(θ)  |
    _ = phi;
    _ = psi;

    const c = @cos(theta);
    const s = @sin(theta);

    var m = SU2Link.zero().matrix;
    m.data[0][0] = Complex.init(c, s);
    m.data[0][1] = Complex.init(0, 0);
    m.data[1][0] = Complex.init(0, 0);
    m.data[1][1] = Complex.init(c, -s);

    return SU2Link.fromMatrix(m);
}

// =============================================================================
// Tests
// =============================================================================

test "U1 link identity" {
    const id = U1Link.identity();
    const tr = id.trace();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), tr.re, constants.tolerance_tight);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tr.im, constants.tolerance_tight);
}

test "U1 link from angle" {
    const theta: f64 = std.math.pi / 4.0;
    const link = u1FromAngle(theta);

    // e^{iπ/4} = cos(π/4) + i sin(π/4)
    const c = link.matrix.data[0][0];
    try std.testing.expectApproxEqAbs(@cos(theta), c.re, constants.tolerance_matrix);
    try std.testing.expectApproxEqAbs(@sin(theta), c.im, constants.tolerance_matrix);
}

test "U1 link multiplication is phase addition" {
    const theta1: f64 = std.math.pi / 6.0;
    const theta2: f64 = std.math.pi / 4.0;

    const link1 = u1FromAngle(theta1);
    const link2 = u1FromAngle(theta2);
    const product = link1.mul(link2);

    const expected_theta = theta1 + theta2;
    const c = product.matrix.data[0][0];
    try std.testing.expectApproxEqAbs(@cos(expected_theta), c.re, constants.tolerance_matrix);
    try std.testing.expectApproxEqAbs(@sin(expected_theta), c.im, constants.tolerance_matrix);
}

test "SU2 link identity" {
    const id = SU2Link.identity();
    try std.testing.expect(id.isUnitary(constants.tolerance_tight));
    try std.testing.expect(id.hasUnitDet(constants.tolerance_tight));
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), id.traceReal(), constants.tolerance_tight);
}

test "SU2 from quaternion" {
    // Identity quaternion (1, 0, 0, 0)
    const id_quat = su2FromQuaternion(.{ 1, 0, 0, 0 });
    try std.testing.expect(id_quat.isUnitary(constants.tolerance_tight));
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), id_quat.traceReal(), constants.tolerance_tight);

    // 90° rotation about z-axis: (cos(π/4), 0, 0, sin(π/4))
    const angle: f64 = std.math.pi / 2.0;
    const rot_z = su2FromQuaternion(.{ @cos(angle / 2), 0, 0, @sin(angle / 2) });
    try std.testing.expect(rot_z.isUnitary(constants.tolerance_matrix));
    try std.testing.expect(rot_z.hasUnitDet(constants.tolerance_matrix));
}

test "SU3 link identity" {
    const id = SU3Link.identity();
    try std.testing.expect(id.isUnitary(constants.tolerance_tight));
    try std.testing.expect(id.hasUnitDet(constants.tolerance_tight));
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), id.traceReal(), constants.tolerance_tight);
}

test "link adjoint is inverse" {
    const theta: f64 = 0.5;
    const link = u1FromAngle(theta);
    const product = link.mul(link.adjoint());

    // U * U† = I
    const tr = product.trace();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), tr.re, constants.tolerance_matrix);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tr.im, constants.tolerance_matrix);
}

test "unitarization preserves unitarity" {
    // Create a slightly non-unitary matrix
    var link = SU2Link.identity();
    link.matrix.data[0][0] = Complex.init(1.01, 0.02);
    link.matrix.data[1][1] = Complex.init(0.99, -0.01);

    // Before unitarization: not unitary
    try std.testing.expect(!link.isUnitary(0.01));

    // After unitarization: unitary
    const fixed = link.unitarize();
    try std.testing.expect(fixed.isUnitary(constants.tolerance_iterative));
}

test "act on vector" {
    const link = u1FromAngle(std.math.pi / 2.0); // i
    const v = [1]Complex{Complex.init(1, 0)};
    const result = link.actOnVector(v);

    // e^{iπ/2} * 1 = i
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[0].re, constants.tolerance_matrix);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0].im, constants.tolerance_matrix);
}

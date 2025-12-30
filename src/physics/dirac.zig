//! Dirac Spinor Module
//!
//! Implements Dirac spinors and gamma matrices built from the SU(2)⊗SU(2) structure
//! of the Lorentz group's double cover. The g=2 magnetic moment emerges naturally
//! from this structure when coupled to electromagnetism via minimal coupling.
//!
//! ## Symmetry Structure
//!
//! The Lorentz group SO(3,1) has double cover SL(2,C) ≃ SU(2)_L × SU(2)_R.
//! Dirac spinors combine left-handed (1/2, 0) and right-handed (0, 1/2) Weyl spinors
//! into a 4-component object that transforms reducibly under Lorentz transformations.
//!
//! ## Gamma Matrices
//!
//! The gamma matrices are constructed from Pauli matrices (SU(2) generators):
//!   γ⁰ = σ₃ ⊗ I₂  (Dirac representation)
//!   γⁱ = iσ₂ ⊗ σᵢ
//!
//! They satisfy the Clifford algebra: {γᵘ, γᵛ} = 2gᵘᵛ I₄
//!
//! ## Dirac Equation
//!
//! The free Dirac equation (iγᵘ∂ᵤ - m)ψ = 0 follows from requiring:
//! - Lorentz covariance
//! - First-order in time derivatives (for probability interpretation)
//! - Correct non-relativistic limit
//!
//! ## Magnetic Moment
//!
//! When minimally coupled to U(1) electromagnetism (∂ᵤ → Dᵤ = ∂ᵤ - ieAᵤ),
//! the non-relativistic limit gives:
//!   H = (p - eA)²/(2m) + eφ - (g/2)(eℏ/2m)σ·B
//!
//! where g = 2 emerges from the Dirac structure (not put in by hand).

const std = @import("std");
const math = @import("math");
const su2 = @import("gauge").su2;
const constants = @import("constants");
const Complex = std.math.Complex(f64);

/// 4-component Dirac spinor
/// Combines upper (left-handed) and lower (right-handed) 2-component Weyl spinors
pub const DiracSpinor = struct {
    /// 4-component spinor as column vector
    components: math.Matrix(Complex, 4, 1),

    const Self = @This();

    pub fn init(c0: Complex, c1: Complex, c2: Complex, c3: Complex) Self {
        return .{
            .components = math.Matrix(Complex, 4, 1).init([4][1]Complex{
                .{c0}, .{c1}, .{c2}, .{c3},
            }),
        };
    }

    pub fn zero() Self {
        return .{
            .components = math.Matrix(Complex, 4, 1).zero(),
        };
    }

    /// Upper 2-component (left-handed Weyl spinor in Dirac rep)
    pub fn upper(self: Self) math.Matrix(Complex, 2, 1) {
        return math.Matrix(Complex, 2, 1).init([2][1]Complex{
            .{self.components.data[0][0]},
            .{self.components.data[1][0]},
        });
    }

    /// Lower 2-component (right-handed Weyl spinor in Dirac rep)
    pub fn lower(self: Self) math.Matrix(Complex, 2, 1) {
        return math.Matrix(Complex, 2, 1).init([2][1]Complex{
            .{self.components.data[2][0]},
            .{self.components.data[3][0]},
        });
    }

    /// Dirac adjoint: ψ̄ = ψ†γ⁰
    pub fn bar(self: Self) math.Matrix(Complex, 1, 4) {
        const psi_dagger = self.components.adjoint();
        return psi_dagger.mul(GammaMatrices.gamma0());
    }

    /// Norm squared: ψ†ψ
    pub fn normSquared(self: Self) f64 {
        var sum: f64 = 0;
        inline for (0..4) |i| {
            const c = self.components.data[i][0];
            sum += c.re * c.re + c.im * c.im;
        }
        return sum;
    }

    /// Normalize the spinor
    pub fn normalize(self: Self) Self {
        const norm = @sqrt(self.normSquared());
        if (norm < 1e-15) return self;

        var result: Self = undefined;
        inline for (0..4) |i| {
            const c = self.components.data[i][0];
            result.components.data[i][0] = Complex.init(c.re / norm, c.im / norm);
        }
        return result;
    }
};

/// Gamma matrices in Dirac (standard) representation
/// Built from SU(2) Pauli matrices via tensor product
pub const GammaMatrices = struct {
    pub const Matrix4x4 = math.Matrix(Complex, 4, 4);
    const Matrix2x2 = su2.Su2.Matrix2x2;

    /// Identity 2x2
    fn I2() Matrix2x2 {
        return su2.Su2.identity();
    }

    /// Zero 2x2
    fn Z2() Matrix2x2 {
        return Matrix2x2.zero();
    }

    /// Tensor product of two 2x2 matrices: A ⊗ B
    /// Result is 4x4 with block structure
    fn tensorProduct(A: Matrix2x2, B: Matrix2x2) Matrix4x4 {
        var result = Matrix4x4.zero();

        // (A ⊗ B)_{(i,k),(j,l)} = A_{i,j} × B_{k,l}
        // Index mapping: row = 2*i + k, col = 2*j + l
        // Unroll loops for performance (small fixed size)
        inline for (0..2) |i| {
            inline for (0..2) |j| {
                const a_ij = A.data[i][j];
                inline for (0..2) |k| {
                    inline for (0..2) |l| {
                        const b_kl = B.data[k][l];
                        const row = 2 * i + k;
                        const col = 2 * j + l;
                        result.data[row][col] = a_ij.mul(b_kl);
                    }
                }
            }
        }
        return result;
    }

    /// γ⁰ = σ₃ ⊗ I₂ = diag(1, 1, -1, -1)
    /// This is the Dirac representation
    pub fn gamma0() Matrix4x4 {
        return tensorProduct(su2.Su2.sigma3(), I2());
    }

    /// γ¹ = iσ₂ ⊗ σ₁
    pub fn gamma1() Matrix4x4 {
        // iσ₂
        const i_sigma2 = su2.Su2.sigma2().scale(Complex.init(0, 1));
        return tensorProduct(i_sigma2, su2.Su2.sigma1());
    }

    /// γ² = iσ₂ ⊗ σ₂
    pub fn gamma2() Matrix4x4 {
        const i_sigma2 = su2.Su2.sigma2().scale(Complex.init(0, 1));
        return tensorProduct(i_sigma2, su2.Su2.sigma2());
    }

    /// γ³ = iσ₂ ⊗ σ₃
    pub fn gamma3() Matrix4x4 {
        const i_sigma2 = su2.Su2.sigma2().scale(Complex.init(0, 1));
        return tensorProduct(i_sigma2, su2.Su2.sigma3());
    }

    /// γ⁵ = iγ⁰γ¹γ²γ³ (chirality matrix)
    pub fn gamma5() Matrix4x4 {
        // In Dirac rep: γ⁵ = σ₁ ⊗ I₂
        return tensorProduct(su2.Su2.sigma1(), I2());
    }

    /// Get gamma matrix by index (0-3)
    pub fn gamma(mu: usize) Matrix4x4 {
        return switch (mu) {
            0 => gamma0(),
            1 => gamma1(),
            2 => gamma2(),
            3 => gamma3(),
            else => @panic("Invalid gamma matrix index"),
        };
    }

    /// Verify Clifford algebra: {γᵘ, γᵛ} = 2gᵘᵛ I₄
    /// Returns true if the anticommutator relation holds
    pub fn verifyClifford(mu: usize, nu: usize) bool {
        const g_mu = gamma(mu);
        const g_nu = gamma(nu);

        // {γᵘ, γᵛ} = γᵘγᵛ + γᵛγᵘ
        const anticomm = g_mu.mul(g_nu).add(g_nu.mul(g_mu));

        // Expected: 2gᵘᵛ I₄
        // Minkowski metric: g⁰⁰ = +1, gⁱⁱ = -1
        const metric_value: f64 = if (mu == nu) (if (mu == 0) 2.0 else -2.0) else 0.0;

        // Check diagonal elements (all diagonal entries should equal metric_value)
        const epsilon = constants.test_epsilon;
        inline for (0..4) |i| {
            const expected = metric_value; // Diagonal: always data[i][i]
            const actual = anticomm.data[i][i].re;
            if (@abs(actual - expected) > epsilon) return false;
            if (@abs(anticomm.data[i][i].im) > epsilon) return false;
        }

        // Check off-diagonal elements are zero
        inline for (0..4) |i| {
            inline for (0..4) |j| {
                if (i != j) {
                    if (@abs(anticomm.data[i][j].re) > epsilon) return false;
                    if (@abs(anticomm.data[i][j].im) > epsilon) return false;
                }
            }
        }

        return true;
    }

    /// Spin matrices Σⁱ = (1/2)εⁱʲᵏ σʲᵏ where σᵘᵛ = (i/2)[γᵘ, γᵛ]
    /// In Dirac rep: Σⁱ = diag(σⁱ, σⁱ) (block diagonal with Pauli matrices)
    pub fn spinMatrix(i: usize) Matrix4x4 {
        const sigma = switch (i) {
            1 => su2.Su2.sigma1(),
            2 => su2.Su2.sigma2(),
            3 => su2.Su2.sigma3(),
            else => @panic("Invalid spin matrix index (use 1, 2, or 3)"),
        };

        // Block diagonal: Σⁱ = diag(σⁱ, σⁱ)
        var result = Matrix4x4.zero();
        inline for (0..2) |r| {
            inline for (0..2) |c| {
                result.data[r][c] = sigma.data[r][c];
                result.data[r + 2][c + 2] = sigma.data[r][c];
            }
        }
        return result;
    }
};

/// Spin operator expectation values
/// These are the SU(2) generators acting on Dirac spinors
pub const SpinOperator = struct {
    /// Compute ⟨ψ|Σⁱ|ψ⟩ for spin component i (1, 2, or 3)
    pub fn expectation(psi: DiracSpinor, component: usize) f64 {
        const sigma_i = GammaMatrices.spinMatrix(component);
        const sigma_psi = sigma_i.mul(psi.components);

        // ⟨ψ|Σⁱ|ψ⟩ = ψ† Σⁱ ψ
        var result = Complex.init(0, 0);
        inline for (0..4) |r| {
            result = result.add(psi.components.data[r][0].conjugate().mul(sigma_psi.data[r][0]));
        }

        // Should be real for Hermitian operator
        return result.re;
    }

    /// Compute spin z-component: ⟨ψ|Σ³|ψ⟩
    /// For a normalized spinor, this gives 2×⟨S_z⟩ (since Σ = 2S/ℏ)
    pub fn spinZ(psi: DiracSpinor) f64 {
        return expectation(psi, 3);
    }

    /// Total spin squared: ⟨ψ|Σ²|ψ⟩ = ⟨Σ₁²⟩ + ⟨Σ₂²⟩ + ⟨Σ₃²⟩
    /// For spin-1/2: should give 3 (since s(s+1) = 3/4 and Σ = 2S)
    pub fn spinSquared(psi: DiracSpinor) f64 {
        var total: f64 = 0;
        inline for (1..4) |i| {
            const sigma_i = GammaMatrices.spinMatrix(i);
            const sigma_i_psi = sigma_i.mul(psi.components);
            const sigma_i_sq_psi = sigma_i.mul(sigma_i_psi);

            var exp_i: f64 = 0;
            inline for (0..4) |r| {
                const val = psi.components.data[r][0].conjugate().mul(sigma_i_sq_psi.data[r][0]);
                exp_i += val.re;
            }
            total += exp_i;
        }
        return total;
    }
};

/// Standard Dirac spinor basis states
pub const BasisSpinors = struct {
    /// Spin-up positive energy: u↑(p=0)
    /// Upper components: (1, 0), Lower: (0, 0) at rest
    pub fn spinUp() DiracSpinor {
        return DiracSpinor.init(
            Complex.init(1, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        );
    }

    /// Spin-down positive energy: u↓(p=0)
    /// Upper components: (0, 1), Lower: (0, 0) at rest
    pub fn spinDown() DiracSpinor {
        return DiracSpinor.init(
            Complex.init(0, 0),
            Complex.init(1, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        );
    }

    /// Spin-up negative energy (antiparticle): v↑(p=0)
    pub fn antiSpinUp() DiracSpinor {
        return DiracSpinor.init(
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(1, 0),
            Complex.init(0, 0),
        );
    }

    /// Spin-down negative energy (antiparticle): v↓(p=0)
    pub fn antiSpinDown() DiracSpinor {
        return DiracSpinor.init(
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(1, 0),
        );
    }
};

// =============================================================================
// Tests
// =============================================================================

test "gamma matrices satisfy Clifford algebra" {
    // {γᵘ, γᵛ} = 2gᵘᵛ I₄
    inline for (0..4) |mu| {
        inline for (0..4) |nu| {
            try std.testing.expect(GammaMatrices.verifyClifford(mu, nu));
        }
    }
}

test "gamma0 is Hermitian" {
    const g0 = GammaMatrices.gamma0();
    const g0_dag = g0.adjoint();

    inline for (0..4) |i| {
        inline for (0..4) |j| {
            try std.testing.expectApproxEqAbs(g0.data[i][j].re, g0_dag.data[i][j].re, constants.test_epsilon);
            try std.testing.expectApproxEqAbs(g0.data[i][j].im, g0_dag.data[i][j].im, constants.test_epsilon);
        }
    }
}

test "spatial gammas are anti-Hermitian" {
    // γⁱ† = -γⁱ for i = 1, 2, 3
    inline for (1..4) |i| {
        const gi = GammaMatrices.gamma(i);
        const gi_dag = gi.adjoint();
        const neg_gi = gi.scale(Complex.init(-1, 0));

        inline for (0..4) |r| {
            inline for (0..4) |c| {
                try std.testing.expectApproxEqAbs(neg_gi.data[r][c].re, gi_dag.data[r][c].re, constants.test_epsilon);
                try std.testing.expectApproxEqAbs(neg_gi.data[r][c].im, gi_dag.data[r][c].im, constants.test_epsilon);
            }
        }
    }
}

test "spin-up has Sz = +1 (in units where Σ = 2S)" {
    const psi_up = BasisSpinors.spinUp();
    const sz = SpinOperator.spinZ(psi_up);
    try std.testing.expectApproxEqAbs(1.0, sz, constants.test_epsilon);
}

test "spin-down has Sz = -1 (in units where Σ = 2S)" {
    const psi_down = BasisSpinors.spinDown();
    const sz = SpinOperator.spinZ(psi_down);
    try std.testing.expectApproxEqAbs(-1.0, sz, constants.test_epsilon);
}

test "spin-1/2 particles have S² = 3/4 (so Σ² = 3)" {
    const psi_up = BasisSpinors.spinUp();
    const s2 = SpinOperator.spinSquared(psi_up);
    // Σ² = 4S² = 4 × (3/4) = 3
    try std.testing.expectApproxEqAbs(3.0, s2, constants.test_epsilon);
}

test "gamma5 squares to identity" {
    const g5 = GammaMatrices.gamma5();
    const g5_sq = g5.mul(g5);
    const I4 = GammaMatrices.Matrix4x4.identity();

    inline for (0..4) |i| {
        inline for (0..4) |j| {
            try std.testing.expectApproxEqAbs(I4.data[i][j].re, g5_sq.data[i][j].re, constants.test_epsilon);
            try std.testing.expectApproxEqAbs(I4.data[i][j].im, g5_sq.data[i][j].im, constants.test_epsilon);
        }
    }
}

test "gamma5 anticommutes with all gamma matrices" {
    const g5 = GammaMatrices.gamma5();

    inline for (0..4) |mu| {
        const g_mu = GammaMatrices.gamma(mu);
        // {γ⁵, γᵘ} = γ⁵γᵘ + γᵘγ⁵ = 0
        const anticomm = g5.mul(g_mu).add(g_mu.mul(g5));

        inline for (0..4) |i| {
            inline for (0..4) |j| {
                try std.testing.expectApproxEqAbs(0.0, anticomm.data[i][j].re, constants.test_epsilon);
                try std.testing.expectApproxEqAbs(0.0, anticomm.data[i][j].im, constants.test_epsilon);
            }
        }
    }
}

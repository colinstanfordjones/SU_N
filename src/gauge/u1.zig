const std = @import("std");
const math = @import("math");
const Complex = std.math.Complex(f64);

/// U(1) Gauge Theory
///
/// The Abelian gauge group with elements U = e^{iθ}.
/// When coupled to matter via minimal coupling D_μ = ∂_μ - igA_μ,
/// the coupling constant g determines the interaction strength.
///
/// The fine structure constant emerges as: α = g²/(4π)
///
/// For QED at low energies: α ≈ 1/137, so g ≈ √(4π/137) ≈ 0.303
pub const U1 = struct {
    pub const Matrix1x1 = math.Matrix(Complex, 1, 1);

    /// U(1) gauge coupling constant g
    /// This is THE fundamental parameter of the gauge theory
    /// All electromagnetic quantities derive from this
    ///
    /// At low energy (Thomson limit): g² = 4πα where α ≈ 1/137.036
    /// The value emerges from matching to experiment or from running
    /// from a high-energy boundary condition
    pub const coupling: f64 = @sqrt(4.0 * std.math.pi / 137.035999084);

    /// Fine structure constant: α = g²/(4π)
    /// This EMERGES from the coupling, not the other way around
    pub fn fineStructure() f64 {
        return coupling * coupling / (4.0 * std.math.pi);
    }

    /// Elementary charge e = g in natural units
    /// The charge appearing in D_μ = ∂_μ - ieA_μ
    pub fn elementaryCharge() f64 {
        return coupling;
    }

    pub fn identity() Matrix1x1 {
        return Matrix1x1.init([1][1]Complex{
            .{ Complex.init(1, 0) },
        });
    }

    /// Gauge transformation U = e^{igθ} where g is the coupling
    /// This is the physical gauge transformation including coupling
    pub fn gaugeTransform(theta: f64) Matrix1x1 {
        const phase = coupling * theta;
        return Matrix1x1.init([1][1]Complex{
            .{ Complex.init(@cos(phase), @sin(phase)) },
        });
    }

    /// Pure rotation e^{iθ} (without coupling factor)
    pub fn rotation(theta: f64) Matrix1x1 {
        return Matrix1x1.init([1][1]Complex{
            .{ Complex.init(@cos(theta), @sin(theta)) },
        });
    }

    /// Generator T = 1, such that U = exp(i * θ * T)
    pub fn generator() Matrix1x1 {
        return Matrix1x1.init([1][1]Complex{
            .{ Complex.init(1, 0) },
        });
    }
};

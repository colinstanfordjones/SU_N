//! Haar Measure Sampling for Gauge Groups
//!
//! This module implements random sampling from SU(N) gauge groups with
//! respect to the invariant Haar measure. This is essential for:
//!
//! - Initializing gauge configurations for Monte Carlo simulations
//! - Heat bath updates in gauge evolution
//! - Testing gauge invariance with random transformations
//!
//! ## Haar Measure
//!
//! The Haar measure is the unique left- and right-invariant measure on
//! a compact Lie group. For SU(N), this means:
//!   ∫ f(U) dU = ∫ f(VU) dU = ∫ f(UV) dU  for any V ∈ SU(N)
//!
//! ## Algorithms
//!
//! - U(1): Uniform random phase θ ∈ [0, 2π)
//! - SU(2): Random point on 3-sphere (quaternion parametrization)
//! - SU(N): Gram-Schmidt orthonormalization of random complex matrix
//!
//! ## Precision
//!
//! All calculations use f64 for SIMD compatibility.

const std = @import("std");
const link_mod = @import("link.zig");
const math_utils = @import("math").utils;
const constants = @import("constants");

pub const Complex = link_mod.Complex;

/// Random number generator wrapper for f64 precision
/// Uses PCG (Permuted Congruential Generator) internally
pub const Random = struct {
    pcg: std.Random.Pcg,

    pub fn init(seed: u64) Random {
        return .{ .pcg = std.Random.Pcg.init(seed) };
    }

    /// Uniform random f64 in [0, 1)
    pub fn uniform(self: *Random) f64 {
        const bits: u64 = self.pcg.random().int(u64);
        return @as(f64, @floatFromInt(bits)) / constants.f64_2_pow_64;
    }

    /// Uniform random f64 in [a, b)
    pub fn uniformRange(self: *Random, a: f64, b: f64) f64 {
        return a + (b - a) * self.uniform();
    }

    /// Standard normal using Box-Muller transform
    pub fn normal(self: *Random) f64 {
        const rand1 = self.uniform();
        const rand2 = self.uniform();

        // Avoid log(0)
        const safe_rand1 = if (rand1 < constants.zero_threshold)
            constants.zero_threshold
        else
            rand1;

        const r = @sqrt(-2.0 * @log(safe_rand1));
        const theta = 2.0 * std.math.pi * rand2;

        return r * @cos(theta);
    }

    /// Complex normal with unit variance
    pub fn complexNormal(self: *Random) Complex {
        const scale: f64 = 1.0 / @sqrt(2.0);
        return Complex.init(self.normal() * scale, self.normal() * scale);
    }
};

/// Haar measure sampling for SU(N) groups
pub fn HaarSampler(comptime N: usize) type {
    const LinkType = link_mod.LinkVariable(N);

    return struct {
        const Self = @This();
        pub const Link = LinkType;

        random: Random,

        pub fn init(seed: u64) Self {
            return .{ .random = Random.init(seed) };
        }

        /// Sample a random SU(N) matrix with Haar measure
        pub fn sample(self: *Self) LinkType {
            if (N == 1) {
                return self.sampleU1();
            } else if (N == 2) {
                return self.sampleSU2();
            } else {
                return self.sampleSUN();
            }
        }

        /// Sample random U(1) element: e^{iθ} with θ uniform in [0, 2π)
        fn sampleU1(self: *Self) LinkType {
            const theta = self.random.uniformRange(0, 2.0 * std.math.pi);
            return link_mod.u1FromAngle(theta);
        }

        /// Sample random SU(2) element using quaternion parametrization
        /// The 3-sphere has uniform measure when sampled via normalized Gaussian
        fn sampleSU2(self: *Self) LinkType {
            // Generate 4 independent Gaussians
            var q: [4]f64 = undefined;
            var norm_sq: f64 = 0;

            for (0..4) |i| {
                q[i] = self.random.normal();
                norm_sq += q[i] * q[i];
            }

            // Normalize to unit quaternion
            const inv_norm = 1.0 / @sqrt(norm_sq);
            for (0..4) |i| {
                q[i] *= inv_norm;
            }

            return link_mod.su2FromQuaternion(q);
        }

        /// Sample random SU(N) element using Gram-Schmidt on random complex matrix
        /// This is the standard algorithm for N > 2
        fn sampleSUN(self: *Self) LinkType {
            var result = LinkType.zero();

            // Fill with random complex Gaussians
            for (0..N) |i| {
                for (0..N) |j| {
                    result.matrix.data[i][j] = self.random.complexNormal();
                }
            }

            // Gram-Schmidt orthonormalization (same as unitarize)
            return result.unitarize();
        }

        /// Sample near-identity SU(N) element for small updates
        /// U = exp(i ε H) where H is random Hermitian with ||H|| ~ 1
        /// epsilon controls the step size
        pub fn sampleNearIdentity(self: *Self, epsilon: f64) LinkType {
            if (N == 1) {
                // U(1): small phase
                const theta = self.random.uniformRange(-epsilon, epsilon);
                return link_mod.u1FromAngle(theta);
            } else if (N == 2) {
                // SU(2): small rotation via quaternion
                var q: [4]f64 = .{ 1.0, 0, 0, 0 };
                for (1..4) |i| {
                    q[i] = self.random.uniformRange(-epsilon, epsilon);
                }
                // Normalize
                var norm_sq: f64 = 0;
                for (q) |qi| norm_sq += qi * qi;
                const inv_norm = 1.0 / @sqrt(norm_sq);
                for (0..4) |i| q[i] *= inv_norm;
                return link_mod.su2FromQuaternion(q);
            } else {
                // General SU(N): exp(i ε H) where H is random Hermitian
                return self.sampleExpHermitian(epsilon);
            }
        }

        /// Sample exp(i ε H) for random Hermitian H
        fn sampleExpHermitian(self: *Self, epsilon: f64) LinkType {
            // Generate random Hermitian matrix
            var h = LinkType.zero();

            // Diagonal: real
            for (0..N) |i| {
                h.matrix.data[i][i] = Complex.init(self.random.uniformRange(-1, 1) * epsilon, 0);
            }

            // Off-diagonal: complex, with H_ij = H*_ji
            for (0..N) |i| {
                for ((i + 1)..N) |j| {
                    const re = self.random.uniformRange(-1, 1) * epsilon;
                    const im = self.random.uniformRange(-1, 1) * epsilon;
                    h.matrix.data[i][j] = Complex.init(re, im);
                    h.matrix.data[j][i] = Complex.init(re, -im);
                }
            }

            // Make traceless (for SU(N))
            const tr = h.trace();
            const correction = Complex.init(tr.re / @as(f64, N), tr.im / @as(f64, N));
            for (0..N) |i| {
                h.matrix.data[i][i] = h.matrix.data[i][i].sub(correction);
            }

            // exp(i H)
            return LinkType.fromAlgebra(h.matrix);
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "U1 Haar sampling" {
    var sampler = HaarSampler(1).init(12345);

    for (0..100) |_| {
        const u = sampler.sample();
        // U(1) elements should have |det| = 1
        const det = u.det();
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), @sqrt(det.re * det.re + det.im * det.im), constants.tolerance_iterative);
    }
}

test "SU2 Haar sampling" {
    var sampler = HaarSampler(2).init(12345);

    for (0..100) |_| {
        const u = sampler.sample();
        // SU(2) elements should be unitary with det = 1
        try std.testing.expect(u.isUnitary(constants.tolerance_iterative));
        try std.testing.expect(u.hasUnitDet(constants.tolerance_iterative));
    }
}

test "SU3 Haar sampling" {
    var sampler = HaarSampler(3).init(12345);

    for (0..100) |_| {
        const u = sampler.sample();
        // SU(3) elements should be unitary with det = 1
        try std.testing.expect(u.isUnitary(constants.tolerance_iterative));
        try std.testing.expect(u.hasUnitDet(constants.tolerance_iterative));
    }
}

test "SU2 Haar measure uniformity" {
    // Test that trace distribution follows Haar measure
    // For SU(2), Tr(U) = 2a where a is the first quaternion component
    // Under Haar measure: <Tr> = 0, <Tr²> = 1 (from character orthogonality)
    var sampler = HaarSampler(2).init(54321);

    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    const n_samples = 1000;

    for (0..n_samples) |_| {
        const u = sampler.sample();
        const tr = u.traceReal();
        sum += tr;
        sum_sq += tr * tr;
    }

    const mean = sum / @as(f64, n_samples);
    const variance = sum_sq / @as(f64, n_samples) - mean * mean;

    // Mean of Tr(U) should be 0 for Haar measure
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mean, constants.tolerance_statistical);

    // Variance of Tr(U) = <Tr²> - <Tr>² = 1 - 0 = 1 for SU(2) Haar measure
    // (from character orthogonality: <|χ_j|²> = 1 for fundamental rep)
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), variance, constants.tolerance_statistical);
}

test "near-identity sampling" {
    var sampler = HaarSampler(2).init(11111);

    for (0..100) |_| {
        const u = sampler.sampleNearIdentity(0.1);
        // Should still be unitary
        try std.testing.expect(u.isUnitary(constants.tolerance_iterative));
        // Trace should be close to 2 (identity has trace 2 for SU(2))
        const tr = u.traceReal();
        try std.testing.expect(tr > 1.8);
    }
}

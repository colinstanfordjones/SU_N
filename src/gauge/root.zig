//! Gauge group modules and AMR-integrated lattice gauge theory infrastructure.
//!
//! ## Gauge Groups (Standard Model)
//!
//! These modules implement the gauge groups with coupling constants:
//! - `su3`: SU(3) color group for QCD
//!   - Gell-Mann generators λᵃ, structure constants fᵃᵇᶜ
//!   - Casimir operators for color singlet verification
//!   - QCD field strength tensor F_μν = ∂_μA_ν - ∂_νA_μ + ig_s[A_μ, A_ν]
//! - `su2`: SU(2) weak isospin and Lorentz group representations
//!   - Pauli matrices σⁱ for isospin doublets (proton, neutron)
//!   - SU(2)⊗SU(2) for Lorentz group double cover (Dirac spinors)
//!   - Used for both nuclear isospin and relativistic spin
//! - `gauge_u1`: U(1) electromagnetic gauge group
//!   - Coupling constant g (elementary charge e = g in natural units)
//!   - Fine structure constant α = g²/(4π) ≈ 1/137 emerges from coupling
//!   - Gauge transformations U = e^{igθ} including coupling factor
//!   - **Key principle**: Elementary charge not hardcoded, derived from gauge coupling
//!
//! ## AMR-Integrated Gauge Infrastructure
//!
//! - `frontend`: GaugeFrontend factory for AMR-compatible frontends
//! - `tree`: GaugeTree wraps AMRTree with gauge link storage and ghost handling
//! - `operators`: LinkOperators for gauge link prolongation/restriction
//! - `link`: Link variables U_μ(x) ∈ SU(N) on lattice edges
//! - `haar`: Haar measure sampling for gauge links
//! - `spacetime`: Minkowski metric and 4-vectors
//!
//! ## Design Philosophy
//!
//! **Gauge Symmetry First:**
//! All physical quantities emerge from gauge structure, not assumed constants.
//! For example, U(1) elementary charge e = g (coupling) appears in covariant derivative,
//! fine structure α = g²/(4π) is derived, not hardcoded.
//!
//! **Locality of Behavior:**
//! Gauge links live on AMR block edges; covariant operations use GaugeTree
//! to keep gauge structure and boundary handling together.
//!
//! **Compile-Time Safety:**
//! Generic over gauge group rank N, lattice dimensions checked at compile time,
//! matrix sizes validated statically.

pub const su2 = @import("su2.zig");
pub const su3 = @import("su3.zig");
pub const gauge_u1 = @import("u1.zig");
pub const @"u1" = gauge_u1; // Alias for convenient access as gauge.u1
pub const link = @import("link.zig");
pub const haar = @import("haar.zig");
pub const spacetime = @import("spacetime.zig");

// AMR Integration
pub const frontend = @import("frontend.zig");
pub const operators = @import("operators.zig");
pub const tree = @import("tree.zig");
pub const field = @import("field.zig");
pub const ghost_policy = @import("ghost_policy.zig");
pub const repartition = @import("repartition.zig");

// Re-export commonly used frontend factories
pub const GaugeFrontend = frontend.GaugeFrontend;
// Re-export gauge-specific AMR operators
pub const LinkOperators = operators.LinkOperators;

// Re-export GaugeTree for high-level AMR usage
pub const GaugeTree = tree.GaugeTree;
// Re-export GaugeField for stateless AMR usage
pub const GaugeField = field.GaugeField;

test {
    _ = frontend;
    _ = operators;
    _ = tree;
    _ = field;
    _ = ghost_policy;
    _ = repartition;
}

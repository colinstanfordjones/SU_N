//! Physics modules implementing quantum field theory and lattice simulations.
//!
//! ## Relativistic Requirement
//!
//! All physics simulations use 4D spacetime (t, x, y, z) to preserve Lorentz covariance.
//! The gauge group dimension N determines internal degrees of freedom.
//!
//! ## AMR Multi-Scale Physics
//! - `hamiltonian_amr`: Block-structured AMR Hamiltonian for multi-scale simulations
//!   - 4D spacetime blocks with gauge-covariant derivatives
//!   - Ghost layer communication for cross-block and cross-level operations
//!   - SIMD-optimized stencils within blocks
//! - `hamiltonian_dirac_amr`: Dirac Hamiltonian on AMR blocks for fermions
//!   - H = α·(-iD) + βm + V with gauge-covariant derivatives
//!   - N_field = 4 * N_gauge (4 spinor × N_gauge gauge indices)
//!   - Coulomb potential with soft-core regularization
//!   - Volume-weighted energy/normalization on multi-level meshes
//!
//! Note: Lattice, spacetime, and gauge link modules are in /src/gauge/.
pub const hamiltonian_amr = @import("hamiltonian_amr.zig");
pub const hamiltonian_dirac_amr = @import("hamiltonian_dirac_amr.zig");
pub const dirac = @import("dirac.zig");

// AMR physics modules (gauge-specific operations on AMR meshes)
pub const force_amr = @import("force_amr.zig");

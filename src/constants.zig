//! Physical constants and algorithm parameters for su_n library.
//!
//! This module centralizes all numerical constants to avoid "magic numbers" in code.
//!
//! ## Organization
//!
//! - **Test tolerances**: Floating-point comparison epsilons
//! - **Mathematical constants**: π, common normalization factors
//! - **Physical constants (SI units)**: ℏ, electron mass, elementary charge, ε₀, Coulomb constant
//! - **Physical constants (atomic units)**: Natural units where ℏ = mₑ = e = 4πε₀ = 1
//! - **Physical constants (nuclear units)**: Natural units with ℏ = c = 1, energy in MeV, length in fm
//! - **Derived scales**: Fine structure constant α, classical electron radius, Compton wavelength
//! - **Algorithm parameters**: Matrix exponential convergence, evolution parameters
//! - **Coulomb regularization**: Soft-core potential parameters from toroidal electron model
//! - **Nuclear constants**: Nucleon masses, pion masses, binding energies, Yukawa parameters
//!
//! ## Atomic Units Convention (QED)
//!
//! The library uses atomic units (Hartree units) for atomic physics calculations:
//! - Length: Bohr radius (a₀ ≈ 0.529 Å)
//! - Energy: Hartree (≈ 27.2 eV)
//! - Time: ℏ/Hartree (≈ 2.42×10⁻¹⁷ s)
//!
//! This makes quantum mechanical equations cleaner and avoids numerical scaling issues.
//!
//! ## Nuclear Natural Units Convention
//!
//! For nuclear physics, we use natural units with ℏ = c = 1:
//! - Energy: MeV
//! - Length: fm (femtometers, 10⁻¹⁵ m)
//! - Mass: MeV/c²
//! - Time: fm/c
//! - Conversion: ℏc = 197.327 MeV·fm
//!
//! This is the standard convention in nuclear and particle physics.
//!
//! ## Why Different Unit Systems?
//!
//! Using appropriate units keeps numerical values O(1), improving floating-point
//! precision and making convergence tolerances intuitive:
//!
//! - **Atomic units (QED)**: Hydrogen ground state is -0.5 Hartree (O(1) in code)
//!   The Bohr radius a₀ ≈ 0.529 Å sets the natural length scale.
//!
//! - **Nuclear units**: Deuteron binding is -2.2 MeV (O(1) in code)
//!   The pion Compton wavelength λ_π ≈ 1.4 fm sets the natural length scale.
//!
//! If we used SI units, hydrogen would be ~10⁻¹⁸ J and deuteron ~10⁻¹³ J,
//! leading to precision loss and awkward convergence criteria.
//!
//! ## Unit Conversion Reference
//!
//! Between atomic and nuclear units:
//! - 1 MeV ≈ 36,749 Hartree (nuclear energies are ~10⁶× atomic)
//! - 1 fm ≈ 1.89×10⁻⁵ Bohr radii (nuclear lengths are ~10⁵× smaller)
//! - Nuclear timesteps must be ~10³× smaller due to higher energy scales

const std = @import("std");

/// Epsilon for floating point comparisons in tests (f64)
pub const test_epsilon = 1e-9;

/// Small epsilon for derivatives / generator expansions
pub const small_epsilon = 1e-5;

/// Pi (convenience alias)
pub const pi = std.math.pi;

/// 1 / sqrt(3), common in SU(3) generators
pub const inv_sqrt3 = 1.0 / std.math.sqrt(3.0);

/// 1 / sqrt(2), common in normalization
pub const inv_sqrt2 = 1.0 / std.math.sqrt(2.0);

// =============================================================================
// F64 Precision Constants and Tolerances
// =============================================================================
// f64 has 53 bits of mantissa (~16 decimal digits).
// Matrix operations accumulate errors, so tolerances scale with complexity.

/// Test tolerance for simple operations (element access, identity)
pub const tolerance_tight: f64 = 1e-14;

/// Test tolerance for matrix operations (multiply, adjoint)
pub const tolerance_matrix: f64 = 1e-10;

/// Test tolerance for iterative algorithms (unitarize, exponential)
pub const tolerance_iterative: f64 = 1e-8;

/// Test tolerance for statistical tests (Haar measure uniformity)
pub const tolerance_statistical: f64 = 0.3;

/// Threshold for avoiding division by zero
pub const zero_threshold: f64 = 1e-15;

/// Very small threshold for norm checks (covariant derivative normalization)
pub const norm_threshold: f64 = 1e-14;

/// Determinant magnitude threshold for unitarization phase correction
pub const det_threshold: f64 = 1e-12;

/// 2^64 as f64 - used for uniform random number generation
/// Maximum value of u64 + 1, for scaling random bits to [0, 1)
pub const f64_2_pow_64: f64 = 18446744073709551616.0;

// =============================================================================
// Physical Constants (SI Units)
// =============================================================================

/// Reduced Planck constant (J·s)
pub const hbar_SI = 1.054571817e-34;

/// Electron mass (kg)
pub const electron_mass_SI = 9.1093837015e-31;

/// Elementary charge (C)
pub const elementary_charge_SI = 1.602176634e-19;

/// Vacuum permittivity (F/m)
pub const epsilon_0_SI = 8.8541878128e-12;

/// Coulomb constant k = 1/(4πε₀) (N·m²/C²)
pub const k_coulomb_SI = 8.9875517923e9;

// =============================================================================
// Atomic Units (ℏ = mₑ = e = 4πε₀ = 1)
// =============================================================================

/// Reduced Planck constant in atomic units
pub const hbar_au = 1.0;

/// Electron mass in atomic units
pub const electron_mass_au = 1.0;

/// Elementary charge in atomic units
pub const elementary_charge_au = 1.0;

/// Coulomb constant in atomic units
pub const k_coulomb_au = 1.0;

/// Bohr radius (m) - natural length scale in atomic physics
pub const bohr_radius = 5.29177210903e-11;

/// Hartree energy to eV conversion
pub const hartree_to_eV = 27.211386245988;

/// Hydrogen ground state energy in Hartree (exact: -0.5 Hartree)
pub const hydrogen_ground_state_hartree = -0.5;

/// Hydrogen ground state energy in eV (≈ -13.6 eV)
pub const hydrogen_ground_state_eV = hydrogen_ground_state_hartree * hartree_to_eV;

/// Fine structure constant α ≈ 1/137
/// DERIVED from U(1) gauge coupling: α = g²/(4π)
/// This is not a separate constant - it emerges from the gauge structure
/// Value: coupling² / 4π where coupling = sqrt(4π/137.035999084)
pub const fine_structure_constant: f64 = 1.0 / 137.035999084;

/// Classical electron radius in meters: r_e = α²a₀ ≈ 2.82e-15 m
pub const classical_electron_radius_SI = 2.8179403262e-15;

/// Classical electron radius in atomic units (Bohr radii): r_e/a₀ = α² ≈ 5.3e-5
pub const classical_electron_radius_au = fine_structure_constant * fine_structure_constant;

/// Compton wavelength in meters: λ_c = h/(m_e c) ≈ 2.43e-12 m
pub const compton_wavelength_SI = 2.42631023867e-12;

/// Compton wavelength in atomic units (Bohr radii): λ_c/a₀ = 2πα ≈ 0.046
pub const compton_wavelength_au = 2.0 * pi * fine_structure_constant;

// =============================================================================
// Algorithm Parameters - Matrix Exponential
// =============================================================================

/// Maximum terms in Taylor series for matrix exponential
pub const matrix_exp_max_terms: usize = 100;

/// Convergence epsilon for matrix exponential
pub const matrix_exp_epsilon: f64 = 1e-14;

/// Norm threshold for scaling-and-squaring: scale matrix until ||A|| < this value
pub const matrix_exp_scaling_threshold: f64 = 0.5;

// =============================================================================
// Algorithm Parameters - Imaginary Time Evolution
// =============================================================================

/// Steps between normalization during imaginary time evolution
pub const evolution_normalize_interval: usize = 10;

/// Steps between energy convergence checks
pub const evolution_convergence_check_interval: usize = 100;

/// Default energy convergence tolerance (Hartree)
pub const evolution_energy_tolerance: f64 = 1e-8;

// =============================================================================
// Coulomb Regularization - Toroidal Model (Williamson & van der Mark)
// =============================================================================
// The electron has finite electromagnetic structure with size scale ~10⁻¹² m.
// We use a soft-core potential V(r) = -k/sqrt(r² + δ²) where δ is the
// regularization scale derived from the toroidal electron model.
// Reference: Williamson & van der Mark, Ann. Fond. Louis de Broglie 22, 133 (1997)

/// Coulomb regularization scale in atomic units
/// Uses classical electron radius as the physical cutoff scale
/// For lattice simulations, use max(this, lattice_spacing)
pub const coulomb_regularization_scale_au = classical_electron_radius_au;

// =============================================================================
// Nuclear Physics Constants (SI Units)
// =============================================================================

/// Proton mass (kg)
pub const proton_mass_SI = 1.67262192369e-27;

/// Neutron mass (kg)
pub const neutron_mass_SI = 1.67492749804e-27;

/// Neutral pion mass (kg)
pub const pion_0_mass_SI = 2.406e-28; // ~135 MeV/c²

/// Charged pion mass (kg)
pub const pion_charged_mass_SI = 2.488e-28; // ~139.6 MeV/c²

/// Speed of light (m/s)
pub const speed_of_light_SI = 299792458.0;

// =============================================================================
// Nuclear Physics Constants (Natural Units, ℏ=c=1, energies in MeV)
// =============================================================================

/// ℏc in nuclear natural units (MeV·fm)
/// Fundamental conversion factor: E [MeV] × r [fm] = ℏc
pub const hbar_c_MeV_fm: f64 = 197.3269804;

/// Proton mass (MeV/c²)
pub const proton_mass_MeV = 938.27208816;

/// Proton g-factor (dimensionless)
/// g_p ≈ 5.5856947
pub const proton_g_factor = 5.585694702;

/// Neutron mass (MeV/c²)
pub const neutron_mass_MeV = 939.56542052;

/// Neutral pion mass (MeV/c²)
pub const pion_0_mass_MeV = 134.9768;

/// Charged pion mass (MeV/c²)
pub const pion_charged_mass_MeV = 139.57039;

/// Nucleon reduced mass for deuteron (MeV/c²): μ = m_p*m_n/(m_p+m_n)
pub const nucleon_reduced_mass_MeV = (proton_mass_MeV * neutron_mass_MeV) / (proton_mass_MeV + neutron_mass_MeV);

/// Pion Compton wavelength (fm): λ_π = ℏ/(m_π c) ≈ 1.4 fm (Yukawa force range)
pub const pion_compton_wavelength_fm = 1.413; // Using charged pion mass

/// MeV to eV conversion
pub const MeV_to_eV = 1.0e6;

/// Femtometer to meter conversion
pub const fm_to_m = 1.0e-15;

/// MeV to Hartree conversion: 1 MeV ≈ 36749.3 Hartree
pub const MeV_to_hartree = MeV_to_eV / hartree_to_eV;

/// Femtometer to Bohr radius conversion: 1 fm ≈ 1.89e-5 Bohr
pub const fm_to_bohr = fm_to_m / bohr_radius;

// =============================================================================
// Nuclear Binding Energies (MeV)
// =============================================================================

/// Deuterium (²H) binding energy (MeV)
pub const deuteron_binding_energy_MeV = -2.224575;

/// Tritium (³H) binding energy (MeV)
pub const tritium_binding_energy_MeV = -8.4820;

/// Helium-3 (³He) binding energy (MeV)
pub const helium3_binding_energy_MeV = -7.7181;

/// Helium-4 (⁴He) binding energy (MeV)
pub const helium4_binding_energy_MeV = -28.2957;

// =============================================================================
// Weak Interaction Constants
// =============================================================================

/// Fermi coupling constant (MeV⁻²): G_F/(ℏc)³ ≈ 1.166×10⁻¹¹ MeV⁻²
pub const fermi_coupling_constant_MeV2 = 1.1663787e-11;

/// Neutron lifetime (seconds): mean lifetime for beta decay
pub const neutron_lifetime_s = 879.4;

/// Tritium beta decay Q-value (MeV): energy released in ³H → ³He + e⁻ + ν̄_e
pub const tritium_beta_decay_Q_MeV = 0.01859; // 18.59 keV

// =============================================================================
// Strong Force Parameters (Effective Theory)
// =============================================================================

/// Strong coupling constant for nucleon-nucleon interaction (dimensionless)
/// This is a phenomenological parameter tuned to reproduce deuteron binding
/// Typical values from effective field theory: g² ≈ 13-15
pub const strong_coupling_g2 = 14.0;

/// Nucleon charge radius (fm) - physical size of nucleon
pub const nucleon_radius_fm = 0.84;

/// Yukawa potential regularization scale (fm)
/// Use nucleon charge radius as physical cutoff
pub const yukawa_regularization_scale_fm = nucleon_radius_fm;

// =============================================================================
// Algorithm Parameters - Nuclear Hamiltonian
// =============================================================================

/// Recommended lattice spacing for nuclear calculations (fm)
/// Should resolve pion Compton wavelength: spacing << λ_π ≈ 1.4 fm
pub const nuclear_lattice_spacing_fm = 0.4;

/// Recommended box size for nuclear bound states (fm)
/// Should contain bound state: ~10 fm for deuteron
pub const nuclear_box_size_fm = 10.0;

/// Nuclear evolution timestep (fm/c in natural units)
pub const nuclear_evolution_timestep = 0.01;

/// Nuclear energy convergence tolerance (MeV)
pub const nuclear_energy_tolerance_MeV = 0.01;

// =============================================================================
// Nuclear Reaction Q-Values (Experimental, MeV)
// =============================================================================

/// D + p → He-3 + γ (radiative capture)
pub const Q_D_p_He3_gamma_MeV = 5.494;

/// D + D → He-3 + n (fusion channel 1)
pub const Q_DD_He3_n_MeV = 3.269;

/// D + D → T + p (fusion channel 2)
pub const Q_DD_T_p_MeV = 4.033;

/// D + T → He-4 + n (DT fusion - highest Q for light nuclei)
pub const Q_DT_He4_n_MeV = 17.590;

/// D + D branching ratio for He-3 + n channel (approximately 50%)
pub const DD_branching_He3_n = 0.5;

/// D + D branching ratio for T + p channel (approximately 50%)
pub const DD_branching_T_p = 0.5;

/// Single-channel reaction branching ratio (probability sums to 1)
pub const single_channel_branching_ratio = 1.0;

/// Tolerance for Q-value comparisons in tests (MeV)
/// Accounts for SEMF approximation errors vs experimental values
pub const reaction_q_value_tolerance_MeV = 0.1;

// =============================================================================
// Semi-Empirical Mass Formula (Bethe-Weizsäcker) Coefficients
// =============================================================================
// The SEMF gives binding energy: BE = a_v*A - a_s*A^(2/3) - a_c*Z(Z-1)/A^(1/3)
//                                    - a_a*(N-Z)²/A + δ(A,Z)
// where δ is the pairing term. All coefficients in MeV.
// Reference: Weizsäcker (1935), Bethe & Bacher (1936)

/// Volume term coefficient (MeV): accounts for bulk nuclear binding
pub const semf_a_volume_MeV = 15.75;

/// Surface term coefficient (MeV): accounts for reduced binding at surface
pub const semf_a_surface_MeV = 17.80;

/// Coulomb term coefficient (MeV): accounts for proton-proton repulsion
pub const semf_a_coulomb_MeV = 0.711;

/// Asymmetry term coefficient (MeV): accounts for neutron-proton imbalance
pub const semf_a_asymmetry_MeV = 23.70;

/// Pairing term coefficient (MeV): accounts for nucleon pairing
pub const semf_a_pairing_MeV = 11.18;

// =============================================================================
// Electron and Lepton Constants
// =============================================================================

/// Electron mass (MeV/c²)
pub const electron_mass_MeV = 0.51099895;

/// Positron mass (MeV/c²) - same as electron
pub const positron_mass_MeV = electron_mass_MeV;

/// Electron neutrino mass upper limit (MeV/c²) - effectively zero for reactions
pub const neutrino_mass_MeV = 0.0;

/// Electron charge in units of elementary charge
pub const electron_charge = -1.0;

/// Positron charge in units of elementary charge
pub const positron_charge = 1.0;

// =============================================================================
// QED Perturbative Coefficients
// =============================================================================
// These are computed from Feynman diagram evaluations and are fundamental
// to perturbative QED. They come from first principles calculations.

/// Two-loop QED anomalous magnetic moment coefficient
/// From: Petermann (1957), Sommerfield (1957)
/// a_e^(2) = (α/π)² × (−0.328478965...)
/// Full expression: -197/144 + π²/12 + 3ζ(3)/4 - π²ln(2)/2
pub const qed_two_loop_coefficient = -0.328478965579193;

/// Three-loop QED anomalous magnetic moment coefficient
/// From: Remiddi & Laporta (1996)
/// a_e^(3) = (α/π)³ × 1.181241456...
pub const qed_three_loop_coefficient = 1.181241456587;

/// Four-loop QED anomalous magnetic moment coefficient
/// From: Aoyama et al. (2012, 2015)
/// a_e^(4) = (α/π)⁴ × (−1.9144...)
pub const qed_four_loop_coefficient = -1.9144;
//! su_n: High-Performance Gauge Theory Library
//!
//! A specialized physics library implementing Quantum Field Theory fundamentals
//! with focus on compile-time correctness and real-time performance.
//!
//! ## Modules
//!
//! - `math`: Generic matrices with SIMD, matrix exponential (scaling-and-squaring)
//! - `gauge`: SU(3), SU(2), U(1) gauge groups, AMR gauge rules
//! - `amr`: Block-structured adaptive mesh refinement with ghost filling
//! - `physics`: AMR Hamiltonians and gauge-covariant dynamics
//! - `stats`: Random number generation and statistical distributions
//! - `constants`: Physical constants (SI/atomic/nuclear units) and algorithm parameters
//! - `platform`: OS/runtime utilities (threading, IO, checkpointing)
//!
//! ## Current Capabilities
//!
//! - Matrix exponential: Lie algebra → Lie group via scaling-and-squaring
//! - Dirac Hamiltonian (AMR): Relativistic fermions on adaptive grids
//! - AMR: Block-structured adaptive mesh refinement for multi-scale simulations
//! - Performance: cache-optimized AMR block stencils and compile-time validation
//!
//! ## Example: AMR Hamiltonian
//!
//! ```zig
//! const su_n = @import("su_n");
//! const gauge = su_n.gauge;
//! const amr = su_n.amr;
//! const physics = su_n.physics;
//!
//! const Frontend = gauge.GaugeFrontend(1, 4, 4, 16);
//! const GaugeTree = gauge.GaugeTree(Frontend);
//! const FieldArena = amr.FieldArena(Frontend);
//! const Topology = amr.topology.OpenTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
//! const Hamiltonian = physics.hamiltonian_dirac_amr.HamiltonianDiracAMR(1, 16, Topology);
//!
//! var tree = try GaugeTree.init(allocator, 0.1, 4, 8);
//! defer tree.deinit();
//! _ = try tree.insertBlock(.{0, 0, 0, 0}, 0);
//! var arena = try FieldArena.init(allocator, 16);
//! defer arena.deinit();
//! ```

pub const math = @import("math");
pub const gauge = @import("gauge");
pub const amr = @import("amr");
pub const stats = @import("stats");
pub const physics = @import("physics");
pub const ai = @import("ai");
pub const constants = @import("constants");
pub const platform = @import("platform");
pub const checkpoint = platform.checkpoint;

test "basic test" {
    const std = @import("std");
    try std.testing.expect(true);
}

// Reference submodules for test discovery
comptime {
    _ = gauge.link;
    _ = gauge.haar;
    _ = gauge.su2;
    _ = gauge.su3;
    _ = gauge.spacetime;
    _ = amr.tree;
    _ = amr.block;
    _ = amr.morton;
    _ = amr.field_arena;
    _ = amr.frontend;
    _ = physics.hamiltonian_amr;
    _ = physics.dirac;
    _ = math.matrix;
    _ = math.utils;
    _ = platform.runtime;
    _ = platform.io;
    _ = platform.checkpoint;
    _ = platform.mpi;
    // ai module is tested separately or via integration tests
}

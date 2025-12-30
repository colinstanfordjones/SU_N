//! Mathematical primitives for gauge theory computations.
//!
//! - `Matrix`: Generic compile-time sized matrices with SIMD operations
//!   - Matrix exponential via scaling-and-squaring
//!   - Type-aware arithmetic (scalars, Complex, generic structs)
//!   - Operations: mul, add, sub, scale, transpose, conjugate, adjoint, trace, det
//! - `tensor`: Structure-of-Arrays field containers for lattice simulations
//! - `utils`: Helper functions for matrix operations

pub const matrix = @import("matrix.zig");
pub const Matrix = matrix.Matrix;
pub const utils = @import("utils.zig");
pub const tensor = @import("tensor.zig");

const std = @import("std");

/// A Tensor Field stored in Structure of Arrays (SoA) format.
/// Rank: Tensor rank (0=Scalar, 1=Vector, 2=Matrix, etc.)
/// Dims: Dimension of the vector space (e.g., 4 for Spacetime)
/// Size: Number of lattice points (Grid volume)
pub fn TensorField(comptime Rank: usize, comptime Dims: usize, comptime Size: usize, comptime T: type) type {
    // Number of components = Dims^Rank
    // e.g. Rank 1 (Vector) in 4D = 4 components.
    const num_components = std.math.pow(usize, Dims, Rank);

    return struct {
        // SoA Storage: One array per component.
        // data[component_index][lattice_site]
        data: [num_components][Size]T,

        const Self = @This();

        pub fn zero() Self {
            return .{
                .data = [_][Size]T{[_]T{0} ** Size} ** num_components,
            };
        }

        /// SIMD Addition of two fields
        pub fn add(self: Self, other: Self) Self {
            var result: Self = undefined;
            const VecType = @Vector(Size, T); 
            // Note: If Size is very large, @Vector might be too big for stack/registers.
            // In a real Lattice implementation, 'Size' would be chunked or iterated.
            // For now, assuming small blocks for demonstration.
            
            inline for (0..num_components) |c| {
                const v1: VecType = self.data[c];
                const v2: VecType = other.data[c];
                result.data[c] = v1 + v2;
            }
            return result;
        }
        
        /// Access a specific tensor at a lattice site
        pub fn get(self: Self, site: usize) [num_components]T {
            var res: [num_components]T = undefined;
            inline for (0..num_components) |c| {
                res[c] = self.data[c][site];
            }
            return res;
        }
    };
}

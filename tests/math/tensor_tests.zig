const std = @import("std");
const su_n = @import("su_n");
const TensorField = su_n.math.tensor.TensorField;

test "SoA Tensor Field SIMD Addition" {
    // 4D Vector Field (Rank 1, Dims 4) over 8 lattice sites
    const Field = TensorField(1, 4, 8, f64);
    
    var f1 = Field.zero();
    var f2 = Field.zero();
    
    // Set Time component (idx 0) at site 0 to 1.0
    f1.data[0][0] = 1.0;
    // Set Time component at site 0 to 2.0
    f2.data[0][0] = 2.0;
    
    const sum = f1.add(f2);
    
    // Result should be 3.0 at data[0][0]
    try std.testing.expectEqual(sum.data[0][0], 3.0);
    // Others 0
    try std.testing.expectEqual(sum.data[1][0], 0.0);
}

const std = @import("std");
const amr = @import("amr");

// Test topology (3D)
const TestTopology3D = amr.topology.OpenTopology(3, .{ 4.0, 4.0, 4.0 });

const TestFrontend = struct {
    pub const Nd: usize = 3; // 3D for non-trivial faces (2D faces)
    pub const block_size: usize = 4;
    pub const FieldType = f64;
    pub const Topology = TestTopology3D;
};

const Ops = amr.AMROperators(TestFrontend);
const Block = amr.block.AMRBlock(TestFrontend);

test "prolongateFace - consistent orientation" {
    // Face size for 3D block (4^3) is 4^2 = 16
    // Coarse face size (level L) corresponds to Fine face size (level L+1) 
    // Wait, prolongateFace maps Coarse Face -> Fine Face.
    // Coarse Face (size 16) maps to 4 Fine Faces (each size 16).
    // NO.
    // Coarse Block Face covers area (4*dx)^2.
    // Fine Block Face covers area (4*dx/2)^2 = (2*dx)^2.
    // So Fine Face is 1/4th of Coarse Face.
    // Operators prolongate "Coarse Face Content" to "Fine Face".
    // Does it prolongate the WHOLE coarse face?
    // Let's check logic.
    // coarse_face len is checked vs `face_size / num_children * 2`.
    // For 3D: face_size=16. num_children=8.
    // coarse_face_size = 16 / 8 * 2 = 4.
    // This implies Coarse Face provided to the function is NOT the full face of the coarse block.
    // It is the subset of the coarse face corresponding to the fine block.
    // 
    // If Fine Block is child of Coarse Block.
    // Its face corresponds to a quadrant of the Coarse Block's face.
    // The Coarse Block's face has 16 elements.
    // The "Coarse Source" for the Fine Face has 16 / 4 = 4 elements?
    // Wait, the code says `coarse_face_size = face_size / num_children * 2`??
    // Nd=3. face_size = 4*4 = 16.
    // num_children = 8.
    // 16 / 8 = 2. * 2 = 4.
    // So coarse_face has 4 elements.
    // A 2x2 coarse patch.
    // A 2x2 coarse patch covers (2*spacing)^2 area.
    // Fine block face is 4x4 fine cells.
    // 4 fine cells = 2 coarse cells (refinement ratio 2).
    // So 4x4 fine cells cover 2x2 coarse cells.
    // Correct.
    
    // So input `coarse_face` is a 2x2 patch (flattened).
    // Output `fine_face` is a 4x4 patch (flattened).
    // Logic: Injection (copy coarse value to fine children).
    
    var coarse_face: [4]f64 = .{ 1.0, 2.0, 3.0, 4.0 };
    // Layout (Row-Major):
    // 1.0  2.0
    // 3.0  4.0
    
    var fine_face: [16]f64 = undefined;
    
    // Call prolongateFace
    // face_dim doesn't matter for logic, but let's say face_dim=0 (X-face, so Y-Z plane)
    Ops.prolongateFace(&coarse_face, &fine_face, 0);
    
    // Expected Result:
    // Fine cells [0,0], [0,1] -> Coarse [0,0] (1.0)
    // Fine cells [1,0], [1,1] -> Coarse [0,0] (1.0)
    // ...
    // Each coarse cell becomes 2x2 fine cells.
    // Coarse 0 (1.0) -> Fine (0,0), (0,1), (1,0), (1,1)
    // Coarse 1 (2.0) -> Fine (0,2), (0,3), (1,2), (1,3)
    // Coarse 2 (3.0) -> Fine (2,0), (2,1), (3,0), (3,1)
    // Coarse 3 (4.0) -> Fine (2,2), (2,3), (3,2), (3,3)
    
    // Indices in fine_face (4x4, row major):
    // 0,1,2,3
    // 4,5,6,7
    // ...
    
    // Check 1.0 block
    try std.testing.expectEqual(1.0, fine_face[0]);
    try std.testing.expectEqual(1.0, fine_face[1]);
    try std.testing.expectEqual(1.0, fine_face[4]);
    try std.testing.expectEqual(1.0, fine_face[5]);
    
    // Check 2.0 block
    try std.testing.expectEqual(2.0, fine_face[2]);
    try std.testing.expectEqual(2.0, fine_face[3]);
    try std.testing.expectEqual(2.0, fine_face[6]);
    try std.testing.expectEqual(2.0, fine_face[7]);
    
    // Check 3.0 block
    try std.testing.expectEqual(3.0, fine_face[8]);
    try std.testing.expectEqual(3.0, fine_face[9]);
    try std.testing.expectEqual(3.0, fine_face[12]);
    try std.testing.expectEqual(3.0, fine_face[13]);
    
    // Check 4.0 block
    try std.testing.expectEqual(4.0, fine_face[10]);
    try std.testing.expectEqual(4.0, fine_face[11]);
    try std.testing.expectEqual(4.0, fine_face[14]);
    try std.testing.expectEqual(4.0, fine_face[15]);
}

test "restrictFace - averaging" {
    // Inverse of above.
    // Fine face 4x4. Coarse face 2x2.
    // Set fine cells to 1.0, 2.0, 3.0, 4.0 in quadrants.
    
    var fine_face: [16]f64 = undefined;
    
    // Quadrant 0 (top-left) -> 1.0
    fine_face[0]=1; fine_face[1]=1; fine_face[4]=1; fine_face[5]=1;
    
    // Quadrant 1 (top-right) -> 2.0
    fine_face[2]=2; fine_face[3]=2; fine_face[6]=2; fine_face[7]=2;
    
    // Quadrant 2 (bottom-left) -> 3.0
    fine_face[8]=3; fine_face[9]=3; fine_face[12]=3; fine_face[13]=3;
    
    // Quadrant 3 (bottom-right) -> 4.0
    fine_face[10]=4; fine_face[11]=4; fine_face[14]=4; fine_face[15]=4;
    
    var coarse_face: [4]f64 = undefined;
    
    Ops.restrictFace(&fine_face, &coarse_face, 0);
    
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), coarse_face[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coarse_face[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coarse_face[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), coarse_face[3], 1e-10);
}

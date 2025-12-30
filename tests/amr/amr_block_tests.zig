//! Tests for AMR Block implementation (domain-agnostic)
//!
//! Tests coordinate operations, block properties, and ghost index calculations.
//! Gauge-specific stencil operations are tested in physics module tests.

const std = @import("std");
const amr = @import("amr");
const Complex = std.math.Complex(f64);

// Test topologies
const TestTopology4D = amr.topology.OpenTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });

// Test Frontends for different configurations
const ScalarFrontend4D = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 8;
    pub const FieldType = f64;
    pub const Topology = TestTopology4D;
};

const ScalarFrontend4D_Small = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
    pub const Topology = TestTopology4D;
};

const ComplexFrontend4D = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 8;
    pub const FieldType = Complex;
    pub const Topology = TestTopology4D;
};

const ArrayFrontend4D = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 8;
    pub const FieldType = [2]Complex;
    pub const Topology = TestTopology4D;
};

test "AMRBlock 4D basic properties" {
    const Block = amr.AMRBlock(ScalarFrontend4D);
    const Info = amr.FrontendInfo(ScalarFrontend4D);

    try std.testing.expectEqual(@as(usize, 4096), Info.volume); // 8^4
    try std.testing.expectEqual(@as(usize, 512), Info.face_size); // 8^3
    try std.testing.expectEqual(@as(usize, 8), Info.num_faces); // 2*4
    try std.testing.expectEqual(@as(usize, 16), Info.num_children); // 2^4

    // Block type exists
    _ = Block;
}

test "AMRBlock coordinate round-trip" {
    const Block = amr.AMRBlock(ScalarFrontend4D);

    // Test a subset to keep test fast
    for (0..4) |t| {
        for (0..4) |x| {
            for (0..4) |y| {
                for (0..4) |z| {
                    const coords = [4]usize{ t, x, y, z };
                    const idx = Block.getLocalIndex(coords);
                    const recovered = Block.getLocalCoords(idx);
                    try std.testing.expectEqual(coords, recovered);
                }
            }
        }
    }
}

test "AMRBlock boundary detection" {
    const Block = amr.AMRBlock(ScalarFrontend4D);

    // Interior point
    try std.testing.expect(!Block.isOnBoundary(.{ 4, 4, 4, 4 }));

    // Boundary points (on any face)
    try std.testing.expect(Block.isOnBoundary(.{ 0, 4, 4, 4 })); // t=0
    try std.testing.expect(Block.isOnBoundary(.{ 7, 4, 4, 4 })); // t=7
    try std.testing.expect(Block.isOnBoundary(.{ 4, 0, 4, 4 })); // x=0
    try std.testing.expect(Block.isOnBoundary(.{ 4, 7, 4, 4 })); // x=7
    try std.testing.expect(Block.isOnBoundary(.{ 4, 4, 0, 4 })); // y=0
    try std.testing.expect(Block.isOnBoundary(.{ 4, 4, 4, 7 })); // z=7
}

test "AMRBlock init and deinit" {
    const Block = amr.AMRBlock(ArrayFrontend4D);

    const block = Block.init(0, .{ 0, 0, 0, 0 }, 1.0);

    try std.testing.expectEqual(@as(u8, 0), block.level);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), block.spacing, 1e-10);
}

test "AMRBlock refinement level spacing" {
    const Block = amr.AMRBlock(ScalarFrontend4D);

    // Level 0: base spacing
    const block0 = Block.init(0, .{ 0, 0, 0, 0 }, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), block0.spacing, 1e-10);

    // Level 1: half spacing
    const block1 = Block.init(1, .{ 0, 0, 0, 0 }, 0.5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), block1.spacing, 1e-10);

    // Level 2: quarter spacing
    const block2 = Block.init(2, .{ 0, 0, 0, 0 }, 0.25);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), block2.spacing, 1e-10);
}

test "AMRBlock small block 4D" {
    const Info = amr.FrontendInfo(ScalarFrontend4D_Small);

    try std.testing.expectEqual(@as(usize, 256), Info.volume); // 4^4
    try std.testing.expectEqual(@as(usize, 64), Info.face_size); // 4^3
    try std.testing.expectEqual(@as(usize, 8), Info.num_faces);

    // Test coordinate round-trip
    const Block = amr.AMRBlock(ScalarFrontend4D_Small);
    const coords = [4]usize{ 1, 2, 3, 0 };
    const idx = Block.getLocalIndex(coords);
    const recovered = Block.getLocalCoords(idx);
    try std.testing.expectEqual(coords, recovered);
}

test "AMRBlock ghost index calculation" {
    const Block = amr.AMRBlock(ScalarFrontend4D_Small);

    // For face 0 (+t), ghost index should use (x, y, z) coordinates
    const coords = [4]usize{ 3, 2, 1, 0 }; // On +t face
    const ghost_idx = Block.getGhostIndex(coords, 0); // face 0 = +t

    // For +t face (dim=0), we skip t and use x, y, z
    // ghost_idx = x * 1 + y * 4 + z * 16 = 2 + 1*4 + 0*16 = 6
    try std.testing.expectEqual(@as(usize, 6), ghost_idx);
}

test "AMRBlock physical position" {
    const Block = amr.AMRBlock(ScalarFrontend4D);

    // Level 0 block at origin with spacing 1.0
    var block = Block.init(0, .{ 0, 0, 0, 0 }, 1.0);

    // Site (1, 2, 3, 4) should be at physical position (1.0, 2.0, 3.0, 4.0)
    const site = Block.getLocalIndex(.{ 1, 2, 3, 4 });
    const pos = block.getPhysicalPosition(site);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), pos[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), pos[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), pos[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), pos[3], 1e-10);
}

test "AMRBlock physical position with offset origin" {
    const Block = amr.AMRBlock(ScalarFrontend4D);

    // Block at origin (8, 0, 0, 0) - one block over in t direction
    var block = Block.init(0, .{ 8, 0, 0, 0 }, 1.0);

    // Site (0, 0, 0, 0) should be at physical position (8.0, 0.0, 0.0, 0.0)
    const site = Block.getLocalIndex(.{ 0, 0, 0, 0 });
    const pos = block.getPhysicalPosition(site);

    try std.testing.expectApproxEqAbs(@as(f64, 8.0), pos[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), pos[1], 1e-10);
}

test "AMRBlock local neighbor fast" {
    const Block = amr.AMRBlock(ScalarFrontend4D);

    // Test interior neighbor calculation
    const center = Block.getLocalIndex(.{ 4, 4, 4, 4 });

    // +t neighbor (face 0)
    const plus_t = Block.localNeighborFast(center, 0);
    const plus_t_coords = Block.getLocalCoords(plus_t);
    try std.testing.expectEqual([4]usize{ 5, 4, 4, 4 }, plus_t_coords);

    // -t neighbor (face 1)
    const minus_t = Block.localNeighborFast(center, 1);
    const minus_t_coords = Block.getLocalCoords(minus_t);
    try std.testing.expectEqual([4]usize{ 3, 4, 4, 4 }, minus_t_coords);

    // +x neighbor (face 2)
    const plus_x = Block.localNeighborFast(center, 2);
    const plus_x_coords = Block.getLocalCoords(plus_x);
    try std.testing.expectEqual([4]usize{ 4, 5, 4, 4 }, plus_x_coords);
}

test "AMRBlock complex field type" {
    const Block = amr.AMRBlock(ComplexFrontend4D);
    const Info = amr.FrontendInfo(ComplexFrontend4D);

    // Check field size
    try std.testing.expectEqual(@as(usize, 16), Info.field_size); // sizeof(Complex) = 16

    _ = Block;
}

test "AMRBlock array field type" {
    const Info = amr.FrontendInfo(ArrayFrontend4D);

    // [2]Complex = 2 * 16 = 32 bytes
    try std.testing.expectEqual(@as(usize, 32), Info.field_size);
}

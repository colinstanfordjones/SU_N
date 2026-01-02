const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");
const topology = amr.topology;

const Complex = std.math.Complex(f64);

// Mock U(1) link for testing
const U1Link = struct {
    phase: f64,

    pub fn identity() U1Link {
        return .{ .phase = 0.0 };
    }
    
    pub fn zero() U1Link {
        return .{ .phase = 0.0 }; // For U(1) zero/identity overlap in this mock, but fine
    }

    pub fn mul(self: U1Link, other: U1Link) U1Link {
        return .{ .phase = self.phase + other.phase };
    }
    
    pub fn add(self: U1Link, other: U1Link) U1Link {
        // Mock addition for averaging
        return .{ .phase = self.phase + other.phase }; 
    }
    
    pub fn norm(self: U1Link) f64 {
        return @abs(self.phase);
    }
    
    pub fn scaleReal(self: U1Link, s: f64) U1Link {
        return .{ .phase = self.phase * s };
    }
    
    pub fn unitarize(self: U1Link) U1Link {
        return self;
    }
    
    pub fn adjoint(self: U1Link) U1Link {
        return .{ .phase = -self.phase };
    }
    
    pub fn trace(self: U1Link) Complex {
        return Complex.init(@cos(self.phase), @sin(self.phase));
    }
};

const TestFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const gauge_group_dim: usize = 1;
    pub const LinkType = U1Link;
    pub const EdgeType = U1Link;
    pub const FieldType = f64; // Dummy
    pub const face_size = 4;
    pub const num_children = 4;
    pub const Topology = topology.PeriodicTopology(2, .{ 4.0, 4.0 });

    pub const LinkOperators = struct {
         pub const LinkType = U1Link;
         pub fn prolongateLink(l: U1Link) [2]U1Link {
             return .{ .{ .phase = l.phase/2.0 }, .{ .phase = l.phase/2.0 } };
         }
         pub fn restrictLink(a: U1Link, b: U1Link) U1Link {
             return a.mul(b);
         }
    };
};

test "GaugeField - initialization and sync" {
    const allocator = std.testing.allocator;
    const Tree = amr.AMRTree(TestFrontend);
    const Field = gauge.GaugeField(TestFrontend);
    
    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();
    
    var field = try Field.init(allocator, &tree);
    defer field.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), field.slots.items.len);
    
    // Insert blocks
    _ = try tree.insertBlock(.{0, 0}, 0);
    _ = try tree.insertBlock(.{4, 4}, 0);
    
    // Sync
    try field.syncWithTree(&tree);
    
    try std.testing.expectEqual(@as(usize, 2), field.slots.items.len);
    try std.testing.expect(field.slots.items[0] != std.math.maxInt(usize));
    try std.testing.expect(field.slots.items[1] != std.math.maxInt(usize));
    
    // Check links initialized to identity
    const link = field.getLink(0, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), link.phase, 1e-10);
}

test "GaugeField - save and restore links" {
    const allocator = std.testing.allocator;
    const Tree = amr.AMRTree(TestFrontend);
    const Field = gauge.GaugeField(TestFrontend);
    
    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();
    var field = try Field.init(allocator, &tree);
    defer field.deinit();
    
    _ = try tree.insertBlock(.{0, 0}, 0);
    try field.syncWithTree(&tree);
    
    // Modify link
    field.setLink(0, 5, 0, .{ .phase = 1.23 });
    
    // Save
    const backup = try field.saveLinks(allocator);
    defer Field.freeBackup(allocator, backup);
    
    // Modify again
    field.setLink(0, 5, 0, .{ .phase = 9.99 });
    
    // Restore
    field.restoreLinks(backup);
    
    const restored = field.getLink(0, 5, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.23), restored.phase, 1e-10);
}

test "GaugeField - computePlaquette" {
    const allocator = std.testing.allocator;
    const Tree = amr.AMRTree(TestFrontend);
    const Field = gauge.GaugeField(TestFrontend);
    const LinkOps = gauge.LinkOperators(TestFrontend);
    
    var tree = try Tree.init(allocator, 1.0, 4, 8);
    defer tree.deinit();
    var field = try Field.init(allocator, &tree);
    defer field.deinit();
    
    _ = try tree.insertBlock(.{0, 0}, 0);
    try field.syncWithTree(&tree);
    try field.fillGhosts(&tree); // Need ghosts for boundary plaquettes
    
    // Set identity links everywhere
    // Plaquette should be identity (phase 0)
    const plaq = LinkOps.computePlaquette(&tree, &field, 0, 0, 0, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), plaq.phase, 1e-10);
    
    // Set a loop
    // U_0(x) = 0.1
    // U_1(x+0) = 0.2
    // U_0(x+1) = 0.3 -> dag = -0.3
    // U_1(x) = 0.4 -> dag = -0.4
    // Sum = 0.1 + 0.2 - 0.3 - 0.4 = -0.4
    
    // We need to set specific links.
    // Site 0 is (0,0).
    // Neighbor+mu(0) is (1,0) -> site 1.
    // Neighbor+nu(0) is (0,1) -> site 4.
    
    field.setLink(0, 0, 0, .{ .phase = 0.1 });
    field.setLink(0, 1, 1, .{ .phase = 0.2 });
    field.setLink(0, 4, 0, .{ .phase = 0.3 });
    field.setLink(0, 0, 1, .{ .phase = 0.4 });
    
    const plaq2 = LinkOps.computePlaquette(&tree, &field, 0, 0, 0, 1);
    try std.testing.expectApproxEqAbs(@as(f64, -0.4), plaq2.phase, 1e-10);
}

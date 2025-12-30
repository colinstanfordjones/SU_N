//! Tests for AMR Operators (Prolongation and Restriction)
//!
//! Tests domain-agnostic field operators. Gauge-specific link operators
//! are tested separately via gauge.operators.LinkOperators.

const std = @import("std");
const amr = @import("amr");
const gauge = @import("gauge");

const Complex = std.math.Complex(f64);

// Test Topologies
const TestTopology4D = amr.topology.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
const TestTopology2D = amr.topology.PeriodicTopology(2, .{ 1.0, 1.0 });

// Test Frontends
const ScalarFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
    pub const Topology = TestTopology4D;
};

const ComplexFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 4;
    pub const FieldType = Complex;
    pub const Topology = TestTopology4D;
};

const ArrayComplexFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 4;
    pub const FieldType = [1]Complex;
    pub const Topology = TestTopology4D;
};

const TwoComponentFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
    pub const Topology = TestTopology2D;
};

test "AMROperators 4D basic properties" {
    const Ops = amr.AMROperators(ScalarFrontend);

    try std.testing.expectEqual(@as(usize, 16), Ops.children_count); // 2^4
    try std.testing.expectEqual(@as(usize, 256), Ops.volume); // 4^4
    try std.testing.expectEqual(@as(usize, 4), Ops.dimensions);
}

test "AMROperators 2D properties" {
    const Ops = amr.AMROperators(TwoComponentFrontend);

    try std.testing.expectEqual(@as(usize, 4), Ops.children_count); // 2^2
    try std.testing.expectEqual(@as(usize, 16), Ops.volume); // 4^2
    try std.testing.expectEqual(@as(usize, 2), Ops.dimensions);
}

test "AMROperators scalar prolongation constant field" {
    const Ops = amr.AMROperators(ScalarFrontend);

    // Create coarse scalar field with constant value
    var coarse: [Ops.volume]f64 = undefined;
    for (&coarse) |*v| v.* = 3.14;

    // Create fine child buffers (16 children for 4D)
    var fine_buffers: [16][Ops.volume]f64 = undefined;
    var fine_children: [16][]f64 = undefined;
    for (0..16) |i| fine_children[i] = &fine_buffers[i];

    // Prolongate without norm preservation
    Ops.prolongateInjection(&coarse, &fine_children, false);

    // All fine values should be 3.14
    for (0..16) |child| {
        for (fine_children[child]) |v| {
            try std.testing.expectApproxEqAbs(@as(f64, 3.14), v, 1e-10);
        }
    }
}

test "AMROperators complex prolongation constant field" {
    const Ops = amr.AMROperators(ComplexFrontend);

    // Create coarse complex field
    var coarse: [Ops.volume]Complex = undefined;
    for (&coarse) |*v| v.* = Complex.init(1.0, 0.5);

    var fine_buffers: [16][Ops.volume]Complex = undefined;
    var fine_children: [16][]Complex = undefined;
    for (0..16) |i| fine_children[i] = &fine_buffers[i];

    Ops.prolongateInjection(&coarse, &fine_children, false);

    for (0..16) |child| {
        for (fine_children[child]) |v| {
            try std.testing.expectApproxEqAbs(@as(f64, 1.0), v.re, 1e-10);
            try std.testing.expectApproxEqAbs(@as(f64, 0.5), v.im, 1e-10);
        }
    }
}

test "AMROperators array complex prolongation" {
    const Ops = amr.AMROperators(ArrayComplexFrontend);

    var coarse: [Ops.volume][1]Complex = undefined;
    for (&coarse) |*v| v.*[0] = Complex.init(2.0, 1.0);

    var fine_buffers: [16][Ops.volume][1]Complex = undefined;
    var fine_children: [16][][1]Complex = undefined;
    for (0..16) |i| fine_children[i] = &fine_buffers[i];

    Ops.prolongateInjection(&coarse, &fine_children, false);

    for (0..16) |child| {
        for (fine_children[child]) |v| {
            try std.testing.expectApproxEqAbs(@as(f64, 2.0), v[0].re, 1e-10);
            try std.testing.expectApproxEqAbs(@as(f64, 1.0), v[0].im, 1e-10);
        }
    }
}

test "AMROperators prolongation preserves norm" {
    const Ops = amr.AMROperators(ComplexFrontend);

    // Create coarse field with varying values
    var coarse: [Ops.volume]Complex = undefined;
    for (&coarse, 0..) |*v, i| {
        const re = @as(f64, @floatFromInt(i % 10));
        const im = @as(f64, @floatFromInt((i * 3) % 7));
        v.* = Complex.init(re, im);
    }

    // Compute coarse norm squared
    var coarse_norm_sq: f64 = 0.0;
    for (coarse) |v| {
        coarse_norm_sq += v.re * v.re + v.im * v.im;
    }

    // Prolongate with norm preservation
    var fine_buffers: [16][Ops.volume]Complex = undefined;
    var fine_children: [16][]Complex = undefined;
    for (0..16) |i| fine_children[i] = &fine_buffers[i];

    Ops.prolongateInjection(&coarse, &fine_children, true);

    // Compute fine norm squared
    var fine_norm_sq: f64 = 0.0;
    for (0..16) |child| {
        for (fine_children[child]) |v| {
            fine_norm_sq += v.re * v.re + v.im * v.im;
        }
    }

    // Norms should match
    try std.testing.expectApproxEqRel(coarse_norm_sq, fine_norm_sq, 1e-6);
}

test "AMROperators restriction constant field" {
    const Ops = amr.AMROperators(ComplexFrontend);

    // Create fine children with constant value
    var fine_buffers: [16][Ops.volume]Complex = undefined;
    var fine_children: [16][]const Complex = undefined;
    for (0..16) |i| {
        for (&fine_buffers[i]) |*v| v.* = Complex.init(2.0, 1.0);
        fine_children[i] = &fine_buffers[i];
    }

    // Restrict
    var coarse: [Ops.volume]Complex = undefined;
    Ops.restrictFullWeight(&fine_children, &coarse, false);

    // Average of constant should be constant
    for (coarse) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 2.0), v.re, 1e-10);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), v.im, 1e-10);
    }
}

test "AMROperators prolongation-restriction round trip" {
    const Ops = amr.AMROperators(ComplexFrontend);

    // Create smooth coarse field
    var coarse_orig: [Ops.volume]Complex = undefined;
    for (0..Ops.volume) |i| {
        const angle = @as(f64, @floatFromInt(i)) * 0.1;
        coarse_orig[i] = Complex.init(@sin(angle), @cos(angle) * 0.5);
    }

    // Prolongate
    var fine_buffers: [16][Ops.volume]Complex = undefined;
    var fine_children_mut: [16][]Complex = undefined;
    var fine_children_const: [16][]const Complex = undefined;
    for (0..16) |i| {
        fine_children_mut[i] = &fine_buffers[i];
        fine_children_const[i] = &fine_buffers[i];
    }

    Ops.prolongateInjection(&coarse_orig, &fine_children_mut, false);

    // Restrict back
    var coarse_back: [Ops.volume]Complex = undefined;
    Ops.restrictFullWeight(&fine_children_const, &coarse_back, false);

    // Should recover original
    for (0..Ops.volume) |i| {
        try std.testing.expectApproxEqAbs(coarse_orig[i].re, coarse_back[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(coarse_orig[i].im, coarse_back[i].im, 1e-10);
    }
}

// ============================================================================
// Link Operator Tests (from gauge.operators)
// ============================================================================

test "LinkOperators U1 prolongation preserves gauge structure" {
    // Use GaugeFrontend to get properly configured LinkOperators
    const Frontend = gauge.GaugeFrontend(1, 1, 4, 4, TestTopology4D); // U(1), scalar, 4D, 4^4 blocks
    const Link = Frontend.LinkType;
    const LinkOps = Frontend.LinkOperators;

    // Create a U(1) link with phase π/4
    var coarse_link = Link.zero();
    const angle = std.math.pi / 4.0;
    coarse_link.matrix.data[0][0] = Complex.init(@cos(angle), @sin(angle));

    // Prolongate to two fine links
    const fine_links = LinkOps.prolongateLink(coarse_link);

    // Product of fine links should equal coarse link
    const product = fine_links[0].mul(fine_links[1]);
    const coarse_phase = std.math.atan2(coarse_link.matrix.data[0][0].im, coarse_link.matrix.data[0][0].re);
    const product_phase = std.math.atan2(product.matrix.data[0][0].im, product.matrix.data[0][0].re);

    try std.testing.expectApproxEqAbs(coarse_phase, product_phase, 1e-10);
}

test "LinkOperators U1 restriction is product" {
    const Frontend = gauge.GaugeFrontend(1, 1, 4, 4, TestTopology4D); // U(1), scalar, 4D, 4^4 blocks
    const Link = Frontend.LinkType;
    const LinkOps = Frontend.LinkOperators;

    // Create two fine U(1) links
    var fine1 = Link.zero();
    var fine2 = Link.zero();
    const angle1 = std.math.pi / 6.0;
    const angle2 = std.math.pi / 3.0;
    fine1.matrix.data[0][0] = Complex.init(@cos(angle1), @sin(angle1));
    fine2.matrix.data[0][0] = Complex.init(@cos(angle2), @sin(angle2));

    // Restrict to coarse link
    const coarse_link = LinkOps.restrictLink(fine1, fine2);

    // Coarse link should have phase = angle1 + angle2
    const expected_angle = angle1 + angle2;
    const coarse_phase = std.math.atan2(coarse_link.matrix.data[0][0].im, coarse_link.matrix.data[0][0].re);

    try std.testing.expectApproxEqAbs(expected_angle, coarse_phase, 1e-10);
}

test "LinkOperators SU2 prolongation" {
    const Frontend = gauge.GaugeFrontend(2, 1, 4, 4, TestTopology4D); // SU(2), scalar, 4D, 4^4 blocks
    const Link = Frontend.LinkType;
    const LinkOps = Frontend.LinkOperators;

    // Create an SU(2) link close to identity
    var coarse_link = Link.identity();
    // Small perturbation
    coarse_link.matrix.data[0][0] = Complex.init(0.99, 0.1);
    coarse_link.matrix.data[0][1] = Complex.init(0.05, 0.05);
    coarse_link.matrix.data[1][0] = Complex.init(-0.05, 0.05);
    coarse_link.matrix.data[1][1] = Complex.init(0.99, -0.1);
    coarse_link = coarse_link.unitarize();

    // Prolongate
    const fine_links = LinkOps.prolongateLink(coarse_link);

    // Product should approximately equal coarse link
    const product = fine_links[0].mul(fine_links[1]);

    // Check all matrix elements are close
    for (0..2) |i| {
        for (0..2) |j| {
            const expected = coarse_link.matrix.data[i][j];
            const actual = product.matrix.data[i][j];
            try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-6);
            try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-6);
        }
    }
}

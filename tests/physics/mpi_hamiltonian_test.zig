const std = @import("std");
const su_n = @import("su_n");
const amr = @import("amr");
const gauge = @import("gauge");
const physics = @import("physics");
const constants = su_n.constants;
const platform = su_n.platform;

const Complex = std.math.Complex(f64);

fn zeroPotential(pos: [2]f64, spacing: f64) f64 {
    _ = pos;
    _ = spacing;
    return 0.0;
}

test "mpi hamiltonian energy is deterministic" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size < 2) return error.SkipZigTest;

    const block_size = 4;
    const Topology = amr.topology.OpenTopology(2, .{ 8.0, 8.0 });
    const Frontend = gauge.GaugeFrontend(1, 1, 2, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const GhostBuffer = amr.GhostBuffer(Frontend);
    const HAMR = physics.hamiltonian_amr.HamiltonianAMR(Frontend);

    var psi = try Arena.init(std.testing.allocator, 4);
    defer psi.deinit();

    var workspace = try Arena.init(std.testing.allocator, 4);
    defer workspace.deinit();

    var ghosts = try GhostBuffer.init(std.testing.allocator, 4);
    defer ghosts.deinit();

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();
    var field = try GaugeField.init(std.testing.allocator, &tree);
    defer field.deinit();

    const block_idx = try tree.insertBlockWithField(.{ 0, 0 }, 0, &psi);
    try field.syncWithTree(&tree);
    const slot = tree.getFieldSlot(block_idx);
    const work_slot = workspace.allocSlot() orelse return error.OutOfMemory;
    try std.testing.expectEqual(slot, work_slot);

    const psi_data = psi.getSlot(slot);
    for (psi_data) |*v| {
        v.*[0] = Complex.init(1.0, 0.0);
    }

    var h = HAMR.init(&tree, &field, 1.0, zeroPotential);
    const local_energy = try h.measureEnergy(&psi, &workspace, &ghosts);
    const total = try platform.mpi.allreduceSum(comm, local_energy);

    const expected = local_energy * @as(f64, @floatFromInt(size));
    try std.testing.expectApproxEqAbs(expected, total, constants.test_epsilon);
}

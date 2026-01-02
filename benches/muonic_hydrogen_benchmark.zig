const std = @import("std");
const su_n = @import("su_n");
const constants = su_n.constants;
const amr = su_n.amr;
const HamiltonianDiracAMR = su_n.physics.hamiltonian_dirac_amr.HamiltonianDiracAMR;
const Complex = std.math.Complex(f64);

// =============================================================================
// Muonic Hydrogen Benchmark
// =============================================================================

const block_size = 16;
const grid_dim = 2; // 2x2x2 blocks
const spacing = 0.05;
const N_gauge = 1;
const domain_extent = @as(f64, @floatFromInt(grid_dim * block_size)) * spacing;
const Topology4D = amr.topology.OpenTopology(4, .{ domain_extent, domain_extent, domain_extent, domain_extent });
const HAMR = HamiltonianDiracAMR(N_gauge, block_size, Topology4D);
const N_field = HAMR.field_dim;
const GT = HAMR.GaugeTreeType;
const Arena = HAMR.FieldArena;
const Ghosts = HAMR.GhostBufferType;

const MuonConstants = struct {
    const alpha: f64 = constants.fine_structure_constant;
    const m_mu: f64 = 206.768283; // Muon mass in m_e units
    const m_p: f64 = 1836.152673; // Proton mass
    const reduced_mass: f64 = (m_mu * m_p) / (m_mu + m_p);
    
    // Theoretical binding energy for point nucleus: -0.5 * alpha^2 * m_mu (approx)
    // Actually uses reduced mass for center of mass frame
    const expected_binding_hartree: f64 = -0.5 * reduced_mass; 
};

fn runMuonicHydrogen(allocator: std.mem.Allocator, max_steps: usize) !void {
    const print = std.debug.print;

    var tree = try GT.init(allocator, spacing, 2, 8);
    defer tree.deinit();

    var psi = try Arena.init(allocator, 32);
    var workspace = try Arena.init(allocator, 32);
    var workspace2 = try Arena.init(allocator, 32);
    var ghosts = try Ghosts.init(allocator, 32);

    defer {
        psi.deinit();
        workspace.deinit();
        workspace2.deinit();
        ghosts.deinit();
    }

    var blocks = std.ArrayList(usize){};
    defer blocks.deinit(allocator);

    for (0..grid_dim) |x| {
        for (0..grid_dim) |y| {
            for (0..grid_dim) |z| {
                const idx = try tree.insertBlock(.{ 0, x * block_size, y * block_size, z * block_size }, 0);
                try blocks.append(allocator, idx);
                const slot = psi.allocSlot() orelse return error.OutOfMemory;
                _ = workspace.allocSlot() orelse return error.OutOfMemory;
                _ = workspace2.allocSlot() orelse return error.OutOfMemory;
                tree.tree.assignFieldSlot(idx, slot);
            }
        }
    }

    var h = HAMR.init(&tree, MuonConstants.m_mu, 1.0, .coulomb);
    defer h.deinit();

    const center_pos = @as(f64, @floatFromInt(grid_dim * block_size)) * spacing / 2.0;
    h.setCenter(.{ center_pos, center_pos, center_pos, center_pos });

    const alpha_mu = MuonConstants.alpha * MuonConstants.m_mu;

    for (blocks.items) |b_idx| {
        const slot = tree.tree.getFieldSlot(b_idx);
        const psi_data = psi.getSlot(slot);
        const block = &tree.tree.blocks.items[b_idx];

        for (0..HAMR.block_volume) |i| {
            const pos = block.getPhysicalPosition(i);

            const dx = pos[1] - center_pos;
            const dy = pos[2] - center_pos;
            const dz = pos[3] - center_pos;
            const r = @sqrt(dx * dx + dy * dy + dz * dz);

            const amp = @exp(-alpha_mu * r);
            psi_data[i][0] = Complex.init(amp, 0);
            for (1..N_field) |c| psi_data[i][c] = Complex.init(0, 0);
        }
    }

    h.normalize(&psi);

    print("Muonic Hydrogen Setup:\n", .{});
    print("  Mass: {d:.4} m_e\n", .{MuonConstants.m_mu});
    print("  Spacing: {d:.4} (natural units)\n", .{spacing});
    print("  Grid: {d}x{d}x{d} blocks (Extent {d:.2})\n", .{ grid_dim, grid_dim, grid_dim, @as(f64, @floatFromInt(grid_dim * block_size)) * spacing });

    var timer = std.time.Timer.start() catch unreachable;
    const result = h.evolveUntilConvergedAMR(&psi, &workspace, &workspace2, &ghosts, 0.00005, max_steps, 1e-4, 50) catch @panic("evolve failed");
    const elapsed = timer.read();

    const energy_hartree = (result.energy - MuonConstants.m_mu) / (MuonConstants.alpha * MuonConstants.alpha);

    print("\nResults:\n", .{});
    print("  Total Energy: {d:.6}\n", .{result.energy});
    print("  Binding Energy: {d:.6} Hartree (Expected ~{d:.2})\n", .{ energy_hartree, MuonConstants.expected_binding_hartree });
    print("  Converged: {s} (Energy={d:.6})\n", .{ if (result.converged) "yes" else "no", result.energy });
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed)) / 1e6});
}

pub fn main() void {
    runMuonicHydrogen(std.heap.page_allocator, 1000) catch |err| {
        std.debug.print("Error: {}\n", .{err});
    };
}

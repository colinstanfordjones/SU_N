const std = @import("std");
const su_n = @import("su_n");
const gauge = su_n.gauge;
const amr = su_n.amr;
const HamiltonianAMR = su_n.physics.hamiltonian_amr.HamiltonianAMR;
const exporter = su_n.exporter;
const constants = su_n.constants;

const Complex = std.math.Complex(f64);

// Global for potential parameters (used by the function pointer potential)
var current_potential = struct {
    centers: [8][4]f64 = undefined,
    charges: [8]f64 = undefined,
    num_centers: usize = 0,
}{};

/// Random Coulomb potential wrapper
fn randomPotential(pos: [4]f64, spacing: f64) f64 {
    var v: f64 = 0.0;
    const delta_sq = spacing * spacing;
    for (0..current_potential.num_centers) |i| {
        const dx = pos[1] - current_potential.centers[i][1];
        const dy = pos[2] - current_potential.centers[i][2];
        const dz = pos[3] - current_potential.centers[i][3];
        const r = @sqrt(dx * dx + dy * dy + dz * dz + delta_sq);
        v -= current_potential.charges[i] * constants.fine_structure_constant / r;
    }
    return v;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <num_samples> <output_dir>\n", .{args[0]});
        return;
    }

    const num_samples = try std.fmt.parseInt(usize, args[1], 10);
    const output_dir = args[2];

    try std.fs.cwd().makePath(output_dir);

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rand = prng.random();

    // Problem parameters
    const N = 1; // U(1) gauge group
    const Nd = 4; // 4D Spacetime
    const block_size = 16;

    // Topology setup
    const Topology = amr.topology.PeriodicTopology(Nd, .{ 16.0, 16.0, 16.0, 16.0 });

    // Frontend setup
    const Frontend = gauge.GaugeFrontend(N, 1, Nd, block_size, Topology); // 1 spinor component (scalar)
    const Tree = amr.AMRTree(Frontend);
    const GaugeField = gauge.GaugeField(Frontend);
    const FieldArena = amr.FieldArena(Frontend);
    const GhostBuffer = amr.GhostBuffer(Frontend);
    const HAMR = HamiltonianAMR(Frontend);
    const Exp = exporter.Exporter(Frontend);

    for (0..num_samples) |s| {
        std.debug.print("Generating sample {d}/{d}...\n", .{ s + 1, num_samples });

        var tree = try Tree.init(allocator, 1.0, 4, 8);
        defer tree.deinit();
        var field = try GaugeField.init(allocator, &tree);
        defer field.deinit();

        var psi = try FieldArena.init(allocator, 256);
        defer psi.deinit();
        var workspace = try FieldArena.init(allocator, 256);
        defer workspace.deinit();
        var ghosts = try GhostBuffer.init(allocator, 256);
        defer ghosts.deinit();

        // Randomize potential: 1 to 4 Coulomb centers
        current_potential.num_centers = rand.intRangeAtMost(usize, 1, 4);
        for (0..current_potential.num_centers) |i| {
            current_potential.centers[i] = .{
                0.0, // t (stationary potential)
                rand.float(f64) * 16.0, // x
                rand.float(f64) * 16.0, // y
                rand.float(f64) * 16.0, // z
            };
            current_potential.charges[i] = rand.float(f64) * 20.0 + 1.0;
        }

        // Initialize root block
        const b0 = try tree.insertBlockWithField(.{ 0, 0, 0, 0 }, 0, &psi);
        try field.syncWithTree(&tree);
        _ = workspace.allocSlot(); // Allocate matching workspace slot

        // Initial noise to seed the evolution
        const slot = tree.getFieldSlot(b0);
        const psi_data = psi.getSlot(slot);
        for (psi_data) |*v| {
            v.*[0] = Complex.init(rand.floatNorm(f64), rand.floatNorm(f64));
        }

        var H = HAMR.init(&tree, &field, 1.0, &randomPotential);
        defer H.deinit();

        H.normalizeAMR(&psi);

        // Relax for 50 steps to get "physical" ground-state-like correlations
        // This ensures the VAE learns physics, not just random noise.
        try H.evolveImaginaryTimeAMR(
            &psi,
            &workspace,
            &ghosts,
            0.01, // delta_tau
            50,   // num_steps
            10,   // normalize_interval
            10,   // adapt_interval
            0.5,  // adapt_threshold
            0.3   // adapt_hysteresis
        );

        // Export the resulting AMR mesh to SUN format
        var filename_buf: [256]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buf, "sample_{d:0>5}.sun", .{s});
        const full_path = try std.fs.path.join(allocator, &.{ output_dir, filename });
        defer allocator.free(full_path);

        const exp = Exp.init(.{ .export_psi = true, .export_full_psi = true, .export_links = true });
        var link_slices = try allocator.alloc([]const Frontend.LinkType, tree.blocks.items.len);
        defer allocator.free(link_slices);
        for (tree.blocks.items, 0..) |*block, idx| {
            if (block.block_index == std.math.maxInt(usize)) {
                link_slices[idx] = &[_]Frontend.LinkType{};
            } else {
                link_slices[idx] = field.getBlockLinks(idx) orelse &[_]Frontend.LinkType{};
            }
        }
        try exp.writeToFile(&tree, &psi, link_slices, full_path);
    }

    std.debug.print("Successfully generated {d} samples in '{s}'\n", .{ num_samples, output_dir });
}

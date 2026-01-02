const std = @import("std");
const su_n = @import("su_n");
const constants = su_n.constants;
const HamiltonianDiracAMR = su_n.physics.hamiltonian_dirac_amr.HamiltonianDiracAMR;
const AMRForce = su_n.physics.force_amr.AMRForce;
const gauge = su_n.gauge;
const Complex = std.math.Complex(f64);

// =============================================================================
// Benchmark Infrastructure
// =============================================================================

const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    total_ns: u64,
    mean_ns: u64,
    min_ns: u64,
    max_ns: u64,
    std_dev_ns: f64,
    ops_per_sec: f64,

    pub fn print(self: BenchmarkResult) void {
        std.debug.print("{s:<50} ", .{self.name});
        std.debug.print("{d:>6} iter  ", .{self.iterations});
        std.debug.print("{d:>10.2} ns/op  ", .{@as(f64, @floatFromInt(self.mean_ns))});
        std.debug.print("{d:>10.2} ops/s  ", .{self.ops_per_sec});
        std.debug.print("(std: {d:.1}%)\n", .{self.std_dev_ns / @as(f64, @floatFromInt(self.mean_ns)) * 100.0});
    }
};

fn benchmark(
    comptime name: []const u8,
    comptime warmup_iters: u64,
    comptime bench_iters: u64,
    comptime func: anytype,
) BenchmarkResult {
    var timer = std.time.Timer.start() catch unreachable;

    // Warmup
    for (0..warmup_iters) |_| {
        func();
    }

    // Benchmark - collect all timings
    var timings: [bench_iters]u64 = undefined;
    var total_ns: u64 = 0;

    for (0..bench_iters) |i| {
        timer.reset();
        func();
        const elapsed = timer.read();
        timings[i] = elapsed;
        total_ns += elapsed;
    }

    // Compute statistics
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    for (timings) |t| {
        min_ns = @min(min_ns, t);
        max_ns = @max(max_ns, t);
    }

    const mean_ns = total_ns / bench_iters;
    const mean_f: f64 = @floatFromInt(mean_ns);

    // Standard deviation
    var variance_sum: f64 = 0;
    for (timings) |t| {
        const diff = @as(f64, @floatFromInt(t)) - mean_f;
        variance_sum += diff * diff;
    }
    const std_dev_ns = @sqrt(variance_sum / @as(f64, @floatFromInt(bench_iters)));

    const ops_per_sec = @as(f64, @floatFromInt(bench_iters)) / (@as(f64, @floatFromInt(total_ns)) / 1e9);

    return .{
        .name = name,
        .iterations = bench_iters,
        .total_ns = total_ns,
        .mean_ns = mean_ns,
        .min_ns = min_ns,
        .max_ns = max_ns,
        .std_dev_ns = std_dev_ns,
        .ops_per_sec = ops_per_sec,
    };
}

// =============================================================================
// Dirac Hamiltonian AMR Benchmarks
// =============================================================================

const amr = su_n.amr;
const TestTopology4D = amr.topology.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });

const block_size = 16;
const N_gauge = 1;
const HAMR = HamiltonianDiracAMR(N_gauge, block_size, TestTopology4D);
const N_field = HAMR.field_dim;
const GT = HAMR.GaugeTreeType;
const Arena = HAMR.FieldArena;
const Ghosts = HAMR.GhostBufferType;

fn setupAMR(allocator: std.mem.Allocator) !struct {
    tree: *GT,
    psi: *Arena,
    workspace: *Arena,
    ghosts: *Ghosts,
    h: HAMR,
} {
    var tree_ptr = try allocator.create(GT);
    tree_ptr.* = try GT.init(allocator, 1.0, 2, 8);
    const block_idx = try tree_ptr.insertBlock(.{ 0, 0, 0, 0 }, 0);

    var psi_ptr = try allocator.create(Arena);
    psi_ptr.* = try Arena.init(allocator, 1);
    var workspace_ptr = try allocator.create(Arena);
    workspace_ptr.* = try Arena.init(allocator, 1);

    const slot = psi_ptr.allocSlot() orelse return error.OutOfMemory;
    tree_ptr.tree.assignFieldSlot(block_idx, slot);
    _ = workspace_ptr.allocSlot() orelse return error.OutOfMemory;

    const ghosts_ptr = try allocator.create(Ghosts);
    ghosts_ptr.* = try Ghosts.init(allocator, 1);

    const h = HAMR.init(tree_ptr, 1.0, 1.0, .none);
    return .{ .tree = tree_ptr, .psi = psi_ptr, .workspace = workspace_ptr, .ghosts = ghosts_ptr, .h = h };
}

fn benchDiracApply1Block() void {
    const allocator = std.heap.page_allocator;
    var setup = setupAMR(allocator) catch @panic("Setup failed");
    defer {
        setup.h.deinit();
        setup.tree.deinit();
        setup.psi.deinit();
        setup.workspace.deinit();
        setup.ghosts.deinit();
        allocator.destroy(setup.tree);
        allocator.destroy(setup.psi);
        allocator.destroy(setup.workspace);
        allocator.destroy(setup.ghosts);
    }

    setup.h.apply(setup.psi, setup.workspace, setup.ghosts, null) catch @panic("apply failed");
    std.mem.doNotOptimizeAway(setup.workspace);
}

fn benchMeasureEnergy1Block() void {
    const allocator = std.heap.page_allocator;
    var setup = setupAMR(allocator) catch @panic("Setup failed");
    defer {
        setup.h.deinit();
        setup.tree.deinit();
        setup.psi.deinit();
        setup.workspace.deinit();
        setup.ghosts.deinit();
        allocator.destroy(setup.tree);
        allocator.destroy(setup.psi);
        allocator.destroy(setup.workspace);
        allocator.destroy(setup.ghosts);
    }

    // Initialize with something non-zero
    setup.h.normalize(setup.psi);
    const energy = setup.h.measureEnergy(setup.psi, setup.workspace, setup.ghosts) catch @panic("measure failed");
    std.mem.doNotOptimizeAway(energy);
}

fn benchNormalize1Block() void {
    const allocator = std.heap.page_allocator;
    var setup = setupAMR(allocator) catch @panic("Setup failed");
    defer {
        setup.h.deinit();
        setup.tree.deinit();
        setup.psi.deinit();
        setup.workspace.deinit();
        setup.ghosts.deinit();
        allocator.destroy(setup.tree);
        allocator.destroy(setup.psi);
        allocator.destroy(setup.workspace);
        allocator.destroy(setup.ghosts);
    }

    setup.h.normalize(setup.psi);
    std.mem.doNotOptimizeAway(setup.psi);
}

// =============================================================================
// HMC (Gauge Force) Benchmarks
// =============================================================================

// Topology for smaller blocks
const HMCTopology4D = amr.topology.PeriodicTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });

// Use GaugeFrontend for HMC benchmarks: U(1), scalar, 4D, 4^4 block
const HMCFrontend = gauge.GaugeFrontend(N_gauge, 1, 4, 4, HMCTopology4D);
const Force = AMRForce(HMCFrontend);
const HMCTree = Force.GaugeTreeType;

/// HMC benchmark state - pre-allocated to avoid timing allocation overhead.
const HMCBenchState = struct {
    tree: *HMCTree,
    forces: *Force.AlgebraBuffer,
    momenta: *Force.AlgebraBuffer,
    prng: std.Random.DefaultPrng,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !HMCBenchState {
        var tree_ptr = try allocator.create(HMCTree);
        tree_ptr.* = try HMCTree.init(allocator, 1.0, 2, 8);
        // Insert 2x2x2 = 8 blocks for a more realistic benchmark
        for (0..2) |x| {
            for (0..2) |y| {
                for (0..2) |z| {
                    _ = try tree_ptr.insertBlock(.{ 0, x * 4, y * 4, z * 4 }, 0);
                }
            }
        }

        const n_blocks = tree_ptr.tree.blocks.items.len;

        // Allocate force buffer (AlgebraBuffer)
        var forces = try allocator.create(Force.AlgebraBuffer);
        forces.* = Force.AlgebraBuffer.init(allocator, n_blocks);
        try forces.ensureBlocks(n_blocks);

        // Allocate momentum buffer
        var momenta = try allocator.create(Force.AlgebraBuffer);
        momenta.* = Force.AlgebraBuffer.init(allocator, n_blocks);
        try momenta.ensureBlocks(n_blocks);

        return .{
            .tree = tree_ptr,
            .forces = forces,
            .momenta = momenta,
            .prng = std.Random.DefaultPrng.init(42),
            .allocator = allocator,
        };
    }

    fn deinit(self: *HMCBenchState) void {
        self.forces.deinit();
        self.allocator.destroy(self.forces);
        self.momenta.deinit();
        self.allocator.destroy(self.momenta);
        self.tree.deinit();
        self.allocator.destroy(self.tree);
    }

    /// Reset state for next benchmark iteration (no allocations).
    fn reset(self: *HMCBenchState) void {
        self.forces.setZero();
        self.momenta.sampleGaussian(self.prng.random());
    }
};

/// Global HMC benchmark state - initialized once, reused across iterations.
var hmc_state: ?HMCBenchState = null;

fn getHMCState() *HMCBenchState {
    if (hmc_state == null) {
        hmc_state = HMCBenchState.init(std.heap.page_allocator) catch @panic("HMC setup failed");
    }
    return &hmc_state.?;
}

fn benchComputeTreeForces() void {
    const state = getHMCState();
    state.reset();

    const beta: f64 = 6.0;
    const count = Force.computeTreeForces(state.tree, state.forces, beta) catch @panic("force compute failed");
    std.mem.doNotOptimizeAway(count);
}

fn benchLeapfrogStep() void {
    const state = getHMCState();
    state.reset();

    const beta: f64 = 6.0;
    const dt: f64 = 0.01;

    // Single leapfrog step (momenta already initialized by reset)
    const count = Force.leapfrogIntegrate(state.tree, state.momenta, state.forces, dt, 1, beta) catch @panic("leapfrog failed");
    std.mem.doNotOptimizeAway(count);
}

fn benchLeapfrogTrajectory() void {
    const state = getHMCState();
    state.reset();

    const beta: f64 = 6.0;
    const dt: f64 = 0.01;
    const n_steps: usize = 10;

    // Full trajectory (momenta already initialized by reset)
    const count = Force.leapfrogIntegrate(state.tree, state.momenta, state.forces, dt, n_steps, beta) catch @panic("leapfrog failed");
    std.mem.doNotOptimizeAway(count);
}

// =============================================================================
// Hydrogen Hyperfine Structure Benchmark (AMR)
// =============================================================================

/// Physical constants for hyperfine calculation
const HyperfineConstants = struct {
    const alpha: f64 = constants.fine_structure_constant;
    const m_p_over_m_e: f64 = 1836.15267343;
    const g_proton: f64 = 5.5856946893;
    const hartree_to_mhz: f64 = 6.579683920502e9;
    const psi_sq_0_theory_au: f64 = 1.0 / std.math.pi;
    const hfs_experimental_mhz: f64 = 1420.405751768;
    const e_binding_exact_hartree: f64 = constants.hydrogen_ground_state_hartree;
};

const HydrogenResult = struct {
    energy_total: f64,
    binding_energy_hartree: f64,
    psi_origin_avg: f64,
    density_3d_au: f64,
    hyperfine_mhz: f64,
    evolution_time_ns: u64,
    evolution_steps: usize,
    converged: bool,
};

fn runHydrogenGroundStateAMR(max_steps: usize) HydrogenResult {
    const allocator = std.heap.page_allocator;

    // Use 2x2x2 grid of blocks.
    // Spacing 20.0. Block size 16. Block extent 320.
    // Grid extent 640. (~4.6 a0). Center at 320.
    const spacing = 20.0;
    const grid_dim = 2;
    
    var tree_ptr = allocator.create(GT) catch @panic("OOM");
    tree_ptr.* = GT.init(allocator, spacing, 2, 8) catch @panic("OOM");

    var psi_ptr = allocator.create(Arena) catch @panic("OOM");
    psi_ptr.* = Arena.init(allocator, 32) catch @panic("OOM");

    var workspace_ptr = allocator.create(Arena) catch @panic("OOM");
    workspace_ptr.* = Arena.init(allocator, 32) catch @panic("OOM");

    var workspace2_ptr = allocator.create(Arena) catch @panic("OOM");
    workspace2_ptr.* = Arena.init(allocator, 32) catch @panic("OOM");

    const ghosts_ptr = allocator.create(Ghosts) catch @panic("OOM");
    ghosts_ptr.* = Ghosts.init(allocator, 32) catch @panic("OOM");

    var blocks = std.ArrayList(usize){};
    defer blocks.deinit(allocator);

    for (0..grid_dim) |x| {
        for (0..grid_dim) |y| {
            for (0..grid_dim) |z| {
                const idx = tree_ptr.insertBlock(.{ 0, x * block_size, y * block_size, z * block_size }, 0) catch @panic("OOM");
                blocks.append(allocator, idx) catch @panic("OOM");
                const slot = psi_ptr.allocSlot() orelse @panic("OOM");
                _ = workspace_ptr.allocSlot() orelse @panic("OOM");
                _ = workspace2_ptr.allocSlot() orelse @panic("OOM");
                tree_ptr.tree.assignFieldSlot(idx, slot);
            }
        }
    }
    
    defer {
        psi_ptr.deinit(); allocator.destroy(psi_ptr);
        workspace_ptr.deinit(); allocator.destroy(workspace_ptr);
        workspace2_ptr.deinit(); allocator.destroy(workspace2_ptr);
        ghosts_ptr.deinit(); allocator.destroy(ghosts_ptr);
        tree_ptr.deinit(); allocator.destroy(tree_ptr);
    }

    var h = HAMR.init(tree_ptr, 1.0, 1.0, .coulomb);
    defer h.deinit();
    
    const center_pos = @as(f64, @floatFromInt(block_size * grid_dim)) * spacing / 2.0;
    h.setCenter(.{center_pos, center_pos, center_pos, center_pos});

    const alpha_val = HyperfineConstants.alpha;
    
    for (blocks.items) |b_idx| {
        const slot = tree_ptr.tree.getFieldSlot(b_idx);
        const psi_data = psi_ptr.getSlot(slot);
        const block = &tree_ptr.tree.blocks.items[b_idx];
        
        for (0..HAMR.block_volume) |i| {
            const pos = block.getPhysicalPosition(i);
            
            const dx = pos[1] - center_pos;
            const dy = pos[2] - center_pos;
            const dz = pos[3] - center_pos;
            const r = @sqrt(dx*dx + dy*dy + dz*dz);
            const amp = @exp(-alpha_val * r);
            
            psi_data[i][0] = Complex.init(amp, 0);
            for (1..N_field) |c| psi_data[i][c] = Complex.init(0, 0);
        }
    }
    h.normalize(psi_ptr);

    var timer = std.time.Timer.start() catch unreachable;
    const result = h.evolveUntilConvergedAMR(psi_ptr, workspace_ptr, workspace2_ptr, ghosts_ptr, 0.01, max_steps, 1e-4, 10) catch @panic("evolve failed");
    const elapsed = timer.read();

    // Metrics
    const energy_total = result.energy;
    const binding_energy_natural = energy_total - 1.0;
    const binding_energy_hartree = binding_energy_natural / (alpha_val * alpha_val);

    var max_sq: f64 = 0.0;
    for (blocks.items) |b_idx| {
        const slot = tree_ptr.tree.getFieldSlot(b_idx);
        const psi_data = psi_ptr.getSlotConst(slot);
        for (0..HAMR.block_volume) |i| {
            var val_sq: f64 = 0;
            for (0..N_field) |c| val_sq += psi_data[i][c].magnitude() * psi_data[i][c].magnitude();
            if (val_sq > max_sq) max_sq = val_sq;
        }
    }
    const psi_0_sq_avg = max_sq; 
    
    const density_3d = psi_0_sq_avg * (block_size * spacing); 
    const a0 = 1.0 / alpha_val;
    const density_au = density_3d * (a0 * a0 * a0);
    
    const hyperfine = density_au * (4.0 * std.math.pi / 3.0) * (alpha_val * alpha_val) * (1.0/HyperfineConstants.m_p_over_m_e) * HyperfineConstants.g_proton * HyperfineConstants.hartree_to_mhz;

    return .{
        .energy_total = energy_total,
        .binding_energy_hartree = binding_energy_hartree,
        .psi_origin_avg = @sqrt(psi_0_sq_avg),
        .density_3d_au = density_au,
        .hyperfine_mhz = hyperfine,
        .evolution_time_ns = elapsed,
        .evolution_steps = result.steps,
        .converged = result.converged,
    };
}

fn printHydrogenResults(result: HydrogenResult) void {
    const print = std.debug.print;
    const energy_error = @abs(result.binding_energy_hartree - HyperfineConstants.e_binding_exact_hartree) / @abs(HyperfineConstants.e_binding_exact_hartree) * 100.0;
    
    print("\n", .{});
    print("Hydrogen Ground State (AMR 2x2x2 blocks):\n", .{});
    print("-" ** 80 ++ "\n", .{});
    print("  Total Energy:          {d:>10.6} m_e\n", .{result.energy_total});
    print("  Binding Energy:        {d:>10.6} Hartree (Target: -0.5, Error: {d:>6.2}%)\n", .{result.binding_energy_hartree, energy_error});
    print("  Hyperfine:             {d:>10.4} MHz\n", .{result.hyperfine_mhz});
    print("  Converged:             {s}\n", .{if (result.converged) "yes" else "no"});
    print("  Time:                  {d:>10.2} ms\n", .{@as(f64, @floatFromInt(result.evolution_time_ns)) / 1e6});
}

// =============================================================================
// Performance Regression Thresholds
// =============================================================================

/// Performance thresholds for regression detection.
/// Values are maximum allowed mean_ns (nanoseconds per operation).
/// Thresholds set at ~2x current performance to allow for system variance.
const PerfThresholds = struct {
    // Dirac AMR operations (16^4 block = 65536 sites)
    const dirac_apply_max_ns: u64 = 10_000_000; // 10ms max for 65k sites
    const measure_energy_max_ns: u64 = 10_000_000; // 10ms max
    const normalize_max_ns: u64 = 4_000_000; // 4ms max

    // HMC operations (8 blocks of 4^4 = 2048 sites total)
    const tree_forces_max_ns: u64 = 500_000; // 500μs max
    const leapfrog_step_max_ns: u64 = 1_000_000; // 1ms max
    const leapfrog_trajectory_max_ns: u64 = 10_000_000; // 10ms max for 10 steps

    // Minimum throughput (link-forces per second)
    const min_link_forces_per_sec: f64 = 20_000_000; // 20M link-forces/sec
};

fn checkPerfRegression(name: []const u8, mean_ns: u64, max_ns: u64) bool {
    if (mean_ns > max_ns) {
        std.debug.print("  ⚠️  REGRESSION: {s} took {d}ns, threshold is {d}ns\n", .{ name, mean_ns, max_ns });
        return true;
    }
    return false;
}

// =============================================================================
// Main
// =============================================================================

pub fn main() void {
    const print = std.debug.print;
    var regressions: usize = 0;

    print("\n", .{});
    print("================================================================================\n", .{});
    print("                         QED / Dirac AMR Benchmarks                             \n", .{});
    print("================================================================================\n", .{});
    print("\n", .{});

    const bench_apply = benchmark("HamiltonianDiracAMR.apply() 16^4 (65k sites)", 5, 20, benchDiracApply1Block);
    bench_apply.print();
    print("  -> {d:.2} million sites/sec\n", .{@as(f64, 65536.0) * bench_apply.ops_per_sec / 1e6});
    if (checkPerfRegression("dirac_apply", bench_apply.mean_ns, PerfThresholds.dirac_apply_max_ns)) regressions += 1;

    const bench_energy = benchmark("measureEnergy() 16^4", 5, 20, benchMeasureEnergy1Block);
    bench_energy.print();
    if (checkPerfRegression("measure_energy", bench_energy.mean_ns, PerfThresholds.measure_energy_max_ns)) regressions += 1;

    const bench_norm = benchmark("normalize() 16^4", 5, 20, benchNormalize1Block);
    bench_norm.print();
    if (checkPerfRegression("normalize", bench_norm.mean_ns, PerfThresholds.normalize_max_ns)) regressions += 1;

    print("\n", .{});
    print("================================================================================\n", .{});
    print("                         HMC / Gauge Force Benchmarks                           \n", .{});
    print("================================================================================\n", .{});
    print("\n", .{});

    const bench_force = benchmark("computeTreeForces() 8 blocks (4^4)", 3, 10, benchComputeTreeForces);
    bench_force.print();
    const sites_per_force = 8 * 256 * 4; // 8 blocks * 256 sites * 4 directions
    const link_forces_per_sec = @as(f64, @floatFromInt(sites_per_force)) * bench_force.ops_per_sec;
    print("  -> {d:.2} million link-forces/sec\n", .{link_forces_per_sec / 1e6});
    if (checkPerfRegression("tree_forces", bench_force.mean_ns, PerfThresholds.tree_forces_max_ns)) regressions += 1;
    if (link_forces_per_sec < PerfThresholds.min_link_forces_per_sec) {
        print("  ⚠️  REGRESSION: link-forces/sec {d:.0} < threshold {d:.0}\n", .{ link_forces_per_sec, PerfThresholds.min_link_forces_per_sec });
        regressions += 1;
    }

    const bench_leapfrog = benchmark("leapfrogStep() 8 blocks", 3, 10, benchLeapfrogStep);
    bench_leapfrog.print();
    if (checkPerfRegression("leapfrog_step", bench_leapfrog.mean_ns, PerfThresholds.leapfrog_step_max_ns)) regressions += 1;

    const bench_trajectory = benchmark("leapfrogIntegrate() 10 steps", 2, 5, benchLeapfrogTrajectory);
    bench_trajectory.print();
    if (checkPerfRegression("leapfrog_trajectory", bench_trajectory.mean_ns, PerfThresholds.leapfrog_trajectory_max_ns)) regressions += 1;

    print("\n", .{});
    print("================================================================================\n", .{});
    print("                    Hydrogen Hyperfine Structure (AMR)                          \n", .{});
    print("================================================================================\n", .{});

    const h_result = runHydrogenGroundStateAMR(500);
    printHydrogenResults(h_result);

    print("\n", .{});
    if (regressions > 0) {
        print("⚠️  {d} performance regression(s) detected!\n", .{regressions});
    } else {
        print("✓ All performance thresholds passed.\n", .{});
    }
    print("Benchmark complete.\n", .{});
}

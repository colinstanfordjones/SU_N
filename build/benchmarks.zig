const std = @import("std");
const Modules = @import("modules.zig");

pub const BenchmarkSpec = struct {
    name: []const u8,
    path: []const u8,
    description: []const u8,
};

pub const Config = struct {
    target: std.Build.ResolvedTarget,
    modules: Modules.ModuleRefs,
    mpi_step: ?*std.Build.Step = null,
};

pub const Collection = struct {
    all: []const *std.Build.Step,
    executables: []const *std.Build.Step.Compile,
};

/// All benchmark specifications
pub const specs = [_]BenchmarkSpec{
    .{
        .name = "qed-benchmarks",
        .path = "benches/qed_benchmarks.zig",
        .description = "QED performance benchmarks",
    },
    .{
        .name = "muonic-hydrogen-benchmark",
        .path = "benches/muonic_hydrogen_benchmark.zig",
        .description = "Muonic hydrogen ground state benchmark",
    },
    .{
        .name = "flux-benchmark",
        .path = "benches/flux_benchmark.zig",
        .description = "Flux Register and AMR evolution benchmark",
    },
    .{
        .name = "repartition-benchmark",
        .path = "benches/repartition_benchmark.zig",
        .description = "Entropy-weighted repartition benchmark",
    },
};

/// Register all benchmark executables and run steps
pub fn register(b: *std.Build, config: Config) !Collection {
    const allocator = b.allocator;

    var all_steps = std.ArrayListUnmanaged(*std.Build.Step){};
    defer all_steps.deinit(allocator);

    var executables = std.ArrayListUnmanaged(*std.Build.Step.Compile){};
    defer executables.deinit(allocator);

    inline for (specs) |spec| {
        const bench_module = b.createModule(.{
            .root_source_file = b.path(spec.path),
            .target = config.target,
            .optimize = .ReleaseFast, // Always optimize benchmarks
            .imports = &.{
                .{ .name = "su_n", .module = config.modules.su_n },
                .{ .name = "amr", .module = config.modules.amr },
                .{ .name = "math", .module = config.modules.math },
                .{ .name = "gauge", .module = config.modules.gauge },
                .{ .name = "physics", .module = config.modules.physics },
                .{ .name = "stats", .module = config.modules.stats },
            },
        });

        const exe = b.addExecutable(.{
            .name = spec.name,
            .root_module = bench_module,
        });
        if (config.mpi_step) |step| {
            exe.step.dependOn(step);
        }
        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        try all_steps.append(allocator, &run.step);
        try executables.append(allocator, exe);
    }

    return .{
        .all = try all_steps.toOwnedSlice(allocator),
        .executables = try executables.toOwnedSlice(allocator),
    };
}

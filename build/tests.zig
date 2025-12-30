const std = @import("std");
const Modules = @import("modules.zig");

pub const Config = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: Modules.ModuleRefs,
    test_filters: []const []const u8,
    mpi_step: ?*std.Build.Step = null,
    mpirun_path: ?[]const u8 = null,
};

pub const Collection = struct {
    all: []const *std.Build.Step,
    internal: []const *std.Build.Step,
    integration: []const *std.Build.Step,
    mpi: ?*std.Build.Step,
};

/// Register all test steps
pub fn register(b: *std.Build, config: Config) !Collection {
    const allocator = b.allocator;

    var all_steps = std.ArrayListUnmanaged(*std.Build.Step){};
    defer all_steps.deinit(allocator);

    var internal_steps = std.ArrayListUnmanaged(*std.Build.Step){};
    defer internal_steps.deinit(allocator);

    var integration_steps = std.ArrayListUnmanaged(*std.Build.Step){};
    defer integration_steps.deinit(allocator);

    // Internal tests (src/root.zig embedded tests)
    {
        const test_compile = b.addTest(.{
            .root_module = config.modules.su_n,
            .filters = config.test_filters,
        });
        if (config.mpi_step) |step| {
            test_compile.step.dependOn(step);
        }
        const run = b.addRunArtifact(test_compile);
        try all_steps.append(allocator, &run.step);
        try internal_steps.append(allocator, &run.step);
    }

    // Integration tests (tests/root.zig)
    {
        const test_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = config.target,
            .optimize = config.optimize,
            .imports = &.{
                .{ .name = "su_n", .module = config.modules.su_n },
                .{ .name = "amr", .module = config.modules.amr },
                .{ .name = "math", .module = config.modules.math },
                .{ .name = "gauge", .module = config.modules.gauge },
                .{ .name = "physics", .module = config.modules.physics },
                .{ .name = "stats", .module = config.modules.stats },
            },
        });

        const test_compile = b.addTest(.{
            .root_module = test_module,
            .filters = config.test_filters,
        });
        if (config.mpi_step) |step| {
            test_compile.step.dependOn(step);
        }
        const run = b.addRunArtifact(test_compile);
        try all_steps.append(allocator, &run.step);
        try integration_steps.append(allocator, &run.step);
    }

    var mpi_step: ?*std.Build.Step = null;
    if (config.mpirun_path) |mpirun_path| {
        const mpi_test_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = config.target,
            .optimize = config.optimize,
            .imports = &.{
                .{ .name = "su_n", .module = config.modules.su_n },
                .{ .name = "amr", .module = config.modules.amr },
                .{ .name = "math", .module = config.modules.math },
                .{ .name = "gauge", .module = config.modules.gauge },
                .{ .name = "physics", .module = config.modules.physics },
                .{ .name = "stats", .module = config.modules.stats },
            },
        });

        const test_compile = b.addTest(.{
            .root_module = mpi_test_module,
            .filters = &.{"mpi"},
        });
        if (config.mpi_step) |step| {
            test_compile.step.dependOn(step);
        }

        const run_mpi = b.addSystemCommand(&.{ mpirun_path, "-n", "2" });
        run_mpi.addFileArg(test_compile.getEmittedBin());
        mpi_step = &run_mpi.step;
    }

    return .{
        .all = try all_steps.toOwnedSlice(allocator),
        .internal = try internal_steps.toOwnedSlice(allocator),
        .integration = try integration_steps.toOwnedSlice(allocator),
        .mpi = mpi_step,
    };
}

const std = @import("std");

/// Module references for use by tests and benchmarks
pub const ModuleRefs = struct {
    su_n: *std.Build.Module,
    math: *std.Build.Module,
    gauge: *std.Build.Module,
    amr: *std.Build.Module,
    physics: *std.Build.Module,
    stats: *std.Build.Module,
    ai: *std.Build.Module,
    platform: *std.Build.Module,
    build_options: *std.Build.Module,
};

pub const Config = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    enable_ai: bool = false,
    enable_mpi: bool = false,
    mpi_include: ?[]const u8 = null,
    mpi_lib: ?[]const u8 = null,
    build_options: *std.Build.Module,
};

/// Create all modules with proper dependency graph
pub fn create(b: *std.Build, config: Config) ModuleRefs {
    // Math module (standalone, no dependencies)
    const math_mod = b.createModule(.{
        .root_source_file = b.path("src/math/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
    });

    // Constants module (standalone, shared by all physics modules)
    const constants_mod = b.createModule(.{
        .root_source_file = b.path("src/constants.zig"),
        .target = config.target,
        .optimize = config.optimize,
    });

    // Platform module (runtime + IO helpers)
    const platform_mod = b.createModule(.{
        .root_source_file = b.path("src/platform/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "build_options", .module = config.build_options },
        },
    });

    if (config.enable_mpi) {
        platform_mod.link_libc = true;
        platform_mod.linkSystemLibrary("mpi", .{});
        if (config.mpi_include) |path| {
            platform_mod.addIncludePath(.{ .cwd_relative = path });
        }
        if (config.mpi_lib) |path| {
            platform_mod.addLibraryPath(.{ .cwd_relative = path });
            platform_mod.addRPath(.{ .cwd_relative = path });
        }
    }

    // Stats module (depends on constants)
    const stats_mod = b.createModule(.{
        .root_source_file = b.path("src/stats/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "constants", .module = constants_mod },
        },
    });

    // AI module
    var ai_mod: *std.Build.Module = undefined;

    if (config.enable_ai) {
        ai_mod = b.createModule(.{
            .root_source_file = b.path("src/ai/root.zig"),
            .target = config.target,
            .optimize = config.optimize,
            .imports = &.{
                .{ .name = "math", .module = math_mod },
            },
        });

        // Add include paths for the C headers
        ai_mod.addIncludePath(b.path("external/onnxruntime/include/onnxruntime/core/session"));
        ai_mod.addIncludePath(b.path("external/faiss/c_api"));

        // Link system libraries
        ai_mod.linkSystemLibrary("onnxruntime", .{});
        ai_mod.linkSystemLibrary("faiss_c", .{});
        ai_mod.link_libc = true;
    } else {
        // Use dummy module
        ai_mod = b.createModule(.{
            .root_source_file = b.path("src/ai/dummy.zig"),
            .target = config.target,
            .optimize = config.optimize,
        });
    }

    // AMR module (domain-agnostic, depends only on math)
    const amr_mod = b.createModule(.{
        .root_source_file = b.path("src/amr/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math_mod },
            .{ .name = "platform", .module = platform_mod },
        },
    });

    // Gauge module (depends on math, amr for LinkOperators, constants)
    const gauge_mod = b.createModule(.{
        .root_source_file = b.path("src/gauge/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math_mod },
            .{ .name = "amr", .module = amr_mod },
            .{ .name = "constants", .module = constants_mod },
            .{ .name = "platform", .module = platform_mod },
        },
    });

    // Physics module (depends on amr, gauge, math, stats, constants)
    const physics_mod = b.createModule(.{
        .root_source_file = b.path("src/physics/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "amr", .module = amr_mod },
            .{ .name = "gauge", .module = gauge_mod },
            .{ .name = "math", .module = math_mod },
            .{ .name = "stats", .module = stats_mod },
            .{ .name = "constants", .module = constants_mod },
        },
    });

    // Main su_n module (aggregates all submodules)
    const su_n_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math_mod },
            .{ .name = "gauge", .module = gauge_mod },
            .{ .name = "amr", .module = amr_mod },
            .{ .name = "physics", .module = physics_mod },
            .{ .name = "stats", .module = stats_mod },
            .{ .name = "ai", .module = ai_mod },
            .{ .name = "constants", .module = constants_mod },
            .{ .name = "platform", .module = platform_mod },
        },
    });

    return .{
        .su_n = su_n_mod,
        .math = math_mod,
        .gauge = gauge_mod,
        .amr = amr_mod,
        .physics = physics_mod,
        .stats = stats_mod,
        .ai = ai_mod,
        .platform = platform_mod,
        .build_options = config.build_options,
    };
}

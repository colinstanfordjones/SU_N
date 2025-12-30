const std = @import("std");
const Modules = @import("build/modules.zig");
const Options = @import("build/options.zig");
const Mpich = @import("build/mpich.zig");
const Tests = @import("build/tests.zig");
const Benchmarks = @import("build/benchmarks.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Options
    const enable_ai = b.option(bool, "enable-ai", "Enable AI inference features (requires onnxruntime and faiss)") orelse false;
    const enable_mpi = b.option(bool, "enable-mpi", "Enable MPI support (MPICH)") orelse false;
    const mpi_include = b.option([]const u8, "mpi-include", "MPI include path (for mpi.h)");
    const mpi_lib = b.option([]const u8, "mpi-lib", "MPI library path (for libmpi)");
    const mpi_jobs = b.option(usize, "mpi-jobs", "MPICH build parallelism");

    const build_options = Options.create(b, .{
        .enable_mpi = enable_mpi,
    });

    var mpich_artifacts: ?Mpich.Artifacts = null;
    if (enable_mpi) {
        const prefix = b.pathJoin(&.{ b.install_path, "mpich" });
        const build_dir = b.pathJoin(&.{ b.install_path, "mpich-build" });
        const jobs = mpi_jobs orelse (std.Thread.getCpuCount() catch 4);

        mpich_artifacts = Mpich.setup(b, .{
            .source_dir = b.path("external/mpich"),
            .prefix = prefix,
            .build_dir = build_dir,
            .jobs = jobs,
        });
    }

    const effective_mpi_include = if (mpi_include) |path|
        path
    else if (mpich_artifacts) |mpich|
        mpich.include_dir
    else
        null;

    const effective_mpi_lib = if (mpi_lib) |path|
        path
    else if (mpich_artifacts) |mpich|
        mpich.lib_dir
    else
        null;
    const mpirun_path = if (mpich_artifacts) |mpich|
        b.pathJoin(&.{ mpich.prefix, "bin", "mpirun" })
    else
        null;

    // Create all modules
    const modules = Modules.create(b, .{
        .target = target,
        .optimize = optimize,
        .enable_ai = enable_ai,
        .enable_mpi = enable_mpi,
        .mpi_include = effective_mpi_include,
        .mpi_lib = effective_mpi_lib,
        .build_options = build_options,
    });

    // Static Library
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "su_n",
        .root_module = modules.su_n,
    });
    if (mpich_artifacts) |mpich| {
        lib.step.dependOn(mpich.step);
    }
    b.installArtifact(lib);

    // Shared Library (C API)
    const c_api_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "su_n", .module = modules.su_n },
        },
    });
    const shlib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "su_n_c",
        .root_module = c_api_mod,
    });
    if (mpich_artifacts) |mpich| {
        shlib.step.dependOn(mpich.step);
    }
    shlib.linkLibC();
    b.installArtifact(shlib);

    // Tests
    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "Skip tests that do not match any filter (e.g., -Dtest-filter=amr -Dtest-filter=ghost)",
    ) orelse &[0][]const u8{};

    const tests = Tests.register(b, .{
        .target = target,
        .optimize = optimize,
        .modules = modules,
        .test_filters = test_filters,
        .mpi_step = if (mpich_artifacts) |mpich| mpich.step else null,
        .mpirun_path = mpirun_path,
    }) catch |err| {
        std.debug.print("Failed to register tests: {}\n", .{err});
        return;
    };

    const test_step = b.step("test", "Run all tests");
    for (tests.all) |step| test_step.dependOn(step);

    const test_internal_step = b.step("test-internal", "Run internal tests only");
    for (tests.internal) |step| test_internal_step.dependOn(step);

    const test_integration_step = b.step("test-integration", "Run integration tests only");
    for (tests.integration) |step| test_integration_step.dependOn(step);

    const test_mpi_step = b.step("test-mpi", "Run MPI tests with mpirun -n 2");
    if (enable_mpi) {
        if (tests.mpi) |step| {
            test_mpi_step.dependOn(step);
        } else {
            const warn = b.addSystemCommand(&.{
                "bash",
                "-lc",
                "echo 'MPI enabled but mpirun was not configured' >&2; exit 1",
            });
            test_mpi_step.dependOn(&warn.step);
        }
    } else {
        const warn = b.addSystemCommand(&.{
            "bash",
            "-lc",
            "echo 'MPI disabled: pass -Denable-mpi=true to run MPI tests' >&2; exit 1",
        });
        test_mpi_step.dependOn(&warn.step);
    }

    // Benchmarks
    const benchmarks = Benchmarks.register(b, .{
        .target = target,
        .modules = modules,
        .mpi_step = if (mpich_artifacts) |mpich| mpich.step else null,
    }) catch |err| {
        std.debug.print("Failed to register benchmarks: {}\n", .{err});
        return;
    };

    const bench_step = b.step("bench", "Run all performance benchmarks");
    for (benchmarks.all) |step| bench_step.dependOn(step);

    // Individual benchmark steps
    inline for (Benchmarks.specs, 0..) |spec, i| {
        const step = b.step("bench-" ++ spec.name, spec.description);
        step.dependOn(benchmarks.all[i]);
    }

    // Tools
    const tools_step = b.step("tools", "Build all tools");

    const generate_dataset = b.addExecutable(.{
        .name = "generate-dataset",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_dataset.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "su_n", .module = modules.su_n },
            },
        }),
    });
    if (mpich_artifacts) |mpich| {
        generate_dataset.step.dependOn(mpich.step);
    }
    const install_generate_dataset = b.addInstallArtifact(generate_dataset, .{});
    tools_step.dependOn(&install_generate_dataset.step);

    const run_generate_dataset = b.addRunArtifact(generate_dataset);
    const run_generate_dataset_step = b.step("generate-dataset", "Run dataset generator");
    if (b.args) |args| {
        run_generate_dataset.addArgs(args);
    }
    run_generate_dataset_step.dependOn(&run_generate_dataset.step);
}

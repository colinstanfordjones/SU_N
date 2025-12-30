# Build System Specification

Module-based build system for su_n quantum field theory library.

## Overview

The build system uses a modular architecture with explicit dependency graph, separating module definitions, test registration, and benchmark configuration into focused files.

## Directory Structure

```
su_n/
├── build.zig           # Main build script (thin wrapper)
├── build.zig.zon       # Package metadata
└── build/
    ├── modules.zig     # Module definitions with dependencies
    ├── mpich.zig       # MPICH build step (configure/make/install)
    ├── options.zig     # Build options exported as a module
    ├── tests.zig       # Test registration
    └── benchmarks.zig  # Benchmark specifications
```

## Build Commands

### Libraries

```bash
zig build              # Build static and shared libraries
```

**Outputs:**
- `zig-out/lib/libsu_n.a` - Static library
- `zig-out/lib/libsu_n_c.so` - Shared library with C API

### Tests

```bash
zig build test                    # Run all tests (internal + integration)
zig build test-internal           # Run internal tests only (src/root.zig)
zig build test-integration        # Run integration tests only (tests/root.zig)
zig build test-mpi                # Run MPI tests with mpirun -n 2 (requires -Denable-mpi=true)
zig build test -- --test-filter "pattern"  # Filter tests by name
```

### Benchmarks

```bash
zig build bench                            # Run all benchmarks
zig build bench-qed-benchmarks             # QED performance benchmarks
zig build bench-muonic-hydrogen-benchmark  # Muonic hydrogen ground state
```

### Build Options

```bash
-Dtarget=<target>      # Cross-compilation target
-Doptimize=<mode>      # Debug/ReleaseSafe/ReleaseFast/ReleaseSmall
-Denable-mpi=true      # Enable MPI support (MPICH)
-Dmpi-include=<path>   # Optional MPI include path (mpi.h)
-Dmpi-lib=<path>       # Optional MPI library path (libmpi)
-Dmpi-jobs=<n>         # MPICH build parallelism (defaults to CPU count)
```

MPI support is optional; enable it with `-Denable-mpi=true`. The build system will compile MPICH from `external/mpich` when MPI is enabled and system headers/libs are not supplied.

## Module Dependency Graph

```
constants (standalone)
math (standalone)
platform (standalone)            # Runtime + IO helpers
stats ──→ constants
amr ──→ math, platform           # Domain-agnostic AMR infrastructure
gauge ──→ math, amr, constants, platform
physics ──→ amr, gauge, math, stats, constants
ai (optional) ──→ math
su_n aggregates all
```

Note: The AMR module is domain-agnostic and does NOT depend on gauge. The gauge module builds ON TOP of AMR.

### Module Details

| Module | Root File | Dependencies | Description |
|--------|-----------|--------------|-------------|
| `math` | `src/math/root.zig` | (none) | Linear algebra, matrices |
| `constants` | `src/constants.zig` | (none) | Physical constants and tolerances |
| `stats` | `src/stats/root.zig` | constants | Statistical utilities |
| `platform` | `src/platform/root.zig` | (none) | Runtime + IO helpers |
| `amr` | `src/amr/root.zig` | math, platform | Adaptive mesh refinement |
| `gauge` | `src/gauge/root.zig` | amr, math, constants, platform | Gauge groups, AMR gauge rules |
| `physics` | `src/physics/root.zig` | amr, gauge, math, stats, constants | AMR Hamiltonians and gauge-covariant dynamics |
| `ai` | `src/ai/root.zig` | math | Optional inference helpers |
| `su_n` | `src/root.zig` | all above | Top-level aggregator |

### Using Modules in Code

```zig
// In test or benchmark files, import directly:
const amr = @import("amr");
const gauge = @import("gauge");
const physics = @import("physics");
const math = @import("math");
const platform = @import("platform");

// Or use the aggregator:
const su_n = @import("su_n");
const amr = su_n.amr;
const gauge = su_n.gauge;
const platform = su_n.platform;
```

## File Reference

### build.zig

Main build script that orchestrates module creation, test registration, and benchmark setup.

```zig
const std = @import("std");
const Modules = @import("build/modules.zig");
const Tests = @import("build/tests.zig");
const Benchmarks = @import("build/benchmarks.zig");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create modules
    const config = .{ .target = target, .optimize = optimize };
    const modules = Modules.create(b, config);

    // Static library
    const lib = b.addStaticLibrary(.{
        .name = "su_n",
        .root_module = modules.su_n,
    });
    b.installArtifact(lib);

    // Shared library with C API
    const c_lib = b.addSharedLibrary(.{
        .name = "su_n_c",
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });
    c_lib.root_module.addImport("su_n", modules.su_n);
    c_lib.linkLibC();
    b.installArtifact(c_lib);

    // Register tests and benchmarks
    const tests = try Tests.register(b, config);
    const benchmarks = try Benchmarks.register(b, config);

    // Create build steps
    // ...
}
```

### build/modules.zig

Defines all modules with their dependencies.

```zig
pub const ModuleRefs = struct {
    su_n: *std.Build.Module,
    math: *std.Build.Module,
    gauge: *std.Build.Module,
    amr: *std.Build.Module,
    physics: *std.Build.Module,
    stats: *std.Build.Module,
    ai: *std.Build.Module,
    constants: *std.Build.Module,
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

pub fn create(b: *std.Build, config: Config) ModuleRefs {
    // Standalone modules (no dependencies)
    const math = b.createModule(.{
        .root_source_file = b.path("src/math/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
    });

    const constants = b.createModule(.{
        .root_source_file = b.path("src/constants.zig"),
        .target = config.target,
        .optimize = config.optimize,
    });

    const platform = b.createModule(.{
        .root_source_file = b.path("src/platform/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "build_options", .module = config.build_options },
        },
    });

    const stats = b.createModule(.{
        .root_source_file = b.path("src/stats/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "constants", .module = constants },
        },
    });

    const ai = if (config.enable_ai) b.createModule(.{
        .root_source_file = b.path("src/ai/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math },
        },
    }) else b.createModule(.{
        .root_source_file = b.path("src/ai/dummy.zig"),
        .target = config.target,
        .optimize = config.optimize,
    });

    // AMR depends on math + platform (domain-agnostic)
    const amr = b.createModule(.{
        .root_source_file = b.path("src/amr/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math },
            .{ .name = "platform", .module = platform },
        },
    });

    // Gauge depends on math, amr, constants, platform (gauge-specific layer on AMR)
    const gauge = b.createModule(.{
        .root_source_file = b.path("src/gauge/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math },
            .{ .name = "amr", .module = amr },
            .{ .name = "constants", .module = constants },
            .{ .name = "platform", .module = platform },
        },
    });

    // Physics depends on amr, gauge, math, stats
    const physics = b.createModule(.{
        .root_source_file = b.path("src/physics/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "amr", .module = amr },
            .{ .name = "gauge", .module = gauge },
            .{ .name = "math", .module = math },
            .{ .name = "stats", .module = stats },
            .{ .name = "constants", .module = constants },
        },
    });

    // Top-level aggregator
    const su_n = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math },
            .{ .name = "gauge", .module = gauge },
            .{ .name = "amr", .module = amr },
            .{ .name = "physics", .module = physics },
            .{ .name = "stats", .module = stats },
            .{ .name = "ai", .module = ai },
            .{ .name = "constants", .module = constants },
            .{ .name = "platform", .module = platform },
        },
    });

    return .{
        .su_n = su_n,
        .math = math,
        .gauge = gauge,
        .amr = amr,
        .physics = physics,
        .stats = stats,
        .ai = ai,
        .constants = constants,
        .platform = platform,
        .build_options = config.build_options,
    };
}
```

### build/tests.zig

Registers internal and integration tests.

```zig
pub const Collection = struct {
    all: []const *std.Build.Step,
    internal: []const *std.Build.Step,
    integration: []const *std.Build.Step,
};

pub fn register(b: *std.Build, config: Config) !Collection {
    const modules = Modules.create(b, config);

    // Internal tests (embedded in source files)
    const internal_tests = b.addTest(.{
        .root_module = modules.su_n,
        .test_filter = test_filter,
    });
    const run_internal = b.addRunArtifact(internal_tests);

    // Integration tests (tests/root.zig)
    const integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .test_filter = test_filter,
    });

    // Add all module imports to integration tests
    integration_tests.root_module.addImport("su_n", modules.su_n);
    integration_tests.root_module.addImport("amr", modules.amr);
    integration_tests.root_module.addImport("gauge", modules.gauge);
    integration_tests.root_module.addImport("physics", modules.physics);
    integration_tests.root_module.addImport("math", modules.math);
    integration_tests.root_module.addImport("stats", modules.stats);

    const run_integration = b.addRunArtifact(integration_tests);

    return .{
        .all = &.{ run_internal, run_integration },
        .internal = &.{run_internal},
        .integration = &.{run_integration},
    };
}
```

### build/benchmarks.zig

Registers benchmark executables.

```zig
pub const BenchmarkSpec = struct {
    name: []const u8,
    path: []const u8,
    description: []const u8,
};

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
};

pub const Collection = struct {
    all: []const *std.Build.Step,
    executables: []const *std.Build.Step.Compile,
};

pub fn register(b: *std.Build, config: Config) !Collection {
    const modules = Modules.create(b, config);

    var run_steps = std.ArrayList(*std.Build.Step).init(b.allocator);
    var executables = std.ArrayList(*std.Build.Step.Compile).init(b.allocator);

    inline for (specs) |spec| {
        // Benchmarks always use ReleaseFast
        const exe = b.addExecutable(.{
            .name = spec.name,
            .root_source_file = b.path(spec.path),
            .target = config.target,
            .optimize = .ReleaseFast,
        });

        // Add module imports
        exe.root_module.addImport("su_n", modules.su_n);
        exe.root_module.addImport("amr", modules.amr);
        exe.root_module.addImport("gauge", modules.gauge);
        exe.root_module.addImport("physics", modules.physics);
        exe.root_module.addImport("math", modules.math);
        exe.root_module.addImport("stats", modules.stats);

        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        try run_steps.append(&run.step);
        try executables.append(exe);
    }

    return .{
        .all = run_steps.items,
        .executables = executables.items,
    };
}
```

### build.zig.zon

Package metadata.

```zig
.{
    .name = .su_n,
    .version = "0.0.1",
    .minimum_zig_version = "0.15.2",
    .paths = .{""},
    .fingerprint = 0x9ddd77111478d18d,
}
```

## Adding New Modules

### Step 1: Create Module Directory

```
src/new_module/
├── root.zig    # Module exports
└── *.zig       # Implementation files
```

### Step 2: Update build/modules.zig

```zig
pub const ModuleRefs = struct {
    // ... existing modules ...
    new_module: *std.Build.Module,
};

pub fn create(b: *std.Build, config: Config) ModuleRefs {
    // ... existing modules ...

    const new_module = b.createModule(.{
        .root_source_file = b.path("src/new_module/root.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "math", .module = math },  // Add dependencies
        },
    });

    // Update su_n aggregator to include new module
    const su_n = b.createModule(.{
        // ...
        .imports = &.{
            // ... existing imports ...
            .{ .name = "new_module", .module = new_module },
        },
    });

    return .{
        // ... existing modules ...
        .new_module = new_module,
    };
}
```

### Step 3: Add to Test Registration

In `build/tests.zig`, add import for integration tests:

```zig
integration_tests.root_module.addImport("new_module", modules.new_module);
```

### Step 4: Export from su_n (src/root.zig)

```zig
pub const new_module = @import("new_module");
```

## Adding New Benchmarks

### Step 1: Create Benchmark File

```zig
// benches/new_benchmark.zig
const std = @import("std");
const physics = @import("physics");

pub fn main() !void {
    var timer = std.time.Timer{};
    timer.reset();

    // Run benchmark...

    const elapsed = timer.read();
    std.debug.print("Elapsed: {} ns\n", .{elapsed});
}
```

### Step 2: Register in build/benchmarks.zig

```zig
pub const specs = [_]BenchmarkSpec{
    // ... existing benchmarks ...
    .{
        .name = "new-benchmark",
        .path = "benches/new_benchmark.zig",
        .description = "Description of new benchmark",
    },
};
```

### Step 3: Run

```bash
zig build bench-new-benchmark
```

## Cross-Compilation

```bash
# Build for specific target
zig build -Dtarget=x86_64-linux-gnu
zig build -Dtarget=aarch64-macos
zig build -Dtarget=x86_64-windows-gnu

# List available targets
zig targets
```

## Optimization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `Debug` | No optimization, full debug info | Development |
| `ReleaseSafe` | Optimized, keeps safety checks | Production with safety |
| `ReleaseFast` | Maximum optimization | Benchmarks, HPC |
| `ReleaseSmall` | Optimize for size | Embedded, WASM |

```bash
zig build -Doptimize=ReleaseFast
zig build test -Doptimize=Debug
```

## C API

The `src/c_api.zig` file provides C-compatible exports:

```zig
// Minimal C API for FFI
export fn su2_get_element(
    generator: c_int,
    row: c_int,
    col: c_int,
    real_out: *f64,
    imag_out: *f64,
) void {
    // ...
}
```

**Usage from C:**

```c
#include <stdio.h>

extern void su2_get_element(int gen, int row, int col, double* re, double* im);

int main() {
    double re, im;
    su2_get_element(1, 0, 1, &re, &im);  // σ₁[0][1]
    printf("σ₁[0][1] = %f + %fi\n", re, im);
    return 0;
}
```

**Linking:**

```bash
gcc -o myprogram myprogram.c -L./zig-out/lib -lsu_n_c
```

## Best Practices

### Module Design

1. **Minimize dependencies**: Only import what you need
2. **Export via root.zig**: Single entry point per module
3. **No circular dependencies**: Graph must be acyclic

### Build Performance

1. **Parallel compilation**: Zig builds modules in parallel automatically
2. **Incremental builds**: Only changed modules recompile
3. **Use ReleaseFast for benchmarks**: Always set in benchmark registration

### Test Organization

1. **Internal tests**: Quick unit tests embedded in source
2. **Integration tests**: Cross-module tests in `tests/` directory
3. **Use test filter**: Focus on specific tests during development

# Frontend Interface Specification

Domain-agnostic interface for AMR-based simulations.

## Overview

The Frontend interface enables the AMR infrastructure to work with any domain-specific application (physics, finance, etc.) without hardcoded dependencies. A Frontend is a compile-time type that defines:

**Critical Architecture:** There is NO universal "Lattice" abstraction. Each domain builds its own lattice-like layer on AMR:
- The gauge module (`/src/gauge/`) IS the gauge-specific lattice layer (links, plaquettes, parallel transport)
- Other domains (Black-Scholes, fluids) would build their own layers with different structures

The Frontend interface defines:

- **Dimensionality**: Number of spacetime/parameter dimensions
- **Block geometry**: Sites per dimension within each block
- **Field type**: Data stored at each lattice site

The AMR backend (tree, blocks, ghost layers, adaptation) operates generically on any conforming Frontend.

## Interface Contract

A valid Frontend must declare these public constants:

```zig
const amr = @import("amr");

const MyFrontend = struct {
    /// Number of dimensions (1-8)
    pub const Nd: usize = 4;

    /// Sites per dimension (power of 2, >= 4)
    pub const block_size: usize = 16;

    /// Type stored at each lattice site
    pub const FieldType = f64;  // or Complex, [N]Complex, custom struct, etc.

    /// Grid topology with boundary conditions
    pub const Topology = amr.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
};
```

### Required Declarations

| Declaration | Type | Constraints | Description |
|-------------|------|-------------|-------------|
| `Nd` | `usize` | 1 <= Nd <= 8 | Number of dimensions |
| `block_size` | `usize` | Power of 2, >= 4 | Sites per dimension |
| `FieldType` | `type` | Any type | Data type at each site |
| `Topology` | `type` | Valid topology type | Boundary conditions per dimension |

### Validation

Use `validateFrontend()` at comptime to get clear error messages:

```zig
const amr = @import("amr");

// Compile-time validation (produces descriptive errors if invalid)
comptime amr.validateFrontend(MyFrontend);

// If validation passes, use with AMR types
const Tree = amr.AMRTree(MyFrontend);
const Arena = amr.FieldArena(MyFrontend);
```

**Error messages for invalid Frontends:**

| Missing/Invalid | Error Message |
|-----------------|---------------|
| No `Nd` | "Frontend must declare `pub const Nd: usize`" |
| `Nd` wrong type | "Frontend.Nd must be of type usize" |
| No `block_size` | "Frontend must declare `pub const block_size: usize`" |
| `block_size < 4` | "Frontend.block_size must be at least 4 for stencil operations" |
| `block_size` not power of 2 | "Frontend.block_size must be power of 2 for efficient indexing" |
| `Nd > 8` | "Frontend.Nd must be between 1 and 8 dimensions" |
| No `FieldType` | "Frontend must declare `pub const FieldType: type`" |
| No `Topology` | "Frontend must declare `pub const Topology: type`" |

## FrontendInfo Helper

`FrontendInfo(Frontend)` extracts and computes derived constants:

```zig
const amr = @import("amr");
const Info = amr.FrontendInfo(MyFrontend);

// Direct copies from Frontend
Info.Nd          // = Frontend.Nd
Info.block_size  // = Frontend.block_size
Info.FieldType   // = Frontend.FieldType
Info.Topology    // = Frontend.Topology

// Derived constants (computed at comptime)
Info.volume       // = block_size^Nd (sites per block)
Info.face_size    // = block_size^(Nd-1) (sites per ghost face)
Info.num_faces    // = 2 * Nd (faces per block)
Info.num_children // = 2^Nd (children when refining)
Info.field_size   // = @sizeOf(FieldType)
```

### Derived Constants Table

| Nd | block_size | volume | face_size | num_faces | num_children |
|----|------------|--------|-----------|-----------|--------------|
| 2 | 8 | 64 | 8 | 4 | 4 |
| 2 | 16 | 256 | 16 | 4 | 4 |
| 3 | 8 | 512 | 64 | 6 | 8 |
| 3 | 16 | 4,096 | 256 | 6 | 8 |
| 4 | 8 | 4,096 | 512 | 8 | 16 |
| 4 | 16 | 65,536 | 4,096 | 8 | 16 |

### Optional Frontend Hooks

Some higher-level modules look for additional declarations on the Frontend:

| Declaration | Type | Description |
|-------------|------|-------------|
| `field_dim` | `usize` | Number of scalar components per site (e.g., N_spinor * N_gauge). Used by physics and field-aware tooling. |

## Example Frontends

### Scalar Field (Heat Equation, Diffusion)

```zig
const amr = @import("amr");

/// 2D scalar field for heat equation simulations
pub const HeatFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 16;
    pub const FieldType = f64;  // Temperature at each site
    pub const Topology = amr.OpenTopology(2, .{ 1.0, 1.0 });  // Open boundaries
};
```

### Complex Field (Quantum Mechanics)

```zig
const std = @import("std");
const amr = @import("amr");
const Complex = std.math.Complex(f64);

/// 4D complex field for quantum mechanics (no gauge)
pub const QuantumFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = Complex;  // Wavefunction psi(x)
    pub const Topology = amr.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
};
```

### Multi-Component Field (Spinors, Multiplets)

```zig
const std = @import("std");
const amr = @import("amr");
const Complex = std.math.Complex(f64);

/// 4D Dirac spinor field (4 components)
pub const DiracFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = [4]Complex;  // 4-spinor at each site
    pub const Topology = amr.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });
};
```

### Financial Model (Black-Scholes)

```zig
const amr = @import("amr");

/// 2D option pricing: (time, underlying price)
/// Open boundaries: time only moves forward, price has boundaries
pub const BlackScholesFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 16;
    pub const FieldType = f64;  // Option value V(S, t)
    pub const Topology = amr.OpenTopology(2, .{ 1.0, 100.0 });
};

/// 3D option pricing: (time, price, volatility)
pub const StochasticVolFrontend = struct {
    pub const Nd: usize = 3;
    pub const block_size: usize = 16;
    pub const FieldType = f64;  // Option value V(S, sigma, t)
    pub const Topology = amr.OpenTopology(3, .{ 1.0, 100.0, 1.0 });
};
```

### Gauge Theory (see gauge.md for details)

For gauge theory applications, use the factories in the gauge module:

```zig
const amr = @import("amr");
const gauge = @import("gauge");

// Define topology first
const Topology4D = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });

// SU(3) with Dirac spinors (N_field = 4 spinors x 3 colors = 12)
const QCDFrontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology4D);

// U(1) scalar field (N_field = 1)
const U1ScalarFrontend = gauge.GaugeFrontend(1, 1, 4, 16, Topology4D);

// SU(2) with Weyl spinors (N_field = 2 spinors x 2 colors = 4)
const WeylSU2Frontend = gauge.GaugeFrontend(2, 2, 4, 16, Topology4D);

// Gauge frontends provide additional gauge-specific fields:
QCDFrontend.gauge_group_dim  // = 3 (SU(3))
QCDFrontend.spinor_dim       // = 4 (Dirac)
QCDFrontend.field_dim        // = 12 (frontend-defined)
QCDFrontend.LinkType         // = LinkVariable(3)
QCDFrontend.LinkOperators    // = LinkOperators(QCDFrontend)
QCDFrontend.Topology         // = Topology4D

// Field utilities
const psi_new = QCDFrontend.applyLinkToField(link, psi);
```

For non-gauge applications (testing, non-physics):

```zig
const amr = @import("amr");

// Scalar field (f64) with open boundaries
const OpenTopo3D = amr.OpenTopology(3, .{ 8.0, 8.0, 8.0 });
const Scalar = amr.frontend.ScalarFrontend(3, 8, OpenTopo3D);

// Complex scalar field with periodic boundaries
const PeriodicTopo4D = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
const ComplexScalar = amr.frontend.ComplexScalarFrontend(4, 16, PeriodicTopo4D);
```

## Usage with AMR Types

Once a Frontend is defined, use it with all AMR types:

```zig
const amr = @import("amr");

const MyFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = f64;
    pub const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
};

// All AMR types parameterized by Frontend
const Tree = amr.AMRTree(MyFrontend);
const Block = amr.AMRBlock(MyFrontend);
const Arena = amr.FieldArena(MyFrontend);
const Info = amr.FrontendInfo(MyFrontend);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Initialize tree with base spacing and bits per dimension
    var tree = try Tree.init(allocator, 1.0, 4, 8);  // 4 bits = max 16 blocks per dim
    defer tree.deinit();

    // Initialize field arena for max blocks
    var arena = try Arena.init(allocator, 256);
    defer arena.deinit();

    // Insert a block and get field storage
    const idx = try tree.insertBlockWithField(.{0, 0, 0, 0}, 0, &arena);
    const field_data = arena.getSlot(tree.getFieldSlot(idx));

    // field_data is []FieldType with Info.volume elements
    for (field_data) |*site| {
        site.* = 0.0;  // Initialize field
    }
}
```

## Design Rationale

### Why Compile-Time Interface?

Zig's compile-time duck typing provides:

1. **Zero runtime overhead**: All type information resolved at compile time
2. **Clear error messages**: `validateFrontend()` produces descriptive compile errors
3. **Type flexibility**: FieldType can be any type (scalar, array, struct)
4. **No vtable overhead**: No dynamic dispatch, all calls inlined

### Why Not Traits/Interfaces?

Traditional trait-based approaches would require:
- Runtime function pointers for operations
- Boxing/unboxing of field data
- Loss of SIMD optimization opportunities

The Frontend pattern preserves:
- Direct array access to field data
- Compile-time loop unrolling
- SIMD-friendly memory layout

### FieldType Flexibility

The `FieldType` can be:

| Type | Use Case | Memory Layout |
|------|----------|---------------|
| `f64` | Scalar fields (temperature, density) | 8 bytes/site |
| `Complex` | Quantum wavefunctions | 16 bytes/site |
| `[N]f64` | Vector fields | N × 8 bytes/site |
| `[N]Complex` | Spinors, multiplets | N × 16 bytes/site |
| Custom struct | Domain-specific data | @sizeOf(struct)/site |

The AMR infrastructure treats FieldType opaquely - it only needs to know the size for memory allocation and copying.

## Migration from Legacy API

### Before (Gauge-Coupled)

```zig
// Old: Four separate type parameters
const Tree = amr.AMRTree(4, 16, 1, 1);  // Nd, block_size, N_gauge, N_field
const Block = amr.AMRBlock(4, 16, 1, 1);
const Arena = amr.PsiArena(4, 16, 1);    // No N_gauge
```

### After (Frontend-Based)

```zig
// New: Single Frontend parameter
const MyFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = [1]Complex;  // N_field = 1
};

const Tree = amr.AMRTree(MyFrontend);
const Block = amr.AMRBlock(MyFrontend);
const Arena = amr.FieldArena(MyFrontend);  // Renamed from PsiArena
```

### Key Changes

| Old API | New API |
|---------|---------|
| `AMRTree(Nd, block_size, N_gauge, N_field)` | `AMRTree(Frontend)` |
| `AMRBlock(Nd, block_size, N_gauge, N_field)` | `AMRBlock(Frontend)` |
| `PsiArena(Nd, block_size, N_field)` | `FieldArena(Frontend)` |
| `Tree.n_field` | `Frontend.field_dim` (if provided) or use `FieldType` directly |
| `Tree.dimensions` | `FrontendInfo(Frontend).Nd` |
| `block.gauge` (embedded links) | Gauge links owned externally |
| `amr.LinkOperators` | `gauge.LinkOperators` (moved to gauge module) |

### Module Organization

The Frontend-based design keeps the AMR module domain-agnostic:

| Module | Contains |
|--------|----------|
| `amr` | Generic AMR infrastructure (tree, block, field_arena, ghost_buffer, operators) |
| `gauge` | `GaugeFrontend`, `LinkOperators`, `GaugeTree` |
| `physics` | `hamiltonian_amr`, `hamiltonian_dirac_amr`, `force_amr` |

## Related Specifications

- [amr.md](amr.md) - AMR module API reference
- [gauge.md](gauge.md) - Gauge module and GaugeFrontend

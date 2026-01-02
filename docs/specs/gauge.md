# Gauge Module Specification

Lattice Gauge Theory Infrastructure for Quantum Field Theory Calculations.

## Overview

The gauge module (`/src/gauge/`) implements AMR-based lattice gauge infrastructure. All electromagnetic and nuclear interactions emerge from the underlying gauge structure following a symmetry-first approach.

**Critical Architecture Note:** The gauge module IS the gauge-specific lattice layer built on AMR. There is no universal "Lattice" abstraction - each domain (gauge theory, Black-Scholes, fluids) builds its own layer on the domain-agnostic AMR infrastructure. The gauge module provides:
- Gauge groups (U(1), SU(2), SU(3)) and their algebra
- Link variables and their operations
- GaugeFrontend for AMR-based gauge simulations
- GaugeField + AMRTree composition for link storage and mesh management
- LinkOperators for gauge-covariant prolongation/restriction at refinement boundaries

## Design Philosophy

### Symmetry-First Approach

1. **Elementary charge**: e = g (U(1) coupling constant), not a separate constant
2. **Fine structure constant**: α = g²/(4π), derived from gauge coupling
3. **Physical emergence**: All QED quantities derive from gauge structure

### Key Principles

- **Locality of behavior**: Gauge-covariant derivatives keep gauge structure with physics
- **Compile-time safety**: Block sizes and matrix dimensions validated statically
- **Performance**: SIMD-friendly f64 precision, precomputed strides for hot paths

## Module Structure

```
src/gauge/
├── root.zig       # Module exports
├── u1.zig         # U(1) electromagnetic gauge group
├── su2.zig        # SU(2) weak isospin / Lorentz
├── su3.zig        # SU(3) color gauge group
├── link.zig       # Gauge link variables U_μ(x)
├── haar.zig       # Haar measure sampling
├── spacetime.zig  # Minkowski metric
├── frontend.zig   # GaugeFrontend for AMR gauge fields
├── operators.zig  # LinkOperators for gauge link prolongation/restriction on AMR
├── field.zig      # GaugeField - link storage and ghost buffers for AMRTree
└── repartition.zig # Gauge-field-aware MPI repartition helpers
```

## API Reference

### u1.zig - U(1) Electromagnetic Gauge Group

The Abelian gauge group for electromagnetism. Elementary charge emerges from the gauge coupling.

```zig
const U1 = @import("gauge").gauge_u1.U1;

// Fine structure constant (α ≈ 1/137)
const alpha = U1.fineStructure();

// Elementary charge e = g
const e = U1.elementaryCharge();

// Coupling constant g ≈ 0.303
const g = U1.coupling;

// Gauge transformation U = e^{igθ}
const U = U1.gaugeTransform(theta);

// Generator (T = 1 for U(1))
const T = U1.generator();
```

**Type: `U1` (struct)**

| Constant/Function | Type | Description |
|-------------------|------|-------------|
| `coupling` | `f64` | Gauge coupling g ≈ 0.303 |
| `fineStructure()` | `f64` | Returns α = g²/(4π) |
| `elementaryCharge()` | `f64` | Returns e = g |
| `identity()` | `Matrix1x1` | Identity element |
| `gaugeTransform(theta)` | `Matrix1x1` | Returns e^{igθ} |
| `rotation(theta)` | `Matrix1x1` | Returns e^{iθ} (pure rotation) |
| `generator()` | `Matrix1x1` | Returns T = 1 |

### su2.zig - SU(2) Gauge Group

Non-Abelian gauge group for weak isospin and Lorentz group representations.

```zig
const Su2 = @import("gauge").su2.Su2;

// Identity
const I = Su2.identity();

// Pauli matrices (generators of SU(2))
const s1 = Su2.sigma1();  // [0,1; 1,0]
const s2 = Su2.sigma2();  // [0,-i; i,0]
const s3 = Su2.sigma3();  // [1,0; 0,-1]
```

**Type: `Su2` (struct)**

| Function | Returns | Description |
|----------|---------|-------------|
| `identity()` | `Matrix2x2` | 2×2 identity matrix |
| `sigma1()` | `Matrix2x2` | Pauli σ₁ (off-diagonal real) |
| `sigma2()` | `Matrix2x2` | Pauli σ₂ (off-diagonal imaginary) |
| `sigma3()` | `Matrix2x2` | Pauli σ₃ (diagonal) |

**Properties:**
- All Pauli matrices are traceless and Hermitian
- Commutation relations: [σᵢ, σⱼ] = 2i εᵢⱼₖ σₖ
- det(σᵢ) = -1

**Dual Role:**
- Nuclear physics: Isospin doublets (proton/neutron)
- Relativistic physics: SU(2)⊗SU(2) Lorentz representations (Dirac spinors)

### su3.zig - SU(3) Color Gauge Group

Non-Abelian gauge group for quantum chromodynamics (QCD).

```zig
const Su3 = @import("gauge").su3.Su3;

// Identity
const I = Su3.identity();

// Gell-Mann matrices (8 generators of SU(3))
const l1 = Su3.lambda1();  // Acts on indices 1,2
const l2 = Su3.lambda2();
const l3 = Su3.lambda3();  // Diagonal in 1,2 subspace
const l4 = Su3.lambda4();  // Acts on indices 1,3
const l5 = Su3.lambda5();
const l6 = Su3.lambda6();  // Acts on indices 2,3
const l7 = Su3.lambda7();
const l8 = Su3.lambda8();  // Diagonal, normalized by 1/√3
```

**Type: `Su3` (struct)**

| Function | Returns | Description |
|----------|---------|-------------|
| `identity()` | `Matrix3x3` | 3×3 identity matrix |
| `lambda1()` - `lambda8()` | `Matrix3x3` | Gell-Mann matrices |

**Properties:**
- All Gell-Mann matrices are traceless and Hermitian
- Form basis of su(3) Lie algebra (8 generators)
- λ₈ has normalization factor 1/√3 for orthogonality

### link.zig - Gauge Link Variables

Link variables U_μ(x) ∈ SU(N) on lattice edges representing gauge connections.

```zig
const link = @import("gauge").link;
const LinkVariable = link.LinkVariable;

// Create link types for different gauge groups
const U1Link = LinkVariable(1);   // U(1)
const SU2Link = LinkVariable(2);  // SU(2)
const SU3Link = LinkVariable(3);  // SU(3)

// Constructors
const identity = SU2Link.identity();
const from_algebra = SU2Link.fromAlgebra(hermitian_matrix);
const from_gen = SU2Link.fromGenerator(theta, generator);

// Group operations
const product = U.mul(V);        // U × V
const adjoint = U.adjoint();     // U†
const inverse = U.inverse();     // U⁻¹ = U†

// Observables
const tr = U.trace();            // Tr(U)
const det = U.det();             // det(U)

// Action on matter fields
const psi_new = U.actOnVector(psi);  // ψ → Uψ

// Unitarization
const U_proj = U.unitarize();    // Project to SU(N)
const is_unitary = U.isUnitary(epsilon);
```

**Type: `LinkVariable(N)`**

| Function | Signature | Description |
|----------|-----------|-------------|
| `identity()` | `Self` | Identity element |
| `zero()` | `Self` | Zero matrix |
| `fromMatrix(m)` | `Self` | From raw matrix |
| `fromAlgebra(H)` | `Self` | U = exp(iH) from Hermitian H |
| `fromGenerator(theta, T)` | `Self` | U = exp(iθT) |
| `mul(other)` | `Self` | Group multiplication |
| `adjoint()` | `Self` | Hermitian conjugate |
| `inverse()` | `Self` | Inverse (= adjoint for unitary) |
| `add(other)` | `Self` | Matrix addition |
| `scale(s)` | `Self` | Complex scalar multiplication |
| `scaleReal(s)` | `Self` | Real scalar multiplication |
| `trace()` | `Complex` | Matrix trace |
| `traceReal()` | `f64` | Real part of trace |
| `det()` | `Complex` | Determinant |
| `norm()` | `f64` | Frobenius norm |
| `unitarize()` | `Self` | Project to SU(N) |
| `isUnitary(eps)` | `bool` | Check unitarity |
| `hasUnitDet(eps)` | `bool` | Check det = 1 |
| `actOnVector(v)` | `[N]Complex` | Fundamental representation |
| `actOnAdjoint(V)` | `Matrix` | Adjoint representation |
| `get(row, col)` | `Complex` | Get element |
| `set(row, col, val)` | `void` | Set element |

**Specialized Helpers:**
```zig
// U(1) helpers
const u1_link = link.u1FromAngle(theta);  // e^{iθ}
const theta = link.u1ToAngle(u1_link);

// SU(2) helpers
const su2_link = link.su2FromQuaternion(q);
const su2_link = link.su2FromAngle(theta, phi, psi);
```

### haar.zig - Random Gauge Configurations

Haar measure sampling for SU(N) groups.

```zig
const amr = @import("amr");
const gauge = @import("gauge");
const haar = gauge.haar;

// Random number generator
var rng = haar.Random.init(seed);
const u = rng.uniform();           // [0, 1)
const n = rng.normal();            // Standard normal
const z = rng.complexNormal();     // Complex normal

// Haar sampler for SU(N)
const Sampler = haar.HaarSampler(2);  // SU(2)
var sampler = Sampler.init(seed);

// Sample random SU(N) element with Haar measure
const U = sampler.sample();

// Sample near-identity element (for MC updates)
const U_near = sampler.sampleNearIdentity(epsilon);

// Initialize random gauge links for a GaugeField
const Topology = amr.PeriodicTopology(4, .{ 4.0, 4.0, 4.0, 4.0 });
const Frontend = gauge.GaugeFrontend(2, 1, 4, 4, Topology);  // SU(2), 1 spinor, 4D, 4^4
const Tree = amr.AMRTree(Frontend);
const Field = gauge.GaugeField(Frontend);

var tree = try Tree.init(allocator, 1.0, 2, 8);
defer tree.deinit();
var field = try Field.init(allocator, &tree);
defer field.deinit();
_ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
try field.syncWithTree(&tree);

if (field.getBlockLinksMut(0)) |links| {
    for (links) |*link| {
        link.* = sampler.sample();
    }
}
```

**Type: `HaarSampler(N)`**

| Function | Description |
|----------|-------------|
| `init(seed)` | Initialize sampler |
| `sample()` | Random SU(N) with Haar measure |
| `sampleNearIdentity(epsilon)` | U = exp(iεH) near identity |

**Sampling Algorithms:**
- **U(1)**: Uniform phase θ ∈ [0, 2π)
- **SU(2)**: Random quaternion on 3-sphere (normalized Gaussian)
- **SU(N)**: Gram-Schmidt on random complex matrix

### frontend.zig - Gauge Frontends for AMR

Factory functions for creating physics-specific AMR frontends.

```zig
const amr = @import("amr");
const gauge = @import("gauge");

// Define topology first (required by all frontends)
const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });

// Create a gauge theory frontend for SU(3) Dirac spinors
const Frontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology);
// N_gauge=3 (SU(3)), N_spinor=4 (Dirac), Nd=4, block_size=16, Topology

// Frontend satisfies AMR Frontend interface
comptime @import("amr").validateFrontend(Frontend);

// Standard Frontend fields
Frontend.Nd           // = 4
Frontend.block_size   // = 16
Frontend.FieldType    // = [12]Complex (4 spinors x 3 colors)
Frontend.Topology     // = Topology (boundary conditions)

// Gauge-specific extensions
Frontend.gauge_group_dim  // = 3 (SU(3))
Frontend.spinor_dim       // = 4 (Dirac)
Frontend.field_dim        // = 12 (frontend-defined)
Frontend.LinkType         // = LinkVariable(3)
Frontend.LinkOperators    // = LinkOperators(Frontend)
Frontend.volume           // = 16^4 = 65536
Frontend.links_per_block  // = 65536 * 4 = 262144

// Field utilities
const psi = Frontend.zeroField();
const norm_sq = Frontend.fieldNormSq(psi);
const psi_new = Frontend.applyLinkToField(link, psi);
const psi_adj = Frontend.applyLinkAdjointToField(link, psi);
const idx = Frontend.fieldIndex(spinor_idx, gauge_idx);
```

**Frontend Factories:**

| Factory | Description |
|---------|-------------|
| `GaugeFrontend(N_gauge, N_spinor, Nd, block_size, Topology)` | Gauge theory with spinors and topology |

For non-gauge scalar frontends, use `amr.frontend.ScalarFrontend(Nd, block_size, Topology)` or `amr.frontend.ComplexScalarFrontend(Nd, block_size, Topology)`.

### operators.zig - Link Operators for AMR

Gauge-covariant prolongation and restriction for link variables at refinement boundaries.

```zig
const amr = @import("amr");
const gauge = @import("gauge");

// Create Frontend for SU(3) Dirac spinors
const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
const Frontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology);  // SU(3), 4 spinor, 4D, 16^4

// Get link operators from Frontend
const LinkOps = gauge.LinkOperators(Frontend);
// Or equivalently: Frontend.LinkOperators

// Prolongate: split coarse link (spanning 2a) into two fine links
const fine_links = LinkOps.prolongateLink(coarse_link);
// U(1): exact angle splitting, exp(iθ) -> exp(iθ/2) * exp(iθ/2)
// SU(N): geodesic midpoint, U^{1/2} ≈ (I + U)/2, unitarized

// Restrict: combine fine links via path-ordered product
const coarse_link = LinkOps.restrictLink(fine1, fine2);
// U_coarse = U_fine1 * U_fine2

// Bulk operations for ghost faces
LinkOps.prolongateLinkFace(coarse_links, fine_links, face_dim);
LinkOps.restrictLinkFace(fine_links, coarse_links, face_dim);
```

**Type: `LinkOperators(Frontend)`**

The Frontend must provide:
- `Nd`: Number of spacetime dimensions
- `block_size`: Sites per dimension
- `gauge_group_dim`: Dimension of gauge group (N for SU(N))
- `LinkType`: The gauge link type
- `face_size`: block_size^(Nd-1)
- `num_children`: 2^Nd

**Why in gauge module?**
Link operators require gauge group structure (matrix logarithm for U(1) splitting, unitarization for SU(N)). The AMR module is domain-agnostic and doesn't know about gauge groups.

### field.zig - GaugeField

Stateless link storage and ghost management for AMR meshes. `GaugeField` does **not**
wrap `AMRTree`; it is attached to a tree and kept in sync after refinement, repartition,
or reordering.

```zig
const amr = @import("amr");
const gauge = @import("gauge");

// Create Frontend for SU(3) Dirac spinors
const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
const Frontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology);  // SU(3), 4 spinor, 4D, 16^4
const Tree = amr.AMRTree(Frontend);
const Field = gauge.GaugeField(Frontend);

var tree = try Tree.init(allocator, 1.0, 4, 8);  // spacing=1.0, bits_per_dim=4
defer tree.deinit();
var field = try Field.init(allocator, &tree);
defer field.deinit();

// Insert blocks, then sync link storage
_ = try tree.insertBlock(.{0, 0, 0, 0}, 0);
try field.syncWithTree(&tree);

// Access links
const link = field.getLink(block_idx, site, mu);
field.setLink(block_idx, site, mu, new_link);

// Fill link ghost layers before cross-block operations
try field.fillGhosts(&tree);
```

**Type: `GaugeField(Frontend)`**

The Frontend must be a `GaugeFrontend` with `gauge_group_dim` and `LinkType`.

**Exports:**
| Export | Description |
|--------|-------------|
| `FrontendType` | The Frontend type |
| `TreeType` | The underlying `amr.AMRTree(Frontend)` |
| `LinkType` | Gauge link type |
| `EdgeArenaType` | Link storage arena (`amr.EdgeArena`) |
| `EdgeGhostFaces` | Link ghost storage type |
| `Policy` | Link ghost exchange policy (`LinkGhostPolicy`) |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `init(allocator, tree)` | Initialize link arena + ghosts for a tree |
| `initWithOptions(allocator, tree, link_spec)` | Initialize with custom link exchange spec |
| `syncWithTree(tree)` | Allocate links/ghosts for new blocks |
| `reorder(perm)` | Reorder link/ghost storage after `tree.reorder()` |
| `getLink(block_idx, site, mu)` | Get link U_mu(x) |
| `setLink(block_idx, site, mu, link)` | Set link U_mu(x) |
| `getBlockLinks(block_idx)` | Read-only link slice for a block |
| `getBlockLinksMut(block_idx)` | Mutable link slice for a block |
| `fillGhosts(tree)` | Fill all link ghost faces (local + MPI) |
| `writeCheckpoint(writer)` | Append link payload after tree checkpoint |
| `readCheckpoint(allocator, tree, reader)` | Restore links for an existing tree |

**Threaded kernel execution uses `AMRTree.apply` + `ApplyContext`:**

Gauge kernels execute through the underlying tree. Field ghosts are handled by
`AMRTree.apply` when `ApplyContext.field_ghosts` is set; link ghosts must be filled
explicitly via `GaugeField.fillGhosts`.

```zig
const ApplyContext = amr.ApplyContext(Frontend);

var ctx = ApplyContext.init(&tree);
ctx.setEdges(&field.arena, &field.ghosts);
try field.fillGhosts(&tree);

try tree.apply(&kernel, &ctx);
```

Neighbor discovery is resolved at runtime by the underlying tree (no cached neighbor arrays).
Gauge kernels should rely on ghost buffers for boundary data.

The kernel must implement:
```zig
fn execute(self: *Self, block_idx: usize, block: *const Block, ctx: *ApplyContext) void
```

**Key Pattern: Tests Use GaugeField + AMRTree**

For gauge-related tests, use the gauge module types alongside AMR infrastructure:
```zig
const gauge = @import("gauge");
const Field = gauge.GaugeField(Frontend);
const FieldArena = amr.FieldArena(Frontend);
```

**Morton Reordering:**

After mesh adaptation or repartition:
1. Call `tree.reorder()` to get the permutation map.
2. Call `field.reorder(perm)` to remap link storage.
3. Call `field.ghosts.invalidateAll()` or `field.fillGhosts(&tree)` before boundary reads.

Call after mesh adaptation when `AdaptResult.changed` is true:
```zig
const result = try adaptation.adaptMesh(...);
if (result.changed) {
    const perm = try tree.reorder();
    defer tree.allocator.free(perm);
    try field.reorder(perm);
}
```

**Warning:** This invalidates all external pointers to blocks, field data, and link slices.

Reference: `src/gauge/field.zig` for implementation.

### spacetime.zig - Minkowski Geometry

Minkowski spacetime structures for relativistic calculations.

```zig
const spacetime = @import("gauge").spacetime;

// Minkowski metric: (+, -, -, -)
const v_lowered = spacetime.Metric.apply(v);  // (t,x,y,z) → (t,-x,-y,-z)

// Four-vectors
const FourVector = spacetime.FourVector;
const p = FourVector.init(E, px, py, pz);

// Minkowski inner product: a^μ η_μν b^ν
const dot = p.dot(q);

// Squared magnitude: s² = v · v
const s2 = p.interval();
// s² > 0: timelike
// s² = 0: lightlike
// s² < 0: spacelike

// Vector addition
const sum = p.add(q);
```

## Usage Examples

### Creating a GaugeField + Tree

```zig
const amr = @import("amr");
const gauge = @import("gauge");

const Topology = amr.PeriodicTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });
const Frontend = gauge.GaugeFrontend(2, 1, 4, 8, Topology);  // SU(2), 1 spinor, 4D, 8^4
const Tree = amr.AMRTree(Frontend);
const Field = gauge.GaugeField(Frontend);

var tree = try Tree.init(allocator, 1.0, 3, 8);
defer tree.deinit();
var field = try Field.init(allocator, &tree);
defer field.deinit();
_ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
try field.syncWithTree(&tree);

// Set a specific link
const U = gauge.link.LinkVariable(2).fromGenerator(0.1, gauge.su2.Su2.sigma3());
field.setLink(0, site, mu, U);
```

### Computing Wilson Action

```zig
const amr = @import("amr");
const gauge = @import("gauge");

const Topology = amr.PeriodicTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });
const Frontend = gauge.GaugeFrontend(3, 1, 4, 8, Topology);  // SU(3), 1 spinor, 4D, 8^4
const Tree = amr.AMRTree(Frontend);
const Field = gauge.GaugeField(Frontend);
const LinkOps = gauge.LinkOperators(Frontend);
const beta = 6.0;  // β = 2N/g² for SU(3)

var tree = try Tree.init(allocator, 1.0, 3, 8);
defer tree.deinit();
var field = try Field.init(allocator, &tree);
defer field.deinit();
_ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
try field.syncWithTree(&tree);

// Initialize links with Haar samples
var sampler = gauge.haar.HaarSampler(3).init(12345);
if (field.getBlockLinksMut(0)) |links| {
    for (links) |*link| {
        link.* = sampler.sample();
    }
}

try field.fillGhosts(&tree);
const action = LinkOps.wilsonAction(&tree, &field, beta);
const avg_plaq = LinkOps.averagePlaquetteTree(&tree, &field);
```

### Gauge-Covariant Laplacian

```zig
const amr = @import("amr");
const gauge = @import("gauge");

const Topology = amr.PeriodicTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });
const Frontend = gauge.GaugeFrontend(1, 1, 4, 8, Topology);  // U(1), 1 spinor, 4D, 8^4
const Tree = amr.AMRTree(Frontend);
const Field = gauge.GaugeField(Frontend);
const LinkOps = gauge.LinkOperators(Frontend);
const Ghosts = amr.GhostBuffer(Frontend);

var tree = try Tree.init(allocator, 1.0, 3, 8);
defer tree.deinit();
var field = try Field.init(allocator, &tree);
defer field.deinit();

var arena = try amr.FieldArena(Frontend).init(allocator, 16);
defer arena.deinit();
_ = try tree.insertBlockWithField(.{ 0, 0, 0, 0 }, 0, &arena);
try field.syncWithTree(&tree);

var ghosts = try Ghosts.init(allocator, 16);
defer ghosts.deinit();
try ghosts.ensureForTree(&tree);
try tree.fillGhostLayers(&arena, ghosts.slice(tree.blocks.items.len));

try field.fillGhosts(&tree); // Link ghosts for backward transport

const slot = tree.getFieldSlot(0);
const psi = arena.getSlot(slot);

if (ghosts.get(0)) |ghost_faces| {
    const Nd = Frontend.Nd;
    var ghost_slices: [2 * Nd][]const Frontend.FieldType = undefined;
    for (0..2 * Nd) |face| {
        ghost_slices[face] = ghost_faces[face][0..];
    }

    const spacing = tree.blocks.items[0].spacing;
    const lap = LinkOps.covariantLaplacianSite(&tree, &field, 0, site, psi, ghost_slices, spacing);
    _ = lap;
}
```

### Fine Structure Constant Derivation

```zig
const U1 = gauge.gauge_u1.U1;

// All derived from gauge coupling g
const g = U1.coupling;                    // ≈ 0.303
const alpha = U1.fineStructure();         // = g²/(4π) ≈ 1/137
const e = U1.elementaryCharge();          // = g

// This is the symmetry-first approach:
// Elementary charge IS the gauge coupling, not a separate constant
```

## Module Dependencies

```
spacetime.zig (standalone - Minkowski geometry)
    ↓
u1.zig, su2.zig, su3.zig (gauge groups, use math module)
    ↓
link.zig (uses math, defines LinkVariable(N))
    ↓
haar.zig (HaarSampler, uses link)
frontend.zig (GaugeFrontend, uses link, validates as AMR Frontend)
operators.zig (LinkOperators, uses link for gauge-covariant prolongation + plaquettes)
field.zig (GaugeField, link storage + ghost buffers for AMRTree)
repartition.zig (MPI repartition helpers for gauge links)
    ↓
physics modules (hamiltonian_amr.zig, hamiltonian_dirac_amr.zig, force_amr.zig)
```

## Best Practices

### Gauge Invariance

1. **Use covariant derivatives**: Never use ordinary derivatives on gauge-coupled fields
2. **Check unitarity**: Periodically verify `isUnitary()` during long simulations
3. **Project after updates**: Call `projectToSU()` to maintain SU(N) manifold

### Performance

1. **Reuse ghost buffers**: Call `fillGhosts()` only after link updates
2. **Batch link updates**: Use `getBlockLinksMut()` to update links in bulk
3. **Avoid per-site allocs**: Use `FieldArena` and `amr.GhostBuffer` for fields

### Numerical Stability

1. **Gram-Schmidt unitarization**: More stable than polar decomposition
2. **Determinant fixing**: Ensure det(U) = 1 after unitarization
3. **Small update steps**: Use `sampleNearIdentity` for MC updates

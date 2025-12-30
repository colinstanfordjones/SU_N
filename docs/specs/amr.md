# AMR Module Specification

Adaptive Mesh Refinement for multi-scale simulations with domain-agnostic Frontend interface.

## Overview

The AMR module (`/src/amr/`) implements block-structured adaptive mesh refinement that is **domain-agnostic**. The core infrastructure works with any physics frontend (gauge theory, Black-Scholes finance, etc.). The architecture follows these principles:

- **Frontend-parameterized design**: All AMR types take a `Frontend` type that defines dimensionality, block size, and field type
- **Linear octree organization**: Blocks indexed by Morton keys with a flat block list
- **Morton indexing**: `reorder()` sorts blocks by Morton index for cache-optimal memory access
- **Gauge-agnostic blocks**: Blocks store metadata only; gauge links are Frontend-owned
- **Namespace-style API**: Ghost filling and adaptation use pure functions (Zig 0.15 compatible)
- **Push Model**: Two-phase ghost communication (pull + push) for refinement boundaries
- **Threaded apply pipeline**: `AMRTree.apply` owns a persistent work-stealing pool and overlaps interior compute with ghost exchange.
- **Kernel Pattern**: State (fields, mesh) is owned by AMR; physics logic is provided as a kernel to `AMRTree.apply`, using `executeInterior`/`executeBoundary` and optional ghost hooks.
- **Checkpoint/Restart**: `platform.checkpoint` snapshots AMR tree/arena (+ optional gauge links) for exact restarts; MPI shard metadata is rebuilt on load.

See also: `docs/specs/amr/apply.md`, `docs/specs/amr/ghost_exchange.md`, `docs/specs/amr/examples.md`, `docs/specs/amr/mpi.md`, `docs/specs/amr/linear_octree.md`, `docs/specs/amr/checkpoint.md`, and `docs/specs/amr/multigrid_flux_register.md`.

## Module Structure

```
src/amr/                    # Domain-agnostic AMR infrastructure
├── root.zig                # Module exports
├── frontend.zig            # Frontend interface definition and validation
├── topology.zig            # Grid topology with configurable boundaries
├── morton.zig              # N-dimensional Morton (Z-order) encoding
├── field_arena.zig         # Pre-allocated field storage (generic FieldType)
├── ghost_buffer.zig        # Ghost face storage for fields
├── tree.zig                # 2^Nd-tree structure and block management
├── block.zig               # Fixed-size gauge-agnostic blocks
├── ghost.zig               # Ghost layer filling
├── adaptation.zig          # Mesh adaptation
└── operators.zig           # Prolongation and restriction for generic FieldType

src/gauge/                  # Gauge-specific AMR extensions
├── frontend.zig            # GaugeFrontend for AMR gauge fields
├── operators.zig           # LinkOperators for gauge link prolongation/restriction
└── tree.zig                # GaugeTree - AMR tree with encapsulated link storage

src/physics/                # Physics modules using AMR
└── force_amr.zig           # HMC force calculation using GaugeTree
```

## Frontend Interface

The Frontend is a compile-time type that defines the domain-specific configuration:

```zig
const amr = @import("amr");

const MyFrontend = struct {
    pub const Nd: usize = 4;           // Number of dimensions
    pub const block_size: usize = 16;  // Sites per dimension (must be power of 2, >= 4)
    pub const FieldType = [4]Complex;  // Type stored at each lattice site
    pub const Topology = amr.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });  // Boundary conditions
};
```

**Required declarations:**
| Declaration | Type | Description |
|-------------|------|-------------|
| `Nd` | `usize` | Number of dimensions (1-8) |
| `block_size` | `usize` | Sites per dimension (power of 2, >= 4) |
| `FieldType` | `type` | Type stored at each lattice site |
| `Topology` | `type` | Grid topology with boundary conditions |

**Optional declarations (used by higher-level modules):**
- `field_dim` - Number of scalar components per site (e.g., N_spinor * N_gauge)

**Validation:**
```zig
const frontend = @import("amr").frontend;

// Validate at compile-time (produces clear error messages)
comptime frontend.validateFrontend(MyFrontend);

// Get derived constants
const Info = frontend.FrontendInfo(MyFrontend);
// Info.volume = block_size^Nd
// Info.face_size = block_size^(Nd-1)
// Info.num_faces = 2 * Nd
// Info.num_children = 2^Nd
// Info.Topology = Frontend.Topology
```

## Linear Octree + Morton Block List

`AMRTree` stores blocks in a flat list and indexes them with a Morton key map:
- `BlockKey = { level, morton }` encodes the refinement level and Morton index.
- `block_index` maps `BlockKey -> block_idx` for fast lookup.
- `reorder()` compacts the list, sorts by Morton, rebuilds the map, and returns a permutation map.

Blocks do not cache neighbor indices. Neighbor relationships are resolved on demand:
- `neighborInfo(block_idx, face)` returns same-level or coarser neighbors (push model).
- `collectFineNeighbors(block_idx, face, out)` enumerates finer neighbors across a face.

This keeps the tree and MPI ownership keyed by stable Morton identifiers while allowing the block list
to be compacted or reordered without pointer chasing.

Adaptation note: `adaptMesh` auto-batches refinement/coarsening using an arena scratch buffer sized
to the current block list. This avoids per-block heap churn but can OOM on very large trees.

## API Reference

### topology.zig - Grid Topology

Defines boundary behavior for AMR grids. The topology interface allows frontends to specify periodic, open, or mixed boundaries per dimension.

```zig
const amr = @import("amr");

// Fully periodic topology (all dimensions wrap)
const Periodic = amr.PeriodicTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });

// Fully open topology (Dirichlet-like boundaries)
const Open = amr.OpenTopology(4, .{ 8.0, 8.0, 8.0, 8.0 });

// Mixed topology (periodic in some dimensions, open in others)
const Mixed = amr.GridTopology(4, .{
    .boundary = .{ .periodic, .periodic, .periodic, .open },
    .domain_size = .{ 8.0, 8.0, 8.0, 8.0 },
});

// Coordinate wrapping
const wrapped = Periodic.wrapCoordinate(9.5, 0);  // Returns 1.5
const invalid = Open.wrapCoordinate(9.5, 0);      // Returns null

// Full coordinate array wrapping
const coords = Periodic.wrapCoordinates(.{ 9.0, -1.0, 0.5, 0.5 });  // Wraps all

// Boundary type queries
const is_periodic = Periodic.isFullyPeriodic();  // true
const is_open = Open.isFullyOpen();              // true
```

**Types:**

| Type | Description |
|------|-------------|
| `BoundaryType` | Enum: `.open` or `.periodic` |
| `TopologyConfig(Nd)` | Configuration struct with `boundary` and `domain_size` arrays |
| `GridTopology(Nd, config)` | Compile-time topology type with coordinate wrapping |
| `PeriodicTopology(Nd, domain_size)` | Convenience: fully periodic topology |
| `OpenTopology(Nd, domain_size)` | Convenience: fully open topology |

**GridTopology Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `wrapCoordinate` | `(f64, comptime dim) ?f64` | Wrap single coordinate (null if open and out of bounds) |
| `wrapCoordinateRuntime` | `(f64, dim) ?f64` | Runtime dimension version |
| `wrapCoordinates` | `([Nd]f64) ?[Nd]f64` | Wrap all coordinates |
| `neighborEpsilon` | `(spacing: f64, dim: usize) f64` | Epsilon used for negative-face neighbor probing |
| `isFullyPeriodic` | `() bool` | Check if all dimensions periodic |
| `isFullyOpen` | `() bool` | Check if all dimensions open |

**Domain Size Coordination:**

The `domain_size` must align with the tree's block layout:
```
domain_size[d] = base_spacing * block_size * num_blocks_per_dimension[d]
```
For example, with `base_spacing=1.0`, `block_size=4`, and 2 blocks per dimension: `domain_size = { 8.0, 8.0, 8.0, 8.0 }`

### morton.zig - Morton (Z-order) Encoding

Maps N-dimensional coordinates to/from 1D Morton indices via bit interleaving.

```zig
const morton = @import("amr").morton;

// Encode coordinates to index
const index = morton.encode(3, .{ x, y, z }, 4);

// Decode index to coordinates
const coords = morton.decode(3, index, 4);
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `encode` | `(comptime Nd: usize, [Nd]usize, u8) u64` | Coordinates to Morton index |
| `decode` | `(comptime Nd: usize, u64, u8) [Nd]usize` | Morton index to coordinates |

`morton.BlockKey` packs the Morton code and refinement level for stable block indexing across ranks.

### field_arena.zig - Pre-allocated Field Storage

Zero-allocation storage for field data using a free-list allocator. Works with any `FieldType` defined by the Frontend.

```zig
const amr = @import("amr");

const MyFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = f64;  // Or any type: Complex, [4]Complex, etc.
};

const Arena = amr.FieldArena(MyFrontend);

var arena = try Arena.init(allocator, max_blocks);
defer arena.deinit();

// Allocate a slot
const slot = arena.allocSlot() orelse return error.Full;

// Access field data
const field_data = arena.getSlot(slot);
// ... use field_data[site] ...

// Return slot when done
arena.freeSlot(slot);
```

**Type: `FieldArena(Frontend)`**

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(Allocator, max_blocks) !Self` | Initialize with capacity |
| `deinit` | `(*Self) void` | Free memory |
| `allocSlot` | `(*Self) ?usize` | Allocate slot (null if full) |
| `freeSlot` | `(*Self, slot) void` | Return slot to free list |
| `getSlot` | `(*Self, slot) []FieldType` | Get mutable data slice |
| `getSlotConst` | `(*const Self, slot) []const FieldType` | Get const data slice |
| `allocatedCount` | `(*const Self) usize` | Number of allocated slots |
| `zeroSlot` | `(*Self, slot) void` | Zero all data in slot |
| `defragmentWithOrder` | `(*Self, []usize, usize) !void` | Reorder field data to match Morton-sorted blocks |

**Constants (from Frontend):**
- `volume` - Total sites per block (`block_size^Nd`)
- `Field` - Alias for `FieldType`

### ghost_buffer.zig - Ghost Face Storage

Optional ghost face storage per block index. Keeps ghost buffers within AMR so
physics/gauge layers can remain stateless.

```zig
const amr = @import("amr");
const Tree = amr.AMRTree(MyFrontend);
const Ghosts = amr.GhostBuffer(MyFrontend);

// Assume tree and field_arena are already initialized.
var ghosts = try Ghosts.init(allocator, max_blocks);
defer ghosts.deinit();

try ghosts.ensureForTree(&tree);
amr.ghost.fillGhostLayers(Tree, &tree, &field_arena, ghosts.slice(tree.blocks.items.len));

// Type alias for direct ghost face access
const faces: *Ghosts.GhostFaces = ghosts.get(block_idx).?;
// GhostFaces = [num_faces][face_size]FieldType
```

**Type: `GhostBuffer(Frontend)`**

| Type Alias | Description |
|------------|-------------|
| `GhostFaces` | `[num_faces][face_size]FieldType` - Ghost face array type |

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(Allocator, max_blocks) !Self` | Initialize with capacity |
| `deinit` | `(*Self) void` | Free memory |
| `ensureCapacity` | `(*Self, max_blocks) !void` | Grow capacity |
| `ensureForTree` | `(*Self, *Tree) !void` | Allocate for all active blocks |
| `get` | `(*const Self, block_idx) ?*GhostFaces` | Get ghost faces for block |
| `slice` | `(*const Self, len) []?*GhostFaces` | Get slice of ghost face pointers |

### tree.zig - AMR Tree Structure

Sparse forest of 2^Nd-trees for organizing blocks in Nd-dimensional space. The domain is defined by a collection of root blocks at Level 0, stored in a spatial hash map. This allows for disjoint or non-cubic domains.

```zig
const amr = @import("amr");

const MyFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = f64;
};

const Tree = amr.AMRTree(MyFrontend);

var tree = try Tree.init(allocator, base_spacing, bits_per_dim, max_level);
defer tree.deinit();

// Insert blocks
const idx = try tree.insertBlock(.{0, 0, 0, 0}, level);

// With field slot allocation
const idx2 = try tree.insertBlockWithField(.{0, 0, 0, 0}, level, &field_arena);

// Refine block into 2^Nd children
try tree.refineBlock(idx);

// Iterate over blocks
var iter = tree.blockIterator();
while (iter.next()) |block| {
    // Process block
}
```

**Type: `AMRTree(Frontend)`**

**Associated Types:**
- `BlockType = AMRBlock(Frontend)`
- `FieldArenaType = FieldArena(Frontend)`
- `FrontendType = Frontend`
- `FieldType = Frontend.FieldType`

**Nested Types:**

```zig
const BlockKey = struct {
    level: u8,
    morton: u64,
};

const NeighborInfo = struct {
    block_idx: usize,  // maxInt if none
    level_diff: i8,    // -1: coarser, 0: same (finer neighbors return none)

    pub fn exists(self) bool;
};
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(Allocator, f64, u8, u8) !Self` | Initialize empty forest (no default root) |
| `initWithOptions` | `(Allocator, f64, u8, u8, ExchangeOptions) !Self` | Initialize with custom exchange spec |
| `deinit` | `(*Self) void` | Free memory |
| `insertBlock` | `(*Self, [Nd]usize, u8) !usize` | Insert at origin, level |
| `insertBlockWithField` | `(*Self, [Nd]usize, u8, *FieldArena) !usize` | Insert with field slot |
| `refineBlock` | `(*Self, usize) !void` | Split into 2^Nd children |
| `blockIterator` | `(*const Self) BlockIterator` | Iterate all blocks |
| `apply` | `(*Self, kernel, inputs, outputs, ghosts) !void` | Threaded kernel execution with pipelined ghost exchange |
| `blockCount` | `(*const Self) usize` | Number of blocks |
| `getBlock` | `(*Self, usize) ?*Block` | Get block by index |
| `getPhysicalOrigin` | `(*const Self, *Block) [Nd]f64` | Block physical coordinates |
| `getMortonIndex` | `(*const Self, [Nd]usize, u8) u64` | Morton index for position |
| `blockKeyFromOrigin` | `(*const Self, [Nd]usize, u8) BlockKey` | Morton key from origin/level |
| `blockKeyForBlock` | `(*const Self, *Block) BlockKey` | Morton key for a block |
| `findBlockByKey` | `(*const Self, BlockKey) ?usize` | Lookup block index from Morton key |
| `neighborInfo` | `(*const Self, usize, usize) NeighborInfo` | Neighbor query for face (same/coarse only) |
| `collectFineNeighbors` | `(*const Self, usize, usize, *[max_fine_neighbors]usize) usize` | Collect finer neighbors across a face |
| `getFieldSlot` | `(*const Self, usize) usize` | Get field arena slot for block |
| `hasFieldSlot` | `(*const Self, usize) bool` | Check if block has field data |
| `invalidateBlock` | `(*Self, usize) void` | Remove block from Morton index and mark invalid |
| `reorder` | `(*Self) ![]usize` | Reorder blocks by Morton index for cache locality |

`AMRTree.init` constructs a persistent work-stealing pool used by `apply` to schedule tasks across worker threads.

## Threaded Apply (AMRTree.apply)

`AMRTree.apply` runs kernels on the AMR mesh with a pipelined, threaded schedule:

1. Ghost pull (field ghosts, plus optional kernel ghost hooks)
2. Interior compute (overlaps with ghost pull)
3. Ghost push (field ghosts, plus optional kernel ghost hooks)
4. Boundary compute (after ghosts are complete)

The tree owns a persistent work-stealing pool; task metadata is allocated from a per-group arena to avoid churn.

**Kernel requirements:**
- `executeInterior(self, block_idx, block, inputs, outputs, ghosts)`
- `executeBoundary(self, block_idx, block, inputs, outputs, ghosts)`

**Optional ghost hooks (all-or-none):**
- `ghostPrepare(self) !bool` (return true if extra exchange needed)
- `ghostPull(self, block_idx)`
- `ghostPush(self, block_idx)`
- `ghostFinalize(self)`

**Example:**
```zig
const Tree = amr.AMRTree(Frontend);
const Arena = amr.FieldArena(Frontend);
const Ghosts = amr.GhostBuffer(Frontend);

const Kernel = struct {
    tree: *const Tree,

    pub fn executeInterior(
        self: *Kernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*Ghosts,
    ) void {
        self.executeRegion(.interior, block_idx, block, inputs, outputs, ghosts);
    }

    pub fn executeBoundary(
        self: *Kernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*Ghosts,
    ) void {
        self.executeRegion(.boundary, block_idx, block, inputs, outputs, ghosts);
    }

    const SiteRegion = enum { interior, boundary };
    fn executeRegion(
        self: *Kernel,
        region: SiteRegion,
        block_idx: usize,
        block: *const Tree.BlockType,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*Ghosts,
    ) void {
        _ = self;
        _ = block;
        const slot = self.tree.getFieldSlot(block_idx);
        const in = inputs.getSlotConst(slot);
        const out = outputs.getSlot(slot);
        _ = in;
        _ = out;
        _ = ghosts;
        _ = region;
    }
};

var kernel = Kernel{ .tree = &tree };
try ghosts.ensureForTree(&tree);
try tree.apply(&kernel, &arena_in, &arena_out, &ghosts);
```

### block.zig - AMR Block Implementation

Fixed-size gauge-agnostic blocks with AMR metadata and coordinate operations. Blocks store only structural metadata; field data (including gauge links) is Frontend-owned.

```zig
const amr = @import("amr");

const MyFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = f64;
};

const Block = amr.AMRBlock(MyFrontend);

// Create a block (no heap allocation needed)
var block = Block.init(level, .{0, 0, 0, 0}, spacing);

// Coordinate operations (compile-time optimized)
const idx = Block.getLocalIndex(.{8, 8, 8, 8});
const coords = Block.getLocalCoords(idx);

// Boundary detection
const on_boundary = Block.isOnBoundary(coords);
const on_face = Block.isOnFace(coords, 0);  // Face 0 = +dimension0

// Interior neighbor (wraps at boundaries)
const neighbor_idx = Block.localNeighborFast(idx, 0);  // +dimension0
```

**Type: `AMRBlock(Frontend)`**

**Constants:**
- `dimensions = Nd`
- `size = block_size`
- `volume = block_size^Nd` (e.g., 65536 for 16^4)
- `ghost_face_size = block_size^(Nd-1)` (e.g., 4096 for 16^3)
- `num_ghost_faces = 2*Nd` (8 faces in 4D)
- `strides` - Precomputed row-major strides for indexing
- `Field = FieldType`

**Fields:**
```zig
level: u8                      // Refinement level (0 = coarsest)
origin: [Nd]usize              // Global coordinates at this level
spacing: f64                   // base_spacing / 2^level
block_index: usize             // Index in tree
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(u8, [Nd]usize, f64) Self` | Create block (no allocation) |
| `getLocalIndex` | `([Nd]usize) usize` | Coordinates to index (inline) |
| `getLocalCoords` | `(usize) [Nd]usize` | Index to coordinates (inline) |
| `isOnBoundary` | `([Nd]usize) bool` | Check if on any boundary |
| `isOnFace` | `([Nd]usize, face_idx) bool` | Check if on specific face |
| `localNeighborFast` | `(usize, face_idx) usize` | Interior neighbor (inline, wraps) |
| `localNeighborRuntime` | `(usize, usize) usize` | Runtime version |
| `getGlobalCoords` | `(*const Self, usize) [Nd]usize` | Local to global coords |
| `getGhostIndex` | `([Nd]usize, face_idx) usize` | Ghost face index |
| `extractBoundaryFace` | `([]FieldType, face_idx, []FieldType) void` | Extract face data |
| `getExtent` | `(*const Self) [Nd]f64` | Physical block size |
| `getPhysicalPosition` | `(*const Self, usize) [Nd]f64` | Site physical coords |

Blocks store no neighbor arrays; use `AMRTree.neighborInfo` and `AMRTree.collectFineNeighbors` for adjacency.

## Ghost Layer Filling (ghost.zig)

Push Model ghost communication: fine blocks pull from coarse, then push to coarse.

```zig
const ghost = @import("amr").ghost;
const Ghosts = amr.GhostBuffer(MyFrontend);
const Tree = amr.AMRTree(MyFrontend);

// Allocate ghost storage for all blocks
var ghosts = try Ghosts.init(allocator, max_blocks);
defer ghosts.deinit();

// Fill all ghosts (two-phase: pull + push)
try ghosts.ensureForTree(&tree);
ghost.fillGhostLayers(Tree, &tree, &field_arena, ghosts.slice(tree.blocks.items.len));
```

**Push Model Architecture:**
1. **Pull phase**: Each block pulls from same-level neighbors and interpolates from coarse parents
2. **Push phase**: Fine blocks push restricted data to coarse neighbors at boundaries

**Split exchange for overlap:**
```zig
try ghosts.ensureForTree(&tree);
const ghost_ptrs = ghosts.slice(tree.blocks.items.len);

ghost.beginGhostExchange(Tree, &tree, &field_arena, ghost_ptrs);
// ... do interior compute here ...
ghost.finishGhostExchange(Tree, &tree, &field_arena, ghost_ptrs);
// ... do boundary compute here ...
```

## Mesh Adaptation (adaptation.zig)

Gradient-based refinement and coarsening with automatic field prolongation/restriction.

```zig
const adaptation = @import("amr").adaptation;
const Tree = amr.AMRTree(MyFrontend);

// Adapt entire mesh based on gradient thresholds
const result = try adaptation.adaptMesh(Tree, &tree, &field_arena, threshold, hysteresis);

// Reorder after adaptation for cache locality
if (result.changed) {
    const perm = try tree.reorder();
    defer tree.allocator.free(perm);
}
```

**AdaptResult Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `refined` | `usize` | Number of blocks refined |
| `coarsened` | `usize` | Number of block groups coarsened |
| `refine_failed` | `usize` | Refinement failures |
| `coarsen_failed` | `usize` | Coarsening failures |
| `changed` | `bool` | True if any refinement or coarsening occurred |

## Morton Reordering

After mesh adaptation, blocks may be scattered in memory (new blocks appended at the end). The `reorder()` method sorts blocks by Morton index to restore cache locality for stencil operations.

**AMRTree.reorder()**

Sorts blocks by Morton index and rebuilds internal indices:
1. Removes invalid (deleted) blocks
2. Sorts remaining blocks by Morton index
3. Rebuilds the Morton key map and permutes field_slots

Returns a permutation map `perm` where `perm[old_idx] = new_idx`; callers must free it.

**FieldArena.defragmentWithOrder()**

Reorders field data to match the Morton-sorted block array. Call after `tree.reorder()`:
1. Copies field data to match new block ordering
2. Zero-initializes slots for blocks without field data
3. Rebuilds the free list for compacted layout

**Pointer Invalidation Warning:**

Both methods invalidate all external pointers:
- `*Block` pointers obtained before `reorder()` are invalid
- Field slice references from `getSlot()` are invalid after `defragmentWithOrder()`
- GhostBuffer references should be re-obtained after reordering

**Typical Usage Pattern:**

Reference: `src/amr/tree.zig` for `reorder()` implementation.
Reference: `src/amr/field_arena.zig` for `defragmentWithOrder()` implementation.

## Field Operators (amr/operators.zig)

Inter-level data transfer with norm preservation for generic FieldType. Domain-agnostic.

```zig
const AMROperators = @import("amr").AMROperators;
const Ops = AMROperators(MyFrontend);

// Prolongate field from coarse to fine
Ops.prolongateField(coarse_field, fine_field);

// Restrict field from fine to coarse
Ops.restrictField(fine_field, coarse_field);
```

## Gauge-Specific: LinkOperators (gauge/operators.zig)

Gauge-covariant link prolongation/restriction for AMR refinement boundaries. Located in gauge module since it requires gauge group structure.

```zig
const amr = @import("amr");
const gauge = @import("gauge");

// Create Frontend for SU(3) Dirac spinors
const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
const Frontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology);  // SU(3), 4 spinor, 4D, 16^4

// Get link operators from Frontend
const LinkOps = gauge.LinkOperators(Frontend);
// Or equivalently: Frontend.LinkOperators

// Split coarse link (spanning 2a) into two fine links (each spanning a)
const fine_links = LinkOps.prolongateLink(coarse_link);
// For U(1): exact angle splitting, exp(iθ) -> exp(iθ/2) * exp(iθ/2)
// For SU(N): geodesic midpoint approximation

// Combine fine links into coarse via path-ordered product
const coarse_link = LinkOps.restrictLink(fine1, fine2);
```

**Type: `LinkOperators(Frontend)`**

The Frontend must provide: `Nd`, `block_size`, `LinkType`, `gauge_group_dim`, `face_size`, `num_children`.

## Gauge-Specific: GaugeFrontend (gauge/frontend.zig)

Factory for creating physics-specific AMR frontends.

```zig
const amr = @import("amr");
const gauge = @import("gauge");

// Define topology first
const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });

// Create frontend for SU(3) with Dirac spinors
const Frontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology);
// N_gauge=3 (SU(3)), N_spinor=4 (Dirac), Nd=4, block_size=16, Topology

// Frontend provides:
// - Standard: Nd, block_size, FieldType = [12]Complex, Topology
// - Gauge extensions: gauge_group_dim, spinor_dim, field_dim (frontend-defined), LinkType
// - Utilities: applyLinkToField, fieldNormSq, zeroField, unitField
// - Link operators: Frontend.LinkOperators

// For non-gauge use, use AMR frontends
const OpenTopo = amr.OpenTopology(3, .{ 8.0, 8.0, 8.0 });
const Scalar = amr.frontend.ScalarFrontend(3, 8, OpenTopo);  // 3D, block_size=8, f64
const PeriodicTopo = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
const ComplexScalar = amr.frontend.ComplexScalarFrontend(4, 16, PeriodicTopo);  // 4D, Complex
```

## Gauge-Specific: GaugeTree (gauge/tree.zig)

High-level API for gauge theory on AMR meshes. Wraps AMR tree with automatic link storage and ghost management.

```zig
const amr = @import("amr");
const gauge = @import("gauge");

const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
const Frontend = gauge.GaugeFrontend(3, 4, 4, 16, Topology);  // SU(3), Dirac, 4D
const GT = gauge.GaugeTree(Frontend);

var tree = try GT.init(allocator, 1.0, 4, 8);
defer tree.deinit();

// GaugeTree exports FieldArena for matter fields
var psi_arena = try GT.FieldArena.init(allocator, 256);
defer psi_arena.deinit();

// Insert blocks (links allocated automatically)
_ = try tree.insertBlock(.{0, 0, 0, 0}, 0);

// Access links
const link = tree.getLink(block_idx, site, mu);
tree.setLink(block_idx, site, mu, new_link);

// Fill ghost layers before cross-block operations
try tree.fillGhosts();

// Compute plaquette (ghost handling is internal)
const plaq = tree.computePlaquette(block_idx, site, mu, nu);
```

Threaded kernel execution uses the underlying AMR tree:
```zig
try tree.tree.apply(&kernel, inputs, outputs, ghosts);
```

If a kernel needs link ghosts, implement the optional ghost hooks by forwarding to
`prepareLinkGhostExchange`, `fillGhostsPull`, `fillGhostsPush`, and `finalizeLinkGhostExchange`.

**Type: `GaugeTree(Frontend)`**

**Exports:**
- `FieldArena` - Re-exports `amr.FieldArena(Frontend)` for convenience
- `FrontendType`, `TreeType`, `BlockType`, `LinkType`, `FieldType`
- `volume`, `dimensions`, `gauge_group_dim`, `N_field`, `num_faces`

**Key Pattern: Tests Use GaugeTree, Not Raw AMR**

For gauge-related tests, import everything from the gauge module:
```zig
const gauge = @import("gauge");
const GT = gauge.GaugeTree(Frontend);
const FieldArena = GT.FieldArena;  // From GaugeTree, not amr directly
```

## Physics: HMC Force (physics/force_amr.zig)

Force computation for Hybrid Monte Carlo using GaugeTree. Features differentiable link prolongation and chain rule transmission across refinement boundaries.

---

## Usage Patterns

### Basic AMR Setup

```zig
const amr = @import("amr");

const MyFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = f64;
    pub const Topology = amr.PeriodicTopology(4, .{ 16.0, 16.0, 16.0, 16.0 });
};

const Tree = amr.AMRTree(MyFrontend);
const Arena = amr.FieldArena(MyFrontend);

// Initialize
var tree = try Tree.init(allocator, 1.0, 4, 8);
defer tree.deinit();

var field_arena = try Arena.init(allocator, 256);
defer field_arena.deinit();

// Insert blocks
_ = try tree.insertBlockWithField(.{0, 0, 0, 0}, 0, &field_arena);
```

### Ghost Filling

```zig
const ghost = @import("amr").ghost;
const Ghosts = amr.GhostBuffer(MyFrontend);

// Allocate ghost storage
var ghosts = try Ghosts.init(allocator, 256);
defer ghosts.deinit();

// Fill ghosts for field data
try ghosts.ensureForTree(&tree);
ghost.fillGhostLayers(Tree, &tree, &field_arena, ghosts.slice(tree.blocks.items.len));
```

### Mesh Adaptation

```zig
const adaptation = @import("amr").adaptation;

// Adapt mesh based on field gradients
const result = try adaptation.adaptMesh(Tree, &tree, &field_arena, 0.1, 0.5);
// threshold=0.1, hysteresis=0.5

// Reorder after adaptation for cache locality
if (result.changed) {
    const perm = try tree.reorder();
    defer tree.allocator.free(perm);
    try field_arena.defragmentWithOrder(tree.field_slots.items, tree.blockCount());
}
```

## Best Practices

### Memory Management

1. **Use defer for cleanup**: Always pair `init` with `defer deinit`
2. **Pre-allocate arenas**: Size FieldArena for expected maximum blocks
3. **Reuse GhostBuffer**: Allocate once, reuse across iterations

### Performance

1. **Block size selection**: Larger blocks (16^4) for SIMD efficiency
2. **Morton reordering**: Call `reorder()` after mesh adaptation to restore cache locality
3. **SIMD alignment**: FieldArena uses 64-byte alignment for vectorization
4. **Defragment fields**: Use `defragmentWithOrder()` to achieve linear memory layout

## Module Dependencies

```
math (standalone)
platform (standalone) - Runtime + IO helpers
  ↓
amr (imports math, platform) - Domain-agnostic infrastructure
  ├── frontend (interface + Scalar/Complex helpers)
  ├── morton (space-filling curves)
  ├── field_arena (generic field storage)
  ├── ghost_buffer (optional ghost face storage)
  ├── tree (Frontend-parameterized)
  ├── block (gauge-agnostic)
  ├── ghost (namespace functions)
  ├── adaptation (namespace functions)
  └── operators (generic field prolongation/restriction)

gauge (imports math, amr, constants, platform) - Gauge-specific extensions
  ├── frontend (GaugeFrontend)
  ├── operators (LinkOperators for gauge links)
  └── tree (GaugeTree - wraps AMR tree with link storage)

physics (imports amr, gauge, math, stats, constants) - Physics modules
  ├── hamiltonian_amr (scalar Hamiltonian using GaugeTree)
  ├── hamiltonian_dirac_amr (Dirac Hamiltonian using GaugeTree)
  └── force_amr (HMC force calculation using GaugeTree)
```

**Key Design: AMR is Domain-Agnostic**
The AMR module does NOT import gauge or physics. All gauge-specific code (LinkOperators, GaugeFrontend) lives in the gauge module. This allows AMR to be reused for non-physics applications.

**Critical Architecture: No Universal Lattice Abstraction**

There is deliberately NO universal "Lattice" layer between AMR and physics. Each domain builds its own lattice-like layer directly on AMR:
- **Gauge theory**: The gauge module (`/src/gauge/`) IS the gauge-specific lattice layer
- **Other domains** (Black-Scholes, fluids, etc.): Would build their own layer on AMR

Different domains have fundamentally different lattice requirements (edge data vs. node data, staggered grids, etc.), making a universal abstraction impractical.

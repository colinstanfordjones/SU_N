# Distributed MPI Sharding (Spec)

This document describes the MPI-based sharding flow for AMR workloads. It focuses on partitioning blocks across ranks, overlapping communication with computation, and keeping the `AMRTree.apply` pipeline intact.

## Goals

- Shard AMR blocks across MPI ranks while preserving existing AMR data structures on each rank.
- Overlap interior compute with ghost exchange (both field ghosts and optional link ghosts).
- Keep the kernel interface unchanged: kernels still implement `executeInterior`/`executeBoundary` and optional ghost hooks.
- Provide deterministic, testable ghost exchange with minimal allocation churn.
- Support explicit, deterministic repartitioning for dynamic load balancing.

## Non-Goals (Initial Scope)

- GPU-aware MPI or RDMA-specific optimizations.
- Cross-rank mesh refinement (initially, refinement happens per rank and is coordinated by a higher-level step).

## Architecture Overview

Each rank owns a subset of AMR blocks and runs a local `AMRTree`. Cross-rank neighbor relationships are tracked via a compact Morton block key.

**Block identity:**

```zig
const BlockKey = struct {
    level: u8,
    morton: u64,
};
```

`BlockKey` is globally unique and stable across ranks; it is used to map neighbor faces to remote owners.
Keys are derived from a block origin/level via `tree.blockKeyFromOrigin` (or `tree.blockKeyForBlock`).

**Ownership map:**

```zig
const BlockOwner = struct {
    key: BlockKey,
    rank: i32,
};
```

The ownership map is created by a partitioning step and used to resolve remote neighbors.

## Partitioning Strategy

Initial partitioning uses Morton (Z-order) ordering of block keys, then divides the sequence into contiguous slices by rank.

```zig
const ShardStrategy = enum {
    morton_contiguous,
    manual,
};
```

Notes:
- `morton_contiguous` is deterministic and keeps spatial locality (ordered by morton, then level).
- `manual` preserves the current per-rank block ownership as reported by MPI allgather.

See `docs/specs/amr/linear_octree.md` for the block-list migration plan and data layout notes.

## Dynamic Load Balancing (Repartitioning)

Repartitioning is an explicit step that migrates blocks (and field data) across ranks using
entropy-weighted Morton ordering. It is intended for coarse-grained load balancing between
kernel steps, not inside hot loops.

Key points:
- Weight per block uses Shannon entropy of field amplitudes (global normalization).
- Blocks are ordered by `(morton, level)` and sliced contiguously across ranks (p4est-style).
- The migration payload is fixed-size: block header + field data + optional extra payload.
- Local storage is updated in-place; invalid blocks are compacted via `reorder()`.
- Use `max_inflight_bytes` or `max_inflight_messages` in `RepartitionOptions` to cap
  migration buffer memory during large transfers.

Field-only usage:

```zig
const opts = amr.repartition.RepartitionOptions{ .compact = true, .defragment = false };
try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, opts);
ghosts.trimForTree(&tree); // optional cleanup
```

Adaptive usage:

```zig
const opts = amr.repartition.RepartitionOptions{ .compact = true };
const adaptive = amr.repartition.AdaptiveOptions{
    .weight_imbalance_threshold = 0.1,
    .block_imbalance_threshold = 0.1,
};
const did = try amr.repartition.repartitionAdaptiveEntropyWeighted(
    Tree,
    &tree,
    &arena,
    &shard,
    opts,
    adaptive,
);
```

Gauge-tree usage:

```zig
const opts = amr.repartition.RepartitionOptions{ .defragment = true };
try gauge.repartition.repartitionEntropyWeighted(GaugeTree, &gauge_tree, &arena, &shard, opts);
```

Detailed algorithm notes: `docs/specs/amr/repartition.md`.

### Shard Context (Implemented)

The current implementation gathers block keys from all ranks, then builds an ownership map using the selected strategy.

```zig
const Tree = amr.AMRTree(Frontend);
const Shard = amr.ShardContext(Tree);

const comm = platform.mpi.commWorld();
var shard = try Shard.initFromTree(allocator, &tree, comm, .morton_contiguous);
defer shard.deinit();

tree.attachShard(&shard);

const local_blocks = shard.localBlockIndices();
```

### Morton Key Generation

Block keys are built from Morton indices derived from block origins and refinement levels.
The tree owns the conversion:

```zig
const key_from_origin = tree.blockKeyFromOrigin(origin, level);
const key_from_block = tree.blockKeyForBlock(&tree.blocks.items[block_idx]);
```

Reordering changes block indices but does not change Morton keys, so ownership remains stable.

## MPI Layer

A platform-level wrapper keeps MPI usage contained and testable.

Module:
- `src/platform/mpi/root.zig` (MPICH-backed wrapper + disabled stub)

Minimal API (error-aware):

```zig
pub const Comm = struct { handle: c.MPI_Comm };

pub fn initThreaded(required: ThreadLevel) !ThreadLevel;
pub fn initSerialized() !ThreadLevel;
pub fn finalize() void;
pub fn commWorld() Comm;
pub fn rank(comm: Comm) !i32;
pub fn size(comm: Comm) !i32;
pub fn barrier(comm: Comm) !void;
pub fn bcastBytes(comm: Comm, buf: []u8, root: i32) !void;

pub fn isend(comm: Comm, buf: []const u8, dest: i32, tag: i32) !Request;
pub fn irecv(comm: Comm, buf: []u8, src: i32, tag: i32) !Request;
pub fn irecvAny(comm: Comm, buf: []u8, tag: i32) !Request;
pub fn waitAll(reqs: []Request) !void;

pub fn allreduceSum(comm: Comm, value: f64) !f64;
pub fn allgatherI32(comm: Comm, value: i32, recv: []i32) !void;
pub fn allgatherBytes(comm: Comm, send: []const u8, recv: []u8) !void;
pub fn allgatherVBytes(comm: Comm, send: []const u8, recv: []u8, recv_counts: []const i32, recv_displs: []const i32) !void;
```

Threading level: use `MPI_THREAD_SERIALIZED` (MPI calls are externally serialized). This aligns with the current threaded AMR pipeline and avoids full `MPI_THREAD_MULTIPLE`.

## Distributed Ghost Exchange

Field ghosts are exchanged across ranks using the same pull/push semantics as the local ghost exchange.
The MPI path is implemented in `src/amr/dist_exchange.zig` and is invoked by `AMRTree.apply` when a shard context is attached via `tree.attachShard`.
Custom exchange specs can be supplied at tree initialization via `Tree.initWithOptions` to override payload sizing or pack/unpack hooks.

### Exchange Phases

1. **Post receives** for all remote neighbor faces.
2. **Pack and send** local boundary faces to remote neighbors.
3. **Local ghost pull** for same-rank neighbors (reuse existing ghost helpers).
4. **Interior compute** (overlaps with MPI in-flight).
5. **Wait for receives** and unpack into ghost buffers.
6. **Local push** (fine -> coarse) plus remote push receive handling.
7. **Boundary compute**.

### Refinement Boundaries and Neighbor Resolution

- Coarse faces adjacent to refined regions are treated as "no neighbor" in the push model.
- Fine blocks pull from same-level or coarse neighbors, then push restricted faces to the coarse side.
- MPI exchange mirrors the local rules by resolving neighbors through the ownership map, not cached neighbor arrays.
- Fine neighbors across a coarse face are enumerated deterministically from Morton keys and the topology, then exchanged as coarse-to-fine messages.
- Fine-to-coarse messages are accumulated into the coarse ghost face (restriction).

Topology wrapping is driven by the frontend's `Topology` type; domain sizes and boundaries must match across ranks.

If you need strict bitwise determinism across MPI stacks, sort received messages by `BlockKey` before applying them.

### API (Implemented)

`AMRTree.apply` uses the attached `ShardContext` and runs the begin/finish calls internally to coordinate remote ghosts.

For manual control (testing or custom scheduling), use `DistExchange` directly:

```zig
const FieldPolicy = amr.ghost_policy.FieldGhostPolicy(Tree);
const Exchange = amr.dist_exchange.DistExchange(Tree, FieldPolicy.Context, FieldPolicy.Payload);
var exchange = Exchange.init(allocator, FieldPolicy.exchangeSpec());
defer exchange.deinit();

const ctx = FieldPolicy.Context{
    .tree = &tree,
    .arena = &arena,
    .ghosts = ghosts.slice(tree.blocks.items.len),
};

const state = try exchange.begin(ctx, &shard);
try exchange.finish(ctx, &state);
```

### Reordering and Shards

Morton reordering keeps ownership keyed to `(morton, level)` but invalidates cached local indices. After `tree.reorder()` call:

```zig
const perm = try tree.reorder();
defer tree.allocator.free(perm);
try shard.refreshLocal(&tree);
```

## Reductions and Global Metrics

Physics routines that compute global scalars (energy, norms) should use `MPI_Allreduce` to aggregate results across ranks.

Example:

```zig
const local_energy = h.measureEnergy(&psi, &workspace, &ghosts);
const total = platform.mpi.allreduceSum(comm, local_energy);
```

## Checkpointing

MPI checkpoints are per-rank snapshots; each rank writes its local tree/arena state. Use a broadcasted step to keep snapshots aligned:

```zig
const checkpoint = platform.checkpoint;
const step = try checkpoint.broadcastCheckpointStep(comm, 0, local_step);
if (schedule.shouldCheckpoint(step)) {
    const path = try checkpoint.formatRankedPath(&buf, "checkpoints", "run", step, rank);
    try Ckpt.writeToFile(&tree, &arena, std.fs.cwd(), path);
}
```

On restart, load per-rank checkpoints and rebuild `ShardContext` from the restored trees.

## Data Packing

Start with explicit packing into contiguous byte buffers. Avoid custom MPI datatypes initially.

```zig
const bytes = std.mem.asBytes(face_slice);
request = mpi.isend(comm, bytes, dest, tag);
```

## Tests

- `tests/platform/mpi_tests.zig`
  - `initSerialized` + world rank/size
  - `allreduceSum` correctness
  - ping-pong send/recv (skips when `mpi.size() < 2` or MPI is disabled)
- `tests/amr/mpi_shard_test.zig`
  - `ShardContext` ownership map and local block selection (skips when `mpi.size() < 2` or MPI is disabled)
- `tests/amr/mpi_repartition_test.zig`
  - Entropy-weighted repartition migrates field data, validates adaptive gating, and rebalances ownership.
- `tests/gauge/mpi_repartition_test.zig`
  - Gauge repartition migrates link payloads across ranks.
- `tests/amr/mpi_ghost_exchange_test.zig`
  - MPI ghost exchange across open boundaries (requires `mpi.size() >= 2`)
- `tests/amr/mpi_apply_test.zig`
  - Distributed apply matches a single-rank reference across coarse/fine boundaries.
- `tests/physics/mpi_hamiltonian_test.zig`
  - Deterministic Hamiltonian energy with MPI allreduce.

Run with `zig build test-mpi -Denable-mpi=true` to execute the MPI test suite under `mpirun -n 2`.

## Build Integration

- MPICH is tracked as a submodule under `external/mpich` for reference/versioning.
- Build flag: `-Denable-mpi=true`
- Optional paths: `-Dmpi-include=<path>`, `-Dmpi-lib=<path>` (useful for system MPICH installs).
- Build parallelism: `-Dmpi-jobs=<n>` (defaults to CPU count).
- When enabled, the build configures and builds MPICH into `zig-out/mpich`, then links `platform/mpi` against it.
- A stub `platform/mpi` is provided when MPI is disabled so code compiles.

The build step runs `git submodule update --init --recursive` inside `external/mpich` to ensure MPICH's own submodules (e.g., hwloc) are available.

## Open Questions

- Do we require cross-rank mesh refinement in v1, or enforce refinement boundaries to remain within a rank?
- Should ownership be based on Morton order or a static block coordinate decomposition?
- Should we explicitly gate to MPICH, or allow OpenMPI if API-compatible with the wrapper?

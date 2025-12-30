# AMR Checkpoint/Restart (Spec)

This document specifies the exact-state checkpoint format and API used to restart AMR runs without loss of fidelity.

## Goals

- Capture AMRTree structure, FieldArena storage, and (optionally) gauge links bit-for-bit.
- Support deterministic restarts for long-running HMC and AMR pipelines.
- Provide MPI-friendly checkpointing (per-rank snapshots + broadcasted schedule).
- Keep checkpoint logic in `platform/` and out of domain modules.

## Non-Goals

- Cross-architecture portability (payload is host-endian raw bytes).
- Visualization/export formatting (no Blender-specific layout).
- Incremental or compressed checkpoints (future work).

## What Is Serialized

Checkpoint snapshots include:

- `AMRTree` block list (raw blocks, including invalid entries)
- `AMRTree.field_slots` (block index -> arena slot map)
- `FieldArena.storage` (full storage for all slots)
- `FieldArena.free_slots` + `free_count`
- Optional gauge link storage (`GaugeTree.links`)

Checkpoint snapshots do **not** include:

- Work-stealing pool state (recreated on load)
- Ghost buffers or ghost validity flags
- MPI shard context (`ShardContext` is rebuilt from tree + MPI size)

## On-Disk Format (`.sunc`)

Header fields are written in little-endian; payload sections are raw bytes.

```
Header:
  magic: "SUNC"
  version: u32
  endian: u8 (little)
  nd: u8
  block_size: u32
  block_volume: u64
  block_bytes: u32
  field_bytes: u32
  usize_bytes: u32
  blocks_len: u64
  field_slots_len: u64
  arena_max_blocks: u64
  arena_free_count: u64
  base_spacing_bits: u64
  bits_per_dim: u8
  max_level: u8
  reserved: [16]u8

Payload (in order):
  blocks[blocks_len]               (raw AMRBlock bytes)
  field_slots[field_slots_len]    (raw usize bytes)
  arena.storage[arena_max_blocks] (raw FieldType bytes)
  arena.free_slots[arena_max_blocks]
```

**Extension Data:**
Consumers (like `GaugeTree`) may append data after the standard payload.
For example, `GaugeTree` appends:
- Magic: "LINK"
- Num Blocks: u64
- Links per Block: u64
- Link Data: `[num_blocks][links_per_block]LinkType`

Notes:
- Payload is host-endian and depends on `FieldType`, `LinkType`, and `usize` sizes.
- `block_index` map is rebuilt from block origins/levels at load time.

## API

Module: `src/platform/checkpoint/root.zig`

```zig
const su_n = @import("su_n");
const checkpoint = su_n.platform.checkpoint;

const Frontend = amr.ScalarFrontend(2, 16, amr.topology.OpenTopology(2, .{ 16.0, 16.0 }));
const Tree = amr.AMRTree(Frontend);
const Arena = amr.FieldArena(Frontend);
const Ckpt = checkpoint.TreeCheckpoint(Tree);

try Ckpt.write(&tree, &arena, writer);
var state = try Ckpt.read(allocator, reader);
defer state.deinit();
```

Gauge tree snapshots use the `GaugeTree` methods which wrap the core checkpointing:

```zig
const Frontend = gauge.GaugeFrontend(1, 1, 4, 16, amr.topology.OpenTopology(4, .{ 16.0, 16.0, 16.0, 16.0 }));
const GaugeTree = gauge.GaugeTree(Frontend);
const Arena = amr.FieldArena(Frontend);

// Write
try gauge_tree.writeCheckpoint(&arena, writer);

// Read (default exchange spec)
var state = try GaugeTree.readCheckpoint(allocator, reader);
var restored_tree = state.tree;
var restored_arena = state.arena;
defer restored_tree.deinit();
defer restored_arena.deinit();

// Read with custom link exchange spec
var state_custom = try GaugeTree.readCheckpointWithOptions(
    allocator,
    reader,
    .{ .link_exchange_spec = custom_spec },
);
```

## Scheduling

`platform.checkpoint.Schedule` provides a lightweight interval-based trigger:

```zig
const schedule = checkpoint.Schedule{ .interval = 100, .start = 0 };
if (schedule.shouldCheckpoint(step)) {
    try Ckpt.writeToFile(&tree, &arena, dir, "run_step100.sunc");
}
```

## MPI Usage

Checkpointing is per-rank:

```zig
const comm = platform.mpi.commWorld();
const rank = try platform.mpi.rank(comm);

var path_buf: [256]u8 = undefined;
const path = try checkpoint.formatRankedPath(&path_buf, "checkpoints", "run", step, rank);
try Ckpt.writeToFile(&tree, &arena, std.fs.cwd(), path);
```

To coordinate snapshot steps across ranks:

```zig
const step = try checkpoint.broadcastCheckpointStep(comm, 0, local_step);
if (schedule.shouldCheckpoint(step)) { /* write */ }
```

After a restart, rebuild sharding:

```zig
var shard = try amr.ShardContext(Tree).initFromTree(allocator, &tree, comm, .morton_contiguous);
defer shard.deinit();
tree.attachShard(&shard);
```

## Frontends and Kernels

Checkpoint boundaries should be aligned with kernel steps. Store checkpoints after `AMRTree.apply` completes and ghosts are consistent. Kernels remain unchanged: checkpoint logic lives outside the kernel and uses `tree` + `arena` handles.

## FAISS Note

FAISS is designed for vector similarity search (ANN) and is not appropriate for exact state serialization. It may still be used later for ML indexing, but not as a checkpoint backend.

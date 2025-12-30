# AMR Repartitioning (Spec)

This document specifies dynamic repartitioning for AMR MPI shards. The goal is
explicit, coarse-grained load balancing between kernel steps using entropy-weighted
Morton ordering, consistent with p4est-style contiguous SFC partitioning.

## Goals

- Rebalance blocks across MPI ranks without changing AMR block semantics.
- Preserve locality by keeping each rank's blocks contiguous in Morton order.
- Move field data (and gauge links when present) with a fixed-size payload.
- Avoid allocations in hot paths; use scratch arenas and adaptive capacity growth.

## Non-Goals

- Per-timestep work stealing or fine-grained balancing.
- GPU-aware MPI.
- Cross-rank refinement (refine/coarsen remains rank-local).

## Weight Definition (Entropy)

Weights are derived from the field's Shannon entropy with global normalization:

- Let w_i = |psi_i|^2 for each field element i.
- Let W = sum_i w_i across all ranks (MPI allreduce).
- For each block, weight = -sum_{i in block} p_i * log(p_i), where p_i = w_i / W.
- If W <= 0, fall back to uniform weight (1.0 per block).
- A minimum weight floor is applied (`min_weight`) to avoid zero-weight blocks.

This yields total weight equal to the global entropy and keeps weights proportional
to the amount of field structure in each block.

Note: Non-scalar field types should implement `Frontend.fieldNormSq` so entropy
weights can be computed consistently.

## Partition Algorithm

1. Gather `(BlockKey, weight)` from all ranks.
2. Sort by `(morton, level)`.
3. Slice the ordered list into contiguous segments using weighted targets
   (total weight / ranks), ensuring at least one block per rank when possible.
4. Build a new ownership map and migrate blocks accordingly.

This matches p4est best practice: contiguous space-filling-curve partitions with
weighted load targets.

## Migration Payload

Each migrated block uses a fixed-size message:

- Header: origin[Nd], level, has_field flag.
- Field data: block_volume * sizeof(FieldType).
- Optional extra payload: gauge links or other per-block storage.

The fixed-size layout keeps MPI send/recv simple (no custom datatypes).

## Allocation Model

- Repartitioning is an explicit call, not part of the hot compute loop.
- Temporary buffers use `ShardContext`'s scratch arena.
- Field storage grows via `FieldArena.ensureCapacity` when needed.
- After migration, `reorder()` compacts invalid blocks.
- Optional `defragmentWithOrder()` can linearize field slots after reorder.
- Ghost storage can be trimmed with `GhostBuffer.trimForTree()`.
- Migration buffers can be batched via `max_inflight_bytes` or `max_inflight_messages`
  to cap peak send/recv memory.
- If the in-flight limit is smaller than a single message (`stride`), repartition
  returns `error.InflightLimitTooSmall`.
- Weight gathering and owner assignment allocate O(total_blocks) scratch memory
  (counts, weights, owner entries); these are not currently batched.

GaugeTree notes:
- The gauge wrapper forces compaction to release link storage.
- `GaugeTree.reorder()` keeps link and ghost arrays consistent.

## API

`RepartitionOptions`:
- `min_weight`: floor for per-block entropy weights.
- `compact`: reorder/compact after migration.
- `defragment`: defragment field arena after reorder.
- `max_inflight_bytes`: cap total bytes of in-flight send/recv buffers per batch (0 = unbounded).
- `max_inflight_messages`: cap number of in-flight send/recv messages per batch (0 = unbounded).

`AdaptiveOptions`:
- `weight_imbalance_threshold`: trigger when max rank weight exceeds avg by this fraction.
- `block_imbalance_threshold`: trigger when max block count exceeds avg by this fraction.

Field-only usage:

```zig
const opts = amr.repartition.RepartitionOptions{ .compact = true, .defragment = false };
try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, opts);
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

GaugeTree usage:

```zig
const opts = amr.repartition.RepartitionOptions{ .defragment = true };
try gauge.repartition.repartitionEntropyWeighted(GaugeTree, &gauge_tree, &arena, &shard, opts);
```

## Tests

- `tests/amr/mpi_repartition_test.zig`
  - Validates block migration, adaptive gating, empty-field handling, and 2/3-rank balance.
- `src/amr/repartition.zig`
  - Unit tests for weighted contiguous owner assignment.
- `tests/gauge/mpi_repartition_test.zig`
  - Validates gauge link payload migration across ranks.

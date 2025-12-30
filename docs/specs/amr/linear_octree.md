# Linear Octree / Block List Plan

This document captures the migration plan for the linear octree (Morton-ordered block list) and
how it impacts neighbor queries, MPI sharding, and tests.

## Goals

- Replace pointer-heavy tree nodes with a flat block list indexed by Morton keys.
- Avoid cached neighbor arrays; resolve neighbors on demand from Morton keys and topology.
- Keep MPI ownership stable across reorder/compaction using `BlockKey` (level + morton).
- Preserve frontend-driven behavior (Topology, block_size, FieldType) and kernel APIs.

## Current State

- Blocks are stored in a flat `ArrayList` and indexed by `BlockKey` via a hash map.
- `BlockKey = { level: u8, morton: u64 }` is derived from origin + level.
- `reorder()` compacts blocks, sorts by Morton, rebuilds the key map, and returns a permutation map.
- Neighbor discovery uses `tree.neighborInfo` (same/coarse only) and `tree.collectFineNeighbors`.
- MPI sharding uses Morton keys for ownership and remote ghost exchange.

## Data Layout Plan

1. **Block list as metadata arrays**
   - Split `AMRBlock` into SoA-style arrays for hot metadata:
     - `origin[Nd]`, `level`, `spacing`, `block_index`, `field_slot`, `morton`.
   - Keep a lightweight `BlockView` for APIs that need a block-shaped interface.

2. **Stable handles**
   - Introduce a `BlockHandle` (index + generation) to survive compaction.
   - Update `reorder()` to return a remap table for external arrays.

3. **Key cache**
   - Store per-block `BlockKey` (level + morton) to avoid recomputing Morton codes.

## Neighbor Query Plan

1. **Runtime queries**
   - Continue to use `neighborInfo` for same/coarse neighbors and `collectFineNeighbors` for fine.
   - Refined faces return `none` (push model) to keep coarse ghost handling deterministic.

2. **Optional caches**
   - Consider a small face cache for same-level neighbors when kernels call repeatedly.
   - Caches must be invalidated on refine/coarsen/reorder.

3. **Topology correctness**
   - All neighbor queries must honor `Frontend.Topology.wrapCoordinateRuntime`.
   - Open boundaries return `none` so ghost faces are zeroed on pull.

## MPI Implications

- Ownership is keyed by `BlockKey`, not by block indices.
- Morton ordering provides deterministic partitioning (`morton_contiguous`).
- Refinement boundaries are handled by the push model:
  - Fine -> coarse restriction accumulates into coarse ghost faces.
  - Coarse -> fine prolongation sends boundary faces to fine neighbors.
- Reorder/compaction and repartition require `shard.refreshLocal(&tree)`.

## Test Plan

- **Key invariants**: `blockKeyFromOrigin`/`findBlockByKey` round-trip.
- **Neighbor queries**: same-level, coarse, and fine neighbors across periodic/open boundaries.
- **Reorder**: permutation map correctness and field_slots stability.
- **MPI**:
  - Shard ownership determinism under `morton_contiguous`.
  - Ghost exchange across refinement boundaries (fine/coarse + open boundaries).
  - Consistent results between MPI and single-rank reference kernels.

## Frontend + Kernel Usage Impact

- Frontends still define `Nd`, `block_size`, `FieldType`, and `Topology`.
- Kernels should consume ghost buffers, not neighbor arrays.
- If a kernel needs adjacency, use `tree.neighborInfo` / `tree.collectFineNeighbors`.
- `AMRTree.apply` continues to own the threaded pipeline and distributed ghost exchange.

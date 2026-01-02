# AMR Audit and Remediation Report
**Date:** 2025-12-30
**Status:** DRAFT

## 1. Executive Summary

The project has made significant progress with the implementation of Multigrid, Flux Registers, and dynamic load balancing. The prior architecture suffered from **leaky abstractions**, **tight coupling**, and **duplicated logic**, particularly in how ghost layers and MPI communication were handled. The legacy separation between `AMRTree` (fields) and a gauge-link wrapper created a "dual ghost management" problem that burdened physics kernels. This refactor is now in progress and largely resolved (see Status Update).

## Status Update (Current Session)

- Legacy GaugeTree wrapper removed; link storage now lives in `GaugeField` with `EdgeArena`/`EdgeGhostBuffer` under `AMRTree`.
- Physics, benchmarks, and tests migrated to `AMRTree` + `GaugeField` APIs; repartition helpers now accept tree + field.
- Compatibility shims (`Link*` aliases, GaugeTree-only helpers) removed from AMR/gauge core.
- Specs updated to reflect `ApplyContext` + `GaugeField`; MPI tests and benchmarks pass in this session.
- Link exchange now includes normal directions via `LinkGhostPolicy` payload packing.

## 2. Findings

### 2.1. Architectural Flaws

*   **Dual Ghost Management (Legacy):** `AMRTree` managed field ghosts via `GhostBuffer`, while a gauge wrapper managed link ghosts. Physics kernels (e.g., `HamiltonianAMR`) had to coordinate both. **Status:** resolved by `GaugeField` + `ApplyContext` with explicit link ghost exchange.
*   **Leaky Abstractions:** `DistExchange` exists to encapsulate MPI, yet `AMRTree.apply` and `HamiltonianAMR` are still responsible for managing the lifecycle of exchanges (`begin`, `end`, `fill`). The user/frontend is exposed to these details.
*   **Tight Coupling:** `HamiltonianAMR` was tightly coupled to the legacy gauge wrapper implementation details. **Status:** resolved by decoupling link storage into `GaugeField`.

### 2.2. Logic Duplication

*   **Exchange Logic:** `src/amr/ghost.zig` implements a "Push Model" for local ghost filling. `src/amr/dist_exchange.zig` implements a "Pull Model" (MPI-compatible) for distributed exchange. These modules duplicate packing/unpacking math and logic.
*   **Math Duplication:** Restriction and prolongation math is repeated across `ghost.zig`, `ghost_policy.zig`, and `multigrid.zig`.

### 2.3. Performance Bottlenecks

*   **Allocations in Hot Path:** `AMRTree.apply` creates `TaskContext` objects dynamically for every block. While small, this adds pressure.
*   **FluxRegister Overhead:** `FluxRegister` uses `std.AutoHashMap` to store flux corrections. This involves hashing in a hot loop. Since boundary faces are structured, a linear or block-indexed storage would be significantly faster (O(1) vs O(1) + Hash).
*   **Thread-Local Storage:** Heavy use of `threadlocal` for scratch arenas in `ghost.zig` might have overhead and complicates thread pooling if not managed carefully.

### 2.4. Implementation Gaps

*   **Normal Ghost Links:** The legacy gauge wrapper only exchanged *tangential* links (for plaquettes). It did NOT exchange *normal* links required for covariant derivatives at boundaries. **Status:** resolved by exchanging all directions in `LinkGhostPolicy`.
*   **Boundary Conditions:** Boundary handling in `HamiltonianAMR` is manual and ad-hoc. A unified `BoundaryCondition` interface is missing.

## 3. Remediation Plan

### Phase 1: Unify Ghost Management (High Priority)
1.  **Create `ApplyContext`:** Bundle `FieldArena`, `GaugeField` link storage, and ghost buffers (fields + links). **Status:** complete.
2.  **Unified Exchange:** Refactor `DistExchange` to handle BOTH local and remote exchanges, superseding `ghost.zig`. The "Push Model" can be an optimization within `DistExchange` for local blocks. **Status:** in progress.
3.  **Automate Lifecycle:** Keep ghost exchange inside `AMRTree.apply` for fields; link ghost exchange remains explicit via `GaugeField.fillGhosts`. **Status:** field path done; link path explicit by design.

### Phase 2: Optimize FluxRegister (Medium Priority)
1.  **Replace HashMap:** Reimplement `FluxRegister` to use a dense array (indexed by `block_idx`) of sparse face data, or a flat array if memory permits.
2.  **Thread Safety:** Ensure `FluxRegister` is thread-safe without heavy locking (e.g., per-thread accumulation or atomic adds).

### Phase 3: Fix Gauge/Link Exchange (High Priority)
1.  **Exchange Normal Links:** Update the gauge link exchange policy to include normal links required for covariant derivatives. **Status:** complete via `LinkGhostPolicy` and `GaugeField`.

### Phase 4: Clean Up Duplication
1.  **Math Module:** Extract restrict/prolongate math into `src/math/interpolation.zig` (or similar).
2.  **Delete Legacy:** Remove `src/amr/ghost.zig` once `dist_exchange.zig` covers all cases.

## 4. Conclusion

The current state is functional for single-node (mostly) but brittle for MPI and maintenance. The proposed refactoring will centralize complexity into `DistExchange` and `Context` structures, liberating the physics kernels to focus on math.

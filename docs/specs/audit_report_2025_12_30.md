# AMR Audit and Remediation Report
**Date:** 2025-12-30
**Status:** DRAFT

## 1. Executive Summary

The project has made significant progress with the implementation of Multigrid, Flux Registers, and dynamic load balancing. However, the current architecture suffers from **leaky abstractions**, **tight coupling**, and **duplicated logic**, particularly in how ghost layers and MPI communication are handled. The separation between `AMRTree` (fields) and `GaugeTree` (links) has created a "dual ghost management" problem that burdens the physics kernels. A refactoring is required to unify these systems and encapsulate MPI details fully.

## 2. Findings

### 2.1. Architectural Flaws

*   **Dual Ghost Management:** `AMRTree` manages field ghosts via `GhostBuffer`, while `GaugeTree` manages link ghosts via `GhostStorage`. Physics kernels (e.g., `HamiltonianAMR`) must manually coordinate both: calling `ghostPrepare` for links and passing `ghosts` buffer for fields. This violates the principle of "Kernel Pattern" where the kernel shouldn't manage infrastructure.
*   **Leaky Abstractions:** `DistExchange` exists to encapsulate MPI, yet `AMRTree.apply` and `HamiltonianAMR` are still responsible for managing the lifecycle of exchanges (`begin`, `end`, `fill`). The user/frontend is exposed to these details.
*   **Tight Coupling:** `HamiltonianAMR` is tightly coupled to `GaugeTree` implementation details. It manually iterates blocks and accesses internal structures to perform flux corrections.

### 2.2. Logic Duplication

*   **Exchange Logic:** `src/amr/ghost.zig` implements a "Push Model" for local ghost filling. `src/amr/dist_exchange.zig` implements a "Pull Model" (MPI-compatible) for distributed exchange. These modules duplicate packing/unpacking math and logic.
*   **Math Duplication:** Restriction and prolongation math is repeated across `ghost.zig`, `ghost_policy.zig`, and `multigrid.zig`.

### 2.3. Performance Bottlenecks

*   **Allocations in Hot Path:** `AMRTree.apply` creates `TaskContext` objects dynamically for every block. While small, this adds pressure.
*   **FluxRegister Overhead:** `FluxRegister` uses `std.AutoHashMap` to store flux corrections. This involves hashing in a hot loop. Since boundary faces are structured, a linear or block-indexed storage would be significantly faster (O(1) vs O(1) + Hash).
*   **Thread-Local Storage:** Heavy use of `threadlocal` for scratch arenas in `ghost.zig` might have overhead and complicates thread pooling if not managed carefully.

### 2.4. Implementation Gaps

*   **Normal Ghost Links:** `GaugeTree` only exchanges *tangential* links (for plaquettes). It does NOT exchange *normal* links required for covariant derivatives ($D_\mu \psi$) at boundaries. This breaks `HamiltonianAMR` at MPI boundaries.
*   **Boundary Conditions:** Boundary handling in `HamiltonianAMR` is manual and ad-hoc. A unified `BoundaryCondition` interface is missing.

## 3. Remediation Plan

### Phase 1: Unify Ghost Management (High Priority)
1.  **Create `PhysicsContext`:** Introduce a struct that bundles `FieldArena`, `LinkStorage`, and `GhostBuffer` (both fields and links).
2.  **Unified Exchange:** Refactor `DistExchange` to handle BOTH local and remote exchanges, superseding `ghost.zig`. The "Push Model" can be an optimization within `DistExchange` for local blocks.
3.  **Automate Lifecycle:** Move `ghostPrepare`/`Finalize` into the `Solver` or `Tree` apply method. The kernel should just receive valid data.

### Phase 2: Optimize FluxRegister (Medium Priority)
1.  **Replace HashMap:** Reimplement `FluxRegister` to use a dense array (indexed by `block_idx`) of sparse face data, or a flat array if memory permits.
2.  **Thread Safety:** Ensure `FluxRegister` is thread-safe without heavy locking (e.g., per-thread accumulation or atomic adds).

### Phase 3: Fix Gauge/Link Exchange (High Priority)
1.  **Exchange Normal Links:** Update `GaugeTree` and `GhostPolicy` to exchange normal links required for covariant derivatives, or implement a mechanism to fetch them from neighbors (if local).

### Phase 4: Clean Up Duplication
1.  **Math Module:** Extract restrict/prolongate math into `src/math/interpolation.zig` (or similar).
2.  **Delete Legacy:** Remove `src/amr/ghost.zig` once `dist_exchange.zig` covers all cases.

## 4. Conclusion

The current state is functional for single-node (mostly) but brittle for MPI and maintenance. The proposed refactoring will centralize complexity into `DistExchange` and `Context` structures, liberating the physics kernels to focus on math.

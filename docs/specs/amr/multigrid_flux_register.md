# Multigrid and Flux Register Specification

This document covers the AMR multigrid operators (`src/amr/multigrid.zig`) and
flux register implementation (`src/amr/flux_register.zig`). It assumes the
AMR tree uses leaf-only refinement (parents invalidated) and enforces 2:1
balance across faces.

## Flux Register

### Conventions

- Fluxes are **+axis fluxes** (F_{+d}).
- Faces are indexed as `0=+d0, 1=-d0, 2=+d1, 3=-d1, ...`.
- Each register entry stores **per-face-cell** flux integrals on the **coarse**
  face. The value type is `[face_cells]FieldType` where `face_cells = block_size^(Nd-1)`.
- Kernels pass **per-face-cell flux values already scaled by face area**
  (`A_cell = h^(Nd-1)`), and `FluxRegister` multiplies by `dt`.

### Accumulation

**Coarse contribution** (on the coarse block face):

```
R_face_cell -= F_coarse * dt * A_coarse_cell
```

**Fine contribution** (mapped onto the coarse face):

```
R_face_cell += sum(F_fine * dt * A_fine_cell)
```

Mapping from fine face cell to coarse face cell uses level-0 coordinates:

```
coarse_coord = ((fine_origin[d] + fine_coord[d]) / 2) - coarse_origin[d]
```

Each coarse face cell receives contributions from `2^(Nd-1)` fine face cells.

### Reflux

Reflux applies the register values back to the coarse cells on that face.
For each face cell, update the corresponding boundary cell:

```
U_coarse += sign(face) * R_face_cell / V_cell
```

- `V_cell = h^Nd` (coarse cell volume)
- `sign(face)` is **-** for `+d` faces, **+** for `-d` faces (matches +axis fluxes)

### Requirements and Assumptions

- The tree must be **2:1 balanced** across faces (neighbor level difference <= 1).
- `block_size` is **even** so face cells map cleanly across 2:1 refinement.
- Fluxes are provided at coarse/fine boundaries only. Leaf-only refinement is
  assumed (parents invalidated).
- `reflux`/`clear` are called after flux accumulation completes (no concurrent
  register updates across threads).
- The design follows p4est best practices: 2:1 balance with coarse-face flux
  corrections applied to coarse cells.

### Performance Notes

- Each thread accumulates into a local register map keyed by `(BlockKey, face)`,
  reducing contention on hot paths. The FluxRegister keeps a thread-local cache
  to avoid locking during steady-state updates.
- `iterAll` yields per-thread entries; reduce by key when inspecting or testing
  aggregate register values.
- `addFine` uses a precomputed fine→coarse face index map to avoid per-cell
  coordinate math in hot paths.
- `clearAndReserve` or `prepare` may be used to pre-size register maps before
  accumulation to minimize allocations.
- `setNoAlloc(true)` can be enabled to force `addFine`/`addCoarse` to return
  `error.OutOfCapacity` if a register map would need to grow.

### Allocation Model

- Allocations are limited to local-register creation and capacity growth in
  `clearAndReserve`/`prepare`; steady-state accumulation should not allocate.
- For strict no-allocation runs, warm up the local register, call
  `clearAndReserve`, then enable `setNoAlloc(true)` and handle
  `error.OutOfCapacity` if capacity is insufficient.
- Per-thread reserve hints should scale with worker count (e.g., total faces
  divided by pool threads) to avoid over-allocating each local register.

## Multigrid (GMG)

### Hierarchy and Constraints

- `multigrid.zig` operates on blocks present in the AMR tree.
- Restriction/prolongation currently assume a **full hierarchy** (parents
  coexisting with children). With leaf-only AMR, build a multigrid hierarchy
  explicitly or materialize coarse blocks for GMG sweeps.
- All levels are expected to be 2:1 balanced.

### Operators

- **Restriction**: average `2^Nd` fine cells into one coarse cell.
- **Prolongation**: injection (copy coarse values into fine cells).

### V-Cycle

1. Pre-smooth on the current level.
2. Compute residual `r = f - A u`.
3. **Zero coarse RHS**, then restrict residual into `f_coarse`.
4. Solve the coarse error equation recursively.
5. Prolongate and add correction to fine level.
6. Post-smooth on the current level.

## 2:1 Balance

`adaptMesh` enforces 2:1 balance after refinement by refining any block that
would otherwise differ by more than one level across a face. This is required
for consistent flux register and multigrid behavior.

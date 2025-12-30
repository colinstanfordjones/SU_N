# Testing Specification

Test organization and current guidance for the su_n library.

## Status

Tests are being refactored alongside the AMR/gauge API cleanup. Physics integration tests
that depended on legacy lattice APIs have been removed and will be rebuilt against the
new AMR + GaugeTree interfaces.

## Current Structure

```
tests/
├── root.zig           # Test registry (imports all test files)
├── test_utils.zig     # Shared utilities (tolerances, helpers)
├── amr/               # AMR infrastructure tests
│   ├── amr_block_tests.zig
│   ├── amr_tree_tests.zig
│   ├── amr_operators_tests.zig
│   ├── amr_topology_tests.zig
│   ├── boundary_test.zig
│   ├── ghost_exchange_test.zig
│   ├── heat_equation_test.zig
│   ├── mpi_apply_test.zig
│   ├── mpi_checkpoint_test.zig
│   ├── mpi_ghost_exchange_test.zig
│   ├── mpi_repartition_test.zig
│   ├── mpi_shard_test.zig
│   ├── pipeline_stress_test.zig
│   └── poisson_test.zig
├── gauge/             # Gauge group tests
│   ├── checkpoint_test.zig
│   ├── ghost_exchange_test.zig
│   ├── mpi_link_exchange_test.zig
│   ├── mpi_repartition_test.zig
│   ├── u1_tests.zig
│   ├── su2_tests.zig
│   └── su3_tests.zig
├── math/              # Linear algebra tests
│   ├── matrix_tests.zig
│   ├── matrix_exp_tests.zig
│   └── tensor_tests.zig
├── physics/           # Physics model tests
│   ├── hamiltonian_amr_test.zig
│   ├── hamiltonian_pipeline_test.zig
│   └── mpi_hamiltonian_test.zig
├── platform/          # Runtime/OS integration tests
│   └── mpi_tests.zig
└── ml/                # Python-based integration tests
    ├── diffusion_test.py
    └── integration_test.py
```

## Running Tests

```bash
zig build test                 # Internal + integration
zig build test-internal        # Source-embedded tests only
zig build test-integration     # tests/ directory only
zig build test -- --test-filter "pattern"
```

### MPI Tests

MPI tests run under the `test-mpi` build step, which invokes `mpirun -n 2` and filters on `mpi`:

```bash
zig build test-mpi -Denable-mpi=true
```

See `docs/specs/amr/mpi.md` for the MPI sharding overview and ghost exchange details.

## Refactor Guidance

- Prefer API behavior checks over full pipeline simulations.
- Use `tests/test_utils.zig` with `constants.test_epsilon` for floating-point checks.
- New physics tests should use `gauge.GaugeTree` and `amr.GhostBuffer`, not raw AMR imports.
- Physics kernels should implement `executeInterior`/`executeBoundary` and run via `AMRTree.apply()`.
- Keep test sizes small (block_size 4 or 8) unless validating scaling.
- Add new test files to `tests/root.zig`.
- Non-AMR code paths have been removed; all physics uses AMR.

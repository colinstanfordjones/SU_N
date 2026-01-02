const std = @import("std");

test {
    _ = @import("math/matrix_tests.zig");
    _ = @import("math/matrix_exp_tests.zig");
    _ = @import("math/tensor_tests.zig");
    _ = @import("gauge/su2_tests.zig");
    _ = @import("gauge/u1_tests.zig");
    _ = @import("gauge/su3_tests.zig");
    _ = @import("gauge/field_test.zig");
    _ = @import("gauge/checkpoint_test.zig");
    _ = @import("gauge/ghost_exchange_test.zig");
    _ = @import("gauge/mpi_link_exchange_test.zig");
    _ = @import("gauge/mpi_repartition_test.zig");
    // AMR tests
    _ = @import("amr/amr_block_tests.zig");
    _ = @import("amr/amr_tree_tests.zig");
    _ = @import("amr/amr_operators_tests.zig");
    _ = @import("amr/amr_topology_tests.zig");
    _ = @import("amr/boundary_test.zig");
    _ = @import("amr/ghost_exchange_test.zig");
    _ = @import("amr/heat_equation_test.zig");
    _ = @import("amr/poisson_test.zig");
    _ = @import("amr/mpi_apply_test.zig");
    _ = @import("amr/mpi_checkpoint_test.zig");
    _ = @import("amr/mpi_ghost_exchange_test.zig");
    _ = @import("amr/mpi_shard_test.zig");
    _ = @import("amr/mpi_repartition_test.zig");
    _ = @import("amr/pipeline_stress_test.zig");
    _ = @import("physics/hamiltonian_amr_test.zig");
    _ = @import("physics/hamiltonian_pipeline_test.zig");
    _ = @import("physics/mpi_hamiltonian_test.zig");
    _ = @import("platform/mpi_tests.zig");
    _ = @import("platform/checkpoint_tests.zig");
}

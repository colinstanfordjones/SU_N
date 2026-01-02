// AMR (Adaptive Mesh Refinement) Module
//
// Block-structured AMR for multi-scale simulations.
// Frontend-parameterized for domain-agnostic operation.

const std = @import("std");

// Frontend interface
pub const frontend = @import("frontend.zig");
pub const topology = @import("topology.zig");

// Core infrastructure (domain-agnostic)
pub const morton = @import("morton.zig");
pub const field_arena = @import("field_arena.zig");
pub const field_math = @import("field_math.zig");
pub const edge_arena = @import("edge_arena.zig");
pub const edge_ghost_buffer = @import("edge_ghost_buffer.zig");
pub const apply_context = @import("apply_context.zig");

// Tree and block structure
pub const tree = @import("tree.zig");
pub const block = @import("block.zig");

// Migrated modules (Frontend-based API)
pub const operators = @import("operators.zig");
pub const dist_exchange = @import("dist_exchange.zig");
pub const ghost_policy = @import("ghost_policy.zig");
pub const flux_register = @import("flux_register.zig");
pub const multigrid = @import("multigrid.zig");
pub const ghost_buffer = @import("ghost_buffer.zig");
pub const adaptation = @import("adaptation.zig");
pub const shard = @import("shard.zig");
pub const repartition = @import("repartition.zig");

// Re-export commonly used types for convenient access.
// Canonical import: `const amr = @import("amr"); const Tree = amr.AMRTree(MyFrontend);`
pub const Morton = morton;
pub const FieldArena = field_arena.FieldArena;
pub const AMRTree = tree.AMRTree;
pub const AMRBlock = block.AMRBlock;
pub const GhostBuffer = ghost_buffer.GhostBuffer;
pub const EdgeArena = edge_arena.EdgeArena;
pub const EdgeGhostBuffer = edge_ghost_buffer.EdgeGhostBuffer;
pub const ApplyContext = apply_context.ApplyContext;
pub const FrontendInfo = frontend.FrontendInfo;
pub const validateFrontend = frontend.validateFrontend;
pub const ScalarFrontend = frontend.ScalarFrontend;
pub const ComplexScalarFrontend = frontend.ComplexScalarFrontend;
pub const AMROperators = operators.AMROperators;
pub const FluxRegister = flux_register.FluxRegister;
pub const ShardStrategy = shard.ShardStrategy;
pub const ShardContext = shard.ShardContext;
pub const BlockKey = shard.BlockKey;
pub const BlockOwner = shard.BlockOwner;

// Topology types
pub const GridTopology = topology.GridTopology;
pub const PeriodicTopology = topology.PeriodicTopology;
pub const OpenTopology = topology.OpenTopology;
pub const BoundaryType = topology.BoundaryType;
pub const TopologyConfig = topology.TopologyConfig;

test {
    _ = frontend;
    _ = topology;
    _ = morton;
    _ = field_arena;
    _ = field_math;
    _ = edge_arena;
    _ = edge_ghost_buffer;
    _ = apply_context;
    _ = tree;
    _ = block;
    _ = operators;
    _ = ghost_buffer;
    _ = dist_exchange;
    _ = ghost_policy;
    _ = adaptation;
    _ = shard;
    _ = repartition;
}

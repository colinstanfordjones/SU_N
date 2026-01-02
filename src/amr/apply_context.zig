//! ApplyContext - Unified state bundle for kernel execution
//!
//! Bundles all state references needed for kernel execution on AMR grids.
//! This enables a clean separation between state management (AMR) and
//! computation (kernels).
//!
//! The context provides:
//! - Tree structure reference
//! - Field input/output arenas
//! - Field ghost buffers
//! - Edge arena (for edge-centered data)
//! - Edge ghost buffers (for edge-centered data)
//! - Flux register (for conservation)

const std = @import("std");
const field_arena_mod = @import("field_arena.zig");
const edge_arena_mod = @import("edge_arena.zig");
const ghost_buffer_mod = @import("ghost_buffer.zig");
const edge_ghost_buffer_mod = @import("edge_ghost_buffer.zig");
const flux_register_mod = @import("flux_register.zig");
const tree_mod = @import("tree.zig");

/// Context for kernel execution on AMR grids.
///
/// Bundles all state references needed for apply() operations.
/// Supports both scalar field kernels and edge-centered kernels.
///
/// ## Usage (Scalar Kernel)
///
/// ```zig
/// var ctx = ApplyContext(Frontend).init(tree);
/// ctx.field_in = &psi_in;
/// ctx.field_out = &psi_out;
/// ctx.field_ghosts = &ghosts;
///
/// try tree.applyWithContext(kernel, &ctx);
/// ```
///
/// ## Usage (Edge-Centered Kernel)
///
/// ```zig
/// var ctx = ApplyContext(Frontend).init(tree);
/// ctx.field_in = &psi_in;
/// ctx.field_out = &psi_out;
/// ctx.field_ghosts = &field_ghosts;
/// ctx.edges = &edge_arena;
/// ctx.edge_ghosts = &edge_ghost_buffer;
///
/// try tree.applyWithContext(kernel, &ctx);
/// ```
pub fn ApplyContext(comptime Frontend: type) type {
    const Tree = tree_mod.AMRTree(Frontend);
    const FieldArena = field_arena_mod.FieldArena(Frontend);
    const GhostBuffer = ghost_buffer_mod.GhostBuffer(Frontend);
    const FluxRegister = flux_register_mod.FluxRegister(Tree);

    const has_edges = @hasDecl(Frontend, "EdgeType");

    return struct {
        const Self = @This();

        // Types exposed for kernel use
        pub const TreeType = Tree;
        pub const FieldArenaType = FieldArena;
        pub const GhostBufferType = GhostBuffer;
        pub const FluxRegisterType = FluxRegister;
        pub const FrontendType = Frontend;

        /// Reference to the AMR tree (required)
        tree: *Tree,

        /// Input field arena (optional - some kernels don't need input)
        field_in: ?*const FieldArena = null,

        /// Output field arena (optional - some kernels only read)
        field_out: ?*FieldArena = null,

        /// Field ghost buffer (optional - interior-only kernels)
        field_ghosts: ?*GhostBuffer = null,

        /// Edge arena (optional - only for edge-centered kernels)
        edges: if (has_edges) ?*const edge_arena_mod.EdgeArena(Frontend) else void = if (has_edges) null else {},

        /// Edge ghost buffer (optional - only for edge-centered kernels)
        edge_ghosts: if (has_edges) ?*edge_ghost_buffer_mod.EdgeGhostBuffer(Frontend) else void = if (has_edges) null else {},

        /// Flux register for conservation (optional)
        flux_reg: ?*FluxRegister = null,

        /// Current timestep (for time-dependent kernels)
        dt: f64 = 0.0,

        /// Whether edge ghosts need exchange
        edge_ghosts_dirty: bool = true,

        /// Whether field ghosts need exchange
        field_ghosts_dirty: bool = true,

        /// Initialize context with required tree reference.
        pub fn init(tree: *Tree) Self {
            return Self{
                .tree = tree,
            };
        }

        /// Set field input/output for a kernel that modifies fields.
        pub fn setFields(self: *Self, in: *const FieldArena, out: *FieldArena) void {
            self.field_in = in;
            self.field_out = out;
        }

        /// Set field ghosts for boundary exchange.
        pub fn setFieldGhosts(self: *Self, ghosts: *GhostBuffer) void {
            self.field_ghosts = ghosts;
        }

        /// Set edges and edge ghosts (for edge-centered kernels).
        pub fn setEdges(
            self: *Self,
            edges: if (has_edges) *const edge_arena_mod.EdgeArena(Frontend) else void,
            edge_ghosts: if (has_edges) *edge_ghost_buffer_mod.EdgeGhostBuffer(Frontend) else void,
        ) void {
            if (comptime has_edges) {
                self.edges = edges;
                self.edge_ghosts = edge_ghosts;
            }
        }

        /// Set flux register for conservation.
        pub fn setFluxRegister(self: *Self, reg: *FluxRegister) void {
            self.flux_reg = reg;
        }

        /// Mark ghosts as needing refresh (after tree modification).
        pub fn invalidateGhosts(self: *Self) void {
            self.field_ghosts_dirty = true;
            self.edge_ghosts_dirty = true;
        }

        /// Check if context has field data configured.
        pub fn hasFieldData(self: *const Self) bool {
            return self.field_in != null and self.field_out != null;
        }

        /// Check if context has edge data configured (always false if no EdgeType).
        pub fn hasEdgeData(self: *const Self) bool {
            if (comptime has_edges) {
                return self.edges != null;
            }
            return false;
        }

        /// Check if context is properly configured for kernel execution.
        pub fn isValid(self: *const Self) bool {
            // At minimum, need tree and either field data or edge data
            return self.hasFieldData() or self.hasEdgeData();
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const TestFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
};

const Complex = std.math.Complex(f64);

const U1Edge = struct {
    phase: f64,
    pub fn identity() U1Edge {
        return .{ .phase = 0.0 };
    }
};

const TestEdgeFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const FieldType = Complex;
    pub const EdgeType = U1Edge;
};

const TestTopology = struct {
    pub const Nd = 2;
    pub const extents = [2]f64{ 4.0, 4.0 };
    pub fn getBoundary(comptime dim: usize, comptime side: u1) type {
        _ = dim;
        _ = side;
        return struct {
            pub const kind = .periodic;
        };
    }
};

test "ApplyContext - scalar frontend" {
    const Ctx = ApplyContext(TestFrontend);

    // Verify it compiles and has expected fields
    try std.testing.expect(@hasField(Ctx, "tree"));
    try std.testing.expect(@hasField(Ctx, "field_in"));
    try std.testing.expect(@hasField(Ctx, "field_out"));
    try std.testing.expect(@hasField(Ctx, "field_ghosts"));
}

test "ApplyContext - edge frontend" {
    const Ctx = ApplyContext(TestEdgeFrontend);

    // Verify edge-specific fields exist
    try std.testing.expect(@hasField(Ctx, "edges"));
    try std.testing.expect(@hasField(Ctx, "edge_ghosts"));
}

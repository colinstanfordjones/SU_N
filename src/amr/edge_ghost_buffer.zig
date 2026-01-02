//! EdgeGhostBuffer - Storage for AMR edge ghost faces
//!
//! Manages optional ghost face buffers for edge-centered data per block index.
//! For each face of a block, stores edges for all Nd directions.

const std = @import("std");
const tree_mod = @import("tree.zig");
const block_mod = @import("block.zig");

pub fn EdgeGhostBuffer(comptime Frontend: type) type {
    if (!@hasDecl(Frontend, "EdgeType")) {
        @compileError("EdgeGhostBuffer requires a Frontend with EdgeType");
    }

    const Nd = Frontend.Nd;
    const EdgeType = Frontend.EdgeType;
    const Block = block_mod.AMRBlock(Frontend);
    const num_faces = Block.num_ghost_faces; // 2 * Nd
    const face_size = Block.ghost_face_size; // block_size^(Nd-1)

    return struct {
        const Self = @This();

        pub const Edge = EdgeType;
        pub const dimensions = Nd;

        fn defaultEdgeValue() EdgeType {
            if (@hasDecl(EdgeType, "identity")) {
                return EdgeType.identity();
            }
            return std.mem.zeroes(EdgeType);
        }

        /// Ghost storage for a single block
        /// Stores [face_idx][edge_direction][site_on_face]
        /// Each face stores all Nd edge directions for proper interpolation
        pub const EdgeGhostFaces = struct {
            allocator: std.mem.Allocator,
            data: [num_faces][Nd][]EdgeType,
            initialized: bool,

            pub fn init(allocator: std.mem.Allocator) EdgeGhostFaces {
                return EdgeGhostFaces{
                    .allocator = allocator,
                    .data = undefined,
                    .initialized = false,
                };
            }

            pub fn deinit(self: *EdgeGhostFaces) void {
                if (self.initialized) {
                    for (0..num_faces) |f| {
                        for (0..Nd) |d| {
                            self.allocator.free(self.data[f][d]);
                        }
                    }
                    self.initialized = false;
                }
            }

            pub fn ensureInitialized(self: *EdgeGhostFaces) !void {
                if (self.initialized) return;

                const value = defaultEdgeValue();
                for (0..num_faces) |f| {
                    for (0..Nd) |d| {
                        self.data[f][d] = try self.allocator.alloc(EdgeType, face_size);
                        for (self.data[f][d]) |*edge| edge.* = value;
                    }
                }
                self.initialized = true;
            }

            /// Get const slice of edges for a face and direction
            pub fn get(self: *const EdgeGhostFaces, face_idx: usize, edge_dim: usize) []const EdgeType {
                std.debug.assert(self.initialized);
                std.debug.assert(face_idx < num_faces);
                std.debug.assert(edge_dim < Nd);
                return self.data[face_idx][edge_dim];
            }

            /// Get mutable slice of edges for a face and direction
            pub fn getMut(self: *EdgeGhostFaces, face_idx: usize, edge_dim: usize) []EdgeType {
                std.debug.assert(self.initialized);
                std.debug.assert(face_idx < num_faces);
                std.debug.assert(edge_dim < Nd);
                return self.data[face_idx][edge_dim];
            }

            /// Set all edges in a face to a default value
            pub fn clearFace(self: *EdgeGhostFaces, face_idx: usize) void {
                std.debug.assert(self.initialized);
                std.debug.assert(face_idx < num_faces);
                const value = defaultEdgeValue();
                for (0..Nd) |d| {
                    for (self.data[face_idx][d]) |*edge| {
                        edge.* = value;
                    }
                }
            }
        };

        pub const LinkGhostFaces = EdgeGhostFaces; // Compatibility alias for gauge code.

        allocator: std.mem.Allocator,
        slots: []?*EdgeGhostFaces,

        pub fn init(allocator: std.mem.Allocator, max_blocks: usize) !Self {
            const slots = try allocator.alloc(?*EdgeGhostFaces, max_blocks);
            for (slots) |*slot| slot.* = null;
            return Self{
                .allocator = allocator,
                .slots = slots,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.slots) |slot| {
                if (slot) |ptr| {
                    ptr.deinit();
                    self.allocator.destroy(ptr);
                }
            }
            self.allocator.free(self.slots);
        }

        /// Grow capacity to hold at least max_blocks.
        pub fn ensureCapacity(self: *Self, max_blocks: usize) !void {
            if (max_blocks <= self.slots.len) return;

            const old_len = self.slots.len;
            self.slots = try self.allocator.realloc(self.slots, max_blocks);
            for (self.slots[old_len..]) |*slot| slot.* = null;
        }

        /// Ensure ghost storage exists for all active blocks in tree.
        pub fn ensureForTree(self: *Self, tree: *const tree_mod.AMRTree(Frontend)) !void {
            try self.ensureCapacity(tree.blocks.items.len);

            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                if (!tree.hasFieldSlot(idx)) continue;
                if (self.slots[idx] == null) {
                    const ghost = try self.allocator.create(EdgeGhostFaces);
                    ghost.* = EdgeGhostFaces.init(self.allocator);
                    try ghost.ensureInitialized();
                    self.slots[idx] = ghost;
                }
            }
        }

        /// Release ghost storage for invalid blocks.
        pub fn trimForTree(self: *Self, tree: *const tree_mod.AMRTree(Frontend)) void {
            const limit = @min(self.slots.len, tree.blocks.items.len);
            for (0..limit) |idx| {
                const block = &tree.blocks.items[idx];
                if (block.block_index != std.math.maxInt(usize) and tree.hasFieldSlot(idx)) continue;
                if (self.slots[idx]) |ptr| {
                    ptr.deinit();
                    self.allocator.destroy(ptr);
                    self.slots[idx] = null;
                }
            }

            if (tree.blocks.items.len < self.slots.len) {
                for (tree.blocks.items.len..self.slots.len) |idx| {
                    if (self.slots[idx]) |ptr| {
                        ptr.deinit();
                        self.allocator.destroy(ptr);
                        self.slots[idx] = null;
                    }
                }
            }
        }

        /// Get ghost storage for a block (may be null)
        pub fn get(self: *const Self, block_idx: usize) ?*EdgeGhostFaces {
            if (block_idx >= self.slots.len) return null;
            return self.slots[block_idx];
        }

        /// Get slice of ghost storage pointers
        pub fn slice(self: *const Self, len: usize) []?*EdgeGhostFaces {
            if (len > self.slots.len) return self.slots;
            return self.slots[0..len];
        }

        /// Mark all ghosts as needing refresh
        pub fn invalidateAll(self: *Self) void {
            for (self.slots) |slot| {
                if (slot) |ghost| {
                    if (ghost.initialized) {
                        for (0..num_faces) |f| {
                            ghost.clearFace(f);
                        }
                    }
                }
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

// Mock edge type for testing
const U1Link = struct {
    phase: f64,

    pub fn identity() U1Link {
        return .{ .phase = 0.0 };
    }
};

const TestEdgeFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const EdgeType = U1Link;
};

test "EdgeGhostBuffer - basic allocation" {
    const Buffer = EdgeGhostBuffer(TestEdgeFrontend);
    var buffer = try Buffer.init(std.testing.allocator, 10);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(usize, 10), buffer.slots.len);
    for (buffer.slots) |slot| {
        try std.testing.expectEqual(@as(?*Buffer.EdgeGhostFaces, null), slot);
    }
}

test "EdgeGhostFaces - init and access" {
    const Buffer = EdgeGhostBuffer(TestEdgeFrontend);
    var ghost = Buffer.EdgeGhostFaces.init(std.testing.allocator);
    defer ghost.deinit();

    try ghost.ensureInitialized();

    const edges = ghost.get(0, 0);
    try std.testing.expectEqual(@as(usize, 4), edges.len);
    for (edges) |edge| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), edge.phase, 1e-10);
    }

    const mut_edges = ghost.getMut(1, 1);
    mut_edges[2].phase = 1.5;
    const updated = ghost.get(1, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), updated[2].phase, 1e-10);
}

test "EdgeGhostFaces - clearFace" {
    const Buffer = EdgeGhostBuffer(TestEdgeFrontend);
    var ghost = Buffer.EdgeGhostFaces.init(std.testing.allocator);
    defer ghost.deinit();

    try ghost.ensureInitialized();

    // Modify some edges
    const mut_edges = ghost.getMut(0, 0);
    mut_edges[1].phase = 2.5;
    mut_edges[3].phase = -1.0;

    ghost.clearFace(0);

    const cleared = ghost.get(0, 0);
    for (cleared) |edge| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), edge.phase, 1e-10);
    }
}

//! EdgeArena - Pre-allocated storage for edge-centered data
//!
//! Enables zero-allocation mesh adaptation by pre-allocating storage for
//! the maximum possible number of blocks. Uses a free-list for O(1) slot
//! allocation and deallocation.
//!
//! This module works with frontends that define EdgeType.

const std = @import("std");

/// Pre-allocated arena for edge-centered data.
///
/// Each block stores volume * Nd edges, where volume = block_size^Nd.
/// Edges are indexed as: edges[site * Nd + mu] for site in 0..volume, mu in 0..Nd.
///
/// ## Usage
///
/// ```zig
/// const MyFrontend = struct {
///     pub const Nd: usize = 4;
///     pub const block_size: usize = 16;
///     pub const EdgeType = MyEdge;
/// };
///
/// const Arena = EdgeArena(MyFrontend);
/// var arena = try Arena.init(allocator, max_blocks);
/// defer arena.deinit();
///
/// // Get a slot for a new block
/// const slot = arena.allocSlot();
/// const edges = arena.getSlot(slot);
/// // edges[site * 4 + mu] = edge value
///
/// // Return slot when block is deallocated
/// arena.freeSlot(slot);
/// ```
pub fn EdgeArena(comptime Frontend: type) type {
    if (!@hasDecl(Frontend, "EdgeType")) {
        @compileError("EdgeArena requires a Frontend with EdgeType");
    }

    const Nd = Frontend.Nd;
    const block_size = Frontend.block_size;
    const EdgeType = Frontend.EdgeType;

    // Calculate volume = block_size^Nd
    const volume: usize = comptime blk: {
        var v: usize = 1;
        for (0..Nd) |_| v *= block_size;
        break :blk v;
    };

    const edges_per_block = volume * Nd;

    return struct {
        const Self = @This();

        pub const Edge = EdgeType;
        pub const block_volume = volume;
        pub const num_edges = edges_per_block;
        pub const num_links = num_edges; // Compatibility alias for gauge code.
        pub const dimensions = Nd;

        /// Pre-allocated edge storage for all possible blocks
        /// Aligned for SIMD access
        storage: []align(64) [edges_per_block]EdgeType,

        /// Free slot indices (stack-based free list)
        free_slots: []usize,

        /// Number of free slots available
        free_count: usize,

        /// Maximum number of blocks this arena can hold
        max_blocks: usize,

        /// Allocator used for this arena
        allocator: std.mem.Allocator,

        /// Initialize arena with pre-allocated storage for max_blocks blocks
        pub fn init(allocator: std.mem.Allocator, max_blocks: usize) !Self {
            const storage = try allocator.alignedAlloc([edges_per_block]EdgeType, .@"64", max_blocks);

            // Initialize free list with all indices (in reverse order for stack pop)
            const free_slots = try allocator.alloc(usize, max_blocks);
            for (free_slots, 0..) |*slot, i| {
                slot.* = max_blocks - 1 - i;
            }

            return Self{
                .storage = storage,
                .free_slots = free_slots,
                .free_count = max_blocks,
                .max_blocks = max_blocks,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.storage);
            self.allocator.free(self.free_slots);
        }

        /// Ensure the arena can hold at least min_blocks slots.
        /// Preserves existing slot indices and edge data.
        pub fn ensureCapacity(self: *Self, min_blocks: usize) !void {
            if (min_blocks <= self.max_blocks) return;

            var new_max = self.max_blocks;
            if (new_max == 0) new_max = 1;
            while (new_max < min_blocks) {
                if (new_max > std.math.maxInt(usize) / 2) {
                    new_max = min_blocks;
                    break;
                }
                new_max *= 2;
            }

            const new_storage = try self.allocator.alignedAlloc([edges_per_block]EdgeType, .@"64", new_max);
            errdefer self.allocator.free(new_storage);

            for (0..self.max_blocks) |i| {
                @memcpy(&new_storage[i], &self.storage[i]);
            }

            const new_free_slots = try self.allocator.alloc(usize, new_max);
            errdefer self.allocator.free(new_free_slots);

            if (self.free_count > 0) {
                std.mem.copyForwards(usize, new_free_slots[0..self.free_count], self.free_slots[0..self.free_count]);
            }

            var idx = self.free_count;
            var slot = new_max;
            while (slot > self.max_blocks) : (slot -= 1) {
                new_free_slots[idx] = slot - 1;
                idx += 1;
            }

            self.allocator.free(self.storage);
            self.allocator.free(self.free_slots);

            self.storage = new_storage;
            self.free_slots = new_free_slots;
            self.free_count = idx;
            self.max_blocks = new_max;
        }

        /// Allocate a slot from the free list
        /// Returns null if no slots available
        pub fn allocSlot(self: *Self) ?usize {
            if (self.free_count == 0) return null;
            self.free_count -= 1;
            return self.free_slots[self.free_count];
        }

        /// Return a slot to the free list
        pub fn freeSlot(self: *Self, slot: usize) void {
            std.debug.assert(slot < self.max_blocks);
            std.debug.assert(self.free_count < self.max_blocks);
            self.free_slots[self.free_count] = slot;
            self.free_count += 1;
        }

        /// Get mutable reference to edge data for a slot
        /// Returns slice of [volume * Nd] edges indexed as edges[site * Nd + mu]
        pub fn getSlot(self: *Self, slot: usize) []EdgeType {
            std.debug.assert(slot < self.max_blocks);
            return &self.storage[slot];
        }

        /// Get const reference to edge data for a slot
        pub fn getSlotConst(self: *const Self, slot: usize) []const EdgeType {
            std.debug.assert(slot < self.max_blocks);
            return &self.storage[slot];
        }

        /// Get edge at specific site and direction
        pub fn getEdge(self: *const Self, slot: usize, site: usize, mu: usize) EdgeType {
            std.debug.assert(slot < self.max_blocks);
            std.debug.assert(site < volume);
            std.debug.assert(mu < Nd);
            return self.storage[slot][site * Nd + mu];
        }

        /// Compatibility alias for getEdge.
        pub fn getLink(self: *const Self, slot: usize, site: usize, mu: usize) EdgeType {
            return self.getEdge(slot, site, mu);
        }

        /// Set edge at specific site and direction
        pub fn setEdge(self: *Self, slot: usize, site: usize, mu: usize, value: EdgeType) void {
            std.debug.assert(slot < self.max_blocks);
            std.debug.assert(site < volume);
            std.debug.assert(mu < Nd);
            self.storage[slot][site * Nd + mu] = value;
        }

        /// Compatibility alias for setEdge.
        pub fn setLink(self: *Self, slot: usize, site: usize, mu: usize, value: EdgeType) void {
            self.setEdge(slot, site, mu, value);
        }

        /// Get pointer to edge at specific site and direction (for in-place modification)
        pub fn getEdgePtr(self: *Self, slot: usize, site: usize, mu: usize) *EdgeType {
            std.debug.assert(slot < self.max_blocks);
            std.debug.assert(site < volume);
            std.debug.assert(mu < Nd);
            return &self.storage[slot][site * Nd + mu];
        }

        /// Compatibility alias for getEdgePtr.
        pub fn getLinkPtr(self: *Self, slot: usize, site: usize, mu: usize) *EdgeType {
            return self.getEdgePtr(slot, site, mu);
        }

        /// Number of currently allocated slots
        pub fn allocatedCount(self: *const Self) usize {
            return self.max_blocks - self.free_count;
        }

        /// Maximum number of slots available.
        pub fn capacity(self: *const Self) usize {
            return self.max_blocks;
        }

        /// Check if arena is full
        pub fn isFull(self: *const Self) bool {
            return self.free_count == 0;
        }

        /// Check if arena is empty
        pub fn isEmpty(self: *const Self) bool {
            return self.free_count == self.max_blocks;
        }

        fn defaultEdgeValue() EdgeType {
            if (@hasDecl(EdgeType, "identity")) {
                return EdgeType.identity();
            }
            return std.mem.zeroes(EdgeType);
        }

        /// Initialize all edges in a slot to a default value.
        pub fn initSlotToDefault(self: *Self, slot: usize) void {
            const edges = self.getSlot(slot);
            const value = defaultEdgeValue();
            for (edges) |*edge| {
                edge.* = value;
            }
        }

        /// Initialize all edges in a slot to identity (falls back to zero if no identity).
        pub fn initSlotToIdentity(self: *Self, slot: usize) void {
            self.initSlotToDefault(slot);
        }

        /// Defragment arena storage to match a new block ordering.
        /// After defragmentation, block i's data is at storage[i].
        pub fn defragmentWithOrder(self: *Self, edge_slots: []usize, active_count: usize) !void {
            if (active_count == 0) {
                self.free_count = self.max_blocks;
                for (self.free_slots, 0..) |*slot, i| {
                    slot.* = self.max_blocks - 1 - i;
                }
                return;
            }

            const new_storage = try self.allocator.alignedAlloc([edges_per_block]EdgeType, .@"64", active_count);
            defer self.allocator.free(new_storage);

            for (0..active_count) |i| {
                const old_slot = edge_slots[i];
                if (old_slot < self.max_blocks) {
                    @memcpy(&new_storage[i], &self.storage[old_slot]);
                } else {
                    const value = defaultEdgeValue();
                    for (&new_storage[i]) |*edge| {
                        edge.* = value;
                    }
                }
            }

            for (0..active_count) |i| {
                @memcpy(&self.storage[i], &new_storage[i]);
            }

            for (0..active_count) |i| {
                edge_slots[i] = i;
            }

            self.free_count = self.max_blocks - active_count;
            for (0..self.free_count) |i| {
                self.free_slots[i] = self.max_blocks - 1 - i;
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

// Mock U(1) edge type for testing
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

test "EdgeArena - basic allocation" {
    const Arena = EdgeArena(TestEdgeFrontend);
    var arena = try Arena.init(std.testing.allocator, 10);
    defer arena.deinit();

    try std.testing.expectEqual(@as(usize, 10), arena.free_count);
    try std.testing.expectEqual(@as(usize, 0), arena.allocatedCount());
    try std.testing.expect(arena.isEmpty());

    const slot1 = arena.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 9), arena.free_count);
    try std.testing.expectEqual(@as(usize, 1), arena.allocatedCount());

    const slot2 = arena.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 8), arena.free_count);

    arena.freeSlot(slot1);
    try std.testing.expectEqual(@as(usize, 9), arena.free_count);

    arena.freeSlot(slot2);
    try std.testing.expectEqual(@as(usize, 10), arena.free_count);
    try std.testing.expect(arena.isEmpty());
}

test "EdgeArena - num_edges calculation" {
    const Arena = EdgeArena(TestEdgeFrontend);
    // 2D with block_size=4: volume = 4^2 = 16, edges = 16 * 2 = 32
    try std.testing.expectEqual(@as(usize, 16), Arena.block_volume);
    try std.testing.expectEqual(@as(usize, 32), Arena.num_edges);
}

test "EdgeArena - edge access" {
    const Arena = EdgeArena(TestEdgeFrontend);
    var arena = try Arena.init(std.testing.allocator, 2);
    defer arena.deinit();

    const slot = arena.allocSlot().?;
    arena.initSlotToIdentity(slot);

    // All edges should be identity
    const edge = arena.getEdge(slot, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), edge.phase, 1e-10);

    // Modify an edge
    arena.setEdge(slot, 5, 1, .{ .phase = 1.5 });
    const modified = arena.getEdge(slot, 5, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), modified.phase, 1e-10);
}

test "EdgeArena - exhaustion" {
    const Arena = EdgeArena(TestEdgeFrontend);
    var arena = try Arena.init(std.testing.allocator, 2);
    defer arena.deinit();

    _ = arena.allocSlot().?;
    _ = arena.allocSlot().?;

    try std.testing.expect(arena.isFull());
    try std.testing.expectEqual(@as(?usize, null), arena.allocSlot());
}

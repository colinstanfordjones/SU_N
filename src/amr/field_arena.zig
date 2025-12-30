//! FieldArena - Pre-allocated storage for field data
//!
//! Enables zero-allocation mesh adaptation by pre-allocating storage for
//! the maximum possible number of blocks. Uses a free-list for O(1) slot
//! allocation and deallocation.
//!
//! This module is domain-agnostic - it works with any FieldType defined
//! by the Frontend interface.

const std = @import("std");
const frontend_mod = @import("frontend.zig");

/// Pre-allocated arena for field storage.
///
/// Enables zero-allocation mesh adaptation by pre-allocating storage for
/// the maximum possible number of blocks. Uses a free-list for O(1) slot
/// allocation and deallocation.
///
/// ## Usage
///
/// ```zig
/// const MyFrontend = struct {
///     pub const Nd: usize = 4;
///     pub const block_size: usize = 16;
///     pub const FieldType = [4]Complex;
/// };
///
/// const Arena = FieldArena(MyFrontend);
/// var arena = try Arena.init(allocator, max_blocks);
/// defer arena.deinit();
///
/// // Get a slot for a new block
/// const slot = arena.allocSlot();
/// const field_data = arena.getSlot(slot);
///
/// // Return slot when block is deallocated
/// arena.freeSlot(slot);
/// ```
pub fn FieldArena(comptime Frontend: type) type {
    const Info = frontend_mod.FrontendInfo(Frontend);
    const FieldType = Info.FieldType;
    const block_volume = Info.volume;

    return struct {
        const Self = @This();

        pub const volume = block_volume;
        pub const Field = FieldType;

        /// Pre-allocated field storage for all possible blocks
        /// Aligned for SIMD access
        storage: []align(64) [block_volume]FieldType,

        /// Free slot indices (stack-based free list)
        /// Indices into storage array that are available
        free_slots: []usize,

        /// Number of free slots available
        free_count: usize,

        /// Maximum number of blocks this arena can hold
        max_blocks: usize,

        /// Allocator used for this arena
        allocator: std.mem.Allocator,

        /// Initialize arena with pre-allocated storage for max_blocks blocks
        pub fn init(allocator: std.mem.Allocator, max_blocks: usize) !Self {
            const storage = try allocator.alignedAlloc([block_volume]FieldType, .@"64", max_blocks);

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
        /// Preserves existing slot indices and field data.
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

            const new_storage = try self.allocator.alignedAlloc([block_volume]FieldType, .@"64", new_max);
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

        /// Get mutable reference to field data for a slot
        pub fn getSlot(self: *Self, slot: usize) []FieldType {
            std.debug.assert(slot < self.max_blocks);
            return &self.storage[slot];
        }

        /// Get const reference to field data for a slot
        pub fn getSlotConst(self: *const Self, slot: usize) []const FieldType {
            std.debug.assert(slot < self.max_blocks);
            return &self.storage[slot];
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

        /// Zero out all data in a slot
        pub fn zeroSlot(self: *Self, slot: usize) void {
            const data = self.getSlot(slot);
            @memset(data, std.mem.zeroes(FieldType));
        }

        /// Defragment arena storage to match a new block ordering.
        ///
        /// After calling tree.reorder(), field data is scattered across the arena
        /// based on original allocation order. This method rearranges field data
        /// so that block i's data is at storage[i], enabling linear memory access.
        ///
        /// Parameters:
        /// - field_slots: The tree's field_slots array (mapping block_idx -> slot_idx)
        /// - active_count: Number of active blocks (length of valid field_slots prefix)
        ///
        /// After defragmentation:
        /// - Field data is compacted to slots 0..active_count-1
        /// - field_slots[i] = i for all active blocks
        /// - Free list contains slots active_count..max_blocks-1
        ///
        /// **IMPORTANT**: This modifies the field_slots array in place.
        pub fn defragmentWithOrder(self: *Self, field_slots: []usize, active_count: usize) !void {
            if (active_count == 0) {
                // Reset free list to all slots
                self.free_count = self.max_blocks;
                for (self.free_slots, 0..) |*slot, i| {
                    slot.* = self.max_blocks - 1 - i;
                }
                return;
            }

            // Allocate temporary storage for reordering
            const new_storage = try self.allocator.alignedAlloc([block_volume]FieldType, .@"64", active_count);
            defer self.allocator.free(new_storage);

            // Copy field data in new order
            // For block i (in new Morton order), copy from storage[field_slots[i]] to new_storage[i]
            for (0..active_count) |i| {
                const old_slot = field_slots[i];
                if (old_slot < self.max_blocks) {
                    @memcpy(&new_storage[i], &self.storage[old_slot]);
                } else {
                    // Zero uninitialized memory for blocks without field slots
                    @memset(&new_storage[i], std.mem.zeroes(FieldType));
                }
            }

            // Copy back to main storage
            for (0..active_count) |i| {
                @memcpy(&self.storage[i], &new_storage[i]);
            }

            // Update field_slots to sequential indices
            for (0..active_count) |i| {
                field_slots[i] = i;
            }

            // Rebuild free list: slots active_count..max_blocks-1 are free
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

const Complex = std.math.Complex(f64);

const TestFrontend = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
};

const ComplexFrontend = struct {
    pub const Nd: usize = 4;
    pub const block_size: usize = 16;
    pub const FieldType = Complex;
};

test "FieldArena - basic allocation" {
    const Arena = FieldArena(TestFrontend);
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

test "FieldArena - volume calculation" {
    const Arena = FieldArena(TestFrontend);
    try std.testing.expectEqual(@as(usize, 16), Arena.volume); // 4^2

    const Arena4D = FieldArena(ComplexFrontend);
    try std.testing.expectEqual(@as(usize, 65536), Arena4D.volume); // 16^4
}

test "FieldArena - data access" {
    const Arena = FieldArena(TestFrontend);
    var arena = try Arena.init(std.testing.allocator, 2);
    defer arena.deinit();

    const slot = arena.allocSlot().?;
    const data = arena.getSlot(slot);

    // Write some data
    data[0] = 42.0;
    data[15] = 99.0;

    // Read it back
    const data_const = arena.getSlotConst(slot);
    try std.testing.expectEqual(@as(f64, 42.0), data_const[0]);
    try std.testing.expectEqual(@as(f64, 99.0), data_const[15]);
}

test "FieldArena - exhaustion" {
    const Arena = FieldArena(TestFrontend);
    var arena = try Arena.init(std.testing.allocator, 2);
    defer arena.deinit();

    _ = arena.allocSlot().?;
    _ = arena.allocSlot().?;

    try std.testing.expect(arena.isFull());
    try std.testing.expectEqual(@as(?usize, null), arena.allocSlot());
}

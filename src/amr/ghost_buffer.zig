//! GhostBuffer - Storage for AMR ghost faces
//!
//! Manages optional ghost face buffers per block index. This keeps ghost storage
//! within the AMR module so higher-level modules can remain stateless.

const std = @import("std");
const frontend_mod = @import("frontend.zig");
const tree_mod = @import("tree.zig");
const block_mod = @import("block.zig");

pub fn GhostBuffer(comptime Frontend: type) type {
    const Info = frontend_mod.FrontendInfo(Frontend);
    const Block = block_mod.AMRBlock(Frontend);
    const FieldType = Info.FieldType;
    const num_faces = Block.num_ghost_faces;
    const face_size = Block.ghost_face_size;

    return struct {
        const Self = @This();

        /// Type alias for ghost face storage
        pub const GhostFaces = [num_faces][face_size]FieldType;

        allocator: std.mem.Allocator,
        slots: []?*GhostFaces,

        pub fn init(allocator: std.mem.Allocator, max_blocks: usize) !Self {
            const slots = try allocator.alloc(?*GhostFaces, max_blocks);
            for (slots) |*slot| slot.* = null;
            return Self{
                .allocator = allocator,
                .slots = slots,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.slots) |slot| {
                if (slot) |ptr| {
                    self.allocator.destroy(ptr);
                }
            }
            self.allocator.free(self.slots);
        }

        /// Grow capacity to hold at least max_blocks.
        /// Note: realloc preserves existing slot pointers. This is allocator-dependent
        /// but holds for Zig's GeneralPurposeAllocator and most system allocators.
        pub fn ensureCapacity(self: *Self, max_blocks: usize) !void {
            if (max_blocks <= self.slots.len) return;

            const old_len = self.slots.len;
            self.slots = try self.allocator.realloc(self.slots, max_blocks);
            for (self.slots[old_len..]) |*slot| slot.* = null;
        }

        pub fn ensureForTree(self: *Self, tree: *const tree_mod.AMRTree(Frontend)) !void {
            try self.ensureCapacity(tree.blocks.items.len);

            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;
                if (!tree.hasFieldSlot(idx)) continue;
                if (self.slots[idx] == null) {
                    self.slots[idx] = try self.allocator.create(GhostFaces);
                }
            }
        }

        /// Release ghost storage for invalid blocks or blocks without field slots.
        pub fn trimForTree(self: *Self, tree: *const tree_mod.AMRTree(Frontend)) void {
            const limit = @min(self.slots.len, tree.blocks.items.len);
            for (0..limit) |idx| {
                const block = &tree.blocks.items[idx];
                if (block.block_index != std.math.maxInt(usize) and tree.hasFieldSlot(idx)) continue;
                if (self.slots[idx]) |ptr| {
                    self.allocator.destroy(ptr);
                    self.slots[idx] = null;
                }
            }

            if (tree.blocks.items.len < self.slots.len) {
                for (tree.blocks.items.len..self.slots.len) |idx| {
                    if (self.slots[idx]) |ptr| {
                        self.allocator.destroy(ptr);
                        self.slots[idx] = null;
                    }
                }
            }
        }

        pub fn get(self: *const Self, block_idx: usize) ?*GhostFaces {
            if (block_idx >= self.slots.len) return null;
            return self.slots[block_idx];
        }

        pub fn slice(self: *const Self, len: usize) []?*GhostFaces {
            if (len > self.slots.len) return self.slots;
            return self.slots[0..len];
        }
    };
}

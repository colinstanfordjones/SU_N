//! Gauge Field - Stateless Gauge Link Management
//!
//! Manages storage and lifecycle for gauge links and their ghosts.
//! Designed to be attached to an AMRTree but not wrap it.
//!
//! ## Usage
//!
//! ```zig
//! const gauge = @import("gauge");
//! var field = try gauge.GaugeField(Frontend).init(allocator, tree);
//! defer field.deinit();
//!
//! // On refinement/coarsening
//! try field.syncWithTree(tree);
//!
//! // Access
//! const links = field.getLinks(block_idx);
//! ```

const std = @import("std");
const amr = @import("amr");
const edge_arena_mod = amr.edge_arena;
const edge_ghost_mod = amr.edge_ghost_buffer;
const dist_exchange_mod = amr.dist_exchange;
const ghost_policy_mod = @import("ghost_policy.zig");

pub fn GaugeField(comptime Frontend: type) type {
    const Tree = amr.AMRTree(Frontend);
    const EdgeArena = edge_arena_mod.EdgeArena(Frontend);
    const EdgeGhostBuffer = edge_ghost_mod.EdgeGhostBuffer(Frontend);
    const Link = Frontend.LinkType;

    return struct {
        const Self = @This();
        // Exports for Policy
        pub const FrontendType = Frontend;
        pub const TreeType = Tree;
        pub const LinkType = Link;
        pub const EdgeArenaType = EdgeArena;
        pub const EdgeGhostFaces = EdgeGhostBuffer.EdgeGhostFaces;

        pub const Policy = ghost_policy_mod.LinkGhostPolicy(Self);
        pub const LinkExchange = dist_exchange_mod.DistExchange(Tree, Policy.Context, Policy.Payload);

        // Instance fields
        allocator: std.mem.Allocator,
        arena: EdgeArena,
        ghosts: EdgeGhostBuffer,
        link_exchange: LinkExchange,
        
        /// Mapping from tree block index to arena slot index.
        /// slots[block_idx] = arena_slot
        /// invalid/missing blocks have slot = maxInt
        slots: std.ArrayList(usize),

        pub fn init(allocator: std.mem.Allocator, tree: *const Tree) !Self {
            return initWithOptions(allocator, tree, null);
        }

        pub fn initWithOptions(allocator: std.mem.Allocator, tree: *const Tree, spec: ?Policy.ExchangeSpec) !Self {
            const max_blocks = tree.blocks.items.len;
            
            // Initialize components
            const arena = try EdgeArena.init(allocator, @max(16, max_blocks));
            const ghosts = try EdgeGhostBuffer.init(allocator, @max(16, max_blocks));
            
            // Initialize slots
            const slots = try std.ArrayList(usize).initCapacity(allocator, max_blocks);
            
            // Exchange
            const exchange_spec = spec orelse Policy.exchangeSpec();
            const link_exchange = LinkExchange.init(allocator, exchange_spec);

            return Self{
                .allocator = allocator,
                .arena = arena,
                .ghosts = ghosts,
                .link_exchange = link_exchange,
                .slots = slots,
            };
        }

        pub fn deinit(self: *Self) void {
            self.link_exchange.deinit();
            self.slots.deinit(self.allocator);
            self.ghosts.deinit();
            self.arena.deinit();
        }

        /// Ensure field storage matches tree structure.
        /// Should be called after tree.insertBlock or tree.refineBlock.
        pub fn syncWithTree(self: *Self, tree: *const Tree) !void {
            const num_blocks = tree.blocks.items.len;
            
            // Ensure capacity
            try self.arena.ensureCapacity(num_blocks);
            try self.ghosts.ensureCapacity(num_blocks);
            try self.slots.ensureTotalCapacity(self.allocator, num_blocks);

            // Fill new slots
            while (self.slots.items.len < num_blocks) {
                // If tree block exists, allocate slot. Else maxInt.
                const idx = self.slots.items.len;
                const block = &tree.blocks.items[idx];
                
                if (block.block_index != std.math.maxInt(usize)) {
                    const slot = self.arena.allocSlot() orelse return error.EdgeArenaFull;
                    self.arena.initSlotToDefault(slot);
                    self.slots.appendAssumeCapacity(slot);
                } else {
                    self.slots.appendAssumeCapacity(std.math.maxInt(usize));
                }
            }
            
            // Also ensure ghosts are allocated for active blocks
            // We use a custom version of ensureForTree logic here because we manage 'slots'
            for (tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    // Check if we have a valid slot
                    if (self.slots.items[idx] != std.math.maxInt(usize)) {
                         if (self.ghosts.slots[idx] == null) {
                            const ghost = try self.allocator.create(EdgeGhostBuffer.EdgeGhostFaces);
                            ghost.* = EdgeGhostBuffer.EdgeGhostFaces.init(self.allocator);
                            try ghost.ensureInitialized();
                            self.ghosts.slots[idx] = ghost;
                        }
                    }
                }
            }
        }

        /// React to tree reordering.
        /// perm[old_idx] = new_idx
        pub fn reorder(self: *Self, perm: []const usize) !void {
            const old_len = self.slots.items.len;
            const new_len = perm.len; // Perm length matches old length, but we filter
            
            // We need to build the new slots array
            var new_slots = try self.allocator.alloc(usize, new_len);
            @memset(new_slots, std.math.maxInt(usize));

            var active_count: usize = 0;
            for (perm, 0..) |new_idx, old_idx| {
                if (new_idx != std.math.maxInt(usize)) {
                    if (old_idx < old_len) {
                        new_slots[new_idx] = self.slots.items[old_idx];
                        if (new_slots[new_idx] != std.math.maxInt(usize)) {
                            active_count += 1;
                        }
                    }
                } else {
                    // Block deleted. Free the slot.
                    if (old_idx < old_len) {
                        const slot = self.slots.items[old_idx];
                        if (slot != std.math.maxInt(usize)) {
                            self.arena.freeSlot(slot);
                        }
                    }
                }
            }
            
            // Find max new index to size the array
            var max_idx: usize = 0;
            for (new_slots, 0..) |slot, idx| {
                if (slot != std.math.maxInt(usize)) max_idx = idx;
            }
            const compact_len = if (active_count > 0) max_idx + 1 else 0;

            // Update slots array
            self.slots.clearRetainingCapacity();
            try self.slots.appendSlice(self.allocator, new_slots[0..compact_len]);
            self.allocator.free(new_slots);

            // Defragment arena: this moves data so slot[i] == i for active blocks
            // This is optional but good for locality.
            // Wait, EdgeArena.defragmentWithOrder expects `link_slots` to be mapped.
            // And it updates `link_slots` in place to be 0..N.
            // But `link_slots` (our self.slots) contains maxInt for holes.
            // `defragmentWithOrder` documentation says:
            // "After defragmentation, block i's data is at storage[i]."
            // But our `self.slots` maps block_idx -> slot_idx.
            // If we have holes in block_idx, we have holes in self.slots.
            // `defragmentWithOrder` seems designed for a compact list of blocks.
            // AMRTree.reorder compacts blocks (removes holes).
            // So `self.slots` should be compact after reorder (no maxInt except maybe at end?).
            // Yes, AMRTree reorder removes invalid blocks.
            
            // So we can call defragmentWithOrder.
            try self.arena.defragmentWithOrder(self.slots.items, active_count);

            // Reorder ghosts
            // EdgeGhostBuffer doesn't have a reorder method, we must do it manually.
            // The `perm` applied to blocks.
            // We need to apply it to `self.ghosts.slots`.
            const old_ghost_slots = try self.allocator.dupe(?*EdgeGhostBuffer.EdgeGhostFaces, self.ghosts.slots);
            defer self.allocator.free(old_ghost_slots);
            
            // Resize ghost slots
            try self.ghosts.ensureCapacity(compact_len);
            // Clear new slots (pointers moved, so we don't own them in old array anymore)
            @memset(self.ghosts.slots, null);
            
            for (perm, 0..) |new_idx, old_idx| {
                if (new_idx != std.math.maxInt(usize)) {
                    if (old_idx < old_ghost_slots.len) {
                        self.ghosts.slots[new_idx] = old_ghost_slots[old_idx];
                    }
                } else {
                    // Deleted block
                    if (old_idx < old_ghost_slots.len) {
                        if (old_ghost_slots[old_idx]) |ptr| {
                            ptr.deinit();
                            self.allocator.destroy(ptr);
                        }
                    }
                }
            }
        }

        /// Get link for a specific block/site/direction
        pub fn getLink(self: *const Self, block_idx: usize, site: usize, mu: usize) Link {
            if (block_idx >= self.slots.items.len) return Link.identity();
            const slot = self.slots.items[block_idx];
            if (slot == std.math.maxInt(usize)) return Link.identity();
            return self.arena.getEdge(slot, site, mu);
        }

        /// Set link for a specific block/site/direction
        pub fn setLink(self: *Self, block_idx: usize, site: usize, mu: usize, link: Link) void {
            if (block_idx >= self.slots.items.len) return;
            const slot = self.slots.items[block_idx];
            if (slot == std.math.maxInt(usize)) return;
            self.arena.setEdge(slot, site, mu, link);
        }
        
        /// Get slice of all links for a block
        pub fn getBlockLinks(self: *const Self, block_idx: usize) ?[]const Link {
            if (block_idx >= self.slots.items.len) return null;
            const slot = self.slots.items[block_idx];
            if (slot == std.math.maxInt(usize)) return null;
            return self.arena.getSlotConst(slot);
        }
        
        /// Get mutable slice of all links for a block
        pub fn getBlockLinksMut(self: *Self, block_idx: usize) ?[]Link {
            if (block_idx >= self.slots.items.len) return null;
            const slot = self.slots.items[block_idx];
            if (slot == std.math.maxInt(usize)) return null;
            return self.arena.getSlot(slot);
        }

        /// Fill ghost layers using MPI exchange.
        pub fn fillGhosts(self: *Self, tree: *amr.AMRTree(Frontend)) !void {
             const shard = tree.shardContext();
             const ctx = Policy.Context{
                 .tree = tree,
                 .arena = &self.arena,
                 .ghosts = self.ghosts.slice(tree.blocks.items.len),
                 .slots = self.slots.items,
             };
             var state = try self.link_exchange.begin(ctx, shard);
             try self.link_exchange.finish(ctx, &state);
        }

        // =====================================================================
        // Backup / Restore (HMC)
        // =====================================================================

        /// Save current link configuration for backup.
        /// Returns a slice of slices. Caller owns the memory.
        pub fn saveLinks(self: *const Self, allocator: std.mem.Allocator) ![][]Link {
            var backup = try allocator.alloc([]Link, self.slots.items.len);
            for (self.slots.items, 0..) |slot, i| {
                if (slot != std.math.maxInt(usize)) {
                    const src = self.arena.getSlotConst(slot);
                    backup[i] = try allocator.alloc(Link, src.len);
                    @memcpy(backup[i], src);
                } else {
                    backup[i] = try allocator.alloc(Link, 0); // Empty slice for invalid blocks
                }
            }
            return backup;
        }

        /// Restore link configuration from backup.
        pub fn restoreLinks(self: *Self, backup: []const []const Link) void {
            // We assume the tree structure hasn't changed (no refinement/reorder)
            // between save and restore.
            for (backup, 0..) |src, i| {
                if (i < self.slots.items.len) {
                    const slot = self.slots.items[i];
                    if (slot != std.math.maxInt(usize) and src.len > 0) {
                        const dest = self.arena.getSlot(slot);
                        if (dest.len == src.len) {
                            @memcpy(dest, src);
                        }
                    }
                }
            }
            // Invalidate ghosts
            self.ghosts.invalidateAll();
        }

        /// Free saved link backup.
        pub fn freeBackup(allocator: std.mem.Allocator, backup: [][]Link) void {
            for (backup) |slice| {
                if (slice.len > 0) allocator.free(slice);
            }
            allocator.free(backup);
        }

        // =====================================================================
        // Checkpoint / Restart
        // =====================================================================

        /// Write a checkpoint of the GaugeField state.
        /// Must be paired with a Tree checkpoint.
        pub fn writeCheckpoint(self: *const Self, writer: anytype) !void {
            const links_magic = "LINK";
            try writer.writeAll(links_magic);
            
            const num_blocks = self.slots.items.len;
            const edges_per_block = edge_arena_mod.EdgeArena(Frontend).num_edges;

            try writer.writeInt(u64, @as(u64, @intCast(num_blocks)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(edges_per_block)), .little);

            // Write links for all blocks to maintain 1-to-1 mapping with tree.blocks
            // For invalid blocks (slot == maxInt), write zeros/identity.
            
            // We need a scratch buffer for identity links
            const identity_scratch = try self.allocator.alloc(Link, edges_per_block);
            defer self.allocator.free(identity_scratch);
            for (identity_scratch) |*l| l.* = Link.identity();
            const identity_bytes = std.mem.sliceAsBytes(identity_scratch);

            for (self.slots.items) |slot| {
                if (slot != std.math.maxInt(usize)) {
                    const slice = self.arena.getSlotConst(slot);
                    if (slice.len != edges_per_block) return error.InvalidLinkLayout;
                    try writer.writeAll(std.mem.sliceAsBytes(slice));
                } else {
                    try writer.writeAll(identity_bytes);
                }
            }
        }

        /// Read a checkpoint and reconstruct the GaugeField.
        /// Requires an initialized Tree to map blocks.
        pub fn readCheckpoint(allocator: std.mem.Allocator, tree: *const Tree, reader: anytype) !Self {
            var magic_buf: [4]u8 = undefined;
            try reader.readNoEof(&magic_buf);
            if (!std.mem.eql(u8, &magic_buf, "LINK")) return error.InvalidMagic;

            const num_blocks = try reader.readInt(u64, .little);
            const stored_links_len = try reader.readInt(u64, .little);
            
            // Validate against tree
            if (num_blocks != tree.blocks.items.len) return error.IncompatibleCheckpoint;
            
            const expected_links_len = edge_arena_mod.EdgeArena(Frontend).num_edges;
            if (stored_links_len != expected_links_len) return error.IncompatibleCheckpoint;

            // Initialize self
            var field = try Self.init(allocator, tree);
            errdefer field.deinit();

            // syncWithTree will allocate slots based on tree blocks
            try field.syncWithTree(tree);

            // Now read data into slots
            const links_byte_len = stored_links_len * @sizeOf(Link);
            
            // We must read exactly num_blocks chunks
            for (0..num_blocks) |i| {
                const slot = field.slots.items[i];
                if (slot != std.math.maxInt(usize)) {
                    const dest = field.arena.getSlot(slot);
                    // Read directly into arena
                    try reader.readNoEof(std.mem.sliceAsBytes(dest));
                } else {
                    // Skip data for invalid block
                    try reader.skipBytes(links_byte_len, . {});
                }
            }
            
            return field;
        }
    };
}

//! AMR Tree Module
//!
//! Implements a linear octree for organizing AMR blocks in Nd-dimensional space.
//! Blocks are stored in a flat list ordered by Morton (Z-order) index for locality.
//!
//! ## Design
//!
//! - **Linear Octree**: Blocks stored in a flat ArrayList with a Morton index map
//! - **Morton Ordering**: Blocks sorted by Morton index for cache locality
//! - **Neighbor Finding**: Runtime queries using Morton keys (no cached neighbor indices)
//! - **Refinement**: Split leaf into 2^Nd children; coarsening merges children
//!
//! ## Frontend-Parameterized
//!
//! The tree is parameterized by a Frontend type that defines:
//! - `Nd`: Number of dimensions
//! - `block_size`: Sites per dimension within each block
//! - `FieldType`: Type stored at each lattice site
//!
//! ## Related Modules
//!
//! - `ghost`: Ghost layer filling (namespace-style functions)
//! - `adaptation`: Mesh adaptation (namespace-style functions)
//!
//! ## Usage
//!
//! ```zig
//! const frontend = @import("frontend.zig");
//! const tree_mod = @import("tree.zig");
//!
//! // Define a frontend
//! const MyFrontend = struct {
//!     pub const Nd: usize = 4;
//!     pub const block_size: usize = 16;
//!     pub const FieldType = [4]Complex;
//! };
//!
//! const Tree = tree_mod.AMRTree(MyFrontend);
//! var tree = try Tree.init(allocator, 1.0, 4, 8);
//! defer tree.deinit();
//! ```

const std = @import("std");
const frontend_mod = @import("frontend.zig");
const block_mod = @import("block.zig");
const field_arena_mod = @import("field_arena.zig");
const ghost_buffer_mod = @import("ghost_buffer.zig");
const dist_exchange_mod = @import("dist_exchange.zig");
const ghost_policy_mod = @import("ghost_policy.zig");
const flux_register_mod = @import("flux_register.zig");
const apply_context_mod = @import("apply_context.zig");
const morton_mod = @import("morton.zig");
const platform = @import("platform");
const shard_mod = @import("shard.zig");

const Complex = std.math.Complex(f64);

/// AMR Tree with 2^Nd children per internal node.
///
/// Parameters:
/// - Frontend: A type implementing the Frontend interface (Nd, block_size, FieldType)
pub fn AMRTree(comptime Frontend: type) type {
    const Info = frontend_mod.FrontendInfo(Frontend);
    const Nd = Info.Nd;
    const block_size = Info.block_size;
    const num_children = Info.num_children;

    const Block = block_mod.AMRBlock(Frontend);
    const ArenaType = field_arena_mod.FieldArena(Frontend);

    return struct {
        const Self = @This();

        const FieldPolicy = ghost_policy_mod.FieldGhostPolicy(Self);
        const FieldExchange = dist_exchange_mod.DistExchange(Self, FieldPolicy.Context, FieldPolicy.Payload);
        pub const FieldExchangeSpec = FieldPolicy.ExchangeSpec;
        pub const ExchangeOptions = struct {
            field_exchange_spec: ?FieldExchangeSpec = null,
        };

        // Compile-time constants exposed for external modules
        pub const dimensions = Nd;
        pub const children_per_node = num_children;
        pub const block_size_const = block_size;
        pub const FrontendType = Frontend;
        pub const BlockType = Block;
        pub const FieldArenaType = ArenaType;
        pub const FieldType = Info.FieldType;
        pub const ShardContext = shard_mod.ShardContext(Self);
        pub const BlockKey = morton_mod.BlockKey;
        pub const max_fine_neighbors = @as(usize, 1) << @intCast(Nd - 1);
        pub const FluxRegister = flux_register_mod.FluxRegister(Self);
        pub const ApplyContext = apply_context_mod.ApplyContext(Frontend);

        /// Information about a neighbor block, including cross-level relationships.
        pub const NeighborInfo = struct {
            block_idx: usize,
            level_diff: i8,

            pub const none = NeighborInfo{
                .block_idx = std.math.maxInt(usize),
                .level_diff = 0,
            };

            pub fn exists(self: NeighborInfo) bool {
                return self.block_idx != std.math.maxInt(usize);
            }
        };

        // Instance fields
        allocator: std.mem.Allocator,
        pool: platform.WorkStealingPool,
        pull_group: platform.TaskGroup,
        interior_group: platform.TaskGroup,
        push_group: platform.TaskGroup,
        boundary_group: platform.TaskGroup,
        field_exchange: FieldExchange,
        shard_context: ?*ShardContext,
        blocks: std.ArrayList(Block),
        field_slots: std.ArrayList(usize),
        block_index: std.AutoHashMap(BlockKey, usize),
        base_spacing: f64,
        bits_per_dim: u8,
        max_level: u8,

        /// Initialize an empty AMR tree.
        pub fn init(
            allocator: std.mem.Allocator,
            base_spacing: f64,
            bits_per_dim: u8,
            max_level: u8,
        ) !Self {
            return initWithOptions(allocator, base_spacing, bits_per_dim, max_level, .{});
        }

        /// Initialize an AMR tree with a custom exchange specification.
        pub fn initWithOptions(
            allocator: std.mem.Allocator,
            base_spacing: f64,
            bits_per_dim: u8,
            max_level: u8,
            options: ExchangeOptions,
        ) !Self {
            try validateMortonConfig(bits_per_dim, max_level);
            const blocks = std.ArrayList(Block){};
            const field_slots = std.ArrayList(usize){};
            const block_index = std.AutoHashMap(BlockKey, usize).init(allocator);
            const pool = try platform.WorkStealingPool.init(allocator, null);

            var pull_group = try platform.TaskGroup.init(allocator);
            errdefer pull_group.deinit();

            var interior_group = try platform.TaskGroup.init(allocator);
            errdefer interior_group.deinit();

            var push_group = try platform.TaskGroup.init(allocator);
            errdefer push_group.deinit();

            var boundary_group = try platform.TaskGroup.init(allocator);
            errdefer boundary_group.deinit();

            const exchange_spec = options.field_exchange_spec orelse FieldPolicy.exchangeSpec();
            const field_exchange = FieldExchange.init(allocator, exchange_spec);

            return Self{
                .allocator = allocator,
                .pool = pool,
                .pull_group = pull_group,
                .interior_group = interior_group,
                .push_group = push_group,
                .boundary_group = boundary_group,
                .field_exchange = field_exchange,
                .shard_context = null,
                .blocks = blocks,
                .field_slots = field_slots,
                .block_index = block_index,
                .base_spacing = base_spacing,
                .bits_per_dim = bits_per_dim,
                .max_level = max_level,
            };
        }

        pub fn deinit(self: *Self) void {
            self.field_exchange.deinit();
            self.boundary_group.deinit();
            self.push_group.deinit();
            self.interior_group.deinit();
            self.pull_group.deinit();
            self.pool.deinit();
            self.blocks.deinit(self.allocator);
            self.field_slots.deinit(self.allocator);
            self.block_index.deinit();
        }

        fn validateMortonConfig(bits_per_dim: u8, max_level: u8) !void {
            const max_bits: u8 = @intCast(64 / Nd);
            const total: u16 = @as(u16, bits_per_dim) + @as(u16, max_level);
            if (total > @as(u16, max_bits)) return error.MortonBitsOverflow;
        }

        pub fn attachShard(self: *Self, shard: *ShardContext) void {
            self.shard_context = shard;
        }

        pub fn detachShard(self: *Self) void {
            self.shard_context = null;
        }

        pub fn shardContext(self: *const Self) ?*ShardContext {
            return self.shard_context;
        }

        // =====================================================================
        // Block Insertion
        // =====================================================================

        pub fn insertBlock(self: *Self, origin: [Nd]usize, level: u8) !usize {
            const spacing = self.base_spacing / @as(f64, @floatFromInt(@as(usize, 1) << @intCast(level)));
            var block = Block.init(level, origin, spacing);
            const block_idx = self.blocks.items.len;
            block.block_index = block_idx;
            const key = self.blockKeyFromOrigin(origin, level);
            if (self.block_index.contains(key)) return error.BlockAlreadyExists;

            try self.blocks.append(self.allocator, block);
            try self.field_slots.append(self.allocator, std.math.maxInt(usize));
            try self.block_index.put(key, block_idx);

            return block_idx;
        }

        pub fn insertBlockWithField(
            self: *Self,
            origin: [Nd]usize,
            level: u8,
            field_arena: *ArenaType,
        ) !usize {
            const block_idx = try self.insertBlock(origin, level);
            const field_slot = field_arena.allocSlot() orelse return error.FieldArenaFull;
            self.field_slots.items[block_idx] = field_slot;
            return block_idx;
        }

        pub fn assignFieldSlot(self: *Self, block_idx: usize, field_slot: usize) void {
            std.debug.assert(block_idx < self.field_slots.items.len);
            self.field_slots.items[block_idx] = field_slot;
        }

        pub fn getFieldSlot(self: *const Self, block_idx: usize) usize {
            std.debug.assert(block_idx < self.field_slots.items.len);
            return self.field_slots.items[block_idx];
        }

        pub fn hasFieldSlot(self: *const Self, block_idx: usize) bool {
            return self.getFieldSlot(block_idx) != std.math.maxInt(usize);
        }

        /// Mark a block invalid and remove it from the Morton index map.
        pub fn invalidateBlock(self: *Self, block_idx: usize) void {
            const block = &self.blocks.items[block_idx];
            if (block.block_index == std.math.maxInt(usize)) return;
            const key = self.blockKeyFromOrigin(block.origin, block.level);
            _ = self.block_index.remove(key);
            self.blocks.items[block_idx].block_index = std.math.maxInt(usize);
        }

        // =====================================================================
        // Block Indexing (Morton)
        // =====================================================================

        fn blockCoords(self: *const Self, origin: [Nd]usize, level: u8) [Nd]usize {
            _ = self;
            const scale = @as(usize, 1) << @intCast(level);
            var coords: [Nd]usize = undefined;
            inline for (0..Nd) |d| {
                coords[d] = (origin[d] / block_size) * scale;
            }
            return coords;
        }

        pub fn getMortonIndex(self: *const Self, origin: [Nd]usize, level: u8) u64 {
            const effective_bits = self.bits_per_dim + level;
            const coords = self.blockCoords(origin, level);
            return morton_mod.encode(Nd, coords, effective_bits);
        }

        pub fn blockKeyFromOrigin(self: *const Self, origin: [Nd]usize, level: u8) BlockKey {
            return .{ .level = level, .morton = self.getMortonIndex(origin, level) };
        }

        pub fn blockKeyForBlock(self: *const Self, block: *const Block) BlockKey {
            return self.blockKeyFromOrigin(block.origin, block.level);
        }

        pub fn findBlockByKey(self: *const Self, key: BlockKey) ?usize {
            return self.block_index.get(key);
        }

        /// Get the extent (in level-0 coordinates) that a block at `level` covers.
        /// At level 0: extent = block_size. At level 1: extent = block_size/2, etc.
        /// Returns 1 for extremely deep levels (>= 64) to avoid shift overflow,
        /// though such levels should not occur in practice (max_level is configured at init).
        fn getBlockExtent(self: *const Self, level: u8) usize {
            _ = self;
            // Levels >= 64 indicate a bad configuration.
            std.debug.assert(level < 64);
            const shift = @as(u6, @intCast(@min(level, 63)));
            return block_size >> shift;
        }

        // =====================================================================
        // Coordinate Conversion Utilities
        // =====================================================================

        pub fn getPhysicalOrigin(self: *const Self, block: *const Block) [Nd]f64 {
            const spacing = self.base_spacing / @as(f64, @floatFromInt(@as(usize, 1) << @intCast(block.level)));
            var physical: [Nd]f64 = undefined;
            inline for (0..Nd) |d| {
                physical[d] = @as(f64, @floatFromInt(block.origin[d])) * spacing;
            }
            return physical;
        }

        pub fn getBlockPhysicalExtent(self: *const Self, level: u8) f64 {
            const spacing = self.base_spacing / @as(f64, @floatFromInt(@as(usize, 1) << @intCast(level)));
            return @as(f64, @floatFromInt(block_size)) * spacing;
        }

        pub fn physicalToBlockOrigin(self: *const Self, physical: [Nd]f64, level: u8) [Nd]usize {
            const spacing = self.base_spacing / @as(f64, @floatFromInt(@as(usize, 1) << @intCast(level)));
            var origin: [Nd]usize = undefined;
            inline for (0..Nd) |d| {
                const site_coord = @as(usize, @intFromFloat(@floor(physical[d] / spacing)));
                origin[d] = (site_coord / block_size) * block_size;
            }
            return origin;
        }

        // =====================================================================
        // Neighbor Finding
        // =====================================================================

        pub fn neighborInfo(self: *const Self, block_idx: usize, face: usize) NeighborInfo {
            const Topology = Frontend.Topology;
            const block = &self.blocks.items[block_idx];
            const dim = face / 2;
            const is_positive = (face % 2) == 0;

            const physical_origin = self.getPhysicalOrigin(block);
            const block_extent = self.getBlockPhysicalExtent(block.level);

            // Compute the raw neighbor coordinate
            var neighbor_physical: [Nd]f64 = physical_origin;
            if (is_positive) {
                neighbor_physical[dim] += block_extent;
            } else {
                neighbor_physical[dim] -= Topology.neighborEpsilon(block.spacing, dim);
            }

            // Apply topology wrapping (handles periodic/open boundaries)
            const wrapped_coord = Topology.wrapCoordinateRuntime(neighbor_physical[dim], dim);
            if (wrapped_coord == null) {
                // Open boundary - no neighbor exists
                return NeighborInfo.none;
            }
            neighbor_physical[dim] = wrapped_coord.?;

            // Search same level first
            if (self.findBlockAtLevel(neighbor_physical, block.level)) |idx| {
                return .{ .block_idx = idx, .level_diff = 0 };
            }

            // Search coarser level
            if (block.level > 0) {
                if (self.findBlockAtLevel(neighbor_physical, block.level - 1)) |idx| {
                    return .{ .block_idx = idx, .level_diff = -1 };
                }
            }

            // Check for finer level (Push Model: return none)
            if (block.level < self.max_level) {
                if (self.findBlockAtLevel(neighbor_physical, block.level + 1)) |_| {
                    return NeighborInfo.none;
                }
            }

            return NeighborInfo.none;
        }

        fn findBlockAtLevel(self: *const Self, physical: [Nd]f64, level: u8) ?usize {
            const origin = self.physicalToBlockOrigin(physical, level);
            return self.findBlockByOrigin(origin, level);
        }

        pub fn findBlockByOrigin(self: *const Self, origin: [Nd]usize, level: u8) ?usize {
            return self.block_index.get(self.blockKeyFromOrigin(origin, level));
        }

        /// Collect indices of finer neighbors across a face.
        /// Returns the count of fine neighbors (0..max_fine_neighbors).
        pub fn collectFineNeighbors(
            self: *const Self,
            block_idx: usize,
            face: usize,
            out: *[max_fine_neighbors]usize,
        ) usize {
            const Topology = Frontend.Topology;
            const block = &self.blocks.items[block_idx];
            if (block.block_index == std.math.maxInt(usize)) return 0;
            if (block.level >= self.max_level) return 0;

            const dim = face / 2;
            const is_positive = (face % 2) == 0;
            const physical_origin = self.getPhysicalOrigin(block);
            const block_extent = self.getBlockPhysicalExtent(block.level);

            var neighbor_physical = physical_origin;
            if (is_positive) {
                neighbor_physical[dim] += block_extent;
            } else {
                neighbor_physical[dim] -= Topology.neighborEpsilon(block.spacing, dim);
            }

            const wrapped = Topology.wrapCoordinateRuntime(neighbor_physical[dim], dim) orelse return 0;
            neighbor_physical[dim] = wrapped;

            const fine_origin_base = self.physicalToBlockOrigin(neighbor_physical, block.level + 1);
            const combos = max_fine_neighbors;
            var count: usize = 0;

            for (0..combos) |combo| {
                var origin = fine_origin_base;
                var combo_idx: usize = 0;
                inline for (0..Nd) |d| {
                    if (d != dim) {
                        const half = (combo >> @intCast(combo_idx)) & 1;
                        origin[d] += half * block_size;
                        combo_idx += 1;
                    }
                }

                if (self.findBlockByOrigin(origin, block.level + 1)) |neighbor_idx| {
                    out[count] = neighbor_idx;
                    count += 1;
                }
            }

            return count;
        }

        // =====================================================================
        // Refinement Operations
        // =====================================================================

        pub fn refineBlock(self: *Self, block_idx: usize) !void {
            const block = &self.blocks.items[block_idx];
            const parent_level = block.level;
            const child_level = parent_level + 1;

            if (child_level > self.max_level) {
                return error.MaxRefinementExceeded;
            }

            const parent_origin = block.origin;
            self.invalidateBlock(block_idx);

            const child_step = block_size;

            for (0..num_children) |child| {
                var child_origin: [Nd]usize = undefined;
                inline for (0..Nd) |d| {
                    const half = (child >> @intCast(d)) & 1;
                    child_origin[d] = parent_origin[d] * 2 + half * child_step;
                }

                _ = try self.insertBlock(child_origin, child_level);
            }
        }

        // =====================================================================
        // Iteration
        // =====================================================================

        pub fn blockIterator(self: *const Self) BlockIterator {
            return .{ .tree = self, .index = 0 };
        }

        pub const BlockIterator = struct {
            tree: *const Self,
            index: usize,

            pub fn next(self: *BlockIterator) ?*Block {
                while (self.index < self.tree.blocks.items.len) {
                    const block = &self.tree.blocks.items[self.index];
                    self.index += 1;
                    if (block.block_index != std.math.maxInt(usize)) {
                        return @constCast(block);
                    }
                }
                return null;
            }
        };

        // =====================================================================
        // Threaded Kernel Execution
        // =====================================================================

        /// Apply a kernel to the mesh using an ApplyContext.
        ///
        /// ApplyContext bundles all state references needed for kernel execution:
        /// - tree: Reference to this AMR tree
        /// - field_in/field_out: Input/output field arenas
        /// - field_ghosts: Ghost buffer for field exchange
        /// - edges/edge_ghosts: Edge-centered storage for staggered kernels
        /// - flux_reg: Optional flux register for conservation
        ///
        /// The kernel must implement:
        /// `fn execute(block_idx: usize, block: *const Block, ctx: *ApplyContext) void`
        ///
        /// Ghost exchange is handled automatically based on context configuration.
        pub fn apply(
            self: *Self,
            kernel: anytype,
            ctx: *ApplyContext,
        ) !void {
            const KernelType = @TypeOf(kernel);
            const KernelDecl = switch (@typeInfo(KernelType)) {
                .pointer => std.meta.Child(KernelType),
                else => KernelType,
            };

            comptime {
                if (!@hasDecl(KernelDecl, "execute")) {
                    @compileError("Kernel must implement execute(block_idx, block, ctx) for applyWithContext.");
                }
            }

            // Ensure ghost buffers are sized for this tree
            if (ctx.field_ghosts) |g| {
                try g.ensureForTree(self);
            }

            // Begin field ghost exchange if we have field data
            const needs_field_ghosts = ctx.field_in != null and ctx.field_ghosts != null;
            var field_arena: ?*const ArenaType = null;
            var field_ghost_slice: []?*ghost_buffer_mod.GhostBuffer(Frontend).GhostFaces = &[_]?*ghost_buffer_mod.GhostBuffer(Frontend).GhostFaces{};

            if (needs_field_ghosts) {
                field_arena = ctx.field_in.?;
                field_ghost_slice = ctx.field_ghosts.?.slice(self.blocks.items.len);
            }

            var dist_state: ?FieldExchange.ExchangeState = null;
            if (needs_field_ghosts) {
                const exchange_ctx = FieldPolicy.Context{
                    .tree = self,
                    .arena = field_arena.?,
                    .ghosts = field_ghost_slice,
                };
                dist_state = try self.field_exchange.begin(exchange_ctx, self.shard_context);
            }

            // Define execution context for task scheduling
            const ExecCtx = struct {
                k: @TypeOf(kernel),
                idx: usize,
                blk: *const Block,
                apply_ctx: *ApplyContext,

                fn run(ctx_ptr: *anyopaque) void {
                    const c: *@This() = @ptrCast(@alignCast(ctx_ptr));
                    c.k.execute(c.idx, c.blk, c.apply_ctx);
                }
            };

            // Execute interior computation (can overlap with ghost exchange)
            self.interior_group.reset();

            for (self.blocks.items, 0..) |*block, idx| {
                if (block.block_index == std.math.maxInt(usize)) continue;

                // Skip blocks without field slots if we have field data
                if (ctx.field_in != null) {
                    const slot = self.getFieldSlot(idx);
                    if (slot == std.math.maxInt(usize)) continue;
                }

                const exec_ctx = self.interior_group.arena.allocatorHandle().create(ExecCtx) catch {
                    kernel.execute(idx, block, ctx);
                    continue;
                };
                exec_ctx.* = .{
                    .k = kernel,
                    .idx = idx,
                    .blk = block,
                    .apply_ctx = ctx,
                };
                self.interior_group.submit(&self.pool, .{ .run = ExecCtx.run, .ctx = exec_ctx });
            }

            // Finish field ghost exchange
            if (dist_state) |*state| {
                const exchange_ctx = FieldPolicy.Context{
                    .tree = self,
                    .arena = field_arena.?,
                    .ghosts = field_ghost_slice,
                };
                try self.field_exchange.finish(exchange_ctx, state);
            }

            // Mark ghosts as clean
            ctx.field_ghosts_dirty = false;

            // Wait for all computation to complete
            self.interior_group.wait();
        }

        pub fn blockCount(self: *const Self) usize {
            var count: usize = 0;
            for (self.blocks.items) |*block| {
                if (block.block_index != std.math.maxInt(usize)) {
                    count += 1;
                }
            }
            return count;
        }

        pub fn threadCount(self: *const Self) usize {
            return self.pool.workerCount();
        }

        /// Fill ghost layers for field data.
        /// Uses the field exchange mechanism to populate ghost buffers.
        pub fn fillGhostLayers(
            self: *Self,
            arena: *const ArenaType,
            ghosts: []?*ghost_buffer_mod.GhostBuffer(Frontend).GhostFaces,
        ) !void {
            const exchange_ctx = FieldPolicy.Context{
                .tree = self,
                .arena = arena,
                .ghosts = ghosts,
            };
            var dist_state = try self.field_exchange.begin(exchange_ctx, self.shard_context);
            try self.field_exchange.finish(exchange_ctx, &dist_state);
        }

        /// State for split ghost exchange (begin/finish pattern).
        pub const GhostExchangeState = struct {
            ctx: FieldPolicy.Context,
            dist_state: FieldExchange.ExchangeState,
        };

        /// Begin a split ghost exchange. Returns state to pass to finishGhostExchange.
        pub fn beginGhostExchange(
            self: *Self,
            arena: *const ArenaType,
            ghosts: []?*ghost_buffer_mod.GhostBuffer(Frontend).GhostFaces,
        ) !GhostExchangeState {
            const exchange_ctx = FieldPolicy.Context{
                .tree = self,
                .arena = arena,
                .ghosts = ghosts,
            };
            const dist_state = try self.field_exchange.begin(exchange_ctx, self.shard_context);
            return .{ .ctx = exchange_ctx, .dist_state = dist_state };
        }

        /// Finish a split ghost exchange started with beginGhostExchange.
        pub fn finishGhostExchange(self: *Self, state: *GhostExchangeState) !void {
            try self.field_exchange.finish(state.ctx, &state.dist_state);
        }

        pub fn getBlock(self: *const Self, idx: usize) ?*const Block {
            if (idx >= self.blocks.items.len) return null;
            const block = &self.blocks.items[idx];
            if (block.block_index == std.math.maxInt(usize)) return null;
            return block;
        }

        // =====================================================================
        // Reordering (Morton Sort + Compaction)
        // =====================================================================

        /// Reorder blocks by Morton index for cache locality.
        ///
        /// This operation:
        /// 1. Removes invalid (deleted) blocks from storage
        /// 2. Sorts remaining blocks by Morton index
        /// 3. Rebuilds the Morton key index and permutes field_slots
        ///
        /// Returns the permutation map `perm` where `perm[old_index] = new_index`.
        /// If a block was removed, `perm[old_index] = std.math.maxInt(usize)`.
        /// **The caller owns the returned slice and must free it.**
        ///
        /// **IMPORTANT**: This invalidates all external pointers to blocks and field data.
        /// After calling reorder(), any `*Block` pointers or field slice references
        /// obtained before the call are invalid.
        ///
        /// Call this after mesh adaptation to restore cache locality.
        ///
        /// Returns error on allocation failure.
        pub fn reorder(self: *Self) ![]usize {
            const invalid = std.math.maxInt(usize);
            const old_len = self.blocks.items.len;

            var active_count: usize = 0;
            for (self.blocks.items) |*block| {
                if (block.block_index != invalid) {
                    active_count += 1;
                }
            }

            const perm = try self.allocator.alloc(usize, old_len);
            @memset(perm, invalid);

            if (active_count == 0) {
                self.blocks.clearRetainingCapacity();
                self.field_slots.clearRetainingCapacity();
                self.block_index.clearRetainingCapacity();
                return perm;
            }

            const SortPair = struct {
                morton: u64,
                level: u8,
                old_idx: usize,

                fn lessThan(_: void, a: @This(), b: @This()) bool {
                    if (a.morton != b.morton) return a.morton < b.morton;
                    if (a.level != b.level) return a.level < b.level;
                    return a.old_idx < b.old_idx;
                }
            };

            var sort_pairs = try self.allocator.alloc(SortPair, active_count);
            defer self.allocator.free(sort_pairs);

            var pair_idx: usize = 0;
            for (self.blocks.items, 0..) |*block, old_idx| {
                if (block.block_index != invalid) {
                    sort_pairs[pair_idx] = .{
                        .morton = self.getMortonIndex(block.origin, block.level),
                        .level = block.level,
                        .old_idx = old_idx,
                    };
                    pair_idx += 1;
                }
            }

            std.mem.sort(SortPair, sort_pairs, {}, SortPair.lessThan);

            for (sort_pairs, 0..) |pair, new_idx| {
                perm[pair.old_idx] = new_idx;
            }

            var new_blocks = try self.allocator.alloc(Block, active_count);
            defer self.allocator.free(new_blocks);

            for (sort_pairs, 0..) |pair, new_idx| {
                new_blocks[new_idx] = self.blocks.items[pair.old_idx];
            }

            var new_field_slots = try self.allocator.alloc(usize, active_count);
            defer self.allocator.free(new_field_slots);

            for (sort_pairs, 0..) |pair, new_idx| {
                if (pair.old_idx < self.field_slots.items.len) {
                    new_field_slots[new_idx] = self.field_slots.items[pair.old_idx];
                } else {
                    new_field_slots[new_idx] = invalid;
                }
            }

            self.blocks.clearRetainingCapacity();
            try self.blocks.appendSlice(self.allocator, new_blocks);

            self.field_slots.clearRetainingCapacity();
            try self.field_slots.appendSlice(self.allocator, new_field_slots);

            self.block_index.clearRetainingCapacity();
            for (self.blocks.items, 0..) |*block, idx| {
                block.block_index = idx;
                const key = self.blockKeyFromOrigin(block.origin, block.level);
                try self.block_index.put(key, idx);
            }

            return perm;
        }
    };
}

// Re-export FieldArena for convenience
pub const FieldArena = field_arena_mod.FieldArena;

// =============================================================================
// Tests
// =============================================================================

const topology = @import("topology.zig");

const TestFrontend2D = struct {
    pub const Nd: usize = 2;
    pub const block_size: usize = 4;
    pub const FieldType = f64;
    pub const Topology = topology.PeriodicTopology(2, .{ 64.0, 64.0 });
};

test "reorder - blocks sorted by Morton index" {
    const Tree = AMRTree(TestFrontend2D);
    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert blocks in non-Morton order (reversed spatial order)
    // Block size is 4, so origins are multiples of 4
    _ = try tree.insertBlock(.{ 12, 12 }, 0); // Far corner
    _ = try tree.insertBlock(.{ 0, 0 }, 0); // Origin
    _ = try tree.insertBlock(.{ 8, 4 }, 0); // Middle-ish
    _ = try tree.insertBlock(.{ 4, 8 }, 0); // Another point

    // Before reorder, blocks are in insertion order
    try std.testing.expectEqual(@as(usize, 4), tree.blocks.items.len);

    // Reorder by Morton index
    try tree.reorder();

    // After reorder, Morton indices should be monotonically increasing
    var prev_morton: u64 = 0;
    for (tree.blocks.items, 0..) |*block, idx| {
        const morton = tree.getMortonIndex(block.origin, block.level);
        if (idx > 0) {
            try std.testing.expect(morton >= prev_morton);
        }
        prev_morton = morton;

        // Self-reference should match position
        try std.testing.expectEqual(idx, block.block_index);
    }
}

test "reorder - neighbor queries remain valid" {
    const Tree = AMRTree(TestFrontend2D);
    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Create a 2x2 grid of blocks
    // Origins at (0,0), (4,0), (0,4), (4,4)
    _ = try tree.insertBlock(.{ 4, 4 }, 0);
    _ = try tree.insertBlock(.{ 0, 4 }, 0);
    _ = try tree.insertBlock(.{ 4, 0 }, 0);
    _ = try tree.insertBlock(.{ 0, 0 }, 0);

    // Capture neighbor relationships before reorder
    // Using physical positions to verify after reorder
    const getBlockByOrigin = struct {
        fn call(t: *Tree, origin: [2]usize) ?usize {
            for (t.blocks.items, 0..) |*blk, idx| {
                if (blk.block_index != std.math.maxInt(usize) and
                    blk.origin[0] == origin[0] and blk.origin[1] == origin[1])
                {
                    return idx;
                }
            }
            return null;
        }
    }.call;

    // Reorder
    try tree.reorder();

    // After reorder, find block at (0,0) and verify its +x neighbor is at (4,0)
    const origin_block = getBlockByOrigin(&tree, .{ 0, 0 }).?;

    // Face 0 is +x direction
    const neighbor_x = tree.neighborInfo(origin_block, 0);
    try std.testing.expect(neighbor_x.exists());
    try std.testing.expectEqual(@as(i8, 0), neighbor_x.level_diff);
    const neighbor = &tree.blocks.items[neighbor_x.block_idx];
    try std.testing.expectEqual(@as(usize, 4), neighbor.origin[0]);
    try std.testing.expectEqual(@as(usize, 0), neighbor.origin[1]);

    // Face 2 is +y direction
    const neighbor_y = tree.neighborInfo(origin_block, 2);
    try std.testing.expect(neighbor_y.exists());
    try std.testing.expectEqual(@as(i8, 0), neighbor_y.level_diff);
    const neighbor_y_block = &tree.blocks.items[neighbor_y.block_idx];
    try std.testing.expectEqual(@as(usize, 0), neighbor_y_block.origin[0]);
    try std.testing.expectEqual(@as(usize, 4), neighbor_y_block.origin[1]);
}

test "reorder - field_slots correctly permuted" {
    const Tree = AMRTree(TestFrontend2D);
    const Arena = FieldArena(TestFrontend2D);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 10);
    defer arena.deinit();

    // Insert blocks in non-Morton order, each with a field slot
    const idx0 = try tree.insertBlock(.{ 8, 8 }, 0);
    const slot0 = arena.allocSlot().?;
    tree.assignFieldSlot(idx0, slot0);
    arena.getSlot(slot0)[0] = 88.0; // Mark with unique value

    const idx1 = try tree.insertBlock(.{ 0, 0 }, 0);
    const slot1 = arena.allocSlot().?;
    tree.assignFieldSlot(idx1, slot1);
    arena.getSlot(slot1)[0] = 0.0;

    const idx2 = try tree.insertBlock(.{ 4, 4 }, 0);
    const slot2 = arena.allocSlot().?;
    tree.assignFieldSlot(idx2, slot2);
    arena.getSlot(slot2)[0] = 44.0;

    // Reorder
    try tree.reorder();

    // Verify: each block's field_slot still points to its original data
    for (tree.blocks.items, 0..) |*block, idx| {
        const slot = tree.getFieldSlot(idx);
        const expected_value: f64 = @floatFromInt(block.origin[0] + block.origin[1]);
        // Our marking was: origin (0,0) -> 0.0, (4,4) -> 44.0, (8,8) -> 88.0
        // So we should be able to find the correct value via the slot
        const actual_value = arena.getSlotConst(slot)[0];

        // Derive expected from origin
        const sum = block.origin[0] + block.origin[1];
        if (sum == 0) {
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), actual_value, 1e-10);
        } else if (sum == 8) {
            try std.testing.expectApproxEqAbs(@as(f64, 44.0), actual_value, 1e-10);
        } else if (sum == 16) {
            try std.testing.expectApproxEqAbs(@as(f64, 88.0), actual_value, 1e-10);
        }

        _ = expected_value;
    }
}

test "reorder - invalid blocks filtered out (compaction)" {
    const Tree = AMRTree(TestFrontend2D);
    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    // Insert blocks
    _ = try tree.insertBlock(.{ 0, 0 }, 0);
    const idx1 = try tree.insertBlock(.{ 4, 0 }, 0);
    _ = try tree.insertBlock(.{ 8, 0 }, 0);

    // Mark one as invalid (simulating refinement)
    tree.blocks.items[idx1].block_index = std.math.maxInt(usize);

    // Before reorder: 3 blocks, 1 invalid
    try std.testing.expectEqual(@as(usize, 3), tree.blocks.items.len);
    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());

    // Reorder (should compact out invalid block)
    try tree.reorder();

    // After reorder: only 2 blocks remain
    try std.testing.expectEqual(@as(usize, 2), tree.blocks.items.len);
    try std.testing.expectEqual(@as(usize, 2), tree.blockCount());

    // All remaining blocks should be valid
    for (tree.blocks.items, 0..) |*block, idx| {
        try std.testing.expectEqual(idx, block.block_index);
        try std.testing.expect(block.block_index != std.math.maxInt(usize));
    }
}

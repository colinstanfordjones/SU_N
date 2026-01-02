//! Generic MPI ghost exchange with persistent buffers.

const std = @import("std");
const platform = @import("platform");
const shard_mod = @import("shard.zig");

pub fn ExchangeSpec(comptime Context: type, comptime Payload: type) type {
    return struct {
        payload_len: usize,
        payload_alignment: std.mem.Alignment,
        pack_same_level: *const fn (Context, usize, usize, []Payload, std.mem.Allocator) void,
        pack_coarse_to_fine: *const fn (Context, usize, usize, []Payload, std.mem.Allocator) void,
        pack_fine_to_coarse: *const fn (Context, usize, usize, []Payload, std.mem.Allocator) void,
        unpack_same_level: *const fn (Context, usize, usize, []const Payload) void,
        unpack_coarse_to_fine: *const fn (Context, usize, usize, []const Payload) void,
        unpack_fine_to_coarse: *const fn (Context, usize, usize, []const Payload) void,
        should_exchange: ?*const fn (Context, usize) bool = null,
        cost_fn: ?*const fn (Context, usize) f64 = null,
    };
}

pub fn NeighborKeyInfo(comptime Tree: type) type {
    return struct {
        key: Tree.BlockKey,
        level_diff: i8,
    };
}

pub fn DistExchange(comptime Tree: type, comptime Context: type, comptime Payload: type) type {
    const Nd = Tree.dimensions;
    const ShardContext = shard_mod.ShardContext(Tree);
    const BlockKey = Tree.BlockKey;
    const Spec = ExchangeSpec(Context, Payload);

    const MessageKind = enum(u8) {
        same_level,
        coarse_to_fine,
        fine_to_coarse,
    };

    const Header = extern struct {
        morton: u64,
        level: u8,
        face: u8,
        kind: u8,
        _pad: [5]u8 = .{0} ** 5,
    };

    const Buffer = struct {
        data: []u8,
        alignment: std.mem.Alignment,

        fn ensureCapacity(self: *@This(), allocator: std.mem.Allocator, size: usize, alignment: std.mem.Alignment) !void {
            if (self.data.len >= size and self.alignment == alignment) return;
            if (self.data.len > 0) allocator.rawFree(self.data, self.alignment, @returnAddress());
            if (size == 0) {
                self.data = &[_]u8{};
                self.alignment = alignment;
                return;
            }
            const ptr = allocator.rawAlloc(size, alignment, @returnAddress()) orelse return error.OutOfMemory;
            self.data = ptr[0..size];
            self.alignment = alignment;
        }

        fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            if (self.data.len > 0) allocator.rawFree(self.data, self.alignment, @returnAddress());
        }
    };

    return struct {
        const Self = @This();
        
        pub const ExchangeState = struct {
            recv_reqs: []platform.mpi.Request,
            send_reqs: []platform.mpi.Request,
        };

        allocator: std.mem.Allocator,
        spec: Spec,
        recv_buffers: std.ArrayListUnmanaged(Buffer),
        send_buffers: std.ArrayListUnmanaged(Buffer),
        recv_reqs: std.ArrayListUnmanaged(platform.mpi.Request),
        send_reqs: std.ArrayListUnmanaged(platform.mpi.Request),

        pub fn init(allocator: std.mem.Allocator, spec: Spec) Self {
            return Self{
                .allocator = allocator,
                .spec = spec,
                .recv_buffers = .{},
                .send_buffers = .{},
                .recv_reqs = .{},
                .send_reqs = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.recv_buffers.items) |*b| b.deinit(self.allocator);
            self.recv_buffers.deinit(self.allocator);
            
            for (self.send_buffers.items) |*b| b.deinit(self.allocator);
            self.send_buffers.deinit(self.allocator);
            
            self.recv_reqs.deinit(self.allocator);
            self.send_reqs.deinit(self.allocator);
        }

        pub fn begin(
            self: *Self,
            ctx: Context,
            shard_opt: ?*ShardContext,
        ) !ExchangeState {
            const tree = ctxTree(ctx);
            var comm: platform.mpi.Comm = undefined;
            var use_mpi = false;

            if (shard_opt) |shard| {
                try shard.refreshLocalBlocks(tree);
                if (platform.mpi.enabled) {
                    comm = shard.comm;
                    use_mpi = true;
                }
            }

            const tag: platform.mpi.types.Tag = 0;
            
            var temp_arena = std.heap.ArenaAllocator.init(self.allocator);
            defer temp_arena.deinit();
            const scratch = if (shard_opt) |s| s.scratchAllocator() else temp_arena.allocator();
            defer {
                if (shard_opt) |s| s.resetScratch();
            }

            if (self.spec.payload_len == 0) return error.InvalidPayloadLength;
            const required_alignment = std.mem.Alignment.of(Payload);
            if (self.spec.payload_alignment.compare(.lt, required_alignment)) {
                return error.PayloadAlignmentTooSmall;
            }

            const payload_bytes = self.spec.payload_len * @sizeOf(Payload);
            const payload_offset = std.mem.alignForward(usize, @sizeOf(Header), @alignOf(Payload));
            const message_size = payload_offset + payload_bytes;
            const payload_alignment = self.spec.payload_alignment;

            // 1. Count messages
            var send_count: usize = 0;
            var recv_count: usize = 0;
            
            if (use_mpi) {
                const shard = shard_opt.?;
                for (shard.localBlockIndices()) |block_idx| {
                    if (!self.shouldExchange(ctx, tree, block_idx)) continue;
                    try self.countBlock(tree, block_idx, shard, &send_count, &recv_count);
                }
            }

            // 2. Resize buffers
            try self.ensureBuffers(send_count, recv_count, message_size, payload_alignment);
            
            self.recv_reqs.clearRetainingCapacity();
            self.send_reqs.clearRetainingCapacity();

            // 3. Post Recvs
            if (use_mpi) {
                const comm_val = comm;
                for (0..recv_count) |i| {
                    const req = try platform.mpi.irecvAny(comm_val, self.recv_buffers.items[i].data, tag);
                    try self.recv_reqs.append(self.allocator, req);
                }
            }

            // 4. Pack and Send (and process Local)
            if (shard_opt) |shard| {
                for (shard.localBlockIndices()) |block_idx| {
                    if (!self.shouldExchange(ctx, tree, block_idx)) continue;
                    try self.processBlock(ctx, tree, block_idx, shard, use_mpi, payload_offset, payload_bytes, scratch);
                }
            } else {
                for (0..tree.blocks.items.len) |block_idx| {
                    const block = &tree.blocks.items[block_idx];
                    if (block.block_index == std.math.maxInt(usize)) continue;
                    if (!self.shouldExchange(ctx, tree, block_idx)) continue;
                    try self.processBlock(ctx, tree, block_idx, null, false, payload_offset, payload_bytes, scratch);
                }
            }

            return ExchangeState{
                .recv_reqs = self.recv_reqs.items,
                .send_reqs = self.send_reqs.items,
            };
        }

        fn countBlock(
            self: *Self,
            tree: *const Tree,
            block_idx: usize,
            shard: *ShardContext,
            send_count: *usize,
            recv_count: *usize,
        ) !void {
            _ = self;
            const block = &tree.blocks.items[block_idx];
            
            inline for (0..(2 * Nd)) |face| {
                if (findNeighborKey(tree, block, face, &shard.owners)) |neighbor| {
                    if (shard.ownerForKey(neighbor.key)) |neighbor_rank| {
                        if (neighbor_rank != shard.rank) {
                            send_count.* += 1;
                            recv_count.* += 1;
                        }
                    }
                }

                if (block.level < tree.max_level) {
                    var fine_neighbors: [maxFineNeighbors(Nd)]NeighborKeyInfo(Tree) = undefined;
                    const fine_count = collectFineNeighbors(tree, block, face, &shard.owners, &fine_neighbors);
                    for (fine_neighbors[0..fine_count]) |fine_neighbor| {
                        if (shard.ownerForKey(fine_neighbor.key)) |fine_rank| {
                            if (fine_rank != shard.rank) {
                                send_count.* += 1;
                                recv_count.* += 1;
                            }
                        }
                    }
                }
            }
        }

        fn processBlock(
            self: *Self,
            ctx: Context,
            tree: *const Tree,
            block_idx: usize,
            shard_opt: ?*ShardContext,
            use_mpi: bool,
            payload_offset: usize,
            payload_bytes: usize,
            scratch: std.mem.Allocator,
        ) !void {
            const block = &tree.blocks.items[block_idx];
            var send_idx = self.send_reqs.items.len;
            
            inline for (0..(2 * Nd)) |face| {
                var neighbor_key_opt: ?NeighborKeyInfo(Tree) = null;
                
                if (shard_opt) |shard| {
                    if (findNeighborKey(tree, block, face, &shard.owners)) |neighbor| {
                        neighbor_key_opt = .{ .key = neighbor.key, .level_diff = neighbor.level_diff };
                    }
                } else {
                    const neighbor_info = tree.neighborInfo(block_idx, face);
                    if (neighbor_info.exists()) {
                        neighbor_key_opt = .{ .key = BlockKey{ .morton = 0, .level = 0 }, .level_diff = neighbor_info.level_diff };
                    }
                }

                if (neighbor_key_opt) |neighbor| {
                    var is_remote = false;
                    var neighbor_rank: i32 = 0;
                    
                    if (shard_opt) |shard| {
                        if (shard.ownerForKey(neighbor.key)) |r| {
                            neighbor_rank = r;
                            is_remote = (r != shard.rank);
                        }
                    }

                    if (is_remote and use_mpi) {
                        const buf = self.send_buffers.items[send_idx].data;
                        const recv_face = face ^ 1;
                        const payload_slice = payloadSlice(Payload, buf, payload_offset, payload_bytes);

                        if (neighbor.level_diff == 0) {
                            packHeader(buf, payload_offset, payload_bytes, neighbor.key, recv_face, MessageKind.same_level);
                            self.spec.pack_same_level(ctx, block_idx, face, payload_slice, scratch);
                        } else if (neighbor.level_diff == -1) {
                            packHeader(buf, payload_offset, payload_bytes, neighbor.key, recv_face, MessageKind.fine_to_coarse);
                            self.spec.pack_fine_to_coarse(ctx, block_idx, face, payload_slice, scratch);
                        }
                        
                        const req = try platform.mpi.isend(shard_opt.?.comm, buf, neighbor_rank, 0);
                        try self.send_reqs.append(self.allocator, req);
                        send_idx += 1;
                    } else if (!is_remote) {
                        var neighbor_idx: ?usize = null;
                        if (shard_opt) |_| {
                            neighbor_idx = tree.findBlockByKey(neighbor.key);
                        } else {
                            const n_info = tree.neighborInfo(block_idx, face);
                            if (n_info.exists()) neighbor_idx = n_info.block_idx;
                        }

                        if (neighbor_idx) |idx| {
                            const buf = try scratch.alloc(Payload, self.spec.payload_len);
                            const recv_face = face ^ 1;
                            
                            if (neighbor.level_diff == 0) {
                                self.spec.pack_same_level(ctx, block_idx, face, buf, scratch);
                                self.spec.unpack_same_level(ctx, idx, recv_face, buf);
                            } else if (neighbor.level_diff == -1) {
                                self.spec.pack_fine_to_coarse(ctx, block_idx, face, buf, scratch);
                                self.spec.unpack_fine_to_coarse(ctx, idx, recv_face, buf);
                            }
                        }
                    }
                }

                if (block.level < tree.max_level) {
                    if (shard_opt) |shard| {
                        var fine_neighbors: [maxFineNeighbors(Nd)]NeighborKeyInfo(Tree) = undefined;
                        const fine_count = collectFineNeighbors(tree, block, face, &shard.owners, &fine_neighbors);
                        for (fine_neighbors[0..fine_count]) |fine_neighbor| {
                            var is_remote = false;
                            var fine_rank: i32 = 0;
                            if (shard.ownerForKey(fine_neighbor.key)) |r| {
                                fine_rank = r;
                                is_remote = (r != shard.rank);
                            }
                            
                            if (is_remote and use_mpi) {
                                const buf = self.send_buffers.items[send_idx].data;
                                const recv_face = face ^ 1;
                                const payload_slice = payloadSlice(Payload, buf, payload_offset, payload_bytes);
                                
                                packHeader(buf, payload_offset, payload_bytes, fine_neighbor.key, recv_face, MessageKind.coarse_to_fine);
                                self.spec.pack_coarse_to_fine(ctx, block_idx, face, payload_slice, scratch);
                                
                                const req = try platform.mpi.isend(shard.comm, buf, fine_rank, 0);
                                try self.send_reqs.append(self.allocator, req);
                                send_idx += 1;
                            } else if (!is_remote) {
                                if (tree.findBlockByKey(fine_neighbor.key)) |fine_idx| {
                                    const buf = try scratch.alloc(Payload, self.spec.payload_len);
                                    const recv_face = face ^ 1;
                                    self.spec.pack_coarse_to_fine(ctx, block_idx, face, buf, scratch);
                                    self.spec.unpack_coarse_to_fine(ctx, fine_idx, recv_face, buf);
                                }
                            }
                        }
                    } else {
                        var local_fine: [Tree.max_fine_neighbors]usize = undefined;
                        const count = tree.collectFineNeighbors(block_idx, face, &local_fine);
                        for (local_fine[0..count]) |fine_idx| {
                            const buf = try scratch.alloc(Payload, self.spec.payload_len);
                            const recv_face = face ^ 1;
                            self.spec.pack_coarse_to_fine(ctx, block_idx, face, buf, scratch);
                            self.spec.unpack_coarse_to_fine(ctx, fine_idx, recv_face, buf);
                        }
                    }
                }
            }
        }

        pub fn finish(
            self: *Self,
            ctx: Context,
            state: *ExchangeState,
        ) !void {
            if (!platform.mpi.enabled and state.recv_reqs.len > 0) return error.MpiDisabled;

            if (state.recv_reqs.len > 0) {
                try platform.mpi.waitAll(state.recv_reqs);
            }

            const payload_bytes = self.spec.payload_len * @sizeOf(Payload);
            const payload_offset = std.mem.alignForward(usize, @sizeOf(Header), @alignOf(Payload));
            const tree = ctxTree(ctx);

            // Process all received buffers
            for (self.recv_buffers.items[0..state.recv_reqs.len]) |buf| {
                var header: Header = undefined;
                std.mem.copyForwards(u8, std.mem.asBytes(&header), buf.data[0..@sizeOf(Header)]);

                const key = BlockKey{
                    .morton = header.morton,
                    .level = header.level,
                };

                const block_idx = tree.findBlockByKey(key) orelse continue;
                const payload_slice = payloadSlice(Payload, buf.data, payload_offset, payload_bytes);
                const kind: MessageKind = @enumFromInt(header.kind);

                switch (kind) {
                    .same_level => self.spec.unpack_same_level(ctx, block_idx, header.face, payload_slice),
                    .coarse_to_fine => self.spec.unpack_coarse_to_fine(ctx, block_idx, header.face, payload_slice),
                    .fine_to_coarse => self.spec.unpack_fine_to_coarse(ctx, block_idx, header.face, payload_slice),
                }
            }

            if (state.send_reqs.len > 0) {
                try platform.mpi.waitAll(state.send_reqs);
            }
        }

        fn ctxTree(ctx: Context) *const Tree {
            if (@TypeOf(ctx) == *Tree or @TypeOf(ctx) == *const Tree) {
                return ctx;
            }
            if (!@hasField(Context, "tree")) {
                @compileError("Context must be *Tree or provide a .tree field");
            }
            if (@TypeOf(ctx.tree) == *Tree or @TypeOf(ctx.tree) == *const Tree) {
                return ctx.tree;
            }
            @compileError("Context.tree must be *Tree or provide a .tree field");
        }
        
        fn ensureBuffers(self: *Self, send_count: usize, recv_count: usize, size: usize, alignment: std.mem.Alignment) !void {
            while (self.send_buffers.items.len < send_count) {
                try self.send_buffers.append(self.allocator, .{ .data = &[_]u8{}, .alignment = std.mem.Alignment.fromByteUnits(1) });
            }
            while (self.recv_buffers.items.len < recv_count) {
                try self.recv_buffers.append(self.allocator, .{ .data = &[_]u8{}, .alignment = std.mem.Alignment.fromByteUnits(1) });
            }
            
            for (0..send_count) |i| {
                try self.send_buffers.items[i].ensureCapacity(self.allocator, size, alignment);
            }
            for (0..recv_count) |i| {
                try self.recv_buffers.items[i].ensureCapacity(self.allocator, size, alignment);
            }
        }

        fn shouldExchange(self: *Self, ctx: Context, tree: *const Tree, block_idx: usize) bool {
            if (self.spec.should_exchange) |predicate| return predicate(ctx, block_idx);
            return tree.getFieldSlot(block_idx) != std.math.maxInt(usize);
        }
    };
}

// =============================================================================
// Neighbor Discovery Helpers
// =============================================================================
//
// These helpers are intentionally local to this module for isolation.
// They provide neighbor-finding logic specific to distributed exchange.

fn maxFineNeighbors(comptime Nd_: usize) usize {
    return @as(usize, 1) << @intCast(Nd_ - 1);
}

fn findNeighborKey(
    tree: anytype,
    block: anytype,
    face: usize,
    owners: anytype,
) ?struct { key: @TypeOf(tree.*).BlockKey, level_diff: i8 } {
    const neighbor_physical = neighborPhysical(tree, block.origin, block.level, face) orelse return null;

    const same_origin = tree.physicalToBlockOrigin(neighbor_physical, block.level);
    const same_key = tree.blockKeyFromOrigin(same_origin, block.level);
    if (owners.contains(same_key)) {
        return .{ .key = same_key, .level_diff = 0 };
    }

    if (block.level > 0) {
        const coarse_origin = tree.physicalToBlockOrigin(neighbor_physical, block.level - 1);
        const coarse_key = tree.blockKeyFromOrigin(coarse_origin, block.level - 1);
        if (owners.contains(coarse_key)) {
            return .{ .key = coarse_key, .level_diff = -1 };
        }
    }

    return null;
}

fn collectFineNeighbors(
    tree: anytype,
    block: anytype,
    face: usize,
    owners: anytype,
    out: anytype,
) usize {
    const Nd = @TypeOf(tree.*).dimensions;
    const Block = @TypeOf(tree.*).BlockType;
    if (block.level + 1 > tree.max_level) return 0;

    const neighbor_physical = neighborPhysical(tree, block.origin, block.level, face) orelse return 0;
    const fine_origin_base = tree.physicalToBlockOrigin(neighbor_physical, block.level + 1);

    const face_dim = face / 2;

    var count: usize = 0;
    const combos = maxFineNeighbors(Nd);
    for (0..combos) |combo| {
        var origin = fine_origin_base;
        var combo_idx: usize = 0;
        inline for (0..Nd) |d| {
            if (d != face_dim) {
                const half = (combo >> @intCast(combo_idx)) & 1;
                origin[d] += half * Block.size;
                combo_idx += 1;
            }
        }

        const fine_key = tree.blockKeyFromOrigin(origin, block.level + 1);
        if (owners.contains(fine_key)) {
            out[count] = .{ .key = fine_key, .level_diff = 1 };
            count += 1;
        }
    }

    return count;
}

fn neighborPhysical(
    tree: anytype,
    origin: anytype,
    level: u8,
    face: usize,
) ?@TypeOf(tree.getPhysicalOrigin(undefined)) {
    const Nd = @TypeOf(tree.*).dimensions;
    const Topology = @TypeOf(tree.*).FrontendType.Topology;
    const dim = face / 2;
    const is_positive = (face % 2) == 0;

    const spacing = tree.base_spacing / @as(f64, @floatFromInt(@as(usize, 1) << @intCast(level)));
    var physical_origin: [Nd]f64 = undefined;
    inline for (0..Nd) |d| {
        physical_origin[d] = @as(f64, @floatFromInt(origin[d])) * spacing;
    }

    const block_extent = tree.getBlockPhysicalExtent(level);
    var neighbor_physical = physical_origin;
    if (is_positive) {
        neighbor_physical[dim] += block_extent;
    } else {
        neighbor_physical[dim] -= Topology.neighborEpsilon(spacing, dim);
    }

    const wrapped = Topology.wrapCoordinateRuntime(neighbor_physical[dim], dim) orelse return null;
    neighbor_physical[dim] = wrapped;
    return neighbor_physical;
}

fn packHeader(
    buffer: []u8,
    payload_offset: usize,
    payload_bytes: usize,
    key: anytype,
    face: usize,
    kind: anytype,
) void {
    std.debug.assert(buffer.len >= payload_offset + payload_bytes);
    const Header = extern struct {
        morton: u64,
        level: u8,
        face: u8,
        kind: u8,
        _pad: [5]u8 = .{0} ** 5,
    };
    var header = Header{
        .morton = key.morton,
        .level = key.level,
        .face = @intCast(face),
        .kind = @intFromEnum(kind),
    };
    std.mem.copyForwards(u8, buffer[0..@sizeOf(Header)], std.mem.asBytes(&header));
}

fn payloadSlice(
    comptime Field: type,
    buffer: []u8,
    payload_offset: usize,
    payload_bytes: usize,
) []Field {
    const payload = buffer[payload_offset .. payload_offset + payload_bytes];
    const payload_ptr: [*]Field = @ptrCast(@alignCast(payload.ptr));
    return payload_ptr[0 .. payload.len / @sizeOf(Field)];
}

//! MPI repartitioning for AMR trees (entropy-weighted Morton contiguous).

const std = @import("std");
const platform = @import("platform");
const morton_mod = @import("morton.zig");

pub const RepartitionOptions = struct {
    min_weight: f64 = 1e-12,
    compact: bool = true,
    defragment: bool = false,
    max_inflight_bytes: usize = 0,
    max_inflight_messages: usize = 0,
};

pub const AdaptiveOptions = struct {
    weight_imbalance_threshold: f64 = 0.1,
    block_imbalance_threshold: f64 = 0.1,
};

pub const BalanceMetrics = struct {
    total_weight: f64,
    max_weight: f64,
    avg_weight: f64,
    weight_imbalance: f64,
    total_blocks: usize,
    max_blocks: usize,
    avg_blocks: f64,
    block_imbalance: f64,
};

const Entry = struct {
    key: morton_mod.BlockKey,
    weight: f64,
    old_owner: i32,

    pub fn lessThan(_: void, a: Entry, b: Entry) bool {
        if (a.key.morton != b.key.morton) return a.key.morton < b.key.morton;
        if (a.key.level != b.key.level) return a.key.level < b.key.level;
        return a.old_owner < b.old_owner;
    }
};

const WeightWire = extern struct {
    morton: u64,
    level: u8,
    _pad: [7]u8 = .{0} ** 7,
    weight: f64,
};

fn BlockHeader(comptime Nd: usize) type {
    return extern struct {
        origin: [Nd]u64,
        level: u8,
        flags: u8,
        _pad: [6]u8 = .{0} ** 6,
    };
}

const flag_has_field: u8 = 1;

pub fn FieldPolicy(comptime Tree: type) type {
    return struct {
        pub const Context = struct {
            tree: *Tree,
            arena: *Tree.FieldArenaType,
        };

        pub fn extraBytes(_: Context) usize {
            return 0;
        }

        pub fn extraAlignment(_: Context) u29 {
            return 1;
        }

        pub fn packExtra(_: Context, _: usize, _: []u8) void {}

        pub fn unpackExtra(_: Context, _: usize, _: []const u8) void {}

        pub fn insertBlock(ctx: Context, origin: [Tree.dimensions]usize, level: u8, has_field: bool) !usize {
            if (has_field) {
                return ctx.tree.insertBlockWithField(origin, level, ctx.arena);
            }
            return ctx.tree.insertBlock(origin, level);
        }

        pub fn compact(ctx: Context, options: RepartitionOptions) !void {
            if (!options.compact) return;
            const perm = try ctx.tree.reorder();
            ctx.tree.allocator.free(perm);
            if (options.defragment) {
                try ctx.arena.defragmentWithOrder(ctx.tree.field_slots.items, ctx.tree.blockCount());
            }
        }
    };
}

pub fn balanceMetricsEntropyWeighted(
    comptime Tree: type,
    tree: *Tree,
    arena: *Tree.FieldArenaType,
    shard: *Tree.ShardContext,
    min_weight: f64,
) !BalanceMetrics {
    if (!platform.mpi.enabled) return error.MpiDisabled;

    const comm = shard.comm;
    const size = shard.size;
    if (size <= 0) return error.InvalidMpiSize;

    const scratch = shard.scratchAllocator();
    defer shard.resetScratch();

    const local_block_count = countValidBlocks(Tree, tree);
    const local_energy = computeLocalEnergy(Tree, tree, arena);
    const global_energy = try platform.mpi.allreduceSum(comm, local_energy);

    const local_weight = computeLocalWeightSum(Tree, tree, arena, global_energy, min_weight);
    const total_weight = try platform.mpi.allreduceSum(comm, local_weight);

    const weights = try scratch.alloc(f64, @intCast(size));
    try platform.mpi.allgatherBytes(
        comm,
        std.mem.asBytes(&local_weight),
        std.mem.sliceAsBytes(weights),
    );

    var max_weight = weights[0];
    for (weights[1..]) |w| {
        if (w > max_weight) max_weight = w;
    }

    const counts_i32 = try scratch.alloc(i32, @intCast(size));
    const local_count_i32 = try castCount(local_block_count);
    try platform.mpi.allgatherI32(comm, local_count_i32, counts_i32);

    var total_blocks: usize = 0;
    var max_blocks: usize = 0;
    for (counts_i32) |count_i32| {
        const count = @as(usize, @intCast(count_i32));
        total_blocks += count;
        if (count > max_blocks) max_blocks = count;
    }

    const size_f64 = @as(f64, @floatFromInt(@as(usize, @intCast(size))));
    const avg_weight = if (size_f64 > 0) total_weight / size_f64 else 0.0;
    const weight_imbalance = if (avg_weight > 0.0) (max_weight - avg_weight) / avg_weight else 0.0;

    const avg_blocks = if (size_f64 > 0) @as(f64, @floatFromInt(total_blocks)) / size_f64 else 0.0;
    const block_imbalance = if (avg_blocks > 0.0)
        (@as(f64, @floatFromInt(max_blocks)) - avg_blocks) / avg_blocks
    else
        0.0;

    return BalanceMetrics{
        .total_weight = total_weight,
        .max_weight = max_weight,
        .avg_weight = avg_weight,
        .weight_imbalance = weight_imbalance,
        .total_blocks = total_blocks,
        .max_blocks = max_blocks,
        .avg_blocks = avg_blocks,
        .block_imbalance = block_imbalance,
    };
}

pub fn repartitionEntropyWeighted(
    comptime Tree: type,
    tree: *Tree,
    arena: *Tree.FieldArenaType,
    shard: *Tree.ShardContext,
    options: RepartitionOptions,
) !void {
    const Policy = FieldPolicy(Tree);
    const ctx = Policy.Context{ .tree = tree, .arena = arena };
    try repartitionEntropyWeightedWithPolicy(Tree, Policy, ctx, shard, options);
}

pub fn repartitionAdaptiveEntropyWeighted(
    comptime Tree: type,
    tree: *Tree,
    arena: *Tree.FieldArenaType,
    shard: *Tree.ShardContext,
    options: RepartitionOptions,
    adaptive: AdaptiveOptions,
) !bool {
    const metrics = try balanceMetricsEntropyWeighted(Tree, tree, arena, shard, options.min_weight);
    if (metrics.weight_imbalance <= adaptive.weight_imbalance_threshold and
        metrics.block_imbalance <= adaptive.block_imbalance_threshold)
    {
        return false;
    }

    try repartitionEntropyWeighted(Tree, tree, arena, shard, options);
    return true;
}

pub fn repartitionAdaptiveEntropyWeightedWithPolicy(
    comptime Tree: type,
    comptime Policy: type,
    ctx: Policy.Context,
    shard: *Tree.ShardContext,
    options: RepartitionOptions,
    adaptive: AdaptiveOptions,
) !bool {
    const metrics = try balanceMetricsEntropyWeighted(Tree, ctx.tree, ctx.arena, shard, options.min_weight);
    if (metrics.weight_imbalance <= adaptive.weight_imbalance_threshold and
        metrics.block_imbalance <= adaptive.block_imbalance_threshold)
    {
        return false;
    }

    try repartitionEntropyWeightedWithPolicy(Tree, Policy, ctx, shard, options);
    return true;
}

pub fn repartitionEntropyWeightedWithPolicy(
    comptime Tree: type,
    comptime Policy: type,
    ctx: Policy.Context,
    shard: *Tree.ShardContext,
    options: RepartitionOptions,
) !void {
    if (!platform.mpi.enabled) return error.MpiDisabled;

    const comm = shard.comm;
    const rank = shard.rank;
    const size = shard.size;
    if (size <= 0) return error.InvalidMpiSize;

    const tree = ctx.tree;
    const arena = ctx.arena;
    const scratch = shard.scratchAllocator();
    defer shard.resetScratch();

    const invalid = std.math.maxInt(usize);

    const local_block_count = countValidBlocks(Tree, tree);
    const local_count_i32 = try castCount(local_block_count);

    const counts_i32 = try scratch.alloc(i32, @intCast(size));
    try platform.mpi.allgatherI32(comm, local_count_i32, counts_i32);

    var displs: []usize = try scratch.alloc(usize, @intCast(size));
    var counts: []usize = try scratch.alloc(usize, @intCast(size));
    var total_blocks: usize = 0;
    for (counts_i32, 0..) |count_i32, idx| {
        const count = @as(usize, @intCast(count_i32));
        counts[idx] = count;
        displs[idx] = total_blocks;
        total_blocks += count;
    }

    const local_wire = try scratch.alloc(WeightWire, local_block_count);
    const local_energy = computeLocalEnergy(Tree, tree, arena);
    const global_energy = try platform.mpi.allreduceSum(comm, local_energy);

    fillLocalWeights(Tree, tree, arena, local_wire, global_energy, options.min_weight);

    const recv_wire = try scratch.alloc(WeightWire, total_blocks);

    var recv_counts_bytes = try scratch.alloc(i32, @intCast(size));
    var recv_displs_bytes = try scratch.alloc(i32, @intCast(size));
    const wire_size = @sizeOf(WeightWire);

    for (counts, 0..) |count, idx| {
        const bytes = count * wire_size;
        recv_counts_bytes[idx] = try castCount(bytes);
        recv_displs_bytes[idx] = try castCount(displs[idx] * wire_size);
    }

    try platform.mpi.allgatherVBytes(
        comm,
        std.mem.sliceAsBytes(local_wire),
        std.mem.sliceAsBytes(recv_wire),
        recv_counts_bytes,
        recv_displs_bytes,
    );

    const entries = try scratch.alloc(Entry, total_blocks);
    var offset: usize = 0;
    for (0..@intCast(size)) |rank_idx| {
        const count = counts[rank_idx];
        for (0..count) |j| {
            const idx = offset + j;
            const wire = recv_wire[idx];
            entries[idx] = .{
                .key = .{ .morton = wire.morton, .level = wire.level },
                .weight = wire.weight,
                .old_owner = @intCast(rank_idx),
            };
        }
        offset += count;
    }

    std.mem.sort(Entry, entries, {}, Entry.lessThan);

    var new_owners = std.AutoHashMap(Tree.BlockKey, i32).init(shard.allocator);
    errdefer new_owners.deinit();
    try new_owners.ensureTotalCapacity(@intCast(entries.len));

    const recv_count = try assignOwnersWeighted(entries, @intCast(size), rank, &new_owners);
    const outgoing = try collectOutgoingBlocks(Tree, tree, &new_owners, rank, scratch);
    const send_blocks = outgoing.blocks;
    const send_count = send_blocks.len;
    const slots_to_free = outgoing.slots_to_free;

    const field_bytes = Tree.BlockType.volume * @sizeOf(Tree.FieldType);
    const field_align: usize = @alignOf(Tree.FieldType);
    const extra_bytes = Policy.extraBytes(ctx);
    const extra_align: usize = blk: {
        const alignment = Policy.extraAlignment(ctx);
        break :blk if (alignment == 0) 1 else alignment;
    };

    const Header = BlockHeader(Tree.dimensions);
    const header_size = @sizeOf(Header);
    const field_offset = std.mem.alignForward(usize, header_size, field_align);
    const extra_offset = std.mem.alignForward(usize, field_offset + field_bytes, extra_align);
    const message_size = extra_offset + extra_bytes;
    const alignment: usize = @max(@as(usize, @alignOf(Header)), @max(field_align, extra_align));
    const stride = std.mem.alignForward(usize, message_size, alignment);

    const max_inflight = try maxInflightCount(options, stride, send_count, recv_count);

    const free_after = arena.free_count + slots_to_free;
    if (free_after < recv_count) {
        const required = arena.max_blocks + (recv_count - free_after);
        try arena.ensureCapacity(required);
    }

    var batch_arena = std.heap.ArenaAllocator.init(tree.allocator);
    defer batch_arena.deinit();

    const tag: platform.mpi.types.Tag = 7;
    var send_cursor: usize = 0;
    var recv_remaining: usize = recv_count;

    while (send_cursor < send_count or recv_remaining > 0) {
        const batch_send = if (send_cursor < send_count)
            @min(send_count - send_cursor, max_inflight)
        else
            0;
        const batch_recv = if (recv_remaining > 0)
            @min(recv_remaining, max_inflight)
        else
            0;

        if (batch_send == 0 and batch_recv == 0) break;

        _ = batch_arena.reset(.retain_capacity);
        const batch_alloc = batch_arena.allocator();

        var send_storage: []u8 = @constCast(&[_]u8{});
        var recv_storage: []u8 = @constCast(&[_]u8{});
        var recv_reqs: []platform.mpi.Request = @constCast(&[_]platform.mpi.Request{});
        var send_reqs: []platform.mpi.Request = @constCast(&[_]platform.mpi.Request{});
        if (batch_send > 0) send_storage = try batch_alloc.alloc(u8, batch_send * stride);
        if (batch_recv > 0) recv_storage = try batch_alloc.alloc(u8, batch_recv * stride);
        if (batch_recv > 0) recv_reqs = try batch_alloc.alloc(platform.mpi.Request, batch_recv);
        if (batch_send > 0) send_reqs = try batch_alloc.alloc(platform.mpi.Request, batch_send);

        for (recv_reqs, 0..) |*req, i| {
            const buf = recv_storage[i * stride .. i * stride + message_size];
            req.* = try platform.mpi.irecvAny(comm, buf, tag);
        }

        for (0..batch_send) |i| {
            const block_idx = send_blocks[send_cursor + i];
            const block = &tree.blocks.items[block_idx];
            const key = tree.blockKeyForBlock(block);
            const new_owner = new_owners.get(key) orelse unreachable;
            std.debug.assert(new_owner != rank);

            const buf = send_storage[i * stride .. i * stride + message_size];
            const has_field = tree.hasFieldSlot(block_idx);
            const header = Header{
                .origin = try castOrigin(Tree.dimensions, block.origin),
                .level = block.level,
                .flags = if (has_field) flag_has_field else 0,
            };
            std.mem.copyForwards(u8, buf[0..header_size], std.mem.asBytes(&header));

            if (has_field) {
                const slot = tree.getFieldSlot(block_idx);
                const field_slice = arena.getSlotConst(slot);
                std.mem.copyForwards(u8, buf[field_offset .. field_offset + field_bytes], std.mem.sliceAsBytes(field_slice));
            } else {
                @memset(buf[field_offset .. field_offset + field_bytes], 0);
            }

            if (extra_bytes > 0) {
                Policy.packExtra(ctx, block_idx, buf[extra_offset .. extra_offset + extra_bytes]);
            }

            send_reqs[i] = try platform.mpi.isend(comm, buf, new_owner, tag);

            if (has_field) {
                const slot = tree.getFieldSlot(block_idx);
                arena.freeSlot(slot);
                tree.assignFieldSlot(block_idx, invalid);
            }
            tree.invalidateBlock(block_idx);
        }

        if (recv_reqs.len > 0) {
            try platform.mpi.waitAll(recv_reqs);
        }

        for (recv_reqs, 0..) |_, i| {
            const buf = recv_storage[i * stride .. i * stride + message_size];
            var header: Header = undefined;
            std.mem.copyForwards(u8, std.mem.asBytes(&header), buf[0..header_size]);

            const origin = try unpackOrigin(Tree.dimensions, header.origin);
            const has_field = (header.flags & flag_has_field) != 0;

            const block_idx = try Policy.insertBlock(ctx, origin, header.level, has_field);
            if (has_field) {
                const slot = tree.getFieldSlot(block_idx);
                const field_slice = arena.getSlot(slot);
                std.mem.copyForwards(u8, std.mem.sliceAsBytes(field_slice), buf[field_offset .. field_offset + field_bytes]);
            }

            if (extra_bytes > 0) {
                Policy.unpackExtra(ctx, block_idx, buf[extra_offset .. extra_offset + extra_bytes]);
            }
        }

        if (send_reqs.len > 0) {
            try platform.mpi.waitAll(send_reqs);
        }

        send_cursor += batch_send;
        recv_remaining -= batch_recv;
    }

    shard.owners.deinit();
    shard.owners = new_owners;

    try Policy.compact(ctx, options);
    try shard.refreshLocal(tree);
}

fn countValidBlocks(comptime Tree: type, tree: *const Tree) usize {
    const invalid = std.math.maxInt(usize);
    var count: usize = 0;
    for (tree.blocks.items) |*block| {
        if (block.block_index != invalid) count += 1;
    }
    return count;
}

fn computeLocalEnergy(comptime Tree: type, tree: *const Tree, arena: *const Tree.FieldArenaType) f64 {
    const invalid = std.math.maxInt(usize);
    const Frontend = Tree.FrontendType;
    var total: f64 = 0.0;
    for (tree.blocks.items, 0..) |*block, block_idx| {
        if (block.block_index == invalid) continue;
        if (!tree.hasFieldSlot(block_idx)) continue;
        const slot = tree.getFieldSlot(block_idx);
        const field_slice = arena.getSlotConst(slot);
        for (field_slice) |value| {
            total += fieldNormSq(Frontend, value);
        }
    }
    return total;
}

fn computeLocalWeightSum(
    comptime Tree: type,
    tree: *const Tree,
    arena: *const Tree.FieldArenaType,
    global_energy: f64,
    min_weight: f64,
) f64 {
    const invalid = std.math.maxInt(usize);
    const Frontend = Tree.FrontendType;
    var total: f64 = 0.0;
    for (tree.blocks.items, 0..) |*block, block_idx| {
        if (block.block_index == invalid) continue;
        const field_slice_opt = if (tree.hasFieldSlot(block_idx))
            arena.getSlotConst(tree.getFieldSlot(block_idx))
        else
            null;
        total += computeBlockWeight(Frontend, field_slice_opt, global_energy, min_weight);
    }
    return total;
}

fn computeBlockWeight(
    comptime Frontend: type,
    field_slice: ?[]const Frontend.FieldType,
    global_energy: f64,
    min_weight: f64,
) f64 {
    if (field_slice == null) return min_weight;
    if (global_energy <= 0.0) {
        return if (min_weight > 1.0) min_weight else 1.0;
    }

    var entropy: f64 = 0.0;
    for (field_slice.?) |value| {
        const w = fieldNormSq(Frontend, value);
        if (w <= 0.0) continue;
        const p = w / global_energy;
        if (p > 0.0) {
            entropy -= p * std.math.log(f64, std.math.e, p);
        }
    }

    var weight: f64 = if (std.math.isFinite(entropy) and entropy > 0.0) entropy else min_weight;
    if (weight < min_weight) weight = min_weight;
    return weight;
}

fn fillLocalWeights(
    comptime Tree: type,
    tree: *const Tree,
    arena: *const Tree.FieldArenaType,
    local_wire: []WeightWire,
    global_energy: f64,
    min_weight: f64,
) void {
    const invalid = std.math.maxInt(usize);
    const Frontend = Tree.FrontendType;
    var idx: usize = 0;
    for (tree.blocks.items, 0..) |*block, block_idx| {
        if (block.block_index == invalid) continue;
        const key = tree.blockKeyForBlock(block);
        const field_slice_opt = if (tree.hasFieldSlot(block_idx))
            arena.getSlotConst(tree.getFieldSlot(block_idx))
        else
            null;
        const weight = computeBlockWeight(Frontend, field_slice_opt, global_energy, min_weight);

        local_wire[idx] = .{
            .morton = key.morton,
            .level = key.level,
            .weight = weight,
        };
        idx += 1;
    }
}

fn assignOwnersWeighted(
    entries: []Entry,
    size: usize,
    rank: i32,
    owners: *std.AutoHashMap(morton_mod.BlockKey, i32),
) !usize {
    if (entries.len == 0 or size == 0) return 0;

    var total_weight: f64 = 0.0;
    for (entries) |entry| total_weight += entry.weight;
    const use_uniform = !std.math.isFinite(total_weight) or total_weight <= 0.0;
    if (use_uniform) {
        total_weight = @as(f64, @floatFromInt(entries.len));
    }

    var recv_count: usize = 0;
    var start: usize = 0;
    var remaining_weight = total_weight;

    for (0..size) |rank_idx| {
        const blocks_left = entries.len - start;
        if (blocks_left == 0) break;

        if (rank_idx == size - 1) {
            for (entries[start..]) |entry| {
                try owners.put(entry.key, @intCast(rank_idx));
                if (@as(i32, @intCast(rank_idx)) == rank and entry.old_owner != rank) recv_count += 1;
            }
            break;
        }

        const ranks_left = size - rank_idx;
        const enforce_min = blocks_left >= ranks_left;
        const min_blocks_left = if (enforce_min) ranks_left - 1 else 0;
        const target_weight = remaining_weight / @as(f64, @floatFromInt(ranks_left));

        var count: usize = 0;
        var acc_weight: f64 = 0.0;
        while (start + count < entries.len - min_blocks_left) {
            const w = if (use_uniform) 1.0 else entries[start + count].weight;
            if (count > 0 and acc_weight + w > target_weight) break;
            acc_weight += w;
            count += 1;
        }

        if (count == 0) {
            count = 1;
            acc_weight = if (use_uniform) 1.0 else entries[start].weight;
        }

        for (entries[start .. start + count]) |entry| {
            try owners.put(entry.key, @intCast(rank_idx));
            if (@as(i32, @intCast(rank_idx)) == rank and entry.old_owner != rank) recv_count += 1;
        }

        start += count;
        remaining_weight = @max(remaining_weight - acc_weight, 0.0);
    }

    return recv_count;
}

fn countOutgoingBlocks(
    comptime Tree: type,
    tree: *const Tree,
    owners: *const std.AutoHashMap(morton_mod.BlockKey, i32),
    rank: i32,
) usize {
    const invalid = std.math.maxInt(usize);
    var count: usize = 0;
    for (tree.blocks.items) |*block| {
        if (block.block_index == invalid) continue;
        const key = tree.blockKeyForBlock(block);
        const new_owner = owners.get(key) orelse continue;
        if (new_owner != rank) count += 1;
    }
    return count;
}

fn collectOutgoingBlocks(
    comptime Tree: type,
    tree: *const Tree,
    owners: *const std.AutoHashMap(morton_mod.BlockKey, i32),
    rank: i32,
    scratch: std.mem.Allocator,
) !struct { blocks: []usize, slots_to_free: usize } {
    const invalid = std.math.maxInt(usize);
    var send_count: usize = 0;
    var slots_to_free: usize = 0;
    for (tree.blocks.items, 0..) |*block, block_idx| {
        if (block.block_index == invalid) continue;
        const key = tree.blockKeyForBlock(block);
        const new_owner = owners.get(key) orelse continue;
        if (new_owner == rank) continue;
        send_count += 1;
        if (tree.hasFieldSlot(block_idx)) slots_to_free += 1;
    }

    const blocks = try scratch.alloc(usize, send_count);
    var cursor: usize = 0;
    for (tree.blocks.items, 0..) |*block, block_idx| {
        if (block.block_index == invalid) continue;
        const key = tree.blockKeyForBlock(block);
        const new_owner = owners.get(key) orelse continue;
        if (new_owner == rank) continue;
        blocks[cursor] = block_idx;
        cursor += 1;
    }

    return .{
        .blocks = blocks,
        .slots_to_free = slots_to_free,
    };
}

fn maxInflightCount(
    options: RepartitionOptions,
    stride: usize,
    send_count: usize,
    recv_count: usize,
) !usize {
    if (send_count == 0 and recv_count == 0) return 0;

    var max_messages: usize = if (send_count > recv_count) send_count else recv_count;
    if (options.max_inflight_messages > 0) {
        max_messages = @min(max_messages, options.max_inflight_messages);
    }
    if (options.max_inflight_bytes > 0) {
        const from_bytes = options.max_inflight_bytes / stride;
        if (from_bytes == 0) return error.InflightLimitTooSmall;
        max_messages = @min(max_messages, from_bytes);
    }
    if (max_messages == 0) return error.InflightLimitTooSmall;
    return max_messages;
}

fn castCount(count: usize) !i32 {
    if (count > std.math.maxInt(i32)) return error.BlockCountOverflow;
    return @intCast(count);
}

fn castOrigin(comptime Nd: usize, origin: [Nd]usize) ![Nd]u64 {
    var out: [Nd]u64 = undefined;
    inline for (0..Nd) |d| {
        if (origin[d] > std.math.maxInt(u64)) return error.OriginOverflow;
        out[d] = @intCast(origin[d]);
    }
    return out;
}

fn unpackOrigin(comptime Nd: usize, origin: [Nd]u64) ![Nd]usize {
    var out: [Nd]usize = undefined;
    inline for (0..Nd) |d| {
        if (origin[d] > std.math.maxInt(usize)) return error.OriginOverflow;
        out[d] = @intCast(origin[d]);
    }
    return out;
}

fn fieldNormSq(comptime Frontend: type, value: Frontend.FieldType) f64 {
    if (@hasDecl(Frontend, "fieldNormSq")) {
        return Frontend.fieldNormSq(value);
    }
    return normSqValue(@TypeOf(value), value);
}

fn normSqValue(comptime T: type, value: T) f64 {
    return switch (@typeInfo(T)) {
        .float => blk: {
            const v: f64 = @floatCast(value);
            break :blk v * v;
        },
        .comptime_float => blk: {
            const v: f64 = value;
            break :blk v * v;
        },
        .int => blk: {
            const v: f64 = @floatFromInt(value);
            break :blk v * v;
        },
        .comptime_int => blk: {
            const v: f64 = @floatFromInt(value);
            break :blk v * v;
        },
        .array => blk: {
            var sum: f64 = 0.0;
            for (value) |elem| {
                sum += normSqValue(@TypeOf(elem), elem);
            }
            break :blk sum;
        },
        .vector => blk: {
            var sum: f64 = 0.0;
            for (value) |elem| {
                sum += normSqValue(@TypeOf(elem), elem);
            }
            break :blk sum;
        },
        .@"struct" => blk: {
            if (@hasField(T, "re") and @hasField(T, "im")) {
                const re = normSqValue(@TypeOf(value.re), value.re);
                const im = normSqValue(@TypeOf(value.im), value.im);
                break :blk re + im;
            }
            @compileError("FieldType requires Frontend.fieldNormSq or numeric/array-compatible type");
        },
        else => @compileError("FieldType requires Frontend.fieldNormSq or numeric/array-compatible type"),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "repartition assigns contiguous owners by weight" {
    const allocator = std.testing.allocator;
    const keys = [_]morton_mod.BlockKey{
        .{ .morton = 0, .level = 0 },
        .{ .morton = 1, .level = 0 },
        .{ .morton = 2, .level = 0 },
        .{ .morton = 3, .level = 0 },
    };

    var entries = try allocator.alloc(Entry, keys.len);
    defer allocator.free(entries);

    for (keys, 0..) |key, idx| {
        entries[idx] = .{ .key = key, .weight = 1.0, .old_owner = 0 };
    }

    var owners = std.AutoHashMap(morton_mod.BlockKey, i32).init(allocator);
    defer owners.deinit();

    _ = try assignOwnersWeighted(entries, 2, 0, &owners);

    try std.testing.expectEqual(@as(i32, 0), owners.get(keys[0]).?);
    try std.testing.expectEqual(@as(i32, 0), owners.get(keys[1]).?);
    try std.testing.expectEqual(@as(i32, 1), owners.get(keys[2]).?);
    try std.testing.expectEqual(@as(i32, 1), owners.get(keys[3]).?);
}

test "repartition falls back to count when weights are zero" {
    const allocator = std.testing.allocator;
    const keys = [_]morton_mod.BlockKey{
        .{ .morton = 0, .level = 0 },
        .{ .morton = 1, .level = 0 },
        .{ .morton = 2, .level = 0 },
        .{ .morton = 3, .level = 0 },
    };

    var entries = try allocator.alloc(Entry, keys.len);
    defer allocator.free(entries);

    for (keys, 0..) |key, idx| {
        entries[idx] = .{ .key = key, .weight = 0.0, .old_owner = 0 };
    }

    var owners = std.AutoHashMap(morton_mod.BlockKey, i32).init(allocator);
    defer owners.deinit();

    _ = try assignOwnersWeighted(entries, 2, 0, &owners);

    try std.testing.expectEqual(@as(i32, 0), owners.get(keys[0]).?);
    try std.testing.expectEqual(@as(i32, 0), owners.get(keys[1]).?);
    try std.testing.expectEqual(@as(i32, 1), owners.get(keys[2]).?);
    try std.testing.expectEqual(@as(i32, 1), owners.get(keys[3]).?);
}

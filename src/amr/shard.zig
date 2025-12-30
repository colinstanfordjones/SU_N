//! MPI sharding utilities for AMR.
//!
//! This module builds global block ownership metadata by exchanging block keys
//! across ranks and applying a partitioning strategy (default: Morton-contiguous).

const std = @import("std");
const platform = @import("platform");
const morton_mod = @import("morton.zig");

pub const ShardStrategy = enum {
    morton_contiguous,
    manual,
};

pub const BlockKey = morton_mod.BlockKey;

pub const BlockOwner = struct {
    key: BlockKey,
    rank: i32,
};

pub fn ShardContext(comptime Tree: type) type {
    const Key = Tree.BlockKey;

    return struct {
        const Self = @This();
        const Wire = BlockKeyWire;

        allocator: std.mem.Allocator,
        comm: platform.mpi.Comm,
        rank: i32,
        size: i32,
        strategy: ShardStrategy,
        owners: std.AutoHashMap(Key, i32),
        local_keys: []Key,
        local_blocks: []usize,
        scratch: platform.TaskArena,

        pub fn initFromTree(
            allocator: std.mem.Allocator,
            tree: *const Tree,
            comm: platform.mpi.Comm,
            strategy: ShardStrategy,
        ) !Self {
            const rank = try platform.mpi.rank(comm);
            const size = try platform.mpi.size(comm);
            if (size <= 0) return error.InvalidMpiSize;

            var scratch = try platform.TaskArena.init(allocator);
            errdefer scratch.deinit();

            const local_keys = try collectLocalKeys(Tree, allocator, tree);
            errdefer allocator.free(local_keys);

            const scratch_alloc = scratch.allocatorHandle();
            const counts = try scratch_alloc.alloc(i32, @intCast(size));

            const local_count_i32 = try castBlockCount(local_keys.len);
            try platform.mpi.allgatherI32(comm, local_count_i32, counts);

            const gather = try gatherKeys(Tree, scratch_alloc, comm, local_keys, counts);

            var owners = std.AutoHashMap(Key, i32).init(allocator);
            errdefer owners.deinit();

            switch (strategy) {
                .manual => {
                    var offset: usize = 0;
                    for (gather.counts, 0..) |count, idx| {
                        const rank_i32: i32 = @intCast(idx);
                        for (0..count) |j| {
                            try owners.put(gather.keys[offset + j], rank_i32);
                        }
                        offset += count;
                    }
                },
                .morton_contiguous => {
                    const total = gather.keys.len;
                    const ranked = try scratch_alloc.alloc(KeyWithMorton, total);

                    for (gather.keys, 0..) |key, idx| {
                        ranked[idx] = .{
                            .key = key,
                            .morton = key.morton,
                        };
                    }
                    std.mem.sort(KeyWithMorton, ranked, {}, KeyWithMorton.lessThan);

                    const size_usize: usize = @intCast(size);
                    const base = total / size_usize;
                    const remainder = total % size_usize;

                    var cursor: usize = 0;
                    for (0..size_usize) |rank_idx| {
                        const count = base + @as(usize, @intFromBool(rank_idx < remainder));
                        for (0..count) |_| {
                            try owners.put(ranked[cursor].key, @intCast(rank_idx));
                            cursor += 1;
                        }
                    }
                },
            }

            scratch.reset();

            const local_blocks = try collectLocalBlocks(Tree, allocator, tree, &owners, rank);
            errdefer allocator.free(local_blocks);

            return Self{
                .allocator = allocator,
                .comm = comm,
                .rank = rank,
                .size = size,
                .strategy = strategy,
                .owners = owners,
                .local_keys = local_keys,
                .local_blocks = local_blocks,
                .scratch = scratch,
            };
        }

        pub fn deinit(self: *Self) void {
            self.owners.deinit();
            self.allocator.free(self.local_keys);
            self.allocator.free(self.local_blocks);
            self.scratch.deinit();
        }

        pub fn ownerForKey(self: *const Self, key: Key) ?i32 {
            return self.owners.get(key);
        }

        pub fn isLocalKey(self: *const Self, key: Key) bool {
            return (self.owners.get(key) orelse -1) == self.rank;
        }

        pub fn isLocalBlock(self: *const Self, block: *const Tree.BlockType) bool {
            return self.isLocalKey(blockKey(Tree, block));
        }

        pub fn localBlockIndices(self: *const Self) []const usize {
            return self.local_blocks;
        }

        pub fn refreshLocalKeys(self: *Self, tree: *const Tree) !void {
            self.allocator.free(self.local_keys);
            self.local_keys = try collectLocalKeys(Tree, self.allocator, tree);
        }

        pub fn refreshLocalBlocks(self: *Self, tree: *const Tree) !void {
            self.allocator.free(self.local_blocks);
            self.local_blocks = try collectLocalBlocks(Tree, self.allocator, tree, &self.owners, self.rank);
        }

        pub fn refreshLocal(self: *Self, tree: *const Tree) !void {
            try self.refreshLocalKeys(tree);
            try self.refreshLocalBlocks(tree);
        }

        pub fn scratchAllocator(self: *Self) std.mem.Allocator {
            return self.scratch.allocatorHandle();
        }

        pub fn resetScratch(self: *Self) void {
            self.scratch.reset();
        }
    };
}

const KeyWithMorton = struct {
    key: BlockKey,
    morton: u64,

    pub fn lessThan(_: void, a: @This(), b: @This()) bool {
        if (a.morton != b.morton) return a.morton < b.morton;
        return a.key.level < b.key.level;
    }
};

const BlockKeyWire = extern struct {
    morton: u64,
    level: u8,
    _pad: [7]u8 = .{0} ** 7,
};

fn castBlockCount(count: usize) !i32 {
    if (count > std.math.maxInt(i32)) return error.BlockCountOverflow;
    return @intCast(count);
}

fn blockKey(comptime Tree: type, tree: *const Tree, block: *const Tree.BlockType) Tree.BlockKey {
    return tree.blockKeyFromOrigin(block.origin, block.level);
}

fn collectLocalKeys(
    comptime Tree: type,
    allocator: std.mem.Allocator,
    tree: *const Tree,
) ![]Tree.BlockKey {
    var count: usize = 0;
    for (tree.blocks.items) |*block| {
        if (block.block_index != std.math.maxInt(usize)) count += 1;
    }

    const keys = try allocator.alloc(Tree.BlockKey, count);
    var idx: usize = 0;
    for (tree.blocks.items) |*block| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        keys[idx] = blockKey(Tree, tree, block);
        idx += 1;
    }
    return keys;
}

fn collectLocalBlocks(
    comptime Tree: type,
    allocator: std.mem.Allocator,
    tree: *const Tree,
    owners: *const std.AutoHashMap(Tree.BlockKey, i32),
    rank: i32,
) ![]usize {
    var count: usize = 0;
    for (tree.blocks.items) |*block| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if ((owners.get(blockKey(Tree, tree, block)) orelse -1) == rank) {
            count += 1;
        }
    }

    const indices = try allocator.alloc(usize, count);
    var idx: usize = 0;
    for (tree.blocks.items, 0..) |*block, block_idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        if ((owners.get(blockKey(Tree, tree, block)) orelse -1) == rank) {
            indices[idx] = block_idx;
            idx += 1;
        }
    }
    return indices;
}

fn gatherKeys(
    comptime Tree: type,
    allocator: std.mem.Allocator,
    comm: platform.mpi.Comm,
    local_keys: []const Tree.BlockKey,
    counts_i32: []const i32,
) !struct {
    keys: []Tree.BlockKey,
    counts: []usize,
    displs: []usize,
} {
    const Wire = BlockKeyWire;
    const wire_size = @sizeOf(Wire);

    const rank_count = counts_i32.len;
    var counts = try allocator.alloc(usize, rank_count);
    errdefer allocator.free(counts);
    var displs = try allocator.alloc(usize, rank_count);
    errdefer allocator.free(displs);

    var total_keys: usize = 0;
    for (counts_i32, 0..) |count_i32, idx| {
        const count = @as(usize, @intCast(count_i32));
        counts[idx] = count;
        displs[idx] = total_keys;
        total_keys += count;
    }

    const send_count_bytes = local_keys.len * wire_size;
    if (send_count_bytes > std.math.maxInt(i32)) return error.ByteCountOverflow;

    var send_wire = try allocator.alloc(Wire, local_keys.len);
    defer allocator.free(send_wire);
    for (local_keys, 0..) |key, idx| {
        send_wire[idx] = packKey(key);
    }

    const recv_wire = try allocator.alloc(Wire, total_keys);
    defer allocator.free(recv_wire);

    var recv_counts_bytes = try allocator.alloc(i32, rank_count);
    defer allocator.free(recv_counts_bytes);
    var recv_displs_bytes = try allocator.alloc(i32, rank_count);
    defer allocator.free(recv_displs_bytes);

    for (counts, 0..) |count, idx| {
        const bytes = count * wire_size;
        if (bytes > std.math.maxInt(i32)) return error.ByteCountOverflow;
        recv_counts_bytes[idx] = @intCast(bytes);
        recv_displs_bytes[idx] = @intCast(displs[idx] * wire_size);
    }

    try platform.mpi.allgatherVBytes(
        comm,
        std.mem.sliceAsBytes(send_wire),
        std.mem.sliceAsBytes(recv_wire),
        recv_counts_bytes,
        recv_displs_bytes,
    );

    var keys = try allocator.alloc(Tree.BlockKey, total_keys);
    for (recv_wire, 0..) |wire, idx| {
        keys[idx] = unpackKey(wire);
    }

    return .{
        .keys = keys,
        .counts = counts,
        .displs = displs,
    };
}

fn packKey(key: BlockKey) BlockKeyWire {
    return .{ .morton = key.morton, .level = key.level };
}

fn unpackKey(wire: BlockKeyWire) BlockKey {
    return .{ .morton = wire.morton, .level = wire.level };
}

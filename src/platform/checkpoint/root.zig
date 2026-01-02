//! Checkpoint/restart serialization for AMR state.
//!
//! Stores AMRTree + FieldArena as a binary snapshot.
//! The payload is a raw byte dump of blocks, field slots, and arena storage.
//! This is host-endian and intended for exact restart on compatible builds.
//!
//! Users (e.g. GaugeField) can append their own data to the stream after calling write().

const std = @import("std");
const mpi = @import("../mpi/root.zig");

pub const magic = [_]u8{ 'S', 'U', 'N', 'C' };
pub const version: u32 = 2; // Bumped version due to format change

const Endian = enum(u8) {
    little = 1,
    big = 2,
};

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedEndian,
    IncompatibleCheckpoint,
    InvalidTreeLayout,
    LengthOverflow,
    CorruptCheckpoint,
};

const Header = struct {
    magic: [4]u8,
    version: u32,
    endian: Endian,
    nd: u8,
    block_size: u32,
    block_volume: u64,
    block_bytes: u32,
    field_bytes: u32,
    usize_bytes: u32,
    blocks_len: u64,
    field_slots_len: u64,
    arena_max_blocks: u64,
    arena_free_count: u64,
    base_spacing_bits: u64,
    bits_per_dim: u8,
    max_level: u8,
    reserved: [16]u8 = [_]u8{0} ** 16,
};

pub const Schedule = struct {
    interval: u64,
    start: u64 = 0,
    enabled: bool = true,

    pub fn shouldCheckpoint(self: Schedule, step: u64) bool {
        if (!self.enabled or self.interval == 0) return false;
        if (step < self.start) return false;
        return ((step - self.start) % self.interval) == 0;
    }
};

pub fn broadcastCheckpointStep(comm: mpi.Comm, root: i32, step: u64) !u64 {
    var value: u64 = step;
    try mpi.bcastBytes(comm, std.mem.asBytes(&value), root);
    return value;
}

pub fn formatRankedPath(
    buf: []u8,
    dir: []const u8,
    prefix: []const u8,
    step: u64,
    rank: i32,
) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/{s}_step{d}_rank{d}.sunc", .{ dir, prefix, step, rank });
}

pub fn formatPath(
    buf: []u8,
    dir: []const u8,
    prefix: []const u8,
    step: u64,
) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/{s}_step{d}.sunc", .{ dir, prefix, step });
}

pub fn TreeCheckpoint(comptime Tree: type) type {
    const Arena = Tree.FieldArenaType;

    return struct {
        const Self = @This();

        pub const State = struct {
            tree: Tree,
            arena: Arena,

            pub fn deinit(self: *State) void {
                self.tree.deinit();
                self.arena.deinit();
            }
        };

        pub fn write(tree: *const Tree, arena: *const Arena, writer: anytype) !void {
            try writeCheckpoint(Tree, tree, arena, writer);
        }

        pub fn writeToFile(tree: *const Tree, arena: *const Arena, dir: std.fs.Dir, path: []const u8) !void {
            var file = try dir.createFile(path, .{ .truncate = true });
            defer file.close();
            try writeCheckpoint(Tree, tree, arena, file.writer());
        }

        pub fn read(allocator: std.mem.Allocator, reader: anytype) !State {
            const header = try readHeader(reader);
            try validateHeader(Tree, header);
            return try readTreeArena(Tree, allocator, header, reader);
        }

        pub fn readFromFile(allocator: std.mem.Allocator, dir: std.fs.Dir, path: []const u8) !State {
            var file = try dir.openFile(path, .{});
            defer file.close();
            return try Self.read(allocator, file.reader());
        }
    };
}

fn writeCheckpoint(
    comptime Tree: type,
    tree: *const Tree,
    arena: *const Tree.FieldArenaType,
    writer: anytype,
) !void {
    const Block = Tree.BlockType;
    const FieldType = Tree.FieldType;
    const nd: usize = Tree.dimensions;
    const block_size: usize = Tree.block_size_const;
    const block_volume: usize = Block.volume;

    if (tree.blocks.items.len != tree.field_slots.items.len) return Error.InvalidTreeLayout;

    const blocks_len = std.math.cast(u64, tree.blocks.items.len) orelse return Error.LengthOverflow;
    const field_slots_len = std.math.cast(u64, tree.field_slots.items.len) orelse return Error.LengthOverflow;
    const arena_max_blocks = std.math.cast(u64, arena.max_blocks) orelse return Error.LengthOverflow;
    const arena_free_count = std.math.cast(u64, arena.free_count) orelse return Error.LengthOverflow;

    const block_size_u32 = std.math.cast(u32, block_size) orelse return Error.LengthOverflow;
    const block_volume_u64 = std.math.cast(u64, block_volume) orelse return Error.LengthOverflow;

    const header = Header{
        .magic = magic,
        .version = version,
        .endian = .little,
        .nd = @intCast(nd),
        .block_size = block_size_u32,
        .block_volume = block_volume_u64,
        .block_bytes = @intCast(@sizeOf(Block)),
        .field_bytes = @intCast(@sizeOf(FieldType)),
        .usize_bytes = @intCast(@sizeOf(usize)),
        .blocks_len = blocks_len,
        .field_slots_len = field_slots_len,
        .arena_max_blocks = arena_max_blocks,
        .arena_free_count = arena_free_count,
        .base_spacing_bits = @bitCast(tree.base_spacing),
        .bits_per_dim = tree.bits_per_dim,
        .max_level = tree.max_level,
    };

    try writeHeader(writer, header);
    try writer.writeAll(std.mem.sliceAsBytes(tree.blocks.items));
    try writer.writeAll(std.mem.sliceAsBytes(tree.field_slots.items));
    try writer.writeAll(std.mem.sliceAsBytes(arena.storage));
    try writer.writeAll(std.mem.sliceAsBytes(arena.free_slots));
}

fn readTreeArena(
    comptime Tree: type,
    allocator: std.mem.Allocator,
    header: Header,
    reader: anytype,
) !TreeCheckpoint(Tree).State {
    const Arena = Tree.FieldArenaType;
    const invalid = std.math.maxInt(usize);

    const blocks_len = try castUsize(header.blocks_len);
    const field_slots_len = try castUsize(header.field_slots_len);
    if (blocks_len != field_slots_len) return Error.InvalidTreeLayout;

    const base_spacing: f64 = @bitCast(header.base_spacing_bits);

    var tree = try Tree.init(allocator, base_spacing, header.bits_per_dim, header.max_level);
    errdefer tree.deinit();

    try tree.blocks.resize(allocator, blocks_len);
    try reader.readNoEof(std.mem.sliceAsBytes(tree.blocks.items));

    try tree.field_slots.resize(allocator, field_slots_len);
    try reader.readNoEof(std.mem.sliceAsBytes(tree.field_slots.items));

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == invalid) continue;
        const key = tree.blockKeyFromOrigin(block.origin, block.level);
        try tree.block_index.put(key, idx);
    }

    const arena_max_blocks = try castUsize(header.arena_max_blocks);
    const arena_free_count = try castUsize(header.arena_free_count);
    if (arena_free_count > arena_max_blocks) return Error.CorruptCheckpoint;

    var arena = try Arena.init(allocator, arena_max_blocks);
    errdefer arena.deinit();

    try reader.readNoEof(std.mem.sliceAsBytes(arena.storage));
    try reader.readNoEof(std.mem.sliceAsBytes(arena.free_slots));
    arena.free_count = arena_free_count;

    return .{
        .tree = tree,
        .arena = arena,
    };
}

fn validateHeader(
    comptime Tree: type,
    header: Header,
) !void {
    const Block = Tree.BlockType;
    const FieldType = Tree.FieldType;
    const nd: usize = Tree.dimensions;
    const block_size: usize = Tree.block_size_const;
    const block_volume: usize = Block.volume;

    if (!std.mem.eql(u8, &header.magic, &magic)) return Error.InvalidMagic;
    if (header.version != version) return Error.UnsupportedVersion;
    if (header.endian != .little) return Error.UnsupportedEndian;

    if (header.nd != @as(u8, @intCast(nd))) return Error.IncompatibleCheckpoint;
    if (header.block_size != @as(u32, @intCast(block_size))) return Error.IncompatibleCheckpoint;
    if (header.block_volume != @as(u64, @intCast(block_volume))) return Error.IncompatibleCheckpoint;
    if (header.block_bytes != @sizeOf(Block)) return Error.IncompatibleCheckpoint;
    if (header.field_bytes != @sizeOf(FieldType)) return Error.IncompatibleCheckpoint;
    if (header.usize_bytes != @sizeOf(usize)) return Error.IncompatibleCheckpoint;
}

fn writeHeader(writer: anytype, header: Header) !void {
    try writer.writeAll(&header.magic);
    try writer.writeInt(u32, header.version, .little);
    try writer.writeByte(@intFromEnum(header.endian));
    try writer.writeByte(header.nd);
    try writer.writeInt(u32, header.block_size, .little);
    try writer.writeInt(u64, header.block_volume, .little);
    try writer.writeInt(u32, header.block_bytes, .little);
    try writer.writeInt(u32, header.field_bytes, .little);
    try writer.writeInt(u32, header.usize_bytes, .little);
    try writer.writeInt(u64, header.blocks_len, .little);
    try writer.writeInt(u64, header.field_slots_len, .little);
    try writer.writeInt(u64, header.arena_max_blocks, .little);
    try writer.writeInt(u64, header.arena_free_count, .little);
    try writer.writeInt(u64, header.base_spacing_bits, .little);
    try writer.writeByte(header.bits_per_dim);
    try writer.writeByte(header.max_level);
    try writer.writeAll(&header.reserved);
}

fn readHeader(reader: anytype) !Header {
    var magic_buf: [4]u8 = undefined;
    try reader.readNoEof(&magic_buf);
    const ver = try reader.readInt(u32, .little);
    const endian_tag = try reader.readByte();
    const nd = try reader.readByte();
    const block_size = try reader.readInt(u32, .little);
    const block_volume = try reader.readInt(u64, .little);
    const block_bytes = try reader.readInt(u32, .little);
    const field_bytes = try reader.readInt(u32, .little);
    const usize_bytes = try reader.readInt(u32, .little);
    const blocks_len = try reader.readInt(u64, .little);
    const field_slots_len = try reader.readInt(u64, .little);
    const arena_max_blocks = try reader.readInt(u64, .little);
    const arena_free_count = try reader.readInt(u64, .little);
    const base_spacing_bits = try reader.readInt(u64, .little);
    const bits_per_dim = try reader.readByte();
    const max_level = try reader.readByte();
    var reserved: [16]u8 = undefined;
    try reader.readNoEof(&reserved);

    const endian = switch (endian_tag) {
        @intFromEnum(Endian.little) => Endian.little,
        @intFromEnum(Endian.big) => Endian.big,
        else => return Error.UnsupportedEndian,
    };

    return Header{
        .magic = magic_buf,
        .version = ver,
        .endian = endian,
        .nd = nd,
        .block_size = block_size,
        .block_volume = block_volume,
        .block_bytes = block_bytes,
        .field_bytes = field_bytes,
        .usize_bytes = usize_bytes,
        .blocks_len = blocks_len,
        .field_slots_len = field_slots_len,
        .arena_max_blocks = arena_max_blocks,
        .arena_free_count = arena_free_count,
        .base_spacing_bits = base_spacing_bits,
        .bits_per_dim = bits_per_dim,
        .max_level = max_level,
        .reserved = reserved,
    };
}

fn castUsize(value: u64) !usize {
    return std.math.cast(usize, value) orelse Error.LengthOverflow;
}

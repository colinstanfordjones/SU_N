//! Ghost exchange policies for AMR.
//!
//! Defines how data is packed, unpacked, restricted, and prolongated during
//! ghost exchange. This decouples the MPI topology logic from the data type.

const std = @import("std");
const dist_exchange = @import("dist_exchange.zig");
const field_math = @import("field_math.zig");

/// Interface for Ghost Policy.
/// Types matching this interface can be used with DistGhostExchange.
///
/// Context: An opaque context passed to all methods (e.g. FieldArena, GaugeField).
/// Block: The AMR block type.
/// Payload: The type of data elements being exchanged (e.g. FieldType, Link).
pub fn FieldGhostPolicy(comptime Tree: type) type {
    const Block = Tree.BlockType;
    const FieldType = Tree.FieldType;
    const Arena = Tree.FieldArenaType;
    const GhostBuffer = @import("ghost_buffer.zig").GhostBuffer(Tree.FrontendType);
    const Nd = Tree.dimensions;

    return struct {
        pub const Payload = FieldType;
        pub const Context = struct {
            tree: *const Tree,
            arena: *const Arena,
            ghosts: []?*GhostBuffer.GhostFaces,
        };
        pub const ExchangeSpec = dist_exchange.ExchangeSpec(Context, Payload);

        pub fn exchangeSpec() ExchangeSpec {
            return .{
                .payload_len = Block.ghost_face_size,
                .payload_alignment = std.mem.Alignment.of(Payload),
                .pack_same_level = packSameLevel,
                .pack_coarse_to_fine = packCoarseToFine,
                .pack_fine_to_coarse = packFineToCoarse,
                .unpack_same_level = unpackSameLevel,
                .unpack_coarse_to_fine = unpackCoarseToFine,
                .unpack_fine_to_coarse = unpackFineToCoarse,
            };
        }

        pub fn packSameLevel(ctx: Context, block_idx: usize, face: usize, dest: []Payload, scratch: std.mem.Allocator) void {
            _ = scratch;
            const slot = ctx.tree.getFieldSlot(block_idx);
            if (slot != std.math.maxInt(usize)) {
                Block.extractBoundaryFaceRuntime(ctx.arena.getSlotConst(slot), face, dest);
            } else {
                zeroSlice(Payload, dest);
            }
        }

        pub fn packFineToCoarse(ctx: Context, block_idx: usize, face: usize, dest: []Payload, scratch: std.mem.Allocator) void {
            zeroSlice(Payload, dest);
            const slot = ctx.tree.getFieldSlot(block_idx);
            if (slot != std.math.maxInt(usize)) {
                const fine_face = scratch.alloc(Payload, Block.ghost_face_size) catch return;
                Block.extractBoundaryFaceRuntime(ctx.arena.getSlotConst(slot), face, fine_face);
                
                const block = &ctx.tree.blocks.items[block_idx];
                restrictFineFaceToCoarse(Block, Payload, Nd, fine_face, block.origin, face, dest);
            }
        }

        pub fn packCoarseToFine(ctx: Context, block_idx: usize, face: usize, dest: []Payload, scratch: std.mem.Allocator) void {
            // Same as SameLevel for Fields - we send the coarse data, receiver prolongates
            packSameLevel(ctx, block_idx, face, dest, scratch);
        }

        pub fn unpackSameLevel(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost_faces = ctx.ghosts[block_idx] orelse return;
            if (face >= ghost_faces.len) return;
            
            std.mem.copyForwards(Payload, ghost_faces[face][0..], src);
        }

        pub fn unpackCoarseToFine(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost_faces = ctx.ghosts[block_idx] orelse return;
            if (face >= ghost_faces.len) return;

            prolongateCoarseToFine(Block, Payload, Nd, src, ghost_faces[face][0..]);
        }

        pub fn unpackFineToCoarse(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost_faces = ctx.ghosts[block_idx] orelse return;
            if (face >= ghost_faces.len) return;

            addFace(Payload, ghost_faces[face][0..], src);
        }
    };
}

fn zeroSlice(comptime T: type, slice: []T) void {
    @memset(slice, std.mem.zeroes(T));
}

// =============================================================================
// Math Helpers (Moved from ghost_mpi.zig)
// =============================================================================

fn prolongateCoarseToFine(
    comptime BlockType: type,
    comptime Field: type,
    comptime Nd: usize,
    coarse_face: []const Field,
    fine_face: []Field,
) void {
    const block_size = BlockType.size;

    const face_strides: [Nd - 1]usize = comptime blk: {
        var strs: [Nd - 1]usize = undefined;
        var acc: usize = 1;
        for (0..Nd - 1) |i| {
            strs[i] = acc;
            acc *= block_size;
        }
        break :blk strs;
    };

    for (0..BlockType.ghost_face_size) |fine_idx| {
        var fine_coords: [Nd - 1]usize = undefined;
        var temp = fine_idx;
        inline for (0..Nd - 1) |k| {
            fine_coords[k] = temp % block_size;
            temp /= block_size;
        }

        var coarse_idx: usize = 0;
        inline for (0..Nd - 1) |k| {
            const coarse_coord = fine_coords[k] / 2;
            coarse_idx += coarse_coord * face_strides[k];
        }

        if (coarse_idx >= coarse_face.len) {
            coarse_idx = coarse_face.len - 1;
        }

        fine_face[fine_idx] = coarse_face[coarse_idx];
    }
}

fn restrictFineFaceToCoarse(
    comptime BlockType: type,
    comptime Field: type,
    comptime Nd: usize,
    fine_face: []const Field,
    fine_origin: [Nd]usize,
    face: usize,
    coarse_face: []Field,
) void {
    const block_size = BlockType.size;

    if (comptime Nd == 1) {
        if (coarse_face.len > 0 and fine_face.len > 0) {
            coarse_face[0] = fine_face[0];
        }
        return;
    }
    const face_dim = face / 2;

    const face_strides: [Nd - 1]usize = comptime blk: {
        var strs: [Nd - 1]usize = undefined;
        var acc: usize = 1;
        for (0..Nd - 1) |i| {
            strs[i] = acc;
            acc *= block_size;
        }
        break :blk strs;
    };

    var fine_offset: [Nd - 1]usize = undefined;
    {
        var k: usize = 0;
        inline for (0..Nd) |d| {
            if (d != face_dim) {
                const fine_in_coarse = fine_origin[d] / 2;
                fine_offset[k] = fine_in_coarse % block_size;
                k += 1;
            }
        }
    }

    const cells_per_dim = block_size / 2;
    var coarse_coords: [Nd - 1]usize = .{0} ** (Nd - 1);

    const num_coarse_cells = comptime blk: {
        var n: usize = 1;
        for (0..Nd - 1) |_| n *= block_size / 2;
        break :blk n;
    };

    const num_fine = @as(usize, 1) << @intCast(Nd - 1);
    const scale = 1.0 / @as(f64, @floatFromInt(num_fine));

    for (0..num_coarse_cells) |_| {
        var fine_base: [Nd - 1]usize = undefined;
        inline for (0..Nd - 1) |k| {
            fine_base[k] = coarse_coords[k] * 2;
        }

        var sum: Field = field_math.zeroField(Field);
        for (0..num_fine) |fine_sub| {
            var fine_coords: [Nd - 1]usize = undefined;
            inline for (0..Nd - 1) |k| {
                fine_coords[k] = fine_base[k] + ((fine_sub >> @intCast(k)) & 1);
            }

            var fine_idx: usize = 0;
            inline for (0..Nd - 1) |k| {
                fine_idx += fine_coords[k] * face_strides[k];
            }

            if (fine_idx < fine_face.len) {
                sum = field_math.addField(Field, sum, fine_face[fine_idx]);
            }
        }

        const avg = field_math.scaleField(Field, sum, scale);

        var coarse_idx: usize = 0;
        inline for (0..Nd - 1) |k| {
            coarse_idx += (fine_offset[k] + coarse_coords[k]) * face_strides[k];
        }

        if (coarse_idx < coarse_face.len) {
            coarse_face[coarse_idx] = avg;
        }

        for (0..Nd - 1) |k| {
            coarse_coords[k] += 1;
            if (coarse_coords[k] < cells_per_dim) break;
            coarse_coords[k] = 0;
        }
    }
}

fn addFace(comptime Field: type, dst: []Field, src: []const Field) void {
    for (dst, src) |*d, s| {
        d.* = field_math.addField(Field, d.*, s);
    }
}

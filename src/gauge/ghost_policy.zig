//! Ghost exchange policy for Gauge Links ($U_\mu$).
//!
//! Handles boundary link packing, unpacking, prolongation, and restriction.

const std = @import("std");
const amr = @import("amr");
const dist_exchange = @import("amr").dist_exchange;

pub fn LinkGhostPolicy(comptime GaugeField: type) type {
    const Frontend = GaugeField.FrontendType;
    const Tree = GaugeField.TreeType;
    const Block = Tree.BlockType;
    const Link = GaugeField.LinkType;
    const Nd = Tree.dimensions;
    const LinkOps = Frontend.LinkOperators;
    const LinkArena = GaugeField.LinkArena;
    const LinkGhostFaces = GaugeField.LinkGhostFaces;

    return struct {
        pub const Payload = Link;
        
        /// Context for Link exchange
        pub const Context = struct {
            tree: *const Tree,
            arena: *const LinkArena,
            ghosts: []?*LinkGhostFaces,
            slots: []const usize,
        };
        pub const ExchangeSpec = dist_exchange.ExchangeSpec(Context, Payload);

        pub fn exchangeSpec() ExchangeSpec {
            return .{
                .payload_len = Block.ghost_face_size * Nd,
                .payload_alignment = std.mem.Alignment.of(Payload),
                .pack_same_level = packSameLevel,
                .pack_coarse_to_fine = packCoarseToFine,
                .pack_fine_to_coarse = packFineToCoarse,
                .unpack_same_level = unpackSameLevel,
                .unpack_coarse_to_fine = unpackCoarseToFine,
                .unpack_fine_to_coarse = unpackFineToCoarse,
                .should_exchange = shouldExchange,
            };
        }

        /// Size of the payload for a single face (including all link directions)
        pub fn facePayloadSize() usize {
            return Block.ghost_face_size * Nd;
        }

        pub fn packSameLevel(ctx: Context, block_idx: usize, face: usize, dest: []Payload, scratch: std.mem.Allocator) void {
            _ = scratch;
            if (block_idx >= ctx.slots.len) return;
            const slot = ctx.slots[block_idx];
            if (slot == std.math.maxInt(usize)) {
                @memset(dest, Link.identity());
                return;
            }
            const links = ctx.arena.getSlotConst(slot);

            var offset: usize = 0;
            inline for (0..Nd) |link_dim| {
                const sub_slice = dest[offset .. offset + Block.ghost_face_size];
                extractBoundaryLinkFace(Block, Nd, links, face, link_dim, sub_slice);
                offset += Block.ghost_face_size;
            }
        }

        pub fn packFineToCoarse(ctx: Context, block_idx: usize, face: usize, dest: []Payload, scratch: std.mem.Allocator) void {
            if (block_idx >= ctx.slots.len) return;
            const slot = ctx.slots[block_idx];
            if (slot == std.math.maxInt(usize)) {
                @memset(dest, Link.zero());
                return;
            }
            const links = ctx.arena.getSlotConst(slot);
            const block = ctx.tree.getBlock(block_idx).?;

            var offset: usize = 0;
            inline for (0..Nd) |link_dim| {
                const sub_slice = dest[offset .. offset + Block.ghost_face_size];

                if (scratch.alloc(Link, Block.ghost_face_size)) |fine_face| {
                    extractBoundaryLinkFace(Block, Nd, links, face, link_dim, fine_face);
                    restrictFineLinkFaceToCoarse(Block, Link, LinkOps, Nd, fine_face, block.origin, face, link_dim, sub_slice);
                    scratch.free(fine_face);
                } else |_| {
                    @memset(sub_slice, Link.zero());
                }

                offset += Block.ghost_face_size;
            }
        }

        pub fn packCoarseToFine(ctx: Context, block_idx: usize, face: usize, dest: []Payload, scratch: std.mem.Allocator) void {
            packSameLevel(ctx, block_idx, face, dest, scratch);
        }

        pub fn unpackSameLevel(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost = ctx.ghosts[block_idx] orelse return;
            
            var offset: usize = 0;
            inline for (0..Nd) |link_dim| {
                const dest_slice = ghost.getMut(face, link_dim);
                std.mem.copyForwards(Link, dest_slice, src[offset .. offset + Block.ghost_face_size]);
                offset += Block.ghost_face_size;
            }
        }

        pub fn unpackCoarseToFine(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost = ctx.ghosts[block_idx] orelse return;
            const block = ctx.tree.getBlock(block_idx).?;

            var offset: usize = 0;
            inline for (0..Nd) |link_dim| {
                const dest_slice = ghost.getMut(face, link_dim);
                const src_slice = src[offset .. offset + Block.ghost_face_size];

                prolongateCoarseToFineLink(Block, Link, LinkOps, Nd, block, face, link_dim, src_slice, dest_slice);
                offset += Block.ghost_face_size;
            }
        }

        pub fn unpackFineToCoarse(ctx: Context, block_idx: usize, face: usize, src: []const Payload) void {
            if (block_idx >= ctx.ghosts.len) return;
            const ghost = ctx.ghosts[block_idx] orelse return;

            var offset: usize = 0;
            inline for (0..Nd) |link_dim| {
                const src_slice = src[offset .. offset + Block.ghost_face_size];
                const dest_slice = ghost.getMut(face, link_dim);

                // Disjoint update: Overwrite if src is non-zero (sentinel)
                for (dest_slice, src_slice) |*d, s| {
                    if (s.norm() > 1e-9) {
                        d.* = s;
                    }
                }
                offset += Block.ghost_face_size;
            }
        }

        pub fn shouldExchange(_: Context, _: usize) bool {
            return true;
        }
    };
}

// =============================================================================
// Helper Functions (LQCD Link Specific)
// =============================================================================

fn extractBoundaryLinkFace(
    comptime Block: type,
    comptime Nd: usize,
    links: anytype,
    face: usize,
    link_dim: usize,
    dest: anytype,
) void {
    const block_size = Block.size;
    const face_dim = face / 2;
    const is_positive = (face % 2) == 0;
    const boundary_coord = if (is_positive) block_size - 1 else 0;

    var dest_idx: usize = 0;
    var coords: [Nd]usize = .{0} ** Nd;
    coords[face_dim] = boundary_coord;

    for (0..Block.ghost_face_size) |_| {
        const site_idx = Block.getLocalIndex(coords);
        dest[dest_idx] = links[site_idx * Nd + link_dim];
        dest_idx += 1;

        for (0..Nd) |d| {
            if (d == face_dim) continue;
            coords[d] += 1;
            if (coords[d] < block_size) break;
            coords[d] = 0;
        }
    }
}

fn prolongateCoarseToFineLink(
    comptime Block: type,
    comptime Link: type,
    comptime LinkOps: anytype,
    comptime Nd: usize,
    fine_block: anytype,
    face_idx: usize,
    link_dim: usize,
    coarse_face: []const Link,
    dest: []Link,
) void {
    const block_size = Block.size;
    const face_dim = face_idx / 2;
    _ = fine_block;

    const face_strides: [Nd - 1]usize = comptime blk: {
        var strs: [Nd - 1]usize = undefined;
        var acc: usize = 1;
        for (0..Nd - 1) |i| {
            strs[i] = acc;
            acc *= Block.size;
        }
        break :blk strs;
    };

    var fine_coords: [Nd - 1]usize = .{0} ** (Nd - 1);

    if (link_dim == face_dim) {
        const part_idx: usize = if ((face_idx % 2) == 0) 1 else 0;
        for (0..Block.ghost_face_size) |fine_idx| {
            var coarse_coords: [Nd - 1]usize = undefined;
            inline for (0..Nd - 1) |k| {
                coarse_coords[k] = fine_coords[k] / 2;
            }

            var coarse_idx: usize = 0;
            inline for (0..Nd - 1) |k| {
                coarse_idx += coarse_coords[k] * face_strides[k];
            }

            if (coarse_idx < coarse_face.len) {
                const u_coarse = coarse_face[coarse_idx];
                const parts = LinkOps.prolongateLink(u_coarse);
                dest[fine_idx] = parts[part_idx];
            } else {
                dest[fine_idx] = Link.identity();
            }

            for (0..Nd - 1) |k| {
                fine_coords[k] += 1;
                if (fine_coords[k] < block_size) break;
                fine_coords[k] = 0;
            }
        }
        return;
    }

    var link_dim_in_face_coords: usize = 0;
    {
        var d: usize = 0;
        var k: usize = 0;
        while (d < Nd) : (d += 1) {
            if (d == face_dim) continue;
            if (d == link_dim) {
                link_dim_in_face_coords = k;
                break;
            }
            k += 1;
        }
    }

    for (0..Block.ghost_face_size) |fine_idx| {
        var coarse_coords: [Nd - 1]usize = undefined;
        inline for (0..Nd - 1) |k| {
            coarse_coords[k] = fine_coords[k] / 2;
        }

        var coarse_idx: usize = 0;
        inline for (0..Nd - 1) |k| {
            coarse_idx += coarse_coords[k] * face_strides[k];
        }

        if (coarse_idx < coarse_face.len) {
            const u_coarse = coarse_face[coarse_idx];
            const parts = LinkOps.prolongateLink(u_coarse);
            const is_second_half = (fine_coords[link_dim_in_face_coords] % 2) != 0;
            dest[fine_idx] = if (is_second_half) parts[1] else parts[0];
        } else {
            dest[fine_idx] = Link.identity();
        }

        // Increment coords
        for (0..Nd - 1) |k| {
            fine_coords[k] += 1;
            if (fine_coords[k] < block_size) break;
            fine_coords[k] = 0;
        }
    }
}

fn restrictFineLinkFaceToCoarse(
    comptime Block: type,
    comptime Link: type,
    comptime LinkOps: anytype,
    comptime Nd: usize,
    fine_face: []const Link,
    fine_origin: [Nd]usize,
    face: usize,
    link_dim: usize,
    coarse_face: []Link,
) void {
    const block_size = Block.size;
    const face_dim = face / 2;
    const half_size = block_size / 2;

    // Initialize to Zero (Sentinel for sparse update)
    for (coarse_face) |*l| l.* = Link.zero();

    // Calculate face strides
    const face_strides: [Nd - 1]usize = comptime blk: {
        var strs: [Nd - 1]usize = undefined;
        var acc: usize = 1;
        for (0..Nd - 1) |i| {
            strs[i] = acc;
            acc *= Block.size;
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

    const total_cells = comptime blk: {
        var n: usize = 1;
        for (0..Nd - 1) |_| n *= Block.size / 2;
        break :blk n;
    };

    if (link_dim == face_dim) {
        // Normal Link: Average over 2^(Nd-1) fine links on the face
        const num_fine = @as(usize, 1) << @intCast(Nd - 1);
        const scale = 1.0 / @as(f64, @floatFromInt(num_fine));

        var rel_coords: [Nd - 1]usize = .{0} ** (Nd - 1);

        for (0..total_cells) |_| {
            var fine_base: [Nd - 1]usize = undefined;
            inline for (0..Nd - 1) |k| {
                fine_base[k] = rel_coords[k] * 2;
            }

            var sum = Link.zero();

            // Iterate fine sub-cells
            for (0..num_fine) |k| {
                var fine_idx: usize = 0;
                inline for (0..Nd - 1) |d| {
                    const fine_coord = fine_base[d] + ((k >> @intCast(d)) & 1);
                    fine_idx += fine_coord * face_strides[d];
                }
                
                if (fine_idx < fine_face.len) {
                    sum = sum.add(fine_face[fine_idx]);
                }
            }

            var coarse_idx: usize = 0;
            inline for (0..Nd - 1) |k| {
                coarse_idx += (fine_offset[k] + rel_coords[k]) * face_strides[k];
            }

            if (coarse_idx < coarse_face.len) {
                // Average and Unitarize
                coarse_face[coarse_idx] = sum.scaleReal(scale).unitarize();
            }

            // Increment coords
            for (0..Nd - 1) |k| {
                rel_coords[k] += 1;
                if (rel_coords[k] < half_size) break;
                rel_coords[k] = 0;
            }
        }
        return;
    }

    // Tangential Link: Path ordering
    var link_dim_in_face_coords: usize = 0;
    {
        var d: usize = 0;
        var k: usize = 0;
        while (d < Nd) : (d += 1) {
            if (d == face_dim) continue;
            if (d == link_dim) {
                link_dim_in_face_coords = k;
                break;
            }
            k += 1;
        }
    }

    const link_stride: usize = face_strides[link_dim_in_face_coords];
    var rel_coords: [Nd - 1]usize = .{0} ** (Nd - 1);

    for (0..total_cells) |_| {
        var fine_base: [Nd - 1]usize = undefined;
        inline for (0..Nd - 1) |k| {
            fine_base[k] = rel_coords[k] * 2;
        }

        var fine_idx1: usize = 0;
        inline for (0..Nd - 1) |k| {
            fine_idx1 += fine_base[k] * face_strides[k];
        }
        const fine_idx2 = fine_idx1 + link_stride;

        var coarse_idx: usize = 0;
        inline for (0..Nd - 1) |k| {
            coarse_idx += (fine_offset[k] + rel_coords[k]) * face_strides[k];
        }

        if (fine_idx1 < fine_face.len and fine_idx2 < fine_face.len and coarse_idx < coarse_face.len) {
            const u_fine1 = fine_face[fine_idx1];
            const u_fine2 = fine_face[fine_idx2];
            coarse_face[coarse_idx] = LinkOps.restrictLink(u_fine1, u_fine2);
        }

        // Increment coords
        for (0..Nd - 1) |k| {
            rel_coords[k] += 1;
            if (rel_coords[k] < half_size) break;
            rel_coords[k] = 0;
        }
    }
}
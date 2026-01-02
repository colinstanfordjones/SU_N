//! MPI repartitioning helpers for gauge fields (entropy-weighted Morton contiguous).

const std = @import("std");
const amr = @import("amr");
const field_mod = @import("field.zig");

pub fn GaugePolicy(comptime Frontend: type) type {
    const Tree = amr.AMRTree(Frontend);
    const FieldArena = amr.FieldArena(Frontend);
    const GaugeField = field_mod.GaugeField(Frontend);
    const Link = Frontend.LinkType;

    return struct {
        pub const Context = struct {
            tree: *Tree,
            arena: *FieldArena,
            field: *GaugeField,
        };

        pub fn extraBytes(_: Context) usize {
            return Tree.BlockType.volume * Tree.dimensions * @sizeOf(Link);
        }

        pub fn extraAlignment(_: Context) u29 {
            return @alignOf(Link);
        }

        pub fn packExtra(ctx: Context, block_idx: usize, dest: []u8) void {
            const links = ctx.field.getBlockLinks(block_idx) orelse {
                @memset(dest, 0);
                return;
            };
            std.debug.assert(dest.len == links.len * @sizeOf(Link));
            std.mem.copyForwards(u8, dest, std.mem.sliceAsBytes(links));
        }

        pub fn unpackExtra(ctx: Context, block_idx: usize, src: []const u8) void {
            if (ctx.field.getBlockLinksMut(block_idx)) |links| {
                std.debug.assert(src.len == links.len * @sizeOf(Link));
                std.mem.copyForwards(u8, std.mem.sliceAsBytes(links), src);
            }
        }

        pub fn insertBlock(
            ctx: Context,
            origin: [Tree.dimensions]usize,
            level: u8,
            has_field: bool,
        ) !usize {
            const block_idx = if (has_field)
                try ctx.tree.insertBlockWithField(origin, level, ctx.arena)
            else
                try ctx.tree.insertBlock(origin, level);
            try ctx.field.syncWithTree(ctx.tree);
            return block_idx;
        }

        pub fn compact(ctx: Context, options: amr.repartition.RepartitionOptions) !void {
            if (!options.compact) return;
            const perm = try ctx.tree.reorder();
            defer ctx.tree.allocator.free(perm);
            try ctx.field.reorder(perm);
            if (options.defragment) {
                try ctx.arena.defragmentWithOrder(ctx.tree.field_slots.items, ctx.tree.blockCount());
            }
        }
    };
}

pub fn repartitionEntropyWeighted(
    comptime Frontend: type,
    tree: *amr.AMRTree(Frontend),
    field: *field_mod.GaugeField(Frontend),
    arena: *amr.FieldArena(Frontend),
    shard: *amr.AMRTree(Frontend).ShardContext,
    options: amr.repartition.RepartitionOptions,
) !void {
    var opts = options;
    opts.compact = true;

    const Policy = GaugePolicy(Frontend);
    const ctx = Policy.Context{ .tree = tree, .arena = arena, .field = field };

    try amr.repartition.repartitionEntropyWeightedWithPolicy(amr.AMRTree(Frontend), Policy, ctx, shard, opts);
    field.ghosts.invalidateAll();
}

pub fn repartitionAdaptiveEntropyWeighted(
    comptime Frontend: type,
    tree: *amr.AMRTree(Frontend),
    field: *field_mod.GaugeField(Frontend),
    arena: *amr.FieldArena(Frontend),
    shard: *amr.AMRTree(Frontend).ShardContext,
    options: amr.repartition.RepartitionOptions,
    adaptive: amr.repartition.AdaptiveOptions,
) !bool {
    var opts = options;
    opts.compact = true;

    const Policy = GaugePolicy(Frontend);
    const ctx = Policy.Context{ .tree = tree, .arena = arena, .field = field };

    const did_repartition = try amr.repartition.repartitionAdaptiveEntropyWeightedWithPolicy(
        amr.AMRTree(Frontend),
        Policy,
        ctx,
        shard,
        opts,
        adaptive,
    );
    if (did_repartition) {
        field.ghosts.invalidateAll();
    }
    return did_repartition;
}

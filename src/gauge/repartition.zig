//! MPI repartitioning helpers for GaugeTree (entropy-weighted Morton contiguous).

const std = @import("std");
const amr = @import("amr");

pub fn GaugePolicy(comptime GaugeTree: type) type {
    return struct {
        pub const Context = struct {
            tree: *GaugeTree.TreeType,
            arena: *GaugeTree.FieldArena,
            gauge: *GaugeTree,
        };

        pub fn extraBytes(_: Context) usize {
            return GaugeTree.BlockType.volume * GaugeTree.dimensions * @sizeOf(GaugeTree.LinkType);
        }

        pub fn extraAlignment(_: Context) u29 {
            return @alignOf(GaugeTree.LinkType);
        }

        pub fn packExtra(ctx: Context, block_idx: usize, dest: []u8) void {
            const links = ctx.gauge.getBlockLinksConst(block_idx) orelse {
                @memset(dest, 0);
                return;
            };
            std.debug.assert(dest.len == links.len * @sizeOf(GaugeTree.LinkType));
            std.mem.copyForwards(u8, dest, std.mem.sliceAsBytes(links));
        }

        pub fn unpackExtra(ctx: Context, block_idx: usize, src: []const u8) void {
            if (ctx.gauge.getBlockLinksMut(block_idx)) |links| {
                std.debug.assert(src.len == links.len * @sizeOf(GaugeTree.LinkType));
                std.mem.copyForwards(u8, std.mem.sliceAsBytes(links), src);
            }
        }

        pub fn insertBlock(
            ctx: Context,
            origin: [GaugeTree.dimensions]usize,
            level: u8,
            has_field: bool,
        ) !usize {
            if (has_field) {
                return ctx.gauge.insertBlockWithField(origin, level, ctx.arena);
            }
            return ctx.gauge.insertBlock(origin, level);
        }

        pub fn compact(ctx: Context, options: amr.repartition.RepartitionOptions) !void {
            if (!options.compact) return;
            try ctx.gauge.reorder();
            if (options.defragment) {
                try ctx.arena.defragmentWithOrder(ctx.tree.field_slots.items, ctx.tree.blockCount());
            }
        }
    };
}

pub fn repartitionEntropyWeighted(
    comptime GaugeTree: type,
    tree: *GaugeTree,
    arena: *GaugeTree.FieldArena,
    shard: *GaugeTree.TreeType.ShardContext,
    options: amr.repartition.RepartitionOptions,
) !void {
    var opts = options;
    opts.compact = true;

    const Policy = GaugePolicy(GaugeTree);
    const ctx = Policy.Context{ .tree = &tree.tree, .arena = arena, .gauge = tree };

    try amr.repartition.repartitionEntropyWeightedWithPolicy(GaugeTree.TreeType, Policy, ctx, shard, opts);
    tree.ghosts_valid = false;
}

pub fn repartitionAdaptiveEntropyWeighted(
    comptime GaugeTree: type,
    tree: *GaugeTree,
    arena: *GaugeTree.FieldArena,
    shard: *GaugeTree.TreeType.ShardContext,
    options: amr.repartition.RepartitionOptions,
    adaptive: amr.repartition.AdaptiveOptions,
) !bool {
    var opts = options;
    opts.compact = true;

    const Policy = GaugePolicy(GaugeTree);
    const ctx = Policy.Context{ .tree = &tree.tree, .arena = arena, .gauge = tree };

    const did_repartition = try amr.repartition.repartitionAdaptiveEntropyWeightedWithPolicy(
        GaugeTree.TreeType,
        Policy,
        ctx,
        shard,
        opts,
        adaptive,
    );
    if (did_repartition) {
        tree.ghosts_valid = false;
    }
    return did_repartition;
}

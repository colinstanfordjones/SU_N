const std = @import("std");
const su_n = @import("su_n");
const amr = su_n.amr;
const platform = su_n.platform;

const Nd = 1;
const block_size = 8;
const blocks_per_rank = 1024;
const domain_extent = @as(f64, @floatFromInt(block_size * blocks_per_rank));

const Topology = amr.topology.OpenTopology(Nd, .{ domain_extent });
const Frontend = amr.ScalarFrontend(Nd, block_size, Topology);
const Tree = amr.AMRTree(Frontend);
const Arena = amr.FieldArena(Frontend);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    if (!platform.mpi.enabled) {
        std.debug.print("MPI disabled; skipping repartition benchmark.\n", .{});
        return;
    }

    _ = try platform.mpi.initSerialized();
    defer platform.mpi.finalize();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    const rank = try platform.mpi.rank(comm);

    if (size != 1) {
        if (rank == 0) {
            std.debug.print("Repartition benchmark expects MPI size 1; got {d}.\n", .{size});
        }
        return;
    }

    var tree = try Tree.init(allocator, 1.0, 10, 4);
    defer tree.deinit();

    var arena = try Arena.init(allocator, blocks_per_rank);
    defer arena.deinit();

    for (0..blocks_per_rank) |block_idx| {
        const origin = .{ block_idx * block_size };
        const idx = try tree.insertBlockWithField(origin, 0, &arena);
        const slot = tree.getFieldSlot(idx);
        const field_slice = arena.getSlot(slot);
        const base = @as(f64, @floatFromInt(block_idx + 1));
        for (field_slice, 0..) |*value, elem_idx| {
            value.* = base + @as(f64, @floatFromInt(elem_idx));
        }
    }

    var shard = try amr.ShardContext(Tree).initFromTree(allocator, &tree, comm, .manual);
    defer shard.deinit();
    tree.attachShard(&shard);

    const options = amr.repartition.RepartitionOptions{ .compact = true };

    try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, options);

    const iterations: usize = 10;
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try amr.repartition.repartitionEntropyWeighted(Tree, &tree, &arena, &shard, options);
    }

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const per_iter_ms = elapsed_s * 1000.0 / @as(f64, @floatFromInt(iterations));

    std.debug.print("Repartition benchmark: {d} blocks, {d} iterations\n", .{ blocks_per_rank, iterations });
    std.debug.print("Total: {d:.6} s, per iteration: {d:.6} ms\n", .{ elapsed_s, per_iter_ms });
}

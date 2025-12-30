const std = @import("std");
const amr = @import("amr");
const su_n = @import("su_n");
const platform = su_n.platform;

test "mpi shard context builds ownership map" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size < 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const domain_extent = 4096.0;
    const Topology = amr.topology.OpenTopology(1, .{domain_extent});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    const origin = .{@as(usize, @intCast(rank)) * block_size};
    _ = try tree.insertBlock(origin, 0);

    var shard = try amr.ShardContext(Tree).initFromTree(
        std.testing.allocator,
        &tree,
        comm,
        .manual,
    );
    defer shard.deinit();

    try std.testing.expectEqual(@as(usize, @intCast(size)), shard.owners.count());
    try std.testing.expectEqual(@as(usize, 1), shard.localBlockIndices().len);

    const key = tree.blockKeyFromOrigin(origin, 0);
    try std.testing.expect(shard.isLocalKey(key));

    const perm = try tree.reorder();
    defer std.testing.allocator.free(perm);
    try shard.refreshLocalBlocks(&tree);
    try std.testing.expectEqual(@as(usize, 1), shard.localBlockIndices().len);
}

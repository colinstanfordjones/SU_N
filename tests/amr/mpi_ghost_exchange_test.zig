const std = @import("std");
const amr = @import("amr");
const su_n = @import("su_n");
const constants = su_n.constants;
const platform = su_n.platform;

test "mpi ghost exchange handles open boundaries" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size < 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);
    const block_size = 4;
    const max_ranks = 64;
    if (size > max_ranks) return error.SkipZigTest;

    const domain_extent = @as(f64, @floatFromInt(block_size * max_ranks));
    const Topology = amr.topology.OpenTopology(1, .{domain_extent});
    const Frontend = amr.ScalarFrontend(1, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Block = Tree.BlockType;
    const FieldArena = amr.FieldArena(Frontend);
    const Ghosts = amr.GhostBuffer(Frontend);
    const ApplyContext = amr.ApplyContext(Frontend);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try FieldArena.init(std.testing.allocator, 8);
    defer arena.deinit();

    const origin = .{@as(usize, @intCast(rank)) * block_size};
    const block_idx = try tree.insertBlockWithField(origin, 0, &arena);

    const slot = tree.getFieldSlot(block_idx);
    const data = arena.getSlot(slot);
    for (data, 0..) |*v, i| {
        v.* = @as(f64, @floatFromInt(rank * 100 + @as(i32, @intCast(i))));
    }

    var ghosts = try Ghosts.init(std.testing.allocator, 8);
    defer ghosts.deinit();
    try ghosts.ensureForTree(&tree);

    var shard = try amr.ShardContext(Tree).initFromTree(std.testing.allocator, &tree, comm, .manual);
    defer shard.deinit();

    const Kernel = struct {
        pub fn execute(
            _: *const @This(),
            _: usize,
            _: *const Block,
            _: *ApplyContext,
        ) void {}
    };

    tree.attachShard(&shard);

    var kernel = Kernel{};
    var ctx = ApplyContext.init(&tree);
    ctx.field_in = &arena;
    ctx.field_out = &arena;
    ctx.field_ghosts = &ghosts;
    try tree.apply(&kernel, &ctx);

    const ghost_faces = ghosts.get(block_idx) orelse return error.TestExpectedEqual;

    if (rank + 1 < size) {
        const pos_expected = @as(f64, @floatFromInt((rank + 1) * 100 + 0));
        try std.testing.expectApproxEqAbs(pos_expected, ghost_faces[0][0], constants.test_epsilon);
    } else {
        try std.testing.expectApproxEqAbs(0.0, ghost_faces[0][0], constants.test_epsilon);
    }

    if (rank > 0) {
        const neg_expected = @as(f64, @floatFromInt((rank - 1) * 100 + (block_size - 1)));
        try std.testing.expectApproxEqAbs(neg_expected, ghost_faces[1][0], constants.test_epsilon);
    } else {
        try std.testing.expectApproxEqAbs(0.0, ghost_faces[1][0], constants.test_epsilon);
    }
}

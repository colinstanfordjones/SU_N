const std = @import("std");
const su_n = @import("su_n");

const amr = su_n.amr;
const checkpoint = su_n.platform.checkpoint;
const mpi = su_n.platform.mpi;

var init_level: ?mpi.ThreadLevel = null;

fn ensureMpi() !mpi.ThreadLevel {
    if (!mpi.enabled) return error.SkipZigTest;
    if (init_level == null) {
        init_level = try mpi.initSerialized();
    }
    return init_level.?;
}

test "mpi checkpoint restart ghost parity" {
    _ = try ensureMpi();

    const comm = mpi.commWorld();
    const size = try mpi.size(comm);
    if (size < 2) return error.SkipZigTest;
    const rank = try mpi.rank(comm);

    const Topology = amr.OpenTopology(2, .{ 8.0, 8.0 });
    const Frontend = amr.ScalarFrontend(2, 4, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const Ghosts = amr.GhostBuffer(Frontend);
    const ApplyContext = amr.ApplyContext(Frontend);
    const Checkpoint = checkpoint.TreeCheckpoint(Tree);

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena = try Arena.init(std.testing.allocator, 4);
    defer arena.deinit();

    const origin = .{ @as(usize, @intCast(rank)) * Frontend.block_size, 0 };
    const block_idx = try tree.insertBlockWithField(origin, 0, &arena);

    const slot = tree.getFieldSlot(block_idx);
    const data = arena.getSlot(slot);
    for (data, 0..) |*val, i| {
        val.* = @as(f64, @floatFromInt(rank * 10000 + @as(i32, @intCast(i))));
    }

    var shard = try amr.ShardContext(Tree).initFromTree(std.testing.allocator, &tree, comm, .morton_contiguous);
    defer shard.deinit();
    tree.attachShard(&shard);
    defer tree.detachShard();

    var ghosts_before = try Ghosts.init(std.testing.allocator, 4);
    defer ghosts_before.deinit();
    try ghosts_before.ensureForTree(&tree);

    const NoopKernel = struct {
        pub fn execute(
            _: *const @This(),
            _: usize,
            _: *const Tree.BlockType,
            _: *ApplyContext,
        ) void {}
    };

    var kernel = NoopKernel{};
    var ctx = ApplyContext.init(&tree);
    ctx.field_in = &arena;
    ctx.field_out = &arena;
    ctx.field_ghosts = &ghosts_before;
    try tree.apply(&kernel, &ctx);

    const ghost_faces_before = ghosts_before.get(block_idx) orelse return error.TestExpectedEqual;
    var before_copy: Ghosts.GhostFaces = ghost_faces_before.*;

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(std.testing.allocator);
    const writer = buffer.writer(std.testing.allocator);
    try Checkpoint.write(&tree, &arena, writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    var restored = try Checkpoint.read(std.testing.allocator, stream.reader());
    defer restored.deinit();

    var shard_restored = try amr.ShardContext(Tree).initFromTree(std.testing.allocator, &restored.tree, comm, .morton_contiguous);
    defer shard_restored.deinit();
    restored.tree.attachShard(&shard_restored);
    defer restored.tree.detachShard();

    var ghosts_after = try Ghosts.init(std.testing.allocator, 4);
    defer ghosts_after.deinit();
    try ghosts_after.ensureForTree(&restored.tree);

    var restored_ctx = ApplyContext.init(&restored.tree);
    restored_ctx.field_in = &restored.arena;
    restored_ctx.field_out = &restored.arena;
    restored_ctx.field_ghosts = &ghosts_after;
    try restored.tree.apply(&kernel, &restored_ctx);

    const ghost_faces_after = ghosts_after.get(block_idx) orelse return error.TestExpectedEqual;
    try std.testing.expect(std.mem.eql(u8, std.mem.asBytes(&before_copy), std.mem.asBytes(&ghost_faces_after.*)));
}

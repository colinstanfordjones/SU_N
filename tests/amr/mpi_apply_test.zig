const std = @import("std");
const amr = @import("amr");
const su_n = @import("su_n");
const constants = su_n.constants;
const platform = su_n.platform;

test "mpi apply matches reference across refinement boundary" {
    if (!platform.mpi.enabled) return error.SkipZigTest;
    _ = try platform.mpi.initSerialized();

    const comm = platform.mpi.commWorld();
    const size = try platform.mpi.size(comm);
    if (size < 2) return error.SkipZigTest;

    const rank = try platform.mpi.rank(comm);

    const block_size = 4;
    const domain_extent = 16.0;
    const Topology = amr.topology.OpenTopology(2, .{ domain_extent, domain_extent });
    const Frontend = amr.ScalarFrontend(2, block_size, Topology);
    const Tree = amr.AMRTree(Frontend);
    const Arena = amr.FieldArena(Frontend);
    const Ghosts = amr.GhostBuffer(Frontend);
    const Block = amr.AMRBlock(Frontend);
    const ApplyContext = amr.ApplyContext(Frontend);

    const coarse_origin = .{ 0, 0 };
    const fine_origin0 = .{ block_size * 2, 0 };
    const fine_origin1 = .{ block_size * 2, block_size };

    var tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer tree.deinit();

    var arena_in = try Arena.init(std.testing.allocator, 4);
    defer arena_in.deinit();

    var arena_out = try Arena.init(std.testing.allocator, 4);
    defer arena_out.deinit();

    if (rank == 0) {
        const idx = try tree.insertBlockWithField(coarse_origin, 0, &arena_in);
        const slot = tree.getFieldSlot(idx);
        const out_slot = arena_out.allocSlot() orelse return error.OutOfMemory;
        try std.testing.expectEqual(slot, out_slot);
    } else {
        const idx0 = try tree.insertBlockWithField(fine_origin0, 1, &arena_in);
        const slot0 = tree.getFieldSlot(idx0);
        const out_slot0 = arena_out.allocSlot() orelse return error.OutOfMemory;
        try std.testing.expectEqual(slot0, out_slot0);

        const idx1 = try tree.insertBlockWithField(fine_origin1, 1, &arena_in);
        const slot1 = tree.getFieldSlot(idx1);
        const out_slot1 = arena_out.allocSlot() orelse return error.OutOfMemory;
        try std.testing.expectEqual(slot1, out_slot1);
    }

    fillFields(Tree, &tree, &arena_in);

    var ghosts = try Ghosts.init(std.testing.allocator, 4);
    defer ghosts.deinit();

    var shard = try amr.ShardContext(Tree).initFromTree(std.testing.allocator, &tree, comm, .manual);
    defer shard.deinit();
    tree.attachShard(&shard);

    var ref_tree = try Tree.init(std.testing.allocator, 1.0, 4, 8);
    defer ref_tree.deinit();

    var ref_in = try Arena.init(std.testing.allocator, 4);
    defer ref_in.deinit();

    var ref_out = try Arena.init(std.testing.allocator, 4);
    defer ref_out.deinit();

    const ref_coarse = try ref_tree.insertBlockWithField(coarse_origin, 0, &ref_in);
    const ref_coarse_slot = ref_tree.getFieldSlot(ref_coarse);
    const ref_out_slot0 = ref_out.allocSlot() orelse return error.OutOfMemory;
    try std.testing.expectEqual(ref_coarse_slot, ref_out_slot0);

    const ref_fine0 = try ref_tree.insertBlockWithField(fine_origin0, 1, &ref_in);
    const ref_fine_slot0 = ref_tree.getFieldSlot(ref_fine0);
    const ref_out_slot1 = ref_out.allocSlot() orelse return error.OutOfMemory;
    try std.testing.expectEqual(ref_fine_slot0, ref_out_slot1);

    const ref_fine1 = try ref_tree.insertBlockWithField(fine_origin1, 1, &ref_in);
    const ref_fine_slot1 = ref_tree.getFieldSlot(ref_fine1);
    const ref_out_slot2 = ref_out.allocSlot() orelse return error.OutOfMemory;
    try std.testing.expectEqual(ref_fine_slot1, ref_out_slot2);

    fillFields(Tree, &ref_tree, &ref_in);

    var ref_ghosts = try Ghosts.init(std.testing.allocator, 4);
    defer ref_ghosts.deinit();

    const Kernel = struct {
        tree_ref: *const Tree,

        pub fn execute(
            self: *const @This(),
            block_idx: usize,
            _: *const Block,
            ctx: *ApplyContext,
        ) void {
            const slot = self.tree_ref.getFieldSlot(block_idx);
            if (slot == std.math.maxInt(usize)) return;

            const inputs = ctx.field_in orelse return;
            const outputs = ctx.field_out orelse return;

            const in_field = inputs.getSlotConst(slot);
            const out_field = outputs.getSlot(slot);

            const ghost_faces = if (ctx.field_ghosts) |g| g.get(block_idx) else null;

            for (in_field, 0..) |val, i| {
                const coords = Block.getLocalCoords(i);
                if (Block.isOnBoundary(coords)) {
                    // Boundary cell - use ghost data
                    var ghost_sum: f64 = 0.0;
                    if (ghost_faces) |gf| {
                        for (0..Block.num_ghost_faces) |face| {
                            const dim = face / 2;
                            const is_positive = (face % 2) == 0;
                            if (is_positive and coords[dim] != Block.size - 1) continue;
                            if (!is_positive and coords[dim] != 0) continue;

                            const ghost_idx = Block.getGhostIndexRuntime(coords, face);
                            ghost_sum += gf[face][ghost_idx];
                        }
                    }
                    out_field[i] = val + ghost_sum;
                } else {
                    // Interior cell
                    out_field[i] = val + 1.0;
                }
            }
        }
    };

    var kernel_ref = Kernel{ .tree_ref = &ref_tree };
    var kernel_local = Kernel{ .tree_ref = &tree };

    var ref_ctx = ApplyContext.init(&ref_tree);
    ref_ctx.field_in = &ref_in;
    ref_ctx.field_out = &ref_out;
    ref_ctx.field_ghosts = &ref_ghosts;
    try ref_tree.apply(&kernel_ref, &ref_ctx);

    // Parallel tree
    var ctx = ApplyContext.init(&tree);
    ctx.field_in = &arena_in;
    ctx.field_out = &arena_out;
    ctx.field_ghosts = &ghosts;
    try tree.apply(&kernel_local, &ctx);

    for (tree.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;

        const ref_idx = ref_tree.findBlockByOrigin(block.origin, block.level) orelse return error.TestExpectedEqual;
        const slot = tree.getFieldSlot(idx);
        const ref_slot = ref_tree.getFieldSlot(ref_idx);

        const out_field = arena_out.getSlot(slot);
        const ref_field = ref_out.getSlot(ref_slot);

        for (out_field, 0..) |val, i| {
            try std.testing.expectApproxEqAbs(ref_field[i], val, constants.test_epsilon);
        }
    }
}

fn fillFields(comptime Tree: type, tree_ptr: *const Tree, arena: *Tree.FieldArenaType) void {
    for (tree_ptr.blocks.items, 0..) |*block, idx| {
        if (block.block_index == std.math.maxInt(usize)) continue;
        const slot = tree_ptr.getFieldSlot(idx);
        if (slot == std.math.maxInt(usize)) continue;

        const data = arena.getSlot(slot);
        const base = @as(f64, @floatFromInt(block.origin[0] + block.origin[1] * 10 + block.level * 100));
        for (data, 0..) |*v, i| {
            v.* = base + @as(f64, @floatFromInt(i)) * 0.01;
        }
    }
}

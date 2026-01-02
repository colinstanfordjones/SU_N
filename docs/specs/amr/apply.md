# AMR Apply Pipeline

`AMRTree.apply` is the single kernel execution entry point. It runs a threaded schedule on a persistent work-stealing pool owned by the tree and uses `ApplyContext` to bind state.

## Pipeline Stages

1. Ensure field ghost buffers are sized for the current tree.
2. Exchange field ghosts when `ctx.field_in` and `ctx.field_ghosts` are set (local + MPI).
3. Execute `kernel.execute` across all blocks on the worker pool.

The pool is created during `AMRTree.init` and defaults to the machine CPU count. Task metadata is allocated from a per-group arena to avoid allocation churn.

## Kernel Interface

Required method:

```zig
fn execute(self: *Self, block_idx: usize, block: *const Block, ctx: *ApplyContext) void
```

## ApplyContext Configuration

- `ctx.setFields(&arena_in, &arena_out)` configures field input/output.
- `ctx.setFieldGhosts(&ghosts)` enables field ghost exchange.
- `ctx.setEdges(&edge_arena, &edge_ghosts)` passes edge-centered data (ghost exchange is explicit).
- Edge-centered storage is supported today; face- and node-centered storage are planned extensions.
- If a shard context is attached via `tree.attachShard(&shard)`, `AMRTree.apply` performs MPI ghost exchange for fields.

## Neighbor Queries (No Cached Neighbor Arrays)

Blocks do not store neighbor indices. If a kernel needs topology-level adjacency information,
query the tree at runtime:

```zig
const info = tree.neighborInfo(block_idx, face);
if (info.exists()) {
    // level_diff: 0 (same), -1 (coarser)
}

var fine_neighbors: [Tree.max_fine_neighbors]usize = undefined;
const fine_count = tree.collectFineNeighbors(block_idx, face, &fine_neighbors);
```

Refined faces return `none` from `neighborInfo` to preserve the push model. Fine neighbors are enumerated separately.

## Example

```zig
const amr = @import("amr");

const Frontend = amr.ScalarFrontend(2, 8, amr.topology.OpenTopology(2, .{ 16.0, 16.0 }));
const Tree = amr.AMRTree(Frontend);
const Arena = amr.FieldArena(Frontend);
const Ghosts = amr.GhostBuffer(Frontend);
const ApplyContext = amr.ApplyContext(Frontend);

const Kernel = struct {
    tree: *const Tree,

    pub fn execute(
        self: *Kernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        ctx: *ApplyContext,
    ) void {
        _ = block;
        const slot = self.tree.getFieldSlot(block_idx);
        const in = ctx.field_in orelse return;
        const out = ctx.field_out orelse return;
        const src = in.getSlotConst(slot);
        const dst = out.getSlot(slot);
        _ = src;
        _ = dst;
    }
};

var tree = try Tree.init(allocator, 1.0, 2, 8);
defer tree.deinit();
var arena_in = try Arena.init(allocator, 16);
defer arena_in.deinit();
var arena_out = try Arena.init(allocator, 16);
defer arena_out.deinit();
var ghosts = try Ghosts.init(allocator, 16);
defer ghosts.deinit();

var ctx = ApplyContext.init(&tree);
ctx.setFields(&arena_in, &arena_out);
ctx.setFieldGhosts(&ghosts);

var kernel = Kernel{ .tree = &tree };
try tree.apply(&kernel, &ctx);
```

## Link Ghosts (Gauge Kernels)

Gauge link ghosts are exchanged explicitly via `GaugeField` (edge ghosts are not exchanged by `AMRTree.apply`):

```zig
const gauge = @import("gauge");
const Field = gauge.GaugeField(Frontend);

var field = try Field.init(allocator, &tree);
defer field.deinit();
try field.syncWithTree(&tree);

var ctx = ApplyContext.init(&tree);
ctx.setEdges(&field.arena, &field.ghosts);
try field.fillGhosts(&tree);

try tree.apply(&kernel, &ctx);
```

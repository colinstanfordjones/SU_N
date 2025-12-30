# AMR Apply Pipeline

`AMRTree.apply` is the single kernel execution entry point. It runs a pipelined schedule on a persistent work-stealing pool owned by the tree.

## Pipeline Stages

1. Ghost pull (field ghosts, plus optional kernel ghost hooks)
2. Interior compute (overlaps with ghost pull)
3. Ghost push (field ghosts, plus optional kernel ghost hooks)
4. Boundary compute

The pool is created during `AMRTree.init` and defaults to the machine CPU count. Task metadata is allocated from a per-group arena to avoid allocation churn.

## Kernel Interface

Required methods:

```zig
fn executeInterior(self: *Self, block_idx: usize, block: *const Block, inputs: anytype, outputs: anytype, ghosts: ?*GhostBuffer) void
fn executeBoundary(self: *Self, block_idx: usize, block: *const Block, inputs: anytype, outputs: anytype, ghosts: ?*GhostBuffer) void
```

Optional ghost hooks (all or none):

```zig
fn ghostPrepare(self: *Self) !bool           // return true if exchange needed
fn ghostPull(self: *Self, block_idx: usize) void
fn ghostPush(self: *Self, block_idx: usize) void
fn ghostFinalize(self: *Self) void
```

`ghostPrepare` is called once at the start. If it returns `false`, the other hook stages are skipped.

## Inputs, Outputs, and Ghosts

- `inputs` and `outputs` are `anytype` so kernels can accept pointers, structs, or `void`.
- If `inputs` is a struct, the first element is treated as the field arena for field ghost exchange.
- If `inputs` is `void`, field ghost exchange is skipped.
- `ghosts` is optional; if provided, `AMRTree.apply` ensures capacity for the tree.
- If a shard context is attached via `tree.attachShard(&shard)`, `AMRTree.apply` performs MPI ghost exchange in addition to local ghosts.

## Neighbor Queries (No Cached Neighbor Arrays)

Blocks no longer store neighbor indices. If a kernel needs topology-level adjacency information,
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

const Kernel = struct {
    tree: *const Tree,

    pub fn executeInterior(
        self: *Kernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*Ghosts,
    ) void {
        self.executeRegion(.interior, block_idx, block, inputs, outputs, ghosts);
    }

    pub fn executeBoundary(
        self: *Kernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*Ghosts,
    ) void {
        self.executeRegion(.boundary, block_idx, block, inputs, outputs, ghosts);
    }

    const SiteRegion = enum { interior, boundary };
    fn executeRegion(
        self: *Kernel,
        region: SiteRegion,
        block_idx: usize,
        block: *const Tree.BlockType,
        inputs: *Arena,
        outputs: *Arena,
        ghosts: ?*Ghosts,
    ) void {
        _ = self;
        _ = block;
        _ = ghosts;
        _ = region;
        const slot = self.tree.getFieldSlot(block_idx);
        const in = inputs.getSlotConst(slot);
        const out = outputs.getSlot(slot);
        _ = in;
        _ = out;
    }
};

var tree = try Tree.init(allocator, 1.0, 2, 8);
var arena_in = try Arena.init(allocator, 16);
var arena_out = try Arena.init(allocator, 16);
var ghosts = try Ghosts.init(allocator, 16);

var kernel = Kernel{ .tree = &tree };
try tree.apply(&kernel, &arena_in, &arena_out, &ghosts);
```

## Link Ghosts (Gauge Kernels)

If a kernel needs gauge link ghosts, use the optional ghost hooks and forward them to `GaugeTree`:

```zig
pub fn ghostPrepare(self: *Self) !bool {
    return try self.gauge_tree.prepareLinkGhostExchange();
}
pub fn ghostPull(self: *Self, block_idx: usize) void {
    self.gauge_tree.fillGhostsPull(block_idx);
}
pub fn ghostPush(self: *Self, block_idx: usize) void {
    self.gauge_tree.fillGhostsPush(block_idx);
}
pub fn ghostFinalize(self: *Self) void {
    self.gauge_tree.finalizeLinkGhostExchange();
}
```

This keeps all pipeline logic inside `AMRTree.apply` while still synchronizing link ghosts.

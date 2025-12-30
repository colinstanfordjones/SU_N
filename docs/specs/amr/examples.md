# AMR Examples

This directory collects focused examples that mirror the unit tests. For full implementations, see the tests referenced below.

## Heat Equation (2D)

The heat equation kernel runs via `AMRTree.apply` using interior/boundary separation and field ghosts.

```zig
const amr = @import("amr");

const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
const Frontend = amr.ScalarFrontend(2, 16, Topology);
const Tree = amr.AMRTree(Frontend);
const Arena = amr.FieldArena(Frontend);
const Ghosts = amr.GhostBuffer(Frontend);

const HeatKernel = struct {
    tree: *const Tree,
    alpha: f64,
    dt: f64,

    pub fn executeInterior(self: *HeatKernel, block_idx: usize, block: *const Tree.BlockType, inputs: *Arena, outputs: *Arena, ghosts: ?*Ghosts) void {
        self.executeRegion(.interior, block_idx, block, inputs, outputs, ghosts);
    }

    pub fn executeBoundary(self: *HeatKernel, block_idx: usize, block: *const Tree.BlockType, inputs: *Arena, outputs: *Arena, ghosts: ?*Ghosts) void {
        self.executeRegion(.boundary, block_idx, block, inputs, outputs, ghosts);
    }

    const SiteRegion = enum { interior, boundary };
    fn executeRegion(self: *HeatKernel, region: SiteRegion, block_idx: usize, block: *const Tree.BlockType, inputs: *Arena, outputs: *Arena, ghosts: ?*Ghosts) void {
        _ = block;
        const slot = self.tree.getFieldSlot(block_idx);
        const u = inputs.getSlotConst(slot);
        const u_new = outputs.getSlot(slot);
        const ghost_faces = ghosts.?.get(block_idx).?;

        // Simple stencil here...
        _ = u;
        _ = u_new;
        _ = ghost_faces;
        _ = region;
        _ = self;
    }
};

// Usage:
// - init tree/arenas/ghosts
// - fill initial condition
// - call tree.apply(&kernel, &arena_in, &arena_out, &ghosts)
```

Reference: `tests/amr/heat_equation_test.zig`.

## Pipeline Stress + Timing

The stress tests verify threaded `AMRTree.apply` matches a sequential baseline and that ghost pull overlaps interior compute on multi-core builds.

Reference: `tests/amr/pipeline_stress_test.zig`.

## Checkpoint/Restart

Use the platform checkpoint module to snapshot tree/arena state for exact restarts.

```zig
const checkpoint = @import("su_n").platform.checkpoint;
const Ckpt = checkpoint.TreeCheckpoint(Tree);

try Ckpt.writeToFile(&tree, &arena, std.fs.cwd(), "run_step100.sunc");

var restored = try Ckpt.readFromFile(allocator, std.fs.cwd(), "run_step100.sunc");
defer restored.deinit();
```

## Notes

- Blocks do not store neighbor arrays; use `tree.neighborInfo` and `tree.collectFineNeighbors` when a kernel needs topology-level adjacency.
- For MPI runs, attach a shard context once and let `AMRTree.apply` handle distributed ghost exchange.

# AMR Examples

This directory collects focused examples that mirror the unit tests. For full implementations, see the tests referenced below.

## Heat Equation (2D)

The heat equation kernel runs via `AMRTree.apply` using `ApplyContext` and field ghosts.

```zig
const amr = @import("amr");

const Topology = amr.topology.OpenTopology(2, .{ 16.0, 16.0 });
const Frontend = amr.ScalarFrontend(2, 16, Topology);
const Tree = amr.AMRTree(Frontend);
const Arena = amr.FieldArena(Frontend);
const Ghosts = amr.GhostBuffer(Frontend);
const ApplyContext = amr.ApplyContext(Frontend);

const HeatKernel = struct {
    tree: *const Tree,
    alpha: f64,
    dt: f64,

    pub fn execute(
        self: *HeatKernel,
        block_idx: usize,
        block: *const Tree.BlockType,
        ctx: *ApplyContext,
    ) void {
        _ = block;
        const slot = self.tree.getFieldSlot(block_idx);
        const in = ctx.field_in orelse return;
        const out = ctx.field_out orelse return;

        const u = in.getSlotConst(slot);
        const u_new = out.getSlot(slot);

        var ghost_slices: [2 * Frontend.Nd][]const Frontend.FieldType = undefined;
        for (0..2 * Frontend.Nd) |f| ghost_slices[f] = &.{};
        if (ctx.field_ghosts) |ghosts| {
            if (ghosts.get(block_idx)) |gp| {
                for (0..2 * Frontend.Nd) |f| ghost_slices[f] = &gp[f];
            }
        }

        // Simple stencil here...
        _ = u;
        _ = u_new;
        _ = ghost_slices;
        _ = self;
    }
};

// Usage:
// - init tree/arenas/ghosts
// - fill initial condition
// - configure ApplyContext with fields + ghosts
// - call tree.apply(&kernel, &ctx)
```

Reference: `tests/amr/heat_equation_test.zig`.

## Pipeline Stress + Timing

The stress tests verify threaded `AMRTree.apply` matches a sequential baseline.

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
- For MPI runs, attach a shard context once and let `AMRTree.apply` handle distributed field ghost exchange.
- Gauge link ghosts are exchanged explicitly via `GaugeField.fillGhosts`.

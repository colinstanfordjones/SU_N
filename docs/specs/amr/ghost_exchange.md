# Ghost Exchange (Push Model)

Field ghost exchange uses a push model to handle refinement boundaries. Fine blocks pull from coarse neighbors, then push restricted data back to coarse neighbors that cannot pull.

## API

```zig
const ghost = @import("amr").ghost;
const Tree = amr.AMRTree(Frontend);
const Ghosts = amr.GhostBuffer(Frontend);

// Convenience: pull + push in one call
ghost.fillGhostLayers(Tree, &tree, &field_arena, ghosts.slice(tree.blocks.items.len));

// Split phases for overlap
ghost.beginGhostExchange(Tree, &tree, &field_arena, ghosts.slice(tree.blocks.items.len));
// ... interior compute ...
ghost.finishGhostExchange(Tree, &tree, &field_arena, ghosts.slice(tree.blocks.items.len));
// ... boundary compute ...
```

Per-block hooks for custom scheduling:

```zig
ghost.fillGhostLayersPull(Tree, &tree, block_idx, &field_arena, all_ghosts);
ghost.fillGhostLayersPush(Tree, &tree, block_idx, &field_arena, all_ghosts);
```

## MPI Ghost Exchange

Distributed runs use `dist_exchange.DistExchange` to exchange remote ghost faces.
`AMRTree.apply` invokes this automatically when a shard context is attached via `tree.attachShard(&shard)`.

```zig
const shard = try amr.ShardContext(Tree).initFromTree(allocator, &tree, comm, .manual);
const ghosts = try amr.GhostBuffer(Frontend).init(allocator, 16);
try ghosts.ensureForTree(&tree);

tree.attachShard(&shard);
try tree.apply(&kernel, &arena, &arena, &ghosts);
```

MPI neighbor resolution uses the ownership map keyed by Morton `BlockKey` values, not cached neighbor arrays.
Refined faces are handled via the same push model (fine -> coarse restriction).

## ExchangeSpec

`DistExchange` is initialized with an `ExchangeSpec` that captures payload sizing
and pack/unpack hooks. This keeps payload layout centralized and consistent across
MPI and local exchange paths (gauge links use the same spec for local ghosts).

```zig
const FieldPolicy = amr.ghost_policy.FieldGhostPolicy(Tree);
const FieldExchange = amr.dist_exchange.DistExchange(Tree, FieldPolicy.Context, FieldPolicy.Payload);
var exchange = FieldExchange.init(allocator, FieldPolicy.exchangeSpec());
```

`ExchangeSpec` also supports an optional `should_exchange` predicate to decide
whether a block participates in MPI exchange (fields default to blocks with
allocated field slots; gauge links exchange for all blocks).

You can also inject a custom exchange spec at initialization:

```zig
const opts = Tree.ExchangeOptions{
    .field_exchange_spec = custom_spec,
};
var tree = try Tree.initWithOptions(allocator, 1.0, 4, 8, opts);
defer tree.deinit();
```

Gauge trees accept a parallel option for link exchange:

```zig
const gt_opts = GaugeTree.ExchangeOptions{
    .tree_options = .{ .field_exchange_spec = custom_field_spec },
    .link_exchange_spec = custom_link_spec,
};
var gauge_tree = try GaugeTree.initWithOptions(allocator, 1.0, 4, 8, gt_opts);
```

For gauge links, `payload_len = block_size^(Nd-1) * Nd` to pack all link directions
(tangential + normal) for each face into a single message.

## Notes

- `GhostBuffer.ensureForTree` must be called whenever the block count changes.
- `GhostBuffer.slice(tree.blocks.items.len)` yields the `[]?*GhostFaces` array expected by the ghost helpers.
- `beginGhostExchange` only performs pull operations; coarse ghost faces are zero-initialized and later overwritten during the push phase.

## Overlap Strategy

A typical pipeline separates interior and boundary updates:

```zig
try ghosts.ensureForTree(&tree);
const ghost_ptrs = ghosts.slice(tree.blocks.items.len);

ghost.beginGhostExchange(Tree, &tree, &arena, ghost_ptrs);
// compute interior sites

ghost.finishGhostExchange(Tree, &tree, &arena, ghost_ptrs);
// compute boundary sites
```

`AMRTree.apply` performs this schedule internally for field ghosts and optional kernel ghost hooks.

//! Gauge Tree: High-level API for Gauge Theory on AMR
//!
//! Wraps AMRTree and GaugeField to provide a unified interface.
//! **Deprecated**: Prefer using AMRTree and GaugeField directly.

const std = @import("std");
const builtin = @import("builtin");
const amr = @import("amr");
const field_mod = @import("field.zig");
const operators_mod = @import("operators.zig");

pub const Complex = std.math.Complex(f64);

/// Gauge Tree wrapper.
pub fn GaugeTree(comptime Frontend: type) type {
    const Tree = amr.AMRTree(Frontend);
    const Field = field_mod.GaugeField(Frontend);
    const LinkOps = operators_mod.LinkOperators(Frontend);
    const Link = Frontend.LinkType;
    const Block = Tree.BlockType;

    return struct {
        const Self = @This();

        // Type exports for compatibility
        pub const FrontendType = Frontend;
        pub const TreeType = Tree;
        pub const BlockType = Block;
        pub const LinkType = Link;
        pub const FieldType = Frontend.FieldType;
        pub const GaugeFieldType = Field;
        pub const FieldArena = amr.FieldArena(Frontend);
        
        // These might be different types now, but tests should use generic types
        pub const GhostStorage = Field.LinkGhostFaces;
        pub const LinkArena = Field.LinkArena;
        pub const LinkGhostFaces = Field.LinkGhostFaces;
        
        allocator: std.mem.Allocator,
        tree: Tree,
        field: Field,
        
        // Compatibility state
        ghosts_valid: bool = false,

        pub const LinkExchangeSpec = Field.Policy.ExchangeSpec;
        pub const ExchangeOptions = struct {
            tree_options: Tree.ExchangeOptions = .{},
            link_exchange_spec: ?LinkExchangeSpec = null,
        };

        pub fn init(
            allocator: std.mem.Allocator,
            base_spacing: f64,
            bits_per_dim: u8,
            max_level: u8,
        ) !Self {
            return initWithOptions(allocator, base_spacing, bits_per_dim, max_level, .{});
        }

        pub fn initWithOptions(
            allocator: std.mem.Allocator,
            base_spacing: f64,
            bits_per_dim: u8,
            max_level: u8,
            options: ExchangeOptions,
        ) !Self {
            var tree = try Tree.initWithOptions(allocator, base_spacing, bits_per_dim, max_level, options.tree_options);
            const field = try Field.initWithOptions(allocator, &tree, options.link_exchange_spec);
            return Self{
                .allocator = allocator,
                .tree = tree,
                .field = field,
            };
        }

        pub fn deinit(self: *Self) void {
            self.field.deinit();
            self.tree.deinit();
        }

        pub fn insertBlock(self: *Self, origin: [Frontend.Nd]usize, level: u8) !usize {
            const idx = try self.tree.insertBlock(origin, level);
            try self.field.syncWithTree(&self.tree);
            self.ghosts_valid = false;
            return idx;
        }

        pub fn insertBlockWithField(self: *Self, origin: [Frontend.Nd]usize, level: u8, arena: anytype) !usize {
            const block_idx = try self.insertBlock(origin, level);
            const field_slot = arena.allocSlot() orelse return error.FieldArenaFull;
            self.tree.assignFieldSlot(block_idx, field_slot);
            return block_idx;
        }

        pub fn blockCount(self: *const Self) usize {
            return self.tree.blockCount();
        }

        pub fn getBlock(self: *const Self, block_idx: usize) ?*const Block {
            return self.tree.getBlock(block_idx);
        }

        pub fn getLink(self: *const Self, block_idx: usize, site: usize, mu: usize) Link {
            return self.field.getLink(block_idx, site, mu);
        }

        pub fn setLink(self: *Self, block_idx: usize, site: usize, mu: usize, link: Link) void {
            self.field.setLink(block_idx, site, mu, link);
            self.ghosts_valid = false;
        }

        pub fn getBlockLinksMut(self: *Self, block_idx: usize) ?[]Link {
            self.ghosts_valid = false;
            return self.field.getBlockLinksMut(block_idx);
        }

        pub fn getBlockLinksConst(self: *const Self, block_idx: usize) ?[]const Link {
            return self.field.getBlockLinks(block_idx);
        }

        pub fn fillGhosts(self: *Self) !void {
            if (self.ghosts_valid) return;
            try self.field.fillGhosts(&self.tree);
            self.ghosts_valid = true;
        }

        pub fn prepareApplyContext(self: *Self, ctx: *Tree.ApplyContext) !void {
            ctx.edges = &self.field.arena;
            ctx.edge_ghosts = &self.field.ghosts;
            if (!self.ghosts_valid or ctx.edge_ghosts_dirty) {
                try self.field.fillGhosts(&self.tree);
                self.ghosts_valid = true;
                ctx.edge_ghosts_dirty = false;
            }
        }

        pub fn apply(self: *Self, kernel: anytype, ctx: *Tree.ApplyContext) !void {
            try self.prepareApplyContext(ctx);
            try self.tree.apply(kernel, ctx);
        }

        // Forward operators
        pub fn computePlaquette(self: *const Self, block_idx: usize, site_idx: usize, comptime mu: usize, comptime nu: usize) Link {
            return LinkOps.computePlaquette(&self.tree, &self.field, block_idx, site_idx, mu, nu);
        }

        pub fn tracePlaquette(self: *const Self, block_idx: usize, site_idx: usize, comptime mu: usize, comptime nu: usize) f64 {
            return LinkOps.tracePlaquette(&self.tree, &self.field, block_idx, site_idx, mu, nu);
        }

        pub fn averagePlaquetteBlock(self: *const Self, block_idx: usize) f64 {
            return LinkOps.averagePlaquetteBlock(&self.tree, &self.field, block_idx);
        }

        pub fn averagePlaquetteTree(self: *const Self) f64 {
            return LinkOps.averagePlaquetteTree(&self.tree, &self.field);
        }

        pub fn wilsonAction(self: *const Self, beta: f64) f64 {
            return LinkOps.wilsonAction(&self.tree, &self.field, beta);
        }
        
        pub fn computeStaple(self: *const Self, block_idx: usize, site_idx: usize, comptime mu: usize) Link {
             return LinkOps.computeStaple(&self.tree, &self.field, block_idx, site_idx, mu);
        }

        pub fn getGhostLink(self: *const Self, block_idx: usize, face_idx: usize, link_dim: usize, ghost_idx: usize) Link {
            if (block_idx >= self.field.ghosts.slots.len) return Link.identity();
            const ghost = self.field.ghosts.slots[block_idx] orelse return Link.identity();
            const slice = ghost.get(face_idx, link_dim);
            if (ghost_idx < slice.len) return slice[ghost_idx];
            return Link.identity();
        }

        pub fn covariantLaplacianSite(
            self: *const Self,
            block_idx: usize,
            site_idx: usize,
            psi_local: anytype,
            psi_ghosts: anytype,
            spacing: f64,
        ) @TypeOf(psi_local[0]) {
            return LinkOps.covariantLaplacianSite(&self.tree, &self.field, block_idx, site_idx, psi_local, psi_ghosts, spacing);
        }

        pub fn reorder(self: *Self) !void {
            const perm = try self.tree.reorder();
            defer self.allocator.free(perm);
            try self.field.reorder(perm);
            self.ghosts_valid = false;
        }
        
        pub fn saveLinks(self: *const Self, allocator: std.mem.Allocator) ![][]Link {
            return self.field.saveLinks(allocator);
        }
        
        pub fn restoreLinks(self: *Self, backup: []const []const Link) void {
            self.field.restoreLinks(backup);
            self.ghosts_valid = false;
        }
        
        pub fn freeBackup(allocator: std.mem.Allocator, backup: [][]Link) void {
            Field.freeBackup(allocator, backup);
        }

        pub const CheckpointState = struct {
            tree: Self,
            arena: FieldArena,
        };

        pub fn writeCheckpoint(self: *const Self, arena: *const FieldArena, writer: anytype) !void {
            const checkpoint = @import("platform").checkpoint;
            try checkpoint.TreeCheckpoint(Tree).write(&self.tree, arena, writer);
            try self.field.writeCheckpoint(writer);
        }

        pub fn readCheckpoint(allocator: std.mem.Allocator, reader: anytype) !CheckpointState {
            return readCheckpointWithOptions(allocator, reader, .{});
        }
        
        pub const ReadOptions = struct {
             link_exchange_spec: ?LinkExchangeSpec = null,
        };

        pub fn readCheckpointWithOptions(
            allocator: std.mem.Allocator,
            reader: anytype,
            options: ReadOptions,
        ) !CheckpointState {
            const checkpoint = @import("platform").checkpoint;

            // 1. Read core AMR state
            const core_state = try checkpoint.TreeCheckpoint(Tree).read(allocator, reader);
            var amr_tree = core_state.tree;
            var arena = core_state.arena;

            errdefer {
                amr_tree.deinit();
                arena.deinit();
            }

            // 2. Read GaugeField
            // Requires reference to tree
            var field = try Field.readCheckpoint(allocator, &amr_tree, reader);
            
            // Apply options if any
             if (options.link_exchange_spec) |spec| {
                 // Re-init exchange with spec?
                 // Field.readCheckpoint uses default init (via syncWithTree logic implies init first).
                 // Field.readCheckpoint calls init.
                 // We should update Field.readCheckpoint to take options or update exchange after.
                 // Currently Field.readCheckpoint calls init(allocator, tree).
                 // We can manually replace link_exchange if needed, but for now ignore spec in readCheckpoint or update Field.
                 field.link_exchange.deinit();
                 field.link_exchange = Field.LinkExchange.init(allocator, spec);
             }

            return .{
                .tree = Self{
                    .allocator = allocator,
                    .tree = amr_tree,
                    .field = field,
                },
                .arena = arena,
            };
        }
    };
}

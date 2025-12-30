//! Gauge Tree: AMR Tree with Encapsulated Link Storage
//!
//! This module provides a high-level API for gauge theory on AMR meshes.
//! It encapsulates the AMR tree, link storage, and ghost layer management
//! so that users don't need to manually manage these components.
//!
//! ## Design Philosophy
//!
//! The gauge tree wraps the domain-agnostic AMR infrastructure and adds:
//! - Automatic link allocation when blocks are inserted
//! - Internal ghost layer management for cross-block operations
//! - Unified API for plaquette computation that handles boundaries
//!
//! ## Usage
//!
//! ```zig
//! const gauge = @import("gauge");
//!
//! const Frontend = gauge.GaugeFrontend(3, 1, 4, 16);  // SU(3), scalar, 4D, 16^4
//! var tree = try gauge.GaugeTree(Frontend).init(allocator, 1.0, 4, 8);
//! defer tree.deinit();
//!
//! // Insert blocks (links allocated automatically)
//! _ = try tree.insertBlock(.{0, 0, 0, 0}, 0);
//!
//! // Access links
//! const link = tree.getLink(0, site, mu);
//! tree.setLink(0, site, mu, new_link);
//!
//! // Compute plaquettes (ghost handling is internal)
//! try tree.fillGhosts();
//! const plaq = tree.computePlaquette(0, site, 0, 1);
//! ```

const std = @import("std");
const builtin = @import("builtin");
const amr = @import("amr");
const link_mod = @import("link.zig");
const dist_exchange_mod = amr.dist_exchange;
const ghost_policy_mod = @import("ghost_policy.zig");

pub const Complex = std.math.Complex(f64);

/// Gauge Tree with encapsulated link storage and ghost management.
///
/// Wraps AMRTree and provides automatic link allocation, ghost filling,
/// and plaquette computation with boundary handling.
pub fn GaugeTree(comptime Frontend: type) type {
    // Validate that this is a gauge frontend
    if (!@hasDecl(Frontend, "gauge_group_dim")) {
        @compileError("GaugeTree requires a GaugeFrontend with gauge_group_dim");
    }
    if (!@hasDecl(Frontend, "LinkType")) {
        @compileError("GaugeTree requires a GaugeFrontend with LinkType");
    }

    const Nd = Frontend.Nd;
    const block_size = Frontend.block_size;
    const N_gauge = Frontend.gauge_group_dim;
    const Link = Frontend.LinkType;

    const Tree = amr.AMRTree(Frontend);
    const Block = amr.AMRBlock(Frontend);
    
    // Number of links per block: volume * Nd
    const links_per_block = Block.volume * Nd;

    return struct {
        const Self = @This();

        const Policy = ghost_policy_mod.LinkGhostPolicy(Self);
        pub const LinkExchange = dist_exchange_mod.DistExchange(Tree, Policy.Context, Policy.Payload);
        pub const LinkExchangeSpec = Policy.ExchangeSpec;
        pub const ExchangeOptions = struct {
            tree_options: Tree.ExchangeOptions = .{},
            link_exchange_spec: ?LinkExchangeSpec = null,
        };

        pub const FrontendType = Frontend;
        pub const TreeType = Tree;
        pub const BlockType = Block;
        pub const LinkType = Link;
        pub const FieldType = Frontend.FieldType;
        pub const FieldArena = amr.FieldArena(Frontend);

        /// Derived constants
        pub const volume = Block.volume;
        pub const dimensions = Nd;
        pub const gauge_group_dim = N_gauge;
        pub const N_field = Frontend.field_dim;
        pub const num_faces = 2 * Nd;

        // =====================================================================
        // Ghost Storage for Plaquettes
        // =====================================================================

        /// Ghost storage for boundary links (tangential + normal).
        /// Each face stores links for every direction so boundary operators can
        /// access both tangential plaquettes and normal transport.
        pub const GhostStorage = struct {
            allocator: std.mem.Allocator,
            /// data[face_idx][link_dim]
            data: [2 * Nd][Nd][]Link,
            initialized: bool,

            pub fn init(allocator: std.mem.Allocator) GhostStorage {
                var storage: GhostStorage = undefined;
                storage.allocator = allocator;
                storage.initialized = false;
                for (0..2 * Nd) |f| {
                    for (0..Nd) |d| {
                        storage.data[f][d] = &[_]Link{};
                    }
                }
                return storage;
            }

            pub fn deinit(self: *GhostStorage) void {
                if (self.initialized) {
                    for (0..2 * Nd) |f| {
                        for (0..Nd) |d| {
                            if (self.data[f][d].len > 0) {
                                self.allocator.free(self.data[f][d]);
                            }
                        }
                    }
                }
            }

            pub fn ensureCapacity(self: *GhostStorage) !void {
                if (self.initialized) return;

                const face_size = Block.ghost_face_size;
                for (0..2 * Nd) |f| {
                    for (0..Nd) |d| {
                        self.data[f][d] = try self.allocator.alloc(Link, face_size);
                        for (self.data[f][d]) |*l| l.* = Link.identity();
                    }
                }
                self.initialized = true;
            }

            pub fn get(self: *const GhostStorage, face_idx: usize, link_dim: usize) []const Link {
                return self.data[face_idx][link_dim];
            }

            pub fn getMut(self: *GhostStorage, face_idx: usize, link_dim: usize) []Link {
                return self.data[face_idx][link_dim];
            }
        };

        // =====================================================================
        // Instance Fields
        // =====================================================================

        allocator: std.mem.Allocator,

        /// The underlying AMR tree
        tree: Tree,

        /// Link storage per block: links[block_idx][site * Nd + mu]
        links: std.ArrayList([]Link),

        /// Ghost storage per block
        ghosts: std.ArrayList(GhostStorage),

        /// MPI exchange for links
        link_exchange: LinkExchange,

        /// Whether ghosts are up to date
        ghosts_valid: bool,

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /// Initialize an empty gauge tree.
        pub fn init(
            allocator: std.mem.Allocator,
            base_spacing: f64,
            bits_per_dim: u8,
            max_level: u8,
        ) !Self {
            return initWithOptions(allocator, base_spacing, bits_per_dim, max_level, .{});
        }

        /// Initialize a gauge tree with custom exchange specifications.
        pub fn initWithOptions(
            allocator: std.mem.Allocator,
            base_spacing: f64,
            bits_per_dim: u8,
            max_level: u8,
            options: ExchangeOptions,
        ) !Self {
            const exchange_spec = options.link_exchange_spec orelse Policy.exchangeSpec();
            return Self{
                .allocator = allocator,
                .tree = try Tree.initWithOptions(allocator, base_spacing, bits_per_dim, max_level, options.tree_options),
                .links = std.ArrayList([]Link){},
                .ghosts = std.ArrayList(GhostStorage){},
                .link_exchange = LinkExchange.init(allocator, exchange_spec),
                .ghosts_valid = false,
            };
        }

        pub fn deinit(self: *Self) void {
            self.link_exchange.deinit();
            self.clearThreadScratch();
            
            // Free link storage
            for (self.links.items) |link_slice| {
                self.allocator.free(link_slice);
            }
            self.links.deinit(self.allocator);

            // Free ghost storage
            for (self.ghosts.items) |*ghost| {
                ghost.deinit();
            }
            self.ghosts.deinit(self.allocator);

            // Free tree
            self.tree.deinit();
        }


        pub fn attachShard(self: *Self, shard: *Tree.ShardContext) void {
            self.tree.attachShard(shard);
        }

        pub fn detachShard(self: *Self) void {
            self.tree.detachShard();
        }

        pub fn shardContext(self: *const Self) ?*Tree.ShardContext {
            return self.tree.shardContext();
        }

        // =====================================================================
        // Block Management
        // =====================================================================

        /// Insert a block at the given origin and level.
        /// Automatically allocates link storage.
        pub fn insertBlock(self: *Self, origin: [Nd]usize, level: u8) !usize {
            // Insert into AMR tree
            const block_idx = try self.tree.insertBlock(origin, level);

            // Ensure we have link storage for this block index
            while (self.links.items.len <= block_idx) {
                const link_slice = try self.allocator.alloc(Link, links_per_block);
                for (link_slice) |*l| l.* = Link.identity();
                try self.links.append(self.allocator, link_slice);

                const ghost = GhostStorage.init(self.allocator);
                try self.ghosts.append(self.allocator, ghost);
            }

            self.ghosts_valid = false;
            return block_idx;
        }

        /// Get the number of blocks in the tree
        pub fn blockCount(self: *const Self) usize {
            return self.tree.blocks.items.len;
        }

        /// Get a block by index
        pub fn getBlock(self: *const Self, block_idx: usize) ?*const Block {
            return self.tree.getBlock(block_idx);
        }

        // =====================================================================
        // Link Access
        // =====================================================================

        /// Get link U_mu(site) in a block
        pub fn getLink(self: *const Self, block_idx: usize, site: usize, mu: usize) Link {
            const idx = site * Nd + mu;
            if (block_idx < self.links.items.len and idx < self.links.items[block_idx].len) {
                return self.links.items[block_idx][idx];
            }
            return Link.identity();
        }

        /// Set link U_mu(site) in a block
        pub fn setLink(self: *Self, block_idx: usize, site: usize, mu: usize, link: Link) void {
            const idx = site * Nd + mu;
            if (block_idx < self.links.items.len and idx < self.links.items[block_idx].len) {
                self.links.items[block_idx][idx] = link;
                self.ghosts_valid = false;
            }
        }

        /// Get mutable slice of all links for a block.
        /// Marks ghost caches dirty because callers may mutate links directly.
        pub fn getBlockLinksMut(self: *Self, block_idx: usize) ?[]Link {
            if (block_idx < self.links.items.len) {
                self.ghosts_valid = false;
                return self.links.items[block_idx];
            }
            return null;
        }

        /// Get const slice of all links for a block
        pub fn getBlockLinksConst(self: *const Self, block_idx: usize) ?[]const Link {
            if (block_idx < self.links.items.len) {
                return self.links.items[block_idx];
            }
            return null;
        }

        // =====================================================================
        // Ghost Layer Management
        // =====================================================================

        /// Fill ghost layers for all blocks.
        /// Must be called before computing plaquettes at boundaries.
        pub fn fillGhosts(self: *Self) !void {
            if (self.ghosts_valid) return;

            for (self.ghosts.items) |*ghost| {
                try ghost.ensureCapacity();
            }

            const shard = self.shardContext();
            var dist_state = try self.beginExchange(shard);
            try self.finishExchange(&dist_state);

            self.ghosts_valid = true;
        }

        pub fn beginExchange(self: *Self, shard: ?*Tree.ShardContext) !LinkExchange.ExchangeState {
            return try self.link_exchange.begin(.{ .tree = self }, shard);
        }

        pub fn finishExchange(self: *Self, state: *LinkExchange.ExchangeState) !void {
            try self.link_exchange.finish(.{ .tree = self }, state);
        }

        pub fn fillGhostsPull(self: *Self, block_idx: usize) void {
            _ = self;
            _ = block_idx;
        }

        pub fn fillGhostsPush(self: *Self, block_idx: usize) void {
            _ = self;
            _ = block_idx;
        }

        fn clearGhostFace(self: *Self, block_idx: usize, face: usize, value: Link) void {
            if (block_idx >= self.ghosts.items.len) return;
            const ghost = &self.ghosts.items[block_idx];
            inline for (0..Nd) |link_dim| {
                const slice = ghost.getMut(face, link_dim);
                for (slice) |*l| l.* = value;
            }
        }

        // Per-thread arena scratch to avoid per-call heap churn.
        fn threadScratchAllocator(self: *Self) std.mem.Allocator {
            const allocator = self.allocator;
            if (!tls_scratch_initialized or tls_scratch_allocator.ptr != allocator.ptr or tls_scratch_allocator.vtable != allocator.vtable) {
                if (tls_scratch_initialized) {
                    tls_scratch.deinit();
                }
                tls_scratch = std.heap.ArenaAllocator.init(allocator);
                tls_scratch_initialized = true;
                tls_scratch_allocator = allocator;
            }
            _ = tls_scratch.reset(.retain_capacity);
            return tls_scratch.allocator();
        }

        fn threadScratchRelease(self: *Self) void {
            const allocator = self.allocator;
            if (!tls_scratch_initialized) return;
            if (comptime builtin.is_test) {
                if (allocator.ptr == std.testing.allocator.ptr and allocator.vtable == std.testing.allocator.vtable) {
                    _ = tls_scratch.reset(.free_all);
                }
            }
        }

        fn clearThreadScratch(self: *Self) void {
            _ = self;
            if (tls_scratch_initialized) {
                tls_scratch.deinit();
                tls_scratch_initialized = false;
            }
        }

        threadlocal var tls_scratch: std.heap.ArenaAllocator = undefined;
        threadlocal var tls_scratch_allocator: std.mem.Allocator = undefined;
        threadlocal var tls_scratch_initialized: bool = false;

        pub fn prepareLinkGhostExchange(self: *Self) !bool {
            if (self.ghosts_valid) return false;
            for (self.ghosts.items) |*ghost| {
                try ghost.ensureCapacity();
            }
            return true;
        }

        pub fn finalizeLinkGhostExchange(self: *Self) void {
            self.ghosts_valid = true;
        }

        // =====================================================================
        // Plaquette Computation
        // =====================================================================

        /// Compute plaquette U_μν at site in a block.
        /// Automatically uses ghost data for boundary sites.
        /// Call fillGhosts() before using this.
        pub fn computePlaquette(
            self: *const Self,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
            comptime nu: usize,
        ) Link {
            if (block_idx >= self.links.items.len) return Link.identity();

            const links = self.links.items[block_idx];
            const coords = Block.getLocalCoords(site_idx);

            const at_mu_boundary = (coords[mu] == block_size - 1);
            const at_nu_boundary = (coords[nu] == block_size - 1);

            if (!at_mu_boundary and !at_nu_boundary) {
                // Interior: use gauge module
                // Create a temporary view and delegate
                const link1 = getLinkLocal(links, site_idx, mu);
                const site_plus_mu = Block.localNeighborFast(site_idx, mu * 2);
                const link2 = getLinkLocal(links, site_plus_mu, nu);
                const site_plus_nu = Block.localNeighborFast(site_idx, nu * 2);
                const link3_dag = getLinkLocal(links, site_plus_nu, mu).adjoint();
                const link4_dag = getLinkLocal(links, site_idx, nu).adjoint();
                return link1.mul(link2).mul(link3_dag).mul(link4_dag);
            }

            // Boundary: use ghosts
            if (block_idx >= self.ghosts.items.len) return Link.identity();
            const ghosts = &self.ghosts.items[block_idx];

            const link1 = getLinkLocal(links, site_idx, mu);

            var link2: Link = undefined;
            if (at_mu_boundary) {
                const face_idx = mu * 2;
                const ghost_idx = Block.getGhostIndex(coords, face_idx);
                link2 = ghosts.get(face_idx, nu)[ghost_idx];
            } else {
                const neighbor = Block.localNeighborFast(site_idx, mu * 2);
                link2 = getLinkLocal(links, neighbor, nu);
            }

            var link3_dag: Link = undefined;
            if (at_nu_boundary) {
                const face_idx = nu * 2;
                const ghost_idx = Block.getGhostIndex(coords, face_idx);
                link3_dag = ghosts.get(face_idx, mu)[ghost_idx].adjoint();
            } else {
                const neighbor = Block.localNeighborFast(site_idx, nu * 2);
                link3_dag = getLinkLocal(links, neighbor, mu).adjoint();
            }

            const link4_dag = getLinkLocal(links, site_idx, nu).adjoint();

            return link1.mul(link2).mul(link3_dag).mul(link4_dag);
        }

        /// Compute trace of plaquette (gauge-invariant scalar)
        pub fn tracePlaquette(
            self: *const Self,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
            comptime nu: usize,
        ) f64 {
            return self.computePlaquette(block_idx, site_idx, mu, nu).trace().re;
        }

        /// Number of plaquette pairs in Nd dimensions: Nd*(Nd-1)/2
        const num_plaquettes: usize = Nd * (Nd - 1) / 2;

        /// Compute average plaquette over a block (generic for any Nd >= 2)
        pub fn averagePlaquetteBlock(self: *const Self, block_idx: usize) f64 {
            comptime {
                if (Nd < 2) @compileError("averagePlaquetteBlock requires Nd >= 2");
            }

            var sum: f64 = 0.0;
            const n_real: f64 = @floatFromInt(N_gauge);

            for (0..volume) |site| {
                // Iterate over all (mu, nu) pairs where mu < nu
                inline for (0..Nd) |mu| {
                    inline for ((mu + 1)..Nd) |nu| {
                        sum += self.tracePlaquette(block_idx, site, mu, nu);
                    }
                }
            }

            const volume_f: f64 = @floatFromInt(volume);
            const num_plaq_f: f64 = @floatFromInt(num_plaquettes);
            return sum / (n_real * volume_f * num_plaq_f);
        }

        /// Compute average plaquette over entire tree
        pub fn averagePlaquetteTree(self: *const Self) f64 {
            var sum: f64 = 0.0;
            var count: usize = 0;

            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    sum += self.averagePlaquetteBlock(idx);
                    count += 1;
                }
            }

            if (count == 0) return 1.0;
            return sum / @as(f64, @floatFromInt(count));
        }

        /// Compute Wilson action for entire tree
        pub fn wilsonAction(self: *const Self, beta: f64) f64 {
            var total_action: f64 = 0.0;
            const num_plaq_f: f64 = @floatFromInt(num_plaquettes);

            for (self.tree.blocks.items, 0..) |*block, idx| {
                if (block.block_index != std.math.maxInt(usize)) {
                    const avg_plaq = self.averagePlaquetteBlock(idx);
                    const n_real: f64 = @floatFromInt(N_gauge);
                    const volume_f: f64 = @floatFromInt(volume);
                    total_action += beta * n_real * volume_f * num_plaq_f * (1.0 - avg_plaq);
                }
            }

            return total_action;
        }

        // =====================================================================
        // Force Calculation Support
        // =====================================================================

        /// Get a ghost link for boundary operations.
        /// face_idx: which face (0-7 for 4D)
        /// link_dim: which link direction (0..Nd-1)
        /// ghost_idx: index within the ghost face
        pub fn getGhostLink(self: *const Self, block_idx: usize, face_idx: usize, link_dim: usize, ghost_idx: usize) Link {
            if (block_idx >= self.ghosts.items.len) return Link.identity();
            const ghost = &self.ghosts.items[block_idx];
            if (!ghost.initialized) return Link.identity();
            const slice = ghost.data[face_idx][link_dim];
            if (ghost_idx < slice.len) return slice[ghost_idx];
            return Link.identity();
        }

        /// Compute staple for link U_mu(site) - sum of plaquettes containing this link.
        /// Used for force calculation in HMC.
        pub fn computeStaple(
            self: *const Self,
            block_idx: usize,
            site_idx: usize,
            comptime mu: usize,
        ) Link {
            if (block_idx >= self.links.items.len) return Link.zero();

            const links = self.links.items[block_idx];
            const coords = Block.getLocalCoords(site_idx);
            var staple = Link.zero();

            inline for (0..Nd) |nu| {
                if (nu == mu) continue;

                const at_mu_boundary = (coords[mu] == block_size - 1);
                const at_nu_boundary = (coords[nu] == block_size - 1);

                // Forward staple: U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag
                const link_nu_xmu = if (at_mu_boundary) blk: {
                    const face_idx = mu * 2;
                    const ghost_idx = Block.getGhostIndex(coords, face_idx);
                    break :blk self.getGhostLink(block_idx, face_idx, nu, ghost_idx);
                } else blk: {
                    const neighbor = Block.localNeighborFast(site_idx, mu * 2);
                    break :blk getLinkLocal(links, neighbor, nu);
                };

                const link_mu_xnu = if (at_nu_boundary) blk: {
                    const face_idx = nu * 2;
                    const ghost_idx = Block.getGhostIndex(coords, face_idx);
                    break :blk self.getGhostLink(block_idx, face_idx, mu, ghost_idx);
                } else blk: {
                    const neighbor = Block.localNeighborFast(site_idx, nu * 2);
                    break :blk getLinkLocal(links, neighbor, mu);
                };

                const link_nu_x = getLinkLocal(links, site_idx, nu);
                const forward = link_nu_xmu.mul(link_mu_xnu.adjoint()).mul(link_nu_x.adjoint());
                staple = staple.add(forward);

                // Backward staple: U_nu(x+mu-nu)^dag U_mu(x-nu)^dag U_nu(x-nu)
                if (coords[nu] > 0) {
                    const site_minus_nu = Block.localNeighborFast(site_idx, nu * 2 + 1);
                    const link_nu_xmnu = getLinkLocal(links, site_minus_nu, nu);
                    const link_mu_xmnu = getLinkLocal(links, site_minus_nu, mu);

                    const link_nu_xmu_mnu = if (at_mu_boundary) blk: {
                        var ghost_coords = coords;
                        ghost_coords[nu] = coords[nu] - 1;
                        const face_idx = mu * 2;
                        const ghost_idx = Block.getGhostIndex(ghost_coords, face_idx);
                        break :blk self.getGhostLink(block_idx, face_idx, nu, ghost_idx);
                    } else blk: {
                        const site_mu_mnu = Block.localNeighborFast(site_minus_nu, mu * 2);
                        break :blk getLinkLocal(links, site_mu_mnu, nu);
                    };

                    const backward = link_nu_xmu_mnu.adjoint().mul(link_mu_xmnu.adjoint()).mul(link_nu_xmnu);
                    staple = staple.add(backward);
                }
            }

            return staple;
        }

        /// Compute staple with runtime mu parameter
        pub fn computeStapleRuntime(
            self: *const Self,
            block_idx: usize,
            site_idx: usize,
            mu: usize,
        ) Link {
            return switch (mu) {
                0 => self.computeStaple(block_idx, site_idx, 0),
                1 => self.computeStaple(block_idx, site_idx, 1),
                2 => self.computeStaple(block_idx, site_idx, 2),
                3 => self.computeStaple(block_idx, site_idx, 3),
                else => Link.zero(),
            };
        }

        /// Save current link configuration for backup (HMC)
        pub fn saveLinks(self: *const Self, allocator: std.mem.Allocator) ![][]Link {
            var backup = try allocator.alloc([]Link, self.links.items.len);
            for (self.links.items, 0..) |link_slice, i| {
                backup[i] = try allocator.alloc(Link, link_slice.len);
                @memcpy(backup[i], link_slice);
            }
            return backup;
        }

        /// Restore link configuration from backup (HMC)
        pub fn restoreLinks(self: *Self, backup: []const []const Link) void {
            for (backup, 0..) |link_slice, i| {
                if (i < self.links.items.len and link_slice.len == self.links.items[i].len) {
                    @memcpy(self.links.items[i], link_slice);
                }
            }
            self.ghosts_valid = false;
        }

        /// Free saved link backup
        pub fn freeBackup(allocator: std.mem.Allocator, backup: [][]Link) void {
            for (backup) |slice| {
                allocator.free(slice);
            }
            allocator.free(backup);
        }

        /// Iterate over all active blocks
        pub fn blockIterator(self: *const Self) Tree.BlockIterator {
            return self.tree.blockIterator();
        }

        /// Insert block with field allocation (convenience wrapper)
        pub fn insertBlockWithField(self: *Self, origin: [Nd]usize, level: u8, arena: anytype) !usize {
            const block_idx = try self.insertBlock(origin, level);
            const field_slot = arena.allocSlot() orelse return error.FieldArenaFull;
            self.tree.assignFieldSlot(block_idx, field_slot);
            return block_idx;
        }

        // =====================================================================
        // Reordering (Morton Sort)
        // =====================================================================

        /// Reorder blocks by Morton index for cache locality.
        ///
        /// This reorders the underlying AMRTree AND permutes the GaugeTree's
        /// link and ghost arrays to maintain consistency.
        ///
        /// **IMPORTANT**: This invalidates all external pointers to blocks,
        /// field data, and link slices. After calling reorder(), any cached
        /// references obtained before the call are invalid.
        ///
        /// Call this after mesh adaptation to restore cache locality.
        pub fn reorder(self: *Self) !void {
            const active_count = self.tree.blockCount();
            if (active_count == 0) {
                const perm = try self.tree.reorder();
                self.allocator.free(perm);
                return;
            }

            // Reorder the underlying tree and get permutation map
            const perm = try self.tree.reorder();
            defer self.allocator.free(perm);

            const invalid = std.math.maxInt(usize);

            // Permute links array
            var new_links = try self.allocator.alloc([]Link, active_count);
            // We need to iterate over the permutation map to fill new_links
            // perm[old_idx] = new_idx
            // So if perm[old_idx] != invalid, new_links[perm[old_idx]] = links[old_idx]

            // Initialize new_links to empty slices just in case (though we should cover all)
            for (new_links) |*slice| slice.* = &[_]Link{};

            for (perm, 0..) |new_idx, old_idx| {
                if (new_idx != invalid) {
                    if (old_idx < self.links.items.len) {
                        new_links[new_idx] = self.links.items[old_idx];
                    } else {
                        // This shouldn't happen if state is consistent, but handle it
                        new_links[new_idx] = try self.allocator.alloc(Link, links_per_block);
                        for (new_links[new_idx]) |*l| l.* = Link.identity();
                    }
                } else {
                    // Old block deleted - free its links if needed?
                    // The ArrayList deinit frees the slices, but here we are moving ownership.
                    // If a block is deleted, we should free its links.
                    if (old_idx < self.links.items.len) {
                        self.allocator.free(self.links.items[old_idx]);
                    }
                }
            }

            // Replace old links array
            self.links.clearRetainingCapacity();
            try self.links.appendSlice(self.allocator, new_links);
            self.allocator.free(new_links);

            // Permute ghosts array
            var new_ghosts = try self.allocator.alloc(GhostStorage, active_count);
            // Initialize new_ghosts to safe state
            for (new_ghosts) |*g| g.* = GhostStorage.init(self.allocator);

            for (perm, 0..) |new_idx, old_idx| {
                if (new_idx != invalid) {
                    if (old_idx < self.ghosts.items.len) {
                        // Move ownership of ghost storage
                        // We must deinit the dummy initialized above to prevent leak
                        // Actually GhostStorage.init doesn't alloc, so it's fine.
                        new_ghosts[new_idx] = self.ghosts.items[old_idx];
                    }
                } else {
                    // Old block deleted - free ghost storage
                    if (old_idx < self.ghosts.items.len) {
                        self.ghosts.items[old_idx].deinit();
                    }
                }
            }

            // Replace old ghosts array
            self.ghosts.clearRetainingCapacity();
            try self.ghosts.appendSlice(self.allocator, new_ghosts);
            self.allocator.free(new_ghosts);

            // Invalidate ghost cache
            self.ghosts_valid = false;
        }

        // =====================================================================
        // Checkpoint/Restart
        // =====================================================================

        /// Write a checkpoint of the GaugeTree state (AMR + Links).
        /// Appends link data after the standard AMR checkpoint.
        pub fn writeCheckpoint(self: *const Self, arena: *const FieldArena, writer: anytype) !void {
            const checkpoint = @import("platform").checkpoint;
            
            // 1. Write core AMR state
            try checkpoint.TreeCheckpoint(Tree).write(&self.tree, arena, writer);

            // 2. Write Link Data Header
            const links_magic = "LINK";
            try writer.writeAll(links_magic);
            
            const num_blocks = self.links.items.len;
            try writer.writeInt(u64, @as(u64, @intCast(num_blocks)), .little);
            
            const links_len = links_per_block;
            try writer.writeInt(u64, @as(u64, @intCast(links_len)), .little);

            // 3. Write Links
            for (self.links.items) |slice| {
                if (slice.len != links_len) return error.InvalidLinkLayout;
                try writer.writeAll(std.mem.sliceAsBytes(slice));
            }
        }

        pub const CheckpointState = struct {
            tree: Self,
            arena: FieldArena,
        };

        pub const ReadOptions = struct {
            link_exchange_spec: ?LinkExchangeSpec = null,
        };

        /// Read a checkpoint and reconstruct the GaugeTree.
        pub fn readCheckpoint(allocator: std.mem.Allocator, reader: anytype) !CheckpointState {
            return readCheckpointWithOptions(allocator, reader, .{});
        }

        /// Read a checkpoint with a custom link exchange specification.
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

            // 2. Read Link Data Header
            var magic_buf: [4]u8 = undefined;
            try reader.readNoEof(&magic_buf);
            if (!std.mem.eql(u8, &magic_buf, "LINK")) return error.InvalidMagic;

            const num_blocks = try reader.readInt(u64, .little);
            const stored_links_len = try reader.readInt(u64, .little);

            if (num_blocks != amr_tree.blocks.items.len) return error.IncompatibleCheckpoint;
            if (stored_links_len != links_per_block) return error.IncompatibleCheckpoint;

            // 3. Read Links
            var links = try std.ArrayList([]Link).initCapacity(allocator, @intCast(num_blocks));
            errdefer {
                for (links.items) |slice| allocator.free(slice);
                links.deinit(allocator);
            }

            for (0..num_blocks) |_| {
                const slice = try allocator.alloc(Link, @intCast(stored_links_len));
                try reader.readNoEof(std.mem.sliceAsBytes(slice));
                links.appendAssumeCapacity(slice);
            }

            // 4. Construct GaugeTree
            const exchange_spec = options.link_exchange_spec orelse Policy.exchangeSpec();
            var gauge_tree = Self{
                .allocator = allocator,
                .tree = amr_tree,
                .links = links,
                .ghosts = std.ArrayList(GhostStorage){},
                .link_exchange = LinkExchange.init(allocator, exchange_spec),
                .ghosts_valid = false,
            };

            // Initialize ghosts (empty but allocated array)
            try gauge_tree.ghosts.resize(allocator, @intCast(num_blocks));
            for (gauge_tree.ghosts.items) |*ghost| {
                ghost.* = GhostStorage.init(allocator);
            }

            return .{
                .tree = gauge_tree,
                .arena = arena,
            };
        }

        // =====================================================================
        // Gauge-Covariant Laplacian
        // =====================================================================

        /// Apply gauge-covariant Laplacian to a matter field at a single site.
        /// Returns ∇²ψ(x) = Σ_μ [U_μ(x)ψ(x+μ) + U†_μ(x-μ)ψ(x-μ) - 2ψ(x)] / a²
        ///
        /// Parameters:
        /// - block_idx: Index of the block
        /// - site_idx: Site index within the block
        /// - psi_local: Matter field values for this block
        /// - psi_ghosts: Ghost values for each face [2*Nd][ghost_face_size]
        /// - spacing: Lattice spacing for this block
        pub fn covariantLaplacianSite(
            self: *const Self,
            block_idx: usize,
            site_idx: usize,
            psi_local: []const FieldType,
            psi_ghosts: [num_faces][]const FieldType,
            spacing: f64,
        ) FieldType {
            const coords = Block.getLocalCoords(site_idx);
            const inv_a_sq = 1.0 / (spacing * spacing);

            // Initialize with -2*Nd*ψ(x) diagonal term
            var result: FieldType = undefined;
            const center_factor = -@as(f64, @floatFromInt(2 * Nd));
            inline for (0..N_field) |a| {
                result[a] = psi_local[site_idx][a].mul(Complex.init(center_factor, 0));
            }

            inline for (0..Nd) |mu| {
                const face_plus = mu * 2;
                const face_minus = mu * 2 + 1;

                // Forward transport: U_μ(x) ψ(x+μ)
                const link_fwd = self.getLink(block_idx, site_idx, mu);
                var psi_plus: FieldType = undefined;

                if (Block.isOnBoundary(coords) and coords[mu] == block_size - 1) {
                    const ghost_idx = Block.getGhostIndex(coords, face_plus);
                    if (ghost_idx < psi_ghosts[face_plus].len) {
                        psi_plus = psi_ghosts[face_plus][ghost_idx];
                    } else {
                        psi_plus = Frontend.zeroField();
                    }
                } else {
                    const neighbor_plus = Block.localNeighborFast(site_idx, face_plus);
                    psi_plus = psi_local[neighbor_plus];
                }

                const transported_fwd = Frontend.applyLinkToField(link_fwd, psi_plus);

                // Backward transport: U†_μ(x-μ) ψ(x-μ)
                var link_bwd: Link = undefined;
                var psi_minus: FieldType = undefined;

                if (Block.isOnBoundary(coords) and coords[mu] == 0) {
                    const ghost_idx = Block.getGhostIndex(coords, face_minus);
                    if (ghost_idx < psi_ghosts[face_minus].len) {
                        psi_minus = psi_ghosts[face_minus][ghost_idx];
                    } else {
                        psi_minus = Frontend.zeroField();
                    }
                    // Get link from ghost storage
                    link_bwd = self.getGhostLink(block_idx, face_minus, mu, ghost_idx).adjoint();
                } else {
                    const neighbor_minus = Block.localNeighborFast(site_idx, face_minus);
                    psi_minus = psi_local[neighbor_minus];
                    link_bwd = self.getLink(block_idx, neighbor_minus, mu).adjoint();
                }

                const transported_bwd = Frontend.applyLinkToField(link_bwd, psi_minus);

                // Accumulate
                inline for (0..N_field) |a| {
                    result[a] = result[a].add(transported_fwd[a]).add(transported_bwd[a]);
                }
            }

            // Scale by 1/a²
            inline for (0..N_field) |a| {
                result[a] = result[a].mul(Complex.init(inv_a_sq, 0));
            }

            return result;
        }

        /// Apply gauge-covariant Laplacian to an entire block.
        pub fn applyCovariantLaplacianBlock(
            self: *const Self,
            block_idx: usize,
            psi_in: []const FieldType,
            psi_out: []FieldType,
            psi_ghosts: [num_faces][]const FieldType,
            spacing: f64,
        ) void {
            for (0..volume) |site| {
                psi_out[site] = self.covariantLaplacianSite(block_idx, site, psi_in, psi_ghosts, spacing);
            }
        }

        // =====================================================================
        // Helper Functions
        // =====================================================================

        inline fn getLinkLocal(links: []const Link, site: usize, mu: usize) Link {
            const idx = site * Nd + mu;
            if (idx < links.len) {
                return links[idx];
            }
            return Link.identity();
        }

        fn extractBoundaryLinkFace(
            links: []const Link,
            face: usize,
            link_dim: usize,
            dest: []Link,
        ) void {
            const face_dim = face / 2;
            const is_positive = (face % 2) == 0;
            const boundary_coord = if (is_positive) block_size - 1 else 0;

            var dest_idx: usize = 0;
            var coords: [Nd]usize = .{0} ** Nd;
            coords[face_dim] = boundary_coord;

            for (0..Block.ghost_face_size) |_| {
                const site_idx = Block.getLocalIndex(coords);
                dest[dest_idx] = getLinkLocal(links, site_idx, link_dim);
                dest_idx += 1;

                for (0..Nd) |d| {
                    if (d == face_dim) continue;
                    coords[d] += 1;
                    if (coords[d] < block_size) break;
                    coords[d] = 0;
                }
            }
        }

        fn incrementCoords(coords: *[Nd - 1]usize, max: usize) void {
            for (0..Nd - 1) |k| {
                coords[k] += 1;
                if (coords[k] < max) break;
                coords[k] = 0;
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const topology = amr.topology;
const TestTopology4D = topology.PeriodicTopology(4, .{ 1.0, 1.0, 1.0, 1.0 });

test "GaugeTree - basic initialization" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D); // U(1), scalar, 4D, 4^4
    const GT = GaugeTree(Frontend);

    var tree = try GT.init(std.testing.allocator, 1.0, 4);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 0), tree.blockCount());
}

test "GaugeTree - insert block allocates links" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const GT = GaugeTree(Frontend);

    var tree = try GT.init(std.testing.allocator, 1.0, 4);
    defer tree.deinit();

    const idx = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try std.testing.expectEqual(@as(usize, 0), idx);
    try std.testing.expectEqual(@as(usize, 1), tree.links.items.len);

    // Links should be identity
    const link = tree.getLink(0, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), link.trace().re, 1e-10);
}

test "GaugeTree - plaquette of identity is identity" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const GT = GaugeTree(Frontend);

    var tree = try GT.init(std.testing.allocator, 1.0, 4);
    defer tree.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.fillGhosts();

    // Interior plaquette
    const plaq = tree.computePlaquette(0, 5, 0, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), plaq.trace().re, 1e-10);

    // Average plaquette
    const avg = tree.averagePlaquetteBlock(0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), avg, 1e-10);
}

test "GaugeTree - Wilson action zero for identity" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const GT = GaugeTree(Frontend);

    var tree = try GT.init(std.testing.allocator, 1.0, 4);
    defer tree.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);
    try tree.fillGhosts();

    const action = tree.wilsonAction(1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), action, 1e-10);
}

test "GaugeTree - set and get link" {
    const frontend = @import("frontend.zig");
    const Frontend = frontend.GaugeFrontend(1, 1, 4, 4, TestTopology4D);
    const GT = GaugeTree(Frontend);

    var tree = try GT.init(std.testing.allocator, 1.0, 4);
    defer tree.deinit();

    _ = try tree.insertBlock(.{ 0, 0, 0, 0 }, 0);

    // Set a non-identity link
    var link = GT.LinkType.zero();
    link.matrix.data[0][0] = Complex.init(0, 1); // exp(iπ/2) = i
    tree.setLink(0, 10, 0, link);

    // Read it back
    const read_link = tree.getLink(0, 10, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), read_link.trace().re, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), read_link.trace().im, 1e-10);
}

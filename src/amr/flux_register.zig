//! Flux Register for AMR Conservation.
//!
//! Maintains conservation at Coarse-Fine boundaries by accumulating the flux mismatch
//! between levels.
//!
//! ## Theory
//! Conservation law: dU/dt + div(F) = 0
//! - Coarse update: U_crse += -dt/dx * F_crse
//! - Fine update:   U_fine += -dt/dx * F_fine
//!
//! At the C-F interface, the flux `F_crse` used to update the coarse cell might differ
//! from the average of `F_fine` used to update the fine cells.
//! This destroys conservation.
//!
//! The Flux Register stores per-face-cell flux integrals:
//! `R = -F_crse * dt * Area_crse_cell`, then `R += sum(F_fine * dt * Area_fine_cell)`.
//! Finally, `reflux` modifies the coarse face cells: `U_crse += R / Volume_crse`.

const std = @import("std");
const field_math = @import("field_math.zig");

var flux_register_cookie: std.atomic.Value(u64) = std.atomic.Value(u64).init(1);

pub fn FluxRegister(comptime Tree: type) type {
    const FieldType = Tree.FieldType;
    const BlockKey = Tree.BlockKey;
    const Block = Tree.BlockType;
    const face_cells = Block.ghost_face_size;
    const FaceField = [face_cells]FieldType;
    const ThreadKey = std.Thread.Id;
    const half_block = Block.size / 2;
    const Shift = std.math.Log2Int(usize);

    comptime {
        if (Block.size % 2 != 0) {
            @compileError("FluxRegister requires even block_size for 2:1 refinement mapping.");
        }
    }

    const fine_to_coarse_face_idx = blk: {
        const Builder = struct {
            fn build() [Tree.dimensions][1 << (Tree.dimensions - 1)][face_cells]u32 {
                @setEvalBranchQuota(2000000);
                const mask_count: usize = 1 << (Tree.dimensions - 1);
                var map: [Tree.dimensions][mask_count][face_cells]u32 = undefined;

                for (0..Tree.dimensions) |dim| {
                    for (0..mask_count) |mask| {
                        for (0..face_cells) |fine_face_idx| {
                            var coords: [Tree.dimensions]usize = undefined;
                            var rem = fine_face_idx;
                            for (0..Tree.dimensions) |d| {
                                if (d == dim) {
                                    coords[d] = 0;
                                } else {
                                    coords[d] = rem % Block.size;
                                    rem /= Block.size;
                                }
                            }

                            var coarse_coords: [Tree.dimensions]usize = coords;
                            var mask_idx: usize = 0;
                            for (0..Tree.dimensions) |d| {
                                if (d == dim) continue;
                                const shift: Shift = @intCast(mask_idx);
                                const half: usize = (mask >> shift) & 1;
                                coarse_coords[d] = (coords[d] >> 1) + (half_block * half);
                                mask_idx += 1;
                            }

                            const coarse_face_idx = Block.getGhostIndexRuntime(coarse_coords, dim * 2);
                            map[dim][mask][fine_face_idx] = @intCast(coarse_face_idx);
                        }
                    }
                }

                return map;
            }
        };

        break :blk Builder.build();
    };

    // Key represents a specific face of a COARSE block.
    // Fine fluxes map to this key by computing their parent's key.
    const FaceKey = struct {
        block_key: BlockKey,
        face: u8,
    };

    return struct {
        const Self = @This();
        const Arena = Tree.FieldArenaType;

        const RegisterMap = std.AutoHashMap(FaceKey, FaceField);
        const LocalRegister = struct {
            registers: RegisterMap,
        };

        const RegisterEntry = RegisterMap.Entry;
        const phase_idle: u8 = 0;
        const phase_reflux: u8 = 1;
        const phase_clear: u8 = 2;

        threadlocal var tls_owner: ?*const Self = null;
        threadlocal var tls_cookie: u64 = 0;
        threadlocal var tls_local: ?*LocalRegister = null;

        allocator: std.mem.Allocator,
        mutex: std.Thread.Mutex = .{},
        locals: std.AutoHashMap(ThreadKey, *LocalRegister),
        phase: std.atomic.Value(u8) = std.atomic.Value(u8).init(phase_idle),
        cookie: u64,
        reserve_hint: RegisterMap.Size = 0,
        no_alloc: bool = false,

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
                .locals = std.AutoHashMap(ThreadKey, *LocalRegister).init(allocator),
                .cookie = flux_register_cookie.fetchAdd(1, .monotonic),
            };
        }

        pub fn deinit(self: *Self) void {
            var iter = self.locals.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.*.registers.deinit();
                self.allocator.destroy(entry.value_ptr.*);
            }
            self.locals.deinit();
        }

        pub fn clear(self: *Self) void {
            self.beginPhase(phase_clear);
            defer self.endPhase();
            self.mutex.lock();
            defer self.mutex.unlock();
            var iter = self.locals.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.*.registers.clearRetainingCapacity();
            }
        }

        pub fn setNoAlloc(self: *Self, enabled: bool) void {
            if (std.debug.runtime_safety) {
                std.debug.assert(self.phase.load(.acquire) == phase_idle);
            }
            self.no_alloc = enabled;
        }

        pub fn clearAndReserve(self: *Self, reserve_hint: usize) !void {
            const hint = clampReserveHint(reserve_hint);
            self.beginPhase(phase_clear);
            defer self.endPhase();
            self.mutex.lock();
            defer self.mutex.unlock();

            if (hint > self.reserve_hint) {
                self.reserve_hint = hint;
            }

            var iter = self.locals.iterator();
            while (iter.next()) |entry| {
                if (self.reserve_hint > 0) {
                    try entry.value_ptr.*.registers.ensureTotalCapacity(self.reserve_hint);
                }
                entry.value_ptr.*.registers.clearRetainingCapacity();
            }
        }

        pub fn prepare(self: *Self, reserve_hint: usize) !void {
            const hint = clampReserveHint(reserve_hint);
            if (hint == 0) return;
            if (std.debug.runtime_safety) {
                std.debug.assert(self.phase.load(.acquire) == phase_idle);
            }

            self.mutex.lock();
            defer self.mutex.unlock();
            if (hint <= self.reserve_hint) return;
            self.reserve_hint = hint;
            var iter = self.locals.iterator();
            while (iter.next()) |entry| {
                try entry.value_ptr.*.registers.ensureTotalCapacity(self.reserve_hint);
            }
        }

        /// Record flux from a Coarse block at a C-F interface.
        /// This SUBTRACTS the flux (or adds negative flux).
        /// scale = dt (flux is pre-scaled by area per face cell).
        pub fn addCoarse(self: *Self, tree: *const Tree, block_idx: usize, face: usize, flux: FaceField, scale: f64) !void {
            self.assertAccumulating();
            const block = tree.getBlock(block_idx) orelse return;

            const key = FaceKey{
                .block_key = tree.blockKeyForBlock(block),
                .face = @intCast(face),
            };

            const local = try self.getLocalRegister();
            const entry = try self.getOrPutRegister(local, key);
            if (!entry.found_existing) {
                entry.value_ptr.* = field_math.zeroField(FaceField);
            }

            // R -= F_coarse * scale
            field_math.addScaledField(FaceField, entry.value_ptr, flux, -scale);
        }

        /// Record flux from a Fine block at a C-F interface.
        /// This ADDS the flux.
        /// scale = dt (flux is pre-scaled by area per face cell).
        pub fn addFine(self: *Self, tree: *const Tree, block_idx: usize, face: usize, flux: FaceField, scale: f64) !void {
            self.assertAccumulating();
            const block = tree.getBlock(block_idx) orelse return;
            if (block.level == 0) return; // Cannot be fine relative to anything if level 0

            // The register lives on the coarse face. Convert fine global coords
            // (level L) to coarse coords (level L-1) by shifting right 1.

            const neighbor_info = tree.neighborInfo(block_idx, face);
            if (neighbor_info.exists() and neighbor_info.level_diff == -1) {
                // We have a direct link to the coarse neighbor!
                const neighbor_idx = neighbor_info.block_idx;
                const neighbor = tree.getBlock(neighbor_idx) orelse return;

                // The coarse face is `face ^ 1` (opposite) on the neighbor?
                // No, we are adding flux *across* the boundary.
                // Flux leaving Fine through Face F enters Coarse through Face F^1.
                // We store the register on the boundary.
                // Let's consistently index by the COARSE block's face.
                // The interface is the Coarse block's `face ^ 1`.

                const coarse_key = FaceKey{
                    .block_key = tree.blockKeyForBlock(neighbor),
                    .face = @intCast(face ^ 1),
                };

                const local = try self.getLocalRegister();
                const entry = try self.getOrPutRegister(local, coarse_key);
                if (!entry.found_existing) {
                    entry.value_ptr.* = field_math.zeroField(FaceField);
                }

                const dim = face / 2;
                const coarse_origin = neighbor.origin;
                const fine_origin = block.origin;

                var mask: usize = 0;
                var mask_idx: usize = 0;
                for (0..Tree.dimensions) |d| {
                    if (d == dim) continue;
                    const fine_origin_coarse = fine_origin[d] >> 1;
                    if (std.debug.runtime_safety) {
                        std.debug.assert(fine_origin_coarse >= coarse_origin[d]);
                    }
                    const offset = fine_origin_coarse - coarse_origin[d];
                    if (std.debug.runtime_safety) {
                        std.debug.assert(offset == 0 or offset == half_block);
                    }
                    const half = offset / half_block;
                    const shift: Shift = @intCast(mask_idx);
                    mask |= half << shift;
                    mask_idx += 1;
                }

                const map = fine_to_coarse_face_idx[dim][mask];
                for (0..face_cells) |fine_face_idx| {
                    const coarse_face_idx = @as(usize, @intCast(map[fine_face_idx]));
                    field_math.addScaledField(FieldType, &entry.value_ptr.*[coarse_face_idx], flux[fine_face_idx], scale);
                }
            }
        }

        /// Apply the reflux correction to the coarse cells.
        pub fn reflux(self: *Self, tree: *const Tree, arena: *Arena) void {
            self.beginPhase(phase_reflux);
            defer self.endPhase();
            // Iterate over all registers
            self.mutex.lock();
            defer self.mutex.unlock();
            var iter = self.iterAll();
            while (iter.next()) |entry| {
                const key = entry.key_ptr;
                const flux_mismatch = entry.value_ptr.*; // (Sum F_fine * dt * A_fine_cell) - (F_crse * dt * A_crse_cell)

                // We need to apply this to the Coarse block.
                const block_idx = tree.findBlockByKey(key.block_key) orelse continue;
                const block = tree.getBlock(block_idx).?;

                const slot = tree.getFieldSlot(block_idx);
                if (slot == std.math.maxInt(usize)) continue;

                const u_coarse = arena.getSlot(slot);

                const vol = block.spacing;

                applyRefluxCorrection(Tree, u_coarse, key.face, flux_mismatch, vol, Tree.dimensions);
            }
        }

        pub fn iterAll(self: *Self) Iterator {
            return .{ .locals_iter = self.locals.iterator() };
        }

        pub fn reduce(self: *Self, allocator: std.mem.Allocator) !RegisterMap {
            var reduced = RegisterMap.init(allocator);
            try self.reduceInto(&reduced);
            return reduced;
        }

        pub fn reduceInto(self: *Self, dst: *RegisterMap) !void {
            dst.clearRetainingCapacity();
            var iter = self.iterAll();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const reduced_entry = try dst.getOrPut(key);
                if (!reduced_entry.found_existing) {
                    reduced_entry.value_ptr.* = field_math.zeroField(FaceField);
                }
                field_math.addScaledField(FaceField, reduced_entry.value_ptr, entry.value_ptr.*, 1.0);
            }
        }

        const Iterator = struct {
            locals_iter: std.AutoHashMap(ThreadKey, *LocalRegister).Iterator,
            current: ?RegisterMap.Iterator = null,

            pub fn next(self: *Iterator) ?RegisterEntry {
                while (true) {
                    if (self.current) |*iter| {
                        if (iter.next()) |entry| return entry;
                    }
                    if (self.locals_iter.next()) |local_entry| {
                        self.current = local_entry.value_ptr.*.registers.iterator();
                        continue;
                    }
                    return null;
                }
            }
        };

        fn getOrPutRegister(self: *Self, local: *LocalRegister, key: FaceKey) !RegisterMap.GetOrPutResult {
            if (!self.no_alloc) {
                return try local.registers.getOrPut(key);
            }

            if (local.registers.count() >= local.registers.capacity()) {
                if (!local.registers.contains(key)) {
                    return error.OutOfCapacity;
                }
            }

            return local.registers.getOrPutAssumeCapacity(key);
        }

        fn assertAccumulating(self: *Self) void {
            if (std.debug.runtime_safety) {
                std.debug.assert(self.phase.load(.acquire) == phase_idle);
            }
        }

        fn beginPhase(self: *Self, phase: u8) void {
            const prev = self.phase.swap(phase, .acq_rel);
            if (std.debug.runtime_safety) {
                std.debug.assert(prev == phase_idle);
            }
        }

        fn endPhase(self: *Self) void {
            self.phase.store(phase_idle, .release);
        }

        fn clampReserveHint(count: usize) RegisterMap.Size {
            const max_hint = std.math.maxInt(RegisterMap.Size);
            if (count >= max_hint) return max_hint;
            return @as(RegisterMap.Size, @intCast(count));
        }

        fn getLocalRegister(self: *Self) !*LocalRegister {
            if (tls_owner == self and tls_cookie == self.cookie) {
                if (tls_local) |local| return local;
            }

            const local = try self.getOrCreateLocalRegister();
            tls_owner = self;
            tls_cookie = self.cookie;
            tls_local = local;
            return local;
        }

        fn getOrCreateLocalRegister(self: *Self) !*LocalRegister {
            const tid = std.Thread.getCurrentId();
            self.mutex.lock();
            defer self.mutex.unlock();

            const entry = try self.locals.getOrPut(tid);
            if (!entry.found_existing) {
                const local = try self.allocator.create(LocalRegister);
                local.* = .{ .registers = RegisterMap.init(self.allocator) };
                if (self.reserve_hint > 0) {
                    try local.registers.ensureTotalCapacity(self.reserve_hint);
                }
                entry.value_ptr.* = local;
            }
            return entry.value_ptr.*;
        }
    };
}

fn applyRefluxCorrection(
    comptime Tree: type,
    field: []Tree.FieldType,
    face: u8,
    correction: [Tree.BlockType.ghost_face_size]Tree.FieldType,
    cell_vol: f64,
    comptime Nd: usize,
) void {
    const block_size = Tree.block_size_const;

    // correction is per-face-cell flux integral. divide by V_cell to get density correction.
    var v_cell: f64 = 1.0;
    inline for (0..Nd) |_| v_cell *= cell_vol;

    // Indices of cells on the face
    const dim = face / 2;
    const is_upper = (face % 2) == 0;

    // If face is upper (+), flux leaves, contributes -F to dU/dt. Correction: -(F_new - F_old).
    // If face is lower (-), flux enters, contributes +F to dU/dt. Correction: +(F_new - F_old).
    const sign: f64 = if (is_upper) -1.0 else 1.0;

    const coord = if (is_upper) block_size - 1 else 0;

    // Iterate over the face cells
    // This is `extractBoundaryFace` logic reversed.
    const face_cells = Tree.BlockType.ghost_face_size; // N^(d-1)

    for (0..face_cells) |face_idx| {
        var coords: [Nd]usize = undefined;
        var rem = face_idx;
        for (0..Nd) |d| {
            if (d == dim) {
                coords[d] = coord;
            } else {
                coords[d] = rem % block_size;
                rem /= block_size;
            }
        }

        const idx = Tree.BlockType.getLocalIndex(coords);
        const dens_correction = field_math.scaleField(Tree.FieldType, correction[face_idx], sign / v_cell);
        field[idx] = field_math.addField(Tree.FieldType, field[idx], dens_correction);
    }
}

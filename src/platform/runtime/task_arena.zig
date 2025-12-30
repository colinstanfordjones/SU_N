//! Arena allocator wrapper for task scheduling.

const std = @import("std");

pub const TaskArena = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator) !Self {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Self{
            .allocator = allocator,
            .arena = arena,
        };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }

    pub fn reset(self: *Self) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn allocatorHandle(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }
};

//! File operation helpers for platform-facing code.

const std = @import("std");

pub const FileOps = struct {
    pub fn ensureDir(dir: std.fs.Dir, path: []const u8) !void {
        dir.makePath(path) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
    }

    pub fn readAllAlloc(allocator: std.mem.Allocator, dir: std.fs.Dir, path: []const u8) ![]u8 {
        var file = try dir.openFile(path, .{});
        defer file.close();
        return try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    }

    pub fn writeAll(dir: std.fs.Dir, path: []const u8, data: []const u8) !void {
        var file = try dir.createFile(path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(data);
    }
};

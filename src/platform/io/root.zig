//! IO helpers for file operations.

pub const file_ops = @import("file_ops.zig");

pub const FileOps = file_ops.FileOps;

test {
    _ = file_ops;
}

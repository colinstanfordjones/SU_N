const types = @import("types.zig");

pub const enabled = false;

pub const Comm = struct {
    handle: usize = 0,
};

pub const Request = struct {
    handle: usize = 0,
};

pub const ThreadLevel = types.ThreadLevel;
pub const preferred_thread_level: ThreadLevel = .serialized;
pub const any_source: i32 = -1;
pub const any_tag: i32 = -1;

pub fn initThreaded(_: ThreadLevel) !ThreadLevel {
    return types.Error.MpiDisabled;
}

pub fn initSerialized() !ThreadLevel {
    return types.Error.MpiDisabled;
}

pub fn finalize() void {}

pub fn commWorld() Comm {
    return .{};
}

pub fn rank(_: Comm) !i32 {
    return types.Error.MpiDisabled;
}

pub fn size(_: Comm) !i32 {
    return types.Error.MpiDisabled;
}

pub fn barrier(_: Comm) !void {
    return types.Error.MpiDisabled;
}

pub fn bcastBytes(_: Comm, _: []u8, _: i32) !void {
    return types.Error.MpiDisabled;
}

pub fn isend(_: Comm, _: []const u8, _: i32, _: types.Tag) !Request {
    return types.Error.MpiDisabled;
}

pub fn irecv(_: Comm, _: []u8, _: i32, _: types.Tag) !Request {
    return types.Error.MpiDisabled;
}

pub fn irecvAny(_: Comm, _: []u8, _: types.Tag) !Request {
    return types.Error.MpiDisabled;
}

pub fn waitAll(_: []Request) !void {
    return types.Error.MpiDisabled;
}

pub fn allreduceSum(_: Comm, _: f64) !f64 {
    return types.Error.MpiDisabled;
}

pub fn allgatherI32(_: Comm, _: i32, _: []i32) !void {
    return types.Error.MpiDisabled;
}

pub fn allgatherBytes(_: Comm, _: []const u8, _: []u8) !void {
    return types.Error.MpiDisabled;
}

pub fn allgatherVBytes(_: Comm, _: []const u8, _: []u8, _: []const i32, _: []const i32) !void {
    return types.Error.MpiDisabled;
}

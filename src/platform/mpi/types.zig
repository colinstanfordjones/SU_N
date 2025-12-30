pub const Error = error{
    MpiFailure,
    MpiDisabled,
    MpiThreadLevelUnsupported,
};

pub const ThreadLevel = enum {
    single,
    funneled,
    serialized,
    multiple,

    pub fn name(level: ThreadLevel) []const u8 {
        return switch (level) {
            .single => "single",
            .funneled => "funneled",
            .serialized => "serialized",
            .multiple => "multiple",
        };
    }
};

pub const Tag = i32;

pub const Status = struct {
    count: usize = 0,
    source: i32 = -1,
    tag: i32 = -1,
};

pub const Stat = struct {
    pub fn ignore() Status {
        return .{};
    }
};

pub const Size = struct {
    pub fn of(comptime T: type, count: usize) usize {
        return @sizeOf(T) * count;
    }
};

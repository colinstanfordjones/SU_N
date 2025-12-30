const std = @import("std");

pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

pub const OrtApi = c.OrtApi;
pub const OrtEnv = c.OrtEnv;
pub const OrtSession = c.OrtSession;
pub const OrtSessionOptions = c.OrtSessionOptions;
pub const OrtMemoryInfo = c.OrtMemoryInfo;
pub const OrtValue = c.OrtValue;

// Helper to check status
pub fn checkStatus(api: *const OrtApi, status: ?*c.OrtStatus) !void {
    if (status) |s| {
        const msg = api.GetErrorMessage.?(s);
        std.debug.print("ORT Error: {s}\n", .{msg});
        api.ReleaseStatus.?(s);
        return error.OnnxRuntimeError;
    }
}


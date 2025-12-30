const std = @import("std");

pub const c = @cImport({
    @cInclude("faiss_c.h");
    @cInclude("Index_c.h");
    @cInclude("IndexFlat_c.h");
});

pub const FaissIndex = c.FaissIndex;
pub const FaissIndexFlatL2 = c.FaissIndexFlatL2;

pub fn checkError(code: c_int) !void {
    if (code != 0) {
        const msg = c.faiss_get_last_error();
        std.debug.print("FAISS Error: {s}\n", .{msg});
        return error.FaissError;
    }
}

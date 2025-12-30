const std = @import("std");

pub const Options = struct {
    enable_mpi: bool = false,
};

pub fn create(b: *std.Build, options: Options) *std.Build.Module {
    var build_options = b.addOptions();
    build_options.addOption(bool, "enable_mpi", options.enable_mpi);
    return build_options.createModule();
}

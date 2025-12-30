const std = @import("std");

pub const Options = struct {
    source_dir: std.Build.LazyPath,
    prefix: []const u8,
    build_dir: []const u8,
    jobs: usize,
};

pub const Artifacts = struct {
    step: *std.Build.Step,
    include_dir: []const u8,
    lib_dir: []const u8,
    prefix: []const u8,
    stamp: std.Build.LazyPath,
};

pub fn setup(b: *std.Build, options: Options) Artifacts {
    const src_path = options.source_dir.getPath(b);
    const build_dir = options.build_dir;
    const prefix = options.prefix;
    const jobs = options.jobs;

    const script = b.fmt(
        \\set -euo pipefail
        \\src="{s}"
        \\build="{s}"
        \\prefix="{s}"
        \\stamp="$1"
        \\mkdir -p "$build"
        \\if [ -f "$src/.gitmodules" ]; then
        \\  (cd "$src" && git submodule update --init --recursive)
        \\fi
        \\if [ ! -f "$src/configure" ]; then
        \\  (cd "$src" && ./autogen.sh)
        \\fi
        \\cd "$build"
        \\if [ ! -f "config.status" ]; then
        \\  "$src/configure" --prefix="$prefix" --disable-fortran --disable-cxx --enable-shared --enable-static
        \\fi
        \\make -j{d}
        \\make install
        \\touch "$stamp"
    , .{ src_path, build_dir, prefix, jobs });

    const run = b.addSystemCommand(&.{ "bash", "-lc", script, "mpich-build" });
    const stamp = run.addOutputFileArg("mpich.stamp");

    run.addFileInput(options.source_dir.path(b, "autogen.sh"));
    run.addFileInput(options.source_dir.path(b, "configure.ac"));
    run.addFileInput(options.source_dir.path(b, ".gitmodules"));
    run.addFileInput(options.source_dir.path(b, "src/include/mpi.h.in"));
    run.addFileInput(b.path("build/mpich.zig"));
    run.addFileInput(b.path("build.zig"));

    return .{
        .step = &run.step,
        .include_dir = b.pathJoin(&.{ prefix, "include" }),
        .lib_dir = b.pathJoin(&.{ prefix, "lib" }),
        .prefix = prefix,
        .stamp = stamp,
    };
}

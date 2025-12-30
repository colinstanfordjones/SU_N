const std = @import("std");
const su_n = @import("su_n");
const Su2 = su_n.gauge.su2.Su2;

export fn su2_get_element(gen_idx: i32, row: usize, col: usize, real: *f64, imag: *f64) void {
    if (row >= 2 or col >= 2) return;

    const matrix = switch (gen_idx) {
        0 => Su2.identity(),
        1 => Su2.sigma1(),
        2 => Su2.sigma2(),
        3 => Su2.sigma3(),
        else => Su2.identity(), // Default
    };

    const val = matrix.data[row][col];
    real.* = val.re;
    imag.* = val.im;
}

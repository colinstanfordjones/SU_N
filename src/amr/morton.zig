//! Morton (Z-order) encoding helpers for AMR block indexing.

const std = @import("std");

pub const BlockKey = struct {
    level: u8,
    morton: u64,
};

pub fn encode(comptime Nd: usize, coords: [Nd]usize, bits: u8) u64 {
    std.debug.assert(@as(usize, bits) * Nd <= 64);

    var code: u64 = 0;
    var bit: usize = 0;
    while (bit < bits) : (bit += 1) {
        inline for (0..Nd) |d| {
            const out_bit = bit * Nd + d;
            const shift: u6 = @intCast(out_bit);
            const bit_val = (coords[d] >> @intCast(bit)) & 1;
            code |= @as(u64, bit_val) << shift;
        }
    }
    return code;
}

pub fn decode(comptime Nd: usize, code: u64, bits: u8) [Nd]usize {
    std.debug.assert(@as(usize, bits) * Nd <= 64);

    var coords: [Nd]usize = .{0} ** Nd;
    var bit: usize = 0;
    while (bit < bits) : (bit += 1) {
        inline for (0..Nd) |d| {
            const out_bit = bit * Nd + d;
            const shift: u6 = @intCast(out_bit);
            const bit_val = (code >> shift) & 1;
            coords[d] |= @as(usize, @intCast(bit_val)) << @intCast(bit);
        }
    }
    return coords;
}

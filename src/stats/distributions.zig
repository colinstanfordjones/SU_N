const std = @import("std");
const constants = @import("constants");

pub const Rng = struct {
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) Rng {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn random(self: *Rng) std.Random {
        return self.prng.random();
    }
};

pub const Distribution = enum {
    Uniform,
    Gaussian,
};

pub const Sampler = struct {
    rng: Rng,

    pub fn init(seed: u64) Sampler {
        return .{
            .rng = Rng.init(seed),
        };
    }

    pub fn sample(self: *Sampler, comptime dist: Distribution, params: anytype) f64 {
        const r = self.rng.random();
        switch (dist) {
            .Uniform => {
                // params expected: struct { min: f64, max: f64 }
                const min = @field(params, "min");
                const max = @field(params, "max");
                return min + (max - min) * r.float(f64);
            },
            .Gaussian => {
                // params expected: struct { mean: f64, stddev: f64 }
                const mean = @field(params, "mean");
                const stddev = @field(params, "stddev");
                return mean + stddev * r.floatNorm(f64);
            },
        }
    }
};

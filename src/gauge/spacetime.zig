const std = @import("std");
const math = @import("math");

/// Minkowski Metric signature (+ - - -)
/// Used to raise/lower indices and calculate invariant intervals.
pub const Metric = struct {
    pub fn apply(v: [4]f64) [4]f64 {
        return .{ v[0], -v[1], -v[2], -v[3] };
    }
};

/// Spacetime Event / Four-Vector (ct, x, y, z)
pub const FourVector = struct {
    data: [4]f64, // (0: Time, 1-3: Space)

    pub fn init(t: f64, x: f64, y: f64, z: f64) FourVector {
        return .{ .data = .{ t, x, y, z } };
    }

    /// Inner product: a . b = a^mu * eta_mu_nu * b^nu
    /// Invariant scalar (s^2)
    pub fn dot(self: FourVector, other: FourVector) f64 {
        const lower = Metric.apply(other.data);
        return self.data[0] * lower[0] +
               self.data[1] * lower[1] +
               self.data[2] * lower[2] +
               self.data[3] * lower[3];
    }

    /// Squared Magnitude (Interval)
    /// > 0: Timelike
    /// = 0: Lightlike
    /// < 0: Spacelike
    pub fn interval(self: FourVector) f64 {
        return self.dot(self);
    }
    
    // Basic addition subtraction for vectors
    pub fn add(self: FourVector, other: FourVector) FourVector {
        return .{ .data = .{
            self.data[0] + other.data[0],
            self.data[1] + other.data[1],
            self.data[2] + other.data[2],
            self.data[3] + other.data[3],
        }};
    }
};

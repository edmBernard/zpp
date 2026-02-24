//! Tests for math.zig - SIMD math functions

const std = @import("std");
const zpp = @import("zpp");

test "SIMD splat helper" {
    const VecF32 = @Vector(4, f32);
    const VecI32 = @Vector(4, i32);

    // Test with f32
    const ones = zpp.splat(VecF32, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), ones[0]);
    try std.testing.expectEqual(@as(f32, 1.0), ones[1]);
    try std.testing.expectEqual(@as(f32, 1.0), ones[2]);
    try std.testing.expectEqual(@as(f32, 1.0), ones[3]);

    // Test with i32
    const fives = zpp.splat(VecI32, 5);
    try std.testing.expectEqual(@as(i32, 5), fives[0]);
    try std.testing.expectEqual(@as(i32, 5), fives[3]);

    // Test with negative values
    const neg = zpp.splat(VecF32, -3.14);
    try std.testing.expectApproxEqAbs(@as(f32, -3.14), neg[0], 0.001);
}

test "SIMD math functions" {
    const VecF32 = @Vector(4, f32);

    // Test abs
    const neg_vec: VecF32 = .{ -1.0, 2.0, -3.0, 4.0 };
    const abs_result = zpp.abs(neg_vec);
    try std.testing.expectEqual(@as(f32, 1.0), abs_result[0]);
    try std.testing.expectEqual(@as(f32, 2.0), abs_result[1]);
    try std.testing.expectEqual(@as(f32, 3.0), abs_result[2]);
    try std.testing.expectEqual(@as(f32, 4.0), abs_result[3]);

    // Test floor
    const float_vec: VecF32 = .{ 1.7, 2.3, -1.5, -2.9 };
    const floor_result = zpp.floor(float_vec);
    try std.testing.expectEqual(@as(f32, 1.0), floor_result[0]);
    try std.testing.expectEqual(@as(f32, 2.0), floor_result[1]);
    try std.testing.expectEqual(@as(f32, -2.0), floor_result[2]);
    try std.testing.expectEqual(@as(f32, -3.0), floor_result[3]);

    // Test ceil
    const ceil_result = zpp.ceil(float_vec);
    try std.testing.expectEqual(@as(f32, 2.0), ceil_result[0]);
    try std.testing.expectEqual(@as(f32, 3.0), ceil_result[1]);
    try std.testing.expectEqual(@as(f32, -1.0), ceil_result[2]);
    try std.testing.expectEqual(@as(f32, -2.0), ceil_result[3]);

    // Test sqrt
    const sqrt_vec: VecF32 = .{ 4.0, 9.0, 16.0, 25.0 };
    const sqrt_result = zpp.sqrt(sqrt_vec);
    try std.testing.expectEqual(@as(f32, 2.0), sqrt_result[0]);
    try std.testing.expectEqual(@as(f32, 3.0), sqrt_result[1]);
    try std.testing.expectEqual(@as(f32, 4.0), sqrt_result[2]);
    try std.testing.expectEqual(@as(f32, 5.0), sqrt_result[3]);

    // Test sign
    const sign_vec: VecF32 = .{ -5.0, 0.0, 3.0, -0.1 };
    const sign_result = zpp.sign(sign_vec);
    try std.testing.expectEqual(@as(f32, -1.0), sign_result[0]);
    try std.testing.expectEqual(@as(f32, 0.0), sign_result[1]);
    try std.testing.expectEqual(@as(f32, 1.0), sign_result[2]);
    try std.testing.expectEqual(@as(f32, -1.0), sign_result[3]);

    // Test min/max
    const a: VecF32 = .{ 1.0, 5.0, 3.0, 8.0 };
    const b: VecF32 = .{ 2.0, 3.0, 4.0, 6.0 };
    const min_result = zpp.min(a, b);
    const max_result = zpp.max(a, b);
    try std.testing.expectEqual(@as(f32, 1.0), min_result[0]);
    try std.testing.expectEqual(@as(f32, 3.0), min_result[1]);
    try std.testing.expectEqual(@as(f32, 3.0), min_result[2]);
    try std.testing.expectEqual(@as(f32, 6.0), min_result[3]);
    try std.testing.expectEqual(@as(f32, 2.0), max_result[0]);
    try std.testing.expectEqual(@as(f32, 5.0), max_result[1]);
    try std.testing.expectEqual(@as(f32, 4.0), max_result[2]);
    try std.testing.expectEqual(@as(f32, 8.0), max_result[3]);

    // Test lerp
    const t: VecF32 = @splat(0.5);
    const lerp_result = zpp.lerp(a, b, t);
    try std.testing.expectEqual(@as(f32, 1.5), lerp_result[0]);
    try std.testing.expectEqual(@as(f32, 4.0), lerp_result[1]);
    try std.testing.expectEqual(@as(f32, 3.5), lerp_result[2]);
    try std.testing.expectEqual(@as(f32, 7.0), lerp_result[3]);

    // Test fma: a * b + c
    const c: VecF32 = .{ 1.0, 1.0, 1.0, 1.0 };
    const fma_result = zpp.fma(a, b, c);
    try std.testing.expectEqual(@as(f32, 3.0), fma_result[0]); // 1*2 + 1
    try std.testing.expectEqual(@as(f32, 16.0), fma_result[1]); // 5*3 + 1
    try std.testing.expectEqual(@as(f32, 13.0), fma_result[2]); // 3*4 + 1
    try std.testing.expectEqual(@as(f32, 49.0), fma_result[3]); // 8*6 + 1
}

test "SIMD trig functions" {
    const VecF32 = @Vector(4, f32);
    const pi = std.math.pi;
    const tolerance: f32 = 0.0001;

    // Test sin at known values
    const sin_vec: VecF32 = .{ 0.0, pi / 6.0, pi / 2.0, pi };
    const sin_result = zpp.sin(sin_vec);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sin_result[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sin_result[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sin_result[2], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sin_result[3], tolerance);

    // Test cos at known values
    const cos_vec: VecF32 = .{ 0.0, pi / 3.0, pi / 2.0, pi };
    const cos_result = zpp.cos(cos_vec);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cos_result[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cos_result[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cos_result[2], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), cos_result[3], tolerance);
}

test "SIMD exp and log functions" {
    const VecF32 = @Vector(4, f32);
    const tolerance: f32 = 0.0001;

    // Test exp
    const exp_vec: VecF32 = .{ 0.0, 1.0, 2.0, -1.0 };
    const exp_result = zpp.exp(exp_vec);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), exp_result[0], tolerance); // e^0 = 1
    try std.testing.expectApproxEqAbs(@as(f32, std.math.e), exp_result[1], tolerance); // e^1 = e
    try std.testing.expectApproxEqAbs(@as(f32, std.math.e * std.math.e), exp_result[2], 0.001); // e^2
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / std.math.e), exp_result[3], tolerance); // e^-1

    // Test log
    const log_vec: VecF32 = .{ 1.0, std.math.e, std.math.e * std.math.e, 10.0 };
    const log_result = zpp.log(log_vec);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), log_result[0], tolerance); // ln(1) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), log_result[1], tolerance); // ln(e) = 1
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), log_result[2], tolerance); // ln(e^2) = 2

    // Test log2
    const log2_vec: VecF32 = .{ 1.0, 2.0, 4.0, 8.0 };
    const log2_result = zpp.log2(log2_vec);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), log2_result[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), log2_result[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), log2_result[2], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), log2_result[3], tolerance);
}

test "SIMD trunc and round" {
    const VecF32 = @Vector(4, f32);

    // Test trunc (round towards zero)
    const trunc_vec: VecF32 = .{ 1.7, -1.7, 2.3, -2.3 };
    const trunc_result = zpp.trunc(trunc_vec);
    try std.testing.expectEqual(@as(f32, 1.0), trunc_result[0]);
    try std.testing.expectEqual(@as(f32, -1.0), trunc_result[1]);
    try std.testing.expectEqual(@as(f32, 2.0), trunc_result[2]);
    try std.testing.expectEqual(@as(f32, -2.0), trunc_result[3]);

    // Test round (round to nearest)
    const round_vec: VecF32 = .{ 1.4, 1.5, -1.4, -1.5 };
    const round_result = zpp.round(round_vec);
    try std.testing.expectEqual(@as(f32, 1.0), round_result[0]);
    try std.testing.expectEqual(@as(f32, 2.0), round_result[1]); // round half up
    try std.testing.expectEqual(@as(f32, -1.0), round_result[2]);
    try std.testing.expectEqual(@as(f32, -2.0), round_result[3]); // round half down
}

test "SIMD clamp" {
    const VecF32 = @Vector(4, f32);

    const values: VecF32 = .{ -5.0, 0.5, 1.5, 10.0 };
    const lo: VecF32 = @splat(0.0);
    const hi: VecF32 = @splat(1.0);
    const result = zpp.clamp(values, lo, hi);

    try std.testing.expectEqual(@as(f32, 0.0), result[0]); // clamped to lo
    try std.testing.expectEqual(@as(f32, 0.5), result[1]); // unchanged
    try std.testing.expectEqual(@as(f32, 1.0), result[2]); // clamped to hi
    try std.testing.expectEqual(@as(f32, 1.0), result[3]); // clamped to hi
}

test "SIMD pow" {
    const VecF32 = @Vector(4, f32);
    const tolerance: f32 = 0.01;

    const bases: VecF32 = .{ 2.0, 3.0, 4.0, 10.0 };
    const exponents: VecF32 = .{ 2.0, 2.0, 0.5, 1.0 };
    const result = zpp.pow(bases, exponents);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[0], tolerance); // 2^2 = 4
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[1], tolerance); // 3^2 = 9
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[2], tolerance); // 4^0.5 = 2
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result[3], tolerance); // 10^1 = 10
}

test "SIMD atan2" {
    const VecF32 = @Vector(4, f32);
    const tolerance: f32 = 0.02; // atan2 approximation has some error

    // Test basic quadrants
    const y: VecF32 = .{ 0.0, 1.0, 0.0, -1.0 };
    const x: VecF32 = .{ 1.0, 0.0, -1.0, 0.0 };
    const result = zpp.atan2(y, x);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], tolerance); // atan2(0, 1) = 0
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), result[1], tolerance); // atan2(1, 0) = pi/2
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi), result[2], tolerance); // atan2(0, -1) = pi
    try std.testing.expectApproxEqAbs(@as(f32, -std.math.pi / 2.0), result[3], tolerance); // atan2(-1, 0) = -pi/2
}

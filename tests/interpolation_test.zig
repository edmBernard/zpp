//! Tests for interpolation.zig - Interpolation support for pixel sampling

const std = @import("std");
const zpp = @import("zpp");

const f32x4 = @Vector(4, f32);

test "InterpLoop Nearest: identity transform" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    // Input: [[1, 2, 3, 4], [5, 6, 7, 8]]
    var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output_data: [8]f32 = .{0} ** 8;

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Identity transform kernel
    const interp_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const ctx = interp_kernel.Context{};
    const result = zpp.InterpLoop(f32x4, .Nearest, source, region, ctx, interp_kernel.process);
    zpp.Process(result, destination);

    // Identity transform should copy input to output
    try std.testing.expectEqual(@as(f32, 1.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[3]);
    try std.testing.expectEqual(@as(f32, 5.0), output_data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), output_data[5]);
}

test "InterpLoop Linear: bilinear interpolation" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Simple 2x2 input: [[0, 2], [0, 2]]
    const input_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    var input_data: [4]f32 = .{ 0.0, 2.0, 0.0, 2.0 };
    var output_data: [4]f32 = .{0} ** 4;

    const source = zpp.In(f32, &input_data, input_region.width, input_region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Sample at half-pixel offsets
    const interp_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            // Sample at x/2 to test interpolation
            const scale: f32x4 = @splat(0.5);
            return interp.sample(x * scale, y * scale);
        }
    };

    const ctx = interp_kernel.Context{};
    const result = zpp.InterpLoop(f32x4, .Linear, source, region, ctx, interp_kernel.process);
    zpp.Process(result, destination);

    // At x=0 -> sample(0, 0) = 0
    // At x=1 -> sample(0.5, 0) = lerp(0, 2, 0.5) = 1
    // At x=2 -> sample(1, 0) = 2
    // At x=3 -> sample(1.5, 0) - may have edge effects
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output_data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output_data[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output_data[2], 0.01);
}

test "InterpLoop: scale transform" {
    const input_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };
    const output_region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 8 };

    // 4x4 gradient
    var input_data: [16]f32 = undefined;
    for (0..16) |i| {
        input_data[i] = @floatFromInt(i);
    }
    var output_data: [64]f32 = .{0} ** 64;

    const source = zpp.In(f32, &input_data, input_region.width, input_region);
    const destination = zpp.Out(f32, &output_data, output_region.width, output_region);

    // 2x upscale: output coords / 2 = input coords
    const scale_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const scale: f32x4 = @splat(0.5);
            return interp.sample(x * scale, y * scale);
        }
    };

    const ctx = scale_kernel.Context{};
    const result = zpp.InterpLoop(f32x4, .Nearest, source, output_region, ctx, scale_kernel.process);
    zpp.Process(result, destination);

    // Output[0,0] should sample input[0,0] = 0
    // Output[2,0] should sample input[1,0] = 1
    // Output[0,2] should sample input[0,1] = 4
    try std.testing.expectEqual(@as(f32, 0.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[16]); // row 2, col 0
}

test "InterpLoop Cubic: smooth interpolation" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Input with edge values for cubic interpolation
    var input_data: [4]f32 = .{ 0.0, 1.0, 1.0, 0.0 };
    var output_data: [4]f32 = .{0} ** 4;

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Identity transform
    const interp_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const ctx = interp_kernel.Context{};
    const result = zpp.InterpLoop(f32x4, .Cubic, source, region, ctx, interp_kernel.process);
    zpp.Process(result, destination);

    // Values should be close to original at integer coordinates
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output_data[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output_data[1], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output_data[2], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output_data[3], 0.1);
}

//! Tests for interpolation.zig - Interpolation support for pixel sampling

const std = @import("std");
const zpp = @import("zpp");

const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);

// MARK: InterpLoop Nearest: identity transform preserves values
test "InterpLoop Nearest: identity transform preserves values" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    var output_data = [_]f32{0} ** 8;

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Identity transform kernel - sample at same coordinates
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const result = zpp.InterpLoop(f32x4, .nearest, source, region, .{}, interp_kernel.process);
    zpp.Process(result, destination);

    // Identity transform should copy input to output
    const expected_data: [8]f32 = .{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: InterpLoop Nearest: slightly shifted transform preserves values
test "InterpLoop Nearest: slightly shifted transform preserves values" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    var output_data = [_]f32{0} ** 8;

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Identity transform kernel - sample at same coordinates
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x + th.splatWithCast(f32x4, 0.1), y + th.splatWithCast(f32x4, 0.1));
        }
    };

    const result = zpp.InterpLoop(f32x4, .nearest, source, region, .{}, interp_kernel.process);
    zpp.Process(result, destination);

    // Identity transform should copy input to output
    const expected_data: [8]f32 = .{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: InterpLoop: 2x scale with nearest produces correct duplication
test "InterpLoop: 2x scale with nearest produces correct duplication" {
    const input_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };
    const output_region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 8 };

    // 4x4 input gradient
    var input_data: [16]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data = [_]f32{0} ** 64;

    const source = zpp.In(f32, &input_data, input_region.width, input_region);
    const destination = zpp.Out(f32, &output_data, output_region.width, output_region);

    // 2x upscale: output coords * 0.5 = input coords
    const scale_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const scale: f32x4 = @splat(0.5);
            return interp.sample(x * scale, y * scale);
        }
    };

    const result = zpp.InterpLoop(f32x4, .nearest, source, output_region, .{}, scale_kernel.process);
    zpp.Process(result, destination);

    const expected_data: [64]f32 = .{
        1.0,  2.0,  2.0,  3.0,  3.0,  4.0,  4.0,  4.0,
        5.0,  6.0,  6.0,  7.0,  7.0,  8.0,  8.0,  8.0,
        5.0,  6.0,  6.0,  7.0,  7.0,  8.0,  8.0,  8.0,
        9.0,  10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0,
        9.0,  10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0,
        13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0,
        13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0,
        13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: InterpLoop: 2x scale with linear produces correct value
test "InterpLoop: 2x scale with linear produces correct value" {
    const input_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };
    const output_region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 8 };

    // 4x4 input gradient
    var input_data: [16]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data = [_]f32{0} ** 64;

    const source = zpp.In(f32, &input_data, input_region.width, input_region);
    const destination = zpp.Out(f32, &output_data, output_region.width, output_region);

    // 2x upscale: output coords * 0.5 = input coords
    const scale_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const scale: f32x4 = @splat(0.5);
            return interp.sample(x * scale, y * scale);
        }
    };

    const result = zpp.InterpLoop(f32x4, .linear, source, output_region, .{}, scale_kernel.process);
    zpp.Process(result, destination);

    const expected_data: [64]f32 = .{
        1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.0,
        3.0,  3.5,  4.0,  4.5,  5.0,  5.5,  6.0,  6.0,
        5.0,  5.5,  6.0,  6.5,  7.0,  7.5,  8.0,  8.0,
        7.0,  7.5,  8.0,  8.5,  9.0,  9.5,  10.0, 10.0,
        9.0,  9.5,  10.0, 10.5, 11.0, 11.5, 12.0, 12.0,
        11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.0,
        13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.0,
        13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.0,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

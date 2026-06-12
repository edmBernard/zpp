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

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    // Identity transform kernel - sample at same coordinates
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const result = zpp.interpLoop(f32x4, .nearest, source, region, .{}, interp_kernel.process);
    zpp.process(result, destination);

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

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    // Identity transform kernel - sample at same coordinates
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x + th.splatWithCast(f32x4, 0.1), y + th.splatWithCast(f32x4, 0.1));
        }
    };

    const result = zpp.interpLoop(f32x4, .nearest, source, region, .{}, interp_kernel.process);
    zpp.process(result, destination);

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

    const source = try zpp.makeSource(f32, &input_data, input_region.width, input_region);
    const destination = try zpp.makeDest(f32, &output_data, output_region.width, output_region);

    // 2x upscale: output coords * 0.5 = input coords
    const scale_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const scale: f32x4 = @splat(0.5);
            return interp.sample(x * scale, y * scale);
        }
    };

    const result = zpp.interpLoop(f32x4, .nearest, source, output_region, .{}, scale_kernel.process);
    zpp.process(result, destination);

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

// MARK: InterpLoop Nearest: identity transform with generator source
test "InterpLoop Nearest: identity transform with generator source" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var output_data = [_]f32{0} ** 8;
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    // Generator that produces y * width + x (ramp pattern)
    const gen_kernel = struct {
        fn process(ctx: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const factor: f32x4 = @splat(4);
            return x + y * factor;
        }
    };

    const generator = zpp.generate(f32x4, .{}, gen_kernel.process);

    // Identity transform kernel - sample at same coordinates
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const result = zpp.interpLoop(f32x4, .nearest, generator, region, .{}, interp_kernel.process);
    zpp.process(result, destination);

    // Generator produces x + y*4, so: row0=[0,1,2,3], row1=[4,5,6,7]
    const expected_data: [8]f32 = .{
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
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

    const source = try zpp.makeSource(f32, &input_data, input_region.width, input_region);
    const destination = try zpp.makeDest(f32, &output_data, output_region.width, output_region);

    // 2x upscale: output coords * 0.5 = input coords
    const scale_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const scale: f32x4 = @splat(0.5);
            return interp.sample(x * scale, y * scale);
        }
    };

    const result = zpp.interpLoop(f32x4, .linear, source, output_region, .{}, scale_kernel.process);
    zpp.process(result, destination);

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

// MARK: InterpLoop Cubic: identity transform preserves values
test "InterpLoop Cubic: identity transform preserves values" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    var output_data = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    // Identity transform kernel - sample at same coordinates
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const result = zpp.interpLoop(f32x4, .cubic, source, region, .{}, interp_kernel.process);
    zpp.process(result, destination);

    // Catmull-Rom weights at t=0 are (0, 1, 0, 0), so integer-coordinate
    // sampling must reproduce the input exactly, even at the padded edges.
    const expected_data: [8]f32 = .{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: InterpLoop Linear: zero padding zeroes out-of-bounds samples
test "InterpLoop Linear: zero padding zeroes out-of-bounds samples" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    var output_data = [_]f32{0} ** 8;

    const source = try zpp.makePaddedSource(f32, zpp.ZeroPadding, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    // Sample half a pixel to the left: at x=0 the left tap reads outside the
    // region and must contribute zero, not the repeated edge pixel.
    const interp_kernel = struct {
        fn process(ctx: anytype, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            const half: f32x4 = @splat(0.5);
            return interp.sample(x - half, y);
        }
    };

    const result = zpp.interpLoop(f32x4, .linear, source, region, .{}, interp_kernel.process);
    zpp.process(result, destination);

    const expected_data: [8]f32 = .{
        0.5, 1.5, 2.5, 3.5,
        2.5, 5.5, 6.5, 7.5,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: InterpLoop: translated sources keep the vectorized clamp gather
test "InterpLoop Linear: identity transform through a translated source" {
    const width = 8;
    const height = 2;
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = width, .height = height };

    var input: [width * height]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    const source = try zpp.makeSource(f32, &input, width, region);
    const translated = zpp.translate(source, 3, 1);

    // Output region follows the translated source; identity sampling must
    // reproduce the original pixels at the shifted positions.
    const out_region = translated.region;
    const stride: u32 = width + 3; // covers the shifted region's stopX
    var output = [_]f32{0} ** (stride * 3);
    const destination = try zpp.makeDest(f32, &output, stride, out_region);

    const kernel = struct {
        fn process(ctx: @TypeOf(.{}), interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const result = zpp.interpLoop(f32x4, .linear, translated, out_region, .{}, kernel.process);
    zpp.process(result, destination);

    for (0..height) |y| {
        for (0..width) |x| {
            const out_idx = (y + 1) * stride + (x + 3);
            try std.testing.expectEqual(input[y * width + x], output[out_idx]);
        }
    }
}

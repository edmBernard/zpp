//! Tests for zip.zig - Zip and Unzip support

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);
const u8x4 = @Vector(4, u8);
const u16x4 = @Vector(4, u16);

// MARK: Zip: two sources add correctly
test "Zip: two sources add correctly" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_a: [8]f32 = undefined;
    th.fillRamp(f32, &input_a, 1, 1);
    var input_b: [8]f32 = undefined;
    th.fillRamp(f32, &input_b, 10, 10);
    var output_data = [_]f32{0} ** 8;

    const source_a = try zpp.makeSource(f32, &input_a, region.width, region);
    const source_b = try zpp.makeSource(f32, &input_b, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    const zipped = zpp.zip(.{ source_a, source_b });

    const add_kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            const a, const b = in.get();
            return a + b;
        }
    };

    const result = zpp.loop(f32x4, .{}, zipped, .{}, add_kernel.process);
    zpp.process(result, destination);

    // output = input_a + input_b = [1+10, 2+20, 3+30, ..., 8+80]
    const expected_data: [8]f32 = .{
        11, 22, 33, 44,
        55, 66, 77, 88,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Zip: fill two destination: kernel output is a struct
test "Zip: fill two destination: kernel output is a struct" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data_a = [_]f32{0} ** 8;
    var output_data_b = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination_a = try zpp.makeDest(f32, &output_data_a, region.width, region);
    const destination_b = try zpp.makeDest(f32, &output_data_b, region.width, region);

    const zipped = zpp.zipDest(.{ destination_a, destination_b });

    const kernel = struct {
        fn process(ctx: anytype, in: anytype) struct { f32x4, f32x4 } {
            _ = ctx;
            const value = in.get();
            return .{ value, value * th.splatWithCast(f32x4, 2) };
        }
    };

    const result = zpp.loop(f32x4, .{}, source, .{}, kernel.process);
    zpp.process(result, zipped);

    // output = input_a + input_b = [1+10, 2+20, 3+30, ..., 8+80]
    const expected_data_a = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    try std.testing.expectEqual(expected_data_a, output_data_a);
    const expected_data_b = [_]f32{
        2,  4,  6,  8,
        10, 12, 14, 16,
    };
    try std.testing.expectEqual(expected_data_b, output_data_b);
}

// MARK: Zip: fill two destination: kernel output is an array
test "Zip: fill two destination: kernel output is an array" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data_a = [_]f32{0} ** 8;
    var output_data_b = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination_a = try zpp.makeDest(f32, &output_data_a, region.width, region);
    const destination_b = try zpp.makeDest(f32, &output_data_b, region.width, region);

    const zipped = zpp.zipDest(.{ destination_a, destination_b });

    const kernel = struct {
        fn process(ctx: anytype, in: anytype) [2]f32x4 {
            _ = ctx;
            const value = in.get();
            return .{ value, value * th.splatWithCast(f32x4, 2) };
        }
    };

    const result = zpp.loop(f32x4, .{}, source, .{}, kernel.process);
    zpp.process(result, zipped);

    // output = input_a + input_b = [1+10, 2+20, 3+30, ..., 8+80]
    const expected_data_a = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    try std.testing.expectEqual(expected_data_a, output_data_a);
    const expected_data_b = [_]f32{
        2,  4,  6,  8,
        10, 12, 14, 16,
    };
    try std.testing.expectEqual(expected_data_b, output_data_b);
}

// MARK: Zip: Can chained several zip kernels together
test "Zip: Can chained several zip kernels together" {
    // first kernel produce 3 outputs, second kernel takes 3 inputs and produces 2 outputs
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data_a = [_]f32{0} ** 8;
    var output_data_b = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination_a = try zpp.makeDest(f32, &output_data_a, region.width, region);
    const destination_b = try zpp.makeDest(f32, &output_data_b, region.width, region);

    const zipped = zpp.zipDest(.{ destination_a, destination_b });

    const kernel_1 = struct {
        fn process(ctx: anytype, in: anytype) struct { f32x4, f32x4, f32x4 } {
            _ = ctx;
            const value = in.get();
            return .{ value, value * th.splatWithCast(f32x4, 10), value * th.splatWithCast(f32x4, 100) };
        }
    };
    const kernel_2 = struct {
        fn process(ctx: anytype, in: anytype) struct { f32x4, f32x4 } {
            _ = ctx;
            const a, const b, const c = in.get();
            return .{ a, b + c };
        }
    };

    const result_kernel_1 = zpp.loop(f32x4, .{}, source, .{}, kernel_1.process);
    const result_kernel_2 = zpp.loop(f32x4, .{}, result_kernel_1, .{}, kernel_2.process);
    zpp.process(result_kernel_2, zipped);

    // output = input_a + input_b = [1+10, 2+20, 3+30, ..., 8+80]
    const expected_data_a = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    try std.testing.expectEqual(expected_data_a, output_data_a);
    const expected_data_b = [_]f32{
        110, 220, 330, 440,
        550, 660, 770, 880,
    };
    try std.testing.expectEqual(expected_data_b, output_data_b);
}

// MARK: Zip: Can zip destination and Stats together
test "Zip: Can zip destination and Stats together" {
    // we transform the input and write to both a normal destination and a Stats accumulator
    // the stat is computed on the transformed value
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    const stat_kernel = struct {
        const Context = struct {
            sum: f32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            ctx.sum += @reduce(.Add, values);
        }
    };
    var stats_ctx = stat_kernel.Context{};

    const stats_dest = zpp.stats(f32x4, &stats_ctx, region, stat_kernel.accumulate);

    const kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            return in.get() * th.splatWithCast(f32x4, 10);
        }
    };

    const result = zpp.loop(f32x4, .{}, source, .{}, kernel.process);
    const zipped_in = zpp.zip(.{ result, result });
    const zipped_dest = zpp.zipDest(.{ destination, stats_dest });
    zpp.process(zipped_in, zipped_dest);

    const expected_data = [_]f32{
        10, 20, 30, 40,
        50, 60, 70, 80,
    };
    try std.testing.expectEqual(expected_data, output_data);

    try std.testing.expectEqual(360, stats_ctx.sum);
}

// MARK: Zip: Can zip destination and Stats together version 2
test "Zip: Can zip destination and Stats together version 2" {
    // we transform the input and write to both a normal destination and a Stats accumulator
    // the stat is computed on the source value
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);

    var output_data = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    const stat_kernel = struct {
        const Context = struct {
            sum: f32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            ctx.sum += @reduce(.Add, values);
        }
    };
    var stats_ctx = stat_kernel.Context{};

    const stats_dest = zpp.stats(f32x4, &stats_ctx, region, stat_kernel.accumulate);

    const kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            return in.get() * th.splatWithCast(f32x4, 10);
        }
    };

    const result = zpp.loop(f32x4, .{}, source, .{}, kernel.process);
    const zipped_in = zpp.zip(.{ result, source });
    const zipped_dest = zpp.zipDest(.{ destination, stats_dest });
    zpp.process(zipped_in, zipped_dest);

    const expected_data = [_]f32{
        10, 20, 30, 40,
        50, 60, 70, 80,
    };
    try std.testing.expectEqual(expected_data, output_data);

    try std.testing.expectEqual(36, stats_ctx.sum);
}

// MARK: Zip: Unzip split source correctly
test "Zip: Unzip split source correctly" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_a: [8]f32 = undefined;
    th.fillRamp(f32, &input_a, 1, 1);
    var input_b: [8]f32 = undefined;
    th.fillRamp(f32, &input_b, 10, 10);
    var output_data = [_]f32{0} ** 8;

    const source_a = try zpp.makeSource(f32, &input_a, region.width, region);
    const source_b = try zpp.makeSource(f32, &input_b, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    const zipped = zpp.zip(.{ source_a, source_b });

    const a, _ = zpp.unzip(zipped);

    zpp.process(a, destination);

    const expected_data = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

test "Zip: direct process preserves mixed integer channel types" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    const input_a: [8]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input_b: [8]u16 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var output_a = [_]u8{0} ** 8;
    var output_b = [_]u16{0} ** 8;

    const source_a = try zpp.makeSource(u8, &input_a, region.width, region);
    const source_b = try zpp.makeSource(u16, &input_b, region.width, region);
    const dest_a = try zpp.makeDest(u8, &output_a, region.width, region);
    const dest_b = try zpp.makeDest(u16, &output_b, region.width, region);

    zpp.process(zpp.zip(.{ source_a, source_b }), zpp.zipDest(.{ dest_a, dest_b }));

    try std.testing.expectEqual(input_a, output_a);
    try std.testing.expectEqual(input_b, output_b);
}

test "Zip: unzip preserves non-f32 channel types" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    const input_a: [8]u8 = .{ 8, 7, 6, 5, 4, 3, 2, 1 };
    const input_b: [8]u16 = .{ 100, 200, 300, 400, 500, 600, 700, 800 };
    var output_a = [_]u8{0} ** 8;
    var output_b = [_]u16{0} ** 8;

    const source_a = try zpp.makeSource(u8, &input_a, region.width, region);
    const source_b = try zpp.makeSource(u16, &input_b, region.width, region);
    const unzipped = zpp.unzip(zpp.zip(.{ source_a, source_b }));

    zpp.process(unzipped[0], try zpp.makeDest(u8, &output_a, region.width, region));
    zpp.process(unzipped[1], try zpp.makeDest(u16, &output_b, region.width, region));

    try std.testing.expectEqual(input_a, output_a);
    try std.testing.expectEqual(input_b, output_b);
}

test "Zip: loop can consume mixed integer channel types" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    const input_a: [8]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input_b: [8]u16 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var output = [_]u16{0} ** 8;

    const source_a = try zpp.makeSource(u8, &input_a, region.width, region);
    const source_b = try zpp.makeSource(u16, &input_b, region.width, region);
    const destination = try zpp.makeDest(u16, &output, region.width, region);

    const kernel = struct {
        fn process(ctx: anytype, in: anytype) u16x4 {
            _ = ctx;
            const a, const b = in.get();
            return @as(u16x4, @intCast(a)) + b;
        }
    };

    const result = zpp.loop(u8x4, .{}, zpp.zip(.{ source_a, source_b }), .{}, kernel.process);
    zpp.process(result, destination);

    const expected = [_]u16{ 11, 22, 33, 44, 55, 66, 77, 88 };
    try std.testing.expectEqual(expected, output);
}

// MARK: Zip: Unzip split source correctly and process
test "Zip: Unzip split source correctly and process" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_a: [8]f32 = undefined;
    th.fillRamp(f32, &input_a, 1, 1);
    var input_b: [8]f32 = undefined;
    th.fillRamp(f32, &input_b, 10, 10);
    var output_data = [_]f32{0} ** 8;

    const source_a = try zpp.makeSource(f32, &input_a, region.width, region);
    const source_b = try zpp.makeSource(f32, &input_b, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    const zipped = zpp.zip(.{ source_a, source_b });

    const a, _ = zpp.unzip(zipped);

    const kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            return in.get() * th.splatWithCast(f32x4, 2);
        }
    };

    const result = zpp.loop(f32x4, .{}, a, .{}, kernel.process);
    zpp.process(result, destination);

    const expected_data = [_]f32{
        2,  4,  6,  8,
        10, 12, 14, 16,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Zip: unzip preserves region dimensions
test "Zip: unzip preserves region dimensions" {
    const region_a: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 4 };
    const region_b: zpp.Region = .{ .x = 2, .y = 1, .width = 6, .height = 3 };

    var data_a = [_]f32{0} ** 32;
    var data_b = [_]f32{0} ** 32;

    const source_a = try zpp.makeSource(f32, &data_a, region_a.width, region_a);
    const source_b = try zpp.makeSource(f32, &data_b, 8, region_b);

    const zipped = zpp.zip(.{ source_a, source_b });

    // Zipped region should be intersection of both regions
    const zipped_region = zipped.region;
    try std.testing.expectEqual(@as(i32, 2), zipped_region.x);
    try std.testing.expectEqual(@as(i32, 1), zipped_region.y);
    try std.testing.expectEqual(@as(u32, 6), zipped_region.width);
    try std.testing.expectEqual(@as(u32, 3), zipped_region.height);

    // Unzip and verify each source preserves the zipped region
    const unzipped = zpp.unzip(zipped);
    const unzipped_0_region = unzipped[0].region;
    const unzipped_1_region = unzipped[1].region;

    try std.testing.expectEqual(zipped_region.x, unzipped_0_region.x);
    try std.testing.expectEqual(zipped_region.y, unzipped_0_region.y);
    try std.testing.expectEqual(zipped_region.width, unzipped_0_region.width);
    try std.testing.expectEqual(zipped_region.height, unzipped_0_region.height);

    try std.testing.expectEqual(zipped_region.x, unzipped_1_region.x);
    try std.testing.expectEqual(zipped_region.y, unzipped_1_region.y);
    try std.testing.expectEqual(zipped_region.width, unzipped_1_region.width);
    try std.testing.expectEqual(zipped_region.height, unzipped_1_region.height);
}

// MARK: Zip: interior fast path
test "Zip: margined loop over wide sources matches scalar reference" {
    // Wide enough that the interior unchecked fast path is taken; the result
    // must be identical to a scalar reference using repeat-edge clamping.
    const width = 64;
    const height = 3;
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = width, .height = height };

    var input_a: [width * height]f32 = undefined;
    var input_b: [width * height]f32 = undefined;
    for (0..width * height) |i| {
        input_a[i] = @floatFromInt(i);
        input_b[i] = @floatFromInt(10000 + i);
    }
    var output = [_]f32{0} ** (width * height);

    const source_a = try zpp.makeSource(f32, &input_a, width, region);
    const source_b = try zpp.makeSource(f32, &input_b, width, region);
    const destination = try zpp.makeDest(f32, &output, width, region);

    const zipped = zpp.zip(.{ source_a, source_b });

    // out(x) = a(x-1) + a(x+1) + b(x)
    const kernel = struct {
        fn process(ctx: @TypeOf(.{}), in: anytype) f32x4 {
            _ = ctx;
            const left = in.getAt(-1, 0);
            const right = in.getAt(1, 0);
            const center = in.get();
            return left[0] + right[0] + center[1];
        }
    };

    const result = zpp.loop(f32x4, .{ .margin = zpp.Margin.horizontal(1) }, zipped, .{}, kernel.process);
    zpp.process(result, destination);

    for (0..height) |y| {
        for (0..width) |x| {
            const xl = if (x == 0) 0 else x - 1;
            const xr = @min(x + 1, width - 1);
            const expected = input_a[y * width + xl] + input_a[y * width + xr] + input_b[y * width + x];
            try std.testing.expectEqual(expected, output[y * width + x]);
        }
    }
}

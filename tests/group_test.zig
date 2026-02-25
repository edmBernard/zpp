//! Tests for group.zig - Group and Ungroup support

const std = @import("std");
const zpp = @import("zpp");

const f32x4 = @Vector(4, f32);

// MARK: Group: 2x2 source downscales region correctly
test "Group: 2x2 source downscales region correctly" {
    // Test multiple input dimensions to verify region arithmetic
    const test_cases = [_]struct { w: u32, h: u32, expected_w: u32, expected_h: u32 }{
        .{ .w = 8, .h = 6, .expected_w = 4, .expected_h = 3 }, // exact division
        .{ .w = 7, .h = 5, .expected_w = 4, .expected_h = 3 }, // ceil(7/2)=4, ceil(5/2)=3
        .{ .w = 4, .h = 4, .expected_w = 2, .expected_h = 2 }, // small exact
        .{ .w = 1, .h = 1, .expected_w = 1, .expected_h = 1 }, // minimal
        .{ .w = 10, .h = 3, .expected_w = 5, .expected_h = 2 }, // wide, ceil(3/2)=2
    };

    for (test_cases) |tc| {
        const region: zpp.Region = .{ .x = 0, .y = 0, .width = tc.w, .height = tc.h };

        var data = [_]f32{0} ** 128;
        const source = zpp.makeSource(f32, data[0 .. tc.w * tc.h], region.width, region);

        const grouped = zpp.group(2, 2, source);
        const grouped_region = grouped.region;

        try std.testing.expectEqual(tc.expected_w, grouped_region.width);
        try std.testing.expectEqual(tc.expected_h, grouped_region.height);
    }
}

// MARK: Group: ungroup round-trip preserves region
test "Group: ungroup round-trip preserves region" {
    // Test the invariant: Ungroup(Group(source)).region dimensions
    const test_cases = [_]struct { w: u32, h: u32, expected_w: u32, expected_h: u32 }{
        .{ .w = 4, .h = 4, .expected_w = 4, .expected_h = 4 }, // exact: 4->2->4
        .{ .w = 8, .h = 6, .expected_w = 8, .expected_h = 6 }, // exact: 8->4->8, 6->3->6
        .{ .w = 3, .h = 3, .expected_w = 4, .expected_h = 4 }, // ceil(3/2)=2, 2*2=4 (rounds up)
        .{ .w = 5, .h = 7, .expected_w = 6, .expected_h = 8 }, // ceil(5/2)=3, 3*2=6; ceil(7/2)=4, 4*2=8
    };

    for (test_cases) |tc| {
        const region: zpp.Region = .{ .x = 0, .y = 0, .width = tc.w, .height = tc.h };

        var data = [_]f32{0} ** 128;
        const source = zpp.makeSource(f32, data[0 .. tc.w * tc.h], region.width, region);

        const grouped = zpp.group(2, 2, source);
        const ungrouped = zpp.ungroup(2, 2, grouped);
        const ungrouped_region = ungrouped.region;

        try std.testing.expectEqual(tc.expected_w, ungrouped_region.width);
        try std.testing.expectEqual(tc.expected_h, ungrouped_region.height);
    }
}

// MARK: Group destination: upsampling nearest neighbor
test "Group destination: upsampling nearest neighbor" {
    const dst_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };
    const in_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    var input: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var output_data = [_]f32{0} ** 16;

    const source = zpp.makeSource(f32, &input, in_region.width, in_region);
    const destination = zpp.makeDest(f32, &output_data, dst_region.width, dst_region);

    // Group the destination - each 2x2 block maps to one source pixel
    const grouped = zpp.groupDest(2, 2, destination);

    // Kernel that replicates input value to all 4 positions in the 2x2 block
    const split_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) struct { f32x4, f32x4, f32x4, f32x4 } {
            _ = ctx;
            const value = in.get();
            return .{ value, value, value, value };
        }
    };

    const ctx = split_kernel.Context{};
    const result = zpp.loop(f32x4, .{}, source, ctx, split_kernel.process);
    zpp.process(result, grouped);

    const expected_data = [_]f32{
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Group: depth to space 2x2 rearrangement
test "Group: depth to space 2x2 rearrangement" {
    // Depth-to-space: 4 channels of 2x2 -> 1 channel of 4x4
    // Input: 4 sources each 2x2
    // Output: 1 destination 4x4 (grouped as 2x2 blocks)
    const in_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var input_c: [4]f32 = .{ 100.0, 200.0, 300.0, 400.0 };
    var input_d: [4]f32 = .{ 1000.0, 2000.0, 3000.0, 4000.0 };
    var output_data = [_]f32{0} ** 16;

    const source_a = zpp.makeSource(f32, &input_a, in_region.width, in_region);
    const source_b = zpp.makeSource(f32, &input_b, in_region.width, in_region);
    const source_c = zpp.makeSource(f32, &input_c, in_region.width, in_region);
    const source_d = zpp.makeSource(f32, &input_d, in_region.width, in_region);
    const destination = zpp.makeDest(f32, &output_data, out_region.width, out_region);

    // Zip the 4 source channels
    const zipped = zpp.zip(.{ source_a, source_b, source_c, source_d });

    // Group the destination (2x2 blocks)
    const grouped = zpp.groupDest(2, 2, destination);

    zpp.process(zipped, grouped);

    const expected_data = [_]f32{
        1,   10,   2,   20,
        100, 1000, 200, 2000,
        3,   30,   4,   40,
        300, 3000, 400, 4000,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Group: depth to space 2x2 rearrangement with intermediate kernel
test "Group: depth to space 2x2 rearrangement with intermediate kernel" {
    // Same as Depth to Space but with a kernel that doubles channel 'a' values
    const in_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var input_c: [4]f32 = .{ 100.0, 200.0, 300.0, 400.0 };
    var input_d: [4]f32 = .{ 1000.0, 2000.0, 3000.0, 4000.0 };
    var output_data = [_]f32{0} ** 16;

    const source_a = zpp.makeSource(f32, &input_a, in_region.width, in_region);
    const source_b = zpp.makeSource(f32, &input_b, in_region.width, in_region);
    const source_c = zpp.makeSource(f32, &input_c, in_region.width, in_region);
    const source_d = zpp.makeSource(f32, &input_d, in_region.width, in_region);
    const destination = zpp.makeDest(f32, &output_data, out_region.width, out_region);

    const zipped = zpp.zip(.{ source_a, source_b, source_c, source_d });

    // Kernel that doubles channel 'a' values
    const double_a_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) struct { f32x4, f32x4, f32x4, f32x4 } {
            _ = ctx;
            const a, const b, const c, const d = in.get();
            return .{ a * @as(f32x4, @splat(2.0)), b, c, d };
        }
    };

    const grouped = zpp.groupDest(2, 2, destination);

    const ctx = double_a_kernel.Context{};
    const result = zpp.loop(f32x4, .{}, zipped, ctx, double_a_kernel.process);
    zpp.process(result, grouped);

    const expected_data = [_]f32{
        2,   10,   4,   20,
        100, 1000, 200, 2000,
        6,   30,   8,   40,
        300, 3000, 400, 4000,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Group: space to depth 2x2 rearrangement
test "Group: space to depth 2x2 rearrangement" {
    // Space-to-depth: 1 channel of 4x4 -> 4 channels of 2x2
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input: [16]f32 = .{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };
    var output_data_a: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_b: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_c: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_d: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.makeSource(f32, &input, region.width, region);
    // Group the source - each 2x2 block becomes one pixel with 4 channels
    const grouped = zpp.group(2, 2, source);

    const destination_region = grouped.region;
    const expected_dst_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    try std.testing.expectEqual(expected_dst_region.width, destination_region.width);
    try std.testing.expectEqual(expected_dst_region.height, destination_region.height);

    const destination_a = zpp.makeDest(f32, &output_data_a, destination_region.width, destination_region);
    const destination_b = zpp.makeDest(f32, &output_data_b, destination_region.width, destination_region);
    const destination_c = zpp.makeDest(f32, &output_data_c, destination_region.width, destination_region);
    const destination_d = zpp.makeDest(f32, &output_data_d, destination_region.width, destination_region);

    // Zip the destinations to receive the 4 channels
    const zipped_dest = zpp.zipDest(.{ destination_a, destination_b, destination_c, destination_d });

    zpp.process(grouped, zipped_dest);

    const expected_data_a = [_]f32{
        1, 3,
        9, 11,
    };
    try std.testing.expectEqual(expected_data_a, output_data_a);
    const expected_data_b = [_]f32{
        2,  4,
        10, 12,
    };
    try std.testing.expectEqual(expected_data_b, output_data_b);
    const expected_data_c = [_]f32{
        5,  7,
        13, 15,
    };
    try std.testing.expectEqual(expected_data_c, output_data_c);
    const expected_data_d = [_]f32{
        6,  8,
        14, 16,
    };
    try std.testing.expectEqual(expected_data_d, output_data_d);
}

// MARK: Group: space to depth 2x2 rearrangement with intermediate kernel
test "Group: space to depth 2x2 rearrangement with intermediate kernel" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input: [16]f32 = .{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };
    var output_data_a: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_b: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_c: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_d: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.makeSource(f32, &input, region.width, region);
    // Group the source
    const grouped = zpp.group(2, 2, source);

    const destination_region = grouped.region;
    const expected_dst_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    try std.testing.expectEqual(expected_dst_region.width, destination_region.width);
    try std.testing.expectEqual(expected_dst_region.height, destination_region.height);

    const destination_a = zpp.makeDest(f32, &output_data_a, destination_region.width, destination_region);
    const destination_b = zpp.makeDest(f32, &output_data_b, destination_region.width, destination_region);
    const destination_c = zpp.makeDest(f32, &output_data_c, destination_region.width, destination_region);
    const destination_d = zpp.makeDest(f32, &output_data_d, destination_region.width, destination_region);

    // Zip the destinations
    const zipped_dest = zpp.zipDest(.{ destination_a, destination_b, destination_c, destination_d });

    // Process with a kernel that outputs to three destinations
    // Kernel returns a tuple, Loop takes just f32x4
    const split_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) struct { f32x4, f32x4, f32x4, f32x4 } {
            _ = ctx;
            const a, const b, const c, const d = in.get();
            return .{ a * @as(f32x4, @splat(2.0)), b, c, d };
        }
    };

    const ctx = split_kernel.Context{};
    const result = zpp.loop(f32x4, .{}, grouped, ctx, split_kernel.process);
    zpp.process(result, zipped_dest);

    const expected_data_a = [_]f32{
        2,  6,
        18, 22,
    };
    try std.testing.expectEqual(expected_data_a, output_data_a);
    const expected_data_b = [_]f32{
        2,  4,
        10, 12,
    };
    try std.testing.expectEqual(expected_data_b, output_data_b);
    const expected_data_c = [_]f32{
        5,  7,
        13, 15,
    };
    try std.testing.expectEqual(expected_data_c, output_data_c);
    const expected_data_d = [_]f32{
        6,  8,
        14, 16,
    };
    try std.testing.expectEqual(expected_data_d, output_data_d);
}

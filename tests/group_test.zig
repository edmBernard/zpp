//! Tests for group.zig - Group and Ungroup support

const std = @import("std");
const zpp = @import("zpp");

const f32x4 = @Vector(4, f32);

test "Group source region downscaling" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 6 };

    var input_data: [48]f32 = undefined;
    for (0..48) |i| {
        input_data[i] = @floatFromInt(i);
    }

    const source = zpp.In(f32, &input_data, region.width, region);

    // Group 2x2 blocks
    const grouped = zpp.Group(2, 2, source);
    const grouped_region = grouped.getRegion();

    // Region should be downscaled
    try std.testing.expectEqual(@as(u32, 4), grouped_region.width); // 8 / 2
    try std.testing.expectEqual(@as(u32, 3), grouped_region.height); // 6 / 2
}

test "Ungroup source region upscaling" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 3 };

    var input_data: [12]f32 = undefined;
    for (0..12) |i| {
        input_data[i] = @floatFromInt(i);
    }

    const source = zpp.In(f32, &input_data, region.width, region);

    // Group then ungroup
    const grouped = zpp.Group(2, 2, source);
    const ungrouped = zpp.Ungroup(2, 2, grouped);
    const ungrouped_region = ungrouped.getRegion();

    // Input: 4x3
    // Grouped (downscaled by 2x2): ceil(4/2)=2, ceil(3/2)=2 -> 2x2
    // Ungrouped (upscaled by 2x2): 2*2=4, 2*2=4 -> 4x4
    try std.testing.expectEqual(@as(u32, 4), ungrouped_region.width);
    try std.testing.expectEqual(@as(u32, 4), ungrouped_region.height);
}

test "Group destination" {
    const dst_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };
    const in_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    var input: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var output_data = [_]f32{0} ** 16;

    const source = zpp.In(f32, &input, in_region.width, in_region);
    const destination = zpp.Out(f32, &output_data, dst_region.width, dst_region);

    // Group the destination - each 2x2 block maps to one source pixel
    const grouped = zpp.GroupOut(2, 2, destination);

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
    const result = zpp.Loop(f32x4, .{}, source, ctx, split_kernel.process);
    zpp.Process(result, grouped);

    // Input 2x2: [1, 2]    Output 4x4 layout (row-major):
    //            [3, 4]    Row 0: [1, 1, 2, 2]
    //                      Row 1: [1, 1, 2, 2]
    //                      Row 2: [3, 3, 4, 4]
    //                      Row 3: [3, 3, 4, 4]
    // Row 0
    try std.testing.expectEqual(@as(f32, 1.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[3]);
    // Row 1
    try std.testing.expectEqual(@as(f32, 1.0), output_data[4]);
    try std.testing.expectEqual(@as(f32, 1.0), output_data[5]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[6]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[7]);
    // Row 2
    try std.testing.expectEqual(@as(f32, 3.0), output_data[8]);
    try std.testing.expectEqual(@as(f32, 3.0), output_data[9]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[10]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[11]);
    // Row 3
    try std.testing.expectEqual(@as(f32, 3.0), output_data[12]);
    try std.testing.expectEqual(@as(f32, 3.0), output_data[13]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[14]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[15]);
}

test "Depth to Space 2x2" {
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

    const source_a = zpp.In(f32, &input_a, in_region.width, in_region);
    const source_b = zpp.In(f32, &input_b, in_region.width, in_region);
    const source_c = zpp.In(f32, &input_c, in_region.width, in_region);
    const source_d = zpp.In(f32, &input_d, in_region.width, in_region);
    const destination = zpp.Out(f32, &output_data, out_region.width, out_region);

    // Zip the 4 source channels
    const zipped = zpp.Zip(.{ source_a, source_b, source_c, source_d });

    // Group the destination (2x2 blocks)
    const grouped = zpp.GroupOut(2, 2, destination);

    zpp.Process(zipped, grouped);

    // Each 2x2 block in output comes from one pixel position across all 4 channels
    // Block layout within each 2x2: [a, b]
    //                               [c, d]
    // Position (0,0) in sources: a=1, b=10, c=100, d=1000
    // -> Output block at (0,0)-(1,1): [1, 10], [100, 1000]
    // Row 0: output[0]=1, output[1]=10, output[2]=2, output[3]=20
    // Row 1: output[4]=100, output[5]=1000, output[6]=200, output[7]=2000
    // Row 2: output[8]=3, output[9]=30, output[10]=4, output[11]=40
    // Row 3: output[12]=300, output[13]=3000, output[14]=400, output[15]=4000
    try std.testing.expectEqual(@as(f32, 1.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 10.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 20.0), output_data[3]);

    try std.testing.expectEqual(@as(f32, 100.0), output_data[4]);
    try std.testing.expectEqual(@as(f32, 1000.0), output_data[5]);
    try std.testing.expectEqual(@as(f32, 200.0), output_data[6]);
    try std.testing.expectEqual(@as(f32, 2000.0), output_data[7]);

    try std.testing.expectEqual(@as(f32, 3.0), output_data[8]);
    try std.testing.expectEqual(@as(f32, 30.0), output_data[9]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[10]);
    try std.testing.expectEqual(@as(f32, 40.0), output_data[11]);

    try std.testing.expectEqual(@as(f32, 300.0), output_data[12]);
    try std.testing.expectEqual(@as(f32, 3000.0), output_data[13]);
    try std.testing.expectEqual(@as(f32, 400.0), output_data[14]);
    try std.testing.expectEqual(@as(f32, 4000.0), output_data[15]);
}

test "Depth to Space 2x2 with intermediate kernel" {
    // Same as Depth to Space but with a kernel that doubles channel 'a' values
    const in_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var input_c: [4]f32 = .{ 100.0, 200.0, 300.0, 400.0 };
    var input_d: [4]f32 = .{ 1000.0, 2000.0, 3000.0, 4000.0 };
    var output_data = [_]f32{0} ** 16;

    const source_a = zpp.In(f32, &input_a, in_region.width, in_region);
    const source_b = zpp.In(f32, &input_b, in_region.width, in_region);
    const source_c = zpp.In(f32, &input_c, in_region.width, in_region);
    const source_d = zpp.In(f32, &input_d, in_region.width, in_region);
    const destination = zpp.Out(f32, &output_data, out_region.width, out_region);

    const zipped = zpp.Zip(.{ source_a, source_b, source_c, source_d });

    // Kernel that doubles channel 'a' values
    const double_a_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) struct { f32x4, f32x4, f32x4, f32x4 } {
            _ = ctx;
            const a, const b, const c, const d = in.get();
            return .{ a * @as(f32x4, @splat(2.0)), b, c, d };
        }
    };

    const grouped = zpp.GroupOut(2, 2, destination);

    const ctx = double_a_kernel.Context{};
    const result = zpp.Loop(f32x4, .{}, zipped, ctx, double_a_kernel.process);
    zpp.Process(result, grouped);

    // Same layout as before but 'a' values are doubled
    // Position (0,0): a=1*2=2, b=10, c=100, d=1000
    try std.testing.expectEqual(@as(f32, 2.0), output_data[0]); // a*2
    try std.testing.expectEqual(@as(f32, 10.0), output_data[1]); // b
    try std.testing.expectEqual(@as(f32, 4.0), output_data[2]); // a*2
    try std.testing.expectEqual(@as(f32, 20.0), output_data[3]); // b

    try std.testing.expectEqual(@as(f32, 100.0), output_data[4]); // c
    try std.testing.expectEqual(@as(f32, 1000.0), output_data[5]); // d
    try std.testing.expectEqual(@as(f32, 200.0), output_data[6]); // c
    try std.testing.expectEqual(@as(f32, 2000.0), output_data[7]); // d

    try std.testing.expectEqual(@as(f32, 6.0), output_data[8]); // a*2
    try std.testing.expectEqual(@as(f32, 30.0), output_data[9]); // b
    try std.testing.expectEqual(@as(f32, 8.0), output_data[10]); // a*2
    try std.testing.expectEqual(@as(f32, 40.0), output_data[11]); // b

    try std.testing.expectEqual(@as(f32, 300.0), output_data[12]); // c
    try std.testing.expectEqual(@as(f32, 3000.0), output_data[13]); // d
    try std.testing.expectEqual(@as(f32, 400.0), output_data[14]); // c
    try std.testing.expectEqual(@as(f32, 4000.0), output_data[15]); // d
}

test "Space to Depth 2x2" {
    // Space-to-depth: 1 channel of 4x4 -> 4 channels of 2x2
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    // Input 4x4:  [1,  2,  3,  4 ]
    //             [5,  6,  7,  8 ]
    //             [9,  10, 11, 12]
    //             [13, 14, 15, 16]
    var input: [16]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    var output_data_a: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_b: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_c: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_d: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.In(f32, &input, region.width, region);
    // Group the source - each 2x2 block becomes one pixel with 4 channels
    const grouped = zpp.Group(2, 2, source);

    const destination_region = grouped.getRegion();
    const expected_dst_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    try std.testing.expectEqual(expected_dst_region.width, destination_region.width);
    try std.testing.expectEqual(expected_dst_region.height, destination_region.height);

    const destination_a = zpp.Out(f32, &output_data_a, destination_region.width, destination_region);
    const destination_b = zpp.Out(f32, &output_data_b, destination_region.width, destination_region);
    const destination_c = zpp.Out(f32, &output_data_c, destination_region.width, destination_region);
    const destination_d = zpp.Out(f32, &output_data_d, destination_region.width, destination_region);

    // Zip the destinations to receive the 4 channels
    const zipped_dest = zpp.ZipOut(.{ destination_a, destination_b, destination_c, destination_d });

    zpp.Process(grouped, zipped_dest);

    // Group(2,2) extracts each 2x2 block as 4 channels in order: (0,0), (1,0), (0,1), (1,1)
    // Block(0,0): 1,2,5,6 -> a[0]=1, b[0]=2, c[0]=5, d[0]=6
    // Block(1,0): 3,4,7,8 -> a[1]=3, b[1]=4, c[1]=7, d[1]=8
    // Block(0,1): 9,10,13,14 -> a[2]=9, b[2]=10, c[2]=13, d[2]=14
    // Block(1,1): 11,12,15,16 -> a[3]=11, b[3]=12, c[3]=15, d[3]=16
    try std.testing.expectEqual(@as(f32, 1.0), output_data_a[0]);
    try std.testing.expectEqual(@as(f32, 3.0), output_data_a[1]);
    try std.testing.expectEqual(@as(f32, 9.0), output_data_a[2]);
    try std.testing.expectEqual(@as(f32, 11.0), output_data_a[3]);

    try std.testing.expectEqual(@as(f32, 2.0), output_data_b[0]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data_b[1]);
    try std.testing.expectEqual(@as(f32, 10.0), output_data_b[2]);
    try std.testing.expectEqual(@as(f32, 12.0), output_data_b[3]);

    try std.testing.expectEqual(@as(f32, 5.0), output_data_c[0]);
    try std.testing.expectEqual(@as(f32, 7.0), output_data_c[1]);
    try std.testing.expectEqual(@as(f32, 13.0), output_data_c[2]);
    try std.testing.expectEqual(@as(f32, 15.0), output_data_c[3]);

    try std.testing.expectEqual(@as(f32, 6.0), output_data_d[0]);
    try std.testing.expectEqual(@as(f32, 8.0), output_data_d[1]);
    try std.testing.expectEqual(@as(f32, 14.0), output_data_d[2]);
    try std.testing.expectEqual(@as(f32, 16.0), output_data_d[3]);
}

test "Space to Depth 2x2 With intermediate kernel" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input: [16]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    var output_data_a: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_b: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_c: [4]f32 = .{ 0, 0, 0, 0 };
    var output_data_d: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.In(f32, &input, region.width, region);
    // Group the source
    const grouped = zpp.Group(2, 2, source);

    const destination_region = grouped.getRegion();
    const expected_dst_region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    try std.testing.expectEqual(expected_dst_region.width, destination_region.width);
    try std.testing.expectEqual(expected_dst_region.height, destination_region.height);

    const destination_a = zpp.Out(f32, &output_data_a, destination_region.width, destination_region);
    const destination_b = zpp.Out(f32, &output_data_b, destination_region.width, destination_region);
    const destination_c = zpp.Out(f32, &output_data_c, destination_region.width, destination_region);
    const destination_d = zpp.Out(f32, &output_data_d, destination_region.width, destination_region);

    // Zip the destinations
    const zipped_dest = zpp.ZipOut(.{ destination_a, destination_b, destination_c, destination_d });

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
    const result = zpp.Loop(f32x4, .{}, grouped, ctx, split_kernel.process);
    zpp.Process(result, zipped_dest);

    // Verify: output_a = channel_a * 2, output_b/c/d = channels unchanged
    // Group(2,2) extracts 2x2 blocks: channel order is (0,0), (1,0), (0,1), (1,1)
    // Input 4x4:  1  2  3  4      Block(0,0): 1,2,5,6   Block(1,0): 3,4,7,8
    //             5  6  7  8      Block(0,1): 9,10,13,14 Block(1,1): 11,12,15,16
    //             9 10 11 12
    //            13 14 15 16
    // Channel a (top-left of each block, doubled): 1*2, 3*2, 9*2, 11*2 = 2, 6, 18, 22
    try std.testing.expectEqual(@as(f32, 2.0), output_data_a[0]);
    try std.testing.expectEqual(@as(f32, 6.0), output_data_a[1]);
    try std.testing.expectEqual(@as(f32, 18.0), output_data_a[2]);
    try std.testing.expectEqual(@as(f32, 22.0), output_data_a[3]);

    // Channel b (top-right of each block): 2, 4, 10, 12
    try std.testing.expectEqual(@as(f32, 2.0), output_data_b[0]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data_b[1]);
    try std.testing.expectEqual(@as(f32, 10.0), output_data_b[2]);
    try std.testing.expectEqual(@as(f32, 12.0), output_data_b[3]);

    // Channel c (bottom-left of each block): 5, 7, 13, 15
    try std.testing.expectEqual(@as(f32, 5.0), output_data_c[0]);
    try std.testing.expectEqual(@as(f32, 7.0), output_data_c[1]);
    try std.testing.expectEqual(@as(f32, 13.0), output_data_c[2]);
    try std.testing.expectEqual(@as(f32, 15.0), output_data_c[3]);

    // Channel d (bottom-right of each block): 6, 8, 14, 16
    try std.testing.expectEqual(@as(f32, 6.0), output_data_d[0]);
    try std.testing.expectEqual(@as(f32, 8.0), output_data_d[1]);
    try std.testing.expectEqual(@as(f32, 14.0), output_data_d[2]);
    try std.testing.expectEqual(@as(f32, 16.0), output_data_d[3]);
}

//! Tests for zip.zig - Zip and Unzip support

const std = @import("std");
const zpp = @import("zpp");

const f32x4 = @Vector(4, f32);

test "Unzip source expressions" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Create two input sources
    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };

    const source_a = zpp.In(f32, &input_a, region.width, region);
    const source_b = zpp.In(f32, &input_b, region.width, region);

    // Zip them together
    const zipped = zpp.Zip(.{ source_a, source_b });

    // Unzip them
    const unzipped = zpp.Unzip(zipped);

    // Verify that unzipped[0] and unzipped[1] can be used independently
    // The unzipped sources should have the same region
    try std.testing.expectEqual(region.width, unzipped[0].getRegion().width);
    try std.testing.expectEqual(region.height, unzipped[0].getRegion().height);
    try std.testing.expectEqual(region.width, unzipped[1].getRegion().width);
    try std.testing.expectEqual(region.height, unzipped[1].getRegion().height);
}

test "Zip and process two sources" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source_a = zpp.In(f32, &input_a, region.width, region);
    const source_b = zpp.In(f32, &input_b, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Zip the sources
    const zipped = zpp.Zip(.{ source_a, source_b });

    // Process with a kernel that adds the two sources
    const add_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            // Get both values and add them
            const values = in.get();
            return values[0] + values[1];
        }
    };

    const ctx = add_kernel.Context{};
    const result = zpp.Loop(f32x4, .{}, zipped, ctx, add_kernel.process);
    zpp.Process(result, destination);

    // Verify: output = input_a + input_b
    try std.testing.expectEqual(@as(f32, 11.0), output_data[0]); // 1 + 10
    try std.testing.expectEqual(@as(f32, 22.0), output_data[1]); // 2 + 20
    try std.testing.expectEqual(@as(f32, 33.0), output_data[2]); // 3 + 30
    try std.testing.expectEqual(@as(f32, 44.0), output_data[3]); // 4 + 40
}

test "Zip with margins" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source_a = zpp.In(f32, &input_a, region.width, region);
    const source_b = zpp.In(f32, &input_b, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    const zipped = zpp.Zip(.{ source_a, source_b });

    // Kernel that uses margins on both zipped sources
    const blur_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            // Sum left, center, right from source A, multiply by center from B
            const left = in.getAt(-1, 0);
            const center = in.getAt(0, 0);
            const right = in.getAt(1, 0);
            return (left[0] + center[0] + right[0]) * center[1] / @as(f32x4, @splat(10.0));
        }
    };

    const ctx = blur_kernel.Context{};
    const result = zpp.Loop(f32x4, .{ .margin = zpp.marginH(1) }, zipped, ctx, blur_kernel.process);
    zpp.Process(result, destination);

    // Verify computation worked (values should be non-zero)
    try std.testing.expect(output_data[0] > 0);
    try std.testing.expect(output_data[1] > 0);
    try std.testing.expect(output_data[2] > 0);
    try std.testing.expect(output_data[3] > 0);
}

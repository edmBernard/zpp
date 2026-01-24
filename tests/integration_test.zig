//! Integration tests that use multiple ZPP modules together

const std = @import("std");
const zpp = @import("zpp");

/// Common SIMD vector types
const f32x4 = @Vector(4, f32);

test "integration: generator to output" {
    const reg: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var output: [4]f32 = .{ 0, 0, 0, 0 };

    const destination = zpp.Out(f32, &output, reg.width, reg);

    const processing_kernel = struct {
        const Context = struct {
            scale: f32x4 = f32x4{ 1.0, 1.0, 1.0, 1.0 },
            offset: f32x4 = f32x4{ 1.0, 1.0, 1.0, 1.0 },
        };

        fn process(ctx: Context, x: f32x4, y: f32x4) f32x4 {
            return x / ctx.scale + y / ctx.scale + ctx.offset;
        }
    };

    const ctx = processing_kernel.Context{};
    const result = zpp.Generate(f32x4, reg, ctx, processing_kernel.process);
    zpp.Process(result, destination);

    try std.testing.expectEqual(@as(f32, 1.0), output[0]);
    try std.testing.expectEqual(@as(f32, 2.0), output[1]);
    try std.testing.expectEqual(@as(f32, 3.0), output[2]);
    try std.testing.expectEqual(@as(f32, 4.0), output[3]);
}

test "integration: zip three sources" {
    const reg: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_a: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var input_b: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var input_c: [4]f32 = .{ 100.0, 200.0, 300.0, 400.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source_a = zpp.In(f32, &input_a, reg.width, reg);
    const source_b = zpp.In(f32, &input_b, reg.width, reg);
    const source_c = zpp.In(f32, &input_c, reg.width, reg);
    const destination = zpp.Out(f32, &output_data, reg.width, reg);

    // Zip the sources
    const zipped = zpp.Zip(.{ source_a, source_b, source_c });

    // Process with a kernel that adds the three sources
    const add_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            // Get all values as a tuple and add them
            const a, const b, const c = in.get();
            return a + b + c;
        }
    };

    const ctx = add_kernel.Context{};
    const result = zpp.Loop(f32x4, .{}, zipped, ctx, add_kernel.process);
    zpp.Process(result, destination);

    // Verify: output = input_a + input_b + input_c
    try std.testing.expectEqual(@as(f32, 111.0), output_data[0]); // 1 + 10 + 100
    try std.testing.expectEqual(@as(f32, 222.0), output_data[1]); // 2 + 20 + 200
    try std.testing.expectEqual(@as(f32, 333.0), output_data[2]); // 3 + 30 + 300
    try std.testing.expectEqual(@as(f32, 444.0), output_data[3]); // 4 + 40 + 400
}

test "integration: stats min/max" {
    const reg: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_data: [4]f32 = .{ 5.0, 2.0, 8.0, 1.0 };

    const source = zpp.In(f32, &input_data, reg.width, reg);

    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    const minmax_kernel = struct {
        const Context = struct {
            min_val: f32 = std.math.inf(f32),
            max_val: f32 = -std.math.inf(f32),
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            ctx.min_val = @min(ctx.min_val, @reduce(.Min, values));
            ctx.max_val = @max(ctx.max_val, @reduce(.Max, values));
        }
    };

    var stats_ctx = minmax_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, reg, minmax_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    try std.testing.expectEqual(@as(f32, 1.0), stats_ctx.min_val);
    try std.testing.expectEqual(@as(f32, 8.0), stats_ctx.max_val);
}

test "integration: group and ungroup roundtrip" {
    const reg: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    var input_data: [16]f32 = undefined;
    for (0..16) |i| {
        input_data[i] = @floatFromInt(i);
    }

    const source = zpp.In(f32, &input_data, reg.width, reg);

    // Group 2x2, then ungroup - should get back original dimensions
    const grouped = zpp.Group(2, 2, source);
    const ungrouped = zpp.Ungroup(2, 2, grouped);

    const ungrouped_region = ungrouped.getRegion();

    // Input: 4x4
    // Grouped (downscaled by 2x2): ceil(4/2)=2, ceil(4/2)=2 -> 2x2
    // Ungrouped (upscaled by 2x2): 2*2=4, 2*2=4 -> 4x4
    try std.testing.expectEqual(@as(u32, 4), ungrouped_region.width);
    try std.testing.expectEqual(@as(u32, 4), ungrouped_region.height);
}

test "integration: interp loop identity" {
    const reg: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    // Input: [[1, 2, 3, 4], [5, 6, 7, 8]]
    var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output_data: [8]f32 = .{0} ** 8;

    const source = zpp.In(f32, &input_data, reg.width, reg);
    const destination = zpp.Out(f32, &output_data, reg.width, reg);

    // Identity transform kernel
    const interp_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
            _ = ctx;
            return interp.sample(x, y);
        }
    };

    const ctx = interp_kernel.Context{};
    const result = zpp.InterpLoop(f32x4, .Nearest, source, reg, ctx, interp_kernel.process);
    zpp.Process(result, destination);

    // Identity transform should copy input to output
    try std.testing.expectEqual(@as(f32, 1.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[3]);
    try std.testing.expectEqual(@as(f32, 5.0), output_data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), output_data[5]);
}

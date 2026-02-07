//! Tests for stats.zig - Stats destination

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };

test "Stats destination: remainder handling with non-aligned width" {
    // Test that stats correctly handles width not divisible by vec_len
    // Width 5 is not divisible by vec_len 4
    // Should process 4 elements in batch, then 1 as remainder
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 1 };

    var input_data: [5]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    const source = zpp.In(f32, &input_data, region.width, region);

    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    // Sum accumulator
    const sum_kernel = struct {
        const Context = struct {
            sum: f32 = 0,
            count: u32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            ctx.sum += @reduce(.Add, values);
            // Count non-zero values (works because we zero out unused lanes)
            inline for (0..4) |i| {
                if (values[i] != 0) ctx.count += 1;
            }
        }
    };

    var stats_ctx = sum_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, region, sum_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    // Sum should be 1+2+3+4+5 = 15
    // Count should be 5 (not 8 which would happen if we counted all 4 lanes twice)
    try std.testing.expectEqual(@as(f32, 15.0), stats_ctx.sum);
    try std.testing.expectEqual(@as(u32, 5), stats_ctx.count);
}

// MARK: Stats destination: sum accumulation with aligned width
test "Stats destination: sum accumulation with aligned width" {
    // Width=4 (aligned to vec_len), 2 rows - ensures no double-counting
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const source = zpp.In(f32, &input_data, region.width, region);

    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    const sum_kernel = struct {
        const Context = struct {
            sum: f32 = 0,
            count: u32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            ctx.sum += @reduce(.Add, values);
            ctx.count += 4;
        }
    };

    var stats_ctx = sum_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, region, sum_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    // Sum: 1+2+3+4+5+6+7+8 = 36
    try std.testing.expectEqual(@as(f32, 36.0), stats_ctx.sum);
    // Count: 2 batches of 4 = 8
    try std.testing.expectEqual(@as(u32, 8), stats_ctx.count);
}

// MARK: Stats destination: min/max tracking
test "Stats destination: min/max tracking" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    // Place min and max at specific known positions
    var input_data: [8]f32 = .{ 5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 0.5, 6.0 };

    const source = zpp.In(f32, &input_data, region.width, region);

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
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, region, minmax_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    try std.testing.expectEqual(@as(f32, 0.5), stats_ctx.min_val);
    try std.testing.expectEqual(@as(f32, 8.0), stats_ctx.max_val);
}

// MARK: Stats destination: compute on the given region
test "Stats destination: compute on the given region" {
    const image_width = 9;
    const image_height = 5;
    const region_in: zpp.Region = .{ .x = 0, .y = 0, .width = image_width, .height = image_height };
    const region_stat: zpp.Region = .{ .x = 2, .y = 1, .width = 4, .height = 2 };

    var input_data = [_]f32{0} ** (image_width * image_height);
    th.fillRamp(f32, &input_data, 1, 1);

    const source = zpp.In(f32, &input_data, region_in.width, region_in);

    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };
    const stat_kernel = struct {
        const Context = struct {
            sum_val: f32 = 0,
        };

        fn accumulate(ctx: *Context, values: anytype) void {
            ctx.sum_val += @reduce(.Add, values);
        }
    };

    var stats_ctx = stat_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, region_stat, stat_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    // 0, 0, 0,  0,  0,  0,  0, 0, 0,
    // 0, 0, 12, 13, 14, 15, 0, 0, 0,
    // 0, 0, 21, 22, 23, 24, 0, 0, 0,
    // 0, 0, 0,  0,  0,  0,  0, 0, 0,
    // 0, 0, 0,  0,  0,  0,  0, 0, 0,
    try std.testing.expectEqual(@as(f32, 144), stats_ctx.sum_val);
}

// TODO: this test should work
// // MARK: Stats destination: compute stat directly from source
// test "Stats destination: compute stat directly from source" {
//     const image_width = 9;
//     const image_height = 5;
//     const region_in: zpp.Region = .{ .x = 0, .y = 0, .width = image_width, .height = image_height };
//     const region_stat: zpp.Region = .{ .x = 2, .y = 1, .width = 4, .height = 2 };

//     var input_data = [_]f32{0} ** (image_width * image_height);
//     th.fillRamp(f32, &input_data, 1, 1);

//     const source = zpp.In(f32, &input_data, region_in.width, region_in);

//     const stat_kernel = struct {
//         const Context = struct {
//             sum_val: f32 = 0,
//         };

//         fn accumulate(ctx: *Context, values: anytype) void {
//             ctx.sum_val += @reduce(.Add, values);
//         }
//     };

//     var stats_ctx = stat_kernel.Context{};
//     const stats_dest = zpp.Stats(f32x4, &stats_ctx, region_stat, stat_kernel.accumulate);
//     zpp.Process(source, stats_dest);

//     // 0, 0, 0,  0,  0,  0,  0, 0, 0,
//     // 0, 0, 12, 13, 14, 15, 0, 0, 0,
//     // 0, 0, 21, 22, 23, 24, 0, 0, 0,
//     // 0, 0, 0,  0,  0,  0,  0, 0, 0,
//     // 0, 0, 0,  0,  0,  0,  0, 0, 0,
//     try std.testing.expectEqual(@as(f32, 144), stats_ctx.sum_val);
// }

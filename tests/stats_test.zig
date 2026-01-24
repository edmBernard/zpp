//! Tests for stats.zig - Stats destination

const std = @import("std");
const zpp = @import("zpp");

const f32x4 = @Vector(4, f32);

test "Stats destination: sum accumulation" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    // Input: [[1, 2, 3, 4], [5, 6, 7, 8]] -> sum = 36
    var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const source = zpp.In(f32, &input_data, region.width, region);

    // Identity kernel to pass through values
    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    // Stats accumulator
    const stats_kernel = struct {
        const Context = struct {
            sum: f32 = 0,
            count: u32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            ctx.sum += @reduce(.Add, values);
            ctx.count += 4;
        }
    };

    var stats_ctx = stats_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, region, stats_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    // Verify accumulated sum
    try std.testing.expectEqual(@as(f32, 36.0), stats_ctx.sum); // 1+2+3+4+5+6+7+8 = 36
    try std.testing.expectEqual(@as(u32, 8), stats_ctx.count);
}

test "Stats destination: min/max tracking" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_data: [4]f32 = .{ 5.0, 2.0, 8.0, 1.0 };

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

    try std.testing.expectEqual(@as(f32, 1.0), stats_ctx.min_val);
    try std.testing.expectEqual(@as(f32, 8.0), stats_ctx.max_val);
}

test "Stats destination with coordinates" {
    const region: zpp.Region = .{ .x = 10, .y = 20, .width = 4, .height = 1 };

    var input_data: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };

    const source = zpp.In(f32, &input_data, region.width, region);

    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    // Stats kernel that tracks coordinates of max value
    const coord_kernel = struct {
        const Context = struct {
            max_val: f32 = -std.math.inf(f32),
            max_x: i32 = 0,
            max_y: i32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4, x: @Vector(4, i32), y: @Vector(4, i32)) void {
            inline for (0..4) |i| {
                if (values[i] > ctx.max_val) {
                    ctx.max_val = values[i];
                    ctx.max_x = x[i];
                    ctx.max_y = y[i];
                }
            }
        }
    };

    var stats_ctx = coord_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.StatsWithCoords(f32x4, &stats_ctx, region, coord_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    try std.testing.expectEqual(@as(f32, 4.0), stats_ctx.max_val);
    try std.testing.expectEqual(@as(i32, 13), stats_ctx.max_x); // x=10 + offset 3
    try std.testing.expectEqual(@as(i32, 20), stats_ctx.max_y);
}

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

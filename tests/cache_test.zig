//! Tests for cache.zig - Row caching for expression trees

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

/// Common SIMD vector types
const f32x4 = @Vector(4, f32);

// MARK: Cache: All loop result fit in cache
test "Cache: All loop result fit in cache" {
    // we transform the input and write to both a normal destination and a Stats accumulator
    // the stat is computed on the transformed value
    // The cache is big enough to contain completly the loop result, so each row is computed only once and reused for both the normal destination and the stat accumulator
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
        const Context = struct {
            calls: u32 = 0,
        };

        fn process(ctx: *Context, in: anytype) f32x4 {
            ctx.calls += 1;
            return in.get() * th.splatWithCast(f32x4, 10);
        }
    };

    var ctx = kernel.Context{};
    const cached = try zpp.cachedLoop(f32x4, .{}, 2, std.testing.allocator, source, &ctx, kernel.process);
    defer cached.deinit();
    const cached_view = cached.view();
    const copied_view = cached_view;
    const zipped_in = zpp.zip(.{ cached_view, copied_view });
    const zipped_dest = zpp.zipDest(.{ destination, stats_dest });
    zpp.process(zipped_in, zipped_dest);

    const expected_data = [_]f32{
        10, 20, 30, 40,
        50, 60, 70, 80,
    };
    try std.testing.expectEqual(expected_data, output_data);

    try std.testing.expectEqual(360, stats_ctx.sum);

    try std.testing.expectEqual(2, ctx.calls);
}

// MARK: Cache: Smaller cache than loop result
test "Cache: Smaller cache than loop result" {
    // we transform the input and write to both a normal destination and a Stats accumulator
    // the stat is computed on the transformed value
    // The cache is only contain one line but each line is still computed only once
    // because the processing is piped and not sequential
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
        const Context = struct {
            calls: u32 = 0,
        };

        fn process(ctx: *Context, in: anytype) f32x4 {
            ctx.calls += 1;
            return in.get() * th.splatWithCast(f32x4, 10);
        }
    };

    var ctx = kernel.Context{};
    const cached = try zpp.cachedLoop(f32x4, .{}, 1, std.testing.allocator, source, &ctx, kernel.process);
    defer cached.deinit();
    const cached_view = cached.view();
    const zipped_in = zpp.zip(.{ cached_view, cached_view });
    const zipped_dest = zpp.zipDest(.{ destination, stats_dest });
    zpp.process(zipped_in, zipped_dest);

    const expected_data = [_]f32{
        10, 20, 30, 40,
        50, 60, 70, 80,
    };
    try std.testing.expectEqual(expected_data, output_data);

    try std.testing.expectEqual(360, stats_ctx.sum);

    try std.testing.expectEqual(2, ctx.calls);
}

// MARK: Cache: Owner/view split allows direct processing and copied views
test "Cache: Owner/view split keeps ownership explicit" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    var input_data: [8]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    var output_data = [_]f32{0} ** 8;

    const source = try zpp.makeSource(f32, &input_data, region.width, region);
    const destination = try zpp.makeDest(f32, &output_data, region.width, region);

    const kernel = struct {
        const Context = struct {
            calls: u32 = 0,
        };

        fn process(ctx: *Context, in: anytype) f32x4 {
            ctx.calls += 1;
            return in.get() * th.splatWithCast(f32x4, 2);
        }
    };

    var ctx = kernel.Context{};
    const cached = try zpp.cachedLoop(f32x4, .{}, 1, std.testing.allocator, source, &ctx, kernel.process);
    defer cached.deinit();

    const cached_view = cached.view();
    const copied_view = cached_view;
    const zipped = zpp.zip(.{ cached_view, copied_view });
    const unzipped = zpp.unzip(zipped);
    zpp.process(unzipped[0], destination);

    const expected_data = [_]f32{
        2,  4,  6,  8,
        10, 12, 14, 16,
    };
    try std.testing.expectEqual(expected_data, output_data);
    try std.testing.expectEqual(@as(u32, 2), ctx.calls);
}

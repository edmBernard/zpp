//! Cache Benchmark
//! Benchmarks cached loop with a separable 3x3 box blur (horizontal pass cached,
//! then vertical pass) on a 2048x2048 image.

const std = @import("std");
const zpp = @import("zpp");

const f32v = zpp.f32v;
const vec_len = @typeInfo(f32v).vector.len;

/// Horizontal box blur kernel (3-tap: left, center, right)
fn hBlurKernel(_: void, in: anytype) f32v {
    const third: f32v = @splat(1.0 / 3.0);
    return (in.getAt(-1, 0) + in.get() + in.getAt(1, 0)) * third;
}

/// Vertical box blur kernel (3-tap: top, center, bottom)
fn vBlurKernel(_: void, in: anytype) f32v {
    const third: f32v = @splat(1.0 / 3.0);
    return (in.getAt(0, -1) + in.get() + in.getAt(0, 1)) * third;
}

const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    const width: u32 = 2048;
    const height: u32 = 2048;
    const num_iterations = 5;

    std.debug.print("Cache Benchmark (Separable 3x3 Box Blur)\n", .{});
    std.debug.print("=========================================\n", .{});
    std.debug.print("Image: {d}x{d}, Iterations: {d}\n", .{ width, height, num_iterations });
    std.debug.print("SIMD vector length: {d}\n\n", .{vec_len});

    // Allocate source data
    const src_data = try allocator.alloc(f32, width * height);
    defer allocator.free(src_data);
    for (0..src_data.len) |i| {
        src_data[i] = @as(f32, @floatFromInt(i % 256)) / 255.0;
    }

    // Allocate output data
    const dst_data = try allocator.alloc(f32, width * height);
    defer allocator.free(dst_data);

    const region = zpp.Region{ .x = 0, .y = 0, .width = width, .height = height };
    const source = try zpp.makeSource(f32, src_data, width, region);
    const dest = try zpp.makeDest(f32, dst_data, width, region);

    var times_with_cache: [num_iterations]u64 = undefined;
    var times_without_cache: [num_iterations]u64 = undefined;

    for (0..num_iterations) |iter| {
        const timer_start = Io.Timestamp.now(io, .awake);

        // Horizontal blur with cache (vertical margin for the subsequent vertical pass)
        const h_blur = try zpp.cachedLoop(
            f32v,
            .{ .margin = .horizontal(1) },
            3, // cache 3 rows for vertical margin
            allocator,
            source,
            {},
            hBlurKernel,
        );
        defer h_blur.deinit();

        // Vertical blur reading from the cached horizontal blur
        const v_blur = zpp.loop(
            f32v,
            .{ .margin = .vertical(1) },
            h_blur.view(),
            {},
            vBlurKernel,
        );

        zpp.process(v_blur, dest);

        times_with_cache[iter] = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    }

    for (0..num_iterations) |iter| {
        const timer_start = Io.Timestamp.now(io, .awake);

        // Horizontal blur without cache (vertical margin for the subsequent vertical pass)
        const h_blur = zpp.loop(
            f32v,
            .{ .margin = .horizontal(1) },
            source,
            {},
            hBlurKernel,
        );

        // Vertical blur reading from the horizontal blur
        const v_blur = zpp.loop(
            f32v,
            .{ .margin = .vertical(1) },
            h_blur,
            {},
            vBlurKernel,
        );

        zpp.process(v_blur, dest);

        times_without_cache[iter] = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    }

    // Sort to find median
    std.mem.sort(u64, &times_with_cache, {}, std.sort.asc(u64));
    std.mem.sort(u64, &times_without_cache, {}, std.sort.asc(u64));
    const median_ns_with_cache = times_with_cache[num_iterations / 2];
    const median_ns_without_cache = times_without_cache[num_iterations / 2];
    const median_ms_with_cache = @as(f64, @floatFromInt(median_ns_with_cache)) / 1_000_000.0;
    const median_ms_without_cache = @as(f64, @floatFromInt(median_ns_without_cache)) / 1_000_000.0;
    const total_pixels: f64 = @floatFromInt(@as(u64, width) * @as(u64, height));
    const mpixels_per_sec_with_cache = total_pixels / (@as(f64, @floatFromInt(median_ns_with_cache)) / 1_000_000_000.0) / 1_000_000.0;
    const mpixels_per_sec_without_cache = total_pixels / (@as(f64, @floatFromInt(median_ns_without_cache)) / 1_000_000_000.0) / 1_000_000.0;

    std.debug.print("  Separable blur with cache: {d:.1} ms  ({d:.1} Mpix/s)\n", .{ median_ms_with_cache, mpixels_per_sec_with_cache });
    std.debug.print("  Separable blur without cache: {d:.1} ms  ({d:.1} Mpix/s)\n", .{ median_ms_without_cache, mpixels_per_sec_without_cache });
    std.debug.print("\nDone.\n", .{});
}

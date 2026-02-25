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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

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
    const source = zpp.makeSource(f32, src_data, width, region);
    const dest = zpp.makeDest(f32, dst_data, width, region);

    var times: [num_iterations]u64 = undefined;

    for (0..num_iterations) |iter| {
        var timer = std.time.Timer.start() catch unreachable;

        // Horizontal blur with cache (vertical margin for the subsequent vertical pass)
        const h_blur = try zpp.cachedLoop(
            f32v,
            .{ .margin = .{ .left = 1, .right = 1 } },
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
            .{ .margin = .{ .top = 1, .bottom = 1 } },
            h_blur,
            {},
            vBlurKernel,
        );

        zpp.process(v_blur, dest);

        times[iter] = timer.read();
    }

    // Sort to find median
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const median_ns = times[num_iterations / 2];
    const median_ms = @as(f64, @floatFromInt(median_ns)) / 1_000_000.0;
    const total_pixels: f64 = @floatFromInt(@as(u64, width) * @as(u64, height));
    const mpixels_per_sec = total_pixels / (@as(f64, @floatFromInt(median_ns)) / 1_000_000_000.0) / 1_000_000.0;

    std.debug.print("  Separable blur: {d:.1} ms  ({d:.1} Mpix/s)\n", .{ median_ms, mpixels_per_sec });
    std.debug.print("\nDone.\n", .{});
}

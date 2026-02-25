//! Interpolation Benchmark
//! Benchmarks nearest, linear, and cubic interpolation methods doing a 2x upscale
//! on a 2048x2048 source to 4096x4096 output.

const std = @import("std");
const zpp = @import("zpp");

const f32v = zpp.f32v;
const vec_len = @typeInfo(f32v).vector.len;

/// Resize kernel using interpolated sampling (2x upscale: sample at half coordinates)
fn resizeKernel(_: void, interp: anytype, x: f32v, y: f32v) f32v {
    const half: f32v = @splat(0.5);
    return interp.sample(x * half, y * half);
}

fn runBenchmark(
    comptime method: zpp.InterpolationMethod,
    comptime label: []const u8,
    source: anytype,
    output_region: zpp.Region,
    dest: anytype,
    comptime num_iterations: u32,
) void {
    var times: [num_iterations]u64 = undefined;

    for (0..num_iterations) |iter| {
        var timer = std.time.Timer.start() catch unreachable;

        const interp = zpp.interpLoop(f32v, method, source, output_region, {}, resizeKernel);
        zpp.process(interp, dest);

        times[iter] = timer.read();
    }

    // Sort to find median
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const median_ns = times[num_iterations / 2];
    const median_ms = @as(f64, @floatFromInt(median_ns)) / 1_000_000.0;
    const total_pixels: f64 = @floatFromInt(@as(u64, output_region.width) * @as(u64, output_region.height));
    const mpixels_per_sec = total_pixels / (@as(f64, @floatFromInt(median_ns)) / 1_000_000_000.0) / 1_000_000.0;

    std.debug.print("  {s}: {d:.1} ms  ({d:.1} Mpix/s)\n", .{ label, median_ms, mpixels_per_sec });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const src_w: u32 = 2048;
    const src_h: u32 = 2048;
    const dst_w: u32 = 4096;
    const dst_h: u32 = 4096;
    const num_iterations = 5;

    std.debug.print("Interpolation Benchmark\n", .{});
    std.debug.print("=======================\n", .{});
    std.debug.print("Source: {d}x{d}, Output: {d}x{d}, Iterations: {d}\n", .{ src_w, src_h, dst_w, dst_h, num_iterations });
    std.debug.print("SIMD vector length: {d}\n\n", .{vec_len});

    // Allocate source data
    const src_data = try allocator.alloc(f32, src_w * src_h);
    defer allocator.free(src_data);
    for (0..src_data.len) |i| {
        src_data[i] = @as(f32, @floatFromInt(i % 256)) / 255.0;
    }

    // Allocate output data
    const dst_data = try allocator.alloc(f32, dst_w * dst_h);
    defer allocator.free(dst_data);

    const src_region = zpp.Region{ .x = 0, .y = 0, .width = src_w, .height = src_h };
    const dst_region = zpp.Region{ .x = 0, .y = 0, .width = dst_w, .height = dst_h };

    const source = zpp.makeSource(f32, src_data, src_w, src_region);
    const dest = zpp.makeDest(f32, dst_data, dst_w, dst_region);

    runBenchmark(.nearest, "Nearest", source, dst_region, dest, num_iterations);
    runBenchmark(.linear, "Linear ", source, dst_region, dest, num_iterations);
    runBenchmark(.cubic, "Cubic  ", source, dst_region, dest, num_iterations);

    std.debug.print("\nDone.\n", .{});
}

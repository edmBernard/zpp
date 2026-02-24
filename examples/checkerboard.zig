//! Checkerboard Example
//! Generates a simple checkerboard pattern and saves it as a PPM file.
//!
//! This example demonstrates:
//! - Using zpp.Generate to create procedural textures from coordinates
//! - Basic SIMD operations for pattern generation
//! - Converting to RGB output using zpp.InterleavedOut

const std = @import("std");
const zpp = @import("zpp");

// ============================================================================
// MARK: SIMD Vector Configuration
// ============================================================================

const u8v = zpp.u8v;
const vec_len = @typeInfo(u8v).vector.len;
const i32v = zpp.VectorLike(u8v, i32);
const f32v = zpp.VectorLike(u8v, f32);

/// Convenience alias for zpp.math.splat with VecF32
inline fn splat(scalar: f32) f32v {
    return zpp.math.splat(f32v, scalar);
}

// ============================================================================
// MARK: ZPP Processing Kernel
// ============================================================================

/// Kernel context for checkerboard generation
const CheckerboardContext = struct {
    /// Size of each checker square in pixels
    square_size: f32v,
};

/// Checkerboard generator kernel for zpp.Generate.
/// Returns RGB values as u8 (black or white based on checker pattern).
fn checkerboardProcess(ctx: CheckerboardContext, x: f32v, y: f32v) [3]u8v {
    // Compute which square we're in
    const col: i32v = @intFromFloat(x / ctx.square_size);
    const row: i32v = @intFromFloat(y / ctx.square_size);

    // XOR the column and row parity to get checkerboard pattern
    const parity = (col ^ row) & @as(i32v, @splat(1));
    // Convert to u8: 0 -> black (0), 1 -> white (255)
    const value: u8v = @intCast(parity * @as(i32v, @splat(255)));

    // Output as grayscale RGB
    return .{ value, value, value };
}

// ============================================================================
// MARK: Image Generation using ZPP
// ============================================================================

/// Generate a checkerboard image using zpp primitives.
fn generateImage(allocator: std.mem.Allocator, width: u32, height: u32, square_size: f32) ![]u8 {
    const size = width * height * 3;
    const data = try allocator.alloc(u8, size);
    @memset(data, 0);

    const context = CheckerboardContext{
        .square_size = splat(square_size),
    };

    const region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };

    const destination = zpp.InterleavedOut(u8, 3, data, width, region);
    const generator = zpp.Generate(f32v, context, checkerboardProcess);
    zpp.Process(generator, destination);

    return data;
}

// ============================================================================
// MARK: PPM Image Output
// ============================================================================

fn writePPM(filename: []const u8, data: []const u8, width: u32, height: u32) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var header_buf: [64]u8 = undefined;
    const header = std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ width, height }) catch unreachable;
    try file.writeAll(header);
    try file.writeAll(data);
}

// ============================================================================
// MARK: Main Entry Point
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const width: u32 = 800;
    const height: u32 = 600;
    const square_size: f32 = 50.0;

    std.debug.print("Checkerboard Example using ZPP\n", .{});
    std.debug.print("==============================\n", .{});
    std.debug.print("SIMD vector length: {d}\n", .{vec_len});
    std.debug.print("Generating {d}x{d} image with {d}px squares...\n", .{ width, height, @as(u32, @intFromFloat(square_size)) });

    const start = std.time.milliTimestamp();

    const image_data = try generateImage(allocator, width, height, square_size);
    defer allocator.free(image_data);

    const elapsed = std.time.milliTimestamp() - start;
    std.debug.print("Generation completed in {d}ms\n", .{elapsed});

    const filename = "checkerboard.ppm";
    try writePPM(filename, image_data, width, height);
    std.debug.print("Image saved to: {s}\n", .{filename});
}

// ============================================================================
// MARK: Tests
// ============================================================================

test "checkerboard kernel produces valid RGB" {
    const ctx = CheckerboardContext{
        .square_size = splat(10.0),
    };

    // Test at origin (should be one color)
    const rgb0 = checkerboardProcess(ctx, splat(0.0), splat(0.0));
    for (0..vec_len) |i| {
        try std.testing.expect(rgb0[0][i] == 0 or rgb0[0][i] == 255);
        try std.testing.expectEqual(rgb0[0][i], rgb0[1][i]);
        try std.testing.expectEqual(rgb0[0][i], rgb0[2][i]);
    }
}

test "checkerboard pattern alternates" {
    const ctx = CheckerboardContext{
        .square_size = splat(10.0),
    };

    // Get values at two adjacent squares
    const rgb1 = checkerboardProcess(ctx, splat(5.0), splat(5.0)); // center of first square
    const rgb2 = checkerboardProcess(ctx, splat(15.0), splat(5.0)); // center of second square

    // They should be opposite colors
    for (0..vec_len) |i| {
        try std.testing.expect(rgb1[0][i] != rgb2[0][i]);
    }
}

test "zpp.Region integration" {
    const region = zpp.Region{ .x = 0, .y = 0, .width = 16, .height = 16 };
    try std.testing.expectEqual(@as(u32, 256), region.area());
    try std.testing.expectEqual(@as(u32, 16), region.width);
}

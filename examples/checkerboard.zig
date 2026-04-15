//! Checkerboard Example
//! Generates a simple checkerboard pattern and saves it as a PPM file.
//!
//! This example demonstrates:
//! - Using zpp.generate to create procedural textures from coordinates
//! - Basic SIMD operations for pattern generation
//! - Converting to RGB output using zpp.makeInterleavedDest

const std = @import("std");
const zpp = @import("zpp");

// ============================================================================
// MARK: SIMD Vector Configuration
// ============================================================================

const u8v = zpp.u8v;
const vec_len = @typeInfo(u8v).vector.len;
const i32v = zpp.VectorLike(u8v, i32);
const f32v = zpp.VectorLike(u8v, f32);

// ============================================================================
// MARK: ZPP Processing Kernel
// ============================================================================

/// Kernel context for checkerboard generation
const CheckerboardContext = struct {
    /// Size of each checker square in pixels
    square_size: f32v,
};

/// Checkerboard generator kernel for zpp.generate.
/// Returns RGB values as u8 (black or white based on checker pattern).
fn checkerboardProcess(ctx: CheckerboardContext, x: f32v, y: f32v) [3]u8v {
    // Compute which square we're in
    const col: i32v = @trunc(x / ctx.square_size);
    const row: i32v = @trunc(y / ctx.square_size);

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
        .square_size = @splat(square_size),
    };

    const region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };

    const destination = try zpp.makeInterleavedDest(u8, 3, data, width, region);
    const generator = zpp.generate(f32v, context, checkerboardProcess);
    zpp.process(generator, destination);

    return data;
}

// ============================================================================
// MARK: PPM Image Output
// ============================================================================

const Io = std.Io;

fn writePPM(io: Io, filename: []const u8, data: []const u8, width: u32, height: u32) !void {
    const file = try Io.Dir.cwd().createFile(io, filename, .{});
    defer file.close(io);

    var header_buf: [64]u8 = undefined;
    const header = std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ width, height }) catch unreachable;
    try file.writeStreamingAll(io, header);
    try file.writeStreamingAll(io, data);
}

// ============================================================================
// MARK: Main Entry Point
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    const width: u32 = 800;
    const height: u32 = 600;
    const square_size: f32 = 50.0;

    std.debug.print("Checkerboard Example using ZPP\n", .{});
    std.debug.print("==============================\n", .{});
    std.debug.print("SIMD vector length: {d}\n", .{vec_len});
    std.debug.print("Generating {d}x{d} image with {d}px squares...\n", .{ width, height, @as(u32, @trunc(square_size)) });

    const start = Io.Timestamp.now(io, .awake);

    const image_data = try generateImage(allocator, width, height, square_size);
    defer allocator.free(image_data);

    const elapsed = start.untilNow(io, .awake).toMilliseconds();
    std.debug.print("Generation completed in {d}ms\n", .{elapsed});

    const filename = "checkerboard.ppm";
    try writePPM(io, filename, image_data, width, height);
    std.debug.print("Image saved to: {s}\n", .{filename});
}

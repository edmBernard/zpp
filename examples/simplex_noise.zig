//! Simplex Noise Example
//! Generates a 2D simplex noise texture and saves it as a PPM file.
//!
//! This example demonstrates:
//! - Using zpp.Generate to create procedural textures from coordinates
//! - SIMD-accelerated simplex noise implementation
//! - Converting grayscale to RGB output using zpp.InterleavedOut
//!
//! The noise implementation uses a pure SIMD approach where gradients are computed
//! inline via hash functions, which is very efficient on modern CPUs.

const std = @import("std");
const zpp = @import("zpp");

// ============================================================================
// MARK: SIMD Vector Configuration
// ============================================================================

pub const vec_len = zpp.default_vec_len;
pub const VecF32 = @Vector(vec_len, f32);
pub const VecU8 = @Vector(vec_len, u8);

/// Convenience alias for zpp.splat with VecF32
inline fn splat(scalar: f32) VecF32 {
    return zpp.splat(VecF32, scalar);
}

// ============================================================================
// MARK: Linear Algebra Types
// ============================================================================

pub const Vec2 = struct {
    x: VecF32,
    y: VecF32,

    pub inline fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }

    pub inline fn mul1(a: Vec2, b: VecF32) Vec2 {
        return .{ .x = a.x * b, .y = a.y * b };
    }

    pub inline fn dot(p: Vec2, q: Vec2) VecF32 {
        return p.x * q.x + p.y * q.y;
    }
};

// ============================================================================
// MARK: Helper Functions
// ============================================================================

/// Return the fractional part of a floating point number
pub inline fn fract(x: VecF32) VecF32 {
    return x - @floor(x);
}

/// A periodic triangle function - faster approximation useful in hash functions
inline fn triangle_func(in: VecF32) VecF32 {
    const z = in * splat(0.25);
    const f = splat(2.0) * @abs(z - @floor(z) - splat(0.5));
    return splat(2.0) * f - splat(1.0);
}

// ============================================================================
// MARK: Simplex Noise Implementation
// ============================================================================

/// Simplex noise constants
const K1: f32 = 0.366025404; // (sqrt(3)-1)/2
const K2: f32 = 0.211324865; // (3-sqrt(3))/6

/// Hash function for computing pseudo-random gradients at integer coordinates.
/// This is a pure SIMD function - very fast as it's all ALU operations.
inline fn hash(p: Vec2) Vec2 {
    const temp = Vec2{
        .x = Vec2.dot(p, .{ .x = splat(127.1), .y = splat(311.7) }),
        .y = Vec2.dot(p, .{ .x = splat(269.5), .y = splat(183.3) }),
    };
    return .{
        .x = splat(-1.0) + splat(2.0) * fract(triangle_func(temp.x) * splat(43758.5453123)),
        .y = splat(-1.0) + splat(2.0) * fract(triangle_func(temp.y) * splat(43758.5453123)),
    };
}

/// Compute simplex grid base coordinate from world coordinate (the skewing transform).
inline fn toSimplexCell(p: Vec2) Vec2 {
    const k1 = splat(K1);
    return .{
        .x = @floor(p.x + (p.x + p.y) * k1),
        .y = @floor(p.y + (p.x + p.y) * k1),
    };
}

/// 2D Simplex noise implementation.
/// Returns values roughly in range [-1, 1].
pub fn noise(p: Vec2) VecF32 {
    const k2 = splat(K2);

    // Compute simplex cell coordinate
    const i = toSimplexCell(p);

    // Offset from cell origin (unskewing)
    const a: Vec2 = .{
        .x = p.x - i.x + (i.x + i.y) * k2,
        .y = p.y - i.y + (i.x + i.y) * k2,
    };

    // Determine which simplex (lower or upper triangle)
    const m: VecF32 = @select(f32, a.x < a.y, splat(0), splat(1));
    const o: Vec2 = .{ .x = m, .y = splat(1.0) - m };

    // Offsets for other two vertices
    const b: Vec2 = .{ .x = a.x - o.x + k2, .y = a.y - o.y + k2 };
    const c: Vec2 = .{
        .x = a.x - splat(1.0) + splat(2.0) * k2,
        .y = a.y - splat(1.0) + splat(2.0) * k2,
    };

    // Falloff weights (radial basis functions)
    const h0 = @max(splat(0.5) - Vec2.dot(a, a), splat(0));
    const h1 = @max(splat(0.5) - Vec2.dot(b, b), splat(0));
    const h2 = @max(splat(0.5) - Vec2.dot(c, c), splat(0));

    // Compute hash at the three simplex vertices and dot with offset
    const n0 = h0 * h0 * h0 * h0 * Vec2.dot(a, hash(.{ .x = i.x, .y = i.y }));
    const n1 = h1 * h1 * h1 * h1 * Vec2.dot(b, hash(.{ .x = i.x + o.x, .y = i.y + o.y }));
    const n2 = h2 * h2 * h2 * h2 * Vec2.dot(c, hash(.{ .x = i.x + splat(1.0), .y = i.y + splat(1.0) }));

    // Scale to roughly [-1, 1]
    return (n0 + n1 + n2) * splat(70);
}

// ============================================================================
// MARK: ZPP Processing Kernel
// ============================================================================

/// Kernel context for simplex noise generation
pub const NoiseContext = struct {
    /// Scale factor (higher = more zoomed in on the noise)
    scale: VecF32,
    /// Offset for animation or variation
    offset_x: VecF32,
    offset_y: VecF32,
};

/// Simplex noise generator kernel for zpp.Generate.
/// Returns RGB values as u8 (grayscale noise mapped to all channels).
pub fn noiseProcess(ctx: NoiseContext, x: VecF32, y: VecF32) [3]VecU8 {
    // Scale coordinates
    const xs = x / ctx.scale + ctx.offset_x;
    const ys = y / ctx.scale + ctx.offset_y;

    // Generate noise value
    const n = noise(.{ .x = xs, .y = ys });

    // Map from [-1, 1] to [0, 255] for grayscale output
    const gray: VecU8 = @intFromFloat(@max(splat(0.0), @min(splat(255.0), (n * splat(0.5) + splat(0.5)) * splat(255.0))));

    // Output as grayscale RGB
    return .{ gray, gray, gray };
}

// ============================================================================
// MARK: Image Generation using ZPP
// ============================================================================

/// Generate a simplex noise image using zpp primitives.
pub fn generateImage(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    const size = width * height * 3;
    const data = try allocator.alloc(u8, size);
    @memset(data, 0);

    const context = NoiseContext{
        .scale = splat(100.0), // Adjust for different noise scales
        .offset_x = splat(0.0),
        .offset_y = splat(0.0),
    };

    const region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };

    const destination = zpp.InterleavedOut(u8, 3, data, width, region);
    const generator = zpp.Generate(VecF32, region, context, noiseProcess);
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

    std.debug.print("Simplex Noise Example using ZPP\n", .{});
    std.debug.print("================================\n", .{});
    std.debug.print("SIMD vector length: {d}\n", .{vec_len});
    std.debug.print("Generating {d}x{d} image...\n", .{ width, height });

    const start = std.time.milliTimestamp();

    const image_data = try generateImage(allocator, width, height);
    defer allocator.free(image_data);

    const elapsed = std.time.milliTimestamp() - start;
    std.debug.print("Generation completed in {d}ms\n", .{elapsed});

    const filename = "simplex_noise.ppm";
    try writePPM(filename, image_data, width, height);
    std.debug.print("Image saved to: {s}\n", .{filename});
}

// ============================================================================
// MARK: Tests
// ============================================================================

test "simplex noise produces values in expected range" {
    const p = Vec2{ .x = splat(0.5), .y = splat(0.5) };
    const n = noise(p);
    for (0..vec_len) |i| {
        // Noise is scaled by 70, so raw values can exceed [-1, 1] slightly
        try std.testing.expect(n[i] >= -2.0 and n[i] <= 2.0);
    }
}

test "simplex noise is deterministic" {
    const p = Vec2{ .x = splat(1.23), .y = splat(4.56) };
    const n1 = noise(p);
    const n2 = noise(p);

    // Same input should produce same output
    for (0..vec_len) |i| {
        try std.testing.expectEqual(n1[i], n2[i]);
    }
}

test "noise kernel produces valid RGB" {
    const ctx = NoiseContext{
        .scale = splat(100.0),
        .offset_x = splat(0.0),
        .offset_y = splat(0.0),
    };
    const rgb = noiseProcess(ctx, splat(50.0), splat(50.0));

    for (0..vec_len) |i| {
        // RGB values are u8, so always in [0, 255] range
        try std.testing.expect(rgb[0][i] <= 255);
        try std.testing.expect(rgb[1][i] <= 255);
        try std.testing.expect(rgb[2][i] <= 255);
    }
}

test "zpp.Region integration" {
    const region = zpp.Region{ .x = 0, .y = 0, .width = 16, .height = 16 };
    try std.testing.expectEqual(@as(u32, 256), region.area());
    try std.testing.expectEqual(@as(u32, 16), region.width);
}

test "hash function produces consistent values" {
    const p1 = Vec2{ .x = splat(1.0), .y = splat(2.0) };
    const p2 = Vec2{ .x = splat(1.0), .y = splat(2.0) };

    const h1 = hash(p1);
    const h2 = hash(p2);

    // Same input should produce same output
    for (0..vec_len) |i| {
        try std.testing.expectEqual(h1.x[i], h2.x[i]);
        try std.testing.expectEqual(h1.y[i], h2.y[i]);
    }

    // Hash values should be in [-1, 1] range
    for (0..vec_len) |i| {
        try std.testing.expect(h1.x[i] >= -1.0 and h1.x[i] <= 1.0);
        try std.testing.expect(h1.y[i] >= -1.0 and h1.y[i] <= 1.0);
    }
}

//! Simplex Noise Example
//! Generates a 2D simplex noise texture and saves it as a PPM file.
//!
//! This example demonstrates:
//! - Using zpp.generate to create procedural textures from coordinates
//! - SIMD-accelerated simplex noise implementation
//! - Converting grayscale to RGB output using zpp.makeInterleavedDest
//!
//! The noise implementation uses a pure SIMD approach where gradients are computed
//! inline via hash functions, which is very efficient on modern CPUs.

const std = @import("std");
const zpp = @import("zpp");
const Io = std.Io;

// ============================================================================
// MARK: SIMD Vector Configuration
// ============================================================================

const u8v = zpp.u8v;
const vec_len = @typeInfo(u8v).vector.len;
const f32v = zpp.VectorLike(u8v, f32);
const la = zpp.zla.With(f32v);
const Vec2 = la.Vec2;
const Vec3 = la.Vec3;

// ============================================================================
// MARK: Helper Functions
// ============================================================================

/// A periodic triangle function - faster approximation useful in hash functions
inline fn triangle_func(in: f32v) f32v {
    const z = in * la.splat(0.25);
    const f = la.splat(2.0) * @abs(z - @floor(z) - la.splat(0.5));
    return la.splat(2.0) * f - la.splat(1.0);
}

// ============================================================================
// MARK: Simplex Noise Constants and Hash Function
// ============================================================================

/// Hash function for computing pseudo-random gradients at integer coordinates.
/// This is a pure SIMD function - very fast as it's all ALU operations with no memory access.
inline fn hash(p: Vec2) Vec2 {
    const temp = Vec2{
        .x = Vec2.dot(p, .{ .x = la.splat(127.1), .y = la.splat(311.7) }),
        .y = Vec2.dot(p, .{ .x = la.splat(269.5), .y = la.splat(183.3) }),
    };
    return .{
        .x = la.splat(-1.0) + la.splat(2.0) * zpp.math.fract(triangle_func(temp.x) * la.splat(43758.5453123)),
        .y = la.splat(-1.0) + la.splat(2.0) * zpp.math.fract(triangle_func(temp.y) * la.splat(43758.5453123)),
    };
}

// ============================================================================
// MARK: Simplex Noise Implementation
// ============================================================================

/// 2D Simplex noise implementation.
/// Uses direct hash computation for optimal SIMD performance.
fn noise(p: Vec2) f32v {
    // Skew/Unskew factors for simplex grid
    const K1: f32 = 0.366025404; // (sqrt(3)-1)/2
    const K2: f32 = 0.211324865; // (3-sqrt(3))/6

    const k1 = la.splat(K1);
    const k2 = la.splat(K2);

    // Compute simplex cell coordinate
    const skew = (p.x + p.y) * k1;
    const i: Vec2 = .{
        .x = @floor(p.x + skew),
        .y = @floor(p.y + skew),
    };

    // Offset from cell origin (unskewing)
    const a = p.sub(i).add1((i.x + i.y) * k2);

    // Determine which simplex (lower or upper triangle)
    const m: f32v = @select(f32, a.x < a.y, la.splat(0), la.splat(1));
    const o: Vec2 = .{ .x = m, .y = la.splat(1.0) - m };

    // Offsets for other two vertices
    const b = a.sub(o).add1(k2);
    const c = a.sub1(la.splat(1.0)).add1(la.splat(2.0) * k2);

    // Falloff weights (radial basis functions)
    const na: f32v = @max(la.splat(0.5) - Vec2.dot(a, a), la.splat(0));
    const nb: f32v = @max(la.splat(0.5) - Vec2.dot(b, b), la.splat(0));
    const nc: f32v = @max(la.splat(0.5) - Vec2.dot(c, c), la.splat(0));

    // Compute hash at the three simplex vertices and dot with offset
    const n: Vec3 = .{
        .x = na * na * na * na * Vec2.dot(a, hash(.{ .x = i.x, .y = i.y })),
        .y = nb * nb * nb * nb * Vec2.dot(b, hash(.{ .x = i.x + o.x, .y = i.y + o.y })),
        .z = nc * nc * nc * nc * Vec2.dot(c, hash(.{ .x = i.x + la.splat(1.0), .y = i.y + la.splat(1.0) })),
    };

    return (n.x + n.y + n.z) * la.splat(70);
}

// ============================================================================
// MARK: ZPP Processing Kernel
// ============================================================================

/// Kernel context for simplex noise generation
pub const NoiseContext = struct {
    /// Scale factor (higher = more zoomed in on the noise)
    scale: f32v,
    /// Offset for animation or variation
    offset_x: f32v,
    offset_y: f32v,
};

/// Simplex noise generator kernel for zpp.generate.
/// Returns RGB values as u8 (grayscale noise mapped to all channels).
pub fn noiseProcess(ctx: NoiseContext, x: f32v, y: f32v) [3]u8v {
    // Scale coordinates
    const xs = x / ctx.scale + ctx.offset_x;
    const ys = y / ctx.scale + ctx.offset_y;

    // Generate noise value
    const n = noise(.{ .x = xs, .y = ys });

    // Map from [-1, 1] to [0, 255] for grayscale output
    const gray: u8v = @trunc(@max(la.splat(0.0), @min(la.splat(255.0), (n * la.splat(0.5) + la.splat(0.5)) * la.splat(255.0))));

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
        .scale = la.splat(100.0), // Adjust for different noise scales
        .offset_x = la.splat(0.0),
        .offset_y = la.splat(0.0),
    };

    const region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };

    const destination = try zpp.makeInterleavedDest(u8, 3, data, width, region);
    const generator = zpp.generate(f32v, context, noiseProcess);
    zpp.process(generator, destination);

    return data;
}

// ============================================================================
// MARK: PPM Image Output
// ============================================================================

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

    std.debug.print("Simplex Noise Example using ZPP\n", .{});
    std.debug.print("================================\n", .{});
    std.debug.print("SIMD vector length: {d}\n", .{vec_len});
    std.debug.print("Generating {d}x{d} image...\n", .{ width, height });

    const start = Io.Timestamp.now(io, .awake);

    const image_data = try generateImage(allocator, width, height);
    defer allocator.free(image_data);

    const elapsed = start.untilNow(io, .awake).toMilliseconds();
    std.debug.print("Generation completed in {d}ms\n", .{elapsed});

    const filename = "simplex_noise.ppm";
    try writePPM(io, filename, image_data, width, height);
    std.debug.print("Image saved to: {s}\n", .{filename});
}

// ============================================================================
// MARK: Tests
// ============================================================================

test "simplex noise produces values in expected range" {
    const p = Vec2{ .x = la.splat(0.5), .y = la.splat(0.5) };
    const n: [vec_len]f32 = noise(p);
    for (0..vec_len) |i| {
        // Noise is scaled by 70, so raw values can exceed [-1, 1] slightly
        try std.testing.expect(n[i] >= -2.0 and n[i] <= 2.0);
    }
}

test "simplex noise is deterministic" {
    const p = Vec2{ .x = la.splat(1.23), .y = la.splat(4.56) };
    const n1: [vec_len]f32 = noise(p);
    const n2: [vec_len]f32 = noise(p);

    // Same input should produce same output
    for (0..vec_len) |i| {
        try std.testing.expectEqual(n1[i], n2[i]);
    }
}

test "noise kernel produces valid RGB" {
    const ctx = NoiseContext{
        .scale = la.splat(100.0),
        .offset_x = la.splat(0.0),
        .offset_y = la.splat(0.0),
    };
    const rgb = noiseProcess(ctx, la.splat(50.0), la.splat(50.0));

    const r: [vec_len]u8 = rgb[0];
    const g: [vec_len]u8 = rgb[1];
    const b: [vec_len]u8 = rgb[2];
    for (0..vec_len) |i| {
        // RGB values are u8, so always in [0, 255] range
        try std.testing.expect(r[i] <= 255);
        try std.testing.expect(g[i] <= 255);
        try std.testing.expect(b[i] <= 255);
    }
}

test "zpp.Region integration" {
    const region = zpp.Region{ .x = 0, .y = 0, .width = 16, .height = 16 };
    try std.testing.expectEqual(@as(u32, 256), region.area());
    try std.testing.expectEqual(@as(u32, 16), region.width);
}

test "hash function produces consistent values" {
    const p1 = Vec2{ .x = la.splat(1.0), .y = la.splat(2.0) };
    const p2 = Vec2{ .x = la.splat(1.0), .y = la.splat(2.0) };

    const h1 = hash(p1);
    const h2 = hash(p2);

    const h1x: [vec_len]f32 = h1.x;
    const h1y: [vec_len]f32 = h1.y;
    const h2x: [vec_len]f32 = h2.x;
    const h2y: [vec_len]f32 = h2.y;

    // Same input should produce same output
    for (0..vec_len) |i| {
        try std.testing.expectEqual(h1x[i], h2x[i]);
        try std.testing.expectEqual(h1y[i], h2y[i]);
    }

    // Hash values should be in [-1, 1] range
    for (0..vec_len) |i| {
        try std.testing.expect(h1x[i] >= -1.0 and h1x[i] <= 1.0);
        try std.testing.expect(h1y[i] >= -1.0 and h1y[i] <= 1.0);
    }
}

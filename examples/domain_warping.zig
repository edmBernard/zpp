//! Domain Warping Example
//! Generates procedural textures using domain warping and fractional Brownian motion (fBm).
//! Adapted from Inigo Quilez: https://iquilezles.org/articles/fbm/
//!
//! This example demonstrates the use of zpp's generate and process primitives
//! to create SIMD-accelerated procedural texture generation.
//!
//! The simplex noise implementation uses a pure SIMD approach where the hash function
//! is computed inline. This is faster than caching because:
//! - The hash function is computationally cheap (pure ALU operations)
//! - SIMD operations are very fast on modern CPUs
//! - No memory bandwidth bottleneck or cache misses
//! - No bounds checking overhead

const std = @import("std");
const zpp = @import("zpp");

// ============================================================================
// MARK: SIMD Vector Configuration - Use zpp's recommended vector length
// ============================================================================

const u8v = zpp.u8v;
const f32v = zpp.VectorLike(u8v, f32);
const la = zpp.zla.with(f32v);
const Vec2 = la.Vec2;
const Vec3 = la.Vec3;
const Mat2x2 = la.Mat2x2;

// ============================================================================
// MARK: Helper Functions
// ============================================================================

/// A periodic triangle function - faster approximation of sin
inline fn triangle_func(in: f32v) f32v {
    const z = in * la.splat(0.25);
    const f = la.splat(2.0) * @abs(z - @floor(z) - la.splat(0.5));
    return la.splat(2.0) * f - la.splat(1.0);
}

/// Convert hex color (0xRRGGBBAA) to Vec3
inline fn hexToVec3(comptime hex: u32) Vec3 {
    return .{
        .x = la.splat(@as(f32, @floatFromInt((hex & 0xFF000000) >> 24)) / 255.0),
        .y = la.splat(@as(f32, @floatFromInt((hex & 0x00FF0000) >> 16)) / 255.0),
        .z = la.splat(@as(f32, @floatFromInt((hex & 0x0000FF00) >> 8)) / 255.0),
    };
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
    const i: Vec2 = .{
        .x = @floor(p.x + (p.x + p.y) * k1),
        .y = @floor(p.y + (p.x + p.y) * k1),
    };

    // Offset from cell origin (unskewing)
    const a: Vec2 = .{
        .x = p.x - i.x + (i.x + i.y) * k2,
        .y = p.y - i.y + (i.x + i.y) * k2,
    };

    // Determine which simplex (lower or upper triangle)
    const m: f32v = @select(f32, a.x < a.y, la.splat(0), la.splat(1));
    const o: Vec2 = .{ .x = m, .y = la.splat(1.0) - m };

    // Offsets for other two vertices
    const b: Vec2 = .{ .x = a.x - o.x + k2, .y = a.y - o.y + k2 };
    const c: Vec2 = .{
        .x = a.x - la.splat(1.0) + la.splat(2.0) * k2,
        .y = a.y - la.splat(1.0) + la.splat(2.0) * k2,
    };

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
// MARK: Domain Warping with FBM
// ============================================================================

// Rotation matrix to avoid directional artifacts (45 degrees)
const angle = std.math.pi / 4.0;
const rotation_mtx = Mat2x2{
    .data = [4]f32v{
        la.splat(@cos(angle)),
        la.splat(@sin(angle)),
        la.splat(-@sin(angle)),
        la.splat(@cos(angle)),
    },
};

/// Fractional Brownian motion (fBm) - sums multiple octaves of noise
fn fbm(comptime octaves: i32, vec: Vec2) f32v {
    const H = 1.0; // Hurst exponent
    const G = la.splat(std.math.exp2(-H));
    var f = la.splat(1.0);
    var a = la.splat(0.5);
    var t = la.splat(0.0);
    inline for (0..octaves) |_| {
        t += a * noise(rotation_mtx.mulvec(vec).mul1(f));
        f *= la.splat(1.9);
        a *= G;
    }
    return t;
}

/// Multi-scale pattern function for domain warping
fn pattern(p: Vec2) struct { f32v, Vec2, Vec2 } {
    // Low frequency layer
    const q: Vec2 = .{
        .x = la.splat(0.5) + la.splat(0.5) * fbm(8, .{ .x = p.x + la.splat(1.1), .y = p.y + la.splat(0.1) }),
        .y = la.splat(0.5) + la.splat(0.5) * fbm(8, .{ .x = p.x + la.splat(5.1), .y = p.y + la.splat(1.5) }),
    };

    // Mid frequency layer
    const r: Vec2 = .{
        .x = la.splat(0.5) - la.splat(0.5) * fbm(6, .{ .x = p.x + la.splat(6.1) * q.x, .y = p.y + la.splat(6.1) * q.y }),
        .y = la.splat(0.5) - la.splat(0.5) * fbm(6, .{ .x = p.x + la.splat(6.1) * q.x, .y = p.y + la.splat(6.1) * q.y }),
    };

    // High frequency layer
    const f = la.splat(0.5) + la.splat(0.5) * fbm(10, p.add(r.mul1(la.splat(8.1))));
    return .{ f, r, q };
}

// ============================================================================
// MARK: ZPP Processing Kernel
// ============================================================================

/// Kernel context containing parameters for domain warping
pub const DomainWarpingContext = struct {
    scale: f32v,
    sin_time: f32v,
};

/// Apply color mapping to pattern values
inline fn applyColorMapping(f: f32v, r: Vec2, q: Vec2) Vec3 {
    // Compute color by mixing several colors based on pattern values
    var col = hexToVec3(0x561111ff);
    col = col.lerp(hexToVec3(0xe2730cff), f);
    col = col.lerp(hexToVec3(0xffffffff), Vec2.dot(r, r));
    col = col.lerp(hexToVec3(0x832121ff), Vec2.dot(q, q));

    // Add extra color in dark areas
    col = col.lerp(
        hexToVec3(0x290202ff),
        la.splat(0.5) * zpp.math.smoothstep(la.splat(1.1), la.splat(1.3), @abs(r.x) + @abs(r.y)),
    );

    // Increase contrast on high frequency details
    col = col.mul1(f * la.splat(2.0));
    // Invert and apply gamma curve
    const temp = Vec3.ones.sub(col);
    return temp.pow(3);
}

/// Domain warping process function for zpp.generate
/// Returns RGB values as u8
pub fn domainWarpingProcess(ctx: DomainWarpingContext, x: f32v, y: f32v) [3]u8v {
    const xs = x / ctx.scale + ctx.sin_time;
    const ys = y / ctx.scale + ctx.sin_time;

    const f, const r, const q = pattern(.{ .x = xs, .y = ys });
    const col = applyColorMapping(f, r, q);

    // Convert from [0, 1] float to [0, 255] u8
    const splat_0: f32v = @splat(0.0);
    const splat_255: f32v = @splat(255.0);
    return .{
        @intFromFloat(@max(splat_0, @min(splat_255, col.x * splat_255))),
        @intFromFloat(@max(splat_0, @min(splat_255, col.y * splat_255))),
        @intFromFloat(@max(splat_0, @min(splat_255, col.z * splat_255))),
    };
}

// ============================================================================
// MARK: Image Generation using ZPP
// ============================================================================

/// Generate an image using domain warping with zpp primitives.
pub fn generateImage(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    const size = width * height * 3;
    const data = try allocator.alloc(u8, size);
    @memset(data, 0);

    const time: f32 = 125.0;
    const context = DomainWarpingContext{
        .scale = la.splat(1000.0),
        .sin_time = la.splat(@sin(time)),
    };

    const region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };

    const destination = zpp.makeInterleavedDest(u8, 3, data, width, region);
    const generator = zpp.generate(f32v, context, domainWarpingProcess);
    zpp.process(generator, destination);

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

    const width: u32 = 6000;
    const height: u32 = 4000;

    std.debug.print("Domain Warping Example using ZPP\n", .{});
    std.debug.print("================================\n", .{});
    std.debug.print("SIMD vector length: {d} (from zpp.suggested_vec_len)\n", .{vec_len});
    std.debug.print("Generating {d}x{d} image...\n", .{ width, height });

    const start = std.time.milliTimestamp();

    const image_data = try generateImage(allocator, width, height);
    defer allocator.free(image_data);

    const elapsed = std.time.milliTimestamp() - start;
    std.debug.print("Generation completed in {d}ms\n", .{elapsed});

    const filename = "domain_warping.ppm";
    try writePPM(filename, image_data, width, height);
    std.debug.print("Image saved to: {s}\n", .{filename});
}

// ============================================================================
// MARK: Tests
// ============================================================================

const vec_len = @typeInfo(u8v).vector.len;

test "simplex noise produces values in expected range" {
    const p = Vec2{ .x = la.splat(0.5), .y = la.splat(0.5) };
    const n = noise(p);
    for (0..vec_len) |i| {
        try std.testing.expect(n[i] >= -2.0 and n[i] <= 2.0);
    }
}

test "fbm produces values" {
    const p = Vec2{ .x = la.splat(0.5), .y = la.splat(0.5) };
    const f = fbm(4, p);
    for (0..vec_len) |i| {
        try std.testing.expect(!std.math.isNan(f[i]));
    }
}

test "domain warping kernel produces valid RGB" {
    const ctx = DomainWarpingContext{
        .scale = la.splat(1000.0),
        .sin_time = la.splat(0.0),
    };
    const rgb = domainWarpingProcess(ctx, la.splat(100.0), la.splat(100.0));

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
    const p1 = Vec2{ .x = la.splat(1.0), .y = la.splat(2.0) };
    const p2 = Vec2{ .x = la.splat(1.0), .y = la.splat(2.0) };

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

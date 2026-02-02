//! Gradient Filter Example
//! Generates a 2D simplex noise texture, then applies different filters by chaining expressions:
//!  - Resize x2 with bilinear interpolation
//!  - Gradient filter (edge detection)
//!  - Gamma correction
//! Then saves the result as a PPM file.
//!
//! This example demonstrates:
//! - Using zpp.Generate to create procedural textures
//! - Using zpp.InterpLoop for interpolated sampling (resize)
//! - Using zpp.Loop with margins for convolution filters (gradient)
//! - Expression tree chaining (composing multiple operations lazily)
//! - Custom kernels for gamma correction
//!
//! Note: This example uses f32x4 (4-element vectors) throughout for consistency
//! with the library's internal processing. The library's Loop and InterpLoop
//! operations work with vectors of length 4.

const std = @import("std");
const zpp = @import("zpp");

// ============================================================================
// MARK: SIMD Vector Configuration - Use 4-element vectors for chained operations
// ============================================================================

/// Vector type for processing - matches library's internal vector size
pub const VecF32 = @Vector(zpp.suggested_vec_len, f32);

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
// MARK: Simplex Noise Implementation (used to generate source texture)
// ============================================================================

const K1: f32 = 0.366025404; // (sqrt(3)-1)/2
const K2: f32 = 0.211324865; // (3-sqrt(3))/6

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

inline fn toSimplexCell(p: Vec2) Vec2 {
    const k1 = splat(K1);
    return .{
        .x = @floor(p.x + (p.x + p.y) * k1),
        .y = @floor(p.y + (p.x + p.y) * k1),
    };
}

pub fn noise(p: Vec2) VecF32 {
    const k2 = splat(K2);
    const i = toSimplexCell(p);

    const a: Vec2 = .{
        .x = p.x - i.x + (i.x + i.y) * k2,
        .y = p.y - i.y + (i.x + i.y) * k2,
    };

    const m: VecF32 = @select(f32, a.x < a.y, splat(0), splat(1));
    const o: Vec2 = .{ .x = m, .y = splat(1.0) - m };

    const b: Vec2 = .{ .x = a.x - o.x + k2, .y = a.y - o.y + k2 };
    const c: Vec2 = .{
        .x = a.x - splat(1.0) + splat(2.0) * k2,
        .y = a.y - splat(1.0) + splat(2.0) * k2,
    };

    const h0 = @max(splat(0.5) - Vec2.dot(a, a), splat(0));
    const h1 = @max(splat(0.5) - Vec2.dot(b, b), splat(0));
    const h2 = @max(splat(0.5) - Vec2.dot(c, c), splat(0));

    const n0 = h0 * h0 * h0 * h0 * Vec2.dot(a, hash(.{ .x = i.x, .y = i.y }));
    const n1 = h1 * h1 * h1 * h1 * Vec2.dot(b, hash(.{ .x = i.x + o.x, .y = i.y + o.y }));
    const n2 = h2 * h2 * h2 * h2 * Vec2.dot(c, hash(.{ .x = i.x + splat(1.0), .y = i.y + splat(1.0) }));

    return (n0 + n1 + n2) * splat(70);
}

// ============================================================================
// MARK: Stage 1: Noise Generator Kernel
// ============================================================================

/// Context for noise generation
pub const NoiseContext = struct {
    scale: VecF32,
};

/// Generate grayscale simplex noise (single channel for processing)
pub fn noiseKernel(ctx: NoiseContext, x: VecF32, y: VecF32) VecF32 {
    const xs = x / ctx.scale;
    const ys = y / ctx.scale;
    const n = noise(.{ .x = xs, .y = ys });
    // Map from [-1, 1] to [0, 1]
    return n * splat(0.5) + splat(0.5);
}

// ============================================================================
// MARK: Stage 2: Resize Kernel (bilinear interpolation, scale 2x)
// ============================================================================

/// Context for resize operation
pub const ResizeContext = struct {
    /// Scale factor (0.5 = double size, as we sample at half coordinates)
    scale: VecF32,
};

/// Resize kernel using interpolated sampling.
/// The kernel receives output coordinates and samples the input at scaled coordinates.
pub fn resizeKernel(ctx: ResizeContext, interp: anytype, x: VecF32, y: VecF32) VecF32 {
    // For 2x upscale, we sample input at x/2, y/2
    return interp.sample(x * ctx.scale, y * ctx.scale);
}

// ============================================================================
// MARK: Stage 3: Gradient Filter Kernel (Sobel-like edge detection)
// ============================================================================

/// Context for gradient filter (empty, could hold filter weights)
pub const GradientContext = struct {};

/// Sobel gradient filter - computes edge magnitude.
/// Uses a 3x3 neighborhood to compute horizontal and vertical gradients.
///
/// Sobel X kernel:     Sobel Y kernel:
/// -1  0  1            -1 -2 -1
/// -2  0  2             0  0  0
/// -1  0  1             1  2  1
pub fn gradientKernel(ctx: GradientContext, in: anytype) VecF32 {
    _ = ctx;

    // Sample 3x3 neighborhood
    const tl = in.getAt(-1, -1); // top-left
    const tc = in.getAt(0, -1); // top-center
    const tr = in.getAt(1, -1); // top-right
    const ml = in.getAt(-1, 0); // middle-left
    const mr = in.getAt(1, 0); // middle-right
    const bl = in.getAt(-1, 1); // bottom-left
    const bc = in.getAt(0, 1); // bottom-center
    const br = in.getAt(1, 1); // bottom-right

    const two: VecF32 = splat(2.0);
    const quarter: VecF32 = splat(0.25);
    const one: VecF32 = splat(1.0);

    // Sobel X gradient: horizontal edges
    const gx = (tr - tl) + two * (mr - ml) + (br - bl);

    // Sobel Y gradient: vertical edges
    const gy = (bl - tl) + two * (bc - tc) + (br - tr);

    // Gradient magnitude (simplified: sqrt(gx^2 + gy^2))
    // Using |gx| + |gy| as faster approximation
    const magnitude = @abs(gx) + @abs(gy);

    // Normalize to [0, 1] range (Sobel max theoretical is ~4 for normalized input)
    return @min(magnitude * quarter, one);
}

// ============================================================================
// MARK: Stage 4: Gamma Correction Kernel
// ============================================================================

/// Context for gamma correction
pub const GammaContext = struct {
    /// Gamma exponent (< 1 brightens, > 1 darkens)
    gamma: f32,
};

/// Apply gamma correction: output = input ^ gamma
pub fn gammaKernel(ctx: GammaContext, in: anytype) VecF32 {
    const value = in.get();
    // Clamp to valid range for pow
    const small: VecF32 = splat(0.0001);
    const clamped = @max(value, small);
    // Apply gamma: v^gamma using exp(gamma * log(v))
    const gamma_vec: VecF32 = splat(ctx.gamma);
    return @exp(gamma_vec * @log(clamped));
}

// ============================================================================
// MARK: Image Generation using ZPP Expression Trees
// ============================================================================

/// Generate an image with the full processing pipeline.
/// Demonstrates chaining: noise -> resize -> gradient -> gamma
/// The result is written as grayscale to an intermediate buffer,
/// then converted to RGB for PPM output.
pub fn generateImage(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    // Allocate grayscale buffer for intermediate processing
    const gray_size = width * height;
    const gray_data = try allocator.alloc(f32, gray_size);
    defer allocator.free(gray_data);
    @memset(gray_data, 0);

    // Output region (final image size)
    const output_region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };

    // Source region (half size, will be upscaled 2x)
    const source_region = zpp.Region{
        .x = 0,
        .y = 0,
        .width = width / 2,
        .height = height / 2,
    };

    // Stage 1: Generate simplex noise at half resolution
    const noise_ctx = NoiseContext{ .scale = splat(50.0) };
    const noise_source = zpp.Generate(VecF32, source_region, noise_ctx, noiseKernel);

    // Stage 2: Resize 2x with bilinear interpolation
    const resize_ctx = ResizeContext{ .scale = splat(0.5) };
    const resized = zpp.InterpLoop(
        VecF32,
        .Linear, // Bilinear interpolation
        noise_source,
        output_region,
        resize_ctx,
        resizeKernel,
    );

    // Stage 3: Apply gradient filter (edge detection)
    const gradient_ctx = GradientContext{};
    const edges = zpp.Loop(
        VecF32,
        .{ .margin = zpp.marginI(1) }, // Need 1-pixel margin for 3x3 kernel
        resized,
        gradient_ctx,
        gradientKernel,
    );

    // Stage 4: Apply gamma correction to enhance contrast
    const gamma_ctx = GammaContext{ .gamma = 0.6 }; // Brighten the edges
    const corrected = zpp.Loop(VecF32, .{}, edges, gamma_ctx, gammaKernel);

    // Process the expression tree and write to grayscale buffer
    const gray_dest = zpp.Out(f32, gray_data, width, output_region);
    zpp.Process(corrected, gray_dest);

    // Convert grayscale to RGB
    const rgb_size = width * height * 3;
    const rgb_data = try allocator.alloc(u8, rgb_size);

    for (0..gray_size) |i| {
        // Clamp and convert to u8
        const gray_val = std.math.clamp(gray_data[i], 0.0, 1.0);
        const byte_val: u8 = @intFromFloat(gray_val * 255.0);
        rgb_data[i * 3 + 0] = byte_val; // R
        rgb_data[i * 3 + 1] = byte_val; // G
        rgb_data[i * 3 + 2] = byte_val; // B
    }

    return rgb_data;
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

    std.debug.print("Gradient Filter Example using ZPP\n", .{});
    std.debug.print("==================================\n", .{});
    std.debug.print("SIMD vector length: {d}\n", .{zpp.suggested_vec_len});
    std.debug.print("Pipeline: noise -> resize(2x) -> gradient -> gamma\n", .{});
    std.debug.print("Generating {d}x{d} image...\n", .{ width, height });

    const start = std.time.milliTimestamp();

    const image_data = try generateImage(allocator, width, height);
    defer allocator.free(image_data);

    const elapsed = std.time.milliTimestamp() - start;
    std.debug.print("Generation completed in {d}ms\n", .{elapsed});

    const filename = "gradient_filter.ppm";
    try writePPM(filename, image_data, width, height);
    std.debug.print("Image saved to: {s}\n", .{filename});
}

// ============================================================================
// MARK: Tests
// ============================================================================

test "noise kernel produces valid values" {
    const ctx = NoiseContext{ .scale = splat(50.0) };
    const result = noiseKernel(ctx, splat(25.0), splat(25.0));

    for (0..zpp.suggested_vec_len) |i| {
        // Output should be in [0, 1] range after mapping
        try std.testing.expect(result[i] >= 0.0 and result[i] <= 1.0);
    }
}

test "gamma kernel with gamma=1.0 is identity" {
    const region = zpp.Region{ .x = 0, .y = 0, .width = 4, .height = 1 };
    var input: [4]f32 = .{ 0.25, 0.5, 0.75, 1.0 };
    var output: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.In(f32, &input, 4, region);
    const dest = zpp.Out(f32, &output, 4, region);

    const ctx = GammaContext{ .gamma = 1.0 };
    const result = zpp.Loop(VecF32, .{}, source, ctx, gammaKernel);
    zpp.Process(result, dest);

    // With gamma=1.0, output should equal input
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(input[i], output[i], 0.001);
    }
}

test "gradient kernel detects edges" {
    // Create a simple edge: 0 0 0 | 1 1 1 (vertical edge at center)
    const region = zpp.Region{ .x = 0, .y = 0, .width = 4, .height = 3 };
    var input: [12]f32 = .{
        0.0, 0.0, 1.0, 1.0, // row 0
        0.0, 0.0, 1.0, 1.0, // row 1
        0.0, 0.0, 1.0, 1.0, // row 2
    };
    var output: [12]f32 = .{0} ** 12;

    const source = zpp.In(f32, &input, 4, region);
    const dest = zpp.Out(f32, &output, 4, region);

    const ctx = GradientContext{};
    const result = zpp.Loop(VecF32, .{ .margin = zpp.marginI(1) }, source, ctx, gradientKernel);
    zpp.Process(result, dest);

    // Center column (x=1, x=2) should have high gradient values
    // The edge is between columns 1 and 2
    // With Sobel, the peak response is at the edge location
    try std.testing.expect(output[1] > 0.1 or output[2] > 0.1); // row 0
    try std.testing.expect(output[5] > 0.1 or output[6] > 0.1); // row 1
}

test "expression tree chaining works" {
    const region = zpp.Region{ .x = 0, .y = 0, .width = 4, .height = 1 };
    var input: [4]f32 = .{ 0.1, 0.2, 0.3, 0.4 };
    var output: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.In(f32, &input, 4, region);
    const dest = zpp.Out(f32, &output, 4, region);

    // Chain: identity -> gamma
    const identity_kernel = struct {
        const Context = struct {};
        pub fn process(ctx: Context, in: anytype) VecF32 {
            _ = ctx;
            return in.get();
        }
    };

    const step1 = zpp.Loop(VecF32, .{}, source, identity_kernel.Context{}, identity_kernel.process);
    const step2 = zpp.Loop(VecF32, .{}, step1, GammaContext{ .gamma = 2.0 }, gammaKernel);
    zpp.Process(step2, dest);

    // Verify output is input^2
    for (0..4) |i| {
        const expected = std.math.pow(f32, input[i], 2.0);
        try std.testing.expectApproxEqAbs(expected, output[i], 0.01);
    }
}

test "resize with InterpLoop" {
    // Test that InterpLoop can be used for 2x upscaling
    const source_region = zpp.Region{ .x = 0, .y = 0, .width = 2, .height = 2 };
    const output_region = zpp.Region{ .x = 0, .y = 0, .width = 4, .height = 4 };

    // Simple 2x2 source: corners have values 0, 1, 2, 3
    var source_data: [4]f32 = .{ 0.0, 1.0, 2.0, 3.0 };
    var output_data: [16]f32 = .{0} ** 16;

    const source = zpp.In(f32, &source_data, 2, source_region);
    const dest = zpp.Out(f32, &output_data, 4, output_region);

    const resize_ctx = ResizeContext{ .scale = splat(0.5) };
    const resized = zpp.InterpLoop(VecF32, .Linear, source, output_region, resize_ctx, resizeKernel);
    zpp.Process(resized, dest);

    // Corner values should be preserved (approximately)
    // Top-left (0,0) -> samples at (0,0) = 0
    // Top-right (3,0) -> samples at (1.5, 0) = interpolation between 0 and 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output_data[0], 0.1);
    // Output should have interpolated values
    try std.testing.expect(output_data[1] > 0.0); // Should be interpolated
}

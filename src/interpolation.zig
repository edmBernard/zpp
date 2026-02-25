//! Interpolation support for pixel sampling at non-integer coordinates.

const std = @import("std");
const sources = @import("sources.zig");
const Region = @import("region.zig").Region;

// ============================================================================
// MARK: Interpolation Method
// ============================================================================

/// Interpolation method for pixel sampling
pub const InterpolationMethod = enum {
    /// Nearest neighbor - rounds to nearest integer coordinate
    nearest,
    /// Bilinear - linear interpolation in both directions
    linear,
    /// Bicubic - cubic interpolation in both directions
    cubic,
};

// ============================================================================
// MARK: Pixel Interpolator
// ============================================================================

/// Pixel interpolator that provides interpolated access to an image.
/// Used within InterpLoop kernels to sample pixels at non-integer coordinates.
pub fn PixelInterpolator(comptime SourceType: type, comptime VecT: type, comptime method: InterpolationMethod) type {
    const vec_len = @typeInfo(VecT).vector.len;
    const ElemT = @typeInfo(VecT).vector.child;

    return struct {
        source: SourceType,
        region: Region,

        const Self = @This();

        /// Sample interpolated pixel values at floating-point coordinates.
        /// Takes vectors of x and y coordinates and returns interpolated values.
        pub inline fn sample(self: Self, x: VecT, y: VecT) VecT {
            return switch (method) {
                .nearest => self.sampleNearest(x, y),
                .linear => self.sampleLinear(x, y),
                .cubic => self.sampleCubic(x, y),
            };
        }

        /// Nearest neighbor interpolation - just round to nearest integer
        inline fn sampleNearest(self: Self, x: VecT, y: VecT) VecT {
            var result: VecT = @splat(0);
            const x_rounded = @round(x);
            const y_rounded = @round(y);

            inline for (0..vec_len) |i| {
                const xi: i32 = @intFromFloat(x_rounded[i]);
                const yi: i32 = @intFromFloat(y_rounded[i]);
                result[i] = self.readPixel(xi, yi);
            }
            return result;
        }

        /// Bilinear interpolation
        inline fn sampleLinear(self: Self, x: VecT, y: VecT) VecT {
            var result: VecT = @splat(0);

            inline for (0..vec_len) |i| {
                const xf = x[i];
                const yf = y[i];

                // Get integer coordinates
                const x0: i32 = @intFromFloat(@floor(xf));
                const y0: i32 = @intFromFloat(@floor(yf));
                const x1 = x0 + 1;
                const y1 = y0 + 1;

                // Get fractional parts
                const fx = xf - @floor(xf);
                const fy = yf - @floor(yf);

                // Sample 4 corners
                const p00 = self.readPixel(x0, y0);
                const p10 = self.readPixel(x1, y0);
                const p01 = self.readPixel(x0, y1);
                const p11 = self.readPixel(x1, y1);

                // Bilinear interpolation
                const top = p00 * (1.0 - fx) + p10 * fx;
                const bottom = p01 * (1.0 - fx) + p11 * fx;
                result[i] = top * (1.0 - fy) + bottom * fy;
            }
            return result;
        }

        /// Bicubic interpolation using Catmull-Rom spline
        inline fn sampleCubic(self: Self, x: VecT, y: VecT) VecT {
            var result: VecT = @splat(0);

            inline for (0..vec_len) |i| {
                const xf = x[i];
                const yf = y[i];

                // Get integer coordinates (center of 4x4 patch)
                const x1: i32 = @intFromFloat(@floor(xf));
                const y1: i32 = @intFromFloat(@floor(yf));

                // Fractional parts
                const fx = xf - @floor(xf);
                const fy = yf - @floor(yf);

                // Sample 4x4 neighborhood
                var rows: [4]ElemT = undefined;
                inline for (0..4) |dy| {
                    const yi = y1 - 1 + @as(i32, @intCast(dy));
                    var cols: [4]ElemT = undefined;
                    inline for (0..4) |dx| {
                        const xi = x1 - 1 + @as(i32, @intCast(dx));
                        cols[dx] = self.readPixel(xi, yi);
                    }
                    rows[dy] = cubicInterpolate(cols, fx);
                }
                result[i] = cubicInterpolate(rows, fy);
            }
            return result;
        }

        /// Cubic interpolation helper using Catmull-Rom weights
        inline fn cubicInterpolate(p: [4]ElemT, t: ElemT) ElemT {
            // Catmull-Rom spline coefficients
            const t2 = t * t;
            const t3 = t2 * t;

            // Catmull-Rom with a = -0.5
            const w0 = -0.5 * t3 + t2 - 0.5 * t;
            const w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            const w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            const w3 = 0.5 * t3 - 0.5 * t2;

            return p[0] * w0 + p[1] * w1 + p[2] * w2 + p[3] * w3;
        }

        /// Read a single pixel at integer coordinates, with padding
        inline fn readPixel(self: Self, xi: i32, yi: i32) ElemT {
            return switch (comptime sources.SourceKind(SourceType)) {
                .eval => self.source.evalAt(xi, yi)[0],
                .read => self.source.read(xi, yi),
            };
        }
    };
}

// ============================================================================
// MARK: InterpLoop Result
// ============================================================================

/// Result type for InterpLoop - evaluates kernel with interpolated pixel access
pub fn InterpLoopResult(
    comptime VecT: type,
    comptime SourceType: type,
    comptime ContextType: type,
    comptime process_fn: anytype,
    comptime method: InterpolationMethod,
) type {
    const vec_len = @typeInfo(VecT).vector.len;
    const InterpolatorType = PixelInterpolator(SourceType, VecT, method);

    return struct {
        source: SourceType,
        context: ContextType,
        region: Region,

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        const Self = @This();

        /// Evaluate at a specific position - calls kernel with interpolator and coords
        pub inline fn evalAt(self: Self, x: i32, y: i32) VecT {
            // Create interpolator for the source
            const interpolator = InterpolatorType{
                .source = self.source,
                .region = self.source.region,
            };

            // Build coordinate vectors for output position
            const iota: VecT = @floatFromInt(std.simd.iota(i32, vec_len));
            const x_vec: VecT = iota + @as(VecT, @splat(@as(f32, @floatFromInt(x))));
            const y_vec: VecT = @splat(@as(f32, @floatFromInt(y)));

            // Call kernel with interpolator and output coordinates
            return process_fn(self.context, interpolator, x_vec, y_vec);
        }
    };
}

/// Create an interpolation loop for coordinate transforms (geometric transformations).
///
/// The kernel receives:
/// - A pixel interpolator that can sample the source at arbitrary float coordinates
/// - The output x and y coordinates (as vectors of floats)
///
/// The kernel should compute input coordinates and sample using the interpolator.
///
/// Example (identity transform):
/// ```zig
/// const kernel = struct {
///     fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
///         return interp.sample(x, y);  // sample at same coordinates
///     }
/// };
/// const result = InterpLoop(f32x4, .linear, source, output_region, ctx, kernel.process);
/// ```
///
/// Example (scale 2x):
/// ```zig
/// const kernel = struct {
///     fn process(ctx: Context, interp: anytype, x: f32x4, y: f32x4) f32x4 {
///         const scale: f32x4 = @splat(0.5);
///         return interp.sample(x * scale, y * scale);
///     }
/// };
/// ```
pub fn interpLoop(
    comptime VecT: type,
    comptime method: InterpolationMethod,
    source: anytype,
    output_region: Region,
    context: anytype,
    comptime process_fn: anytype,
) InterpLoopResult(VecT, @TypeOf(source), @TypeOf(context), process_fn, method) {
    comptime sources.assertIsSource(@TypeOf(source));
    return .{
        .source = source,
        .context = context,
        .region = output_region,
    };
}

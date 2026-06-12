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
///
/// Coordinate math, interpolation weights, and blending are evaluated on
/// whole vectors (one lane per output pixel). Only the pixel fetches are
/// per-lane gathers, since each lane samples an arbitrary source position.
pub fn PixelInterpolator(comptime SourceType: type, comptime VecT: type, comptime method: InterpolationMethod) type {
    const vec_len = @typeInfo(VecT).vector.len;
    const ElemT = @typeInfo(VecT).vector.child;
    const IndexVecT = @Vector(vec_len, i32);
    const Traits = sources.SourceTraits(SourceType);

    // Sources that expose their padding policy (InputSource and wrappers such
    // as translate) support a branch-free gather: clamp the coordinates with
    // vector min/max, read unchecked, and mask lanes to zero if the policy
    // requires it. `void` marks a wrapper whose nested source has no policy.
    const has_clamp_gather = Traits.kind == .read and Traits.has_unchecked and
        @hasDecl(SourceType, "PaddingPolicyType") and SourceType.PaddingPolicyType != void;

    return struct {
        source: SourceType,

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
            @setEvalBranchQuota(1000 + 32 * vec_len);
            const xi: IndexVecT = @intFromFloat(@round(x));
            const yi: IndexVecT = @intFromFloat(@round(y));
            return self.gather(xi, yi);
        }

        /// Bilinear interpolation
        inline fn sampleLinear(self: Self, x: VecT, y: VecT) VecT {
            @setEvalBranchQuota(1000 + 128 * vec_len);
            const x0f = @floor(x);
            const y0f = @floor(y);

            // Fractional parts
            const fx = x - x0f;
            const fy = y - y0f;

            // Integer coordinates of the top-left corner per lane
            const x0: IndexVecT = @intFromFloat(x0f);
            const y0: IndexVecT = @intFromFloat(y0f);
            const one_i: IndexVecT = @splat(1);
            const x1 = x0 + one_i;
            const y1 = y0 + one_i;

            // Gather the 4 corners, one full vector per tap
            const p00 = self.gather(x0, y0);
            const p10 = self.gather(x1, y0);
            const p01 = self.gather(x0, y1);
            const p11 = self.gather(x1, y1);

            // Bilinear blend
            const ones: VecT = @splat(1.0);
            const top = p00 * (ones - fx) + p10 * fx;
            const bottom = p01 * (ones - fx) + p11 * fx;
            return top * (ones - fy) + bottom * fy;
        }

        /// Bicubic interpolation using Catmull-Rom spline
        inline fn sampleCubic(self: Self, x: VecT, y: VecT) VecT {
            @setEvalBranchQuota(1000 + 512 * vec_len);
            const x1f = @floor(x);
            const y1f = @floor(y);

            // Fractional parts
            const fx = x - x1f;
            const fy = y - y1f;

            // Integer coordinates (center of 4x4 patch) per lane
            const x1v: IndexVecT = @intFromFloat(x1f);
            const y1v: IndexVecT = @intFromFloat(y1f);

            // Column coordinates of the 4x4 patch, shared by all rows
            var xs: [4]IndexVecT = undefined;
            inline for (0..4) |dx| {
                xs[dx] = x1v + @as(IndexVecT, @splat(@as(i32, @intCast(dx)) - 1));
            }

            // Gather and interpolate row by row, then across rows
            var rows: [4]VecT = undefined;
            inline for (0..4) |dy| {
                const yi = y1v + @as(IndexVecT, @splat(@as(i32, @intCast(dy)) - 1));
                var cols: [4]VecT = undefined;
                inline for (0..4) |dx| {
                    cols[dx] = self.gather(xs[dx], yi);
                }
                rows[dy] = cubicInterpolate(cols, fx);
            }
            return cubicInterpolate(rows, fy);
        }

        /// Cubic interpolation helper using Catmull-Rom weights,
        /// evaluated on whole vectors (one lane per output pixel).
        inline fn cubicInterpolate(p: [4]VecT, t: VecT) VecT {
            // Catmull-Rom spline coefficients
            const t2 = t * t;
            const t3 = t2 * t;

            const half: VecT = @splat(0.5);
            const one: VecT = @splat(1.0);
            const one_and_half: VecT = @splat(1.5);
            const two: VecT = @splat(2.0);
            const two_and_half: VecT = @splat(2.5);

            // Catmull-Rom with a = -0.5
            const w0 = -half * t3 + t2 - half * t;
            const w1 = one_and_half * t3 - two_and_half * t2 + one;
            const w2 = -one_and_half * t3 + two * t2 + half * t;
            const w3 = half * t3 - half * t2;

            return p[0] * w0 + p[1] * w1 + p[2] * w2 + p[3] * w3;
        }

        /// Gather one pixel per lane at integer coordinates.
        /// Each lane samples an independent source position, so the fetches
        /// themselves run per lane; bounds handling is vectorized when the
        /// source exposes its padding policy.
        inline fn gather(self: Self, xi: IndexVecT, yi: IndexVecT) VecT {
            if (comptime has_clamp_gather) {
                const r = self.source.region;
                if (r.width > 0 and r.height > 0) {
                    return self.gatherClamped(xi, yi);
                }
            }
            return self.gatherChecked(xi, yi);
        }

        /// Branch-free gather: clamp coordinates into the source region with
        /// vector min/max, read unchecked (clamped coordinates are always
        /// in-bounds), then zero out-of-bounds lanes if the padding policy
        /// requires it (ZeroPadding). RepeatEdgePadding is the clamp itself.
        inline fn gatherClamped(self: Self, xi: IndexVecT, yi: IndexVecT) VecT {
            const r = self.source.region;
            const min_x: IndexVecT = @splat(r.x);
            const max_x: IndexVecT = @splat(r.stopX() - 1);
            const min_y: IndexVecT = @splat(r.y);
            const max_y: IndexVecT = @splat(r.stopY() - 1);
            const cx: [vec_len]i32 = @max(min_x, @min(xi, max_x));
            const cy: [vec_len]i32 = @max(min_y, @min(yi, max_y));

            var out: VecT = undefined;
            inline for (0..vec_len) |i| {
                out[i] = sources.readSourceScalarUnchecked(SourceType, self.source, cx[i], cy[i]);
            }

            if (comptime SourceType.PaddingPolicyType.needs_mask) {
                const in_bounds = (xi >= min_x) & (xi <= max_x) & (yi >= min_y) & (yi <= max_y);
                const zero: VecT = @splat(0);
                out = @select(ElemT, in_bounds, out, zero);
            }
            return out;
        }

        /// Generic fallback gather using checked per-pixel reads. Used for
        /// eval-kind sources (expression trees) and custom padding policies.
        inline fn gatherChecked(self: Self, xi: IndexVecT, yi: IndexVecT) VecT {
            const xa: [vec_len]i32 = xi;
            const ya: [vec_len]i32 = yi;
            var out: VecT = undefined;
            inline for (0..vec_len) |i| {
                out[i] = sources.readSourceScalarChecked(SourceType, self.source, xa[i], ya[i]);
            }
            return out;
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

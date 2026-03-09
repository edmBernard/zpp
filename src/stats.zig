//! Stats destination for computing statistics without writing to memory.

const std = @import("std");
const Region = @import("region.zig").Region;

// ============================================================================
// MARK: Stats Destination
// ============================================================================

/// A Stats destination that calls a kernel function for each processed pixel
/// without writing to memory. This is useful for computing statistics like
/// histograms, min/max values, sums, etc.
///
/// The kernel receives the pixel value and coordinates, allowing it to accumulate
/// statistics into a context structure.
pub fn StatsDest(
    comptime VecT: type,
    comptime ContextType: type,
    comptime accumulate_fn: anytype,
    comptime has_coords: bool,
) type {
    const vec_len = @typeInfo(VecT).vector.len;

    return struct {
        context: *ContextType,
        region: Region,

        pub const InputScalarType = @typeInfo(VecT).vector.child;
        const Self = @This();

        inline fn scalarBatch(value: InputScalarType) VecT {
            var single: VecT = @splat(value);
            single[0] = value;
            return single;
        }

        /// Calls the user's stats kernel on a simd batch.
        pub fn write(self: Self, x: u32, y: u32, values: VecT) void {
            if (has_coords) {
                const iota = std.simd.iota(i32, vec_len);
                const xi: i32 = @intCast(x);
                const yi: i32 = @intCast(y);
                const x_base = self.region.x + xi;
                const y_base = self.region.y + yi;
                const x_vec: @Vector(vec_len, i32) = iota + @as(@Vector(vec_len, i32), @splat(x_base));
                const y_vec: @Vector(vec_len, i32) = @splat(y_base);
                accumulate_fn(self.context, values, x_vec, y_vec);
            } else {
                accumulate_fn(self.context, values);
            }
        }

        /// "Write" a single scalar value by calling the stats function.
        /// Only lane 0 is populated; the remaining lanes are zeroed for checked remainders.
        pub fn writeScalar(self: Self, x: u32, y: u32, value: InputScalarType) void {
            const single = scalarBatch(value);

            if (has_coords) {
                const xi: i32 = @intCast(x);
                const yi: i32 = @intCast(y);
                const x_base = self.region.x + xi;
                const y_base = self.region.y + yi;
                const x_vec: @Vector(vec_len, i32) = @splat(x_base);
                const y_vec: @Vector(vec_len, i32) = @splat(y_base);
                accumulate_fn(self.context, single, x_vec, y_vec);
            } else {
                accumulate_fn(self.context, single);
            }
        }
    };
}

/// Create a Stats destination expression.
/// The stats_fn receives pixel values (and optionally coordinates) and can accumulate
/// statistics into the context. The context is passed by pointer so it can be mutated.
///
/// Usage:
/// ```zig
/// const stats_kernel = struct {
///     const Context = struct { sum: f32 = 0, count: u32 = 0 };
///     pub fn accumulate(ctx: *Context, values: f32x4) void {
///         ctx.sum += @reduce(.Add, values);
///         ctx.count += 4;
///     }
/// };
/// var ctx = stats_kernel.Context{};
/// const stats_dest = stats(f32x4, &ctx, region, stats_kernel.accumulate);
/// process(source, stats_dest);
/// // ctx.sum and ctx.count now contain accumulated values
/// ```
pub fn stats(
    comptime VecT: type,
    context: anytype,
    region: Region,
    comptime stats_fn: anytype,
) StatsDest(VecT, @typeInfo(@TypeOf(context)).pointer.child, stats_fn, false) {
    return .{
        .context = context,
        .region = region,
    };
}

/// Create a Stats destination expression with coordinate support.
/// The stats_fn receives pixel values and x/y coordinate vectors.
pub fn statsWithCoords(
    comptime VecT: type,
    context: anytype,
    region: Region,
    comptime stats_fn: anytype,
) StatsDest(VecT, @typeInfo(@TypeOf(context)).pointer.child, stats_fn, true) {
    return .{
        .context = context,
        .region = region,
    };
}

//! Stats destination for computing statistics without writing to memory.

const std = @import("std");
const region_mod = @import("region.zig");

const Region = region_mod.Region;

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
    comptime stats_fn: anytype,
    comptime has_coords: bool,
) type {
    const vec_len = @typeInfo(VecT).vector.len;

    return struct {
        context: *ContextType,
        region: Region,

        pub const InputType = @typeInfo(VecT).vector.child;
        pub const InputScalarType = InputType;
        const Self = @This();

        /// "Write" a SIMD batch by calling the stats function
        /// This doesn't actually write to memory - it calls the user's stats kernel
        pub fn write(self: Self, x: u32, y: u32, values: VecT) void {
            if (has_coords) {
                // Build coordinate vectors
                const iota = std.simd.iota(i32, vec_len);
                const x_vec: @Vector(vec_len, i32) = iota + @as(@Vector(vec_len, i32), @splat(@as(i32, @intCast(x)) + self.region.x));
                const y_vec: @Vector(vec_len, i32) = @splat(@as(i32, @intCast(y)) + self.region.y);
                stats_fn(self.context, values, x_vec, y_vec);
            } else {
                stats_fn(self.context, values);
            }
        }

        /// "Write" a single scalar value by calling the stats function
        /// Only processes lane 0 to avoid overcounting in remainder handling
        pub fn writeScalar(self: Self, x: u32, y: u32, value: InputType) void {
            // Create a vector with only lane 0 populated, others zeroed
            // This ensures @reduce operations only count the valid scalar value
            var single: VecT = @splat(0);
            single[0] = value;

            if (has_coords) {
                const x_vec: @Vector(vec_len, i32) = @splat(@as(i32, @intCast(x)) + self.region.x);
                const y_vec: @Vector(vec_len, i32) = @splat(@as(i32, @intCast(y)) + self.region.y);
                stats_fn(self.context, single, x_vec, y_vec);
            } else {
                stats_fn(self.context, single);
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

//! Padding policies and loop options for pixel processing.

const Region = @import("region.zig").Region;

// ============================================================================
// MARK: Padding Policies
// ============================================================================

/// Padding policy that returns zero for out-of-bounds access.
pub const ZeroPadding = struct {
    pub fn apply(comptime T: type, _: []const T, _: u32, _: i32, _: i32, _: Region) T {
        return 0;
    }

    /// Clamp X coordinates for vectorized access. For zero padding, clamp is
    /// still applied; the mask from `inBoundsX` determines which lanes are zeroed.
    pub fn clampX(comptime vec_len: comptime_int, x_vec: @Vector(vec_len, i32), region: Region) @Vector(vec_len, i32) {
        const min_x: @Vector(vec_len, i32) = @splat(region.x);
        const max_x: @Vector(vec_len, i32) = @splat(region.stopX() - 1);
        return @max(min_x, @min(x_vec, max_x));
    }

    /// Return a boolean mask of which X lanes are in bounds.
    pub fn inBoundsX(comptime vec_len: comptime_int, x_vec: @Vector(vec_len, i32), region: Region) @Vector(vec_len, bool) {
        const min_x: @Vector(vec_len, i32) = @splat(region.x);
        const stop_x: @Vector(vec_len, i32) = @splat(region.stopX());
        return (x_vec >= min_x) & (x_vec < stop_x);
    }

    /// Whether this policy needs zero masking for out-of-bounds lanes.
    pub const needs_mask = true;
};

/// Padding policy that repeats the edge pixel for out-of-bounds access (default).
pub const RepeatEdgePadding = struct {
    pub fn apply(comptime T: type, data: []const T, stride: u32, x: i32, y: i32, region: Region) T {
        // Handle empty regions - return zero
        if (region.width == 0 or region.height == 0) {
            return 0;
        }

        // Clamp coordinates to valid region
        const clamped_x = @max(region.x, @min(x, region.stopX() - 1));
        const clamped_y = @max(region.y, @min(y, region.stopY() - 1));

        const ux: u32 = @intCast(clamped_x);
        const uy: u32 = @intCast(clamped_y);
        const idx = uy * stride + ux;

        if (idx < data.len) {
            return data[idx];
        }
        return 0;
    }

    /// Clamp X coordinates to valid region bounds (vector operation).
    pub fn clampX(comptime vec_len: comptime_int, x_vec: @Vector(vec_len, i32), region: Region) @Vector(vec_len, i32) {
        const min_x: @Vector(vec_len, i32) = @splat(region.x);
        const max_x: @Vector(vec_len, i32) = @splat(region.stopX() - 1);
        return @max(min_x, @min(x_vec, max_x));
    }

    /// Clamp Y coordinate to valid region bounds (scalar, since Y is uniform across lanes).
    pub fn clampY(y: i32, region: Region) i32 {
        return @max(region.y, @min(y, region.stopY() - 1));
    }

    /// RepeatEdge doesn't need masking — clamped coordinates always point to valid data.
    pub const needs_mask = false;
};

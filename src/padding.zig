//! Padding policies and loop options for pixel processing.

const region_mod = @import("region.zig");
const Region = region_mod.Region;
const Margin = region_mod.Margin;

// ============================================================================
// MARK: Padding Policies
// ============================================================================

/// Padding policy that returns zero for out-of-bounds access.
pub const ZeroPadding = struct {
    pub fn apply(comptime T: type, _: []const T, _: u32, _: i32, _: i32, _: Region) T {
        return 0;
    }
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

        const ux: u32 = @intCast(clamped_x - region.x);
        const uy: u32 = @intCast(clamped_y - region.y);
        const idx = uy * stride + ux;

        if (idx < data.len) {
            return data[idx];
        }
        return 0;
    }
};

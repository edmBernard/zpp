//! Region and Margin types for defining rectangular areas and neighborhood access.

const std = @import("std");

// ============================================================================
// MARK: Margin
// ============================================================================

/// Margin specification for neighborhood access
pub const Margin = struct {
    top: u32 = 0,
    left: u32 = 0,
    bottom: u32 = 0,
    right: u32 = 0,

    const Self = @This();

    /// Returns true if this is a zero margin (no neighborhood access needed).
    pub fn isZero(self: Self) bool {
        return self.left == 0 and self.right == 0 and self.top == 0 and self.bottom == 0;
    }

    /// Returns true if this is a purely vertical margin (left == right == 0).
    pub fn isVertical(self: Self) bool {
        return self.left == 0 and self.right == 0 and self.top == self.bottom and self.top > 0;
    }

    /// Returns true if this is a purely horizontal margin (top == bottom == 0).
    pub fn isHorizontal(self: Self) bool {
        return self.top == 0 and self.bottom == 0 and self.left == self.right and self.left > 0;
    }

    /// Returns true if this is an isotropic margin (all sides equal).
    pub fn isIsotropic(self: Self) bool {
        return self.left == self.right and self.top == self.bottom and
            self.left == self.top and self.left > 0;
    }

    /// Returns true if this is a 2D margin (both horizontal and vertical components).
    pub fn is2D(self: Self) bool {
        return !self.isZero() and !self.isVertical() and !self.isHorizontal();
    }

    /// Returns the maximum extent in any direction.
    pub fn maxExtent(self: Self) u32 {
        return @max(@max(self.left, self.right), @max(self.top, self.bottom));
    }
};

/// Creates a horizontal symmetric margin (left and right equal, top and bottom zero).
pub fn marginH(n: u32) Margin {
    return .{ .left = n, .right = n, .top = 0, .bottom = 0 };
}

/// Creates a vertical symmetric margin (top and bottom equal, left and right zero).
pub fn marginV(n: u32) Margin {
    return .{ .left = 0, .right = 0, .top = n, .bottom = n };
}

/// Creates an isotropic margin (all sides equal).
pub fn marginI(n: u32) Margin {
    return .{ .left = n, .right = n, .top = n, .bottom = n };
}

// ============================================================================
// MARK: Region
// ============================================================================

/// Region defines a rectangular area for processing.
/// Uses start/stop coordinates (start inclusive, stop exclusive).
pub const Region = struct {
    /// The first column inside the region (at the left).
    x: i32 = 0,
    /// The first row inside the region (at the top).
    y: i32 = 0,
    /// Width of the region (number of columns).
    width: u32,
    /// Height of the region (number of rows).
    height: u32,

    const Self = @This();

    /// Returns the area of this region (total number of pixels).
    pub fn area(self: Self) u32 {
        return self.width * self.height;
    }

    /// Returns the stop X coordinate (first column outside the region at the right).
    pub fn stopX(self: Self) i32 {
        return self.x + @as(i32, @intCast(self.width));
    }

    /// Returns the stop Y coordinate (first row outside the region at the bottom).
    pub fn stopY(self: Self) i32 {
        return self.y + @as(i32, @intCast(self.height));
    }

    /// Returns true if the point (px, py) lies inside this region.
    pub fn contains(self: Self, px: i32, py: i32) bool {
        return self.x <= px and px < self.stopX() and self.y <= py and py < self.stopY();
    }

    /// Returns true if the given region lies entirely inside this region.
    pub fn containsRegion(self: Self, other: Self) bool {
        return self.x <= other.x and self.stopX() >= other.stopX() and
            self.y <= other.y and self.stopY() >= other.stopY();
    }

    /// Returns a region starting left columns and above rows before and stopping
    /// right columns and below rows after the current region.
    pub fn inflated(self: Self, left: i32, above: i32, right: i32, below: i32) Self {
        const new_x = self.x - left;
        const new_y = self.y - above;
        const new_stop_x = self.stopX() + right;
        const new_stop_y = self.stopY() + below;
        return .{
            .x = new_x,
            .y = new_y,
            .width = @intCast(@max(0, new_stop_x - new_x)),
            .height = @intCast(@max(0, new_stop_y - new_y)),
        };
    }

    /// Returns a region inflated by the same margin on all sides.
    pub fn inflatedUniform(self: Self, margin: i32) Self {
        return self.inflated(margin, margin, margin, margin);
    }

    /// Returns a region inflated by the dimensions of the given Margin.
    pub fn inflatedByMargin(self: Self, margin: Margin) Self {
        return self.inflated(
            @intCast(margin.left),
            @intCast(margin.top),
            @intCast(margin.right),
            @intCast(margin.bottom),
        );
    }

    /// Returns a region starting left columns and above rows after and stopping
    /// right columns and below rows before the current region.
    pub fn deflated(self: Self, left: i32, above: i32, right: i32, below: i32) Self {
        return self.inflated(-left, -above, -right, -below);
    }

    /// Returns a region deflated by the same margin on all sides.
    pub fn deflatedUniform(self: Self, margin: i32) Self {
        return self.deflated(margin, margin, margin, margin);
    }

    /// Returns a region representing the current region in an image horizontally times wider
    /// and vertically times higher than the current.
    pub fn upscaled(self: Self, horizontally: i32, vertically: i32) Self {
        return .{
            .x = self.x * horizontally,
            .y = self.y * vertically,
            .width = self.width * @as(u32, @intCast(horizontally)),
            .height = self.height * @as(u32, @intCast(vertically)),
        };
    }

    /// Returns a region representing the current region in an image horizontally times narrower
    /// and vertically times lower than the current. Start coordinates are rounded down and stop
    /// coordinates are rounded up.
    pub fn downscaled(self: Self, horizontally: i32, vertically: i32) Self {
        const new_x = @divFloor(self.x, horizontally);
        const new_y = @divFloor(self.y, vertically);
        const new_stop_x = divCeil(self.stopX(), horizontally);
        const new_stop_y = divCeil(self.stopY(), vertically);
        return .{
            .x = new_x,
            .y = new_y,
            .width = @intCast(@max(0, new_stop_x - new_x)),
            .height = @intCast(@max(0, new_stop_y - new_y)),
        };
    }

    /// Returns true if the other region intersects with this region.
    pub fn intersectsWith(self: Self, other: Self) bool {
        return @max(self.x, other.x) < @min(self.stopX(), other.stopX()) and
            @max(self.y, other.y) < @min(self.stopY(), other.stopY());
    }

    /// Returns the intersection of two regions.
    pub fn intersection(self: Self, other: Self) Self {
        const new_x = @max(self.x, other.x);
        const new_y = @max(self.y, other.y);
        const new_stop_x = @min(self.stopX(), other.stopX());
        const new_stop_y = @min(self.stopY(), other.stopY());
        return .{
            .x = new_x,
            .y = new_y,
            .width = @intCast(@max(0, new_stop_x - new_x)),
            .height = @intCast(@max(0, new_stop_y - new_y)),
        };
    }

    /// Returns a region shifted by (dx, dy) — same size, moved origin.
    pub fn shifted(self: Self, dx: i32, dy: i32) Self {
        return .{
            .x = self.x + dx,
            .y = self.y + dy,
            .width = self.width,
            .height = self.height,
        };
    }

    /// Returns the smallest region containing both regions a and b.
    pub fn Union(self: Self, b: Self) Self {
        const new_x = @min(self.x, b.x);
        const new_y = @min(self.y, b.y);
        const new_stop_x = @max(self.stopX(), b.stopX());
        const new_stop_y = @max(self.stopY(), b.stopY());
        return .{
            .x = new_x,
            .y = new_y,
            .width = @intCast(@max(0, new_stop_x - new_x)),
            .height = @intCast(@max(0, new_stop_y - new_y)),
        };
    }
};

/// Ceiling division for signed integers.
fn divCeil(a: i32, b: i32) i32 {
    return -@divFloor(-a, b);
}

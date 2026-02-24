//! Tests for region.zig - Region and Margin types

const std = @import("std");
const zpp = @import("zpp");

// MARK: Margin tests

test "Margin predicates comprehensive" {
    // Zero margin
    const zero: zpp.Margin = .{};
    try std.testing.expect(zero.isZero());
    try std.testing.expect(!zero.isHorizontal());
    try std.testing.expect(!zero.isVertical());
    try std.testing.expect(!zero.isIsotropic());
    try std.testing.expect(!zero.is2D());
    try std.testing.expectEqual(@as(u32, 0), zero.maxExtent());

    // Horizontal margin
    const horiz_manual: zpp.Margin = .{ .left = 2, .right = 2 };
    const horiz = zpp.Margin.horizontal(2);
    try std.testing.expectEqual(horiz, horiz_manual);
    try std.testing.expect(!horiz.isZero());
    try std.testing.expect(horiz.isHorizontal());
    try std.testing.expect(!horiz.isVertical());
    try std.testing.expect(!horiz.isIsotropic());
    try std.testing.expect(!horiz.is2D());
    try std.testing.expectEqual(@as(u32, 2), horiz.maxExtent());

    // Vertical margin
    const vert_manual: zpp.Margin = .{ .top = 3, .bottom = 3 };
    const vert = zpp.Margin.vertical(3);
    try std.testing.expectEqual(vert, vert_manual);
    try std.testing.expect(!vert.isZero());
    try std.testing.expect(!vert.isHorizontal());
    try std.testing.expect(vert.isVertical());
    try std.testing.expect(!vert.isIsotropic());
    try std.testing.expect(!vert.is2D());
    try std.testing.expectEqual(@as(u32, 3), vert.maxExtent());

    // Isotropic margin
    const isot_manual: zpp.Margin = .{ .left = 5, .right = 5, .top = 5, .bottom = 5 };
    const isot = zpp.Margin.uniform(5);
    try std.testing.expectEqual(isot, isot_manual);
    try std.testing.expect(!isot.isZero());
    try std.testing.expect(!isot.isHorizontal());
    try std.testing.expect(!isot.isVertical());
    try std.testing.expect(isot.isIsotropic());
    try std.testing.expect(isot.is2D());
    try std.testing.expectEqual(@as(u32, 5), isot.maxExtent());

    // 2D margin (asymmetric)
    const asym: zpp.Margin = .{ .left = 1, .right = 2, .top = 3, .bottom = 4 };
    try std.testing.expect(!asym.isZero());
    try std.testing.expect(!asym.isHorizontal());
    try std.testing.expect(!asym.isVertical());
    try std.testing.expect(!asym.isIsotropic());
    try std.testing.expect(asym.is2D());
    try std.testing.expectEqual(@as(u32, 4), asym.maxExtent());
}

test "Margin helpers" {
    // Margin.horizontal
    const h = zpp.Margin.horizontal(3);
    try std.testing.expectEqual(@as(u32, 3), h.left);
    try std.testing.expectEqual(@as(u32, 3), h.right);
    try std.testing.expectEqual(@as(u32, 0), h.top);
    try std.testing.expectEqual(@as(u32, 0), h.bottom);
    try std.testing.expect(h.isHorizontal());
    try std.testing.expect(!h.isVertical());

    // Margin.vertical
    const v = zpp.Margin.vertical(2);
    try std.testing.expectEqual(@as(u32, 0), v.left);
    try std.testing.expectEqual(@as(u32, 0), v.right);
    try std.testing.expectEqual(@as(u32, 2), v.top);
    try std.testing.expectEqual(@as(u32, 2), v.bottom);
    try std.testing.expect(v.isVertical());
    try std.testing.expect(!v.isHorizontal());

    // Margin.uniform
    const i = zpp.Margin.uniform(4);
    try std.testing.expectEqual(@as(u32, 4), i.left);
    try std.testing.expectEqual(@as(u32, 4), i.right);
    try std.testing.expectEqual(@as(u32, 4), i.top);
    try std.testing.expectEqual(@as(u32, 4), i.bottom);
    try std.testing.expect(i.isIsotropic());
}

// MARK: Region tests

test "Region operations" {
    const region: zpp.Region = .{ .x = 10, .y = 20, .width = 100, .height = 50 };

    // Basic properties
    try std.testing.expectEqual(@as(i32, 110), region.stopX());
    try std.testing.expectEqual(@as(i32, 70), region.stopY());
    try std.testing.expectEqual(@as(u32, 5000), region.area());

    // Contains point
    try std.testing.expect(region.contains(10, 20)); // Top-left corner (inclusive)
    try std.testing.expect(region.contains(50, 40)); // Interior
    try std.testing.expect(!region.contains(110, 70)); // Stop coordinates (exclusive)
    try std.testing.expect(!region.contains(9, 20)); // Left of region
    try std.testing.expect(!region.contains(10, 19)); // Above region

    // Inflated
    const inflated = region.inflatedUniform(5);
    try std.testing.expectEqual(@as(i32, 5), inflated.x);
    try std.testing.expectEqual(@as(i32, 15), inflated.y);
    try std.testing.expectEqual(@as(u32, 110), inflated.width);
    try std.testing.expectEqual(@as(u32, 60), inflated.height);

    // Deflated
    const deflated = region.deflatedUniform(5);
    try std.testing.expectEqual(@as(i32, 15), deflated.x);
    try std.testing.expectEqual(@as(i32, 25), deflated.y);
    try std.testing.expectEqual(@as(u32, 90), deflated.width);
    try std.testing.expectEqual(@as(u32, 40), deflated.height);

    // Upscaled
    const upscaled = region.upscaled(2, 3);
    try std.testing.expectEqual(@as(i32, 20), upscaled.x);
    try std.testing.expectEqual(@as(i32, 60), upscaled.y);
    try std.testing.expectEqual(@as(u32, 200), upscaled.width);
    try std.testing.expectEqual(@as(u32, 150), upscaled.height);

    // Intersection
    const other: zpp.Region = .{ .x = 50, .y = 40, .width = 100, .height = 50 };
    const inter = region.intersection(other);
    try std.testing.expectEqual(@as(i32, 50), inter.x);
    try std.testing.expectEqual(@as(i32, 40), inter.y);
    try std.testing.expectEqual(@as(u32, 60), inter.width);
    try std.testing.expectEqual(@as(u32, 30), inter.height);

    // Union
    const uni = region.merge(other);
    try std.testing.expectEqual(@as(i32, 10), uni.x);
    try std.testing.expectEqual(@as(i32, 20), uni.y);
    try std.testing.expectEqual(@as(u32, 140), uni.width);
    try std.testing.expectEqual(@as(u32, 70), uni.height);
}

test "Region edge cases" {
    // Zero-area region
    const zero_region: zpp.Region = .{ .x = 0, .y = 0, .width = 0, .height = 0 };
    try std.testing.expectEqual(@as(u32, 0), zero_region.area());
    try std.testing.expectEqual(@as(i32, 0), zero_region.stopX());
    try std.testing.expectEqual(@as(i32, 0), zero_region.stopY());
    try std.testing.expect(!zero_region.contains(0, 0));

    // Single pixel region
    const single: zpp.Region = .{ .x = 5, .y = 10, .width = 1, .height = 1 };
    try std.testing.expectEqual(@as(u32, 1), single.area());
    try std.testing.expect(single.contains(5, 10));
    try std.testing.expect(!single.contains(6, 10));
    try std.testing.expect(!single.contains(5, 11));

    // Negative origin region
    const neg_region: zpp.Region = .{ .x = -10, .y = -20, .width = 30, .height = 40 };
    try std.testing.expectEqual(@as(i32, 20), neg_region.stopX());
    try std.testing.expectEqual(@as(i32, 20), neg_region.stopY());
    try std.testing.expect(neg_region.contains(-5, 0));
    try std.testing.expect(neg_region.contains(0, 0));

    // Non-intersecting regions
    const r1: zpp.Region = .{ .x = 0, .y = 0, .width = 10, .height = 10 };
    const r2: zpp.Region = .{ .x = 20, .y = 20, .width = 10, .height = 10 };
    try std.testing.expect(!r1.intersectsWith(r2));
    const inter = r1.intersection(r2);
    try std.testing.expectEqual(@as(u32, 0), inter.width);
    try std.testing.expectEqual(@as(u32, 0), inter.height);
}

test "Region downscaled with rounding" {
    // Test that downscaled properly rounds start down and stop up
    const region: zpp.Region = .{ .x = 3, .y = 5, .width = 7, .height = 9 };
    const downscaled = region.downscaled(2, 2);

    // startX: floor(3/2) = 1, stopX: ceil(10/2) = 5, width = 4
    // startY: floor(5/2) = 2, stopY: ceil(14/2) = 7, height = 5
    try std.testing.expectEqual(@as(i32, 1), downscaled.x);
    try std.testing.expectEqual(@as(i32, 2), downscaled.y);
    try std.testing.expectEqual(@as(u32, 4), downscaled.width);
    try std.testing.expectEqual(@as(u32, 5), downscaled.height);
}

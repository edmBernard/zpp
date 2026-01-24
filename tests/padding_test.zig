//! Tests for padding.zig - Padding policies and loop options

const std = @import("std");
const zpp = @import("zpp");

test "RepeatEdgePadding with empty region" {
    const empty_region: zpp.Region = .{ .x = 0, .y = 0, .width = 0, .height = 0 };
    var data: [0]f32 = .{};

    // Should return 0 for any coordinate with empty region
    const result = zpp.RepeatEdgePadding.apply(f32, &data, 0, 0, 0, empty_region);
    try std.testing.expectEqual(@as(f32, 0), result);

    const result2 = zpp.RepeatEdgePadding.apply(f32, &data, 0, 10, 10, empty_region);
    try std.testing.expectEqual(@as(f32, 0), result2);
}

test "RepeatEdgePadding with zero width" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 0, .height = 5 };
    var data: [0]f32 = .{};

    const result = zpp.RepeatEdgePadding.apply(f32, &data, 0, 0, 0, region);
    try std.testing.expectEqual(@as(f32, 0), result);
}

test "RepeatEdgePadding with zero height" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 0 };
    var data: [0]f32 = .{};

    const result = zpp.RepeatEdgePadding.apply(f32, &data, 5, 0, 0, region);
    try std.testing.expectEqual(@as(f32, 0), result);
}

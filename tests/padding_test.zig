//! Tests for padding.zig - Padding policies and loop options

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

/// Common SIMD vector types
const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };
const AllScalarTypes = [_]type{ f32, u16, u8 };

test "Source: Default padding repeats edge pixels correctly" {
    const image_width = 9;
    const image_height = 5;

    const input_region: zpp.Region = .{ .x = 1, .y = 1, .width = 4, .height = 2 };
    const input_stride = image_width;

    const output_region: zpp.Region = .{ .x = 2, .y = 1, .width = 5, .height = 3 };
    const output_stride = image_width;

    inline for (AllTypes) |DataType| {
        const ScalarType = @typeInfo(DataType).vector.child;

        var source_data = [_]ScalarType{0} ** (image_width * image_height);
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.In(ScalarType, &source_data, input_stride, input_region);

        var output_data = [_]ScalarType{0} ** (image_width * image_height);
        const destination = zpp.Out(ScalarType, &output_data, output_stride, output_region);

        zpp.Process(source, destination);

        const expected_data: [45]ScalarType = .{
            0, 0, 0,  0,  0,  0,  0,  0, 0,
            0, 0, 12, 13, 14, 14, 14, 0, 0,
            0, 0, 21, 22, 23, 23, 23, 0, 0,
            0, 0, 21, 22, 23, 23, 23, 0, 0,
            0, 0, 0,  0,  0,  0,  0,  0, 0,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

test "Source: RepeatPadding repeats edge pixels correctly" {
    const image_width = 9;
    const image_height = 5;

    const input_region: zpp.Region = .{ .x = 1, .y = 1, .width = 4, .height = 2 };
    const input_stride = image_width;

    const output_region: zpp.Region = .{ .x = 2, .y = 1, .width = 5, .height = 3 };
    const output_stride = image_width;

    inline for (AllTypes) |DataType| {
        const ScalarType = @typeInfo(DataType).vector.child;

        var source_data = [_]ScalarType{0} ** (image_width * image_height);
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.InWithPadding(ScalarType, zpp.RepeatEdgePadding, &source_data, input_stride, input_region);

        var output_data = [_]ScalarType{0} ** (image_width * image_height);
        const destination = zpp.Out(ScalarType, &output_data, output_stride, output_region);

        zpp.Process(source, destination);

        const expected_data: [45]ScalarType = .{
            0, 0, 0,  0,  0,  0,  0,  0, 0,
            0, 0, 12, 13, 14, 14, 14, 0, 0,
            0, 0, 21, 22, 23, 23, 23, 0, 0,
            0, 0, 21, 22, 23, 23, 23, 0, 0,
            0, 0, 0,  0,  0,  0,  0,  0, 0,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

test "Source: ZeroPadding fill edge pixels correctly" {
    const image_width = 9;
    const image_height = 5;

    const input_region: zpp.Region = .{ .x = 1, .y = 1, .width = 4, .height = 2 };
    const input_stride = image_width;

    const output_region: zpp.Region = .{ .x = 2, .y = 1, .width = 5, .height = 3 };
    const output_stride = image_width;

    inline for (AllTypes) |DataType| {
        const ScalarType = @typeInfo(DataType).vector.child;

        var source_data = [_]ScalarType{0} ** (image_width * image_height);
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.InWithPadding(ScalarType, zpp.ZeroPadding, &source_data, input_stride, input_region);

        var output_data = [_]ScalarType{1} ** (image_width * image_height);
        const destination = zpp.Out(ScalarType, &output_data, output_stride, output_region);

        zpp.Process(source, destination);

        const expected_data: [45]ScalarType = .{
            1, 1, 1,  1,  1,  1, 1, 1, 1,
            1, 1, 12, 13, 14, 0, 0, 1, 1,
            1, 1, 21, 22, 23, 0, 0, 1, 1,
            1, 1, 0,  0,  0,  0, 0, 1, 1,
            1, 1, 1,  1,  1,  1, 1, 1, 1,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: RepeatEdgePadding: empty and zero-dimension regions across types
test "RepeatEdgePadding: empty and zero-dimension regions across types" {
    inline for (AllScalarTypes) |ScalarType| {
        // Zero width region
        {
            const region: zpp.Region = .{ .x = 0, .y = 0, .width = 0, .height = 5 };
            var data: [0]ScalarType = .{};
            const result = zpp.RepeatEdgePadding.apply(ScalarType, &data, 0, 0, 0, region);
            try std.testing.expectEqual(@as(ScalarType, 0), result);
        }

        // Zero height region
        {
            const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 0 };
            var data: [0]ScalarType = .{};
            const result = zpp.RepeatEdgePadding.apply(ScalarType, &data, 5, 0, 0, region);
            try std.testing.expectEqual(@as(ScalarType, 0), result);
        }

        // Fully empty region (width=0, height=0)
        {
            const region: zpp.Region = .{ .x = 0, .y = 0, .width = 0, .height = 0 };
            var data: [0]ScalarType = .{};
            const result = zpp.RepeatEdgePadding.apply(ScalarType, &data, 0, 0, 0, region);
            try std.testing.expectEqual(@as(ScalarType, 0), result);

            // Also test with non-zero coordinates
            const result2 = zpp.RepeatEdgePadding.apply(ScalarType, &data, 0, 10, 10, region);
            try std.testing.expectEqual(@as(ScalarType, 0), result2);
        }
    }
}

// MARK: ZeroPadding: boundary values across types
test "ZeroPadding: boundary values across types" {
    inline for (AllScalarTypes) |ScalarType| {
        const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };
        // data: [10, 20, 30, 40, 50, 60, 70, 80]
        var data: [8]ScalarType = undefined;
        th.fillRamp(ScalarType, &data, 10, 10);

        // Last valid x position (3, 0) -> should return data
        const valid = zpp.ZeroPadding.apply(ScalarType, &data, 4, 3, 0, region);
        // ZeroPadding.apply returns 0 for everything (it's the OOB handler)
        // The check for in-bounds is done in InputSource.read, not in the policy
        // ZeroPadding.apply is only called for OOB, so it always returns 0
        try std.testing.expectEqual(@as(ScalarType, 0), valid);

        // First invalid x position (4, 0) -> zero
        const invalid_right = zpp.ZeroPadding.apply(ScalarType, &data, 4, 4, 0, region);
        try std.testing.expectEqual(@as(ScalarType, 0), invalid_right);

        // First invalid y position (0, 2) -> zero
        const invalid_bottom = zpp.ZeroPadding.apply(ScalarType, &data, 4, 0, 2, region);
        try std.testing.expectEqual(@as(ScalarType, 0), invalid_bottom);

        // Negative coordinates -> zero
        const invalid_left = zpp.ZeroPadding.apply(ScalarType, &data, 4, -1, 0, region);
        try std.testing.expectEqual(@as(ScalarType, 0), invalid_left);

        const invalid_top = zpp.ZeroPadding.apply(ScalarType, &data, 4, 0, -1, region);
        try std.testing.expectEqual(@as(ScalarType, 0), invalid_top);
    }
}

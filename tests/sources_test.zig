//! Tests for sources.zig - Input and Output sources

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

/// Common SIMD vector types
const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };
const AllScalarTypes = [_]type{ f32, u16, u8 };

test "Source: Naked In source reads correct values" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    inline for (AllTypes) |DataType| {
        const ScalarType = @typeInfo(DataType).vector.child;

        var output_data = [_]ScalarType{0} ** 4;
        const destination = zpp.Out(ScalarType, &output_data, region.width, region);

        var source_data = [_]ScalarType{0} ** 4;
        th.fillRamp(ScalarType, &source_data, 1, 3);
        const source = zpp.In(ScalarType, &source_data, region.width, region);

        zpp.Process(source, destination);

        try std.testing.expectEqual(source_data, output_data);
    }
}

test "Source: Naked In source reads correct values: larger and odd region" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 3 };

    inline for (AllTypes) |DataType| {
        const ScalarType = @typeInfo(DataType).vector.child;

        var output_data = [_]ScalarType{0} ** 15;
        const destination = zpp.Out(ScalarType, &output_data, region.width, region);

        var source_data = [_]ScalarType{0} ** 15;
        th.fillRamp(ScalarType, &source_data, 1, 3);
        const source = zpp.In(ScalarType, &source_data, region.width, region);

        zpp.Process(source, destination);

        try std.testing.expectEqual(source_data, output_data);
    }
}

test "Source: Only fill requested region: Same global size, different regions" {
    const input_region: zpp.Region = .{ .x = 0, .y = 0, .width = 9, .height = 5 };

    const output_region: zpp.Region = .{ .x = 2, .y = 1, .width = 5, .height = 3 };
    const output_stride = input_region.width;

    inline for (AllTypes) |DataType| {
        const ScalarType = @typeInfo(DataType).vector.child;

        var source_data = [_]ScalarType{0} ** 45;
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.In(ScalarType, &source_data, input_region.width, input_region);

        var output_data = [_]ScalarType{0} ** 45;
        const destination = zpp.Out(ScalarType, &output_data, output_stride, output_region);

        zpp.Process(source, destination);

        const expected_data: [45]ScalarType = .{
            0, 0, 0,  0,  0,  0,  0,  0, 0,
            0, 0, 12, 13, 14, 15, 16, 0, 0,
            0, 0, 21, 22, 23, 24, 25, 0, 0,
            0, 0, 30, 31, 32, 33, 34, 0, 0,
            0, 0, 0,  0,  0,  0,  0,  0, 0,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

test "Source: Only fill requested region: Same global size, different regions, padding" {
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

// test "Padding policy edge behavior" {
//     const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

//     // 2x4 input: [[1,2,3,4], [5,6,7,8]]
//     var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

//     // Test RepeatEdgePadding
//     const repeat_source = zpp.In(f32, &input_data, region.width, region);
//     try std.testing.expectEqual(@as(f32, 1.0), repeat_source.read(-1, 0)); // left edge
//     try std.testing.expectEqual(@as(f32, 4.0), repeat_source.read(4, 0)); // right edge
//     try std.testing.expectEqual(@as(f32, 1.0), repeat_source.read(0, -1)); // top edge
//     try std.testing.expectEqual(@as(f32, 5.0), repeat_source.read(0, 2)); // bottom edge
//     try std.testing.expectEqual(@as(f32, 1.0), repeat_source.read(-1, -1)); // corner

//     // Test ZeroPadding
//     const zero_source = zpp.InWithPadding(f32, zpp.ZeroPadding, &input_data, region.width, region);
//     try std.testing.expectEqual(@as(f32, 0.0), zero_source.read(-1, 0));
//     try std.testing.expectEqual(@as(f32, 0.0), zero_source.read(4, 0));
//     try std.testing.expectEqual(@as(f32, 0.0), zero_source.read(0, -1));
//     try std.testing.expectEqual(@as(f32, 0.0), zero_source.read(0, 2));
// }

// test "readVec in-bounds SIMD load" {
//     const region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 2 };

//     // Input: [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16]]
//     var input_data: [16]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
//     const source = zpp.In(f32, &input_data, region.width, region);

//     // In-bounds read at (0, 0) -> should load [1, 2, 3, 4]
//     const vec1 = source.readVec(f32x4, 0, 0);
//     try std.testing.expectEqual(f32x4{ 1, 2, 3, 4 }, vec1);

//     // In-bounds read at (4, 0) -> should load [5, 6, 7, 8]
//     const vec2 = source.readVec(f32x4, 4, 0);
//     try std.testing.expectEqual(f32x4{ 5, 6, 7, 8 }, vec2);

//     // In-bounds read at (2, 1) -> should load [11, 12, 13, 14]
//     const vec3 = source.readVec(f32x4, 2, 1);
//     try std.testing.expectEqual(f32x4{ 11, 12, 13, 14 }, vec3);
// }

// test "readVec out-of-bounds with RepeatEdgePadding" {
//     const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

//     // Input: [[1,2,3,4], [5,6,7,8]]
//     var input_data: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
//     const source = zpp.In(f32, &input_data, region.width, region);

//     // Read starting at x=-1 (left edge clamp) -> [1, 1, 2, 3]
//     const vec1 = source.readVec(f32x4, -1, 0);
//     try std.testing.expectEqual(f32x4{ 1, 1, 2, 3 }, vec1);

//     // Read starting at x=2 (right edge extends past) -> [3, 4, 4, 4]
//     const vec2 = source.readVec(f32x4, 2, 0);
//     try std.testing.expectEqual(f32x4{ 3, 4, 4, 4 }, vec2);

//     // Read at y=-1 (top edge clamp) -> [1, 2, 3, 4]
//     const vec3 = source.readVec(f32x4, 0, -1);
//     try std.testing.expectEqual(f32x4{ 1, 2, 3, 4 }, vec3);

//     // Read at y=2 (bottom edge clamp) -> [5, 6, 7, 8]
//     const vec4 = source.readVec(f32x4, 0, 2);
//     try std.testing.expectEqual(f32x4{ 5, 6, 7, 8 }, vec4);
// }

// test "readVec out-of-bounds with ZeroPadding" {
//     const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

//     // Input: [[1,2,3,4], [5,6,7,8]]
//     var input_data: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
//     const source = zpp.InWithPadding(f32, zpp.ZeroPadding, &input_data, region.width, region);

//     // Read starting at x=-1 (left edge zero) -> [0, 1, 2, 3]
//     const vec1 = source.readVec(f32x4, -1, 0);
//     try std.testing.expectEqual(f32x4{ 0, 1, 2, 3 }, vec1);

//     // Read starting at x=2 (right edge extends past) -> [3, 4, 0, 0]
//     const vec2 = source.readVec(f32x4, 2, 0);
//     try std.testing.expectEqual(f32x4{ 3, 4, 0, 0 }, vec2);

//     // Read at y=-1 (top edge zero) -> [0, 0, 0, 0]
//     const vec3 = source.readVec(f32x4, 0, -1);
//     try std.testing.expectEqual(f32x4{ 0, 0, 0, 0 }, vec3);

//     // Read at y=2 (bottom edge zero) -> [0, 0, 0, 0]
//     const vec4 = source.readVec(f32x4, 0, 2);
//     try std.testing.expectEqual(f32x4{ 0, 0, 0, 0 }, vec4);
// }

// test "readVec non-origin region" {
//     const region: zpp.Region = .{ .x = 10, .y = 20, .width = 4, .height = 2 };

//     // Input: [[1,2,3,4], [5,6,7,8]]
//     var input_data: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
//     const source = zpp.In(f32, &input_data, region.width, region);

//     // In-bounds read at region origin (10, 20) -> should load [1, 2, 3, 4]
//     const vec1 = source.readVec(f32x4, 10, 20);
//     try std.testing.expectEqual(f32x4{ 1, 2, 3, 4 }, vec1);

//     // Out-of-bounds left of region (9, 20) -> [1, 1, 2, 3] with RepeatEdge
//     const vec2 = source.readVec(f32x4, 9, 20);
//     try std.testing.expectEqual(f32x4{ 1, 1, 2, 3 }, vec2);
// }

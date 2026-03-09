const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };

test "Generator: produce correct coordinates type: smaller region than batch size" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output: [4]ScalarType = .{ 0, 0, 0, 0 };
            const destination = try zpp.makeDest(ScalarType, &output, region.width, region);

            const processing_kernel = struct {
                fn process(ctx: anytype, x: anytype, y: anytype) OutputType {
                    _ = ctx;
                    const VecType = @TypeOf(x);
                    if (VecType != CoordType) {
                        @compileError("Expected coordinate vectors to match CoordType");
                    }
                    return th.vectorCast(OutputType, x) + th.vectorCast(OutputType, y) * th.splatWithCast(OutputType, 10);
                }
            };

            const result = zpp.generate(CoordType, .{}, processing_kernel.process);
            zpp.process(result, destination);

            try std.testing.expectEqual(@as(ScalarType, 0.0), output[0]);
            try std.testing.expectEqual(@as(ScalarType, 1.0), output[1]);
            try std.testing.expectEqual(@as(ScalarType, 10.0), output[2]);
            try std.testing.expectEqual(@as(ScalarType, 11.0), output[3]);
        }
    }
}

test "Generator: produce correct coordinates type: larger region than batch size" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output = [_]ScalarType{0} ** 16;
            const destination = try zpp.makeDest(ScalarType, &output, region.width, region);

            const processing_kernel = struct {
                fn process(ctx: anytype, x: anytype, y: anytype) OutputType {
                    _ = ctx;
                    const VecType = @TypeOf(x);
                    if (VecType != CoordType) {
                        @compileError("Expected coordinate vectors to match CoordType");
                    }
                    return th.vectorCast(OutputType, x) + th.vectorCast(OutputType, y) * th.splatWithCast(OutputType, 10);
                }
            };

            const result = zpp.generate(CoordType, .{}, processing_kernel.process);
            zpp.process(result, destination);

            var expectedOutput = [_]ScalarType{0} ** 16;
            for (0..region.height) |y| {
                for (0..region.width) |x| {
                    const idx = y * region.width + x;
                    expectedOutput[idx] += th.scalarCast(ScalarType, x) + th.scalarCast(ScalarType, y) * 10;
                }
            }
            try std.testing.expectEqual(expectedOutput, output);
        }
    }
}

test "Generator: produce correct coordinates type: region width not multiple of batch size" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 4 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output = [_]ScalarType{0} ** 20;
            const destination = try zpp.makeDest(ScalarType, &output, region.width, region);

            const processing_kernel = struct {
                fn process(ctx: anytype, x: anytype, y: anytype) OutputType {
                    _ = ctx;
                    const VecType = @TypeOf(x);
                    if (VecType != CoordType) {
                        @compileError("Expected coordinate vectors to match CoordType");
                    }
                    return th.vectorCast(OutputType, x) + th.vectorCast(OutputType, y) * th.splatWithCast(OutputType, 10);
                }
            };

            const result = zpp.generate(CoordType, .{}, processing_kernel.process);
            zpp.process(result, destination);

            var expectedOutput = [_]ScalarType{0} ** 20;
            for (0..region.height) |y| {
                for (0..region.width) |x| {
                    const idx = y * region.width + x;
                    expectedOutput[idx] += th.scalarCast(ScalarType, x) + th.scalarCast(ScalarType, y) * 10;
                }
            }
            try std.testing.expectEqual(expectedOutput, output);
        }
    }
}

test "Generator: Only fill requested region: Same global size, different regions" {
    const image_width = 9;
    const image_height = 5;

    const output_region: zpp.Region = .{ .x = 2, .y = 1, .width = 5, .height = 3 };
    const output_stride = image_width;

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output_data = [_]ScalarType{0} ** (image_width * image_height);
            const destination = try zpp.makeDest(ScalarType, &output_data, output_stride, output_region);

            const processing_kernel = struct {
                fn process(ctx: anytype, x: anytype, y: anytype) OutputType {
                    _ = ctx;
                    const VecType = @TypeOf(x);
                    if (VecType != CoordType) {
                        @compileError("Expected coordinate vectors to match CoordType");
                    }
                    return th.vectorCast(OutputType, x) + th.vectorCast(OutputType, y) * th.splatWithCast(OutputType, 10);
                }
            };
            const result = zpp.generate(CoordType, .{}, processing_kernel.process);
            zpp.process(result, destination);

            const expected_data: [45]ScalarType = .{
                0, 0, 0,  0,  0,  0,  0,  0, 0,
                0, 0, 12, 13, 14, 15, 16, 0, 0,
                0, 0, 22, 23, 24, 25, 26, 0, 0,
                0, 0, 32, 33, 34, 35, 36, 0, 0,
                0, 0, 0,  0,  0,  0,  0,  0, 0,
            };
            try std.testing.expectEqual(expected_data, output_data);
        }
    }
}

// FIXME: This is a design change currently only process can take a generator loop and other stuff expect a source with a region
// test "Generator: Generate followed by loop" {
//     const image_width = 9;
//     const image_height = 5;

//     const output_region: zpp.Region = .{ .x = 2, .y = 1, .width = 5, .height = 3 };
//     const output_stride = image_width;

//     inline for (AllTypes) |CoordType| {
//         inline for (AllTypes) |OutputType| {
//             const ScalarType = @typeInfo(OutputType).vector.child;

//             var output_data = [_]ScalarType{0} ** (image_width * image_height);
//             const destination = try zpp.makeDest(ScalarType, &output_data, output_stride, output_region);
//             const generator_kernel = struct {
//                 fn process(ctx: anytype, x: anytype, y: anytype) OutputType {
//                     _ = ctx;
//                     return th.vectorCast(OutputType, x) + th.vectorCast(OutputType, y);
//                 }
//             };
//             const processing_kernel = struct {
//                 fn process(ctx: anytype, in: anytype) OutputType {
//                     _ = ctx;
//                     return in.get() * th.splatWithCast(OutputType, 10);
//                 }
//             };
//             const generator = zpp.generate(CoordType, .{}, generator_kernel.process);
//             const result = zpp.loop(OutputType, .{}, generator, .{}, processing_kernel.process);
//             zpp.process(result, destination);

//             const expected_data: [45]ScalarType = .{
//                 0, 0, 0,  0,  0,  0,  0,  0, 0,
//                 0, 0, 12, 13, 14, 15, 16, 0, 0,
//                 0, 0, 22, 23, 24, 25, 26, 0, 0,
//                 0, 0, 32, 33, 34, 35, 36, 0, 0,
//                 0, 0, 0,  0,  0,  0,  0,  0, 0,
//             };
//             try std.testing.expectEqual(expected_data, output_data);
//         }
//     }
// }

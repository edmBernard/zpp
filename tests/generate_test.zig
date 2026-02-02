const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };

test "Generator: produce correct coordinates type: smaller width than batch size" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output: [4]ScalarType = .{ 0, 0, 0, 0 };
            const destination = zpp.Out(ScalarType, &output, region.width, region);

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

            const result = zpp.Generate(CoordType, region, .{}, processing_kernel.process);
            zpp.Process(result, destination);

            try std.testing.expectEqual(@as(ScalarType, 0.0), output[0]);
            try std.testing.expectEqual(@as(ScalarType, 1.0), output[1]);
            try std.testing.expectEqual(@as(ScalarType, 10.0), output[2]);
            try std.testing.expectEqual(@as(ScalarType, 11.0), output[3]);
        }
    }
}

test "Generator: produce correct coordinates type: larger width than batch size" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 4 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output = [_]ScalarType{0} ** 16;
            const destination = zpp.Out(ScalarType, &output, region.width, region);

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

            const result = zpp.Generate(CoordType, region, .{}, processing_kernel.process);
            zpp.Process(result, destination);

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

test "Generator: produce correct coordinates type: width not multiple of batch size" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 4 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output = [_]ScalarType{0} ** 20;
            const destination = zpp.Out(ScalarType, &output, region.width, region);

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

            const result = zpp.Generate(CoordType, region, .{}, processing_kernel.process);
            zpp.Process(result, destination);

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

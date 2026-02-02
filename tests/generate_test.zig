const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };

test "Generator: produce correct coordinates type" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        var output: [4]ScalarType = .{ 0, 0, 0, 0 };
        const destination = zpp.Out(ScalarType, &output, region.width, region);

        const processing_kernel = struct {
            fn process(ctx: anytype, x: anytype, y: anytype) LoopType {
                _ = ctx;
                const VecType = @TypeOf(x);
                if (VecType != LoopType) {
                    @compileError("Expected coordinate vectors to match LoopType");
                }
                return x + y * zpp.splat(VecType, 10);
            }
        };

        const result = zpp.Generate(LoopType, region, .{}, processing_kernel.process);
        zpp.Process(result, destination);

        try std.testing.expectEqual(@as(ScalarType, 0.0), output[0]);
        try std.testing.expectEqual(@as(ScalarType, 1.0), output[1]);
        try std.testing.expectEqual(@as(ScalarType, 10.0), output[2]);
        try std.testing.expectEqual(@as(ScalarType, 11.0), output[3]);
    }
}

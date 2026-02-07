//! Tests for loop.zig - Core processing loop primitives

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

/// Common SIMD vector types
const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };
const AllScalarTypes = [_]type{ f32, u16, u8 };

// MARK: Loop: produce correct type accessor
test "Loop: produce correct type accessor" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        var input_data: [4]ScalarType = .{ 1.0, 2.0, 3.0, 4.0 };
        const source = zpp.In(ScalarType, &input_data, region.width, region);

        var output: [4]ScalarType = .{ 0, 0, 0, 0 };
        const destination = zpp.Out(ScalarType, &output, region.width, region);

        const processing_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                const value = in.get();
                const VecType = @TypeOf(value);
                if (VecType != LoopType) {
                    @compileError("Expected coordinate vectors to match LoopType");
                }
                return value * zpp.splat(VecType, 10);
            }
        };

        const result = zpp.Loop(LoopType, .{}, source, .{}, processing_kernel.process);
        zpp.Process(result, destination);

        try std.testing.expectEqual(@as(ScalarType, 10.0), output[0]);
        try std.testing.expectEqual(@as(ScalarType, 20.0), output[1]);
        try std.testing.expectEqual(@as(ScalarType, 30.0), output[2]);
        try std.testing.expectEqual(@as(ScalarType, 40.0), output[3]);
    }
}

// MARK: Loop: produce correct coordinates and loop type
test "Loop: produce correct coordinates and loop type" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };

    inline for (AllTypes) |LoopType| {
        inline for (AllTypes) |CoordType| {
            const ScalarType = @typeInfo(LoopType).vector.child;

            var input_data: [4]ScalarType = .{ 1.0, 2.0, 3.0, 4.0 };
            const source = zpp.In(ScalarType, &input_data, region.width, region);

            var output: [4]ScalarType = .{ 0, 0, 0, 0 };
            const destination = zpp.Out(ScalarType, &output, region.width, region);

            const processing_kernel = struct {
                fn process(ctx: anytype, in: anytype, x: anytype, y: anytype) LoopType {
                    _ = ctx;
                    const value = in.get();
                    const VecType = @TypeOf(value);
                    if (VecType != LoopType) {
                        @compileError("Expected coordinate vectors to match LoopType");
                    }
                    if (@TypeOf(x) != CoordType) {
                        @compileError("Expected coordinate vectors to match LoopType");
                    }
                    if (@TypeOf(y) != CoordType) {
                        @compileError("Expected coordinate vectors to match LoopType");
                    }
                    return th.vectorCast(VecType, x) + th.vectorCast(VecType, y) * th.splatWithCast(VecType, 10) + value * th.splatWithCast(VecType, 20);
                }
            };

            const result = zpp.Loop(LoopType, .{ .need_coordinates = CoordType }, source, .{}, processing_kernel.process);
            zpp.Process(result, destination);

            try std.testing.expectEqual(@as(ScalarType, 20.0), output[0]);
            try std.testing.expectEqual(@as(ScalarType, 41.0), output[1]);
            try std.testing.expectEqual(@as(ScalarType, 70.0), output[2]);
            try std.testing.expectEqual(@as(ScalarType, 91.0), output[3]);
        }
    }
}

// MARK: Loop: expression tree chains two kernels across types
test "Loop: expression tree chains two kernels across types" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        // 2x4 input
        var input_data: [8]ScalarType = undefined;
        th.fillRamp(ScalarType, &input_data, 1, 1);
        var output_data = [_]ScalarType{0} ** 8;

        const source = zpp.In(ScalarType, &input_data, region.width, region);
        const destination = zpp.Out(ScalarType, &output_data, region.width, region);

        // First kernel: scale by 2
        const scale_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                return in.get() * th.splatWithCast(LoopType, 2);
            }
        };

        // Second kernel: add 10
        const offset_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                return in.get() + th.splatWithCast(LoopType, 10);
            }
        };

        const step1 = zpp.Loop(LoopType, .{}, source, .{}, scale_kernel.process);
        const step2 = zpp.Loop(LoopType, .{}, step1, .{}, offset_kernel.process);
        zpp.Process(step2, destination);

        // output = input * 2 + 10
        const expected_data: [8]ScalarType = .{
            12, 14, 16, 18,
            20, 22, 24, 26,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Loop: non-origin region preserves data
test "Loop: non-origin region preserves data" {
    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        // Use non-origin region with a buffer large enough to cover absolute coordinates
        // Region starts at (2, 1), so data indices are: y*stride + x = 1*9 + 2 = 11 onwards
        const image_width = 9;
        const image_height = 3;
        const region: zpp.Region = .{ .x = 2, .y = 1, .width = 4, .height = 2 };

        var input_data = [_]ScalarType{0} ** (image_width * image_height);
        th.fillRamp(ScalarType, &input_data, 1, 1);
        var output_data = [_]ScalarType{0} ** (image_width * image_height);

        const source = zpp.In(ScalarType, &input_data, image_width, region);
        const destination = zpp.Out(ScalarType, &output_data, image_width, region);

        // Identity kernel
        const id_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                return in.get();
            }
        };

        const result = zpp.Loop(LoopType, .{}, source, .{}, id_kernel.process);
        zpp.Process(result, destination);

        const expected_data = [_]ScalarType{
            0, 0, 0,  0,  0,  0,  0, 0, 0,
            0, 0, 12, 13, 14, 15, 0, 0, 0,
            0, 0, 21, 22, 23, 24, 0, 0, 0,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Loop: margins with RepeatEdgePadding across types
test "Loop: margins with RepeatEdgePadding across types" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        var input_data: [4]ScalarType = .{ 1.0, 2.0, 3.0, 4.0 };
        var output_data = [_]ScalarType{0} ** 4;

        const source = zpp.In(ScalarType, &input_data, region.width, region);
        const destination = zpp.Out(ScalarType, &output_data, region.width, region);

        const processing_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                // Horizontal blur: left + center + right
                return in.getAt(-1, 0) + in.getAt(0, 0) + in.getAt(1, 0);
            }
        };

        const result = zpp.Loop(LoopType, .{ .margin = .{ .left = 1, .right = 1 } }, source, .{}, processing_kernel.process);
        zpp.Process(result, destination);

        const expected_data = [_]ScalarType{ 4, 6, 9, 11 };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Loop: generator with coordinates at non-origin region
test "Loop: generator with coordinates at non-origin region" {
    // Non-origin region: output buffer must cover absolute coordinates
    const image_width: u32 = 8;
    const image_height: u32 = 6;
    const region: zpp.Region = .{ .x = 2, .y = 3, .width = 4, .height = 2 };

    inline for (AllTypes) |CoordType| {
        inline for (AllTypes) |OutputType| {
            const ScalarType = @typeInfo(OutputType).vector.child;

            var output = [_]ScalarType{0} ** (image_width * image_height);
            const destination = zpp.Out(ScalarType, &output, image_width, region);

            const gen_kernel = struct {
                fn process(ctx: anytype, x: anytype, y: anytype) OutputType {
                    _ = ctx;
                    // Generate x + y * 10
                    return th.vectorCast(OutputType, x) + th.vectorCast(OutputType, y) * th.splatWithCast(OutputType, 10);
                }
            };

            const generator = zpp.Generate(CoordType, region, .{}, gen_kernel.process);
            zpp.Process(generator, destination);

            const expected_data: [image_width * image_height]ScalarType = .{
                0, 0, 0,  0,  0,  0,  0, 0,
                0, 0, 0,  0,  0,  0,  0, 0,
                0, 0, 0,  0,  0,  0,  0, 0,
                0, 0, 32, 33, 34, 35, 0, 0,
                0, 0, 42, 43, 44, 45, 0, 0,
                0, 0, 0,  0,  0,  0,  0, 0,
            };
            try std.testing.expectEqual(expected_data, output);
        }
    }
}

// MARK: Loop: width=1 processes single pixel correctly
test "Loop: width=1 processes single pixel correctly" {
    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        const region: zpp.Region = .{ .x = 0, .y = 0, .width = 1, .height = 1 };

        var input_data: [1]ScalarType = .{42.0};
        var output_data: [1]ScalarType = .{0};

        const source = zpp.In(ScalarType, &input_data, region.width, region);
        const destination = zpp.Out(ScalarType, &output_data, region.width, region);

        const double_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                return in.get() * th.splatWithCast(LoopType, 2);
            }
        };

        const result = zpp.Loop(LoopType, .{}, source, .{}, double_kernel.process);
        zpp.Process(result, destination);

        try std.testing.expectEqual(@as(ScalarType, 84.0), output_data[0]);
    }
}

// MARK: Loop: Process with Stat call the correct number of time the kernel
test "Loop: Process with Stat call the correct number of time the kernel" {
    // Width=5 with vec_len=4: should process 1 full batch + 1 scalar remainder
    // Stats destination does NOT support overlapping writes, so we get exact scalar remainder
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 2 };

    var input_data: [10]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    const source = zpp.In(f32, &input_data, region.width, region);

    const id_kernel = struct {
        const Context = struct {};
        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    // Count how many batch and scalar calls we get
    const counting_kernel = struct {
        const Context = struct {
            batch_calls: u32 = 0,
            scalar_calls: u32 = 0,
            sum: f32 = 0,
        };

        fn accumulate(ctx: *Context, values: f32x4) void {
            // Determine if this is a scalar call (lanes 1-3 are zero) or batch
            // writeScalar zeroes out non-lane-0 values
            if (values[1] == 0 and values[2] == 0 and values[3] == 0) {
                ctx.scalar_calls += 1;
            } else {
                ctx.batch_calls += 1;
            }
            ctx.sum += @reduce(.Add, values);
        }
    };

    var stats_ctx = counting_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, id_kernel.Context{}, id_kernel.process);
    const stats_dest = zpp.Stats(f32x4, &stats_ctx, region, counting_kernel.accumulate);
    zpp.Process(loop_result, stats_dest);

    // 2 rows * (1 batch of 4 + 1 scalar remainder) = 2 batch calls + 2 scalar calls
    try std.testing.expectEqual(@as(u32, 2), stats_ctx.batch_calls);
    try std.testing.expectEqual(@as(u32, 2), stats_ctx.scalar_calls);
    // Sum: 1+2+3+4+5 + 6+7+8+9+10 = 55
    try std.testing.expectEqual(@as(f32, 55.0), stats_ctx.sum);
}

// MARK: Loop: Process with Loop call the correct number of time the kernel
test "Loop: Process with Loop call the correct number of time the kernel" {
    // Width=5 with vec_len=4: should process 1 full batch + 1 scalar remainder
    // Loop destination does support overlapping writes, so we don't compute in scalar the remainder
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 2 };

    var input_data: [10]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    const source = zpp.In(f32, &input_data, region.width, region);
    var output_data = [_]f32{0} ** (10);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Count how many batch and scalar calls we get
    const identity_kernel = struct {
        const Context = struct {
            batch_calls: u32 = 0,
            scalar_calls: u32 = 0,
        };

        fn process(ctx: *Context, in: anytype) f32x4 {
            // Determine if this is a scalar call (lanes 1-3 are zero) or batch
            // writeScalar zeroes out non-lane-0 values
            const values = in.get();
            if (values[1] == 0 and values[2] == 0 and values[3] == 0) {
                ctx.scalar_calls += 1;
            } else {
                ctx.batch_calls += 1;
            }
            return values;
        }
    };

    var context = identity_kernel.Context{};
    const loop_result = zpp.Loop(f32x4, .{}, source, &context, identity_kernel.process);
    zpp.Process(loop_result, destination);

    try std.testing.expectEqual(@as(u32, 4), context.batch_calls);
    try std.testing.expectEqual(@as(u32, 0), context.scalar_calls);

    const expected_data = [_]f32{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Loop: Process with Loop process pixel two time if trying inplace computation
test "Loop: Process with Loop process pixel two time if trying inplace computation" {
    // it's a current limitation that when performing btach overlaping we process 2 time some of the pixel
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 5, .height = 2 };

    var input_data: [10]f32 = undefined;
    th.fillRamp(f32, &input_data, 1, 1);
    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &input_data, region.width, region);

    // Count how many batch and scalar calls we get
    const identity_kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            // Determine if this is a scalar call (lanes 1-3 are zero) or batch
            // writeScalar zeroes out non-lane-0 values
            return in.get() * th.splatWithCast(f32x4, 2);
        }
    };

    const loop_result = zpp.Loop(f32x4, .{}, source, .{}, identity_kernel.process);
    zpp.Process(loop_result, destination);

    const expected_data = [_]f32{
        2,  8,  12, 16, 10,
        12, 28, 32, 36, 20,
    };
    try std.testing.expectEqual(expected_data, input_data);
}

// MARK: Loop: width=2 processes sub-vector region correctly
test "Loop: width=2 processes sub-vector region correctly" {
    inline for (AllTypes) |LoopType| {
        const ScalarType = @typeInfo(LoopType).vector.child;

        const region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 3 };

        var input_data: [6]ScalarType = undefined;
        th.fillRamp(ScalarType, &input_data, 1, 1);
        var output_data = [_]ScalarType{0} ** 6;

        const source = zpp.In(ScalarType, &input_data, region.width, region);
        const destination = zpp.Out(ScalarType, &output_data, region.width, region);

        const triple_kernel = struct {
            fn process(ctx: anytype, in: anytype) LoopType {
                _ = ctx;
                return in.get() * th.splatWithCast(LoopType, 3);
            }
        };

        const result = zpp.Loop(LoopType, .{}, source, .{}, triple_kernel.process);
        zpp.Process(result, destination);

        // output = input * 3: [1,2,3,4,5,6] * 3 = [3,6,9,12,15,18]
        const expected_data: [6]ScalarType = .{
            3,  6,  9,
            12, 15, 18,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

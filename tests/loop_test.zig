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

test "basic Processing loop: Use coordinates" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_data: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    const processing_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype, x: u16x4, y: u16x4) f32x4 {
            _ = ctx;
            // Return input + x + y
            const x_f: f32x4 = .{
                @floatFromInt(x[0]),
                @floatFromInt(x[1]),
                @floatFromInt(x[2]),
                @floatFromInt(x[3]),
            };
            const y_f: f32x4 = .{
                @floatFromInt(y[0]),
                @floatFromInt(y[1]),
                @floatFromInt(y[2]),
                @floatFromInt(y[3]),
            };
            return in.get() + x_f + y_f;
        }
    };

    const ctx = processing_kernel.Context{};
    const result = zpp.Loop(f32x4, .{ .need_coordinates = u16x4 }, source, ctx, processing_kernel.process);
    zpp.Process(result, destination);

    // input[i] + x + y = input[i] + i + 0
    try std.testing.expectEqual(@as(f32, 10.0), output_data[0]); // 10 + 0 + 0
    try std.testing.expectEqual(@as(f32, 21.0), output_data[1]); // 20 + 1 + 0
    try std.testing.expectEqual(@as(f32, 32.0), output_data[2]); // 30 + 2 + 0
    try std.testing.expectEqual(@as(f32, 43.0), output_data[3]); // 40 + 3 + 0
}

test "basic Processing loop: Use Margins" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Input: [1, 2, 3, 4]
    var input_data: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    const processing_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            // Horizontal blur: left + center + right
            return in.getAt(-1, 0) + in.getAt(0, 0) + in.getAt(1, 0);
        }
    };

    const ctx = processing_kernel.Context{};
    const result = zpp.Loop(f32x4, .{ .margin = .{ .left = 1, .right = 1 } }, source, ctx, processing_kernel.process);
    zpp.Process(result, destination);

    // With RepeatEdgePadding (default):
    // Position 0: in(-1,0)=1 (edge repeat) + in(0,0)=1 + in(1,0)=2 = 4
    // Position 1: in(-1,0)=1 + in(0,0)=2 + in(1,0)=3 = 6
    // Position 2: in(-1,0)=2 + in(0,0)=3 + in(1,0)=4 = 9
    // Position 3: in(-1,0)=3 + in(0,0)=4 + in(1,0)=4 (edge repeat) = 11
    try std.testing.expectEqual(@as(f32, 4.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 11.0), output_data[3]);
}

test "Expression tree" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    // 2x4 input:
    // Row 0: [1, 2, 3, 4]
    // Row 1: [5, 6, 7, 8]
    var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output_data: [8]f32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // First kernel: horizontal blur (left + center + right)
    const kernel1 = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.getAt(-1, 0) + in.getAt(0, 0) + in.getAt(1, 0);
        }
    };

    // Second kernel: vertical blur (top + center + bottom)
    const kernel2 = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.getAt(0, -1) + in.getAt(0, 0) + in.getAt(0, 1);
        }
    };

    const ctx1 = kernel1.Context{};
    const ctx2 = kernel2.Context{};

    const result1 = zpp.Loop(f32x4, .{ .margin = .{ .left = 1, .right = 1 } }, source, ctx1, kernel1.process);
    const result2 = zpp.Loop(f32x4, .{ .margin = .{ .top = 1, .bottom = 1 } }, result1, ctx2, kernel2.process);
    zpp.Process(result2, destination);

    // The expression tree chains kernel1 -> kernel2
    // For each output pixel, kernel2 reads from kernel1's virtual output

    // Verify the chained processing produces expected results
    // This is a box blur (3x3 separable = horizontal then vertical)
    try std.testing.expect(output_data[0] > 0);
    try std.testing.expect(output_data[4] > 0);
}

test "multi-channel RGB generator" {
    const vec_len = 4;
    const VecF32 = @Vector(vec_len, f32);
    const VecU8 = @Vector(vec_len, u8);

    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Output buffer: 4 pixels * 3 channels = 12 bytes
    var output: [12]u8 = .{0} ** 12;

    const destination = zpp.InterleavedOut(u8, 3, &output, region.width, region);

    // Kernel that produces RGB based on coordinates
    // User handles normalization: converts f32 [0,1] to u8 [0,255]
    const rgb_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, x: VecF32, y: VecF32) [3]VecU8 {
            _ = ctx;
            _ = y;
            // R = x/4, G = 0.5, B = 1.0 (scaled to [0,1], then converted to u8)
            const splat_4: VecF32 = @splat(4.0);
            const splat_255: VecF32 = @splat(255.0);
            return .{
                @intFromFloat(x / splat_4 * splat_255), // R: 0, 63, 127, 191
                @intFromFloat(@as(VecF32, @splat(0.5)) * splat_255), // G: 127
                @intFromFloat(@as(VecF32, @splat(1.0)) * splat_255), // B: 255
            };
        }
    };

    const ctx = rgb_kernel.Context{};
    const generator = zpp.Generate(VecF32, region, ctx, rgb_kernel.process);
    zpp.Process(generator, destination);

    // Check RGB interleaving: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3
    // R values: 0, 63, 127, 191 (approximately 0, 0.25, 0.5, 0.75 * 255)
    // G values: 127 (0.5 * 255)
    // B values: 255 (1.0 * 255)
    try std.testing.expectEqual(@as(u8, 0), output[0]); // R0
    try std.testing.expectEqual(@as(u8, 127), output[1]); // G0
    try std.testing.expectEqual(@as(u8, 255), output[2]); // B0
    try std.testing.expectEqual(@as(u8, 63), output[3]); // R1
    try std.testing.expectEqual(@as(u8, 127), output[4]); // G1
    try std.testing.expectEqual(@as(u8, 255), output[5]); // B1
}

test "Multiple kernels in expression tree" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 4 };

    // Create input data
    var input_data: [32]f32 = undefined;
    for (0..32) |i| {
        input_data[i] = @floatFromInt(i);
    }
    var output_data: [32]f32 = .{0} ** 32;

    const source = zpp.In(f32, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    // Chain 3 kernels: scale -> offset -> abs
    const scale_kernel = struct {
        const Context = struct { scale: f32 };
        fn process(ctx: Context, in: anytype) f32x4 {
            return in.get() * @as(f32x4, @splat(ctx.scale));
        }
    };

    const offset_kernel = struct {
        const Context = struct { offset: f32 };
        fn process(ctx: Context, in: anytype) f32x4 {
            return in.get() + @as(f32x4, @splat(ctx.offset));
        }
    };

    const step1 = zpp.Loop(f32x4, .{}, source, scale_kernel.Context{ .scale = 2.0 }, scale_kernel.process);
    const step2 = zpp.Loop(f32x4, .{}, step1, offset_kernel.Context{ .offset = -10.0 }, offset_kernel.process);
    zpp.Process(step2, destination);

    // Verify: output = input * 2 - 10
    try std.testing.expectEqual(@as(f32, -10.0), output_data[0]); // 0*2-10
    try std.testing.expectEqual(@as(f32, -8.0), output_data[1]); // 1*2-10
    try std.testing.expectEqual(@as(f32, 10.0), output_data[10]); // 10*2-10
}

// test "Generator with coordinates" {
//     const region: zpp.Region = .{ .x = 10, .y = 20, .width = 4, .height = 2 };
//     var output: [8]f32 = .{0} ** 8;

//     const destination = zpp.Out(f32, &output, region.width, region);

//     // Generate x + y
//     const gen_kernel = struct {
//         const Context = struct {};
//         fn process(ctx: Context, x: f32x4, y: f32x4) f32x4 {
//             _ = ctx;
//             return x + y;
//         }
//     };

//     const ctx = gen_kernel.Context{};
//     const generator = zpp.Generate(f32x4, region, ctx, gen_kernel.process);
//     zpp.Process(generator, destination);

//     // Row 0: x=10,11,12,13 + y=20 = 30,31,32,33
//     try std.testing.expectEqual(@as(f32, 30.0), output[0]);
//     try std.testing.expectEqual(@as(f32, 31.0), output[1]);
//     try std.testing.expectEqual(@as(f32, 32.0), output[2]);
//     try std.testing.expectEqual(@as(f32, 33.0), output[3]);
//     // Row 1: x=10,11,12,13 + y=21 = 31,32,33,34
//     try std.testing.expectEqual(@as(f32, 31.0), output[4]);
//     try std.testing.expectEqual(@as(f32, 32.0), output[5]);
// }

// test "Non-origin region processing" {
//     const region: zpp.Region = .{ .x = 5, .y = 10, .width = 4, .height = 1 };
//     var input_data: [4]f32 = .{ 100.0, 200.0, 300.0, 400.0 };
//     var output_data: [4]f32 = .{0} ** 4;

//     const source = zpp.In(f32, &input_data, region.width, region);
//     const destination = zpp.Out(f32, &output_data, region.width, region);

//     // Identity kernel
//     const id_kernel = struct {
//         const Context = struct {};
//         fn process(ctx: Context, in: anytype) f32x4 {
//             _ = ctx;
//             return in.get();
//         }
//     };

//     const ctx = id_kernel.Context{};
//     const result = zpp.Loop(f32x4, .{}, source, ctx, id_kernel.process);
//     zpp.Process(result, destination);

//     // Should copy input to output despite non-zero origin
//     try std.testing.expectEqual(@as(f32, 100.0), output_data[0]);
//     try std.testing.expectEqual(@as(f32, 200.0), output_data[1]);
//     try std.testing.expectEqual(@as(f32, 300.0), output_data[2]);
//     try std.testing.expectEqual(@as(f32, 400.0), output_data[3]);
// }

test "basic Processing loop: Use Margins with ZeroPadding" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Input: [1, 2, 3, 4]
    var input_data: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source = zpp.InWithPadding(f32, zpp.ZeroPadding, &input_data, region.width, region);
    const destination = zpp.Out(f32, &output_data, region.width, region);

    const processing_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            // Horizontal blur: left + center + right
            return in.getAt(-1, 0) + in.getAt(0, 0) + in.getAt(1, 0);
        }
    };

    const ctx = processing_kernel.Context{};
    const result = zpp.Loop(f32x4, .{ .margin = .{ .left = 1, .right = 1 } }, source, ctx, processing_kernel.process);
    zpp.Process(result, destination);

    // With ZeroPadding:
    // Position 0: in(-1,0)=0 + in(0,0)=1 + in(1,0)=2 = 3
    // Position 1: in(-1,0)=1 + in(0,0)=2 + in(1,0)=3 = 6
    // Position 2: in(-1,0)=2 + in(0,0)=3 + in(1,0)=4 = 9
    // Position 3: in(-1,0)=3 + in(0,0)=4 + in(1,0)=0 = 7
    try std.testing.expectEqual(@as(f32, 3.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 7.0), output_data[3]);
}

test "multi-channel output with non-power-of-2 width" {
    // Test remainder handling when width is not divisible by vec_len
    const vec_len = 4;
    const VecF32 = @Vector(vec_len, f32);
    const VecU8 = @Vector(vec_len, u8);

    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 6, .height = 1 }; // 6 is not divisible by 4

    // Output buffer: 6 pixels * 3 channels = 18 bytes
    var output: [18]u8 = .{0} ** 18;

    const destination = zpp.InterleavedOut(u8, 3, &output, region.width, region);

    // Constant color kernel - user handles normalization
    const const_kernel = struct {
        const Context = struct {};

        fn process(ctx: Context, x: VecF32, y: VecF32) [3]VecU8 {
            _ = ctx;
            _ = x;
            _ = y;
            return .{
                @as(VecU8, @splat(255)), // R
                @as(VecU8, @splat(127)), // G
                @as(VecU8, @splat(0)), // B
            };
        }
    };

    const ctx = const_kernel.Context{};
    const generator = zpp.Generate(VecF32, region, ctx, const_kernel.process);
    zpp.Process(generator, destination);

    // All pixels should have the same color
    for (0..6) |i| {
        try std.testing.expectEqual(@as(u8, 255), output[i * 3 + 0]); // R
        try std.testing.expectEqual(@as(u8, 127), output[i * 3 + 1]); // G
        try std.testing.expectEqual(@as(u8, 0), output[i * 3 + 2]); // B
    }
}

// TODO: add test to verify we only compute what needed (no extra evalAt calls)
// TODO: add test that the kernel recieve the correct VecT type in InputAccessor
// TODO: test to check that we only process within the defined region boundaries
// TODO: add test that we perform correct conversion when Loop have different VecT than source/dest
// TODO: add test that we perform correct conversion when several Loop have different VecT between them

// TODO: add test to check that the coordinate type match the type passed in need_coordinates option

// TODO: does it make sense that generate requires region, while loop infers it from destination ?

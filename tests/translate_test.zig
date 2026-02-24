//! Tests for translate.zig - Integer pixel shift without interpolation

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

const f32x4 = @Vector(4, f32);
const u16x4 = @Vector(4, u16);
const u8x4 = @Vector(4, u8);

const AllTypes = [_]type{ f32x4, u16x4, u8x4 };

// MARK: Translate: identity shift copies data unchanged
test "Translate: identity shift copies data unchanged" {
    const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    inline for (AllTypes) |VecType| {
        const ScalarType = @typeInfo(VecType).vector.child;

        var source_data: [8]ScalarType = undefined;
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.makeSource(ScalarType, &source_data, region.width, region);
        const translated = zpp.translate(source, 0, 0);

        var output_data = [_]ScalarType{0} ** 8;
        const dest = zpp.makeDest(ScalarType, &output_data, region.width, region);

        zpp.process(translated, dest);

        try std.testing.expectEqual(source_data, output_data);
    }
}

// MARK: Translate: shifts data to new position
test "Translate: shifts data to new position" {
    const src_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    inline for (AllTypes) |VecType| {
        const ScalarType = @typeInfo(VecType).vector.child;

        var source_data: [8]ScalarType = undefined;
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.makeSource(ScalarType, &source_data, src_region.width, src_region);
        const translated = zpp.translate(source, 2, 1);

        const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 6, .height = 3 };
        var output_data = [_]ScalarType{0} ** (out_region.area());
        const dest = zpp.makeDest(ScalarType, &output_data, out_region.width, out_region);

        zpp.process(translated, dest);

        const expected_data = [_]ScalarType{
            1, 1, 1, 2, 3, 4,
            1, 1, 1, 2, 3, 4,
            5, 5, 5, 6, 7, 8,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Translate: through Loop with identity kernel
test "Translate: through Loop with identity kernel" {
    const src_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    inline for (AllTypes) |VecType| {
        const ScalarType = @typeInfo(VecType).vector.child;

        var source_data: [8]ScalarType = undefined;
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.makeSource(ScalarType, &source_data, src_region.width, src_region);
        const translated = zpp.translate(source, 2, 1);

        const identity_kernel = struct {
            fn process(ctx: anytype, in: anytype) VecType {
                _ = ctx;
                return in.get();
            }
        };

        const result = zpp.loop(VecType, .{}, translated, .{}, identity_kernel.process);

        const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 6, .height = 3 };
        var output_data = [_]ScalarType{0} ** (out_region.area());
        const dest = zpp.makeDest(ScalarType, &output_data, out_region.width, out_region);

        zpp.process(result, dest);

        const expected_data = [_]ScalarType{
            1, 1, 1, 2, 3, 4,
            1, 1, 1, 2, 3, 4,
            5, 5, 5, 6, 7, 8,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Translate: of LoopResult shifts computation
test "Translate: of LoopResult shifts computation" {
    // Source at origin, Loop doubles values, then translate shifts the result
    const src_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    inline for (AllTypes) |VecType| {
        const ScalarType = @typeInfo(VecType).vector.child;

        var source_data: [8]ScalarType = undefined;
        th.fillRamp(ScalarType, &source_data, 1, 1);
        const source = zpp.makeSource(ScalarType, &source_data, src_region.width, src_region);
        const double_kernel = struct {
            fn process(ctx: anytype, in: anytype) VecType {
                _ = ctx;
                return in.get() * @as(VecType, @splat(2));
            }
        };

        const doubled = zpp.loop(VecType, .{}, source, .{}, double_kernel.process);
        const translated = zpp.translate(doubled, 2, 1);

        const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 6, .height = 3 };
        const out_stride = 6;
        var output_data = [_]ScalarType{0} ** (out_stride * 3);
        const dest = zpp.makeDest(ScalarType, &output_data, out_stride, out_region);

        zpp.process(translated, dest);

        const expected_data = [_]ScalarType{
            2,  2,  2,  4,  6,  8,
            2,  2,  2,  4,  6,  8,
            10, 10, 10, 12, 14, 16,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Translate: with Zip combines shifted sources
test "Translate: with Zip combines shifted sources" {
    const src_region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 1 };

    var input_a: [8]f32 = undefined;
    th.fillRamp(f32, &input_a, 1, 1);
    var input_b: [8]f32 = undefined;
    th.fillRamp(f32, &input_b, 10, 10);

    const source_a = zpp.makeSource(f32, &input_a, src_region.width, src_region);
    const source_b = zpp.makeSource(f32, &input_b, src_region.width, src_region);

    const translated_b = zpp.translate(source_b, 4, 0);

    const zipped = zpp.zip(.{ source_a, translated_b });

    const add_kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            const a, const b = in.get();
            return a + b;
        }
    };

    const result = zpp.loop(f32x4, .{}, zipped, .{}, add_kernel.process);

    const out_region: zpp.Region = .{ .x = 4, .y = 0, .width = 4, .height = 1 };
    var output_data = [_]f32{0} ** 8;
    const dest = zpp.makeDest(f32, &output_data, 8, out_region);

    zpp.process(result, dest);

    const expected_data = [_]f32{
        0, 0, 0, 0, 15, 26, 37, 48,
    };
    try std.testing.expectEqual(expected_data, output_data);
}

// MARK: Translate: preserves padding at shifted boundaries
test "Translate: preserves padding at shifted boundaries" {
    const src_region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var source_data: [4]f32 = .{ 10, 20, 30, 40 };
    const source = zpp.makePaddedSource(f32, zpp.ZeroPadding, &source_data, src_region.width, src_region);

    const translated = zpp.translate(source, 2, 0);

    const identity_kernel = struct {
        fn process(ctx: anytype, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    const result = zpp.loop(f32x4, .{}, translated, .{}, identity_kernel.process);

    const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 8, .height = 1 };
    var output_data = [_]f32{0} ** 8;
    const dest = zpp.makeDest(f32, &output_data, out_region.width, out_region);

    zpp.process(result, dest);

    const expected: [8]f32 = .{ 0, 0, 10, 20, 30, 40, 0, 0 };
    try std.testing.expectEqual(expected, output_data);
}

// MARK: Translate: negative shift
test "Translate: negative shift" {
    const src_region: zpp.Region = .{ .x = 4, .y = 3, .width = 4, .height = 2 };

    inline for (AllTypes) |VecType| {
        const ScalarType = @typeInfo(VecType).vector.child;

        const stride = 8;
        var source_data = [_]ScalarType{0} ** (stride * 5);
        th.fillRamp(ScalarType, &source_data, 1, 1);

        const source = zpp.makeSource(ScalarType, &source_data, stride, src_region);

        // Translate by (-2, -1) → region becomes (2, 2, 4, 2)
        const translated = zpp.translate(source, -2, -1);

        const out_region: zpp.Region = .{ .x = 0, .y = 0, .width = 6, .height = 4 };
        var output_data = [_]ScalarType{0} ** (out_region.area());
        const dest = zpp.makeDest(ScalarType, &output_data, out_region.width, out_region);

        zpp.process(translated, dest);

        const expected_data = [_]ScalarType{
            29, 29, 29, 30, 31, 32,
            29, 29, 29, 30, 31, 32,
            29, 29, 29, 30, 31, 32,
            37, 37, 37, 38, 39, 40,
        };
        try std.testing.expectEqual(expected_data, output_data);
    }
}

// MARK: Translate: region is correctly shifted
test "Translate: region is correctly shifted" {
    const src_region: zpp.Region = .{ .x = 5, .y = 10, .width = 20, .height = 15 };

    var source_data: [1]f32 = .{0};
    const source = zpp.makeSource(f32, &source_data, 1, src_region);

    const translated = zpp.translate(source, 3, -7);
    const region = translated.getRegion();

    try std.testing.expectEqual(@as(i32, 8), region.x);
    try std.testing.expectEqual(@as(i32, 3), region.y);
    try std.testing.expectEqual(@as(u32, 20), region.width);
    try std.testing.expectEqual(@as(u32, 15), region.height);
}

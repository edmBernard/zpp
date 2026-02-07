//! Tests for cache.zig - Row caching for expression trees

const std = @import("std");
const zpp = @import("zpp");
const th = @import("test_helpers.zig");

/// Common SIMD vector types
const f32x4 = @Vector(4, f32);

// TODO: this test should work
// // MARK: Cache: Cache is use when kernel is reuse in different branch
// test "Cache: Cache is use when kernel is reuse in different branch" {
//     // we transform the input and write to both a normal destination and a Stats accumulator
//     // the stat is computed on the transformed value
//     const region: zpp.Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

//     var input_data: [8]f32 = undefined;
//     th.fillRamp(f32, &input_data, 1, 1);

//     var output_data = [_]f32{0} ** 8;

//     const source = zpp.In(f32, &input_data, region.width, region);
//     const destination = zpp.Out(f32, &output_data, region.width, region);

//     const stat_kernel = struct {
//         const Context = struct {
//             sum: f32 = 0,
//         };

//         fn accumulate(ctx: *Context, values: f32x4) void {
//             ctx.sum += @reduce(.Add, values);
//         }
//     };
//     var stats_ctx = stat_kernel.Context{};

//     const stats_dest = zpp.Stats(f32x4, &stats_ctx, region, stat_kernel.accumulate);

//     const kernel = struct {
//         const Context = struct {
//             calls: u32 = 0,
//         };

//         fn process(ctx: *Context, in: anytype) f32x4 {
//             ctx.calls += 1;
//             return in.get() * th.splatWithCast(f32x4, 10);
//         }
//     };

//     var ctx = kernel.Context{};
//     const result = zpp.CachedLoop(f32x4, .{}, 2, source, &ctx, kernel.process);
//     const zipped_in = zpp.Zip(.{ result, result });
//     const zipped_dest = zpp.ZipOut(.{ destination, stats_dest });
//     zpp.Process(zipped_in, zipped_dest);

//     const expected_data = [_]f32{
//         10, 20, 30, 40,
//         50, 60, 70, 80,
//     };
//     try std.testing.expectEqual(expected_data, output_data);

//     try std.testing.expectEqual(360, stats_ctx.sum);

//     try std.testing.expectEqual(2, ctx.calls);
// }

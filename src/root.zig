//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

test "basic generator" {

    var region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    var output = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer output.deinit();
    for (region.Area()) |i| {
        try output.appendAssumeCapacity(0);
    }

    const f32_v: type = @Vector(4, f32);

    const source = zpp.In(input.items, region.Width(), region);
    const destination = zpp.Out(output.items, region.Width(), region);

    const processing_kernel = struct {
        context = .{
            .scale : f32_v = f32_v{ 1.0, 1.0, 1.0, 1.0 },
            .sin_time : f32_v = f32_v{ 0.0, 0.0, 0.0, 0.0 },
        },

        pub inline fn process(ctx: ProcessingKernel, x: anytype, y: anytype) f32_v {
            const xs = x / ctx.scale + ctx.sin_time;
            const ys = y / ctx.scale + ctx.sin_time;
            return xs + ys;
        }
    };

    const result = zpp.Generate(f32_v, .{}, processing_kernel.context, processing_kernel.process);
    zpp.Process(result, destination);

    try std.testing.expect(output.items[0] == 1.0 + 1.0);
    try std.testing.expect(output.items[1] == 2.0 + 1.0);
    try std.testing.expect(output.items[2] == 1.0 + 2.0);
    try std.testing.expect(output.items[3] == 2.0 + 2.0);

  }


test "basic Processing loop: Identity" {

    var region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    var input = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer input.deinit();
    for (region.Area()) |i| {
        try input.appendAssumeCapacity(@as(f32, i + 1));
    }
    var output = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer output.deinit();
    for (region.Area()) |i| {
        try output.appendAssumeCapacity(0);
    }

    const f32_v: type = @Vector(4, f32);

    const source = zpp.In(input.items, region.Width(), region);
    const destination = zpp.Out(output.items, region.Width(), region);

    const processing_kernel = struct {
        context = .{
            .scale : f32_v = f32_v{ 1.0, 1.0, 1.0, 1.0 },
            .sin_time : f32_v = f32_v{ 0.0, 0.0, 0.0, 0.0 },
        },

        pub inline fn process(ctx: ProcessingKernel, in: anytype) f32_v {
            return in;
        }
    };

    const result = zpp.Loop(f32_v, .{}, source, processing_kernel.context, processing_kernel.process);
    zpp.Process(result, destination);

    try std.testing.expect(output.items[0] == 1.0 + 1.0);
    try std.testing.expect(output.items[1] == 2.0 + 1.0);
    try std.testing.expect(output.items[2] == 1.0 + 2.0);
    try std.testing.expect(output.items[3] == 2.0 + 2.0);

  }


test "basic Processing loop: Use coordinates" {

    var region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    var input = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer input.deinit();
    for (region.Area()) |i| {
        try input.appendAssumeCapacity(@as(f32, i + 1));
    }
    var output = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer output.deinit();
    for (region.Area()) |i| {
        try output.appendAssumeCapacity(0);
    }

    const f32_v: type = @Vector(4, f32);

    const source = zpp.In(input.items, region.Width(), region);
    const destination = zpp.Out(output.items, region.Width(), region);

    const processing_kernel = struct {
        context = .{
            .scale : f32_v = f32_v{ 1.0, 1.0, 1.0, 1.0 },
            .sin_time : f32_v = f32_v{ 0.0, 0.0, 0.0, 0.0 },
        },

        pub inline fn process(ctx: ProcessingKernel, in: anytype, x: u16_v, y: u16_v) f32_v {
            return in() + @cast(f32_v, x) + @cast(f32_v, y);
        }
    };

    const result = zpp.Loop(f32_v, .{.need_coordinates = u16_v}, source, processing_kernel.context, processing_kernel.process);
    zpp.Process(result, destination);

    try std.testing.expect(output.items[0] == 1.0 + 1.0);
    try std.testing.expect(output.items[1] == 2.0 + 1.0);
    try std.testing.expect(output.items[2] == 1.0 + 2.0);
    try std.testing.expect(output.items[3] == 2.0 + 2.0);

  }

  test "basic Processing loop: Use Margins" {

    var region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    var input = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer input.deinit();
    for (region.Area()) |i| {
        try input.appendAssumeCapacity(@as(f32, i + 1));
    }
    var output = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer output.deinit();
    for (region.Area()) |i| {
        try output.appendAssumeCapacity(0);
    }

    const f32_v: type = @Vector(4, f32);

    const source = zpp.In(input.items, region.Width(), region);
    const destination = zpp.Out(output.items, region.Width(), region);

    const processing_kernel = struct {
        context = .{
            .scale : f32_v = f32_v{ 1.0, 1.0, 1.0, 1.0 },
            .sin_time : f32_v = f32_v{ 0.0, 0.0, 0.0, 0.0 },
        },

        pub inline fn process(ctx: ProcessingKernel, in: anytype) f32_v {
            return in(-1,0) + in(0,0) + in(1,0);
        }
    };

    const result = zpp.Loop(f32_v, .{.margin = .{.top=1, .bottom=1, .left=1, .right=1}}, source, processing_kernel.context, processing_kernel.process);
    zpp.Process(result, destination);

    try std.testing.expect(output.items[0] == 1.0 + 1.0);
    try std.testing.expect(output.items[1] == 2.0 + 1.0);
    try std.testing.expect(output.items[2] == 1.0 + 2.0);
    try std.testing.expect(output.items[3] == 2.0 + 2.0);

  }

  test "Expression tree" {

    var region: zpp.Region = .{ .x = 0, .y = 0, .width = 2, .height = 2 };
    var input = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer input.deinit();
    for (region.Area()) |i| {
        try input.appendAssumeCapacity(@as(f32, i + 1));
    }
    var output = std.ArrayList(f32).initWithCapacity(std.testing.allocator, region.Area());
    defer output.deinit();
    for (region.Area()) |i| {
        try output.appendAssumeCapacity(0);
    }

    const f32_v: type = @Vector(4, f32);

    const source = zpp.In(input.items, region.Width(), region);
    const destination = zpp.Out(output.items, region.Width(), region);

    const processing_kernel_1 = struct {
        context = .{
            .scale : f32_v = f32_v{ 1.0, 1.0, 1.0, 1.0 },
            .sin_time : f32_v = f32_v{ 0.0, 0.0, 0.0, 0.0 },
        },

        pub inline fn process(ctx: ProcessingKernel, in: anytype) f32_v {
            return in(-1,0) + in(0,0) + in(1,0);
        }
    };
    const processing_kernel_2 = struct {
        context = .{
            .scale : f32_v = f32_v{ 1.0, 1.0, 1.0, 1.0 },
            .sin_time : f32_v = f32_v{ 0.0, 0.0, 0.0, 0.0 },
        },

        pub inline fn process(ctx: ProcessingKernel, in: anytype) f32_v {
            return in(0,-1) + in(0,0) + in(0,1);
        }
    };

    const result1 = zpp.Loop(f32_v, .{.margin = .{.left=1, .right=1}}, source, processing_kernel_1.context, processing_kernel_1.process);
    const result2 = zpp.Loop(f32_v, .{.margin = .{.top=1, .bottom=1}}, result1, processing_kernel_2.context, processing_kernel_2.process);
    zpp.Process(result2, destination);

    try std.testing.expect(output.items[0] == 1.0 + 1.0);
    try std.testing.expect(output.items[1] == 2.0 + 1.0);
    try std.testing.expect(output.items[2] == 1.0 + 2.0);
    try std.testing.expect(output.items[3] == 2.0 + 2.0);

  }
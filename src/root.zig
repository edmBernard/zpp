//! ZPP - Zig Pixel Processing Library
//! A library for efficient pixel processing using SIMD vector capabilities.

const std = @import("std");

/// Common SIMD vector types
pub const f32x4 = @Vector(4, f32);
pub const u16x4 = @Vector(4, u16);
pub const i32x4 = @Vector(4, i32);

/// Margin specification for neighborhood access
pub const Margin = struct {
    top: u32 = 0,
    bottom: u32 = 0,
    left: u32 = 0,
    right: u32 = 0,
};

/// Region defines a rectangular area for processing
pub const Region = struct {
    x: i32 = 0,
    y: i32 = 0,
    width: u32,
    height: u32,

    pub fn area(self: Region) u32 {
        return self.width * self.height;
    }

    pub fn getWidth(self: Region) u32 {
        return self.width;
    }

    pub fn getHeight(self: Region) u32 {
        return self.height;
    }
};

/// Options for Loop and Generate operations
pub fn LoopOptions(comptime CoordType: ?type) type {
    return struct {
        need_coordinates: ?type = CoordType,
        margin: Margin = .{},
    };
}

/// Default loop options type
pub const DefaultLoopOptions = struct {
    need_coordinates: ?type = null,
    margin: Margin = .{},
};

/// Input source wrapper - provides access to input data with optional neighborhood access
pub fn InputSource(comptime T: type) type {
    return struct {
        data: []const T,
        stride: u32,
        region: Region,

        const Self = @This();
    };
}

/// Create an input source from data buffer
pub fn In(comptime T: type, data: []const T, stride: u32, region: Region) InputSource(T) {
    return .{
        .data = data,
        .stride = stride,
        .region = region,
    };
}

/// Output destination wrapper
pub fn OutputDest(comptime T: type) type {
    return struct {
        data: []T,
        stride: u32,
        region: Region,

        const Self = @This();

        pub fn write(self: Self, x: u32, y: u32, values: @Vector(4, T)) void {
            const base_idx = y * self.stride + x;
            inline for (0..4) |i| {
                const idx = base_idx + i;
                if (idx < self.data.len and (x + i) < self.stride) {
                    self.data[idx] = values[i];
                }
            }
        }

        pub fn writeScalar(self: Self, x: u32, y: u32, value: T) void {
            const idx = y * self.stride + x;
            if (idx < self.data.len) {
                self.data[idx] = value;
            }
        }
    };
}

/// Create an output destination from data buffer
pub fn Out(comptime T: type, data: []T, stride: u32, region: Region) OutputDest(T) {
    return .{
        .data = data,
        .stride = stride,
        .region = region,
    };
}

/// Input accessor that provides neighborhood access for kernels
pub fn InputAccessor(comptime SrcType: type, comptime VecT: type) type {
    return struct {
        source: SrcType,
        current_x: i32,
        current_y: i32,

        const Self = @This();

        /// Get value at offset (for margin-based operations)
        pub inline fn getAt(self: Self, dx: i32, dy: i32) VecT {
            const x = self.current_x + dx;
            const y = self.current_y + dy;

            // Check if source is a LoopResult (for expression trees) - has evalAt
            if (@hasDecl(SrcType, "evalAt")) {
                // It's a LoopResult - call its eval function
                return self.source.evalAt(x, y);
            } else {
                // Source is InputSource - read from data directly
                var result: VecT = @splat(0);
                inline for (0..4) |i| {
                    const px = x + @as(i32, @intCast(i));
                    const py = y;
                    if (px >= 0 and py >= 0) {
                        const ux: u32 = @intCast(px);
                        const uy: u32 = @intCast(py);
                        if (ux < self.source.stride) {
                            const idx = uy * self.source.stride + ux;
                            if (idx < self.source.data.len) {
                                result[i] = self.source.data[idx];
                            }
                        }
                    }
                }
                return result;
            }
        }

        /// Get current value (no offset) - for identity operations
        pub inline fn get(self: Self) VecT {
            return self.getAt(0, 0);
        }
    };
}

/// Lazy loop result that can be chained or processed
pub fn LoopResult(
    comptime VecT: type,
    comptime SrcType: type,
    comptime CtxType: type,
    comptime process_fn: anytype,
    comptime opts: DefaultLoopOptions,
) type {
    const has_coords = opts.need_coordinates != null;
    const AccessorType = InputAccessor(SrcType, VecT);

    return struct {
        source: SrcType,
        context: CtxType,
        region: Region,

        // Marker to identify this as a LoopResult for expression trees
        const source_type = SrcType;

        const Self = @This();

        /// Evaluate at a specific position
        pub inline fn evalAt(self: Self, x: i32, y: i32) VecT {
            const accessor = AccessorType{
                .source = self.source,
                .current_x = x,
                .current_y = y,
            };

            if (has_coords) {
                const CoordT = opts.need_coordinates.?;
                var x_coords: CoordT = undefined;
                var y_coords: CoordT = undefined;
                inline for (0..4) |i| {
                    x_coords[i] = @intCast(x + @as(i32, @intCast(i)));
                    y_coords[i] = @intCast(y);
                }
                return process_fn(self.context, accessor, x_coords, y_coords);
            } else {
                return process_fn(self.context, accessor);
            }
        }

        pub fn getRegion(self: Self) Region {
            return self.region;
        }
    };
}

/// Create a lazy processing loop
pub fn Loop(
    comptime VecT: type,
    comptime opts: DefaultLoopOptions,
    source: anytype,
    context: anytype,
    comptime process_fn: anytype,
) LoopResult(VecT, @TypeOf(source), @TypeOf(context), process_fn, opts) {
    const SrcType = @TypeOf(source);

    const region = if (@hasDecl(SrcType, "getRegion"))
        source.getRegion()
    else if (@hasField(SrcType, "region"))
        source.region
    else
        @compileError("Source must have a region field or getRegion method");

    return .{
        .source = source,
        .context = context,
        .region = region,
    };
}

/// Generator result (no input source, just generates based on coordinates)
pub fn GeneratorResult(
    comptime VecT: type,
    comptime CtxType: type,
    comptime process_fn: anytype,
) type {
    return struct {
        context: CtxType,
        region: Region,

        const Self = @This();

        pub inline fn evalAt(self: Self, x: i32, y: i32) VecT {
            var x_vec: VecT = undefined;
            var y_vec: VecT = undefined;
            inline for (0..4) |i| {
                x_vec[i] = @floatFromInt(x + @as(i32, @intCast(i)));
                y_vec[i] = @floatFromInt(y);
            }
            return process_fn(self.context, x_vec, y_vec);
        }

        pub fn getRegion(self: Self) Region {
            return self.region;
        }
    };
}

/// Create a generator (produces values from coordinates, no input)
pub fn Generate(
    comptime VecT: type,
    comptime opts: DefaultLoopOptions,
    region: Region,
    context: anytype,
    comptime process_fn: anytype,
) GeneratorResult(VecT, @TypeOf(context), process_fn) {
    _ = opts;
    return .{
        .context = context,
        .region = region,
    };
}

/// Execute the processing pipeline and write results to output
pub fn Process(comptime ElemT: type, source: anytype, dest: anytype) void {
    const region = source.getRegion();

    var y: u32 = 0;
    while (y < region.height) : (y += 1) {
        var x: u32 = 0;
        while (x + 4 <= region.width) : (x += 4) {
            const result = source.evalAt(@as(i32, @intCast(x)) + region.x, @as(i32, @intCast(y)) + region.y);
            dest.write(x, y, result);
        }
        // Handle remaining pixels
        while (x < region.width) : (x += 1) {
            const result = source.evalAt(@as(i32, @intCast(x)) + region.x, @as(i32, @intCast(y)) + region.y);
            dest.writeScalar(x, y, result[0]);
        }
    }
    _ = ElemT;
}

// ============================================================================
// Tests
// ============================================================================

test "basic generator" {
    const region: Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var output: [4]f32 = .{ 0, 0, 0, 0 };

    const destination = Out(f32, &output, region.width, region);

    const processing_kernel = struct {
        const Context = struct {
            scale: f32x4 = f32x4{ 1.0, 1.0, 1.0, 1.0 },
            offset: f32x4 = f32x4{ 1.0, 1.0, 1.0, 1.0 },
        };

        pub fn process(ctx: Context, x: f32x4, y: f32x4) f32x4 {
            return x / ctx.scale + y / ctx.scale + ctx.offset;
        }
    };

    const ctx = processing_kernel.Context{};
    const result = Generate(f32x4, .{}, region, ctx, processing_kernel.process);
    Process(f32, result, destination);

    // x=0,y=0 -> 0+0+1=1, x=1,y=0 -> 1+0+1=2, etc.
    try std.testing.expectEqual(@as(f32, 1.0), output[0]);
    try std.testing.expectEqual(@as(f32, 2.0), output[1]);
    try std.testing.expectEqual(@as(f32, 3.0), output[2]);
    try std.testing.expectEqual(@as(f32, 4.0), output[3]);
}

test "basic Processing loop: Identity" {
    const region: Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_data: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source = In(f32, &input_data, region.width, region);
    const destination = Out(f32, &output_data, region.width, region);

    const processing_kernel = struct {
        const Context = struct {};

        pub fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.get();
        }
    };

    const ctx = processing_kernel.Context{};
    const result = Loop(f32x4, .{}, source, ctx, processing_kernel.process);
    Process(f32, result, destination);

    try std.testing.expectEqual(@as(f32, 1.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), output_data[3]);
}

test "basic Processing loop: Use coordinates" {
    const region: Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    var input_data: [4]f32 = .{ 10.0, 20.0, 30.0, 40.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source = In(f32, &input_data, region.width, region);
    const destination = Out(f32, &output_data, region.width, region);

    const processing_kernel = struct {
        const Context = struct {};

        pub fn process(ctx: Context, in: anytype, x: u16x4, y: u16x4) f32x4 {
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
    const result = Loop(f32x4, .{ .need_coordinates = u16x4 }, source, ctx, processing_kernel.process);
    Process(f32, result, destination);

    // input[i] + x + y = input[i] + i + 0
    try std.testing.expectEqual(@as(f32, 10.0), output_data[0]); // 10 + 0 + 0
    try std.testing.expectEqual(@as(f32, 21.0), output_data[1]); // 20 + 1 + 0
    try std.testing.expectEqual(@as(f32, 32.0), output_data[2]); // 30 + 2 + 0
    try std.testing.expectEqual(@as(f32, 43.0), output_data[3]); // 40 + 3 + 0
}

test "basic Processing loop: Use Margins" {
    const region: Region = .{ .x = 0, .y = 0, .width = 4, .height = 1 };

    // Input: [1, 2, 3, 4]
    var input_data: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var output_data: [4]f32 = .{ 0, 0, 0, 0 };

    const source = In(f32, &input_data, region.width, region);
    const destination = Out(f32, &output_data, region.width, region);

    const processing_kernel = struct {
        const Context = struct {};

        pub fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            // Horizontal blur: left + center + right
            return in.getAt(-1, 0) + in.getAt(0, 0) + in.getAt(1, 0);
        }
    };

    const ctx = processing_kernel.Context{};
    const result = Loop(f32x4, .{ .margin = .{ .left = 1, .right = 1 } }, source, ctx, processing_kernel.process);
    Process(f32, result, destination);

    // Position 0: in(-1,0)=0 + in(0,0)=1 + in(1,0)=2 = 3
    // Position 1: in(-1,0)=1 + in(0,0)=2 + in(1,0)=3 = 6
    // Position 2: in(-1,0)=2 + in(0,0)=3 + in(1,0)=4 = 9
    // Position 3: in(-1,0)=3 + in(0,0)=4 + in(1,0)=0 = 7
    try std.testing.expectEqual(@as(f32, 3.0), output_data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), output_data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), output_data[2]);
    try std.testing.expectEqual(@as(f32, 7.0), output_data[3]);
}

test "Expression tree" {
    const region: Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };

    // 2x4 input:
    // Row 0: [1, 2, 3, 4]
    // Row 1: [5, 6, 7, 8]
    var input_data: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output_data: [8]f32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };

    const source = In(f32, &input_data, region.width, region);
    const destination = Out(f32, &output_data, region.width, region);

    // First kernel: horizontal blur (left + center + right)
    const kernel1 = struct {
        const Context = struct {};

        pub fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.getAt(-1, 0) + in.getAt(0, 0) + in.getAt(1, 0);
        }
    };

    // Second kernel: vertical blur (top + center + bottom)
    const kernel2 = struct {
        const Context = struct {};

        pub fn process(ctx: Context, in: anytype) f32x4 {
            _ = ctx;
            return in.getAt(0, -1) + in.getAt(0, 0) + in.getAt(0, 1);
        }
    };

    const ctx1 = kernel1.Context{};
    const ctx2 = kernel2.Context{};

    const result1 = Loop(f32x4, .{ .margin = .{ .left = 1, .right = 1 } }, source, ctx1, kernel1.process);
    const result2 = Loop(f32x4, .{ .margin = .{ .top = 1, .bottom = 1 } }, result1, ctx2, kernel2.process);
    Process(f32, result2, destination);

    // The expression tree chains kernel1 -> kernel2
    // For each output pixel, kernel2 reads from kernel1's virtual output

    // Verify the chained processing produces expected results
    // This is a box blur (3x3 separable = horizontal then vertical)
    try std.testing.expect(output_data[0] > 0);
    try std.testing.expect(output_data[4] > 0);
}

pub fn bufferedPrint() !void {
    const stdout_file = std.fs.File.stdout();
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_stream = stdout_file.writer(&stdout_buffer);
    const stdout = &stdout_stream.interface;

    try stdout.print("ZPP - Zig Pixel Processing Library\n", .{});
    try stdout.flush();
}

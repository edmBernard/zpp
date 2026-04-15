const std = @import("std");

/// fill an array with a ramp of values starting from `start` and incrementing by `step`
pub fn fillRamp(comptime T: type, arr: []T, comptime start: T, comptime step: T) void {
    var value: T = start;
    for (arr) |*elem| {
        elem.* = value;
        value += step;
    }
}

pub fn splatWithCast(comptime VecT: type, value: anytype) VecT {
    switch (@typeInfo(@typeInfo(VecT).vector.child)) {
        .float => {
            switch (@typeInfo(@TypeOf(value))) {
                .int, .comptime_int => return @as(VecT, @splat(@floatFromInt(value))),
                .float, .comptime_float => return @as(VecT, @splat(@floatCast(value))),
                else => @compileError("splatWithCast only supports int and float scalar values to float"),
            }
        },
        .int => {
            switch (@typeInfo(@TypeOf(value))) {
                .int, .comptime_int => return @as(VecT, @splat(@intCast(value))),
                .float, .comptime_float => return @as(VecT, @splat(@trunc(value))),
                else => @compileError("splatWithCast only supports int and float scalar values to int"),
            }
        },
        else => @compileError("splatWithCast only supports vector of f32, u16, u8"),
    }
}

pub fn vectorCast(comptime VecT: type, value: anytype) VecT {
    switch (@typeInfo(@typeInfo(VecT).vector.child)) {
        .float => {
            switch (@typeInfo(@typeInfo(@TypeOf(value)).vector.child)) {
                .int, .comptime_int => return @as(VecT, @floatFromInt(value)),
                .float, .comptime_float => return @as(VecT, @floatCast(value)),
                else => @compileError("vectorCast only supports int and float scalar values to float"),
            }
        },
        .int => {
            switch (@typeInfo(@typeInfo(@TypeOf(value)).vector.child)) {
                .int, .comptime_int => return @as(VecT, @intCast(value)),
                .float, .comptime_float => return @as(VecT, @trunc(value)),
                else => @compileError("splatWithConversion only supports int and float scalar values to int"),
            }
        },
        else => @compileError("vectorCast only supports vector of f32, u16, u8"),
    }
}

pub fn scalarCast(comptime T: type, value: anytype) T {
    switch (@typeInfo(T)) {
        .float => {
            switch (@typeInfo(@TypeOf(value))) {
                .int, .comptime_int => return @as(T, @floatFromInt(value)),
                .float, .comptime_float => return @as(T, @floatCast(value)),
                else => @compileError("vectorCast only supports int and float scalar values to float"),
            }
        },
        .int => {
            switch (@typeInfo(@TypeOf(value))) {
                .int, .comptime_int => return @as(T, @intCast(value)),
                .float, .comptime_float => return @as(T, @trunc(value)),
                else => @compileError("vectorCast only supports int and float scalar values to int"),
            }
        },
        else => @compileError("scalarCast only supports vector of f32, u16, u8"),
    }
}

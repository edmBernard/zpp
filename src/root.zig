//! ZPP - Zig Pixel Processing Library
//!
//! A library for efficient pixel processing using SIMD vector capabilities.
//!
//! This is the main entry point that re-exports all public API symbols from
//! the individual modules.

const std = @import("std");

// ============================================================================
// MARK: Module Imports
// ============================================================================

const region = @import("region.zig");
const padding = @import("padding.zig");
const math = @import("math.zig");
const sources = @import("sources.zig");
const zip = @import("zip.zig");
const group = @import("group.zig");
const stats = @import("stats.zig");
const interpolation = @import("interpolation.zig");
const cache = @import("cache.zig");
const loop = @import("loop.zig");

// ============================================================================
// MARK: Re-exported Types and Constants
// ============================================================================

/// Default vector length - uses platform optimal size based on u8 processing
pub const suggested_vec_len = std.simd.suggestVectorLength(u8) orelse 1;
pub const f32v = @Vector(std.simd.suggestVectorLength(f32) orelse 1, f32);
pub const i32v = @Vector(std.simd.suggestVectorLength(u16) orelse 1, i32);
pub const u16v = @Vector(std.simd.suggestVectorLength(u16) orelse 1, u16);
pub const u8v = @Vector(std.simd.suggestVectorLength(u8) orelse 1, u8);

// TODO: add tests for this utility
pub fn VectorLike(comptime VectorType: type, comptime ScalarType: type) type {
    return @Vector(@typeInfo(VectorType).vector.len, ScalarType);
}

// Region module
pub const Region = region.Region;
pub const Margin = region.Margin;
pub const marginH = region.marginH;
pub const marginV = region.marginV;
pub const marginI = region.marginI;

// Padding module
pub const ZeroPadding = padding.ZeroPadding;
pub const RepeatEdgePadding = padding.RepeatEdgePadding;

// Sources module
pub const In = sources.In;
pub const InWithPadding = sources.InWithPadding;
pub const Out = sources.Out;
pub const InterleavedOut = sources.InterleavedOut;

// Zip module
pub const Zip = zip.Zip;
pub const ZipOut = zip.ZipOut;
pub const Unzip = zip.Unzip;

// Group module
pub const Group = group.Group;
pub const GroupOut = group.GroupOut;
pub const Ungroup = group.Ungroup;

// Stats module
pub const Stats = stats.Stats;
pub const StatsWithCoords = stats.StatsWithCoords;

// Interpolation module
pub const InterpolationMethod = interpolation.InterpolationMethod;
pub const InterpLoop = interpolation.InterpLoop;
pub const LoopOptions = loop.LoopOptions;
pub const DefaultLoopOptions = loop.DefaultLoopOptions;

// Loop module (core processing)
pub const Loop = loop.Loop;
pub const Generate = loop.Generate;
pub const Process = loop.Process;
// Cache module
pub const CachedLoop = cache.CachedLoop;

// ============================================================================
// MARK: Math Functions
// ============================================================================

pub const splat = math.splat;
pub const abs = math.abs;
pub const floor = math.floor;
pub const ceil = math.ceil;
pub const trunc = math.trunc;
pub const round = math.round;
pub const sqrt = math.sqrt;
pub const sin = math.sin;
pub const cos = math.cos;
pub const tan = math.tan;
pub const exp = math.exp;
pub const exp2 = math.exp2;
pub const log = math.log;
pub const log2 = math.log2;
pub const log10 = math.log10;
pub const sign = math.sign;
pub const pow = math.pow;
pub const atan2 = math.atan2;
pub const min = math.min;
pub const max = math.max;
pub const clamp = math.clamp;
pub const lerp = math.lerp;
pub const fma = math.fma;

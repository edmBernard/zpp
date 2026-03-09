//! Public ZPP API.

const std = @import("std");

const cache = @import("cache.zig");
const group_mod = @import("group.zig");
const interpolation = @import("interpolation.zig");
const loop_mod = @import("loop.zig");
const padding = @import("padding.zig");
const region = @import("region.zig");
const sources_mod = @import("sources.zig");
const stats_mod = @import("stats.zig");
const translate_mod = @import("translate.zig");
const zip_mod = @import("zip.zig");

fn SuggestedVector(comptime T: type) type {
    return @Vector(std.simd.suggestVectorLength(T) orelse 1, T);
}

/// Platform-suggested SIMD lane count for 8bit image workloads.
pub const suggested_vec_len = std.simd.suggestVectorLength(u8) orelse 1;
/// Convenience SIMD aliases using the platform-suggested lane count.
pub const f32v = SuggestedVector(f32);
pub const i32v = SuggestedVector(i32);
pub const u16v = SuggestedVector(u16);
pub const u8v = SuggestedVector(u8);

/// Reuse a vector type's lane count with a different scalar type.
pub fn VectorLike(comptime VectorType: type, comptime ScalarType: type) type {
    return @Vector(@typeInfo(VectorType).vector.len, ScalarType);
}

pub const Region = region.Region;
pub const Margin = region.Margin;

pub const ZeroPadding = padding.ZeroPadding;
pub const RepeatEdgePadding = padding.RepeatEdgePadding;

pub const makeSource = sources_mod.makeSource;
pub const makePaddedSource = sources_mod.makePaddedSource;
pub const makeDest = sources_mod.makeDest;
pub const makeInterleavedDest = sources_mod.makeInterleavedDest;

pub const translate = translate_mod.translate;

pub const zip = zip_mod.zip;
pub const zipDest = zip_mod.zipDest;
pub const unzip = zip_mod.unzip;

pub const group = group_mod.group;
pub const groupDest = group_mod.groupDest;
pub const ungroup = group_mod.ungroup;

pub const stats = stats_mod.stats;
pub const statsWithCoords = stats_mod.statsWithCoords;

pub const InterpolationMethod = interpolation.InterpolationMethod;
pub const interpLoop = interpolation.interpLoop;

pub const LoopOptions = loop_mod.LoopOptions;
pub const loop = loop_mod.loop;
pub const generate = loop_mod.generate;
pub const process = loop_mod.process;

pub const cachedLoop = cache.cachedLoop;

/// SIMD math helpers such as `fract`, `pow`, and `lerp`.
pub const math = @import("math.zig");
/// Small vector and matrix helpers parameterized by SIMD vector type.
pub const zla = @import("zla");

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
const sources_mod = @import("sources.zig");
const translate_mod = @import("translate.zig");
const zip_mod = @import("zip.zig");
const group_mod = @import("group.zig");
const stats_mod = @import("stats.zig");
const interpolation = @import("interpolation.zig");
const loop_mod = @import("loop.zig");
const cache = @import("cache.zig");

// ============================================================================
// MARK: Re-exported Types and Constants
// ============================================================================

fn suggestedVectorType(comptime T: type) type {
    return @Vector(std.simd.suggestVectorLength(T) orelse 1, T);
}

/// Convenience alias for a platform-suggested SIMD lane count.
pub const suggested_vec_len = std.simd.suggestVectorLength(u8) orelse 1;
/// Convenience aliases for common SIMD types; the API does not require them.
pub const f32v = suggestedVectorType(f32);
pub const i32v = suggestedVectorType(i32);
pub const u16v = suggestedVectorType(u16);
pub const u8v = suggestedVectorType(u8);

// TODO: add tests for this utility
/// Method to create a vector type compatible with a given vector type but with a different scalar type.
pub fn VectorLike(comptime VectorType: type, comptime ScalarType: type) type {
    return @Vector(@typeInfo(VectorType).vector.len, ScalarType);
}

// Region module
pub const Region = region.Region;
pub const Margin = region.Margin;

// Padding module
pub const ZeroPadding = padding.ZeroPadding;
pub const RepeatEdgePadding = padding.RepeatEdgePadding;

// Sources module
pub const makeSource = sources_mod.makeSource;
pub const makePaddedSource = sources_mod.makePaddedSource;
pub const makeDest = sources_mod.makeDest;
pub const makeInterleavedDest = sources_mod.makeInterleavedDest;

// Translate module
pub const translate = translate_mod.translate;

// Zip module
pub const zip = zip_mod.zip;
pub const zipDest = zip_mod.zipDest;
pub const unzip = zip_mod.unzip;

// Group module
pub const group = group_mod.group;
pub const groupDest = group_mod.groupDest;
pub const ungroup = group_mod.ungroup;

// Stats module
pub const stats = stats_mod.stats;
pub const statsWithCoords = stats_mod.statsWithCoords;

// Interpolation module
pub const InterpolationMethod = interpolation.InterpolationMethod;
pub const interpLoop = interpolation.interpLoop;

// Loop module (core processing)
pub const LoopOptions = loop_mod.LoopOptions;
pub const loop = loop_mod.loop;
pub const generate = loop_mod.generate;
pub const process = loop_mod.process;

// Cache module
pub const cachedLoop = cache.cachedLoop;

// ============================================================================
// MARK: Math Functions
// ============================================================================

pub const math = @import("math.zig");

// Linear algebra module
pub const zla = @import("zla");

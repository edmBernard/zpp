//! Core processing loop primitives for ZPP.
//!
//! This module provides the fundamental building blocks for pixel processing:
//! - InputAccessor: provides neighborhood access for kernels
//! - LoopResult: lazy evaluation of processing operations
//! - GeneratorResult: generates values from coordinates
//! - Loop, Generate, Process: main API functions

const std = @import("std");
const region_mod = @import("region.zig");
const padding_mod = @import("padding.zig");
const zip_mod = @import("zip.zig");
const group_mod = @import("group.zig");

const Region = region_mod.Region;
const Margin = region_mod.Margin;

// ============================================================================
// MARK: Loop Options
// ============================================================================

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

// ============================================================================
// MARK: Helper Functions
// ============================================================================

/// Helper to detect if a type is a ZipSource
pub fn isZipSourceType(comptime T: type) bool {
    return zip_mod.isZipSourceType(T);
}

/// Helper to detect if a type is a GroupSource
pub fn isGroupSourceType(comptime T: type) bool {
    return group_mod.isGroupSourceType(T);
}

/// Extract the inner vector type from a type.
/// If T is a vector, returns T.
/// If T is an array of vectors (e.g., [3]@Vector(4, f32)), returns the vector type.
/// If T is a tuple of vectors (e.g., struct{f32x4, f32x4}), returns the first vector type.
pub fn innerVectorType(comptime T: type) type {
    const type_info = @typeInfo(T);
    if (type_info == .vector) {
        return T;
    } else if (type_info == .array) {
        const child_info = @typeInfo(type_info.array.child);
        if (child_info == .vector) {
            return type_info.array.child;
        }
    } else if (type_info == .@"struct" and type_info.@"struct".is_tuple) {
        // Tuple of vectors - return the first element's type
        if (type_info.@"struct".fields.len > 0) {
            const first_field_type = type_info.@"struct".fields[0].type;
            if (@typeInfo(first_field_type) == .vector) {
                return first_field_type;
            }
        }
    }
    @compileError("Expected vector, array of vectors, or tuple of vectors, got " ++ @typeName(T));
}

/// Get the vector length from either a vector type, array of vectors, or tuple of vectors
pub fn vectorLen(comptime T: type) comptime_int {
    const type_info = @typeInfo(T);
    if (type_info == .vector) {
        return type_info.vector.len;
    } else if (type_info == .array) {
        const child_info = @typeInfo(type_info.array.child);
        if (child_info == .vector) {
            return child_info.vector.len;
        }
    } else if (type_info == .@"struct" and type_info.@"struct".is_tuple) {
        if (type_info.@"struct".fields.len > 0) {
            const first_field_type = type_info.@"struct".fields[0].type;
            const first_info = @typeInfo(first_field_type);
            if (first_info == .vector) {
                return first_info.vector.len;
            }
        }
    }
    @compileError("Expected vector, array of vectors, or tuple of vectors, got " ++ @typeName(T));
}

/// Get the return type of a process function
pub fn ProcessReturnType(comptime process_fn: anytype) type {
    const fn_info = @typeInfo(@TypeOf(process_fn)).@"fn";
    return fn_info.return_type.?;
}

// ============================================================================
// MARK: Input Accessor
// ============================================================================

/// Input accessor that provides neighborhood access for kernels.
/// Supports configurable vector length and uses the source's padding policy.
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
            } else if (@hasDecl(SrcType, "readVec")) {
                // Source is an InputSource with readVec method (vectorized SIMD read)
                return self.source.readVec(VecT, x, y);
            } else {
                @compileError("Invalid source type for InputAccessor: missing evalAt or readVec");
            }
        }

        /// Get current value (no offset) - for identity operations
        pub inline fn get(self: Self) VecT {
            return self.getAt(0, 0);
        }
    };
}

// ============================================================================
// MARK: Loop Result
// ============================================================================

/// Lazy loop result that can be chained or processed
pub fn LoopResult(
    comptime VecT: type,
    comptime SrcType: type,
    comptime CtxType: type,
    comptime process_fn: anytype,
    comptime opts: DefaultLoopOptions,
) type {
    const has_coords = opts.need_coordinates != null;
    const is_zip_source = isZipSourceType(SrcType);
    const is_group_source = isGroupSourceType(SrcType);

    // VecT is the base vector type for the accessor
    const AccessorType = if (comptime is_zip_source)
        zip_mod.ZipAccessor(SrcType, VecT)
    else if (comptime is_group_source)
        group_mod.GroupAccessor(SrcType, VecT, SrcType.group_width, SrcType.group_height)
    else
        InputAccessor(SrcType, VecT);
    const vec_len = @typeInfo(VecT).vector.len;

    // Get the actual return type from the process function
    const ReturnType = ProcessReturnType(process_fn);

    return struct {
        source: SrcType,
        context: CtxType,
        region: Region,

        // Marker to identify this as a LoopResult for expression trees
        const source_type = SrcType;

        /// The output type of evalAt (may be VecT, [N]VecT, or tuple)
        pub const OutputType = ReturnType;

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        const Self = @This();

        /// Evaluate at a specific position
        pub inline fn evalAt(self: Self, x: i32, y: i32) ReturnType {
            const accessor = AccessorType{
                .source = self.source,
                .current_x = x,
                .current_y = y,
            };

            if (comptime has_coords) {
                const CoordT = opts.need_coordinates.?;
                const CoordElemT = @typeInfo(CoordT).vector.child;
                const iota = std.simd.iota(CoordElemT, vec_len);
                const x_coords: CoordT = iota + @as(CoordT, @splat(@intCast(x)));
                const y_coords: CoordT = @splat(@intCast(y));
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

// ============================================================================
// MARK: Generator Result
// ============================================================================

/// Generator result (no input source, just generates based on coordinates)
/// Supports both single-channel (VecT) and multi-channel ([N]VecT) output.
pub fn GeneratorResult(
    comptime VecT: type,
    comptime CtxType: type,
    comptime process_fn: anytype,
    comptime vec_len: comptime_int,
) type {
    // Determine the output type from the process function's return type
    const OutputType = ProcessReturnType(process_fn);

    return struct {
        context: CtxType,
        region: Region,

        const Self = @This();

        /// The output type of this generator (e.g., VecF32 or [3]VecF32)
        pub const output_type = OutputType;

        /// The vector type used for coordinates
        pub const coord_vec_type = VecT;

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        const iota = std.simd.iota(f32, vec_len);

        pub inline fn evalAt(self: Self, x: i32, y: i32) OutputType {
            const x_vec: VecT = iota + @as(VecT, @splat(@floatFromInt(x)));
            const y_vec: VecT = @splat(@floatFromInt(y));
            return process_fn(self.context, x_vec, y_vec);
        }

        pub fn getRegion(self: Self) Region {
            return self.region;
        }
    };
}

/// Create a generator (produces values from coordinates, no input)
/// Supports both single-channel (VecT) and multi-channel ([N]VecT) output.
/// The vector length is inferred from the VecT type.
pub fn Generate(
    comptime VecT: type,
    region: Region,
    context: anytype,
    comptime process_fn: anytype,
) GeneratorResult(VecT, @TypeOf(context), process_fn, @typeInfo(VecT).vector.len) {
    return .{
        .context = context,
        .region = region,
    };
}

// ============================================================================
// MARK: Process
// ============================================================================

/// Execute the processing pipeline and write results to output.
/// Supports both single-channel and multi-channel sources.
/// The destination region drives what gets computed (pull model).
/// The destination must have write() and writeScalar() methods matching the source output type.
pub fn Process(source: anytype, dest: anytype) void {
    const region = dest.region;
    const SourceType = @TypeOf(source);

    // Determine vector length - check for generator metadata or default to 4
    const vec_len = SourceType.vector_length;

    const DestType = @TypeOf(dest);
    // Check if destination supports idempotent/overlapping writes (e.g., pixel buffers)
    // Accumulators like Stats do not support this and must use scalar remainder handling
    const supports_overlapping_writes = @hasDecl(DestType, "supports_overlapping_writes") and DestType.supports_overlapping_writes;

    for (0..region.height) |y| {
        const y_coord: i32 = @as(i32, @intCast(y)) + region.y;

        // Process full vectors
        var x: u32 = 0;
        while (x + vec_len <= region.width) : (x += vec_len) {
            const result = source.evalAt(@as(i32, @intCast(x)) + region.x, y_coord);
            dest.write(x, @intCast(y), result);
        }

        // Handle remaining pixels
        if (x < region.width) {
            if (supports_overlapping_writes and region.width >= vec_len) {
                // Optimization: shift x back to process the last vec_len pixels as a full vector
                // This overlaps with already-written pixels but is safe for idempotent destinations
                const aligned_x = region.width - vec_len;
                const result = source.evalAt(@as(i32, @intCast(aligned_x)) + region.x, y_coord);
                dest.write(aligned_x, @intCast(y), result);
            } else {
                // Scalar fallback for accumulators or narrow regions
                while (x < region.width) : (x += 1) {
                    const result = source.evalAt(@as(i32, @intCast(x)) + region.x, y_coord);
                    dest.writeScalar(x, @intCast(y), result);
                }
            }
        }
    }
}

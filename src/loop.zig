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

/// Options for Loop operations
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
// MARK: Generator Result
// ============================================================================

/// Generator result (no input source, just generates based on coordinates)
/// Supports both single-channel (VecT) and multi-channel ([N]VecT) output.
pub fn GeneratorResult(
    comptime CoordT: type,
    comptime ContextType: type,
    comptime process_fn: anytype,
    comptime vec_len: comptime_int,
) type {
    // Determine the output type from the process function's return type
    return struct {
        context: ContextType,

        const Self = @This();

        /// The output type of this generator (e.g., VecF32 or [3]VecF32)
        pub const OutputType = ProcessReturnType(process_fn);

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        const iota = std.simd.iota(@typeInfo(CoordT).vector.child, vec_len);

        pub inline fn evalAt(self: Self, x: i32, y: i32) OutputType {
            const x_vec: CoordT = iota + CastScalarCoordToVector(CoordT, x);
            const y_vec: CoordT = CastScalarCoordToVector(CoordT, y);
            return process_fn(self.context, x_vec, y_vec);
        }
    };
}

inline fn CastScalarCoordToVector(comptime CoordT: type, value: i32) CoordT {
    switch (@typeInfo(CoordT).vector.child) {
        f32 => {
            return @as(CoordT, @splat(@floatFromInt(value)));
        },
        u16, u8 => {
            return @as(CoordT, @splat(@intCast(value)));
        },
        else => @compileError("CastScalarCoordToVector only supports f32, u16, u8 scalars"),
    }
}

/// Create a generator (produces values from coordinates, no input)
/// Supports both single-channel (VecT) and multi-channel ([N]VecT) output.
/// The vector length is inferred from the VecT type.
pub fn Generate(
    comptime VecT: type,
    context: anytype,
    comptime process_fn: anytype,
) GeneratorResult(VecT, @TypeOf(context), process_fn, @typeInfo(VecT).vector.len) {
    return .{
        .context = context,
    };
}

// ============================================================================
// MARK: Input Accessor
// ============================================================================

/// Input accessor that provides neighborhood access for kernels.
/// Supports configurable vector length and uses the source's padding policy.
/// When unchecked=true, skips bounds checking for the interior fast path.
pub fn InputAccessorGeneric(comptime SourceType: type, comptime VecT: type, comptime unchecked: bool) type {
    const ReturnType = if (@hasDecl(SourceType, "OutputType")) SourceType.OutputType else VecT;

    return struct {
        source: SourceType,
        current_x: i32,
        current_y: i32,

        const Self = @This();

        /// Get value at offset (for margin-based operations)
        pub inline fn getAt(self: Self, dx: i32, dy: i32) ReturnType {
            const x = self.current_x + dx;
            const y = self.current_y + dy;

            // Check if source is a LoopResult (for expression trees) - has evalAt
            if (@hasDecl(SourceType, "evalAt")) {
                if (comptime unchecked and @hasDecl(SourceType, "evalAtUnchecked")) {
                    return self.source.evalAtUnchecked(x, y);
                } else {
                    return self.source.evalAt(x, y);
                }
            } else if (@hasDecl(SourceType, "readVec")) {
                if (comptime unchecked and @hasDecl(SourceType, "readVecUnchecked")) {
                    return self.source.readVecUnchecked(VecT, x, y);
                } else {
                    return self.source.readVec(VecT, x, y);
                }
            } else {
                @compileError("Invalid source type for InputAccessor: missing evalAt or readVec");
            }
        }

        /// Get current value (no offset) - for identity operations
        pub inline fn get(self: Self) ReturnType {
            return self.getAt(0, 0);
        }
    };
}

/// Standard input accessor with bounds checking (default).
pub fn InputAccessor(comptime SourceType: type, comptime VecT: type) type {
    return InputAccessorGeneric(SourceType, VecT, false);
}

/// Unchecked input accessor that skips bounds checking (for interior fast path).
pub fn UncheckedInputAccessor(comptime SourceType: type, comptime VecT: type) type {
    return InputAccessorGeneric(SourceType, VecT, true);
}

// ============================================================================
// MARK: Loop Result
// ============================================================================

/// Lazy loop result that can be chained or processed
pub fn LoopResult(
    comptime VecT: type,
    comptime SourceType: type,
    comptime ContextType: type,
    comptime process_fn: anytype,
    comptime opts: DefaultLoopOptions,
) type {
    const has_coords = opts.need_coordinates != null;
    const is_zip_source = isZipSourceType(SourceType);
    const is_group_source = isGroupSourceType(SourceType);

    // VecT is the base vector type for the accessor
    const AccessorType = if (comptime is_zip_source)
        zip_mod.ZipAccessor(SourceType, VecT)
    else if (comptime is_group_source)
        group_mod.GroupAccessor(SourceType, VecT, SourceType.group_width, SourceType.group_height)
    else
        InputAccessor(SourceType, VecT);

    // Unchecked accessor for interior fast path (only for simple sources, not zip/group)
    const UncheckedAccessorType = if (comptime is_zip_source)
        zip_mod.ZipAccessor(SourceType, VecT)
    else if (comptime is_group_source)
        group_mod.GroupAccessor(SourceType, VecT, SourceType.group_width, SourceType.group_height)
    else
        UncheckedInputAccessor(SourceType, VecT);

    const vec_len = @typeInfo(VecT).vector.len;

    // Get the actual return type from the process function
    const ReturnType = ProcessReturnType(process_fn);

    // Check if the source chain supports unchecked access
    const source_has_unchecked = @hasDecl(SourceType, "readVecUnchecked") or
        @hasDecl(SourceType, "evalAtUnchecked");

    return struct {
        source: SourceType,
        context: ContextType,
        region: Region,

        // Marker to identify this as a LoopResult for expression trees
        const source_type = SourceType;

        /// The output type of evalAt (may be VecT, [N]VecT, or tuple)
        pub const OutputType = ReturnType;

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        /// The margin used by this loop's kernel
        pub const margin = opts.margin;

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
                const x_coords: CoordT = iota + CastScalarCoordToVector(CoordT, x);
                const y_coords: CoordT = CastScalarCoordToVector(CoordT, y);
                return process_fn(self.context, accessor, x_coords, y_coords);
            } else {
                return process_fn(self.context, accessor);
            }
        }

        /// Evaluate at a specific position without bounds checking.
        /// Caller MUST guarantee that all accessed positions (including margin offsets)
        /// are within the source's data region.
        pub inline fn evalAtUnchecked(self: Self, x: i32, y: i32) ReturnType {
            if (comptime !source_has_unchecked) {
                return self.evalAt(x, y);
            }
            const accessor = UncheckedAccessorType{
                .source = self.source,
                .current_x = x,
                .current_y = y,
            };

            if (comptime has_coords) {
                const CoordT = opts.need_coordinates.?;
                const CoordElemT = @typeInfo(CoordT).vector.child;
                const iota = std.simd.iota(CoordElemT, vec_len);
                const x_coords: CoordT = iota + CastScalarCoordToVector(CoordT, x);
                const y_coords: CoordT = CastScalarCoordToVector(CoordT, y);
                return process_fn(self.context, accessor, x_coords, y_coords);
            } else {
                return process_fn(self.context, accessor);
            }
        }

        /// Returns the region where evalAtUnchecked can be safely called.
        /// This is the source region deflated by the kernel's margin and vec_len.
        /// For position x in this region, all reads x-margin.left .. x+margin.right+vec_len-1
        /// are guaranteed in-bounds of the source data.
        /// Returns an empty region if the margin is zero (no benefit from split iteration)
        /// or if the source doesn't support unchecked access.
        pub fn getInteriorRegion(self: Self) Region {
            if (comptime !source_has_unchecked or opts.margin.isZero()) {
                // No unchecked path available or no margin → no benefit from split
                return .{ .x = 0, .y = 0, .width = 0, .height = 0 };
            }
            // The source region is where data is available
            const src_region = self.region;
            // Deflate by margin to get where all margin reads are in-bounds
            // Left: need margin.left pixels to the left
            // Right: need margin.right + vec_len - 1 pixels to the right (for the vector read)
            // Top: need margin.top rows above
            // Bottom: need margin.bottom rows below
            return src_region.deflated(
                @intCast(opts.margin.left),
                @intCast(opts.margin.top),
                @as(i32, @intCast(opts.margin.right)) + vec_len - 1,
                @intCast(opts.margin.bottom),
            );
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
    const SourceType = @TypeOf(source);

    const region = if (@hasDecl(SourceType, "getRegion"))
        source.getRegion()
    else if (@hasField(SourceType, "region"))
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
// MARK: Process
// ============================================================================

/// Get compatible vector length.
/// In case of conversion the length must accommodate both source and destination.
fn getCompatibleVectorLen(comptime SourceType: type, comptime DestType: type) comptime_int {
    const suggested_source_len = std.simd.suggestVectorLength(SourceType.OutputScalarType) orelse 1;
    const suggested_dest_len = std.simd.suggestVectorLength(DestType.InputScalarType) orelse 1;
    return @max(suggested_source_len, suggested_dest_len);
}

/// Get vector length from a source type.
/// Returns the source's vector_length if it has one, otherwise returns 4
/// (a conservative default for direct Process calls).
fn getSourceVecLen(comptime SourceType: type) ?comptime_int {
    if (@hasDecl(SourceType, "vector_length")) {
        return SourceType.vector_length;
    }
    return null;
}

/// Execute the processing pipeline and write results to output.
/// Supports both single-channel and multi-channel sources.
/// The destination region drives what gets computed (pull model).
/// The destination must have write() and writeScalar() methods matching the source output type.
///
/// When the source supports it (LoopResult with margins over an InputSource),
/// splits each row into left-edge / interior / right-edge strips. The interior
/// strip uses unchecked reads (no bounds checking), which eliminates per-vector
/// bounds comparisons for the majority of pixels.
pub fn Process(source: anytype, dest: anytype) void {
    const region = dest.region;
    const SourceType = @TypeOf(source);
    const DestType = @TypeOf(dest);

    // Determine vector length from Source or from Source and Destination
    const vec_len = getSourceVecLen(SourceType) orelse getCompatibleVectorLen(SourceType, DestType);

    // Check if destination supports idempotent/overlapping writes (e.g., pixel buffers)
    // Accumulators like Stats do not support this and must use scalar remainder handling
    const supports_overlapping_writes = @hasDecl(DestType, "supports_overlapping_writes") and DestType.supports_overlapping_writes;

    // Check if source supports split iteration (unchecked interior path)
    const has_unchecked = @hasDecl(SourceType, "evalAtUnchecked") or @hasDecl(SourceType, "readVecUnchecked");
    const has_interior = @hasDecl(SourceType, "getInteriorRegion") and has_unchecked;

    if (has_interior) {
        const interior = source.getInteriorRegion();
        // Only use split path if interior region is non-empty
        if (interior.width > 0 and interior.height > 0) {
            processSplit(SourceType, DestType, source, dest, region, interior, vec_len, supports_overlapping_writes);
            return;
        }
    }

    // Standard path (no split optimization)
    processStandard(SourceType, DestType, source, dest, region, vec_len, supports_overlapping_writes);
}

/// Standard processing path without split iteration.
fn processStandard(
    comptime SourceType: type,
    comptime DestType: type,
    source: SourceType,
    dest: DestType,
    region: Region,
    comptime vec_len: comptime_int,
    comptime supports_overlapping_writes: bool,
) void {
    for (0..region.height) |y| {
        const y_coord: i32 = @as(i32, @intCast(y)) + region.y;
        processRowChecked(SourceType, DestType, source, dest, region.x, region.x + @as(i32, @intCast(region.width)), y_coord, vec_len, supports_overlapping_writes);
    }
}

/// Split processing path: left-edge (checked) / interior (unchecked) / right-edge (checked).
fn processSplit(
    comptime SourceType: type,
    comptime DestType: type,
    source: SourceType,
    dest: DestType,
    region: Region,
    interior: Region,
    comptime vec_len: comptime_int,
    comptime supports_overlapping_writes: bool,
) void {
    const dst_x = region.x;
    const dst_stop_x = region.x + @as(i32, @intCast(region.width));

    // Interior X bounds (clamped to destination region)
    const int_x_start = @max(dst_x, interior.x);
    const int_x_stop = @min(dst_stop_x, interior.stopX());

    for (0..region.height) |y| {
        const y_coord: i32 = @as(i32, @intCast(y)) + region.y;

        // Check if this row is within the interior's Y range
        const y_in_interior = y_coord >= interior.y and y_coord < interior.stopY();

        if (y_in_interior and int_x_start < int_x_stop) {
            // Left edge: checked path
            if (dst_x < int_x_start) {
                processRowChecked(SourceType, DestType, source, dest, dst_x, int_x_start, y_coord, vec_len, supports_overlapping_writes);
            }

            // Interior: unchecked fast path
            {
                var x_coord: i32 = int_x_start;
                while (x_coord + vec_len <= int_x_stop) : (x_coord += vec_len) {
                    const result = blk: {
                        if (@hasDecl(SourceType, "evalAtUnchecked")) {
                            break :blk source.evalAtUnchecked(x_coord, y_coord);
                        } else if (@hasDecl(SourceType, "readVecUnchecked")) {
                            break :blk source.readVecUnchecked(@Vector(vec_len, SourceType.OutputScalarType), x_coord, y_coord);
                        } else {
                            @compileError("Source must have evalAtUnchecked or readVecUnchecked for split iteration");
                        }
                    };
                    dest.write(@intCast(x_coord), @intCast(y_coord), result);
                }
                // Interior remainder (still unchecked, use overlapping write if possible)
                if (x_coord < int_x_stop) {
                    if (supports_overlapping_writes and int_x_stop - int_x_start >= vec_len) {
                        const aligned_x = int_x_stop - vec_len;
                        const result = blk: {
                            if (@hasDecl(SourceType, "evalAtUnchecked")) {
                                break :blk source.evalAtUnchecked(aligned_x, y_coord);
                            } else if (@hasDecl(SourceType, "readVecUnchecked")) {
                                break :blk source.readVecUnchecked(@Vector(vec_len, SourceType.OutputScalarType), aligned_x, y_coord);
                            } else {
                                @compileError("Source must have evalAtUnchecked or readVecUnchecked for split iteration");
                            }
                        };
                        dest.write(@intCast(aligned_x), @intCast(y_coord), result);
                    } else {
                        // Fall back to checked for remaining interior pixels
                        processRowChecked(SourceType, DestType, source, dest, x_coord, int_x_stop, y_coord, vec_len, supports_overlapping_writes);
                    }
                }
            }

            // Right edge: checked path
            if (int_x_stop < dst_stop_x) {
                processRowChecked(SourceType, DestType, source, dest, int_x_stop, dst_stop_x, y_coord, vec_len, supports_overlapping_writes);
            }
        } else {
            // Entire row is in edge zone: use checked path
            processRowChecked(SourceType, DestType, source, dest, dst_x, dst_stop_x, y_coord, vec_len, supports_overlapping_writes);
        }
    }
}

/// Extract the first SIMD lane from a result value.
/// For vectors: returns the scalar at lane 0.
/// For struct tuples (from Zip/Group): returns an array of scalars from each element's lane 0.
fn firstLane(result: anytype) FirstLaneType(@TypeOf(result)) {
    const T = @TypeOf(result);
    const info = @typeInfo(T);
    switch (info) {
        .vector => return result[0],
        .array => |a| {
            var out: [a.len]FirstLaneType(a.child) = undefined;
            inline for (0..a.len) |i| {
                out[i] = firstLane(result[i]);
            }
            return out;
        },
        .@"struct" => |s| {
            var out: FirstLaneType(T) = undefined;
            inline for (0..s.fields.len) |i| {
                out[i] = firstLane(result[i]);
            }
            return out;
        },
        else => @compileError("Unsupported type for firstLane: expected vector, array, or struct tuple"),
    }
}

fn FirstLaneType(comptime T: type) type {
    const info = @typeInfo(T);
    return switch (info) {
        .vector => |v| v.child,
        .array => |a| [a.len]FirstLaneType(a.child),
        .@"struct" => |s| [s.fields.len]FirstLaneType(s.fields[0].type),
        else => @compileError("Unsupported type for FirstLaneType: " ++ @typeName(T)),
    };
}

/// Process a row segment [x_start, x_stop) using checked (bounds-checking) reads.
fn processRowChecked(
    comptime SourceType: type,
    comptime DestType: type,
    source: SourceType,
    dest: DestType,
    x_start: i32,
    x_stop: i32,
    y_coord: i32,
    comptime vec_len: comptime_int,
    comptime supports_overlapping_writes: bool,
) void {
    const width = x_stop - x_start;
    if (width <= 0) return;

    // Process full vectors
    var x_coord: i32 = x_start;
    while (x_coord + vec_len <= x_stop) : (x_coord += vec_len) {
        const result = blk: {
            if (@hasDecl(SourceType, "evalAt")) {
                break :blk source.evalAt(x_coord, y_coord);
            } else if (@hasDecl(SourceType, "readVec")) {
                break :blk source.readVec(@Vector(vec_len, SourceType.OutputScalarType), x_coord, y_coord);
            } else {
                @compileError("Source must have evalAt or readVec method");
            }
        };
        dest.write(@intCast(x_coord), @intCast(y_coord), result);
    }

    // Handle remaining pixels
    if (x_coord < x_stop) {
        if (supports_overlapping_writes and width >= vec_len) {
            const aligned_x = x_stop - vec_len;
            const result = blk: {
                if (@hasDecl(SourceType, "evalAt")) {
                    break :blk source.evalAt(aligned_x, y_coord);
                } else if (@hasDecl(SourceType, "readVec")) {
                    break :blk source.readVec(@Vector(vec_len, SourceType.OutputScalarType), aligned_x, y_coord);
                } else {
                    @compileError("Source must have evalAt or readVec method");
                }
            };
            dest.write(@intCast(aligned_x), @intCast(y_coord), result);
        } else {
            while (x_coord < x_stop) : (x_coord += 1) {
                const result = blk: {
                    if (@hasDecl(SourceType, "evalAt")) {
                        break :blk source.evalAt(x_coord, y_coord);
                    } else if (@hasDecl(SourceType, "readVec")) {
                        break :blk source.readVec(@Vector(vec_len, SourceType.OutputScalarType), x_coord, y_coord);
                    } else {
                        @compileError("Source must have evalAt or readVec method");
                    }
                };
                dest.writeScalar(@intCast(x_coord), @intCast(y_coord), firstLane(result));
            }
        }
    }
}

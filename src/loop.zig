//! Core processing loop primitives for ZPP.
//!
//! This module provides the fundamental building blocks for pixel processing:
//! - InputAccessorGeneric: provides neighborhood access for kernels
//! - LoopResult: lazy evaluation of processing operations
//! - GeneratorResult: generates values from coordinates
//! - Loop, Generate, Process: main API functions

const std = @import("std");
const sources = @import("sources.zig");
const zip = @import("zip.zig");
const group = @import("group.zig");
const Region = @import("region.zig").Region;
const Margin = @import("region.zig").Margin;

// ============================================================================
// MARK: Loop Options
// ============================================================================

/// Options for Loop operations.
pub const LoopOptions = struct {
    coord_type: ?type = null,
    /// The maximum neighborhood the kernel is allowed to read through the
    /// accessor: `getAt(dx, dy)` requires dx in [-left, right] and
    /// dy in [-top, bottom].
    ///
    /// This declaration is a contract, not a hint: the interior fast path
    /// uses unchecked reads sized from it, so reading outside the declared
    /// margin is undefined behavior in unsafe builds. Violations trip an
    /// assertion in safe builds.
    margin: Margin = .{},
};

// ============================================================================
// MARK: Helper Functions
// ============================================================================

/// Helper to detect if a type is a ZipSource
pub fn isZipSourceType(comptime T: type) bool {
    return zip.isZipSourceType(T);
}

/// Helper to detect if a type is a GroupSource
pub fn isGroupSourceType(comptime T: type) bool {
    return group.isGroupSourceType(T);
}

/// Get the return type of a process function
fn ProcessReturnType(comptime process_fn: anytype) type {
    const fn_info = @typeInfo(@TypeOf(process_fn)).@"fn";
    return fn_info.return_type.?;
}

/// Comptime check that a chained source's vector width matches the loop's
/// VecT lane count. A mismatch would make process() advance positions by the
/// wrong step and silently corrupt the output.
///
/// Read-kind sources (InputSource and wrappers) adapt to any lane count and
/// are not checked. Zip/group sources are checked recursively: their
/// accessors re-read adaptive nested sources at VecT's lane count, so only
/// fixed-width (eval-kind) nested stages constrain the loop.
pub fn assertSourceVectorLength(comptime VecT: type, comptime SourceType: type) void {
    if (comptime isZipSourceType(SourceType)) {
        inline for (zip.sourceTypesFromTuple(SourceType.Sources)) |NestedType| {
            assertSourceVectorLength(VecT, NestedType);
        }
        return;
    }
    if (comptime isGroupSourceType(SourceType)) {
        assertSourceVectorLength(VecT, @FieldType(SourceType, "nested"));
        return;
    }
    if (comptime sources.SourceTraits(SourceType).kind == .read) {
        return;
    }

    const vec_len = @typeInfo(VecT).vector.len;
    if (zip.getSourceVecLen(SourceType)) |src_len| {
        if (src_len != vec_len) {
            @compileError(std.fmt.comptimePrint(
                "loop vector type {s} has {d} lanes but source {s} produces {d}-lane vectors; every stage of an expression tree must use the same lane count",
                .{ @typeName(VecT), vec_len, @typeName(SourceType), src_len },
            ));
        }
    }
}

// ============================================================================
// MARK: Generator Result
// ============================================================================

fn coordVectorError(comptime CoordVecT: type, comptime detail: []const u8) noreturn {
    @compileError("Coordinate vector type " ++ @typeName(CoordVecT) ++ " " ++ detail);
}

fn CoordScalarType(comptime CoordVecT: type) type {
    return switch (@typeInfo(CoordVecT)) {
        .vector => |info| switch (@typeInfo(info.child)) {
            .int, .comptime_int, .float, .comptime_float => info.child,
            else => coordVectorError(CoordVecT, "must use an integer or float scalar, got " ++ @typeName(info.child)),
        },
        else => coordVectorError(CoordVecT, "must be a vector type"),
    };
}

fn coordVectorLen(comptime CoordVecT: type) comptime_int {
    _ = CoordScalarType(CoordVecT);
    return switch (@typeInfo(CoordVecT)) {
        .vector => |info| info.len,
        else => unreachable,
    };
}

/// Generator result (no input source, just generates based on coordinates)
/// Supports both single-channel (VecT) and multi-channel ([N]VecT) output.
pub fn GeneratorResult(
    comptime CoordVecT: type,
    comptime ContextType: type,
    comptime process_fn: anytype,
    comptime vec_len: comptime_int,
) type {
    // Determine the output type from the process function's return type
    return struct {
        context: ContextType,

        /// The output type of this generator (e.g., VecF32 or [3]VecF32)
        pub const OutputType = ProcessReturnType(process_fn);

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        const CoordScalarT = CoordScalarType(CoordVecT);
        const iota = std.simd.iota(CoordScalarT, vec_len);
        const Self = @This();

        pub inline fn evalAt(self: Self, x: i32, y: i32) OutputType {
            const x_vec: CoordVecT = iota + splatCoordScalar(CoordVecT, x);
            const y_vec: CoordVecT = splatCoordScalar(CoordVecT, y);
            return process_fn(self.context, x_vec, y_vec);
        }
    };
}

pub inline fn splatCoordScalar(comptime CoordVecT: type, value: i32) CoordVecT {
    const CoordScalarT = CoordScalarType(CoordVecT);
    return switch (@typeInfo(CoordScalarT)) {
        .float, .comptime_float => @as(CoordVecT, @splat(@as(CoordScalarT, @floatFromInt(value)))),
        .int, .comptime_int => @as(CoordVecT, @splat(@as(CoordScalarT, @intCast(value)))),
        else => unreachable,
    };
}

/// Create a generator (produces values from coordinates, no input)
/// Supports both single-channel (VecT) and multi-channel ([N]VecT) output.
/// The coordinate vector length is inferred from `CoordVecT`.
pub fn generate(
    comptime CoordVecT: type,
    context: anytype,
    comptime process_fn: anytype,
) GeneratorResult(CoordVecT, @TypeOf(context), process_fn, coordVectorLen(CoordVecT)) {
    return .{
        .context = context,
    };
}

// ============================================================================
// MARK: Input Accessor
// ============================================================================

/// Input accessor that provides neighborhood access for kernels.
/// Supports configurable vector length and uses the source's padding policy.
/// With `.unchecked` bounds, skips bounds checking for the interior fast path.
/// When a margin is provided, safe builds assert that every getAt offset
/// stays within it (the declared margin sizes the unchecked interior, so
/// out-of-margin reads are undefined behavior in unsafe builds).
pub fn InputAccessorGeneric(comptime SourceType: type, comptime VecT: type, comptime bounds: sources.BoundsCheck, comptime margin: ?Margin) type {
    const SourceInfo = sources.SourceTraits(SourceType);
    const ReturnType = if (SourceInfo.has_output_type) SourceType.OutputType else VecT;

    return struct {
        source: SourceType,
        current_x: i32,
        current_y: i32,

        const Self = @This();

        /// Get value at offset (for margin-based operations).
        /// The offset must stay within the loop's declared margin.
        pub inline fn getAt(self: Self, dx: i32, dy: i32) ReturnType {
            if (margin) |m| {
                std.debug.assert(dx >= -@as(i32, @intCast(m.left)) and dx <= @as(i32, @intCast(m.right)));
                std.debug.assert(dy >= -@as(i32, @intCast(m.top)) and dy <= @as(i32, @intCast(m.bottom)));
            }
            const x = self.current_x + dx;
            const y = self.current_y + dy;

            return switch (comptime SourceInfo.kind) {
                .eval => if (comptime bounds == .unchecked and SourceInfo.unchecked_kind == .eval)
                    self.source.evalAtUnchecked(x, y)
                else
                    self.source.evalAt(x, y),
                .read => if (comptime bounds == .unchecked and SourceInfo.unchecked_kind == .read)
                    self.source.readVecUnchecked(VecT, x, y)
                else
                    self.source.readVec(VecT, x, y),
            };
        }

        /// Get current value (no offset) - for identity operations
        pub inline fn get(self: Self) ReturnType {
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
    comptime SourceType: type,
    comptime ContextType: type,
    comptime process_fn: anytype,
    comptime opts: LoopOptions,
) type {
    const SourceInfo = sources.SourceTraits(SourceType);
    const has_coords = opts.coord_type != null;
    const is_zip_source = isZipSourceType(SourceType);
    const is_group_source = isGroupSourceType(SourceType);

    // VecT is the base vector type for the accessor
    const AccessorType = if (comptime is_zip_source)
        zip.ZipAccessor(SourceType, VecT, .checked)
    else if (comptime is_group_source)
        group.GroupAccessor(SourceType, VecT, SourceType.group_width, SourceType.group_height, .checked)
    else
        InputAccessorGeneric(SourceType, VecT, .checked, opts.margin);

    // Unchecked accessor for interior fast path
    const UncheckedAccessorType = if (comptime is_zip_source)
        zip.ZipAccessor(SourceType, VecT, .unchecked)
    else if (comptime is_group_source)
        group.GroupAccessor(SourceType, VecT, SourceType.group_width, SourceType.group_height, .unchecked)
    else
        InputAccessorGeneric(SourceType, VecT, .unchecked, opts.margin);

    const vec_len = @typeInfo(VecT).vector.len;

    // A chained source must produce vectors with the same lane count as VecT,
    // otherwise process() would advance positions by the wrong step.
    comptime assertSourceVectorLength(VecT, SourceType);

    // Get the actual return type from the process function
    const ReturnType = ProcessReturnType(process_fn);

    // Check if the source chain supports unchecked access
    const source_has_unchecked = SourceInfo.has_unchecked;

    return struct {
        source: SourceType,
        context: ContextType,
        region: Region,

        /// The output type of evalAt (may be VecT, [N]VecT, or tuple)
        pub const OutputType = ReturnType;

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        /// The margin used by this loop's kernel
        pub const margin = opts.margin;

        const Self = @This();

        /// Shared evaluation logic parameterized by accessor type.
        inline fn evalWith(self: Self, comptime Accessor: type, x: i32, y: i32) ReturnType {
            const accessor = Accessor{
                .source = self.source,
                .current_x = x,
                .current_y = y,
            };

            if (comptime has_coords) {
                const CoordVecT = opts.coord_type.?;
                const CoordScalarT = CoordScalarType(CoordVecT);
                if (comptime coordVectorLen(CoordVecT) != vec_len) {
                    @compileError("Loop coordinate vector " ++ @typeName(CoordVecT) ++ " must use the same lane count as " ++ @typeName(VecT));
                }
                const iota = std.simd.iota(CoordScalarT, vec_len);
                const x_coords: CoordVecT = iota + splatCoordScalar(CoordVecT, x);
                const y_coords: CoordVecT = splatCoordScalar(CoordVecT, y);
                return process_fn(self.context, accessor, x_coords, y_coords);
            } else {
                return process_fn(self.context, accessor);
            }
        }

        /// Evaluate at a specific position
        pub inline fn evalAt(self: Self, x: i32, y: i32) ReturnType {
            return self.evalWith(AccessorType, x, y);
        }

        /// Evaluate at a specific position without bounds checking.
        /// Caller MUST guarantee that all accessed positions (including margin offsets)
        /// are within the source's data region.
        pub inline fn evalAtUnchecked(self: Self, x: i32, y: i32) ReturnType {
            if (comptime !source_has_unchecked) {
                return self.evalAt(x, y);
            }
            return self.evalWith(UncheckedAccessorType, x, y);
        }

        /// Returns the region where evalAtUnchecked can be safely called.
        /// For position x in this region, all reads through the full source chain
        /// are guaranteed in-bounds.
        /// Returns an empty region if the margin is zero (no benefit from split iteration)
        /// or if the source doesn't support unchecked access.
        pub fn getInteriorRegion(self: Self) Region {
            if (comptime !source_has_unchecked or opts.margin.isZero()) {
                // No unchecked path available or no margin → no benefit from split
                return .{ .width = 0, .height = 0 };
            }

            // If the source is a chained source with its own interior region,
            // use that as the base. It already accounts for its own lane count
            // and deeper margins. We only need to deflate by our own kernel's
            // margin offsets.
            if (comptime SourceInfo.has_interior_region) {
                const src_interior = self.source.getInteriorRegion();
                if (src_interior.width == 0 or src_interior.height == 0) {
                    return .{ .width = 0, .height = 0 };
                }
                // Zip/group sources compute their interior for their own lane
                // count, but their accessors re-read nested sources at this
                // loop's lane count. When this loop reads wider vectors,
                // deflate the right edge by the difference.
                const src_vec_len = comptime getSourceVecLen(SourceType) orelse vec_len;
                const extra_right: i32 = comptime @max(0, vec_len - src_vec_len);
                return src_interior.deflated(
                    @intCast(opts.margin.left),
                    @intCast(opts.margin.top),
                    @as(i32, @intCast(opts.margin.right)) + extra_right,
                    @intCast(opts.margin.bottom),
                );
            }

            // Leaf source (InputSource): deflate self.region by margin + vec_len.
            // Right side needs margin.right + vec_len - 1 for the vector read width.
            const src_region = self.region;
            const right: i32 = @intCast(opts.margin.right);
            return src_region.deflated(
                @intCast(opts.margin.left),
                @intCast(opts.margin.top),
                right + vec_len - 1,
                @intCast(opts.margin.bottom),
            );
        }
    };
}

/// Create a lazy processing loop
pub fn loop(
    comptime VecT: type,
    comptime opts: LoopOptions,
    source: anytype,
    context: anytype,
    comptime process_fn: anytype,
) LoopResult(VecT, @TypeOf(source), @TypeOf(context), process_fn, opts) {
    comptime sources.assertIsSource(@TypeOf(source));
    comptime sources.assertSourceHasRegion(@TypeOf(source));
    return .{
        .source = source,
        .context = context,
        .region = source.region,
    };
}

// ============================================================================
// MARK: Process
// ============================================================================

/// Get compatible vector length.
/// In case of conversion the length must accommodate both source and destination.
fn getCompatibleVectorLen(comptime SourceType: type, comptime DestType: type) comptime_int {
    const SourceInfo = sources.SourceTraits(SourceType);
    comptime sources.assertDestHasInputScalarType(DestType);
    const suggested_source_len = std.simd.suggestVectorLength(SourceInfo.output_scalar_type) orelse 1;
    const suggested_dest_len = std.simd.suggestVectorLength(DestType.InputScalarType) orelse 1;
    return @max(suggested_source_len, suggested_dest_len);
}

/// Get vector length from a source type.
const getSourceVecLen = zip.getSourceVecLen;

/// Execute the processing pipeline and write results to output.
/// Supports both single-channel and multi-channel sources.
/// The destination region drives what gets computed (pull model).
/// The destination must have write() and writeScalar() methods matching the source output type.
/// `write()` handles full SIMD batches; `writeScalar()` is used for checked remainder
/// pixels when overlapping vector writes are not part of the destination contract.
///
/// When the source supports it (LoopResult with margins over an InputSource),
/// splits each row into left-edge / interior / right-edge strips. The interior
/// strip uses unchecked reads (no bounds checking), which eliminates per-vector
/// bounds comparisons for the majority of pixels.
pub fn process(source: anytype, dest: anytype) void {
    const SourceType = @TypeOf(source);
    const DestType = @TypeOf(dest);
    const SourceInfo = sources.SourceTraits(SourceType);
    const DestInfo = sources.DestTraits(DestType);
    comptime sources.assertIsSource(SourceType);
    comptime sources.assertIsDest(DestType);

    const region = dest.region;

    // Determine vector length from Source or from Source and Destination
    const vec_len = getSourceVecLen(SourceType) orelse getCompatibleVectorLen(SourceType, DestType);

    // Check if destination supports idempotent/overlapping writes (e.g., pixel buffers)
    // Accumulators like Stats do not support this and must use scalar remainder handling
    const supports_overlapping_writes = DestInfo.supports_overlapping_writes;

    // Check if source supports split iteration (unchecked interior path)
    const has_unchecked = SourceInfo.has_unchecked;
    const has_interior = SourceInfo.has_interior_region and has_unchecked;

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

const evalSourceChecked = sources.evalSourceChecked;
const evalSourceUnchecked = sources.evalSourceUnchecked;
const readSourceScalarChecked = sources.readSourceScalarChecked;

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
    const stop_x = region.stopX();
    for (0..region.height) |y| {
        const y_offset: i32 = @intCast(y);
        const y_coord = region.y + y_offset;
        processRowChecked(SourceType, DestType, source, dest, region.x, stop_x, y_coord, vec_len, supports_overlapping_writes);
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
    const dst_stop_x = region.stopX();

    // Interior X bounds (clamped to destination region)
    const int_x_start = @max(dst_x, interior.x);
    const int_x_stop = @min(dst_stop_x, interior.stopX());

    for (0..region.height) |y| {
        const y_offset: i32 = @intCast(y);
        const y_coord = region.y + y_offset;

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
                    dest.write(@intCast(x_coord), @intCast(y_coord), evalSourceUnchecked(SourceType, source, vec_len, x_coord, y_coord));
                }
                // Interior remainder (still unchecked, use overlapping write if possible)
                if (x_coord < int_x_stop) {
                    if (supports_overlapping_writes and int_x_stop - int_x_start >= vec_len) {
                        const aligned_x = int_x_stop - vec_len;
                        dest.write(@intCast(aligned_x), @intCast(y_coord), evalSourceUnchecked(SourceType, source, vec_len, aligned_x, y_coord));
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
        dest.write(@intCast(x_coord), @intCast(y_coord), evalSourceChecked(SourceType, source, vec_len, x_coord, y_coord));
    }

    // Handle remaining pixels
    if (x_coord < x_stop) {
        if (supports_overlapping_writes and width >= vec_len) {
            const aligned_x = x_stop - vec_len;
            dest.write(@intCast(aligned_x), @intCast(y_coord), evalSourceChecked(SourceType, source, vec_len, aligned_x, y_coord));
        } else {
            while (x_coord < x_stop) : (x_coord += 1) {
                dest.writeScalar(@intCast(x_coord), @intCast(y_coord), readSourceScalarChecked(SourceType, source, x_coord, y_coord));
            }
        }
    }
}

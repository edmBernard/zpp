//! Input and Output source/destination types for pixel processing.

const std = @import("std");
const Region = @import("region.zig").Region;
const Margin = @import("region.zig").Margin;
const padding = @import("padding.zig");

pub const RepeatEdgePadding = padding.RepeatEdgePadding;

// ============================================================================
// MARK: Source Tags
// ============================================================================

/// Tag to identify composite source types.
/// Used instead of marker booleans for type identification.
pub const SourceTag = enum {
    zip,
    group,
};

/// Check if a type has a specific source tag.
pub fn hasSourceTag(comptime T: type, comptime tag: SourceTag) bool {
    if (@hasDecl(T, "source_tag")) {
        return T.source_tag == tag;
    }
    return false;
}

// ============================================================================
// MARK: Traits
// ============================================================================

pub const SourceAccessKind = enum {
    eval,
    read,
};

/// Whether an accessor may skip bounds checking.
/// `.unchecked` is only valid on the interior fast path, where the caller
/// guarantees every covered position is in-bounds; sources without an
/// unchecked entry point fall back to checked reads.
pub const BoundsCheck = enum {
    checked,
    unchecked,
};

fn sourceContractError(comptime T: type, comptime detail: []const u8) noreturn {
    @compileError(@typeName(T) ++ " does not satisfy the Source interface: " ++ detail);
}

fn destContractError(comptime T: type, comptime detail: []const u8) noreturn {
    @compileError(@typeName(T) ++ " does not satisfy the Destination interface: " ++ detail);
}

/// Centralized comptime traits for source types.
pub fn SourceTraits(comptime T: type) type {
    const source_has_eval = @hasDecl(T, "evalAt");
    const source_has_read_vec = @hasDecl(T, "readVec");
    const source_kind: SourceAccessKind = if (source_has_eval)
        .eval
    else if (source_has_read_vec)
        .read
    else
        sourceContractError(T, "must implement evalAt() or readVec()");

    const source_has_output_scalar_type = @hasDecl(T, "OutputScalarType");
    if (source_kind == .read and !source_has_output_scalar_type) {
        sourceContractError(T, "readVec()-based sources must declare OutputScalarType");
    }

    const source_unchecked_kind: ?SourceAccessKind = if (@hasDecl(T, "evalAtUnchecked"))
        .eval
    else if (@hasDecl(T, "readVecUnchecked"))
        .read
    else
        null;

    return struct {
        pub const kind = source_kind;
        pub const unchecked_kind = source_unchecked_kind;
        pub const has_unchecked = source_unchecked_kind != null;
        pub const has_eval = source_has_eval;
        pub const has_read_vec = source_has_read_vec;
        pub const has_read_scalar = @hasDecl(T, "read");
        pub const has_region = @hasField(T, "region");
        pub const has_output_type = @hasDecl(T, "OutputType");
        pub const has_output_scalar_type = source_has_output_scalar_type;
        pub const has_vector_length = @hasDecl(T, "vector_length");
        pub const has_margin = @hasDecl(T, "margin");
        pub const has_interior_region = @hasDecl(T, "getInteriorRegion");
        pub const output_scalar_type = if (source_has_output_scalar_type) T.OutputScalarType else void;
        pub const output_type = if (@hasDecl(T, "OutputType")) T.OutputType else void;
    };
}

/// Centralized comptime traits for destination types.
pub fn DestTraits(comptime T: type) type {
    const dest_has_write = @hasDecl(T, "write");
    if (!dest_has_write) {
        destContractError(T, "must implement write()");
    }

    const dest_has_write_scalar = @hasDecl(T, "writeScalar");
    if (!dest_has_write_scalar) {
        destContractError(T, "must implement writeScalar()");
    }

    const dest_has_region = @hasField(T, "region");
    if (!dest_has_region) {
        destContractError(T, "must have a region field");
    }

    return struct {
        pub const has_write = dest_has_write;
        pub const has_write_scalar = dest_has_write_scalar;
        pub const has_region = dest_has_region;
        pub const has_input_scalar_type = @hasDecl(T, "InputScalarType");
        pub const supports_overlapping_writes = @hasDecl(T, "supports_overlapping_writes") and T.supports_overlapping_writes;
    };
}

// ============================================================================
// MARK: Interface Validation
// ============================================================================

/// Validate that a type satisfies the Source interface.
/// A Source must provide either `evalAt` or `readVec` for reading data.
pub fn assertIsSource(comptime T: type) void {
    _ = SourceTraits(T);
}

/// Validate that a source is region-addressable.
pub fn assertSourceHasRegion(comptime T: type) void {
    if (!SourceTraits(T).has_region) {
        sourceContractError(T, "must have a region field for this operation");
    }
}

/// Validate that a type satisfies the Destination interface.
/// A Destination must provide `write` and `writeScalar` methods and a `region` field.
pub fn assertIsDest(comptime T: type) void {
    _ = DestTraits(T);
}

/// Validate that a destination exposes an input scalar type.
pub fn assertDestHasInputScalarType(comptime T: type) void {
    if (!DestTraits(T).has_input_scalar_type) {
        destContractError(T, "must declare InputScalarType for vector-length inference");
    }
}

// ============================================================================
// MARK: Source Helpers
// ============================================================================

/// Return type for evalSourceChecked/evalSourceUnchecked.
fn EvalReturnType(comptime SourceType: type, comptime vec_len: comptime_int, comptime decl_name: []const u8) type {
    const Traits = SourceTraits(SourceType);
    if (@hasDecl(SourceType, decl_name)) {
        const fn_info = @typeInfo(@TypeOf(@field(SourceType, decl_name)));
        return fn_info.@"fn".return_type.?;
    } else {
        return @Vector(vec_len, Traits.output_scalar_type);
    }
}

/// The value type a source produces for the requested vector length.
pub fn SourceValueType(comptime SourceType: type, comptime vec_len: comptime_int) type {
    return EvalReturnType(SourceType, vec_len, "evalAt");
}

fn FirstLaneTupleType(comptime T: type) type {
    const fields = @typeInfo(T).@"struct".fields;
    var types: [fields.len]type = undefined;
    inline for (fields, 0..) |field, i| {
        types[i] = FirstLaneType(field.type);
    }
    return std.meta.Tuple(&types);
}

/// Scalar type produced by taking the first lane from a source result.
fn FirstLaneType(comptime T: type) type {
    const info = @typeInfo(T);
    return switch (info) {
        .vector => |v| v.child,
        .array => |a| [a.len]FirstLaneType(a.child),
        .@"struct" => |s| if (s.is_tuple) FirstLaneTupleType(T) else @compileError("Unsupported struct type for FirstLaneType: " ++ @typeName(T)),
        else => @compileError("Unsupported type for FirstLaneType: " ++ @typeName(T)),
    };
}

/// Extract lane 0 from a vector-like source result.
pub fn firstLane(result: anytype) FirstLaneType(@TypeOf(result)) {
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
            if (!s.is_tuple) {
                @compileError("Unsupported struct type for firstLane: " ++ @typeName(T));
            }
            var out: FirstLaneType(T) = undefined;
            inline for (0..s.fields.len) |i| {
                out[i] = firstLane(result[i]);
            }
            return out;
        },
        else => @compileError("Unsupported type for firstLane: expected vector, array, or struct tuple"),
    }
}

/// The scalar value type a source produces when only lane 0 is requested.
fn SourceScalarValueType(comptime SourceType: type) type {
    const Traits = SourceTraits(SourceType);
    return switch (comptime Traits.kind) {
        .eval => FirstLaneType(EvalReturnType(SourceType, 1, "evalAt")),
        .read => Traits.output_scalar_type,
    };
}

/// Evaluate a source at position (x, y) using checked reads.
/// This is the vector-native helper. Scalar callers should use readSourceScalarChecked.
pub inline fn evalSourceChecked(comptime SourceType: type, source: SourceType, comptime vec_len: comptime_int, x: i32, y: i32) EvalReturnType(SourceType, vec_len, "evalAt") {
    const Traits = SourceTraits(SourceType);
    return switch (comptime Traits.kind) {
        .eval => source.evalAt(x, y),
        .read => source.readVec(@Vector(vec_len, Traits.output_scalar_type), x, y),
    };
}

/// Evaluate a source at position (x, y) using unchecked reads (no bounds checking).
/// This is the vector-native helper. Scalar callers should use readSourceScalarUnchecked.
pub inline fn evalSourceUnchecked(comptime SourceType: type, source: SourceType, comptime vec_len: comptime_int, x: i32, y: i32) EvalReturnType(SourceType, vec_len, "evalAtUnchecked") {
    const Traits = SourceTraits(SourceType);
    return switch (comptime Traits.unchecked_kind orelse sourceContractError(SourceType, "must implement evalAtUnchecked() or readVecUnchecked() to support unchecked access")) {
        .eval => source.evalAtUnchecked(x, y),
        .read => source.readVecUnchecked(@Vector(vec_len, Traits.output_scalar_type), x, y),
    };
}

/// Read a single scalar value from a source using checked access.
pub inline fn readSourceScalarChecked(comptime SourceType: type, source: SourceType, x: i32, y: i32) SourceScalarValueType(SourceType) {
    const Traits = SourceTraits(SourceType);
    return switch (comptime Traits.kind) {
        .eval => firstLane(source.evalAt(x, y)),
        .read => if (comptime Traits.has_read_scalar)
            source.read(x, y)
        else
            firstLane(source.readVec(@Vector(1, Traits.output_scalar_type), x, y)),
    };
}

/// Read a single scalar value from a source using unchecked access.
pub inline fn readSourceScalarUnchecked(comptime SourceType: type, source: SourceType, x: i32, y: i32) SourceScalarValueType(SourceType) {
    const Traits = SourceTraits(SourceType);
    return switch (comptime Traits.unchecked_kind orelse sourceContractError(SourceType, "must implement evalAtUnchecked() or readVecUnchecked() to support unchecked access")) {
        .eval => firstLane(source.evalAtUnchecked(x, y)),
        .read => firstLane(source.readVecUnchecked(@Vector(1, Traits.output_scalar_type), x, y)),
    };
}

fn validatePackedLayout(data_len: usize, stride: u32, region: Region) !void {
    if (region.area() == 0) return;
    if (region.x < 0 or region.y < 0) return error.NegativeRegionOrigin;

    const stop_x = @as(u64, @intCast(region.x)) + region.width;
    if (stop_x > stride) return error.StrideTooSmall;

    const stop_y = @as(u64, @intCast(region.y)) + region.height;
    const required_len = (stop_y - 1) * @as(u64, stride) + stop_x;
    if (required_len > data_len) return error.BufferTooSmall;
}

fn validateInterleavedLayout(comptime num_channels: comptime_int, data_len: usize, width: u32, region: Region) !void {
    if (num_channels <= 0) {
        @compileError("Interleaved destinations must use at least one channel");
    }
    if (region.area() == 0) return;
    if (region.x < 0 or region.y < 0) return error.NegativeRegionOrigin;

    const stop_x = @as(u64, @intCast(region.x)) + region.width;
    if (stop_x > width) return error.StrideTooSmall;

    const stop_y = @as(u64, @intCast(region.y)) + region.height;
    const pixels_needed = (stop_y - 1) * @as(u64, width) + stop_x;
    const required_len = pixels_needed * @as(u64, @intCast(num_channels));
    if (required_len > data_len) return error.BufferTooSmall;
}

// ============================================================================
// MARK: Input Sources
// ============================================================================

/// Input source wrapper - provides access to input data with optional neighborhood access
/// Supports padding policies for out-of-bounds access.
pub fn InputSource(comptime T: type, comptime PaddingPolicy: type) type {
    return struct {
        data: []const T,
        stride: u32,
        region: Region,

        pub const OutputScalarType = T;
        /// The padding policy used for out-of-bounds reads. Exposed so that
        /// consumers (e.g. PixelInterpolator) can use vectorized clamp/mask
        /// gathers instead of per-pixel checked reads.
        pub const PaddingPolicyType = PaddingPolicy;
        const Self = @This();

        /// Read a value at the given position, applying padding policy for out-of-bounds.
        pub fn read(self: Self, x: i32, y: i32) T {
            // In-region reads are always in-buffer: makeSource validates the layout.
            if (x >= self.region.x and x < self.region.stopX() and
                y >= self.region.y and y < self.region.stopY())
            {
                const ux: u32 = @intCast(x);
                const uy: u32 = @intCast(y);
                return self.data[uy * self.stride + ux];
            }
            // Apply padding policy for out-of-bounds access
            return PaddingPolicy.apply(T, self.data, self.stride, x, y, self.region);
        }

        /// Read a vector of values starting at the given position (horizontal SIMD load).
        /// Returns vec_len values: [data[x], data[x+1], ..., data[x+vec_len-1]]
        /// When all indices are in-bounds, performs a direct SIMD load from memory.
        /// When out-of-bounds, uses vectorized coordinate clamping where possible.
        pub fn readVec(self: Self, comptime VecT: type, x: i32, y: i32) VecT {
            const vec_len = @typeInfo(VecT).vector.len;

            // Fast path: all indices are within bounds
            if (x >= self.region.x and
                x + vec_len <= self.region.stopX() and
                y >= self.region.y and y < self.region.stopY())
            {
                return self.readVecUnchecked(VecT, x, y);
            }

            // Vectorized path: y in bounds, x straddles boundary
            if (y >= self.region.y and y < self.region.stopY() and
                self.region.width > 0)
            {
                const x_vec = std.simd.iota(i32, vec_len) + @as(@Vector(vec_len, i32), @splat(x));
                const clamped_x = PaddingPolicy.clampX(vec_len, x_vec, self.region);
                const uy: u32 = @intCast(y);
                const base = uy * self.stride;
                var result: VecT = undefined;
                inline for (0..vec_len) |i| {
                    result[i] = self.data[base + @as(u32, @intCast(clamped_x[i]))];
                }
                if (PaddingPolicy.needs_mask) {
                    // Zero out-of-bounds lanes (for ZeroPadding)
                    const mask = PaddingPolicy.inBoundsX(vec_len, x_vec, self.region);
                    const zero: VecT = @splat(0);
                    result = @select(T, mask, result, zero);
                }
                return result;
            }

            // Slow path: y out of bounds (rare — only at top/bottom edges)
            var result: VecT = undefined;
            inline for (0..vec_len) |i| {
                result[i] = self.read(x + @as(i32, @intCast(i)), y);
            }
            return result;
        }

        /// Read a vector of values without bounds checking.
        /// Caller MUST guarantee all vec_len elements are within the source region.
        pub inline fn readVecUnchecked(self: Self, comptime VecT: type, x: i32, y: i32) VecT {
            const vec_len = @typeInfo(VecT).vector.len;
            const ux: u32 = @intCast(x);
            const uy: u32 = @intCast(y);
            const start_idx = uy * self.stride + ux;
            return self.data[start_idx..][0..vec_len].*;
        }
    };
}

/// Create an input source from data buffer with default RepeatEdgePadding
pub fn makeSource(comptime T: type, data: []const T, stride: u32, region: Region) !InputSource(T, RepeatEdgePadding) {
    try validatePackedLayout(data.len, stride, region);
    return .{
        .data = data,
        .stride = stride,
        .region = region,
    };
}

/// Create an input source from data buffer with specified padding policy
pub fn makePaddedSource(
    comptime T: type,
    comptime PaddingPolicy: type,
    data: []const T,
    stride: u32,
    region: Region,
) !InputSource(T, PaddingPolicy) {
    try validatePackedLayout(data.len, stride, region);
    return .{
        .data = data,
        .stride = stride,
        .region = region,
    };
}

// ============================================================================
// MARK: Output Destinations
// ============================================================================

/// Output destination wrapper
pub fn OutputDest(comptime T: type) type {
    return struct {
        data: []T,
        stride: u32,
        region: Region,

        pub const InputScalarType = T;
        const Self = @This();

        // TODO: Add heavy testing for this property espectially with Group and Zip
        /// Writes to pixel buffers are idempotent (overwriting same pixel is safe)
        pub const supports_overlapping_writes = true;

        pub fn write(self: Self, x: u32, y: u32, values: anytype) void {
            const VecType = @TypeOf(values);
            const vec_len = @typeInfo(VecType).vector.len;
            const base_idx = y * self.stride + x;
            // Process guarantees x + vec_len <= region.width, so direct SIMD store is safe.
            // The slice pointer cast enables a single vector store instruction.
            const dest: *[vec_len]T = @ptrCast(self.data[base_idx..][0..vec_len]);
            dest.* = values;
        }

        /// Write a single scalar value (for remainder handling)
        pub fn writeScalar(self: Self, x: u32, y: u32, value: T) void {
            const idx = y * self.stride + x;
            if (idx < self.data.len) {
                self.data[idx] = value;
            }
        }
    };
}

/// Create an output destination from data buffer
pub fn makeDest(comptime T: type, data: []T, stride: u32, region: Region) !OutputDest(T) {
    try validatePackedLayout(data.len, stride, region);
    return .{
        .data = data,
        .stride = stride,
        .region = region,
    };
}

// ============================================================================
// MARK: Interleaved Output
// ============================================================================

/// Interleaved multi-channel output destination (e.g., RGBRGBRGB... for 3 channels)
/// Generic over element type T and number of channels.
/// No automatic conversion is performed - values are written directly.
pub fn InterleavedOutput(comptime T: type, comptime num_channels: comptime_int) type {
    return struct {
        data: []T,
        width: u32,
        region: Region,

        pub const InputScalarType = T;
        const Self = @This();

        /// Writes to pixel buffers are idempotent (overwriting same pixel is safe)
        pub const supports_overlapping_writes = true;

        /// Write a SIMD batch of multi-channel pixels at position (x, y)
        /// Takes an array of vectors (one per channel), interlaces them, and writes directly.
        pub fn write(self: Self, x: u32, y: u32, values: anytype) void {
            const ValuesType = @TypeOf(values);
            const values_type_info = @typeInfo(ValuesType);
            if (comptime values_type_info != .array) {
                @compileError("InterleavedOutput.write expects an array of vectors");
            }
            if (comptime values_type_info.array.len != num_channels) {
                @compileError("InterleavedOutput.write expects exactly " ++ std.fmt.comptimePrint("{}", .{num_channels}) ++ " channels");
            }
            const VecType = values_type_info.array.child;
            const vec_info = @typeInfo(VecType);
            if (comptime vec_info != .vector) {
                @compileError("InterleavedOutput.write expects an array of vectors");
            }
            if (comptime vec_info.vector.child != T) {
                @compileError("InterleavedOutput.write expects vectors of type " ++ @typeName(T));
            }
            const vec_len = vec_info.vector.len;

            // Interlace the channels using SIMD
            const interlaced = std.simd.interlace(values);

            // Write to output buffer
            const offset = y * self.width * num_channels + x * num_channels;
            const dest = self.data[offset..][0 .. vec_len * num_channels];
            dest.* = interlaced;
        }

        /// Write a single multi-channel pixel (for remainder handling)
        pub fn writeScalar(self: Self, x: u32, y: u32, values: anytype) void {
            const offset = y * self.width * num_channels + x * num_channels;
            inline for (0..num_channels) |c| {
                self.data[offset + c] = values[c];
            }
        }
    };
}

/// Create an interleaved multi-channel output destination
pub fn makeInterleavedDest(
    comptime T: type,
    comptime num_channels: comptime_int,
    data: []T,
    width: u32,
    region: Region,
) !InterleavedOutput(T, num_channels) {
    try validateInterleavedLayout(num_channels, data.len, width, region);
    return .{
        .data = data,
        .width = width,
        .region = region,
    };
}

test "scalar helpers read checked and unchecked values from read-based sources" {
    const region: Region = .{ .x = 0, .y = 0, .width = 4, .height = 2 };
    var input_data: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const source = try makeSource(f32, &input_data, region.width, region);

    try std.testing.expectEqual(@as(f32, 1), readSourceScalarChecked(@TypeOf(source), source, -2, 0));
    try std.testing.expectEqual(@as(f32, 6), readSourceScalarChecked(@TypeOf(source), source, 1, 1));
    try std.testing.expectEqual(@as(f32, 7), readSourceScalarUnchecked(@TypeOf(source), source, 2, 1));
}

test "scalar helpers extract lane 0 from eval-based sources" {
    const EvalSource = struct {
        fn valueAt(x: i32, y: i32) @Vector(4, f32) {
            const base = @as(f32, @floatFromInt(x + y * 4));
            return .{ base, base + 1, base + 2, base + 3 };
        }

        pub fn evalAt(self: @This(), x: i32, y: i32) @Vector(4, f32) {
            _ = self;
            return valueAt(x, y);
        }

        pub fn evalAtUnchecked(self: @This(), x: i32, y: i32) @Vector(4, f32) {
            _ = self;
            return valueAt(x, y);
        }
    };

    const source = EvalSource{};
    try std.testing.expectEqual(@as(f32, 6), readSourceScalarChecked(@TypeOf(source), source, 2, 1));
    try std.testing.expectEqual(@as(f32, 6), readSourceScalarUnchecked(@TypeOf(source), source, 2, 1));
}

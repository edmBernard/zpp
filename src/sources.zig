//! Input and Output source/destination types for pixel processing.

const std = @import("std");
const region_mod = @import("region.zig");
const padding_mod = @import("padding.zig");

const Region = region_mod.Region;
pub const RepeatEdgePadding = padding_mod.RepeatEdgePadding;

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

/// Tag to identify composite destination types.
pub const DestTag = enum {
    zip,
};

/// Check if a type has a specific destination tag.
pub fn hasDestTag(comptime T: type, comptime tag: DestTag) bool {
    if (@hasDecl(T, "dest_tag")) {
        return T.dest_tag == tag;
    }
    return false;
}

// ============================================================================
// MARK: Interface Validation
// ============================================================================

/// Validate that a type satisfies the Source interface.
/// A Source must provide either `evalAt` or `readVec` for reading data.
pub fn assertIsSource(comptime T: type) void {
    const has_eval = @hasDecl(T, "evalAt");
    const has_read = @hasDecl(T, "readVec");
    if (!has_eval and !has_read) {
        @compileError(@typeName(T) ++ " does not satisfy the Source interface: must implement evalAt() or readVec()");
    }
}

/// Validate that a type satisfies the Destination interface.
/// A Destination must provide `write` and `writeScalar` methods,
/// an `InputScalarType` declaration, and a `region` field.
pub fn assertIsDest(comptime T: type) void {
    if (!@hasDecl(T, "write")) {
        @compileError(@typeName(T) ++ " does not satisfy the Destination interface: must implement write()");
    }
    if (!@hasDecl(T, "writeScalar")) {
        @compileError(@typeName(T) ++ " does not satisfy the Destination interface: must implement writeScalar()");
    }
    if (!@hasField(T, "region")) {
        @compileError(@typeName(T) ++ " does not satisfy the Destination interface: must have a region field");
    }
}

// ============================================================================
// MARK: Source Helpers
// ============================================================================

/// Get the region from any source (has getRegion method or region field).
pub fn getSourceRegion(source: anytype) Region {
    const T = @TypeOf(source);
    if (@hasDecl(T, "getRegion")) {
        return source.getRegion();
    } else if (@hasField(T, "region")) {
        return source.region;
    } else {
        @compileError("Source must have a region field or getRegion method");
    }
}

/// Return type for evalSourceChecked/evalSourceUnchecked.
fn EvalReturnType(comptime SourceType: type, comptime vec_len: comptime_int, comptime decl_name: []const u8) type {
    if (@hasDecl(SourceType, decl_name)) {
        const fn_info = @typeInfo(@TypeOf(@field(SourceType, decl_name)));
        return fn_info.@"fn".return_type.?;
    } else {
        return @Vector(vec_len, SourceType.OutputScalarType);
    }
}

/// Evaluate a source at position (x, y) using checked reads.
pub inline fn evalSourceChecked(comptime SourceType: type, source: SourceType, comptime vec_len: comptime_int, x: i32, y: i32) EvalReturnType(SourceType, vec_len, "evalAt") {
    if (@hasDecl(SourceType, "evalAt")) {
        return source.evalAt(x, y);
    } else if (@hasDecl(SourceType, "readVec")) {
        return source.readVec(@Vector(vec_len, SourceType.OutputScalarType), x, y);
    } else {
        @compileError("Source must have evalAt or readVec method");
    }
}

/// Evaluate a source at position (x, y) using unchecked reads (no bounds checking).
pub inline fn evalSourceUnchecked(comptime SourceType: type, source: SourceType, comptime vec_len: comptime_int, x: i32, y: i32) EvalReturnType(SourceType, vec_len, "evalAtUnchecked") {
    if (@hasDecl(SourceType, "evalAtUnchecked")) {
        return source.evalAtUnchecked(x, y);
    } else if (@hasDecl(SourceType, "readVecUnchecked")) {
        return source.readVecUnchecked(@Vector(vec_len, SourceType.OutputScalarType), x, y);
    } else {
        @compileError("Source must have evalAtUnchecked or readVecUnchecked for split iteration");
    }
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
        const Self = @This();

        /// Read a value at the given position, applying padding policy for out-of-bounds.
        pub fn read(self: Self, x: i32, y: i32) T {
            // Check if within valid region
            if (x >= self.region.x and x < self.region.stopX() and
                y >= self.region.y and y < self.region.stopY())
            {
                const ux: u32 = @intCast(x);
                const uy: u32 = @intCast(y);
                const idx = uy * self.stride + ux;
                if (idx < self.data.len) {
                    return self.data[idx];
                }
            }
            // Apply padding policy for out-of-bounds access
            return PaddingPolicy.apply(T, self.data, self.stride, x, y, self.region);
        }

        /// Read a vector of values starting at the given position (horizontal SIMD load).
        /// Returns vec_len values: [data[x], data[x+1], ..., data[x+vec_len-1]]
        /// When all indices are in-bounds, performs a direct SIMD load from memory.
        /// When out-of-bounds, falls back to element-by-element padding application.
        pub fn readVec(self: Self, comptime VecT: type, x: i32, y: i32) VecT {
            const vec_len = @typeInfo(VecT).vector.len;

            // Fast path: all indices are within bounds
            if (x >= self.region.x and
                x + vec_len <= self.region.stopX() and
                y >= self.region.y and y < self.region.stopY())
            {
                return self.readVecUnchecked(VecT, x, y);
            }

            // TODO: Implement vectorized padding
            // Slow path: apply padding policy element-by-element
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
pub fn makeSource(comptime T: type, data: []const T, stride: u32, region: Region) InputSource(T, RepeatEdgePadding) {
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
) InputSource(T, PaddingPolicy) {
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
pub fn makeDest(comptime T: type, data: []T, stride: u32, region: Region) OutputDest(T) {
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
) InterleavedOutput(T, num_channels) {
    return .{
        .data = data,
        .width = width,
        .region = region,
    };
}

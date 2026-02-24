//! Input and Output source/destination types for pixel processing.

const std = @import("std");
const region_mod = @import("region.zig");
const padding_mod = @import("padding.zig");

const Region = region_mod.Region;
pub const RepeatEdgePadding = padding_mod.RepeatEdgePadding;

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
pub fn In(comptime T: type, data: []const T, stride: u32, region: Region) InputSource(T, RepeatEdgePadding) {
    return .{
        .data = data,
        .stride = stride,
        .region = region,
    };
}

/// Create an input source from data buffer with specified padding policy
pub fn InWithPadding(
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
pub fn Out(comptime T: type, data: []T, stride: u32, region: Region) OutputDest(T) {
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
pub fn InterleavedOut(
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

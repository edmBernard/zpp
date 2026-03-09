//! Group and Ungroup support for Bayer patterns and P x Q pixel grouping.

const std = @import("std");
const sources = @import("sources.zig");
const zip = @import("zip.zig");
const Region = @import("region.zig").Region;
const VecTuple = zip.VecTuple;

fn deriveSourceVecLen(comptime SourceType: type) comptime_int {
    if (zip.getSourceVecLen(SourceType)) |len| {
        return len;
    }

    const Traits = sources.SourceTraits(SourceType);
    if (Traits.has_output_scalar_type) {
        return std.simd.suggestVectorLength(Traits.output_scalar_type) orelse 1;
    }

    @compileError(@typeName(SourceType) ++ " does not expose enough type information to infer a group vector length");
}

fn requireVectorValueType(comptime SourceType: type, comptime vec_len: comptime_int) type {
    const ValueT = sources.SourceValueType(SourceType, vec_len);
    if (@typeInfo(ValueT) != .vector) {
        @compileError(@typeName(SourceType) ++ " must evaluate to a vector type to be used with group/ungroup");
    }
    return ValueT;
}

// ============================================================================
// MARK: Group Source
// ============================================================================

/// A grouped source that combines PxQ pixels from a nested source into a single "pixel".
/// This is useful for Bayer pattern processing where you want to treat 2x2 blocks as units.
pub fn GroupSource(comptime NestedSource: type, comptime P: comptime_int, comptime Q: comptime_int) type {
    const NestedInfo = sources.SourceTraits(NestedSource);
    const vec_len = deriveSourceVecLen(NestedSource);
    const VecT = requireVectorValueType(NestedSource, vec_len);
    const ResultTuple = VecTuple(P * Q, VecT);
    const ElemT = @typeInfo(VecT).vector.child;
    const WideVecT = @Vector(P * vec_len, ElemT);

    return struct {
        nested: NestedSource,
        region: Region,

        pub const source_tag = sources.SourceTag.group;

        /// The group dimensions
        pub const group_width = P;
        pub const group_height = Q;
        pub const count = P * Q;

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        /// The vector type used by this GroupSource
        pub const VectorType = VecT;
        pub const OutputType = ResultTuple;

        const Self = @This();

        /// Evaluate at a grouped position - returns a tuple of PxQ values
        pub inline fn evalAt(self: Self, x: i32, y: i32) ResultTuple {
            var result: ResultTuple = undefined;

            // Map grouped coordinates to nested coordinates
            const nested_x = x * P;
            const nested_y = y * Q;

            switch (comptime NestedInfo.kind) {
                .eval => inline for (0..Q) |dy| {
                    inline for (0..P) |dx| {
                        const idx = dy * P + dx;
                        result[idx] = self.nested.evalAt(nested_x + @as(i32, @intCast(dx)), nested_y + @as(i32, @intCast(dy)));
                    }
                },
                .read => inline for (0..Q) |dy| {
                    const row_data = self.nested.readVec(WideVecT, nested_x, nested_y + @as(i32, @intCast(dy)));
                    const deinterlaced = std.simd.deinterlace(P, row_data);
                    inline for (0..P) |dx| {
                        result[dy * P + dx] = deinterlaced[dx];
                    }
                },
            }

            return result;
        }
    };
}

/// Group a source expression, treating PxQ blocks as single pixels.
/// The output region is downscaled by P horizontally and Q vertically.
pub fn group(comptime P: comptime_int, comptime Q: comptime_int, source: anytype) GroupSource(@TypeOf(source), P, Q) {
    comptime if (P <= 0 or Q <= 0) {
        @compileError("Group dimensions must be positive");
    };
    comptime sources.assertIsSource(@TypeOf(source));
    comptime sources.assertSourceHasRegion(@TypeOf(source));
    const nested_region = source.region;

    // The grouped region is downscaled
    const grouped_region = nested_region.downscaled(P, Q);

    return .{
        .nested = source,
        .region = grouped_region,
    };
}

// ============================================================================
// MARK: Ungroup Source
// ============================================================================

/// An ungrouped source that extracts individual pixels from a grouped source.
/// Reverses the grouping operation.
pub fn UngroupSource(comptime GroupedSource: type, comptime P: comptime_int, comptime Q: comptime_int) type {
    const vec_len = deriveSourceVecLen(GroupedSource);
    const VecT = GroupedSource.VectorType;

    return struct {
        grouped: GroupedSource,
        region: Region,

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;
        pub const OutputType = VecT;

        const Self = @This();

        /// Evaluate at an ungrouped position
        pub inline fn evalAt(self: Self, x: i32, y: i32) VecT {
            // Map ungrouped coordinates to grouped coordinates
            const group_x = @divFloor(x, P);
            const group_y = @divFloor(y, Q);
            const local_x: usize = @intCast(@mod(x, P));
            const local_y: usize = @intCast(@mod(y, Q));

            // Get the grouped values
            const grouped_values = self.grouped.evalAt(group_x, group_y);

            // Extract the appropriate pixel
            inline for (0..Q) |dy| {
                inline for (0..P) |dx| {
                    if (local_x == dx and local_y == dy) {
                        return grouped_values[dy * P + dx];
                    }
                }
            }
            unreachable;
        }
    };
}

/// Ungroup a grouped source expression back to individual pixels.
/// The output region is upscaled by P horizontally and Q vertically.
pub fn ungroup(comptime P: comptime_int, comptime Q: comptime_int, source: anytype) UngroupSource(@TypeOf(source), P, Q) {
    comptime if (P <= 0 or Q <= 0) {
        @compileError("Ungroup dimensions must be positive");
    };
    comptime sources.assertIsSource(@TypeOf(source));
    comptime sources.assertSourceHasRegion(@TypeOf(source));
    const grouped_region = source.region;

    // The ungrouped region is upscaled
    const ungrouped_region = grouped_region.upscaled(P, Q);

    return .{
        .grouped = source,
        .region = ungrouped_region,
    };
}

// ============================================================================
// MARK: Group Destination
// ============================================================================

/// A grouped destination that writes PxQ pixel blocks to a nested destination.
/// This is the destination counterpart to GroupSource.
pub fn GroupDest(comptime NestedDest: type, comptime P: comptime_int, comptime Q: comptime_int) type {
    const NestedDestInfo = sources.DestTraits(NestedDest);
    return struct {
        nested: NestedDest,
        region: Region,

        pub const InputScalarType = if (NestedDestInfo.has_input_scalar_type) NestedDest.InputScalarType else void;

        /// The group dimensions
        pub const group_width = P;
        pub const group_height = Q;
        pub const count = P * Q;

        /// Writes are idempotent if the nested destination supports overlapping writes.
        pub const supports_overlapping_writes = NestedDestInfo.supports_overlapping_writes;

        const Self = @This();

        /// Write values to a grouped position
        /// values should be a tuple of PxQ vectors
        pub fn write(self: Self, x: u32, y: u32, values: anytype) void {
            // Map grouped coordinates to nested coordinates
            const nested_x = x * P;
            const nested_y = y * Q;

            inline for (0..Q) |dy| {
                inline for (0..P) |dx| {
                    const idx = dy * P + dx;
                    self.nested.write(
                        nested_x + @as(u32, @intCast(dx)),
                        nested_y + @as(u32, @intCast(dy)),
                        values[idx],
                    );
                }
            }
        }

        /// Write scalar values to a grouped position
        pub fn writeScalar(self: Self, x: u32, y: u32, values: anytype) void {
            // Map grouped coordinates to nested coordinates
            const nested_x = x * P;
            const nested_y = y * Q;

            inline for (0..Q) |dy| {
                inline for (0..P) |dx| {
                    const idx = dy * P + dx;
                    self.nested.writeScalar(
                        nested_x + @as(u32, @intCast(dx)),
                        nested_y + @as(u32, @intCast(dy)),
                        values[idx],
                    );
                }
            }
        }
    };
}

/// Group a destination expression, treating PxQ blocks as single output pixels.
/// The input region is downscaled by P horizontally and Q vertically.
pub fn groupDest(comptime P: comptime_int, comptime Q: comptime_int, dest: anytype) GroupDest(@TypeOf(dest), P, Q) {
    comptime if (P <= 0 or Q <= 0) {
        @compileError("Group destination dimensions must be positive");
    };
    comptime sources.assertIsDest(@TypeOf(dest));
    const nested_region = dest.region;

    // The grouped region is downscaled
    const grouped_region = nested_region.downscaled(P, Q);

    return .{
        .nested = dest,
        .region = grouped_region,
    };
}

// ============================================================================
// MARK: Group Accessor
// ============================================================================

/// Accessor for grouped pixels in kernel functions.
/// Provides access to PxQ pixel groups as tuples.
pub fn GroupAccessor(comptime SrcType: type, comptime VecT: type, comptime P: comptime_int, comptime Q: comptime_int) type {
    const NestedType = @TypeOf(@as(SrcType, undefined).nested);
    const NestedInfo = sources.SourceTraits(NestedType);
    const vec_len = @typeInfo(VecT).vector.len;
    const ValueT = requireVectorValueType(NestedType, vec_len);
    const ResultTuple = VecTuple(P * Q, ValueT);
    const ElemT = @typeInfo(ValueT).vector.child;
    // Wide vector for reading P pixels at once per row
    const WideVecT = @Vector(P * vec_len, ElemT);

    return struct {
        source: SrcType,
        current_x: i32,
        current_y: i32,

        const Self = @This();

        /// Get all PxQ values at the current position as a tuple
        pub inline fn get(self: Self) ResultTuple {
            return self.getAt(0, 0);
        }

        /// Get all PxQ values at offset (dx, dy) in group coordinates as a tuple
        pub inline fn getAt(self: Self, dx: i32, dy: i32) ResultTuple {
            const x = self.current_x + dx;
            const y = self.current_y + dy;

            var result: ResultTuple = undefined;

            // Map grouped coordinates to nested coordinates
            const nested_x = x * P;
            const nested_y = y * Q;

            const nested = self.source.nested;
            switch (comptime NestedInfo.kind) {
                .eval => inline for (0..Q) |qy| {
                    inline for (0..P) |px| {
                        const idx = qy * P + px;
                        result[idx] = nested.evalAt(nested_x + @as(i32, @intCast(px)), nested_y + @as(i32, @intCast(qy)));
                    }
                },
                .read => inline for (0..Q) |qy| {
                    const row_data = nested.readVec(WideVecT, nested_x, nested_y + @as(i32, @intCast(qy)));
                    const deinterlaced = std.simd.deinterlace(P, row_data);
                    inline for (0..P) |px| {
                        result[qy * P + px] = deinterlaced[px];
                    }
                },
            }

            return result;
        }

        /// Get a specific pixel within the group at offset (dx, dy)
        pub inline fn getPixel(self: Self, dx: i32, dy: i32, px: usize, py: usize) ValueT {
            const grouped = self.getAt(dx, dy);
            return grouped[py * P + px];
        }
    };
}

/// Helper to detect if a type is a GroupSource
pub fn isGroupSourceType(comptime T: type) bool {
    return sources.hasSourceTag(T, .group);
}

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

fn RequireVectorValueType(comptime SourceType: type, comptime vec_len: comptime_int) type {
    const ValueT = sources.SourceValueType(SourceType, vec_len);
    if (@typeInfo(ValueT) != .vector) {
        @compileError(@typeName(SourceType) ++ " must evaluate to a vector type to be used with group/ungroup");
    }
    return ValueT;
}

/// Ceiling division for signed integers (infallible, unlike std.math.divCeil).
fn divCeil(a: i32, b: i32) i32 {
    return -@divFloor(-a, b);
}

/// Evaluate the PxQ grouped tuple at grouped position (x, y).
///
/// Contract: lane i of tuple element (dy * P + dx) is channel (dx, dy) of
/// group x + i. The nested pixels of one group row are strided by P, so a
/// P*vec_len-wide contiguous row is fetched and deinterlaced; for eval-kind
/// nested sources the wide row is assembled from P consecutive vec_len-wide
/// evaluations.
///
/// With `.unchecked` bounds, nested reads skip bounds checking where the
/// nested source supports it; the caller must guarantee the covered nested
/// area is in-bounds (see GroupSource.getInteriorRegion).
inline fn evalGroupTuple(
    comptime NestedSource: type,
    comptime P: comptime_int,
    comptime Q: comptime_int,
    comptime vec_len: comptime_int,
    comptime bounds: sources.BoundsCheck,
    nested: NestedSource,
    x: i32,
    y: i32,
) VecTuple(P * Q, RequireVectorValueType(NestedSource, vec_len)) {
    const NestedInfo = sources.SourceTraits(NestedSource);
    const VecT = RequireVectorValueType(NestedSource, vec_len);
    const ElemT = @typeInfo(VecT).vector.child;
    const WideVecT = @Vector(P * vec_len, ElemT);

    var result: VecTuple(P * Q, VecT) = undefined;

    // Map grouped coordinates to nested coordinates
    const nested_x = x * P;
    const nested_y = y * Q;

    inline for (0..Q) |dy| {
        const row_y = nested_y + @as(i32, @intCast(dy));
        const row_data: WideVecT = switch (comptime NestedInfo.kind) {
            .read => if (comptime bounds == .unchecked and NestedInfo.unchecked_kind == .read)
                nested.readVecUnchecked(WideVecT, nested_x, row_y)
            else
                nested.readVec(WideVecT, nested_x, row_y),
            .eval => blk: {
                var row: [P * vec_len]ElemT = undefined;
                inline for (0..P) |p| {
                    const chunk_x = nested_x + @as(i32, @intCast(p * vec_len));
                    const chunk = if (comptime bounds == .unchecked and NestedInfo.unchecked_kind == .eval)
                        nested.evalAtUnchecked(chunk_x, row_y)
                    else
                        nested.evalAt(chunk_x, row_y);
                    row[p * vec_len ..][0..vec_len].* = @as([vec_len]ElemT, chunk);
                }
                break :blk row;
            },
        };
        const deinterlaced = std.simd.deinterlace(P, row_data);
        inline for (0..P) |dx| {
            result[dy * P + dx] = deinterlaced[dx];
        }
    }

    return result;
}

// ============================================================================
// MARK: Group Source
// ============================================================================

/// A grouped source that combines PxQ pixels from a nested source into a single "pixel".
/// This is useful for Bayer pattern processing where you want to treat 2x2 blocks as units.
pub fn GroupSource(comptime NestedSource: type, comptime P: comptime_int, comptime Q: comptime_int) type {
    const NestedInfo = sources.SourceTraits(NestedSource);
    const vec_len = deriveSourceVecLen(NestedSource);
    const VecT = RequireVectorValueType(NestedSource, vec_len);
    const ResultTuple = VecTuple(P * Q, VecT);

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

        /// Evaluate at a grouped position - returns a tuple of PxQ values.
        /// Lane i of tuple element (dy * P + dx) is channel (dx, dy) of group x + i.
        pub inline fn evalAt(self: Self, x: i32, y: i32) ResultTuple {
            return evalGroupTuple(NestedSource, P, Q, vec_len, .checked, self.nested, x, y);
        }

        /// Evaluate at a grouped position using unchecked nested reads.
        /// Caller MUST guarantee (x, y) lies inside getInteriorRegion().
        pub inline fn evalAtUnchecked(self: Self, x: i32, y: i32) ResultTuple {
            return evalGroupTuple(NestedSource, P, Q, vec_len, .unchecked, self.nested, x, y);
        }

        /// Returns the grouped region where evalAtUnchecked is safe: every
        /// covered nested position is in-bounds for the whole nested chain.
        /// Returns an empty region when the nested source has no unchecked path.
        pub fn getInteriorRegion(self: Self) Region {
            const inner: Region = blk: {
                if (comptime NestedInfo.has_interior_region) {
                    break :blk self.nested.getInteriorRegion();
                }
                if (comptime NestedInfo.kind == .read and NestedInfo.has_unchecked and NestedInfo.has_region) {
                    break :blk self.nested.region;
                }
                break :blk .{ .width = 0, .height = 0 };
            };
            if (inner.width == 0 or inner.height == 0) {
                return .{ .width = 0, .height = 0 };
            }
            // Grouped position (x, y) is safe when the nested columns
            // [x*P, x*P + P*vec_len) and rows [y*Q, (y+1)*Q) lie inside `inner`.
            const x_start = divCeil(inner.x, P);
            const x_stop = @divFloor(inner.stopX() - P * vec_len, P) + 1;
            const y_start = divCeil(inner.y, Q);
            const y_stop = @divFloor(inner.stopY(), Q);
            return .{
                .x = x_start,
                .y = y_start,
                .width = @intCast(@max(0, x_stop - x_start)),
                .height = @intCast(@max(0, y_stop - y_start)),
            };
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
    const GroupedInfo = sources.SourceTraits(GroupedSource);

    return struct {
        grouped: GroupedSource,
        region: Region,

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;
        pub const OutputType = VecT;

        const Self = @This();

        /// Lane i of the result is the ungrouped pixel (x + i, y), matching the
        /// contract expected by process(). In a grouped result the lanes span
        /// consecutive *groups*, so the P channels of the matching row are
        /// re-interlaced back into pixel order.
        inline fn evalImpl(self: Self, comptime bounds: sources.BoundsCheck, x: i32, y: i32) VecT {
            // Map ungrouped coordinates to grouped coordinates
            const group_x = @divFloor(x, P);
            const group_y = @divFloor(y, Q);
            // Phase of x within its group: pixel (x + i) lives in channel
            // (i + local_x) % P of group group_x + (i + local_x) / P.
            const local_x: usize = @intCast(@mod(x, P));
            const local_y: usize = @intCast(@mod(y, Q));

            // One grouped evaluation covers groups group_x .. group_x + vec_len - 1,
            // which is enough for all vec_len ungrouped pixels.
            const grouped_values = if (comptime bounds == .unchecked and GroupedInfo.unchecked_kind == .eval)
                self.grouped.evalAtUnchecked(group_x, group_y)
            else
                self.grouped.evalAt(group_x, group_y);

            inline for (0..Q) |dy| {
                if (local_y == dy) {
                    // Interlace the P channels of this row: wide[j] holds the
                    // ungrouped pixel at group_x * P + j.
                    var channels: [P]VecT = undefined;
                    inline for (0..P) |dx| {
                        channels[dx] = grouped_values[dy * P + dx];
                    }
                    const wide = std.simd.interlace(channels);
                    inline for (0..P) |shift| {
                        if (local_x == shift) {
                            return std.simd.extract(wide, shift, vec_len);
                        }
                    }
                }
            }
            unreachable;
        }

        /// Evaluate at an ungrouped position.
        pub inline fn evalAt(self: Self, x: i32, y: i32) VecT {
            return self.evalImpl(.checked, x, y);
        }

        /// Evaluate at an ungrouped position using unchecked grouped reads.
        /// Caller MUST guarantee (x, y) lies inside getInteriorRegion().
        pub inline fn evalAtUnchecked(self: Self, x: i32, y: i32) VecT {
            return self.evalImpl(.unchecked, x, y);
        }

        /// Returns the ungrouped region where evalAtUnchecked is safe.
        /// An ungrouped position only touches the group containing it, so this
        /// is the grouped interior scaled back to pixel coordinates.
        pub fn getInteriorRegion(self: Self) Region {
            if (comptime !GroupedInfo.has_interior_region) {
                return .{ .width = 0, .height = 0 };
            }
            return self.grouped.getInteriorRegion().upscaled(P, Q);
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
        /// values should be a tuple of PxQ vectors.
        /// Lane i of values[dy * P + dx] is channel (dx, dy) of group x + i, so
        /// nested pixels of one row are strided by P: the P channel vectors of
        /// each row are interlaced back into pixel order and written as one
        /// contiguous wide store.
        pub fn write(self: Self, x: u32, y: u32, values: anytype) void {
            // Map grouped coordinates to nested coordinates
            const nested_x = x * P;
            const nested_y = y * Q;

            inline for (0..Q) |dy| {
                const row_y = nested_y + @as(u32, @intCast(dy));
                if (comptime P == 1) {
                    self.nested.write(nested_x, row_y, values[dy]);
                } else {
                    const VecT = @TypeOf(values[dy * P]);
                    var channels: [P]VecT = undefined;
                    inline for (0..P) |dx| {
                        channels[dx] = values[dy * P + dx];
                    }
                    self.nested.write(nested_x, row_y, std.simd.interlace(channels));
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
/// With `.unchecked` bounds, nested reads skip bounds checking where the
/// nested source supports it (interior fast path).
pub fn GroupAccessor(comptime SrcType: type, comptime VecT: type, comptime P: comptime_int, comptime Q: comptime_int, comptime bounds: sources.BoundsCheck) type {
    const NestedType = @TypeOf(@as(SrcType, undefined).nested);
    const vec_len = @typeInfo(VecT).vector.len;
    const ValueT = RequireVectorValueType(NestedType, vec_len);
    const ResultTuple = VecTuple(P * Q, ValueT);

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
            return evalGroupTuple(
                NestedType,
                P,
                Q,
                vec_len,
                bounds,
                self.source.nested,
                self.current_x + dx,
                self.current_y + dy,
            );
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

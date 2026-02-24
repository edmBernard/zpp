//! Group and Ungroup support for Bayer patterns and P x Q pixel grouping.

const std = @import("std");
const region_mod = @import("region.zig");
const zip_mod = @import("zip.zig");
const sources_mod = @import("sources.zig");

const Region = region_mod.Region;
const VecTuple = zip_mod.VecTuple;

/// Get vector length from a source type, defaulting to 4.
fn getSourceVecLen(comptime SourceType: type) comptime_int {
    return zip_mod.getSourceVecLen(SourceType) orelse 4;
}

// ============================================================================
// MARK: Group Source
// ============================================================================

/// A grouped source that combines PxQ pixels from a nested source into a single "pixel".
/// This is useful for Bayer pattern processing where you want to treat 2x2 blocks as units.
pub fn GroupSource(comptime NestedSource: type, comptime P: comptime_int, comptime Q: comptime_int) type {
    // Derive vector length from nested source
    const vec_len = getSourceVecLen(NestedSource);
    const VecT = @Vector(vec_len, f32);
    const ResultTuple = VecTuple(P * Q, VecT);

    return struct {
        nested: NestedSource,
        region: Region,

        const Self = @This();

        pub const is_group_source = true;

        /// The group dimensions
        pub const group_width = P;
        pub const group_height = Q;
        pub const count = P * Q;

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        /// The vector type used by this GroupSource
        pub const VectorType = VecT;

        pub fn getRegion(self: Self) Region {
            return self.region;
        }

        /// Evaluate at a grouped position - returns a tuple of PxQ values
        pub inline fn evalAt(self: Self, x: i32, y: i32) ResultTuple {
            var result: ResultTuple = undefined;

            // Map grouped coordinates to nested coordinates
            const nested_x = x * P;
            const nested_y = y * Q;

            inline for (0..Q) |dy| {
                inline for (0..P) |dx| {
                    const idx = dy * P + dx;
                    if (@hasDecl(NestedSource, "evalAt")) {
                        result[idx] = self.nested.evalAt(nested_x + @as(i32, @intCast(dx)), nested_y + @as(i32, @intCast(dy)));
                    } else if (@hasDecl(NestedSource, "read")) {
                        var vec: VecT = @splat(0);
                        inline for (0..vec_len) |i| {
                            vec[i] = self.nested.read(
                                nested_x + @as(i32, @intCast(dx)) + @as(i32, @intCast(i * P)),
                                nested_y + @as(i32, @intCast(dy)),
                            );
                        }
                        result[idx] = vec;
                    }
                }
            }

            return result;
        }
    };
}

/// Group a source expression, treating PxQ blocks as single pixels.
/// The output region is downscaled by P horizontally and Q vertically.
pub fn group(comptime P: comptime_int, comptime Q: comptime_int, source: anytype) GroupSource(@TypeOf(source), P, Q) {
    const nested_region = sources_mod.getSourceRegion(source);

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
    // Derive vector length from the grouped source
    const vec_len = getSourceVecLen(GroupedSource);
    const VecT = @Vector(vec_len, f32);

    return struct {
        grouped: GroupedSource,
        region: Region,

        const Self = @This();

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        pub fn getRegion(self: Self) Region {
            return self.region;
        }

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
            const idx = local_y * P + local_x;
            return grouped_values[idx];
        }
    };
}

/// Ungroup a grouped source expression back to individual pixels.
/// The output region is upscaled by P horizontally and Q vertically.
pub fn ungroup(comptime P: comptime_int, comptime Q: comptime_int, source: anytype) UngroupSource(@TypeOf(source), P, Q) {
    const grouped_region = sources_mod.getSourceRegion(source);

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
    return struct {
        nested: NestedDest,
        region: Region,

        const Self = @This();

        /// The group dimensions
        pub const group_width = P;
        pub const group_height = Q;
        pub const count = P * Q;

        /// Writes are idempotent if the nested destination supports overlapping writes.
        pub const supports_overlapping_writes = @hasDecl(NestedDest, "supports_overlapping_writes") and NestedDest.supports_overlapping_writes;

        pub fn getRegion(self: Self) Region {
            return self.region;
        }

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
    const nested_region = sources_mod.getSourceRegion(dest);

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
    const ResultTuple = VecTuple(P * Q, VecT);
    const vec_len = @typeInfo(VecT).vector.len;
    const ElemT = @typeInfo(VecT).vector.child;
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

            // Get access to the nested source (inside GroupSource)
            const nested = self.source.nested;
            const NestedType = @TypeOf(nested);

            if (@hasDecl(NestedType, "evalAt")) {
                // Expression tree source - must evaluate each position
                inline for (0..Q) |qy| {
                    inline for (0..P) |px| {
                        const idx = qy * P + px;
                        result[idx] = nested.evalAt(nested_x + @as(i32, @intCast(px)), nested_y + @as(i32, @intCast(qy)));
                    }
                }
            } else if (@hasDecl(NestedType, "readVec")) {
                // InputSource - use deinterlace for efficient SIMD loads
                // Read P*vec_len contiguous pixels per row, then deinterlace into P vectors
                inline for (0..Q) |qy| {
                    const row_data = nested.readVec(WideVecT, nested_x, nested_y + @as(i32, @intCast(qy)));
                    const deinterlaced = std.simd.deinterlace(P, row_data);
                    inline for (0..P) |px| {
                        result[qy * P + px] = deinterlaced[px];
                    }
                }
            } else {
                @compileError("Nested source must have evalAt or readVec method");
            }

            return result;
        }

        /// Get a specific pixel within the group at offset (dx, dy)
        pub inline fn getPixel(self: Self, dx: i32, dy: i32, px: usize, py: usize) VecT {
            const grouped = self.getAt(dx, dy);
            return grouped[py * P + px];
        }
    };
}

/// Helper to detect if a type is a GroupSource
pub fn isGroupSourceType(comptime T: type) bool {
    return @hasDecl(T, "is_group_source");
}

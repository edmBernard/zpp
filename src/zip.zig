//! Zip and Unzip support for combining multiple source/destination expressions.

const std = @import("std");
const sources = @import("sources.zig");
const Region = @import("region.zig").Region;

// ============================================================================
// MARK: Helper Types
// ============================================================================

/// Helper to create a tuple type from an array of types
pub fn SourceTuple(comptime Types: []const type) type {
    return std.meta.Tuple(Types);
}

/// Helper to create a tuple type of N identical types
pub fn VecTuple(comptime count: comptime_int, comptime VecT: type) type {
    return std.meta.Tuple(&([1]type{VecT} ** count));
}

/// Helper to extract source types from a tuple type as an array
pub fn sourceTypesFromTuple(comptime TupleType: type) [tupleLen(TupleType)]type {
    const type_info = @typeInfo(TupleType);
    const fields = type_info.@"struct".fields;
    var types: [fields.len]type = undefined;
    for (fields, 0..) |field, i| {
        types[i] = field.type;
    }
    return types;
}

/// Helper to get the length of a tuple type
pub fn tupleLen(comptime TupleType: type) comptime_int {
    return @typeInfo(TupleType).@"struct".fields.len;
}

fn innerVectorLen(comptime T: type) ?comptime_int {
    switch (@typeInfo(T)) {
        .vector => |info| return info.len,
        .array => |info| {
            if (@typeInfo(info.child) == .vector) {
                return @typeInfo(info.child).vector.len;
            }
        },
        .@"struct" => |info| {
            if (info.is_tuple and info.fields.len > 0) {
                return innerVectorLen(info.fields[0].type);
            }
        },
        else => {},
    }
    return null;
}

fn suggestedSourceVecLen(comptime SourceType: type) ?comptime_int {
    const Traits = sources.SourceTraits(SourceType);
    if (Traits.has_output_scalar_type) {
        return std.simd.suggestVectorLength(Traits.output_scalar_type) orelse 1;
    }
    if (Traits.has_output_type) {
        return innerVectorLen(Traits.output_type);
    }
    if (Traits.has_eval) {
        const fn_info = @typeInfo(@TypeOf(SourceType.evalAt)).@"fn";
        return innerVectorLen(fn_info.return_type.?);
    }
    return null;
}

fn SourceValueTupleType(comptime SourceTypes: anytype, comptime vec_len: comptime_int) type {
    var types: [SourceTypes.len]type = undefined;
    inline for (SourceTypes, 0..) |SourceType, i| {
        types[i] = sources.SourceValueType(SourceType, vec_len);
    }
    return std.meta.Tuple(&types);
}

fn AccessorVecType(comptime SourceType: type, comptime BaseVecT: type) type {
    const Traits = sources.SourceTraits(SourceType);
    if (Traits.kind == .read) {
        return @Vector(@typeInfo(BaseVecT).vector.len, Traits.output_scalar_type);
    }
    return BaseVecT;
}

/// Helper to detect if a type is a ZipSource
pub fn isZipSourceType(comptime T: type) bool {
    return sources.hasSourceTag(T, .zip);
}

// ============================================================================
// MARK: Helper: Get Vector Length from Source
// ============================================================================

/// Get vector length from a source type.
/// Returns the source's vector_length if it has one, otherwise returns null.
pub fn getSourceVecLen(comptime SourceType: type) ?comptime_int {
    if (@hasDecl(SourceType, "vector_length")) {
        return SourceType.vector_length;
    }
    if (@hasDecl(SourceType, "OutputType")) {
        return innerVectorLen(SourceType.OutputType);
    }
    if (@hasDecl(SourceType, "evalAt")) {
        const fn_info = @typeInfo(@TypeOf(SourceType.evalAt)).@"fn";
        return innerVectorLen(fn_info.return_type.?);
    }
    return null;
}

/// Derive vector length from an array of source types.
/// If sources declare a vector width, they must agree.
/// Otherwise, use the smallest platform-suggested width across the nested scalars.
fn deriveVecLen(comptime SourceTypes: anytype) comptime_int {
    var explicit_len: ?comptime_int = null;
    inline for (SourceTypes) |ST| {
        if (getSourceVecLen(ST)) |len| {
            if (explicit_len) |existing| {
                if (existing != len) {
                    @compileError("Zip requires all nested sources to agree on vector length");
                }
            } else {
                explicit_len = len;
            }
        }
    }
    if (explicit_len) |len| {
        return len;
    }

    var suggested_len: ?comptime_int = null;
    inline for (SourceTypes) |ST| {
        if (suggestedSourceVecLen(ST)) |len| {
            suggested_len = if (suggested_len) |existing| @min(existing, len) else len;
        }
    }
    if (suggested_len) |len| {
        return len;
    }

    @compileError("Zip could not infer a vector length from nested sources; expose vector_length or OutputScalarType");
}

// ============================================================================
// MARK: Zip Source
// ============================================================================

/// A zipped source that combines N source expressions.
/// When processed, the kernel receives an array of values.
pub fn ZipSource(comptime source_count: comptime_int, comptime SourceTypes: [source_count]type) type {
    const vec_len = deriveVecLen(SourceTypes);
    const ResultTuple = SourceValueTupleType(SourceTypes, vec_len);

    return struct {
        sources: SourceTuple(&SourceTypes),
        region: Region,

        pub const source_tag = sources.SourceTag.zip;

        /// Number of sources in this zip
        pub const count = source_count;

        /// The tuple type holding all sources
        pub const Sources = SourceTuple(&SourceTypes);

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        /// The value type produced by this ZipSource.
        pub const OutputType = ResultTuple;

        const Self = @This();

        /// Evaluate at a specific position - returns a tuple of values from all sources
        /// This allows ZipSource to be used directly with Process (without a kernel)
        pub inline fn evalAt(self: Self, x: i32, y: i32) ResultTuple {
            var result: ResultTuple = undefined;

            inline for (0..source_count) |i| {
                result[i] = sources.evalSourceChecked(SourceTypes[i], self.sources[i], vec_len, x, y);
            }

            return result;
        }

        /// Evaluate at a specific position using unchecked reads for every
        /// nested source that supports them (the rest stay checked).
        /// Caller MUST guarantee (x, y) lies inside getInteriorRegion().
        pub inline fn evalAtUnchecked(self: Self, x: i32, y: i32) ResultTuple {
            var result: ResultTuple = undefined;

            inline for (0..source_count) |i| {
                result[i] = if (comptime sources.SourceTraits(SourceTypes[i]).has_unchecked)
                    sources.evalSourceUnchecked(SourceTypes[i], self.sources[i], vec_len, x, y)
                else
                    sources.evalSourceChecked(SourceTypes[i], self.sources[i], vec_len, x, y);
            }

            return result;
        }

        /// Returns the region where evalAtUnchecked is safe for every nested
        /// source: the intersection of the nested interiors. Sources without
        /// an unchecked path stay checked and add no constraint.
        pub fn getInteriorRegion(self: Self) Region {
            var interior: ?Region = null;
            inline for (0..source_count) |i| {
                const Traits = sources.SourceTraits(SourceTypes[i]);
                const src_interior: ?Region = blk: {
                    if (comptime Traits.has_interior_region) {
                        break :blk self.sources[i].getInteriorRegion();
                    }
                    if (comptime !Traits.has_unchecked) {
                        break :blk null;
                    }
                    if (comptime Traits.kind == .read and Traits.has_region) {
                        // Leaf data source: a vector read at x covers
                        // columns x .. x + vec_len - 1.
                        break :blk self.sources[i].region.deflated(0, 0, vec_len - 1, 0);
                    }
                    // Unchecked-capable source without interior information:
                    // disable the fast path entirely.
                    break :blk Region{ .width = 0, .height = 0 };
                };
                if (src_interior) |si| {
                    interior = if (interior) |acc| acc.intersection(si) else si;
                }
            }
            return interior orelse .{ .width = 0, .height = 0 };
        }
    };
}

/// Zip any number of source expressions together.
/// The kernel will receive an array [N]VecT that can be unpacked: `const [a, b, c] = in.get();`
pub fn zip(input_sources: anytype) ZipSource(tupleLen(@TypeOf(input_sources)), sourceTypesFromTuple(@TypeOf(input_sources))) {
    const SourcesTuple = @TypeOf(input_sources);
    const type_info = @typeInfo(SourcesTuple);

    if (type_info != .@"struct" or !type_info.@"struct".is_tuple) {
        @compileError("Zip requires a tuple of sources");
    }

    const field_count = type_info.@"struct".fields.len;
    if (field_count < 2) {
        @compileError("Zip requires at least 2 sources");
    }

    // Compute intersection of all regions
    var combined_region = input_sources[0].region;
    inline for (1..field_count) |i| {
        combined_region = combined_region.intersection(input_sources[i].region);
    }

    return .{
        .sources = input_sources,
        .region = combined_region,
    };
}

// ============================================================================
// MARK: Zip Destination
// ============================================================================

/// A zipped destination that combines N destination expressions.
pub fn ZipDest(comptime dest_count: comptime_int, comptime DestTypes: [dest_count]type) type {
    return struct {
        dests: SourceTuple(&DestTypes),
        region: Region,

        pub const count = dest_count;
        const Self = @This();

        pub fn write(self: Self, x: u32, y: u32, values: anytype) void {
            inline for (0..dest_count) |i| {
                self.dests[i].write(x, y, values[i]);
            }
        }

        pub fn writeScalar(self: Self, x: u32, y: u32, values: anytype) void {
            inline for (0..dest_count) |i| {
                self.dests[i].writeScalar(x, y, values[i]);
            }
        }
    };
}

/// Zip any number of destination expressions together.
pub fn zipDest(dests: anytype) ZipDest(tupleLen(@TypeOf(dests)), sourceTypesFromTuple(@TypeOf(dests))) {
    const DestsTuple = @TypeOf(dests);
    const type_info = @typeInfo(DestsTuple);

    if (type_info != .@"struct" or !type_info.@"struct".is_tuple) {
        @compileError("ZipOut requires a tuple of destinations");
    }

    const field_count = type_info.@"struct".fields.len;
    if (field_count < 2) {
        @compileError("ZipOut requires at least 2 destinations");
    }

    // Compute intersection of all regions
    var combined_region = dests[0].region;
    inline for (1..field_count) |i| {
        combined_region = combined_region.intersection(dests[i].region);
    }

    return .{
        .dests = dests,
        .region = combined_region,
    };
}

// ============================================================================
// MARK: Unzip Source
// ============================================================================

/// An unzipped source that extracts a single component from a zipped source.
/// This allows individual processing of zipped channels.
pub fn UnzipSource(comptime ZippedSource: type, comptime channel: usize) type {
    // Get the type of the nested source at this channel index
    const SourceTypes = @typeInfo(ZippedSource.Sources).@"struct".fields;
    if (channel >= SourceTypes.len) {
        @compileError("Unzip channel index out of bounds");
    }
    const NestedSourceType = SourceTypes[channel].type;

    // Derive vector length from the zipped source
    const vec_len = ZippedSource.vector_length;
    const ValueT = sources.SourceValueType(NestedSourceType, vec_len);

    return struct {
        zipped: ZippedSource,
        region: Region,

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;
        pub const OutputType = ValueT;

        const Self = @This();

        /// Get the nested source for this channel
        pub fn getNestedSource(self: Self) NestedSourceType {
            return self.zipped.sources[channel];
        }

        /// For expression tree chaining - evaluate at position
        pub inline fn evalAt(self: Self, x: i32, y: i32) ValueT {
            return sources.evalSourceChecked(NestedSourceType, self.getNestedSource(), vec_len, x, y);
        }
    };
}

/// Helper type for Unzip return value - creates a tuple of UnzipSource types
fn UnzipResultType(comptime ZippedType: type) type {
    const source_count = ZippedType.count;
    var types: [source_count]type = undefined;
    inline for (0..source_count) |i| {
        types[i] = UnzipSource(ZippedType, i);
    }
    return std.meta.Tuple(&types);
}

/// Unzip a zipped source into its component sources.
/// Returns a tuple containing the individual sources.
/// This allows you to process each channel independently after zipping.
/// Example: `const unzipped = Unzip(zipped); const first = unzipped[0];`
pub fn unzip(zipped: anytype) UnzipResultType(@TypeOf(zipped)) {
    const ZippedType = @TypeOf(zipped);
    const source_count = ZippedType.count;

    var result: UnzipResultType(ZippedType) = undefined;
    inline for (0..source_count) |i| {
        result[i] = .{ .zipped = zipped, .region = zipped.region };
    }
    return result;
}

// ============================================================================
// MARK: Zip Accessor
// ============================================================================

/// Zip accessor for kernel functions - provides access to zipped pixel values.
/// Returns tuples that can be unpacked: `const [a, b] = in.get();`
/// With `.unchecked` bounds, nested reads skip bounds checking where the
/// nested source supports it (interior fast path).
pub fn ZipAccessor(comptime SrcType: type, comptime VecT: type, comptime bounds: sources.BoundsCheck) type {
    const source_count = SrcType.count;
    const SourceTypes = @typeInfo(SrcType.Sources).@"struct".fields;
    const ResultTuple = SourceValueTupleType(sourceTypesFromTuple(SrcType.Sources), @typeInfo(VecT).vector.len);
    const InputAccessorGeneric = @import("loop.zig").InputAccessorGeneric;

    return struct {
        source: SrcType,
        current_x: i32,
        current_y: i32,

        /// Number of zipped sources
        pub const num_sources = source_count;

        const Self = @This();

        /// Get values at offset as a tuple
        pub inline fn getAt(self: Self, dx: i32, dy: i32) ResultTuple {
            var result: ResultTuple = undefined;
            inline for (0..source_count) |i| {
                const SourceT = SourceTypes[i].type;
                const Accessor = InputAccessorGeneric(SourceT, AccessorVecType(SourceT, VecT), bounds, null);
                const accessor = Accessor{
                    .source = self.source.sources[i],
                    .current_x = self.current_x,
                    .current_y = self.current_y,
                };
                result[i] = accessor.getAt(dx, dy);
            }
            return result;
        }

        /// Get current values (no offset) as a tuple
        pub inline fn get(self: Self) ResultTuple {
            return self.getAt(0, 0);
        }
    };
}

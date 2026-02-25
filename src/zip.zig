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

/// Helper to detect if a type is a ZipSource
pub fn isZipSourceType(comptime T: type) bool {
    return sources.hasSourceTag(T, .zip);
}

/// Check if a type is a ZipDest
pub fn isZipDestType(comptime T: type) bool {
    return sources.hasDestTag(T, .zip);
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
    return null;
}

/// Derive vector length from an array of source types.
/// Returns the first source's vector_length if any has one, otherwise returns 4
/// (a conservative default for direct Process calls).
fn deriveVecLen(comptime SourceTypes: anytype) comptime_int {
    for (SourceTypes) |ST| {
        if (getSourceVecLen(ST)) |len| {
            return len;
        }
    }
    // Conservative default for direct Process calls without Loop wrapper
    return 4;
}

// ============================================================================
// MARK: Zip Source
// ============================================================================

/// A zipped source that combines N source expressions.
/// When processed, the kernel receives an array of values.
pub fn ZipSource(comptime source_count: comptime_int, comptime SourceTypes: [source_count]type) type {
    // Derive vector length from nested sources, defaulting to 4 if none specify it
    const vec_len = deriveVecLen(SourceTypes);

    const VecT = @Vector(vec_len, f32);
    const ResultTuple = VecTuple(source_count, VecT);

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

        /// The vector type used by this ZipSource
        pub const VectorType = VecT;

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

        pub const dest_tag = sources.DestTag.zip;
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
    const VecT = @Vector(vec_len, f32);

    return struct {
        zipped: ZippedSource,
        region: Region,

        /// Number of elements processed per evalAt call
        pub const vector_length = vec_len;

        const Self = @This();

        /// Get the nested source for this channel
        pub fn getNestedSource(self: Self) NestedSourceType {
            return self.zipped.sources[channel];
        }

        /// For expression tree chaining - evaluate at position
        pub inline fn evalAt(self: Self, x: i32, y: i32) VecT {
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
pub fn ZipAccessor(comptime SrcType: type, comptime VecT: type) type {
    const source_count = SrcType.count;
    const SourceTypes = @typeInfo(SrcType.Sources).@"struct".fields;
    const ResultTuple = VecTuple(source_count, VecT);
    const InputAccessor = @import("loop.zig").InputAccessor;

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
                const Accessor = InputAccessor(SourceT, VecT);
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

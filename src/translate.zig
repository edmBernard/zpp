//! Translation support for shifting sources by integer pixel offsets.
//!
//! Translate wraps any source (InputSource or LoopResult) and remaps
//! coordinates by a fixed (dx, dy) offset. The output region is shifted
//! accordingly while all reads are forwarded to the original source at
//! the remapped position.
//!
//! This is a zero-cost abstraction at the interior: translated reads
//! remain contiguous SIMD loads, unlike InterpLoop(.nearest) which
//! would fall back to scalar gather.

const sources = @import("sources.zig");
const Region = @import("region.zig").Region;
const Margin = @import("region.zig").Margin;

// ============================================================================
// MARK: Translated Source Types
// ============================================================================

/// Returns the appropriate translated source type for the given source.
/// InputSource-like sources (with readVec) get DataTranslatedSource.
/// LoopResult-like sources (with evalAt) get EvalTranslatedSource.
fn TranslatedSource(comptime SourceType: type) type {
    return switch (comptime sources.SourceTraits(SourceType).kind) {
        .eval => EvalTranslatedSource(SourceType),
        .read => DataTranslatedSource(SourceType),
    };
}

/// Translated wrapper for InputSource-like sources (readVec-based).
/// Forwards all reads to the underlying source at (x - dx, y - dy).
fn DataTranslatedSource(comptime SourceType: type) type {
    return struct {
        source: SourceType,
        dx: i32,
        dy: i32,
        region: Region,

        pub const OutputScalarType = SourceType.OutputScalarType;
        /// Forward the nested padding policy so PixelInterpolator keeps its
        /// vectorized clamp gather through translated sources (clamping to the
        /// shifted region composes with the coordinate remap). `void` when the
        /// nested source doesn't expose one.
        pub const PaddingPolicyType = if (@hasDecl(SourceType, "PaddingPolicyType"))
            SourceType.PaddingPolicyType
        else
            void;
        const Self = @This();

        pub fn read(self: Self, x: i32, y: i32) OutputScalarType {
            return self.source.read(x - self.dx, y - self.dy);
        }

        pub fn readVec(self: Self, comptime VecT: type, x: i32, y: i32) VecT {
            return self.source.readVec(VecT, x - self.dx, y - self.dy);
        }

        pub fn readVecUnchecked(self: Self, comptime VecT: type, x: i32, y: i32) VecT {
            return self.source.readVecUnchecked(VecT, x - self.dx, y - self.dy);
        }
    };
}

/// Translated wrapper for LoopResult-like sources (evalAt-based).
/// Forwards all evaluations to the underlying source at (x - dx, y - dy).
fn EvalTranslatedSource(comptime SourceType: type) type {
    const SourceInfo = sources.SourceTraits(SourceType);

    return struct {
        source: SourceType,
        dx: i32,
        dy: i32,
        region: Region,

        pub const OutputType = SourceType.OutputType;
        pub const vector_length = SourceType.vector_length;
        pub const margin = if (SourceInfo.has_margin) SourceType.margin else Margin{};
        const Self = @This();

        pub inline fn evalAt(self: Self, x: i32, y: i32) OutputType {
            return self.source.evalAt(x - self.dx, y - self.dy);
        }

        pub inline fn evalAtUnchecked(self: Self, x: i32, y: i32) OutputType {
            if (comptime SourceInfo.unchecked_kind == .eval) {
                return self.source.evalAtUnchecked(x - self.dx, y - self.dy);
            }
            return self.source.evalAt(x - self.dx, y - self.dy);
        }

        pub fn getInteriorRegion(self: Self) Region {
            if (comptime SourceInfo.has_interior_region) {
                return self.source.getInteriorRegion().shifted(self.dx, self.dy);
            }
            return .{ .width = 0, .height = 0 };
        }
    };
}

// ============================================================================
// MARK: Public API
// ============================================================================

/// Shift a source by (dx, dy) pixels without interpolation.
///
/// The output region is the source region moved by the offset.
/// All reads are forwarded to the original source at (x - dx, y - dy).
///
/// This preserves contiguous SIMD loads (unlike InterpLoop with Nearest)
/// and composes naturally in expression trees, Zip, etc.
///
/// Example — shift an image 10 pixels right and 5 pixels down:
/// ```zig
/// const shifted = zpp.translate(source, 10, 5);
/// zpp.process(shifted, dest);
/// ```
///
/// Example — blend two shifted copies:
/// ```zig
/// const left  = zpp.translate(source, -5, 0);
/// const right = zpp.translate(source,  5, 0);
/// const zipped = zpp.zip(.{left, right});
/// const blended = zpp.loop(f32x4, .{}, zipped, .{}, blend_kernel);
/// ```
pub fn translate(source: anytype, dx: i32, dy: i32) TranslatedSource(@TypeOf(source)) {
    comptime sources.assertIsSource(@TypeOf(source));
    comptime sources.assertSourceHasRegion(@TypeOf(source));
    return .{
        .source = source,
        .dx = dx,
        .dy = dy,
        .region = source.region.shifted(dx, dy),
    };
}

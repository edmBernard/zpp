//! Row caching for expression trees.
//!
//! This module provides caching support for intermediate results in expression trees,
//! avoiding recomputation when accessing neighboring rows (for margins).

const std = @import("std");
const region_mod = @import("region.zig");
const loop_mod = @import("loop.zig");
const zip_mod = @import("zip.zig");

const Region = region_mod.Region;
const DefaultLoopOptions = loop_mod.DefaultLoopOptions;

// Forward declaration - these will be resolved at comptime
// InputAccessor and ZipAccessor are imported from loop.zig and zip.zig respectively

/// Row cache for intermediate results in expression trees.
/// Uses a circular buffer to store a sliding window of rows, avoiding recomputation
/// when accessing neighboring rows (for margins).
///
/// The cache stores `num_rows` rows in a circular manner. When row Y is requested,
/// it is stored in slot `Y mod num_rows`. The cache tracks which actual image row
/// is stored in each slot to determine if recomputation is needed.
fn RowCache(comptime T: type, comptime max_rows: usize) type {
    return struct {
        /// Storage for cached rows. Each row can hold `width` elements.
        data: [max_rows][]T,
        /// Tracks which image row is currently in each cache slot.
        /// A value of min_i64 indicates the slot is uninitialized.
        image_rows_in_slots: [max_rows]i64,
        /// Width of each row in the cache.
        width: u32,
        /// Number of active rows in the cache (based on margin requirements).
        num_rows: usize,
        /// Allocator used for row data.
        allocator: std.mem.Allocator,
        /// Whether the cache has been initialized.
        initialized: bool,

        const Self = @This();
        const min_i64: i64 = std.math.minInt(i64);

        /// Initialize a new row cache.
        pub fn init(allocator: std.mem.Allocator) Self {
            var self = Self{
                .data = undefined,
                .image_rows_in_slots = [_]i64{min_i64} ** max_rows,
                .width = 0,
                .num_rows = 0,
                .allocator = allocator,
                .initialized = false,
            };
            for (0..max_rows) |i| {
                self.data[i] = &[_]T{};
            }
            return self;
        }

        /// Setup the cache for a given width and margin requirements.
        /// `above` is how many rows above the current row are needed.
        /// `below` is how many rows below the current row are needed.
        pub fn setup(self: *Self, width: u32, above: u32, below: u32) !void {
            const needed_rows = above + 1 + below;
            if (needed_rows > max_rows) {
                return error.CacheTooSmall;
            }

            self.width = width;
            self.num_rows = needed_rows;

            // Allocate row buffers
            for (0..needed_rows) |i| {
                if (self.data[i].len != width) {
                    if (self.data[i].len > 0) {
                        self.allocator.free(self.data[i]);
                    }
                    self.data[i] = try self.allocator.alloc(T, width);
                }
                self.image_rows_in_slots[i] = min_i64;
            }
            self.initialized = true;
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            for (0..max_rows) |i| {
                if (self.data[i].len > 0) {
                    self.allocator.free(self.data[i]);
                    self.data[i] = &[_]T{};
                }
            }
            self.initialized = false;
        }

        /// Get the cache slot for a given image row.
        fn getSlot(self: Self, y: i64) usize {
            // Use positive modulo
            const mod_y = @mod(y, @as(i64, @intCast(self.num_rows)));
            return @intCast(mod_y);
        }

        /// Check if row `y` is currently cached and valid.
        pub fn isRowUpToDate(self: Self, y: i64) bool {
            const slot = self.getSlot(y);
            return self.image_rows_in_slots[slot] == y;
        }

        /// Mark row `y` as up to date in the cache.
        pub fn markRowAsUpToDate(self: *Self, y: i64) void {
            const slot = self.getSlot(y);
            self.image_rows_in_slots[slot] = y;
        }

        /// Get the buffer for row `y` (for writing).
        pub fn getRowBuffer(self: *Self, y: i64) []T {
            const slot = self.getSlot(y);
            return self.data[slot];
        }

        /// Get the buffer for row `y` (for reading).
        pub fn getRowData(self: Self, y: i64) []const T {
            const slot = self.getSlot(y);
            return self.data[slot];
        }

        /// Invalidate all cached rows.
        pub fn invalidate(self: *Self) void {
            for (0..max_rows) |i| {
                self.image_rows_in_slots[i] = min_i64;
            }
        }
    };
}

/// A cached loop result that stores intermediate rows to avoid recomputation.
/// This is the caching version of LoopResult
/// The cache is heap-allocated so that copies of this struct (e.g. in Zip)
/// share the same underlying cache, ensuring rows are computed only once.
pub fn CachedLoopResult(
    comptime VecT: type,
    comptime SrcType: type,
    comptime CtxType: type,
    comptime process_fn: anytype,
    comptime opts: DefaultLoopOptions,
    comptime max_cache_rows: usize,
) type {
    const vec_len = @typeInfo(VecT).vector.len;
    const ElemT = @typeInfo(VecT).vector.child;
    const has_coords = opts.need_coordinates != null;
    const is_zip_source = zip_mod.isZipSourceType(SrcType);

    // Import the accessor types - these need to be resolved at comptime
    const AccessorType = if (is_zip_source)
        zip_mod.ZipAccessor(SrcType, VecT)
    else
        loop_mod.InputAccessor(SrcType, VecT);

    const Cache = RowCache(ElemT, max_cache_rows);

    return struct {
        source: SrcType,
        context: CtxType,
        region: Region,
        cache: *Cache,
        allocator: std.mem.Allocator,

        const Self = @This();
        pub const vector_length = vec_len;

        /// Free cache memory and the heap allocation.
        pub fn deinit(self: Self) void {
            self.cache.deinit();
            self.allocator.destroy(self.cache);
        }

        /// Ensure rows needed for position (x, y) are computed and cached.
        fn ensureRowsCached(self: Self, y: i32) void {
            const y64: i64 = y;
            const above: i64 = opts.margin.top;
            const below: i64 = opts.margin.bottom;

            var j: i64 = -above;
            while (j <= below) : (j += 1) {
                const row = y64 + j;
                if (!self.cache.isRowUpToDate(row)) {
                    self.computeRow(row);
                    self.cache.markRowAsUpToDate(row);
                }
            }
        }

        /// Compute a single row and store it in the cache.
        fn computeRow(self: Self, y: i64) void {
            const row_buffer = self.cache.getRowBuffer(y);
            const y32: i32 = @intCast(y);

            var x: u32 = 0;
            while (x < self.region.width) : (x += @intCast(vec_len)) {
                const x32: i32 = @as(i32, @intCast(x)) + self.region.x;

                const accessor = AccessorType{
                    .source = self.source,
                    .current_x = x32,
                    .current_y = y32,
                };

                const result: VecT = if (has_coords) blk: {
                    const CoordT = opts.need_coordinates.?;
                    var x_coords: CoordT = undefined;
                    var y_coords: CoordT = undefined;
                    inline for (0..vec_len) |i| {
                        x_coords[i] = @intCast(x32 + @as(i32, @intCast(i)));
                        y_coords[i] = @intCast(y32);
                    }
                    break :blk process_fn(self.context, accessor, x_coords, y_coords);
                } else process_fn(self.context, accessor);

                // Store result in cache
                inline for (0..vec_len) |i| {
                    if (x + i < self.region.width) {
                        row_buffer[x + i] = result[i];
                    }
                }
            }
        }

        /// Evaluate at a specific position, using cached data.
        pub fn evalAt(self: Self, x: i32, y: i32) VecT {
            // Ensure needed rows are cached
            self.ensureRowsCached(y);

            // Read from cache
            var result: VecT = @splat(0);
            inline for (0..vec_len) |i| {
                const px = x - self.region.x + @as(i32, @intCast(i));
                if (px >= 0 and px < @as(i32, @intCast(self.region.width))) {
                    const row_data = self.cache.getRowData(y);
                    result[i] = row_data[@intCast(px)];
                }
            }
            return result;
        }

        pub fn getRegion(self: Self) Region {
            return self.region;
        }
    };
}

/// Create a cached processing loop.
/// Use this when you have vertical margins and want to avoid recomputing rows.
/// The cache size should be at least `margin.top + 1 + margin.bottom`.
/// The cache is heap-allocated so that copies share the same underlying data.
pub fn CachedLoop(
    comptime VecT: type,
    comptime opts: DefaultLoopOptions,
    comptime max_cache_rows: usize,
    source: anytype,
    context: anytype,
    comptime process_fn: anytype,
    allocator: std.mem.Allocator,
) !CachedLoopResult(VecT, @TypeOf(source), @TypeOf(context), process_fn, opts, max_cache_rows) {
    const SrcType = @TypeOf(source);
    const ElemT = @typeInfo(VecT).vector.child;
    const Cache = RowCache(ElemT, max_cache_rows);

    const region = if (@hasDecl(SrcType, "getRegion"))
        source.getRegion()
    else if (@hasField(SrcType, "region"))
        source.region
    else
        @compileError("Source must have a region field or getRegion method");

    const cache = try allocator.create(Cache);
    errdefer allocator.destroy(cache);
    cache.* = Cache.init(allocator);
    errdefer cache.deinit();
    try cache.setup(region.width, opts.margin.top, opts.margin.bottom);

    return .{
        .source = source,
        .context = context,
        .region = region,
        .cache = cache,
        .allocator = allocator,
    };
}

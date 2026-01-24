//! Tests for cache.zig - Row caching for expression trees

const std = @import("std");
const zpp = @import("zpp");

test "RowCache basic operations" {
    var cache = zpp.RowCache(f32, 4).init(std.testing.allocator);
    defer cache.deinit();

    try cache.setup(8, 1, 1); // width=8, above=1, below=1 -> 3 rows

    // Initially no rows are up to date
    try std.testing.expect(!cache.isRowUpToDate(0));
    try std.testing.expect(!cache.isRowUpToDate(1));
    try std.testing.expect(!cache.isRowUpToDate(2));

    // Fill row 0
    const row0 = cache.getRowBuffer(0);
    for (0..8) |i| {
        row0[i] = @floatFromInt(i);
    }
    cache.markRowAsUpToDate(0);

    // Row 0 should now be up to date
    try std.testing.expect(cache.isRowUpToDate(0));
    try std.testing.expect(!cache.isRowUpToDate(1));

    // Verify data
    const read_row0 = cache.getRowData(0);
    try std.testing.expectEqual(@as(f32, 0.0), read_row0[0]);
    try std.testing.expectEqual(@as(f32, 7.0), read_row0[7]);

    // Test circular wrapping: row 3 should use slot 0 (3 mod 3 = 0)
    cache.markRowAsUpToDate(3);
    try std.testing.expect(cache.isRowUpToDate(3));
    try std.testing.expect(!cache.isRowUpToDate(0)); // slot was reused
}

test "RowCache invalidation" {
    var cache = zpp.RowCache(f32, 4).init(std.testing.allocator);
    defer cache.deinit();

    try cache.setup(4, 0, 1); // 2 rows

    cache.markRowAsUpToDate(0);
    cache.markRowAsUpToDate(1);
    try std.testing.expect(cache.isRowUpToDate(0));
    try std.testing.expect(cache.isRowUpToDate(1));

    cache.invalidate();
    try std.testing.expect(!cache.isRowUpToDate(0));
    try std.testing.expect(!cache.isRowUpToDate(1));
}

const std = @import("std");
const zpp = @import("zpp");

test "VectorLike keeps lane count and swaps scalar type" {
    const VecT = zpp.VectorLike(zpp.u8v, f32);
    const info = @typeInfo(VecT).vector;

    try std.testing.expectEqual(@typeInfo(zpp.u8v).vector.len, info.len);
    try std.testing.expectEqual(f32, info.child);
}

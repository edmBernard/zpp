const std = @import("std");

/// Generic linear algebra types parameterized by a SIMD vector type.
///
/// Usage:
/// ```
/// const zla = @import("zla").with(@Vector(4, f32));
/// const v = zla.Vec2{ .x = @splat(1.0), .y = @splat(2.0) };
/// ```
pub fn with(comptime VectorType: type) type {
    const vec_info = @typeInfo(VectorType).vector;

    return struct {
        pub const InnerType: type = VectorType;
        pub const ScalarType: type = vec_info.child;
        pub const vec_len: comptime_int = vec_info.len;

        // MARK: Helper functions

        /// Convert a scalar to a vector by splatting it.
        /// Most of the time we can directly use @splat, but this helper is easier to use inside expression.
        pub inline fn splat(scalar: ScalarType) InnerType {
            return @splat(scalar);
        }

        // MARK: Vec2

        pub const Vec2 = struct {
            x: InnerType,
            y: InnerType,

            pub const zero: Vec2 = .{ .x = @splat(0), .y = @splat(0) };
            pub const ones: Vec2 = .{ .x = @splat(1), .y = @splat(1) };

            /// a * b (scalar)
            pub inline fn mul1(a: Vec2, b: InnerType) Vec2 {
                return .{ .x = a.x * b, .y = a.y * b };
            }

            /// a + b (scalar)
            pub inline fn add1(a: Vec2, b: InnerType) Vec2 {
                return .{ .x = a.x + b, .y = a.y + b };
            }

            /// a + b
            pub inline fn add(a: Vec2, b: Vec2) Vec2 {
                return .{ .x = a.x + b.x, .y = a.y + b.y };
            }

            /// a - b (scalar)
            pub inline fn sub1(a: Vec2, b: InnerType) Vec2 {
                return .{ .x = a.x - b, .y = a.y - b };
            }

            /// a - b
            pub inline fn sub(a: Vec2, b: Vec2) Vec2 {
                return .{ .x = a.x - b.x, .y = a.y - b.y };
            }

            /// Dot product
            pub inline fn dot(p: Vec2, q: Vec2) InnerType {
                return p.x * q.x + p.y * q.y;
            }
        };

        // MARK: Vec3

        pub const Vec3 = struct {
            x: InnerType,
            y: InnerType,
            z: InnerType,

            pub const zero: Vec3 = .{ .x = @splat(0), .y = @splat(0), .z = @splat(0) };
            pub const ones: Vec3 = .{ .x = @splat(1), .y = @splat(1), .z = @splat(1) };

            /// a * b (scalar)
            pub inline fn mul1(a: Vec3, b: InnerType) Vec3 {
                return .{ .x = a.x * b, .y = a.y * b, .z = a.z * b };
            }

            /// a * b (component-wise)
            pub inline fn mul(a: Vec3, b: Vec3) Vec3 {
                return .{ .x = a.x * b.x, .y = a.y * b.y, .z = a.z * b.z };
            }

            /// a + b (scalar)
            pub inline fn add1(a: Vec3, b: InnerType) Vec3 {
                return .{ .x = a.x + b, .y = a.y + b, .z = a.z + b };
            }

            /// a + b
            pub inline fn add(a: Vec3, b: Vec3) Vec3 {
                return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
            }
            /// a - b (scalar)
            pub inline fn sub1(a: Vec3, b: InnerType) Vec3 {
                return .{ .x = a.x - b, .y = a.y - b, .z = a.z - b };
            }

            /// a - b
            pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
                return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
            }

            /// Dot product
            pub inline fn dot(p: Vec3, q: Vec3) InnerType {
                return p.x * q.x + p.y * q.y + p.z * q.z;
            }

            /// Normalize the vector
            pub inline fn normalize(v: Vec3) Vec3 {
                const len = @sqrt(Vec3.dot(v, v));
                return .{ .x = v.x / len, .y = v.y / len, .z = v.z / len };
            }

            /// Linear interpolation between two Vec3
            pub inline fn lerp(a: Vec3, b: Vec3, t: InnerType) Vec3 {
                return .{
                    .x = std.math.lerp(a.x, b.x, t),
                    .y = std.math.lerp(a.y, b.y, t),
                    .z = std.math.lerp(a.z, b.z, t),
                };
            }

            /// Power function for integer exponents
            pub inline fn pow(a: Vec3, comptime b: u32) Vec3 {
                if (comptime b == 0) {
                    return Vec3.ones;
                }
                var res = a;
                inline for (1..b) |_| {
                    res = res.mul(a);
                }
                return res;
            }
        };

        // MARK: Mat2x2

        pub const Mat2x2 = struct {
            data: [4]InnerType,

            pub const identity: Mat2x2 = .{ .data = .{
                @splat(1), @splat(0),
                @splat(0), @splat(1),
            } };

            pub inline fn rotation(angle: InnerType) Mat2x2 {
                const c = @cos(angle);
                const s = @sin(angle);
                return .{ .data = .{
                    c, -s,
                    s, c,
                } };
            }

            pub inline fn scaling(sx: InnerType, sy: InnerType) Mat2x2 {
                return .{ .data = .{
                    sx,        @splat(0),
                    @splat(0), sy,
                } };
            }

            pub inline fn mul1(m: Mat2x2, s: InnerType) Mat2x2 {
                return .{ .data = .{
                    m.data[0] * s, m.data[1] * s,
                    m.data[2] * s, m.data[3] * s,
                } };
            }

            pub inline fn mulvec(m: Mat2x2, b: Vec2) Vec2 {
                return .{
                    .x = m.data[0] * b.x + m.data[1] * b.y,
                    .y = m.data[2] * b.x + m.data[3] * b.y,
                };
            }

            pub inline fn mul(a: Mat2x2, b: Mat2x2) Mat2x2 {
                return .{ .data = .{
                    a.data[0] * b.data[0] + a.data[1] * b.data[2],
                    a.data[0] * b.data[1] + a.data[1] * b.data[3],
                    a.data[2] * b.data[0] + a.data[3] * b.data[2],
                    a.data[2] * b.data[1] + a.data[3] * b.data[3],
                } };
            }
        };

        // MARK: Mat3x3

        pub const Mat3x3 = struct {
            data: [9]InnerType,

            pub const identity: Mat3x3 = .{ .data = .{
                @splat(1), @splat(0), @splat(0),
                @splat(0), @splat(1), @splat(0),
                @splat(0), @splat(0), @splat(1),
            } };

            pub inline fn scaling(sx: InnerType, sy: InnerType, sz: InnerType) Mat3x3 {
                return .{ .data = .{
                    sx,        @splat(0), @splat(0),
                    @splat(0), sy,        @splat(0),
                    @splat(0), @splat(0), sz,
                } };
            }

            pub inline fn rotationX(angle: InnerType) Mat3x3 {
                const c = @cos(angle);
                const s = @sin(angle);
                return .{ .data = .{
                    @splat(1), @splat(0), @splat(0),
                    @splat(0), c,         -s,
                    @splat(0), s,         c,
                } };
            }

            pub inline fn rotationY(angle: InnerType) Mat3x3 {
                const c = @cos(angle);
                const s = @sin(angle);
                return .{ .data = .{
                    c,         @splat(0), s,
                    @splat(0), @splat(1), @splat(0),
                    -s,        @splat(0), c,
                } };
            }

            pub inline fn rotationZ(angle: InnerType) Mat3x3 {
                const c = @cos(angle);
                const s = @sin(angle);
                return .{ .data = .{
                    c,         -s,        @splat(0),
                    s,         c,         @splat(0),
                    @splat(0), @splat(0), @splat(1),
                } };
            }

            pub inline fn mul1(m: Mat3x3, s: InnerType) Mat3x3 {
                return .{ .data = .{
                    m.data[0] * s, m.data[1] * s, m.data[2] * s,
                    m.data[3] * s, m.data[4] * s, m.data[5] * s,
                    m.data[6] * s, m.data[7] * s, m.data[8] * s,
                } };
            }

            pub inline fn mulvec(m: Mat3x3, b: Vec3) Vec3 {
                return .{
                    .x = m.data[0] * b.x + m.data[1] * b.y + m.data[2] * b.z,
                    .y = m.data[3] * b.x + m.data[4] * b.y + m.data[5] * b.z,
                    .z = m.data[6] * b.x + m.data[7] * b.y + m.data[8] * b.z,
                };
            }

            pub inline fn mul(a: Mat3x3, b: Mat3x3) Mat3x3 {
                return .{ .data = .{
                    a.data[0] * b.data[0] + a.data[1] * b.data[3] + a.data[2] * b.data[6],
                    a.data[0] * b.data[1] + a.data[1] * b.data[4] + a.data[2] * b.data[7],
                    a.data[0] * b.data[2] + a.data[1] * b.data[5] + a.data[2] * b.data[8],
                    a.data[3] * b.data[0] + a.data[4] * b.data[3] + a.data[5] * b.data[6],
                    a.data[3] * b.data[1] + a.data[4] * b.data[4] + a.data[5] * b.data[7],
                    a.data[3] * b.data[2] + a.data[4] * b.data[5] + a.data[5] * b.data[8],
                    a.data[6] * b.data[0] + a.data[7] * b.data[3] + a.data[8] * b.data[6],
                    a.data[6] * b.data[1] + a.data[7] * b.data[4] + a.data[8] * b.data[7],
                    a.data[6] * b.data[2] + a.data[7] * b.data[5] + a.data[8] * b.data[8],
                } };
            }
        };
    };
}

// MARK: Tests

const testing = std.testing;

fn expectVecEqual(comptime len: comptime_int, actual: @Vector(len, f32), expected: @Vector(len, f32)) !void {
    for (0..len) |i| {
        try testing.expectApproxEqAbs(expected[i], actual[i], 1e-6);
    }
}

const zla_f32 = with(@Vector(4, f32));

// MARK: Tests Vec2

test "Vec2 mul1" {
    const v = zla_f32.Vec2{ .x = @splat(2.0), .y = @splat(3.0) };
    const r = v.mul1(@splat(4.0));
    try testing.expect(std.meta.eql(r.x, @splat(8.0)));
    try testing.expect(std.meta.eql(r.y, @splat(12.0)));
}

test "Vec2 add1" {
    const v = zla_f32.Vec2{ .x = @splat(2.0), .y = @splat(3.0) };
    const r = v.add1(@splat(4.0));
    try testing.expect(std.meta.eql(r.x, @splat(6.0)));
    try testing.expect(std.meta.eql(r.y, @splat(7.0)));
}

test "Vec2 add" {
    const v = zla_f32.Vec2{ .x = @splat(2.0), .y = @splat(3.0) };
    const r = v.add(.{ .x = @splat(4.0), .y = @splat(5.0) });
    try testing.expect(std.meta.eql(r.x, @splat(6.0)));
    try testing.expect(std.meta.eql(r.y, @splat(8.0)));
}

test "Vec2 sub1" {
    const v = zla_f32.Vec2{ .x = @splat(5.0), .y = @splat(7.0) };
    const r = v.sub1(@splat(2.0));
    try testing.expect(std.meta.eql(r.x, @splat(3.0)));
    try testing.expect(std.meta.eql(r.y, @splat(5.0)));
}

test "Vec2 sub" {
    const v = zla_f32.Vec2{ .x = @splat(5.0), .y = @splat(7.0) };
    const r = v.sub(.{ .x = @splat(2.0), .y = @splat(3.0) });
    try testing.expect(std.meta.eql(r.x, @splat(3.0)));
    try testing.expect(std.meta.eql(r.y, @splat(4.0)));
}

test "Vec2 dot" {
    const a = zla_f32.Vec2{ .x = @splat(2.0), .y = @splat(3.0) };
    const b = zla_f32.Vec2{ .x = @splat(4.0), .y = @splat(5.0) };
    const r = zla_f32.Vec2.dot(a, b);
    try testing.expect(std.meta.eql(r, @splat(23.0)));
}

// MARK: Tests Vec3

test "Vec3 pow" {
    const v = zla_f32.Vec3{ .x = @splat(-2.0), .y = @splat(2.0), .z = @splat(3.0) };

    const r0 = v.pow(0);
    try testing.expect(std.meta.eql(r0.x, @splat(1.0)));
    try testing.expect(std.meta.eql(r0.y, @splat(1.0)));
    try testing.expect(std.meta.eql(r0.z, @splat(1.0)));

    const r3 = v.pow(3);
    try testing.expect(std.meta.eql(r3.x, @splat(-8.0)));
    try testing.expect(std.meta.eql(r3.y, @splat(8.0)));
    try testing.expect(std.meta.eql(r3.z, @splat(27.0)));

    const r10 = v.pow(10);
    try testing.expect(std.meta.eql(r10.x, @splat(1024.0)));
    try testing.expect(std.meta.eql(r10.y, @splat(1024.0)));
    try testing.expect(std.meta.eql(r10.z, @splat(59049.0)));
}

test "Vec3 normalize" {
    const v = zla_f32.Vec3{ .x = @splat(3.0), .y = @splat(0.0), .z = @splat(0.0) };
    const n = v.normalize();
    try expectVecEqual(4, n.x, @splat(1.0));
    try expectVecEqual(4, n.y, @splat(0.0));
    try expectVecEqual(4, n.z, @splat(0.0));
}

test "Vec3 lerp" {
    const a = zla_f32.Vec3{ .x = @splat(0.0), .y = @splat(0.0), .z = @splat(0.0) };
    const b = zla_f32.Vec3{ .x = @splat(10.0), .y = @splat(20.0), .z = @splat(30.0) };
    const r = a.lerp(b, @splat(0.5));
    try expectVecEqual(4, r.x, @splat(5.0));
    try expectVecEqual(4, r.y, @splat(10.0));
    try expectVecEqual(4, r.z, @splat(15.0));
}

test "Vec3 mul" {
    const a = zla_f32.Vec3{ .x = @splat(2.0), .y = @splat(3.0), .z = @splat(4.0) };
    const b = zla_f32.Vec3{ .x = @splat(5.0), .y = @splat(6.0), .z = @splat(7.0) };
    const r = a.mul(b);
    try testing.expect(std.meta.eql(r.x, @splat(10.0)));
    try testing.expect(std.meta.eql(r.y, @splat(18.0)));
    try testing.expect(std.meta.eql(r.z, @splat(28.0)));
}

test "Vec3 dot" {
    const a = zla_f32.Vec3{ .x = @splat(1.0), .y = @splat(2.0), .z = @splat(3.0) };
    const b = zla_f32.Vec3{ .x = @splat(4.0), .y = @splat(5.0), .z = @splat(6.0) };
    const r = zla_f32.Vec3.dot(a, b);
    try testing.expect(std.meta.eql(r, @splat(32.0)));
}

// MARK: Tests Mat2x2

test "Mat2x2 identity" {
    const v = zla_f32.Vec2{ .x = @splat(3.0), .y = @splat(4.0) };
    const r = zla_f32.Mat2x2.identity.mulvec(v);
    try testing.expect(std.meta.eql(r.x, @splat(3.0)));
    try testing.expect(std.meta.eql(r.y, @splat(4.0)));
}

test "Mat2x2 rotation" {
    const half_pi = zla_f32.splat(std.math.pi / 2.0);
    const m = zla_f32.Mat2x2.rotation(half_pi);
    const v = zla_f32.Vec2{ .x = @splat(1.0), .y = @splat(0.0) };
    const r = m.mulvec(v);
    try expectVecEqual(4, r.x, @splat(0.0));
    try expectVecEqual(4, r.y, @splat(1.0));
}

test "Mat2x2 scaling" {
    const m = zla_f32.Mat2x2.scaling(@splat(2.0), @splat(3.0));
    const v = zla_f32.Vec2{ .x = @splat(4.0), .y = @splat(5.0) };
    const r = m.mulvec(v);
    try testing.expect(std.meta.eql(r.x, @splat(8.0)));
    try testing.expect(std.meta.eql(r.y, @splat(15.0)));
}

test "Mat2x2 mul1" {
    const m = zla_f32.Mat2x2{ .data = .{ @splat(1.0), @splat(2.0), @splat(3.0), @splat(4.0) } };
    const r = m.mul1(@splat(2.0));
    try testing.expect(std.meta.eql(r.data[0], @splat(2.0)));
    try testing.expect(std.meta.eql(r.data[1], @splat(4.0)));
    try testing.expect(std.meta.eql(r.data[2], @splat(6.0)));
    try testing.expect(std.meta.eql(r.data[3], @splat(8.0)));
}

test "Mat2x2 mulvec2" {
    // 90 degree rotation (approximate)
    const m2 = zla_f32.Mat2x2{ .data = .{ @splat(0.0), @splat(-1.0), @splat(1.0), @splat(0.0) } };
    const v = zla_f32.Vec2{ .x = @splat(3.0), .y = @splat(4.0) };
    const r2 = m2.mulvec(v);
    try testing.expect(std.meta.eql(r2.x, @splat(-4.0)));
    try testing.expect(std.meta.eql(r2.y, @splat(3.0)));
}

test "Mat2x2 mul" {
    // rotation by pi/2 twice should give rotation by pi
    const half_pi = zla_f32.splat(std.math.pi / 2.0);
    const r90 = zla_f32.Mat2x2.rotation(half_pi);
    const r180 = r90.mul(r90);
    const v = zla_f32.Vec2{ .x = @splat(1.0), .y = @splat(0.0) };
    const r = r180.mulvec(v);
    try expectVecEqual(4, r.x, @splat(-1.0));
    try expectVecEqual(4, r.y, @splat(0.0));
}

// MARK: Tests Mat3x3

test "Mat3x3 identity" {
    const v = zla_f32.Vec3{ .x = @splat(3.0), .y = @splat(4.0), .z = @splat(5.0) };
    const r = zla_f32.Mat3x3.identity.mulvec(v);
    try testing.expect(std.meta.eql(r.x, @splat(3.0)));
    try testing.expect(std.meta.eql(r.y, @splat(4.0)));
    try testing.expect(std.meta.eql(r.z, @splat(5.0)));
}

test "Mat3x3 scaling" {
    const m = zla_f32.Mat3x3.scaling(@splat(2.0), @splat(3.0), @splat(4.0));
    const v = zla_f32.Vec3{ .x = @splat(1.0), .y = @splat(2.0), .z = @splat(3.0) };
    const r = m.mulvec(v);
    try testing.expect(std.meta.eql(r.x, @splat(2.0)));
    try testing.expect(std.meta.eql(r.y, @splat(6.0)));
    try testing.expect(std.meta.eql(r.z, @splat(12.0)));
}

test "Mat3x3 rotationZ" {
    // Rotating (1,0,0) by pi/2 around Z should give (0,1,0)
    const half_pi = zla_f32.splat(std.math.pi / 2.0);
    const m = zla_f32.Mat3x3.rotationZ(half_pi);
    const v = zla_f32.Vec3{ .x = @splat(1.0), .y = @splat(0.0), .z = @splat(0.0) };
    const r = m.mulvec(v);
    try expectVecEqual(4, r.x, @splat(0.0));
    try expectVecEqual(4, r.y, @splat(1.0));
    try expectVecEqual(4, r.z, @splat(0.0));
}

test "Mat3x3 rotationX" {
    // Rotating (0,1,0) by pi/2 around X should give (0,0,1)
    const half_pi = zla_f32.splat(std.math.pi / 2.0);
    const m = zla_f32.Mat3x3.rotationX(half_pi);
    const v = zla_f32.Vec3{ .x = @splat(0.0), .y = @splat(1.0), .z = @splat(0.0) };
    const r = m.mulvec(v);
    try expectVecEqual(4, r.x, @splat(0.0));
    try expectVecEqual(4, r.y, @splat(0.0));
    try expectVecEqual(4, r.z, @splat(1.0));
}

test "Mat3x3 rotationY" {
    // Rotating (0,0,1) by pi/2 around Y should give (1,0,0)
    const half_pi = zla_f32.splat(std.math.pi / 2.0);
    const m = zla_f32.Mat3x3.rotationY(half_pi);
    const v = zla_f32.Vec3{ .x = @splat(0.0), .y = @splat(0.0), .z = @splat(1.0) };
    const r = m.mulvec(v);
    try expectVecEqual(4, r.x, @splat(1.0));
    try expectVecEqual(4, r.y, @splat(0.0));
    try expectVecEqual(4, r.z, @splat(0.0));
}

test "Mat3x3 mul1" {
    const m = zla_f32.Mat3x3.identity;
    const r = m.mul1(@splat(5.0));
    try testing.expect(std.meta.eql(r.data[0], @splat(5.0)));
    try testing.expect(std.meta.eql(r.data[1], @splat(0.0)));
    try testing.expect(std.meta.eql(r.data[4], @splat(5.0)));
    try testing.expect(std.meta.eql(r.data[8], @splat(5.0)));
}

test "Mat3x3 mulvec3" {
    const v = zla_f32.Vec3{ .x = @splat(3.0), .y = @splat(4.0), .z = @splat(5.0) };
    const r = zla_f32.Mat3x3.identity.mulvec(v);
    try testing.expect(std.meta.eql(r.x, @splat(3.0)));
    try testing.expect(std.meta.eql(r.y, @splat(4.0)));
    try testing.expect(std.meta.eql(r.z, @splat(5.0)));
}

test "Mat3x3 mul" {
    // rotationZ by pi/2 twice should give rotation by pi
    const half_pi = zla_f32.splat(std.math.pi / 2.0);
    const r90 = zla_f32.Mat3x3.rotationZ(half_pi);
    const r180 = r90.mul(r90);
    const v = zla_f32.Vec3{ .x = @splat(1.0), .y = @splat(0.0), .z = @splat(0.0) };
    const r = r180.mulvec(v);
    try expectVecEqual(4, r.x, @splat(-1.0));
    try expectVecEqual(4, r.y, @splat(0.0));
    try expectVecEqual(4, r.z, @splat(0.0));
}

// MARK: Tests helper functions

test "splat" {
    const v = zla_f32.splat(42.0);
    try testing.expect(std.meta.eql(v, @splat(42.0)));
}

// MARK: Multi-type test

test "instantiate with i32 vector" {
    const zla_i32 = with(@Vector(4, i32));
    const v = zla_i32.Vec2{ .x = zla_i32.splat(2), .y = zla_i32.splat(3) };
    const r = v.mul1(zla_i32.splat(4));
    try testing.expect(std.meta.eql(r.x, zla_i32.splat(8)));
    try testing.expect(std.meta.eql(r.y, zla_i32.splat(12)));

    const v3 = zla_i32.Vec3{ .x = zla_i32.splat(1), .y = zla_i32.splat(2), .z = zla_i32.splat(3) };
    const dot = zla_i32.Vec3.dot(v3, v3);
    try testing.expect(std.meta.eql(dot, zla_i32.splat(14)));

    const p = v3.pow(0);
    try testing.expect(std.meta.eql(p.x, zla_i32.splat(1)));
}

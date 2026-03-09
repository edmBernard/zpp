# ZLA - Zig Linear Algebra Library

A SIMD linear algebra library for Zig.
This library provide some basic linear algebra such as Vec2, Vec3, Mat2x2 etc... The library is designed to be used with [ZPP](https://github.com/edmbernard/zpp). With that in mind, I have made design choice that make sense only in this context. The inner type is a simd batch. For example Vec2 is a structure that contain 2 @Vector and not a @Vector(2, T). All the type are parametrized by the inner type.

## Quick Start

```zig
const zla = @import("zla");

// Instantiate with your SIMD vector type
const zlaf = zla.With(@Vector(4, f32));

// Create vectors
const a = zlaf.Vec2{ .x = @splat(1.0), .y = @splat(2.0) };
const b = zlaf.Vec2{ .x = @splat(3.0), .y = @splat(4.0) };

// Arithmetic
const sum = a.add(b);
const scaled = a.mul1(zla.splat(2.0));
const d = a.dot(b); // or zlaf.dot(a, b)

// 3D vectors
const v = zlaf.Vec3{ .x = @splat(2.0), .y = @splat(0.0), .z = @splat(0.0) };
const n = v.normalize();

// Matrix-vector multiplication
const m = zlaf.Mat2x2{ .data = .{
    zla.splat(0.0), zla.splat(-1.0),
    zla.splat(1.0), zla.splat(0.0),
} };
const rotated = m.mulvec(a);  // 90-degree rotation
```

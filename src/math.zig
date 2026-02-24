//! SIMD math functions for pixel processing.

const std = @import("std");

// ============================================================================
// MARK: Utility Functions
// ============================================================================

/// Create a SIMD vector with all elements set to the same scalar value.
/// This is a convenience wrapper around @splat that infers the vector type.
///
/// Example:
/// ```zig
/// const VecF32 = @Vector(4, f32);
/// const ones = splat(VecF32, 1.0);  // {1.0, 1.0, 1.0, 1.0}
/// const twos = splat(VecF32, @as(f32, 2.0));
/// ```
pub inline fn splat(comptime VecT: type, scalar: @typeInfo(VecT).vector.child) VecT {
    return @splat(scalar);
}

// ============================================================================
// MARK: Basic Math Functions
// ============================================================================

/// Element-wise absolute value for SIMD vectors.
pub inline fn abs(v: anytype) @TypeOf(v) {
    return @abs(v);
}

/// Element-wise floor for SIMD vectors.
pub inline fn floor(v: anytype) @TypeOf(v) {
    return @floor(v);
}

/// Element-wise ceiling for SIMD vectors.
pub inline fn ceil(v: anytype) @TypeOf(v) {
    return @ceil(v);
}

/// Element-wise truncation (round towards zero) for SIMD vectors.
pub inline fn trunc(v: anytype) @TypeOf(v) {
    return @trunc(v);
}

/// Element-wise round for SIMD vectors.
pub inline fn round(v: anytype) @TypeOf(v) {
    return @round(v);
}

/// Element-wise square root for SIMD vectors.
pub inline fn sqrt(v: anytype) @TypeOf(v) {
    return @sqrt(v);
}

// ============================================================================
// MARK: Trigonometric Functions
// ============================================================================

/// Element-wise sine for SIMD vectors.
pub inline fn sin(v: anytype) @TypeOf(v) {
    return @sin(v);
}

/// Element-wise cosine for SIMD vectors.
pub inline fn cos(v: anytype) @TypeOf(v) {
    return @cos(v);
}

/// Element-wise tangent for SIMD vectors.
pub inline fn tan(v: anytype) @TypeOf(v) {
    return @tan(v);
}

// ============================================================================
// MARK: Exponential and Logarithmic Functions
// ============================================================================

/// Element-wise exponential (e^x) for SIMD vectors.
pub inline fn exp(v: anytype) @TypeOf(v) {
    return @exp(v);
}

/// Element-wise base-2 exponential for SIMD vectors.
pub inline fn exp2(v: anytype) @TypeOf(v) {
    return @exp2(v);
}

/// Element-wise natural logarithm for SIMD vectors.
pub inline fn log(v: anytype) @TypeOf(v) {
    return @log(v);
}

/// Element-wise base-2 logarithm for SIMD vectors.
pub inline fn log2(v: anytype) @TypeOf(v) {
    return @log2(v);
}

/// Element-wise base-10 logarithm for SIMD vectors.
pub inline fn log10(v: anytype) @TypeOf(v) {
    return @log10(v);
}

// ============================================================================
// MARK: Sign and Power Functions
// ============================================================================

/// Element-wise sign function for SIMD vectors.
/// Returns -1 for negative, 0 for zero, 1 for positive.
pub inline fn sign(v: anytype) @TypeOf(v) {
    const VecT = @TypeOf(v);
    const ElemT = @typeInfo(VecT).vector.child;
    const zero: VecT = @splat(0);
    const one: VecT = @splat(1);
    const neg_one: VecT = @splat(-1);

    // Use select: (v > 0) ? 1 : ((v < 0) ? -1 : 0)
    const pos_mask = v > zero;
    const neg_mask = v < zero;

    return @select(ElemT, pos_mask, one, @select(ElemT, neg_mask, neg_one, zero));
}

/// Element-wise power function for SIMD vectors (base^exp).
pub inline fn pow(base: anytype, exponent: @TypeOf(base)) @TypeOf(base) {
    // Zig doesn't have a built-in @pow for SIMD, so we use exp(exp * log(base))
    // This is valid for positive bases
    return @exp(exponent * @log(base));
}

/// Element-wise atan2 for SIMD vectors.
/// Returns the angle in radians between the positive x-axis and the point (x, y).
pub inline fn atan2(y: anytype, x: @TypeOf(y)) @TypeOf(y) {
    const VecT = @TypeOf(y);
    // Zig doesn't have built-in @atan2 for SIMD, implement using atan approximation
    // Using the identity: atan2(y, x) = atan(y/x) with quadrant correction
    const ElemT = @typeInfo(VecT).vector.child;
    const pi: VecT = @splat(std.math.pi);
    const pi_2: VecT = @splat(std.math.pi / 2.0);
    const zero: VecT = @splat(0);

    // Compute atan(y/x) using polynomial approximation
    const ratio = y / x;
    const atan_val = atanApprox(ratio);

    // Quadrant correction
    const x_neg = x < zero;
    const y_neg = y < zero;
    const y_pos = y >= zero;

    // When x < 0 and y >= 0: add pi
    // When x < 0 and y < 0: subtract pi
    // When x >= 0: no correction needed
    const correction = @select(ElemT, x_neg, @select(ElemT, y_pos, pi, -pi), zero);

    // Handle x == 0 cases
    const x_zero = x == zero;
    const result_x_zero = @select(ElemT, y_neg, -pi_2, pi_2);

    return @select(ElemT, x_zero, result_x_zero, atan_val + correction);
}

/// Polynomial approximation for atan for SIMD vectors.
/// Uses a rational approximation accurate for |x| <= 1.
fn atanApprox(x: anytype) @TypeOf(x) {
    const VecT = @TypeOf(x);
    const ElemT = @typeInfo(VecT).vector.child;

    // For |x| > 1, use atan(x) = pi/2 - atan(1/x)
    const one: VecT = @splat(1);
    const pi_2: VecT = @splat(std.math.pi / 2.0);

    const abs_x = @abs(x);
    const large = abs_x > one;

    // Compute atan for |x| <= 1 using polynomial
    const x_small = @select(ElemT, large, one / x, x);
    const x2 = x_small * x_small;

    // Polynomial coefficients for atan approximation
    const c1: VecT = @splat(0.9998660);
    const c3: VecT = @splat(-0.3302995);
    const c5: VecT = @splat(0.1801410);
    const c7: VecT = @splat(-0.0851330);
    const c9: VecT = @splat(0.0208351);

    // Horner's method: x * (c1 + x^2 * (c3 + x^2 * (c5 + x^2 * (c7 + x^2 * c9))))
    var result = c9;
    result = result * x2 + c7;
    result = result * x2 + c5;
    result = result * x2 + c3;
    result = result * x2 + c1;
    result = result * x_small;

    // For |x| > 1: atan(x) = sign(x) * pi/2 - atan(1/x)
    const sign_x = sign(x);
    const result_large = sign_x * pi_2 - result;

    return @select(ElemT, large, result_large, result);
}

// ============================================================================
// MARK: Min/Max/Clamp Functions
// ============================================================================

/// Element-wise minimum of two SIMD vectors.
pub inline fn min(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return @min(a, b);
}

/// Element-wise maximum of two SIMD vectors.
pub inline fn max(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return @max(a, b);
}

/// Element-wise clamp of SIMD vector to range [lo, hi].
pub inline fn clamp(v: anytype, lo: @TypeOf(v), hi: @TypeOf(v)) @TypeOf(v) {
    return @min(@max(v, lo), hi);
}

// ============================================================================
// MARK: Interpolation Functions
// ============================================================================

/// Linear interpolation between two SIMD vectors.
/// Returns a + t * (b - a), equivalent to mix(a, b, t).
pub inline fn lerp(a: anytype, b: @TypeOf(a), t: @TypeOf(a)) @TypeOf(a) {
    return a + t * (b - a);
}

/// Fused multiply-add for SIMD vectors: a * b + c
pub inline fn fma(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
    return @mulAdd(@TypeOf(a), a, b, c);
}

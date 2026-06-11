# ZPP - Zig Pixel Processing Library

A SIMD pixel processing library for Zig, designed for image manipulation using expression trees and lazy evaluation.

- GitHub: [https://github.com/edmbernard/zpp](https://github.com/edmbernard/zpp)

## Features

- **Expression Trees**: Chain multiple operations lazily - only computed when results are needed
- **Zero Dependencies**: Pure Zig implementation with no external libraries
- **Flexible Region Processing**: Process arbitrary rectangular regions with margin support
- **Multiple Interpolation Methods**: Nearest neighbor, bilinear, and bicubic sampling
- **Configurable Padding**: Edge repeat or zero padding for boundary handling
- **Multi-Channel Support**: Native RGB output with interleaved destinations
- **Row Caching**: Efficient caching for kernels requiring vertical neighborhood access

## Requirements

- **Zig**: 0.16.0 or later
- **Dependencies**: None

## Installation

### Using Zig Package Manager

Add ZPP to your project by running:

```bash
zig fetch --save git+https://github.com/edmbernard/zpp
```

Then add the import in your `build.zig`:

```zig
const zpp_dep = b.dependency("zpp", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("zpp", zpp_dep.module("zpp"));
```

### Manual Installation

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .zpp = .{
        .url = "git+https://github.com/edmbernard/zpp",
        .hash = "...", // Use hash from zig fetch
    },
},
```

## Quick Start

Here's a simple example that generates an RGB gradient:

```zig
const std = @import("std");
const zpp = @import("zpp");

const u8v = zpp.u8v;
const f32v = zpp.VectorLike(u8v, f32);

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const width: u32 = 800;
    const height: u32 = 600;

    // Allocate RGB output buffer
    const data = try allocator.alloc(u8, width * height * 3);
    defer allocator.free(data);

    // Define processing region
    const region = zpp.Region{ .x = 0, .y = 0, .width = width, .height = height };

    // Create RGB interleaved output destination (3 channels)
    const dest = try zpp.makeInterleavedDest(u8, 3, data, width, region);

    // Define kernel context and process function
    const GradientKernel = struct {
        const Context = struct {
            width: f32v,
            height: f32v,
        };

        fn toBytes(value: f32v) u8v {
            const zero: f32v = @splat(0.0);
            const max_byte: f32v = @splat(255.0);
            return @trunc(@max(zero, @min(max_byte, value * max_byte)));
        }

        fn process(ctx: Context, x: f32v, y: f32v) [3]u8v {
            const r = x / ctx.width;   // Red increases left to right.
            const g = y / ctx.height;  // Green increases top to bottom.
            const b: f32v = @splat(0.5);
            return .{ toBytes(r), toBytes(g), toBytes(b) };
        }
    };

    const ctx = GradientKernel.Context{
        .width = @splat(@floatFromInt(width)),
        .height = @splat(@floatFromInt(height)),
    };

    // Generate and process
    const generator = zpp.generate(f32v, ctx, GradientKernel.process);
    zpp.process(generator, dest);

    // 'data' now contains the RGB image
}
```

## Core Concepts

### Regions and Margins

A `Region` defines a rectangular area for processing:

```zig
const region = zpp.Region{
    .x = 0,      // Start X coordinate
    .y = 0,      // Start Y coordinate
    .width = 800,
    .height = 600,
};

// Region operations
const area = region.area();                    // Total pixels
const inflated = region.inflatedUniform(5);   // Expand by 5 pixels
const shifted = region.shifted(10, 5);        // Move origin by (dx, dy)
const intersection = region.intersection(other);
```

`Margin` specifies neighborhood access for loop operation:

```zig
const margin = zpp.Margin.uniform(1);     // 1 pixel in all directions (3x3 kernel)
const h_margin = zpp.Margin.horizontal(2); // 2 pixels horizontally only
const v_margin = zpp.Margin.vertical(1);   // 1 pixel vertically only
```

### Sources and Destinations

**Input sources** read from memory buffers:

```zig
// Basic input (uses edge-repeat padding by default)
const source = try zpp.makeSource(f32, input_data[0..], stride, region);

// Input with explicit zero padding
const source_zero = try zpp.makePaddedSource(f32, zpp.ZeroPadding, input_data[0..], stride, region);
```

**Output destinations** write processed results:

```zig
// Single-channel output
const dest = try zpp.makeDest(f32, output_data[0..], stride, region);

// RGB interleaved output (3 channels, values must already be `u8`)
const rgb_dest = try zpp.makeInterleavedDest(u8, 3, rgb_data[0..], width, region);
```

All destination and source constructors validate the region origin, row width/stride, and buffer size up front.

### Kernels

Kernels follow the pattern of a **context struct** containing parameters and a **process function**:

```zig
const MyKernel = struct {
    const Context = struct {
        scale: f32v,
        offset: f32v,
    };

    // For loop: receives input accessor
    fn process(ctx: Context, in: anytype) f32v {
        return in.get() * ctx.scale + ctx.offset;
    }
    // For generate: receives x, y coordinates
    fn generateProcess(ctx: Context, x: f32v, y: f32v) f32v {
        return x / ctx.scale + y / ctx.scale;
    }
};
```

### Processing Primitives

**generate** creates values from coordinates (no input source):

```zig
const generator = zpp.generate(f32v, context, processFunc);
```

The first comptime argument is the coordinate vector type. The output type is inferred from the generator kernel's return value. Coordinate vectors may use integer or float scalars.

**loop** transforms input data through a kernel:

```zig
const result = zpp.loop(f32v, .{}, source, context, processFunc);

// With margin for neighborhood access
const result = zpp.loop(f32v, .{ .margin = zpp.Margin.uniform(1) }, source, context, kernelFunc);
```

**process** executes the pipeline and writes to destination:

```zig
zpp.process(result, destination);
```

`generate()` uses the destination region as its output region. Coordinates passed to the generator kernel are absolute image coordinates inside that destination region.

### Expression Trees

One of ZPP's most powerful features is lazy evaluation through expression trees. Operations are chained without intermediate storage:

```zig
// Chain multiple operations: source -> blur -> sharpen -> gamma
const step1 = zpp.loop(f32v, .{ .margin = zpp.Margin.uniform(1) }, source, blur_ctx, blurKernel);
const step2 = zpp.loop(f32v, .{ .margin = zpp.Margin.uniform(1) }, step1, sharpen_ctx, sharpenKernel);
const step3 = zpp.loop(f32v, .{}, step2, gamma_ctx, gammaKernel);

// Only now is computation triggered - all stages fuse together
zpp.process(step3, destination);
```

### Cached Loops

`cachedLoop()` returns an owning handle. Keep that owner alive for the whole pipeline, and pass `owner.view()` into expression trees:

```zig
const cached = try zpp.cachedLoop(f32v, .{ .margin = zpp.Margin.vertical(1) }, 3, allocator, source, ctx, blurKernel);
defer cached.deinit();

const cached_view = cached.view();
const sharpened = zpp.loop(f32v, .{}, cached_view, sharpen_ctx, sharpenKernel);
zpp.process(sharpened, destination);
```

The returned view is cheap to copy and safe to use inside `zip`, `group`, or other composed pipelines. Only the owner has `deinit()`.

### Stats Destinations

`stats()` and `statsWithCoords()` do not allow overlapping remainder writes. Full SIMD batches arrive through `write()`. Checked remainders arrive through `writeScalar()`, which means only lane 0 is populated and the remaining lanes are zero.

### Translation

For integer pixel offsets (shifting without interpolation):

```zig
// Shift source 10 pixels right and 5 pixels down
const shifted = zpp.translate(source, 10, 5);
zpp.process(shifted, dest);

// Blend two shifted copies via zip
const left  = zpp.translate(source, -5, 0);
const right = zpp.translate(source,  5, 0);
const zipped = zpp.zip(.{left, right});
const blended = zpp.loop(f32v, .{}, zipped, blend_ctx, blendKernel);
```

`translate` preserves contiguous SIMD loads (unlike `interpLoop` with `.nearest`) and composes naturally in expression trees.

### Interpolated Sampling

For geometric transformations (scaling, rotation, warping):

```zig
const ResizeKernel = struct {
    const Context = struct { scale: f32v };

    fn process(ctx: Context, interp: anytype, x: f32v, y: f32v) f32v {
        // Sample source at scaled coordinates
        return interp.sample(x * ctx.scale, y * ctx.scale);
    }
};

const resized = zpp.interpLoop(
    f32v,
    .linear,        // Interpolation method: .nearest, .linear, or .cubic
    source,
    output_region,
    context,
    ResizeKernel.process,
);
```

## API Overview

### Main Modules

| Module | Description |
|--------|-------------|
| `Region`, `Margin` | Rectangular areas and neighborhood specification |
| `makeSource`, `makeDest`, `makeInterleavedDest` | Input/output buffer wrappers |
| `suggested_vec_len`, `f32v`, `u8v`, `VectorLike` | SIMD convenience helpers |
| `generate`, `loop`, `process` | Core processing primitives |
| `interpLoop` | Interpolated sampling for geometric transforms |
| `translate` | Zero-cost integer pixel offset (shift without interpolation) |
| `zip`, `unzip` | Combine/split multiple sources |
| `group`, `ungroup` | Block-based operations |
| `stats`, `statsWithCoords` | Statistical accumulation without memory writes |
| `cachedLoop` | Cached loop owner; call `.view()` to compose the cached source |

### SIMD Math Functions

ZPP exposes vectorized math helpers under `zpp.math`:

```zig
// Vector creation
zpp.math.splat // Create a vector with all elements set to the same value

// Basic operations
zpp.math.abs, zpp.math.floor, zpp.math.ceil, zpp.math.trunc, zpp.math.round, zpp.math.sqrt

// Trigonometric
zpp.math.sin, zpp.math.cos, zpp.math.tan, zpp.math.atan2

// Exponential/Logarithmic
zpp.math.exp, zpp.math.exp2, zpp.math.log, zpp.math.log2, zpp.math.log10

// Utility
zpp.math.sign, zpp.math.pow, zpp.math.min, zpp.math.max, zpp.math.clamp, zpp.math.lerp, zpp.math.fma
```

### Padding Strategies

- `RepeatEdgePadding` (default): Clamps coordinates to valid range
- `ZeroPadding`: Returns zero for out-of-bounds access

## Examples

The `examples/` directory contains working demonstrations:

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `checkerboard` | Generates a checkerboard pattern | `generate`, `makeInterleavedDest`, basic SIMD |
| `simplex_noise` | Procedural noise texture | SIMD noise implementation, hash functions |
| `domain_warping` | fBm-based procedural textures | Multiple noise octaves, color mapping |
| `gradient_filter` | Edge detection pipeline | Expression trees, `interpLoop`, margins |

Run examples with:

```bash
zig build run-checkerboard
zig build run-simplex-noise
zig build run-domain-warping
zig build run-gradient-filter
```

## Build Commands

| Command | Description |
|---------|-------------|
| `zig build` | Build all targets |
| `zig build test` | Run all tests (library + examples) |
| `zig build docs` | Generate documentation to `zig-out/docs/` |
| `zig build --release=fast` | Build with speed optimizations |
| `zig build run-<example>` | Run a specific example |

## Project Structure

```
src/
├── root.zig          # Library entry point - re-exports all public API
├── region.zig        # Region and Margin types
├── sources.zig       # Input/Output buffer wrappers (makeSource, makeDest, makeInterleavedDest)
├── translate.zig     # Zero-cost integer translation (translate)
├── loop.zig          # Core processing primitives (loop, generate, process)
├── math.zig          # SIMD math functions (sin, cos, exp, pow, etc.)
├── interpolation.zig # Interpolation methods (nearest, linear, cubic)
├── padding.zig       # Padding policies (ZeroPadding, RepeatEdgePadding)
├── cache.zig         # Row caching for expression trees (CachedLoopOwner, cachedLoop)
├── zip.zig           # zip/unzip for multiple sources
├── group.zig         # group/ungroup for Bayer patterns
└── stats.zig         # Statistics accumulation (stats, statsWithCoords)
tests/
├── root.zig              # Test entry point - imports all test modules
├── test_helpers.zig      # Shared test utilities (fillRamp, vectorCast, etc.)
├── region_test.zig       # Region and Margin tests
├── sources_test.zig      # Input/Output source tests
├── translate_test.zig    # Translation tests
├── loop_test.zig         # loop processing tests
├── generate_test.zig     # generate processing tests
├── math_test.zig         # SIMD math function tests
├── interpolation_test.zig # Interpolation tests
├── padding_test.zig      # Padding policy tests
├── cache_test.zig        # Row caching tests
├── zip_test.zig          # zip/unzip tests
├── group_test.zig        # group/ungroup tests
├── stats_test.zig        # Statistics tests
└── integration_test.zig  # Cross-module integration tests
```

## Code Style

When contributing, follow these conventions:

- **Imports**: `std` first, then external modules
- **Variables/functions**: `snake_case`
- **Types/structs**: `PascalCase`
- **SIMD functions**: Mark as `inline` for performance
- **Resource cleanup**: Always pair allocations with `defer`

## License

This project is licensed under the Apache License 2.0.

## Disclaimer

It's a toy project. I'm not really good at Zig and the current implementation is mainly written by an LLM. Tests are mainly written by hand. Use it at your own risk. If you spot error, improvement, comments are welcome.

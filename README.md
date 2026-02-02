# ZPP - Zig Pixel Processing Library

A SIMD pixel processing library for Zig, designed for image manipulation using expression trees and lazy evaluation.

- GitHub: [https://github.com/edmbernard/zpp](https://github.com/edmbernard/zpp)

## Features

- **Expression Trees**: Chain multiple operations lazily - only computed when results are needed
- **Zero Dependencies**: Pure Zig implementation with no external libraries
- **Flexible Region Processing**: Process arbitrary rectangular regions with margin support
- **Multiple Interpolation Methods**: Nearest neighbor, bilinear, and bicubic sampling
- **Configurable Padding**: Edge repeat or zero padding for boundary handling
- **Multi-Channel Support**: Native RGB output with automatic interleaving
- **Row Caching**: Efficient caching for kernels requiring vertical neighborhood access

## Requirements

- **Zig**: 0.15.2 or later
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

Here's a simple example that generates a gradient image:

```zig
const std = @import("std");
const zpp = @import("zpp");

// Use platform-optimal vector length
const vec_len = zpp.suggested_vec_len;
const VecF32 = @Vector(vec_len, f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const width: u32 = 800;
    const height: u32 = 600;

    // Allocate RGB output buffer
    const data = try allocator.alloc(u8, width * height * 3);
    defer allocator.free(data);

    // Define processing region
    const region = zpp.Region{ .x = 0, .y = 0, .width = width, .height = height };

    // Create RGB output destination
    const dest = zpp.RgbOut(vec_len, data, width, region);

    // Define kernel context and process function
    const GradientKernel = struct {
        const Context = struct {
            width: VecF32,
            height: VecF32,
        };

        fn process(ctx: Context, x: VecF32, y: VecF32) [3]VecF32 {
            const r = x / ctx.width;       // Red increases left to right
            const g = y / ctx.height;      // Green increases top to bottom
            const b: VecF32 = @splat(0.5); // Constant blue
            return .{ r, g, b };
        }
    };

    const ctx = GradientKernel.Context{
        .width = @splat(@floatFromInt(width)),
        .height = @splat(@floatFromInt(height)),
    };

    // Generate and process
    const generator = zpp.Generate(VecF32, .{}, region, ctx, GradientKernel.process);
    zpp.Process(u8, generator, dest);

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
const intersection = region.intersection(other);
```

`Margin` specifies neighborhood access for convolution kernels:

```zig
const margin = zpp.marginI(1);  // 1 pixel in all directions (3x3 kernel)
const h_margin = zpp.marginH(2); // 2 pixels horizontally only
const v_margin = zpp.marginV(1); // 1 pixel vertically only
```

### Sources and Destinations

**Input sources** read from memory buffers:

```zig
// Basic input (uses edge-repeat padding by default)
const source = zpp.In(f32, &input_data, stride, region);

// Input with explicit zero padding
const source_zero = zpp.InWithPadding(f32, zpp.ZeroPadding, &input_data, stride, region);
```

**Output destinations** write processed results:

```zig
// Single-channel output
const dest = zpp.Out(f32, &output_data, stride, region);

// RGB interleaved output (automatically converts float [0,1] to u8 [0,255])
const rgb_dest = zpp.RgbOut(vec_len, &rgb_data, width, region);
```

### Kernels

Kernels follow the pattern of a **context struct** containing parameters and a **process function**:

```zig
const MyKernel = struct {
    const Context = struct {
        scale: VecF32,
        offset: VecF32,
    };

    // For Loop: receives input accessor
    fn process(ctx: Context, in: anytype) VecF32 {
        return in.get() * ctx.scale + ctx.offset;
    }
};

// For Generate: receives x, y coordinates
fn generateProcess(ctx: Context, x: VecF32, y: VecF32) VecF32 {
    return x / ctx.scale + y / ctx.scale;
}
```

### Processing Primitives

**Generate** creates values from coordinates (no input source):

```zig
const generator = zpp.Generate(VecF32, .{}, region, context, processFunc);
```

**Loop** transforms input data through a kernel:

```zig
const result = zpp.Loop(VecF32, .{}, source, context, processFunc);

// With margin for neighborhood access
const result = zpp.Loop(VecF32, .{ .margin = zpp.marginI(1) }, source, context, kernelFunc);
```

**Process** executes the pipeline and writes to destination:

```zig
zpp.Process(f32, result, destination);
```

### Expression Trees

One of ZPP's most powerful features is lazy evaluation through expression trees. Operations are chained without intermediate storage:

```zig
// Chain multiple operations: source -> blur -> sharpen -> gamma
const step1 = zpp.Loop(VecF32, .{ .margin = zpp.marginI(1) }, source, blur_ctx, blurKernel);
const step2 = zpp.Loop(VecF32, .{ .margin = zpp.marginI(1) }, step1, sharpen_ctx, sharpenKernel);
const step3 = zpp.Loop(VecF32, .{}, step2, gamma_ctx, gammaKernel);

// Only now is computation triggered - all stages fuse together
zpp.Process(f32, step3, destination);
```

### Interpolated Sampling

For geometric transformations (scaling, rotation, warping):

```zig
const ResizeKernel = struct {
    const Context = struct { scale: VecF32 };

    fn process(ctx: Context, interp: anytype, x: VecF32, y: VecF32) VecF32 {
        // Sample source at scaled coordinates
        return interp.sample(x * ctx.scale, y * ctx.scale);
    }
};

const resized = zpp.InterpLoop(
    VecF32,
    .Linear,        // Interpolation method: .Nearest, .Linear, or .Cubic
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
| `In`, `Out`, `RgbOut` | Input/output buffer wrappers |
| `Generate`, `Loop`, `Process` | Core processing primitives |
| `InterpLoop` | Interpolated sampling for geometric transforms |
| `Zip`, `Unzip` | Combine/split multiple sources |
| `Group`, `Ungroup` | Block-based operations |
| `Stats`, `StatsWithCoords` | Statistical accumulation without memory writes |
| `RowCache`, `CachedLoop` | Row caching for complex vertical operations |

### SIMD Math Functions

ZPP provides vectorized versions of common math functions:

```zig
// Vector creation
zpp.splat  // Create vector with all elements set to same value

// Basic operations
zpp.abs, zpp.floor, zpp.ceil, zpp.trunc, zpp.round, zpp.sqrt

// Trigonometric
zpp.sin, zpp.cos, zpp.tan, zpp.atan2

// Exponential/Logarithmic
zpp.exp, zpp.exp2, zpp.log, zpp.log2, zpp.log10

// Utility
zpp.sign, zpp.pow, zpp.min, zpp.max, zpp.clamp, zpp.lerp, zpp.fma
```

### Padding Strategies

- `RepeatEdgePadding` (default): Clamps coordinates to valid range
- `ZeroPadding`: Returns zero for out-of-bounds access

## Examples

The `examples/` directory contains working demonstrations:

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `checkerboard` | Generates a checkerboard pattern | `Generate`, `RgbOut`, basic SIMD |
| `simplex_noise` | Procedural noise texture | SIMD noise implementation, hash functions |
| `domain_warping` | fBm-based procedural textures | Multiple noise octaves, color mapping |
| `gradient_filter` | Edge detection pipeline | Expression trees, `InterpLoop`, margins |

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
├── root.zig        # Library entry point - re-exports all public API
├── region.zig      # Region and Margin types
├── sources.zig     # Input/Output buffer wrappers (In, Out, InterleavedOut)
├── loop.zig        # Core processing primitives (Loop, Generate, Process)
├── math.zig        # SIMD math functions (sin, cos, exp, pow, etc.)
├── interpolation.zig # Interpolation methods (Nearest, Linear, Cubic)
├── padding.zig     # Padding policies (ZeroPadding, RepeatEdgePadding)
├── cache.zig       # Row caching for expression trees (RowCache, CachedLoop)
├── zip.zig         # Zip/Unzip for multiple sources
├── group.zig       # Group/Ungroup for Bayer patterns
└── stats.zig       # Statistics accumulation (Stats, StatsWithCoords)
```

## Code Style

When contributing, follow these conventions:

- **Imports**: `std` first, then external modules
- **Variables/functions**: `snake_case`
- **Types/structs**: `PascalCase`
- **SIMD functions**: Mark as `inline` for performance
- **Resource cleanup**: Always pair allocations with `defer`

See `AGENTS.md` for detailed guidelines.

## License

This project is licensed under the Apache License 2.0.

## Disclaimer

It's a toy project. I'm not really good at Zig and the current implementation is mainly written by an LLM. Use it at your own risk. If you spot error, improvement, comments are welcome.

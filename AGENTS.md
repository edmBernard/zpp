# AGENTS.md - Coding Agent Guidelines for zpp

Guidelines for AI coding agents working on the zpp (Zig Pixel Processing) codebase.

## Project Overview

- **Language:** Zig (minimum version 0.15.2)
- **Type:** SIMD pixel processing library with examples
- **Purpose:** Efficient image processing using Zig's vector capabilities
- **License:** Apache 2.0

## Project Structure

```
/
├── build.zig           # Build configuration
├── build.zig.zon       # Package manifest (no external dependencies)
├── src/
│   ├── root.zig        # Library entry point - re-exports all public API
│   ├── region.zig      # Region and Margin types
│   ├── sources.zig     # Input/Output buffer wrappers (In, Out, InterleavedOut)
│   ├── loop.zig        # Core processing primitives (Loop, Generate, Process)
│   ├── math.zig        # SIMD math functions (sin, cos, exp, pow, etc.)
│   ├── interpolation.zig # Interpolation methods (Nearest, Linear, Cubic)
│   ├── padding.zig     # Padding policies (ZeroPadding, RepeatEdgePadding)
│   ├── cache.zig       # Row caching for expression trees (RowCache, CachedLoop)
│   ├── zip.zig         # Zip/Unzip for multiple sources
│   ├── group.zig       # Group/Ungroup for Bayer patterns
│   └── stats.zig       # Statistics accumulation (Stats, StatsWithCoords)
├── tests/
│   ├── root.zig              # Test entry point - imports all test modules
│   ├── test_helpers.zig      # Shared test utilities (fillRamp, vectorCast, etc.)
│   ├── region_test.zig       # Region and Margin tests
│   ├── sources_test.zig      # Input/Output source tests
│   ├── loop_test.zig         # Loop processing tests
│   ├── generate_test.zig     # Generate processing tests
│   ├── math_test.zig         # SIMD math function tests
│   ├── interpolation_test.zig # Interpolation tests
│   ├── padding_test.zig      # Padding policy tests
│   ├── cache_test.zig        # Row caching tests
│   ├── zip_test.zig          # Zip/Unzip tests
│   ├── group_test.zig        # Group/Ungroup tests
│   ├── stats_test.zig        # Statistics tests
│   └── integration_test.zig  # Cross-module integration tests
├── examples/
│   ├── checkerboard.zig      # Pattern generation example
│   ├── simplex_noise.zig     # Procedural noise example
│   ├── domain_warping.zig    # fBm domain warping example
│   └── gradient_filter.zig   # Edge detection with expression trees
└── zig-out/            # Build artifacts (gitignored)
```

## Build Commands

| Command | Description |
|---------|-------------|
| `zig build` | Build all targets |
| `zig build test` | Run all tests (library + examples) |
| `zig build docs` | Generate documentation to `zig-out/docs/` |
| `zig build run-checkerboard` | Run checkerboard example |
| `zig build run-simplex-noise` | Run simplex noise example |
| `zig build run-domain-warping` | Run domain warping example |
| `zig build run-gradient-filter` | Run gradient filter example |
| `zig build --release=fast` | Build with speed optimizations |

## Running Tests

Tests are in the `tests/` directory, separate from the library source.

```bash
zig build test              # Run all tests (via tests/root.zig)
zig build test --summary all  # Run with verbose output
```

## Code Formatting

```bash
zig fmt src/                # Format library
zig fmt examples/           # Format examples
zig fmt --check .           # Check without modifying (CI)
```

## Code Style Guidelines

### Imports
Always place `std` first, then external modules:
```zig
const std = @import("std");
const zpp = @import("zpp");
```

### Naming Conventions
- **Variables/functions:** `snake_case` (`vec_len`, `generate_image`)
- **Types/structs:** `PascalCase` (`Region`, `Margin`, `VecF32`)
- **Constants:** `snake_case` or `UPPER_SNAKE_CASE` for special values

### SIMD Vector Types
```zig
pub const vec_len = zpp.suggested_vec_len;
pub const VecF32 = @Vector(vec_len, f32);
pub const VecI32 = @Vector(vec_len, i32);

pub inline fn splat(scalar: f32) VecF32 {
    return @splat(scalar);
}
```

### Struct Definitions
Use `const Self = @This();` for self-referencing methods:
```zig
pub const Region = struct {
    x: i32 = 0,
    y: i32 = 0,
    width: u32,
    height: u32,
    const Self = @This();
    pub fn area(self: Self) u32 {
        return self.width * self.height;
    }
};
```

### Kernel Pattern
Kernels have a context struct and a process function:
```zig
pub const KernelContext = struct {
    scale: VecF32,
};

pub fn kernelProcess(ctx: KernelContext, in: anytype) VecF32 {
    return in.get() * ctx.scale;
}
```

### Function Signatures
- Use `inline` for performance-critical SIMD functions
- Use `anytype` for generic accessor parameters
- Return `!T` for fallible operations

### Error Handling
- Use `try` for error propagation
- Always handle or propagate errors explicitly

### Resource Management
Always pair allocations with `defer` cleanup immediately:
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();

const file = try std.fs.cwd().createFile(filename, .{});
defer file.close();
```

### Comments
```zig
//! Module-level documentation (top of file)
/// Function/type documentation
// MARK: Section Headers
```

### Testing
Tests live in the `tests/` directory, one file per module (e.g., `tests/region_test.zig` for `src/region.zig`). Shared utilities are in `tests/test_helpers.zig`. Use descriptive names:
```zig
test "region inflation preserves center" {
    const region = Region{ .x = 10, .y = 10, .width = 20, .height = 20 };
    const inflated = region.inflatedUniform(5);
    try std.testing.expectEqual(@as(i32, 5), inflated.x);
}
```

## API Patterns

### Region and Processing
```zig
const region = zpp.Region{ .x = 0, .y = 0, .width = 800, .height = 600 };

// Input/Output sources
const source = zpp.In(f32, &input_data, stride, region);
const destination = zpp.Out(f32, &output_data, stride, region);
const rgb_dest = zpp.InterleavedOut(u8, 3, rgb_data, width, region);

// Generator (creates from coordinates)
const generator = zpp.Generate(VecF32, context, processFunc);
zpp.Process(generator, destination);

// Loop (transforms input)
const result = zpp.Loop(VecF32, .{}, source, context, processFunc);
zpp.Process(result, destination);

// With margins for convolution
const result = zpp.Loop(VecF32, .{ .margin = zpp.marginI(1) }, source, ctx, kernel);
```

### Expression Trees (lazy chaining)
```zig
const step1 = zpp.Loop(VecF32, .{}, source, ctx1, kernel1);
const step2 = zpp.InterpLoop(VecF32, .Linear, step1, output_region, ctx2, resize_kernel);
const step3 = zpp.Loop(VecF32, .{ .margin = zpp.marginI(1) }, step2, ctx3, gradient_kernel);
zpp.Process(f32, step3, destination);
```

## Common Pitfalls

1. **Vector length mismatch:** Ensure all SIMD operations use consistent vector lengths
2. **Memory leaks:** Always pair allocations with `defer` cleanup
3. **Missing error handling:** Use `try` or handle errors explicitly

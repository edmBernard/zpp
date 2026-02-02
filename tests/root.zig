//! Test entry point - imports all test modules
//!
//! This file serves as the root for running all tests in the test suite.

const std = @import("std");

// Import all test modules to run their tests
comptime {
    _ = @import("region_test.zig");
    _ = @import("generate_test.zig");
    _ = @import("sources_test.zig");
    _ = @import("padding_test.zig");

    _ = @import("loop_test.zig");
    _ = @import("math_test.zig");
    _ = @import("zip_test.zig");
    _ = @import("interpolation_test.zig");
    _ = @import("group_test.zig");
    _ = @import("stats_test.zig");
    _ = @import("cache_test.zig");
    _ = @import("integration_test.zig");
}

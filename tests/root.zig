//! Test entry point that imports each test module.

comptime {
    _ = @import("cache_test.zig");
    _ = @import("generate_test.zig");
    _ = @import("group_test.zig");
    _ = @import("integration_test.zig");
    _ = @import("interpolation_test.zig");
    _ = @import("loop_test.zig");
    _ = @import("math_test.zig");
    _ = @import("padding_test.zig");
    _ = @import("region_test.zig");
    _ = @import("root_api_test.zig");
    _ = @import("sources_test.zig");
    _ = @import("stats_test.zig");
    _ = @import("translate_test.zig");
    _ = @import("zip_test.zig");
}

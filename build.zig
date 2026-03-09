const std = @import("std");

const ExampleSpec = struct {
    name: []const u8,
    source_path: []const u8,
    run_step_name: []const u8,
    run_step_description: []const u8,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zla_mod = b.addModule("zla", .{
        .root_source_file = b.path("src/zla/root.zig"),
        .target = target,
    });

    const zpp_mod = b.addModule("zpp", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "zla", .module = zla_mod },
        },
    });

    const zla_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/zla/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const zpp_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpp", .module = zpp_mod },
            },
        }),
    });

    const test_step = b.step("test", "Run tests");
    const run_zla_tests = b.addRunArtifact(zla_tests);
    const run_zpp_tests = b.addRunArtifact(zpp_tests);
    test_step.dependOn(&run_zpp_tests.step);
    test_step.dependOn(&run_zla_tests.step);

    const install_tests = b.addInstallArtifact(zpp_tests, .{});
    const test_install_step = b.step("test-install", "Build and install test binary for debugging");
    test_install_step.dependOn(&install_tests.step);

    const examples = [_]ExampleSpec{
        .{
            .name = "domain_warping",
            .source_path = "examples/domain_warping.zig",
            .run_step_name = "run-domain-warping",
            .run_step_description = "Run the domain warping example",
        },
        .{
            .name = "simplex_noise",
            .source_path = "examples/simplex_noise.zig",
            .run_step_name = "run-simplex-noise",
            .run_step_description = "Run the simplex noise example",
        },
        .{
            .name = "gradient_filter",
            .source_path = "examples/gradient_filter.zig",
            .run_step_name = "run-gradient-filter",
            .run_step_description = "Run the gradient filter example",
        },
        .{
            .name = "checkerboard",
            .source_path = "examples/checkerboard.zig",
            .run_step_name = "run-checkerboard",
            .run_step_description = "Run the checkerboard example",
        },
        .{
            .name = "bench_interpolation",
            .source_path = "examples/bench_interpolation.zig",
            .run_step_name = "run-bench-interpolation",
            .run_step_description = "Run the interpolation benchmark",
        },
        .{
            .name = "bench_cache",
            .source_path = "examples/bench_cache.zig",
            .run_step_name = "run-bench-cache",
            .run_step_description = "Run the cache benchmark",
        },
    };

    for (examples) |example| {
        const exe = b.addExecutable(.{
            .name = example.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(example.source_path),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zpp", .module = zpp_mod },
                },
            }),
        });

        b.installArtifact(exe);

        const run_example = b.addRunArtifact(exe);
        run_example.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_example.addArgs(args);
        }

        const run_step = b.step(example.run_step_name, example.run_step_description);
        run_step.dependOn(&run_example.step);

        const example_tests = b.addTest(.{
            .root_module = exe.root_module,
        });
        const run_example_tests = b.addRunArtifact(example_tests);
        test_step.dependOn(&run_example_tests.step);
    }

    const docs_lib = b.addLibrary(.{
        .name = "zpp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zla", .module = zla_mod },
            },
        }),
    });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate library documentation");
    docs_step.dependOn(&install_docs.step);
}

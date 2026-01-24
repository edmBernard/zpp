const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("zpp", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
    });

    // Creates an executable that will run tests from the tests/ directory.
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpp", .module = mod },
            },
        }),
    });

    // A run step that will run the test executable.
    const run_tests = b.addRunArtifact(tests);

    // A top level step for running all tests.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);

    // =========================================================================
    // MARK: Examples
    // =========================================================================

    // Domain Warping Example
    const domain_warping_exe = b.addExecutable(.{
        .name = "domain_warping",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/domain_warping.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpp", .module = mod },
            },
        }),
    });

    b.installArtifact(domain_warping_exe);

    // Run step for domain warping example
    const run_domain_warping = b.addRunArtifact(domain_warping_exe);
    run_domain_warping.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_domain_warping.addArgs(args);
    }

    const domain_warping_step = b.step("run-domain-warping", "Run the domain warping example");
    domain_warping_step.dependOn(&run_domain_warping.step);

    // Test for domain warping example
    const domain_warping_tests = b.addTest(.{
        .root_module = domain_warping_exe.root_module,
    });
    const run_domain_warping_tests = b.addRunArtifact(domain_warping_tests);
    test_step.dependOn(&run_domain_warping_tests.step);

    // Simplex Noise Example
    const simplex_noise_exe = b.addExecutable(.{
        .name = "simplex_noise",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/simplex_noise.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpp", .module = mod },
            },
        }),
    });

    b.installArtifact(simplex_noise_exe);

    // Run step for simplex noise example
    const run_simplex_noise = b.addRunArtifact(simplex_noise_exe);
    run_simplex_noise.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_simplex_noise.addArgs(args);
    }

    const simplex_noise_step = b.step("run-simplex-noise", "Run the simplex noise example");
    simplex_noise_step.dependOn(&run_simplex_noise.step);

    // Test for simplex noise example
    const simplex_noise_tests = b.addTest(.{
        .root_module = simplex_noise_exe.root_module,
    });
    const run_simplex_noise_tests = b.addRunArtifact(simplex_noise_tests);
    test_step.dependOn(&run_simplex_noise_tests.step);

    // Gradient Filter Example
    const gradient_filter_exe = b.addExecutable(.{
        .name = "gradient_filter",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/gradient_filter.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpp", .module = mod },
            },
        }),
    });

    b.installArtifact(gradient_filter_exe);

    // Run step for gradient filter example
    const run_gradient_filter = b.addRunArtifact(gradient_filter_exe);
    run_gradient_filter.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_gradient_filter.addArgs(args);
    }

    const gradient_filter_step = b.step("run-gradient-filter", "Run the gradient filter example");
    gradient_filter_step.dependOn(&run_gradient_filter.step);

    // Test for gradient filter example
    const gradient_filter_tests = b.addTest(.{
        .root_module = gradient_filter_exe.root_module,
    });
    const run_gradient_filter_tests = b.addRunArtifact(gradient_filter_tests);
    test_step.dependOn(&run_gradient_filter_tests.step);

    // Checkerboard Example
    const checkerboard_exe = b.addExecutable(.{
        .name = "checkerboard",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/checkerboard.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpp", .module = mod },
            },
        }),
    });

    b.installArtifact(checkerboard_exe);

    // Run step for checkerboard example
    const run_checkerboard = b.addRunArtifact(checkerboard_exe);
    run_checkerboard.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_checkerboard.addArgs(args);
    }

    const checkerboard_step = b.step("run-checkerboard", "Run the checkerboard example");
    checkerboard_step.dependOn(&run_checkerboard.step);

    // Test for checkerboard example
    const checkerboard_tests = b.addTest(.{
        .root_module = checkerboard_exe.root_module,
    });
    const run_checkerboard_tests = b.addRunArtifact(checkerboard_tests);
    test_step.dependOn(&run_checkerboard_tests.step);

    // =========================================================================
    // MARK: Documentation
    // =========================================================================

    // Generate documentation for the zpp library module.
    // The documentation is built from the doc comments (/// and //!) in the source.
    const docs_lib = b.addLibrary(.{
        .name = "zpp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Enable documentation generation - outputs to zig-out/docs/
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    // Top level step for generating documentation: `zig build docs`
    const docs_step = b.step("docs", "Generate library documentation");
    docs_step.dependOn(&install_docs.step);
}

# ZPP Idiomatic Zig Refactor Plan

## Goals

- Make the public API and internal abstractions more idiomatic Zig.
- Reduce repeated comptime introspection and centralize interface logic.
- Tighten invariants so invalid layouts fail early instead of being silently tolerated.
- Preserve current behavior and keep the test suite green during refactors.

## Non-goals

- No semantic redesign of the expression-tree model.
- No large feature additions unless they directly support API cleanup.
- No premature micro-optimization without measurements.

## Phase 1: Establish Shared Traits and Invariants

### Scope

- `src/sources.zig`
- `src/loop.zig`
- `src/group.zig`
- `src/translate.zig`
- `src/interpolation.zig`

### Tasks

1. Introduce a centralized comptime traits layer for source and destination types.
2. Replace repeated `@hasDecl` branching with `SourceTraits(T)` and `DestTraits(T)` style helpers.
3. Standardize compile-time validation and error messages for source and destination contracts.
4. Add explicit invariant checks for region, stride, and buffer shape assumptions in source and destination constructors.
5. Decide where debug assertions are sufficient and where constructors should return errors.

### Expected outcome

- A single source of truth for source and destination capabilities.
- Simpler internal code paths in loop, translate, group, and interpolation.
- Better failure modes for invalid caller input.

## Phase 2: Remove Hardcoded Type Assumptions

### Scope

- `src/zip.zig`
- `src/group.zig`
- `src/root.zig`

### Tasks

1. Remove hardcoded `@Vector(_, f32)` assumptions from zip, unzip, group, and ungroup.
2. Infer vector scalar types from nested sources instead of manufacturing `f32` vectors internally.
3. Replace fallback constants like `4` for vector width with explicit inference rules or documented defaults.
4. Review `suggested_vec_len`, `f32v`, `u8v`, and similar exports to ensure they are convenience aliases rather than hidden requirements.
5. Add tests that exercise non-`f32` paths where the API claims generic support.

### Expected outcome

- More honest generics.
- Fewer hidden constraints in composite sources.
- Clearer type behavior for library users.

## Phase 3: Make Scalar and Vector Access Uniform

### Scope

- `src/sources.zig`
- `src/interpolation.zig`
- `src/loop.zig`
- `src/stats.zig`

### Tasks

1. Add shared helpers for checked and unchecked scalar reads alongside the existing vector helpers.
2. Remove ad hoc patterns like `evalAt(...)[0]` when only one scalar lane is needed.
3. Review remainder handling and first-lane extraction to ensure the implementation matches the documented destination contract.
4. Clarify which APIs are vector-native and which intentionally support scalar access.

### Expected outcome

- Cleaner interpolation and stats code.
- Less coupling between scalar logic and vector implementation details.
- Easier-to-read processing code.

## Phase 4: Simplify Public Surface and Naming

### Scope

- `src/root.zig`
- `README.md`
- selected examples and tests

### Tasks

1. Reduce comment noise in `src/root.zig` and keep it focused on exports.
2. Review exported names for Zig conventions.
3. Consider renaming helpers like `VectorLike`, `SourceKind`, and `ProcessReturnType` if a more idiomatic lowerCamelCase or noun-based name improves clarity.
4. Align README examples with the refined APIs and invariants.
5. Add short doc comments only where they improve discoverability in generated docs.

### Expected outcome

- A smaller and clearer top-level module.
- Documentation that matches actual API expectations.
- More consistent naming across the library.

## Phase 5: Refactor Build Script and Project Structure

### Scope

- `build.zig`
- `tests/root.zig`

### Tasks

1. Extract helpers in `build.zig` for registering examples, run steps, and example tests.
2. Remove repetitive boilerplate while keeping build steps explicit.
3. Keep test registration simple and predictable.
4. Verify the build still exposes the same commands or intentionally document any step-name changes.

### Expected outcome

- A shorter, more maintainable build script.
- Less copy-paste across example definitions.
- Lower cost for adding new examples.

## Phase 6: Expand Tests Around Contracts

### Scope

- `tests/`

### Tasks

1. Add tests for constructor invariants and invalid input handling.
2. Add tests for trait-driven dispatch across direct sources, translated sources, loop results, zip, and group.
3. Add coverage for scalar helper behavior and remainder handling.
4. Add regression tests for any behavior changed during refactoring.

### Expected outcome

- Confidence that refactors preserve behavior.
- Better coverage of currently implicit contracts.

## Recommended Execution Order

1. Phase 1
2. Phase 6 for the new contracts introduced by Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5

## Verification Checklist

- `zig build test`
- Spot-check example builds after `build.zig` cleanup
- Re-read public examples in `README.md` for API consistency
- Confirm compile errors remain actionable for misuse cases

## Risks

- Over-generalizing traits can make compile errors worse if not designed carefully.
- Constructor validation may expose previously hidden caller bugs.
- Type inference cleanup in zip/group can break assumptions in tests or examples that accidentally rely on `f32`.
- Build script deduplication can accidentally change step names or import wiring.

## Suggested First PR

- Implement Phase 1 only.
- Add the minimum new tests needed to lock down invariants and trait behavior.
- Keep naming changes out of the first PR unless required by the trait refactor.

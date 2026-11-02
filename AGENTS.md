# AGENTS.md — Fink-FAT Coding Agent Guide

## Project Overview

**fink-fat** is a Rust workspace for asteroid detection and trajectory reconstruction
from photometric alert streams (ZTF, Vera Rubin/LSST). Despite being under `Python/`,
the project is **primarily Rust** (Rust 2024 edition) with two small Python sub-projects
for ML tooling.

**Workspace members:**
- `src/` — root binary crate: CLI entrypoint, logging, progress, pipeline orchestration
- `crates/fink-fat-engine/` — core library: alerts, seeding, edges, solver, persistence
- `crates/fink-fat-eval/` — evaluation/diagnostic tooling against labelled datasets

**All code, comments, variable names, and commit messages must be in English only.**

---

## Build Commands

```bash
# Build (debug)
cargo build

# Build (release)
cargo build --release

# Check compilation without producing binaries
cargo check --all-targets --all-features

# Check a single crate
cargo check -p fink-fat-engine
```

---

## Test Commands

```bash
# Run all tests in the workspace
cargo test --workspace

# Run all tests for the engine crate only
cargo test -p fink-fat-engine

# Run a single test by name (substring match)
cargo test -p fink-fat-engine <test_name>
# Example:
cargo test -p fink-fat-engine ingest_then_build_seeds_produces_seeds_for_each_night

# Run CLI integration tests
cargo test --test cli_integration

# Run tests with CI environment (sequential, limited Rayon threads)
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=2 cargo test -p fink-fat-engine

# Run benchmarks
cargo bench -p fink-fat-engine
cargo bench --bench generate_topk_edges
```

---

## Lint / Format Commands

```bash
# Format all code (apply)
cargo fmt --all

# Format check only (as in CI / pre-commit)
cargo fmt --all -- --check

# Clippy — warnings are errors (as in CI / pre-commit)
cargo clippy --all-targets --all-features -- -D warnings

# Clippy for a single crate
cargo clippy -p fink-fat-engine -- -D warnings

# Build documentation (with KaTeX header, no warnings)
RUSTDOCFLAGS="-D warnings --html-in-header $(pwd)/katex-header.html" \
  cargo doc --no-deps --all-features -p fink-fat -p fink-fat-engine

# Full doc build for browsing locally
RUSTDOCFLAGS="--html-in-header $(pwd)/katex-header.html" cargo doc --workspace --open
```

**Always run `cargo check` or `cargo clippy` after non-trivial edits.**

The pre-commit hook (enforced via `husky-rs`) runs fmt check, clippy, check, and doc
in sequence. CI runs the same four jobs plus coverage (`cargo llvm-cov`).

---

## Code Style — Rust

### Error Handling
- Use `?` for all error propagation; never `.unwrap()` in library code.
- Every domain module owns a typed error enum in `<module>/error.rs`.
- A top-level `EngineError` in `crates/fink-fat-engine/src/error.rs` aggregates
  all sub-errors via `#[error(transparent)]` + `#[from]`.
- Derive errors with `thiserror`.

### Naming Conventions
| Pattern       | Examples                                    |
|---------------|---------------------------------------------|
| Newtypes      | `NightId(u32)`, `DiaSourceId`, `InputUri`   |
| Stores        | `AlertStore`, `SeedStore`                   |
| Builders      | `SyntheticDatasetBuilder`, `PipelineRunner` |
| Config structs| `EngineConfig`, `EdgeConfig`, `PairConfig`  |
| Error enums   | `EngineError`, `SeedError`, `EdgeBuilderError` |
| Pools         | `EdgeRankingModelPool`                      |
| Spatial types | `SeedSpatialIndex`, `SpacetimeBucket`       |

### Collections & Data Structures
- Use `AHashMap` / `AHashSet` (from `ahash`) in hot paths instead of `std::HashMap`.
- Use `SmallVec` for collections that are typically small.
- Use `camino::Utf8Path` / `Utf8PathBuf` for all file system paths (never `std::path::Path`).
- Use `Arc<String>` for shared immutable strings.

### Iterators & Patterns
- Prefer iterator chains (`.map()`, `.filter()`, `.collect()`) over explicit loops.
- Use `f64::total_cmp()` for floating-point ordering (not `partial_cmp`).
- Use `f64::to_bits()` for hashing floats.
- Use `rayon` for data-parallel operations.
- Use `thread_local!` for ONNX model pools (not `Arc<Mutex<_>>`).

### Module Structure
- Every module has a `mod.rs` re-exporting its public API.
- Every domain has an `error.rs` with a typed error enum.
- New top-level domains: `src/<domain>/mod.rs` + `src/<domain>/error.rs` + `src/<domain>/<impl>.rs`.

### Configuration / Serde
- `#[serde(default, deny_unknown_fields)]` on all config structs.
- Implement custom deserializers for unit-bearing strings
  (e.g. `"86.4 min"`, `"35 arcmin/day"`).
- Config environment overrides use prefix `FINK_FAT__` with `__` separator.

---

## Code Style — Documentation

### Comment Syntax
- Module-level: `//!` inner doc comments at the **top of every file** (every blank
  separator line must also start with `//!`).
- Items: `///` outer doc comments on all `pub` items.

### Required Sections (functions / methods)
1. One-line summary (first line).
2. `Arguments` — one entry per parameter.
3. `Return` — description of return value and error variants.

### Optional Sections (include only when meaningful)
- Extended description paragraph
- `Behavior` — when the function has multiple modes
- `Parallelism` — when concurrency affects observable behavior
- `Errors` — detailed error conditions
- `Panics` — conditions that cause panics
- `Notes` — caveats, complexity, ordering guarantees

Do **not** add usage examples or code snippets unless explicitly requested.

### LaTeX in Doc Comments (KaTeX is enabled)

KaTeX is loaded via `--html-in-header katex-header.html`.

| Rule | Correct | Wrong |
|------|---------|-------|
| Subscript underscore | `$x\_i$` | `$x_i$` |
| Thin space | `\frac{1}{2}\chi^2` | `\frac{1}{2}\,\chi^2` |
| Block formula | single line with `\begin{align}` | split across `///` lines |
| Subscript label | `\mathrm{pos}` | `\text{pos}` |

Block formulas must be on a single line:
```
/// $$\begin{align} c &= \frac{1}{2}\chi^2\_{\text{pos}} \\ &+ \frac{1}{2}\chi^2\_{\text{vel}} \end{align}$$
```

### Cross-References
- Use full `crate::` paths for intra-doc links: `[`SeedNode`](crate::seeds::SeedNode)`.
- Only add cross-references when they genuinely aid understanding.
- Verify the referenced path exists before inserting the link.

---

## Testing Conventions

- **Unit tests:** `#[cfg(test)] mod tests { use super::*; ... }` inline in source files.
- **Integration tests:** `crates/fink-fat-engine/tests/` with `tests/mod.rs` declaring submodules.
- **CLI tests:** `tests/cli_integration.rs` at workspace root, using `assert_cmd`.
- **Synthetic data:** `SyntheticDatasetBuilder` with deterministic RNG (default seed = 42).
- **Property-based tests:** `proptest!` macro with `ProptestConfig { cases: 64, .. }` for CI speed.
- **Filesystem isolation:** always use `tempfile::TempDir` for test files.
- **Floating-point assertions:** use `assert_relative_eq!` / `assert_ulps_eq!` from `approx`.
- No real external data required — all tests use synthetic or temp files.

---

## Code Style — Python (sub-projects only)

Python sub-projects live in `crates/fink-fat-eval/src/bin/edge_ml_prediction/` and
`crates/fink-fat-eval/src/bin/lsst_experiment/`. They use PDM for dependency management.

- Target Python **3.12**.
- Use `pathlib.Path` over `os.path`.
- Prefer numpy vectorised operations over Python loops on large arrays.
- Format with `black` (declared in `[dependency-groups] dev`).
- Use PDM for dependency management (`pdm add`, `pdm install`).

---

## Key Domain Concepts

| Concept | Description |
|---------|-------------|
| `Alert` | Single photometric detection: RA/Dec (radians), MJD TT, magnitude, band, observer code |
| `NightId` | `u32` newtype representing one observation night |
| `SeedNode` | Intra-night motion vector from 2–3 same-night alerts; encodes velocity on tangent plane |
| `Edge` | Inter-night kinematic hypothesis linking two `SeedNode`s from different nights |
| `TrackHypothesis` | Ordered multi-night sequence of seeds connected by edges |
| `AlertLinkageDAG` | Runtime directed acyclic graph of all seeds and edges |
| `PipelineStage` | One of 7 ordered stages: Load → Ingest → BuildSeeds → BuildEdges → Solve → FitOrbit → Save |
| `PersistenceManager` | Journal-based on-disk state; bitcode-serialized + optional LZ4/Zstd/Gzip |
| `EngineConfig` | Single YAML-loaded config; `deny_unknown_fields`; env-overridable with `FINK_FAT__` prefix |

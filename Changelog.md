# Changelog

All notable changes to this project are tracked here. Dates use the ISO-8601 format (YYYY-MM-DD).

## [1.0.0] - 2024-??-??

This release marks the first stable version of the fully rewritten `fink-fat` stack. The project now combines a Rust core for heavy computations with Python bindings for ergonomics and interoperability. The previous pure-Python implementation has been retired.

### Added
- **Rust core crate (`src/`)** implementing alert ingestion, seeding, propagation, scoring, and conflict resolution. The codebase is now organized around dedicated modules such as `alerts`, `propagation`, `track_registry`, and `params` to clarify responsibilities and make the Rust API usable on its own.
- **Python bindings via PyO3 (`python/fink_fat`)** that expose the Rust functionality as a compiled extension module. The bindings include `.pyi` stub files for editors/type checkers and re-export the Rust classes (`Alert`, `AlertStore`, `RollingLinkState`, `DetectConflictPolicy`, etc.) directly to Python callers.
- **Alert data model refresh**: The new `Alert` struct carries all astrometry, photometry, and metadata with explicit units in docstrings. `AlertStore` now supports zero-copy construction from NumPy arrays, deterministic identifiers, rich string representations, and helpers to build link identifiers.
- **Configuration bridge with `FinkFatParams`**: A builder pattern mirrors the strongly typed Rust configuration structs, ensuring Python users can tweak binning, seeding, and propagation tolerances while benefiting from validation performed on the Rust side.
- **Propagation pipeline enhancements**: Introduction of `RollingLinkState`, improved solver/scoring modules, and a clearer conflict policy system (`DetectConflictPolicy`). These are geared toward nightly link creation with better control over heuristics and tie-breaking.
- **Expanded testing and benchmarking**: Criterion benches exercise critical update paths (`benches/track_registry_update`). Property-based tests cover edge cases in seeding and propagation (`tests/`, `proptest-regressions/`). Pytest suites validate the Python API contract (`python/tests/`), including determinism around time-bin behaviour and parameter toggles.
- **Modern project scaffolding**: Added `pyproject.toml` and `Cargo.toml` definitions tuned for building both an `rlib` and `cdylib`, ensuring the crate can serve standalone Rust consumers and the Python wheel. Release builds now use `lto = "thin"` with preserved debug symbols to aid profiling while keeping artifacts production ready.
- **Graph-based tracking architecture**: The linkage pipeline now models the alert stream as a multi-stage graph. Alerts feed into spatial-temporal buckets whose adjacency is controlled by configurable radii and time windows. Pair and triplet generators emit `SeedNode` structures enriched with kinematic features (angular velocity, acceleration proxies, residuals) to populate an intermediate graph layer. `NightSnapshot` snapshots capture the evolving edge set, while the propagation engine scores paths, prunes conflicts, and maintains consistency via `track_registry`. This modular graph decomposition makes it easier to swap heuristics—e.g., custom feature extractors or scoring strategies—without touching upstream ingestion.


### Changed
- **Pipeline implementation**: The entire linkage workflow has been reimplemented in Rust. Compared with the legacy Python pipeline, this delivers large gains in throughput, predictable memory usage, and easier parallelisation. Python now delegates computational hotspots to Rust while keeping the scripting ergonomics intact.
- **Public API surface**: Exported classes and functions have been renamed and regrouped around the new Rust types. Existing Python callers need to adopt `AlertStore.from_numpy(...)`, the new parameter builder, and the updated return types (`Pairs`, `Triplets`, `RollingLinkState`).
- **Build and packaging process**: Building wheels now compiles the Rust crate through `maturin`/`pyo3` conventions. Continuous integration uses the Rust toolchain and runs both `cargo` and `pytest` suites to guarantee parity across languages.
- **Documentation and developer workflow**: README, inline module docs, and type hints now reflect the Rust + Python hybrid architecture, making onboarding smoother and clarifying performance assumptions (units, error budgets, binning strategies).

### Removed
- **Legacy Python-only backend**: All pure-Python implementations of the alert store, seeding, and propagation logic have been removed in favour of the Rust rewrite. This includes bespoke data structures, NumPy-centric loops, and glue scripts that are now superseded by the compiled extension.
- **Ad-hoc configuration scripts**: Older configuration helpers and JSON/TOML loaders are deprecated; the builder pattern centralises configuration flow and ensures validation happens before a pipeline run.

## [Unreleased]
- No additional changes recorded.

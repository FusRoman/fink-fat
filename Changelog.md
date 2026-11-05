# Changelog

All notable changes to this project will be documented in this file.

This project follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Nothing yet.

### Changed
- Nothing yet.

### Removed
- Nothing yet.

## [1.0.0] - 2026-03-19

### Added
- Rust workspace reorganisation with three crates: `fink-fat` (binary entrypoint), `fink-fat-engine` (core engine), and `fink-fat-eval` (evaluation and diagnostics).
- Core engine modules in Rust for alert ingestion, seed generation, inter-night edge construction, solver routing, persistence, and configuration loading.
- Command-line runtime components for typed CLI parsing, progress bars, structured logging, and pipeline orchestration.
- Evaluation tooling in Rust for seeding, edge, solver, and model evaluation, plus feature export and diagnostic plots.
- Documentation overhaul with module-level rustdoc, a new project README, and crate-level README files.
- Rust-focused CI, packaging metadata, and documentation generation with KaTeX support.
- Integration tests and Criterion benchmarks covering seeding, edge generation, solver behaviour, and persistence primitives.

### Changed
- The detection-linking workflow is now implemented natively in Rust instead of Python.
- Runtime behaviour now includes first-class logging and progress reporting via `tracing` and `indicatif`.
- The project now uses validated Cargo manifests, workspace dependencies, and a documented engine configuration model instead of the previous Python packaging flow.
- The root README and crate-level documentation now describe the workspace in terms of user-facing workflows, command-line usage, and crate responsibilities.

### Removed
- The legacy pure-Python `fink_fat/` package, `setup.py` packaging flow, and Python-only linkage implementation.
- Standalone CLI scripts, notebook-driven analysis workflow, and legacy configuration helpers from the primary release path.
- Python linting/test workflows and release-time shell scripts in favour of Rust-oriented CI jobs and Cargo-based packaging.

[Unreleased]: https://github.com/FusRoman/fink-fat/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FusRoman/fink-fat/compare/v0.17.0...v1.0.0


//! Shared test infrastructure for fink-fat CLI integration tests.
//!
//! # Modules
//!
//! - [`config`] — write a fink-fat YAML config pointing to a temp storage directory.
//! - [`night_run`] — invoke `fink-fat night-run` through the compiled binary.
//! - [`reconstruction`] — measure overlap between pipeline output tracks and
//!   ground-truth trajectories stored in the fixture parquets.
//!
//! # Fixture data
//!
//! The fixture parquet files live in `tests/fixtures/nights/`.
//! Each file covers one night and contains only the observations belonging to
//! the five ground-truth trajectories under test (1732, 1789, 1882, 2154, 2169).
//! The `trajectory_id` column is preserved from the source data so ground-truth
//! verification can be done without any external label file.

pub mod config;
pub mod night_run;
pub mod reconstruction;

use std::path::{Path, PathBuf};

/// Absolute path to `tests/fixtures/nights/` inside the workspace.
pub fn fixture_nights_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data_tests")
}

/// Sorted list of all fixture night parquet files.
///
/// Files are sorted lexicographically, which corresponds to ascending
/// `night_id` order because the filenames follow the pattern `night_XXXX.parquet`.
pub fn fixture_night_files() -> Vec<PathBuf> {
    let dir = fixture_nights_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read fixture nights dir {dir:?}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .collect();
    files.sort();
    files
}

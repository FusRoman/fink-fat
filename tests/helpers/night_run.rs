//! CLI invocation helpers for `fink-fat night-run` integration tests.
//!
//! Uses [`assert_cmd`] to invoke the compiled `fink-fat` binary in a
//! subprocess, capturing stdout/stderr for diagnostic output on failure.

use std::path::Path;

use assert_cmd::Command;

/// Invoke `fink-fat night-run --alerts <alerts_path> --config <config_path>`.
///
/// Panics with a detailed message (including stdout/stderr) if the binary
/// exits with a non-zero status code.
///
/// The alert path is formatted as a `file://` URI because [`InputUri`] requires
/// a URI scheme (bare absolute paths are rejected by the URL parser as
/// "relative URL without a base").  For `/abs/path/file.parquet` the resulting
/// argument is `file:///abs/path/file.parquet`.
///
/// Uses the `CARGO_BIN_EXE_fink-fat` environment variable set by Cargo when
/// compiling integration tests, which avoids requiring `cargo` to be on `PATH`
/// and is compatible with custom `build.target-dir` configurations.
///
/// # Arguments
///
/// * `alerts_path` — path to the alert parquet file for this night.
/// * `config_path` — path to the fink-fat YAML configuration file.
pub fn run_night(alerts_path: &Path, config_path: &Path) {
    let alerts_uri = format!(
        "file://{}",
        alerts_path
            .to_str()
            .expect("alerts_path must be valid UTF-8")
    );

    Command::new(env!("CARGO_BIN_EXE_fink-fat"))
        .args([
            "night-run",
            "--alerts",
            &alerts_uri,
            "--config",
            config_path
                .to_str()
                .expect("config_path must be valid UTF-8"),
        ])
        .assert()
        .success();
}

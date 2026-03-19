//! # Fink-FAT
//!
//! Fink-FAT is the command-line entrypoint of the asteroid-linking pipeline.
//! It ingests photometric alerts, groups detections into intra-night seeds,
//! links those seeds across nights, and forwards the resulting trajectories to
//! orbit fitting and evaluation stages.
//!
//! The workspace is split into three crates with clear responsibilities:
//!
//! - `fink-fat` — binary crate. Parses the CLI, loads the engine configuration,
//!   initialises logging and progress reporting, and launches the runtime
//!   pipeline.
//! - `fink-fat-engine` — core library. Implements alert models, seeding, edge
//!   construction, solver routing, persistence, and configuration loading.
//! - `fink-fat-eval` — evaluation binaries. Runs the engine on labelled data to
//!   measure seeding, edge, solver, and model quality.
//!
//! ## Typical usage
//!
//! Run the binary with an alerts URI and a validated configuration file:
//!
//! ```bash
//! fink-fat --alerts file:///path/to/alerts.parquet --config /path/to/config.yml
//! ```
//!
//! Optional flags:
//!
//! - `--progress` enables `indicatif` progress bars.
//! - `--logs` enables terminal and file logging.
//!
//! The configuration file controls the pipeline policy, seeding thresholds,
//! edge generation rules, solver parameters, persistence layout, and optional
//! ONNX-based ranking.
//!
//! ## Main entry points
//!
//! - [`main()`] — parses the CLI and starts the runner.
//! - [`load_config`] — loads and validates an [`EngineConfig`] from disk.
//!
//! ## Runtime flow
//!
//! 1. Parse CLI arguments in [`crate::init_cli`].
//! 2. Load the validated engine configuration.
//! 3. Open or create the persistence layout.
//! 4. Configure progress reporting and logging if requested.
//! 5. Execute the pipeline through [`crate::main_runner::fink_fat_runner`].
//!
//! ## Notes
//!
//! - See [`crate::logging`] for the tracing setup used by the binary.
//! - See [`crate::progress`] for the indicatif-backed progress hooks.
//! - For algorithmic details and configuration schemas, refer to
//!   `fink-fat-engine`.
//!

pub mod init_cli;
pub mod logging;
pub mod main_runner;
pub mod progress;

use camino::Utf8Path;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    error::EngineError,
};

use crate::{init_cli::cli_builder, main_runner::fink_fat_runner};

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig, EngineError> {
    Ok(load_engine_config_validated(config_path)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli_builder();

    fink_fat_runner(cli)?;

    Ok(())
}

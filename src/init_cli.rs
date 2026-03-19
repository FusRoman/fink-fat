//! Command-line argument definitions for the `fink-fat` binary.
//!
//! This module exposes `FinkFatCliArgs`, a `clap`-derived structure that
//! captures the common command-line flags used to run the pipeline, and
//! `cli_builder()` which parses `std::env::args()` and returns the parsed
//! structure.
//!
//! # Examples
//!
//! ```no_run
//! let args = fink_fat::init_cli::cli_builder();
//! // use `args.alerts`, `args.config`, …
//! ```

use camino::Utf8PathBuf;

use clap::Parser;
use fink_fat_engine::pipeline::stages::alert_inputs::input_uri::InputUri;

/// Command-line arguments for the `fink-fat` binary.
///
/// This struct is parsed with `clap` and mirrors the primary flags used by
/// the runner: input alerts URI, engine configuration file, and runtime
/// toggles for progress bar rendering and logging.
///
/// Arguments
/// ---------
/// * `alerts` — Input alert URI. Accepts the same `InputUri` variants used by
///   the engine (`file://`, `s3://`, …). Contains the per-night Parquet
///   alert batch to process.
/// * `config` — Path to the `EngineConfig` YAML file used to configure the
///   pipeline stages, thresholds, solver policy, and persistence layout.
/// * `progress` — When `true`, render `indicatif` progress bars during the
///   run and enable related hooks that provide fine-grained stage progress.
/// * `logs` — When `true`, initialise file + terminal logging according to
///   the log level present in the configuration file.
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct FinkFatCliArgs {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: InputUri,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,

    /// Display indicatif progress bars during the pipeline run
    #[arg(long, default_value_t = false)]
    pub progress: bool,

    /// Enable logging to file and terminal (log level is read from the config file)
    #[arg(long, default_value_t = false)]
    pub logs: bool,
}

/// Parse command-line arguments and return a fully-populated
/// `FinkFatCliArgs` structure.
///
/// Arguments
/// ---------
/// * None — reads arguments from the process environment.
///
/// Return
/// ------
/// * `FinkFatCliArgs` — parsed argument structure ready to be passed to the
///   runner.
pub fn cli_builder() -> FinkFatCliArgs {
    FinkFatCliArgs::parse()
}

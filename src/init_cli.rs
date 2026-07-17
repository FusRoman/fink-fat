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

/// Command-line arguments for the `fink-fat` binary.
///
/// This struct is parsed with `clap` and mirrors the primary flags used by
/// the runner: input alerts URI, engine configuration file, and runtime
/// toggles for progress bar rendering and logging.
///
/// Arguments
/// ---------
/// * `alerts` — Path to the per-night Parquet alert batch to process.
///   Required unless `--list-log-targets` is given.
/// * `config` — Path to the `EngineConfig` YAML file used to configure the
///   pipeline stages, thresholds, solver policy, and persistence layout.
///   Required unless `--list-log-targets` is given.
/// * `progress` — When `true`, render `indicatif` progress bars during the
///   run and enable related hooks that provide fine-grained stage progress.
/// * `logs` — When `true`, initialise file + terminal logging according to
///   the log level (and per-target overrides) from the configuration file
///   and `--log-target`.
/// * `log_targets` — Per-target log level overrides (`TARGET=LEVEL`,
///   repeatable), merged over the configuration file's `log_targets` (this
///   flag wins on conflict). See `--list-log-targets` for valid target names.
/// * `list_log_targets` — Print every available tracing target (name,
///   description, levels) and exit, without loading `--alerts`/`--config`.
/// * `snapshot_every` — In batch mode (multi-night input), write the
///   snapshot every N nights processed, in addition to always after the
///   last night. Omit to write only once, at the end. Ignored in
///   single-night mode.
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct FinkFatCliArgs {
    /// Path to the file containing the night's alerts (Parquet)
    #[arg(
        short,
        long,
        value_name = "ALERTS_FILE",
        required_unless_present = "list_log_targets"
    )]
    pub alerts: Option<Utf8PathBuf>,

    /// Path to the fink-fat configuration file
    #[arg(
        short,
        long,
        value_name = "CONFIG_FILE",
        required_unless_present = "list_log_targets"
    )]
    pub config: Option<Utf8PathBuf>,

    /// Display indicatif progress bars during the pipeline run
    #[arg(long, default_value_t = false)]
    pub progress: bool,

    /// Enable logging to file and terminal. Log files (`fink_fat.<date>.log`)
    /// are written to the config's `storage_path`, alongside the
    /// `BranchCollection` snapshot, rotated daily, and pruned beyond
    /// `log_retention_days` (see `--log-retention-days`).
    #[arg(long, default_value_t = false)]
    pub logs: bool,

    /// Override or add a per-target log level, e.g. `--log-target propagation=trace`
    /// (repeatable). See `--list-log-targets` for valid target names.
    #[arg(long = "log-target", value_name = "TARGET=LEVEL")]
    pub log_targets: Vec<String>,

    /// Print every available tracing target with its description and levels, then exit.
    #[arg(long)]
    pub list_log_targets: bool,

    /// Override the number of daily log files to retain (see the config
    /// file's `log_retention_days`). Only meaningful with `--logs`.
    #[arg(long, value_name = "N")]
    pub log_retention_days: Option<usize>,

    /// In batch mode (multi-night input), write the snapshot to disk every
    /// N nights processed (plus always after the last night). Omit to
    /// write only once, after the final night. Ignored in single-night
    /// mode (the snapshot is always written once, as today).
    #[arg(long, value_name = "N")]
    pub snapshot_every: Option<usize>,
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

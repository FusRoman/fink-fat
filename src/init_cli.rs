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

use clap::{Parser, Subcommand, ValueEnum};
use fink_fat_engine::engine_config::log_level::LogLevel;

#[derive(Parser)]
#[command(name = "fink-fat")]
pub struct FinkFatCli {
    #[command(subcommand)]
    pub command: FinkFatCommands,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum ConvertFormat {
    Parquet,
    SQL,
}

/// Which MPC submission tier `fink-fat submit` targets. Mirrors
/// [`fink_fat_ades::mpc_submission::SubmitEndpoint`] (a `clap`-free type, so
/// this thin CLI-facing wrapper exists instead of deriving `ValueEnum`
/// directly on it). `Test` is the default (see `--endpoint`'s doc on
/// [`FinkFatCommands::Submit`]) — only `Production` sends a real submission.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SubmitEndpointArg {
    Test,
    Production,
}

impl From<SubmitEndpointArg> for fink_fat_ades::mpc_submission::SubmitEndpoint {
    fn from(value: SubmitEndpointArg) -> Self {
        match value {
            SubmitEndpointArg::Test => Self::Test,
            SubmitEndpointArg::Production => Self::Production,
        }
    }
}

#[derive(Subcommand)]
pub enum FinkFatCommands {
    /// Track asteroids in an alert stream from photometric alerts
    Track(FinkFatCliArgs),

    /// Convert the binary output of fink-fat into the requested format
    Convert {
        /// Path to the fink-fat configuration file
        #[arg(short, long, value_name = "CONFIG_FILE")]
        config: Utf8PathBuf,
        #[arg(short, long, value_name = "FORMAT")]
        format: ConvertFormat,
        /// Postgres connection string, e.g. `postgres://user:pass@host/db`.
        /// Required when `--format sql`, ignored otherwise.
        #[arg(long, value_name = "DATABASE_URL")]
        database_url: Option<String>,
        /// Path to the raw observation parquet (e.g. `sso_dataset_eval.parquet`)
        /// to load into the `observations` table. Required when `--format sql`,
        /// ignored otherwise.
        #[arg(long, value_name = "OBSERVATIONS_PARQUET")]
        path_observation: Option<Utf8PathBuf>,
    },

    /// Submit MPC-eligible lineages' observations to the Minor Planet Center
    /// as ADES XML files, recording the outcome in the `mpc_submissions`
    /// table so a later run (or fink-fat-explorer's submission dashboard)
    /// can tell what has already been sent. See `src/submit/mod.rs`'s
    /// module docs for the full per-lineage pipeline.
    Submit {
        /// Comma-separated list of `lineage_designation`s to submit.
        /// Mutually exclusive with `--csv`; exactly one of the two is
        /// required.
        #[arg(long, value_name = "DESIGNATIONS", conflicts_with = "csv")]
        lineages: Option<String>,

        /// Path to a CSV file with a `lineage_designation` column (e.g. the
        /// one downloaded from fink-fat-explorer's Submission page).
        /// Mutually exclusive with `--lineages`.
        #[arg(long, value_name = "CSV_FILE", conflicts_with = "lineages")]
        csv: Option<Utf8PathBuf>,

        /// Postgres connection string, e.g. `postgres://user:pass@host/db`.
        #[arg(long, value_name = "DATABASE_URL")]
        database_url: String,

        /// Path to a YAML file with the submitter/telescope identity fields
        /// (submitter name, observers, measurers, telescope, ack contact) —
        /// see `src/submit::SubmitterConfig`.
        #[arg(long, value_name = "SUBMITTER_CONFIG")]
        submitter_config: Utf8PathBuf,

        /// Which MPC tier to submit to. `test` (the default) hits MPC's
        /// `submit_xml_test` integration-testing endpoint — safe to run
        /// repeatedly, never a real submission. `production` hits the real,
        /// irreversible `submit_xml` endpoint.
        #[arg(long, value_enum, default_value_t = SubmitEndpointArg::Test)]
        endpoint: SubmitEndpointArg,

        /// Stop after building and locally validating each lineage's ADES
        /// document — nothing is sent to MPC and nothing is written to
        /// `mpc_submissions`. The safe way to test the whole pipeline
        /// (already-submitted check, eligibility, ADES generation) end to
        /// end.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Bypass the already-submitted and eligibility gates (steps 0-1).
        /// The ADES schema validation gate (step 2) still applies.
        #[arg(long, default_value_t = false)]
        force: bool,

        /// Enable structured logging to stderr during the submission
        /// process.
        #[arg(long, default_value_t = false)]
        logs: bool,

        /// Also write logs to this file, in addition to stderr. Only
        /// meaningful with `--logs`.
        #[arg(long, value_name = "LOG_FILE")]
        log_file: Option<Utf8PathBuf>,

        /// Minimum log level recorded. Only meaningful with `--logs`.
        #[arg(long, value_name = "LEVEL", default_value = "info")]
        log_level: LogLevel,
    },
}

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
pub fn cli_builder() -> FinkFatCli {
    FinkFatCli::parse()
}

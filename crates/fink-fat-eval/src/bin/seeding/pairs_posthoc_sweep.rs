//! Post-hoc sweep for pair-seeding thresholds (diagnostic plots).
//!
//! Overview
//! --------
//! This binary is a **configuration-tuning and diagnostic tool** for the
//! *pair-seeding* stage of the Fink-FAT pipeline.
//!
//! The core idea is to decouple **pair generation** from **threshold tuning**:
//!
//! 1. Ingest a ZTF/LSST-like alert dataset from a Parquet file,
//! 2. Generate a **superset of candidate pairs once**, using a deliberately
//!    large angular speed (`--gen-max-omega`),
//! 3. Apply a **post-hoc sweep** of angular-speed thresholds
//!    (`sep <= omega * dt`) *without re-running seeding*,
//! 4. Produce diagnostic plots to study tradeoffs between completeness and
//!    contamination.
//!
//! This approach makes it possible to explore threshold sensitivity quickly
//! and reproducibly, even on large datasets.
//!
//! Typical questions this tool helps answer
//! -----------------------------------------
//! * Where is the knee point between false links and missed associations?
//! * Do spurious pairs concentrate at large Δt or large separation?
//! * How sensitive are metrics to the angular-speed cut?
//!
//! What this tool does NOT do
//! --------------------------
//! * It does **not** re-run pair generation for each threshold.
//! * It does **not** optimize multiple parameters simultaneously.
//!
//! Because of this, it is critical that `--gen-max-omega` is **greater than or
//! equal to** the maximum angular-speed threshold tested during the sweep.
//!
//! Outputs
//! -------
//! The tool writes deterministic PNG files to `--out-dir`, typically including:
//! * Δt histogram,
//! * separation histogram,
//! * Δt vs separation scatter plot,
//! * tradeoff curve (quality vs omega threshold).
//!
//! Usage
//! -----
//! ```bash
//! cargo run -p fink-fat-eval --bin pairs-posthoc-sweep -- \
//!   alerts.parquet \
//!   --out-dir out_pairs \
//!   --mode fink \
//!   --gen-max-omega "30 arcsec/min" \
//!   --sweep-min-omega "1 arcsec/min" \
//!   --sweep-steps 40 \
//!   --logspace
//! ```
//!
//! Notes
//! -----
//! * The spatial binning uses a HEALPix partition, while time binning uses
//!   uniform bins whose origin is inferred from the dataset.
//! * This binary is intentionally thin: all reusable logic lives in
//!   `fink-fat-eval` library modules, making it easy to write additional tools
//!   on top of the same post-hoc sweep infrastructure.

use anyhow::Result;
use camino::Utf8PathBuf;
use clap::{ArgAction, Parser, ValueEnum};

use fink_fat_eval::angular_speed::AngularSpeed;
use fink_fat_eval::dataset::ingest_config::AlertIngestConfig;
use fink_fat_eval::dataset::ztf_alerts::{AlertLoadMode, ZtfAlertScan};
use fink_fat_eval::seeding::pairs_sweep::{
    PairsPosthocSweepConfig, default_plot_config, run_pairs_posthoc_sweep,
};
use fink_fat_eval::seeding::plotting::AngularUnit;

/// CLI-facing loading mode.
///
/// This maps to [`AlertLoadMode`] in the library.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum CliLoadMode {
    /// Load only "oracle" asteroids:
    /// `trajectory_id > 0` OR `fink_class == "Solar System MPC"`.
    Oracle,
    /// Load broker-plausible asteroid-like alerts:
    /// `fink_class ∈ {"Solar System MPC", "Solar System candidate", "Unknown"}`.
    Fink,
    /// Load everything (no truth/class filtering).
    All,
}

impl From<CliLoadMode> for AlertLoadMode {
    fn from(v: CliLoadMode) -> Self {
        match v {
            CliLoadMode::Oracle => AlertLoadMode::Oracle,
            CliLoadMode::Fink => AlertLoadMode::Fink,
            CliLoadMode::All => AlertLoadMode::All,
        }
    }
}

/// Command-line interface for the post-hoc pair sweep tool.
///
/// This structure defines all parameters controlling:
/// - dataset ingestion,
/// - pair generation (superset),
/// - post-hoc threshold sweep,
/// - diagnostic plot generation.
///
/// Angular speeds are parsed using [`AngularSpeed`], which reuses [`Angle`] for
/// the numerator and supports a small set of time units for the denominator.
/// Internally, values are converted to **radians per day**.
#[derive(Parser, Debug)]
#[command(
    name = "pairs-posthoc-sweep",
    version,
    about = "Generate pairs with a large max_omega, then post-hoc sweep omega thresholds and plot metrics.",
    long_about = None,
    disable_help_subcommand = true,
    verbatim_doc_comment,
    after_help = "\
Examples:
  pairs-posthoc-sweep alerts.parquet --out-dir out_pairs \\
    --mode fink \\
    --gen-max-omega \"30 arcsec/min\" \\
    --sweep-min-omega \"1 arcsec/min\" \\
    --sweep-steps 40 --logspace

Modes:
  --mode all     : ingest all alerts (no class/truth filtering)
  --mode fink    : keep only broker-plausible asteroid-like classes
  --mode oracle  : keep only known asteroids (truth OR 'Solar System MPC')

Tips:
  - Ensure: --gen-max-omega >= max swept omega (otherwise the sweep is biased).
  - Use --minimal/--no-minimal to control I/O (projection pushdown).
  - Start with a single night using --nid for quick iteration.
",
    help_template = "\
{before-help}{name} {version}
{about-with-newline}
{usage-heading} {usage}

{all-args}{after-help}
"
)]
struct Cli {
    /// Input ZTF/LSST-like alerts Parquet file.
    ///
    /// This file must follow the alert schema expected by
    /// `fink-fat-eval::dataset::ztf_alerts`.
    ///
    /// The dataset may contain one or multiple nights; optional filtering
    /// can be applied via `--nid`.
    #[arg(value_name = "ALERTS.parquet", help_heading = "I/O")]
    parquet: Utf8PathBuf,

    /// Output directory where all diagnostic plots will be written.
    ///
    /// The directory is created if it does not exist. Existing files
    /// may be overwritten.
    #[arg(
        short,
        long,
        default_value = "out_pairs",
        value_name = "DIR",
        help_heading = "I/O"
    )]
    out_dir: Utf8PathBuf,

    /// Optional filter on the night identifier (`nid`).
    ///
    /// When specified, only alerts belonging to this night are ingested.
    /// This is particularly useful for:
    /// - rapid debugging,
    /// - small-scale experiments,
    /// - visually inspecting a single night's behavior.
    ///
    /// If omitted, alerts from all nights present in the dataset are used.
    #[arg(long, value_name = "NID", help_heading = "Scan")]
    nid: Option<i32>,

    /// Alert loading mode (truth / broker filtering).
    ///
    /// - `oracle`: keep only known asteroids (truth OR "Solar System MPC").
    /// - `fink`: keep only broker-plausible asteroid-like classes.
    /// - `all`: keep all alerts.
    #[arg(
        long,
        value_enum,
        default_value_t = CliLoadMode::All,
        value_name = "MODE",
        help_heading = "Scan"
    )]
    mode: CliLoadMode,

    /// Use a minimal column projection when scanning the Parquet file.
    ///
    /// This reduces I/O and memory usage by loading only the columns
    /// strictly required for pair generation and diagnostics.
    ///
    /// This should almost always be enabled unless additional columns
    /// are explicitly needed for custom analysis.
    ///
    /// Disable with `--no-minimal`.
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help_heading = "Scan"
    )]
    minimal: bool,

    /// Maximum allowed time difference Δt for pair generation (days, TT).
    ///
    /// This constraint is applied **during pair generation** and defines
    /// the temporal search window for candidate pairs.
    ///
    /// This parameter is *not* swept post-hoc: changing it requires
    /// regenerating the pair superset.
    #[arg(
        long,
        default_value_t = 0.06,
        value_name = "DAYS",
        help_heading = "Pair generation"
    )]
    max_dt: f64,

    /// Maximum angular speed used to generate the pair superset.
    ///
    /// Pairs are generated with the kinematic cut:
    /// `sep(a,b) <= omega_max * dt(a,b)`.
    ///
    /// This value must be **greater than or equal to** the maximum omega
    /// threshold tested during the post-hoc sweep.
    ///
    /// A larger value increases completeness of the superset but may
    /// significantly increase runtime and memory usage.
    ///
    /// Examples:
    /// - "10 arcsec/hour"
    /// - "30 arcsec/min"
    /// - "0.02 rad/day"
    #[arg(
        long,
        default_value = "30 arcsec/min",
        value_name = "OMEGA",
        help_heading = "Pair generation",
        long_help = "Maximum angular speed used to generate the pair superset.\n\
Pairs are generated with: sep(a,b) <= omega_max * dt(a,b).\n\
This value must be >= the maximum swept omega (otherwise the sweep is biased).\n\
Examples: \"10 arcsec/hour\", \"30 arcsec/min\", \"0.02 rad/day\"."
    )]
    gen_max_omega: AngularSpeed,

    /// Minimum angular-speed threshold for the post-hoc sweep.
    ///
    /// The sweep evaluates cuts of the form `sep <= omega * dt` starting from
    /// this minimum value up to `--gen-max-omega`.
    #[arg(
        long,
        default_value = "1 arcsec/min",
        value_name = "OMEGA",
        help_heading = "Sweep"
    )]
    sweep_min_omega: AngularSpeed,

    /// Number of omega thresholds evaluated in the sweep.
    ///
    /// This controls the resolution of the tradeoff curve:
    /// - too small → coarse diagnostics,
    /// - too large → diminishing returns and longer runtime.
    #[arg(long, default_value_t = 40, value_name = "N", help_heading = "Sweep")]
    sweep_steps: usize,

    /// Use logarithmic spacing for the omega threshold sweep.
    ///
    /// Log spacing is strongly recommended when thresholds span several
    /// orders of magnitude, as it provides better resolution at small
    /// omega where performance often changes rapidly.
    ///
    /// Disable with `--no-logspace`.
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help_heading = "Sweep"
    )]
    logspace: bool,

    /// Maximum allowed flux difference between alerts in a pair.
    ///
    /// This parameter controls photometric gating during pair generation.
    /// Setting it to a very large value effectively disables flux-based
    /// filtering.
    #[arg(
        long,
        default_value_t = 1.0e9,
        value_name = "FLUX",
        help_heading = "Pair generation"
    )]
    max_flux_difference: f32,

    /// Allow pairs formed by alerts falling in the same time bin.
    ///
    /// When disabled, pairs must belong to strictly different time bins,
    /// which can reduce false positives at the cost of missing very
    /// closely spaced detections.
    #[arg(long, default_value_t = false, help_heading = "Pair generation")]
    allow_same_timebin: bool,

    /// HEALPix depth used for spatial candidate binning.
    ///
    /// Higher values correspond to smaller sky cells:
    /// - larger depth → fewer spatial candidates per cell,
    /// - smaller depth → more candidates but lower binning overhead.
    #[arg(
        long,
        default_value_t = 8,
        value_name = "DEPTH",
        help_heading = "Binning"
    )]
    healpix_depth: u8,

    /// Width of uniform time bins (days).
    ///
    /// This controls the temporal discretization used to limit candidate
    /// searches. Smaller bins reduce temporal fan-out but increase the
    /// number of bins.
    #[arg(
        long,
        default_value_t = 0.01,
        value_name = "DAYS",
        help_heading = "Binning"
    )]
    time_bin_days: f64,

    /// Width of generated plots, in pixels.
    ///
    /// This affects all output figures uniformly.
    #[arg(long, default_value_t = 1400, value_name = "PX", help_heading = "Plot")]
    width: u32,

    /// Height of generated plots, in pixels.
    ///
    /// This affects all output figures uniformly.
    #[arg(long, default_value_t = 900, value_name = "PX", help_heading = "Plot")]
    height: u32,

    /// Angular unit used for display in plots.
    ///
    /// This setting affects axis labels, tick formatting, and legends,
    /// but does not change any internal computations (which are always
    /// performed in radians).
    #[arg(
        long,
        value_enum,
        default_value_t = AngularUnit::Radian,
        value_name = "UNIT",
        help_heading = "Plot"
    )]
    angular_unit: AngularUnit,
}

/// Entry point for the post-hoc pair sweep binary.
///
/// This function:
/// * parses CLI arguments,
/// * maps them into a reusable [`PairsPosthocSweepConfig`],
/// * runs the full post-hoc sweep pipeline,
/// * writes diagnostic plots to disk.
///
/// Arguments
/// ---------
/// * None (arguments are provided via the command line).
///
/// Return
/// ------
/// * `Ok(())` on success.
/// * `Err(anyhow::Error)` if ingestion, generation, or plotting fails.
fn main() -> Result<()> {
    let cli = Cli::parse();

    let cfg = PairsPosthocSweepConfig {
        parquet: cli.parquet,
        out_dir: cli.out_dir,

        scan: ZtfAlertScan {
            nid: cli.nid,
            mode: cli.mode.into(),
            minimal: cli.minimal,
        },
        ingest: AlertIngestConfig::default(),

        max_dt_days: cli.max_dt,

        // Omega-based sweep (stored as rad/day)
        gen_max_angular_speed_rad_per_day: cli.gen_max_omega.as_rad_per_day(),
        sweep_min_angular_speed_rad_per_day: cli.sweep_min_omega.as_rad_per_day(),
        sweep_steps: cli.sweep_steps,
        logspace: cli.logspace,

        max_flux_difference: cli.max_flux_difference,
        allow_same_timebin: cli.allow_same_timebin,

        healpix_depth: cli.healpix_depth,
        time_bin_days: cli.time_bin_days,

        plot: default_plot_config(cli.width, cli.height, cli.angular_unit),
    };

    run_pairs_posthoc_sweep(&cfg)?;
    eprintln!("Wrote plots to {}", cfg.out_dir);
    Ok(())
}

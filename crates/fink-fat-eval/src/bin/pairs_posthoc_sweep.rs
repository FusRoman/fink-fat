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
//!    large angular separation (`--gen-max-sep`),
//! 3. Apply a **post-hoc sweep** of separation thresholds
//!    (`sep <= thr`) *without re-running seeding*,
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
//! * How sensitive are metrics to the angular separation cut?
//!
//! What this tool does NOT do
//! --------------------------
//! * It does **not** re-run pair generation for each threshold.
//! * It does **not** optimize multiple parameters simultaneously.
//!
//! Because of this, it is critical that `--gen-max-sep` is **greater than or
//! equal to** the maximum separation threshold tested during the sweep.
//!
//! Outputs
//! -------
//! The tool writes deterministic PNG files to `--out-dir`, typically including:
//! * Δt histogram,
//! * separation histogram,
//! * Δt vs separation scatter plot,
//! * tradeoff curve (quality vs separation threshold).
//!
//! Usage
//! -----
//! ```bash
//! cargo run -p fink-fat-eval --bin pairs-posthoc-sweep -- \
//!   alerts.parquet \
//!   --out-dir out_pairs \
//!   --gen-max-sep 0.01 rad \
//!   --sweep-min-sep 10 arcsec \
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
use clap::Parser;

use fink_fat_eval::angle::Angle;
use fink_fat_eval::dataset::ingest_config::AlertIngestConfig;
use fink_fat_eval::dataset::ztf_alerts::ZtfAlertScan;
use fink_fat_eval::seeding::pairs_sweep::{
    PairsPosthocSweepConfig, default_plot_config, run_pairs_posthoc_sweep,
};
use fink_fat_eval::seeding::plotting::AngularUnit;

/// Command-line interface for the post-hoc pair sweep tool.
///
/// This structure defines all parameters controlling:
/// - dataset ingestion,
/// - pair generation (superset),
/// - post-hoc threshold sweep,
/// - diagnostic plot generation.
///
/// All angles are parsed using [`Angle`], allowing human-friendly units
/// while maintaining an internal representation in radians.
#[derive(Parser, Debug)]
#[command(
    name = "pairs-posthoc-sweep",
    about = "Generate pairs with a large max_sep, then post-hoc sweep sep thresholds and plot metrics."
)]
struct Cli {
    /// Input ZTF/LSST-like alerts Parquet file.
    ///
    /// This file must follow the alert schema expected by
    /// `fink-fat-eval::dataset::ztf_alerts`.
    ///
    /// The dataset may contain one or multiple nights; optional filtering
    /// can be applied via `--nid`.
    #[arg(value_name = "ALERTS.parquet")]
    parquet: Utf8PathBuf,

    /// Output directory where all diagnostic plots will be written.
    ///
    /// The directory is created if it does not exist. Existing files
    /// may be overwritten.
    #[arg(short, long, default_value = "out_pairs")]
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
    #[arg(long)]
    nid: Option<i32>,

    /// Keep only alerts with an associated truth trajectory.
    ///
    /// When enabled, alerts without a known trajectory identifier are
    /// discarded at scan time. This simplifies diagnostics by removing
    /// "unknown" regions but may hide areas where truth information is
    /// missing or incomplete.
    #[arg(long)]
    only_truth: bool,

    /// Use a minimal column projection when scanning the Parquet file.
    ///
    /// This reduces I/O and memory usage by loading only the columns
    /// strictly required for pair generation and diagnostics.
    ///
    /// This should almost always be enabled unless additional columns
    /// are explicitly needed for custom analysis.
    #[arg(long, default_value_t = true)]
    minimal: bool,

    /// Maximum allowed time difference Δt for pair generation (days, TT).
    ///
    /// This constraint is applied **during pair generation** and defines
    /// the temporal search window for candidate pairs.
    ///
    /// This parameter is *not* swept post-hoc: changing it requires
    /// regenerating the pair superset.
    #[arg(long, default_value_t = 0.06)]
    max_dt: f64,

    /// Maximum angular separation used to generate the pair superset.
    ///
    /// This value must be **greater than or equal to** the maximum separation
    /// threshold tested during the post-hoc sweep.
    ///
    /// A larger value increases completeness of the superset but may
    /// significantly increase runtime and memory usage.
    #[arg(long, default_value = "0.01 rad")]
    gen_max_sep: Angle,

    /// Minimum angular separation threshold for the post-hoc sweep.
    ///
    /// The sweep evaluates cuts of the form `sep <= thr` starting from
    /// this minimum value up to `--gen-max-sep`.
    #[arg(long, default_value = "1e-5 rad")]
    sweep_min_sep: Angle,

    /// Number of separation thresholds evaluated in the sweep.
    ///
    /// This controls the resolution of the tradeoff curve:
    /// - too small → coarse diagnostics,
    /// - too large → diminishing returns and longer runtime.
    #[arg(long, default_value_t = 40)]
    sweep_steps: usize,

    /// Use logarithmic spacing for the separation threshold sweep.
    ///
    /// Log spacing is strongly recommended when thresholds span several
    /// orders of magnitude, as it provides better resolution at small
    /// separations where performance often changes rapidly.
    #[arg(long, default_value_t = true)]
    logspace: bool,

    /// Maximum allowed flux difference between alerts in a pair.
    ///
    /// This parameter controls photometric gating during pair generation.
    /// Setting it to a very large value effectively disables flux-based
    /// filtering.
    #[arg(long, default_value_t = 1.0e9)]
    max_flux_difference: f32,

    /// Allow pairs formed by alerts falling in the same time bin.
    ///
    /// When disabled, pairs must belong to strictly different time bins,
    /// which can reduce false positives at the cost of missing very
    /// closely spaced detections.
    #[arg(long, default_value_t = false)]
    allow_same_timebin: bool,

    /// HEALPix depth used for spatial candidate binning.
    ///
    /// Higher values correspond to smaller sky cells:
    /// - larger depth → fewer spatial candidates per cell,
    /// - smaller depth → more candidates but lower binning overhead.
    #[arg(long, default_value_t = 8)]
    healpix_depth: u8,

    /// Width of uniform time bins (days).
    ///
    /// This controls the temporal discretization used to limit candidate
    /// searches. Smaller bins reduce temporal fan-out but increase the
    /// number of bins.
    #[arg(long, default_value_t = 0.01)]
    time_bin_days: f64,

    /// Width of generated plots, in pixels.
    ///
    /// This affects all output figures uniformly.
    #[arg(long, default_value_t = 1400)]
    width: u32,

    /// Height of generated plots, in pixels.
    ///
    /// This affects all output figures uniformly.
    #[arg(long, default_value_t = 900)]
    height: u32,

    /// Angular unit used for display in plots.
    ///
    /// This setting affects axis labels, tick formatting, and legends,
    /// but does not change any internal computations (which are always
    /// performed in radians).
    #[arg(long, value_enum, default_value_t = AngularUnit::Radian)]
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
            only_truth: cli.only_truth,
            minimal: cli.minimal,
        },
        ingest: AlertIngestConfig::default(),

        max_dt_days: cli.max_dt,
        gen_max_sep_rad: cli.gen_max_sep.as_radians(),
        sweep_min_sep_rad: cli.sweep_min_sep.as_radians(),
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

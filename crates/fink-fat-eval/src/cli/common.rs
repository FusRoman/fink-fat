//! Shared CLI building blocks for `fink-fat-eval` binaries.
//!
//! Overview
//! --------
//! Many binaries in this crate share the same *plumbing*:
//! - reading a ZTF/LSST-like alert dataset from a Parquet file,
//! - optionally restricting the scan to a single night (`nid`),
//! - choosing a truth/broker filtering mode,
//! - configuring pair seeding (Δt, ω_max, flux gating, time-bin constraints),
//! - configuring bucketization (HEALPix + uniform time bins),
//! - configuring plotting output (size, display units).
//!
//! This module centralizes `clap` argument structs so binaries can compose
//! them via `#[command(flatten)]`.
//!
//! Why centralize CLI args?
//! ------------------------
//! - **Consistency**: binaries expose the same flags with the same semantics.
//! - **Less boilerplate**: avoid copy/paste and keep `--help` coherent.
//! - **Safer refactors**: changing default values or wording happens in one place.
//!
//! How to compose in a binary
//! --------------------------
//! ```ignore
//! #[derive(Parser)]
//! struct Cli {
//!     #[command(flatten)]
//!     scan: CommonScanArgs,
//!     #[command(flatten)]
//!     gen: CommonPairGenArgs,
//!     #[command(flatten)]
//!     binning: CommonBinningArgs,
//!     #[command(flatten)]
//!     plot: CommonPlotArgs,
//!     // ... binary-specific options ...
//! }
//! ```
//!
//! Notes on performance
//! --------------------
//! - `--minimal` is almost always recommended: it enables **projection pushdown**
//!   during Parquet scans (lower I/O and memory).
//! - `--healpix-depth` and `--time-bin-days` control candidate fan-out. They are
//!   often the dominant knobs for runtime once you move to large volumes.
//!
//! See also
//! --------
//! - [`crate::dataset::ztf_alerts::scan_ztf_alerts`]
//! - [`crate::dataset::ztf_alerts::ZtfAlertScan`]
//! - [`crate::seeding`] (pair/triplet generation and diagnostics)

use camino::Utf8PathBuf;
use clap::{ArgAction, Args, ValueEnum};

use crate::angular_speed::AngularSpeed;
use crate::dataset::ztf_alerts::AlertLoadMode;
use crate::seeding::plotting::AngularUnit;

/// CLI-facing loading mode.
///
/// This maps to [`AlertLoadMode`] in the library.
///
/// The intent is to support different evaluation scenarios:
/// - "oracle" truth-only evaluation,
/// - broker-like selection (Fink classes),
/// - or raw unfiltered ingestion.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum CliLoadMode {
    /// Load only "oracle" asteroids.
    ///
    /// Keeps alerts that are confidently associated to known Solar System objects:
    /// - `trajectory_id > 0` (truth association present), OR
    /// - `fink_class == "Solar System MPC"`.
    ///
    /// Typical use
    /// -----------
    /// * Algorithm development / debugging where you want to ignore the broker noise floor.
    /// * Measuring upper-bound performance (best-case purity/recall).
    Oracle,

    /// Load broker-plausible asteroid-like alerts (truth + candidates).
    ///
    /// Keeps:
    /// - `fink_class ∈ {"Solar System MPC", "Solar System candidate"}`, OR
    /// - `fink_class == "Unknown"` and `nalerthist <= 1` (early unknowns).
    ///
    /// Typical use
    /// -----------
    /// * Stress-testing seeding under a "broker-realistic" mixture.
    /// * Studying contamination introduced by uncertain/early alerts.
    FinkTruth,

    /// Load broker-plausible asteroid-like alerts + candidates (more aggressive).
    ///
    /// Keeps:
    /// - `fink_class == "Solar System candidate"`, OR
    /// - `fink_class == "Unknown"` and `nalerthist <= 1`.
    ///
    /// Typical use
    /// -----------
    /// * Evaluating seeding when the broker is *not* providing known MPC objects,
    ///   e.g. focusing on discovery-like candidates.
    FinkCandidate,

    /// Load everything (no truth/class filtering).
    ///
    /// Typical use
    /// -----------
    /// * Benchmarking I/O and scanning cost.
    /// * Measuring worst-case fan-out and memory pressure.
    All,
}

impl From<CliLoadMode> for AlertLoadMode {
    fn from(v: CliLoadMode) -> Self {
        match v {
            CliLoadMode::Oracle => AlertLoadMode::Oracle,
            CliLoadMode::FinkTruth => AlertLoadMode::FinkTruth,
            CliLoadMode::FinkCandidate => AlertLoadMode::FinkCandidate,
            CliLoadMode::All => AlertLoadMode::All,
        }
    }
}

// Common scan + I/O arguments shared by most tools.
//
// These options control *what* is read from the Parquet dataset
// and *where* outputs are written.
//
// Notes
// -----
// - Most tools accept multi-night Parquet files. Use `--nid` to restrict to a
//   single night for quick iteration.
// - `--minimal` should stay enabled unless you explicitly need non-core columns
//   for custom diagnostics.
#[derive(Args, Debug, Clone)]
pub struct CommonScanArgs {
    /// Input ZTF/LSST-like alerts Parquet file.
    ///
    /// The file must follow the alert schema expected by `crate::dataset::ztf_alerts`.
    #[arg(value_name = "ALERTS.parquet", help_heading = "I/O")]
    pub parquet: Utf8PathBuf,

    /// Output directory (plots, CSV summaries, etc.).
    ///
    /// The directory is created if it does not exist. Existing files may be overwritten.
    #[arg(
        short,
        long,
        default_value = "out",
        value_name = "DIR",
        help_heading = "I/O"
    )]
    pub out_dir: Utf8PathBuf,

    /// Optional filter on the night identifier (`nid`).
    ///
    /// When set, only alerts belonging to this night are ingested.
    #[arg(long, value_name = "NID", help_heading = "Scan")]
    pub nid: Option<u32>,

    /// Alert loading mode (truth / broker filtering).
    #[arg(
        long,
        value_enum,
        default_value_t = CliLoadMode::All,
        value_name = "MODE",
        help_heading = "Scan"
    )]
    pub mode: CliLoadMode,

    /// Use a minimal column projection when scanning the Parquet file.
    ///
    /// When enabled, only the columns required for seeding + truth metrics are read.
    /// This reduces:
    /// 
    /// - Parquet I/O (projection pushdown),
    /// 
    /// - memory use,
    /// 
    /// - deserialization overhead.
    ///
    /// You should disable this only if a downstream analysis explicitly needs extra columns.
    ///
    /// Disable with `--no-minimal`.
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help_heading = "Scan"
    )]
    pub minimal: bool,
}

// Common binning parameters (spatial + temporal discretization).
//
// These parameters control how candidate searches are bucketized to keep
// pair generation tractable.
//
// Design intuition :
// 
// - Higher `--healpix-depth` → smaller sky cells → fewer candidates per cell
//   (often faster, but more overhead in indexing).
// 
// - Smaller `--time-bin-days` → finer time bins → fewer candidates per bin
//   (often faster, but more bins to manage).
//
// If runtime explodes, these are usually the first knobs to inspect.
#[derive(Args, Debug, Clone)]
pub struct CommonBinningArgs {
    /// HEALPix depth used for spatial candidate binning.
    ///
    /// Higher values correspond to smaller sky pixels.
    ///
    /// Tradeoff :
    /// 
    /// - Larger depth: lower candidate fan-out, potentially faster pair search.
    /// 
    /// - Too large: more indexing overhead and risk of splitting close neighbors
    /// across many pixels (may require looking at neighbor pixels).
    #[arg(
        long,
        default_value_t = 8,
        value_name = "DEPTH",
        help_heading = "Binning"
    )]
    pub healpix_depth: u8,

    /// Width of uniform time bins (days).
    ///
    /// This controls temporal discretization for bucket-based candidate search.
    ///
    /// Notes :
    /// 
    /// - Smaller bins reduce temporal fan-out but increase the number of bins.
    /// 
    /// - This is independent from `--max-dt` which controls the *search horizon*.
    #[arg(
        long,
        default_value_t = 0.01,
        value_name = "DAYS",
        help_heading = "Binning"
    )]
    pub time_bin_days: f64,
}

// Common pair-seeding generation arguments.
//
// These parameters define *which* candidate pairs can be generated.
// They typically have the largest impact on seeding volume and quality.
//
// Key constraints
// ---------------
// - Time window: `dt(a,b) <= max_dt`
// - Kinematic gate: `sep(a,b) <= omega_max * dt(a,b)`
// - Optional photometric gate: `|flux(a) - flux(b)| <= max_flux_difference`
// - Optional time-bin rule: whether same-bin pairs are allowed
#[derive(Args, Debug, Clone)]
pub struct CommonPairGenArgs {
    /// Maximum allowed time difference Δt for pair generation (days, TT).
    ///
    /// This defines the temporal search window. Larger values:
    /// - increase completeness (more potential true pairs),
    /// - but can sharply increase the number of candidates and runtime.
    #[arg(
        long,
        default_value_t = 0.06,
        value_name = "DAYS",
        help_heading = "Pair generation"
    )]
    pub max_dt: f64,

    /// Maximum angular speed used to generate pairs.
    ///
    /// Candidate pairs must satisfy:
    /// `sep(a,b) <= omega_max * dt(a,b)`.
    ///
    /// This is the main "motion plausibility" gate.
    #[arg(
        long,
        default_value = "30 arcsec/min",
        value_name = "OMEGA",
        help_heading = "Pair generation"
    )]
    pub gen_max_omega: AngularSpeed,

    /// Maximum allowed flux difference between alerts in a pair.
    ///
    /// This provides a coarse photometric consistency gate.
    /// Setting a very large value effectively disables flux-based filtering.
    ///
    /// Notes :
    /// 
    /// - The unit corresponds to whatever "flux" quantity the engine uses for
    ///   gating (often derived from magnitude; see engine config).
    /// 
    /// - In many survey-like datasets, photometric gating is a second-order
    ///   effect compared to kinematic gating.
    #[arg(
        long,
        default_value_t = 1.0e9,
        value_name = "FLUX",
        help_heading = "Pair generation"
    )]
    pub max_flux_difference: f32,

    /// Allow pairs formed by alerts falling in the same time bin.
    ///
    /// When disabled (default), pairs must belong to strictly different time bins.
    #[arg(long, default_value_t = false, help_heading = "Pair generation")]
    pub allow_same_timebin: bool,
}

// Common plot sizing + display parameters.
//
// These options affect only output rendering (PNG size and angle units).
// They do not change internal computations (which use radians).
#[derive(Args, Debug, Clone)]
pub struct CommonPlotArgs {
    /// Width of generated plots, in pixels.
    ///
    /// Larger values improve readability (labels, legends) at the cost of file size.
    #[arg(long, default_value_t = 1400, value_name = "PX", help_heading = "Plot")]
    pub width: u32,

    /// Height of generated plots, in pixels.
    #[arg(long, default_value_t = 900, value_name = "PX", help_heading = "Plot")]
    pub height: u32,

    /// Angular unit used for display in plots.
    ///
    /// This affects axis labels and tick formatting only.
    #[arg(
        long,
        value_enum,
        default_value_t = AngularUnit::Radian,
        value_name = "UNIT",
        help_heading = "Plot"
    )]
    pub angular_unit: AngularUnit,
}

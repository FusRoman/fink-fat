//! Post-hoc sweep utilities for pair-seeding analysis.
//!
//! Overview
//! --------
//! This module implements a reusable pipeline to **tune** and **diagnose**
//! *pair-seeding* thresholds without re-running pair generation for every
//! candidate threshold.
//!
//! The workflow is designed for survey-scale datasets (ZTF/LSST-like):
//!
//! 1. **Scan + ingest** an alert Parquet dataset into an [`AlertStoreWithTruth`].
//! 2. **Generate pairs once** using a deliberately large maximum separation
//!    (`gen_max_sep_rad`) so the candidate set is a superset.
//! 3. **Extract features** on that fixed superset (Δt, separation, labels).
//! 4. **Plot diagnostics** (histograms, scatter) on the full superset.
//! 5. **Sweep post-hoc cuts** of the form `sep <= thr` for a grid of thresholds,
//!    and plot a tradeoff curve (e.g., quality vs threshold).
//!
//! This approach is extremely useful to answer questions like:
//! - Where is the knee point between completeness and contamination?
//! - Do false links concentrate at large Δt or large separation?
//! - How sensitive are metrics to the separation threshold?
//!
//! What this module does NOT do
//! ----------------------------
//! * It does **not** re-run seeding for each threshold. The sweep is purely a
//!   post-hoc filter on a pre-generated superset.
//! * It does **not** perform a multi-parameter optimization loop (only a 1D
//!   separation threshold sweep, as written here).
//!
//! Notes
//! -----
//! * You must ensure `gen_max_sep_rad >= sweep_min_sep_rad` and that
//!   `gen_max_sep_rad` is at least as large as the **largest** threshold you
//!   want to test.
//! * The uniform time binner origin `t0` is inferred from the minimum `mjd_tt`
//!   in the dataset by default (see [`infer_t0_mjd_tt`]) to make binning stable
//!   and reproducible for a fixed dataset.
//!
//! See also
//! --------
//! * [`crate::seeding::plotting`] – Plot implementations and feature extraction.
//! * [`crate::seeding::seed_gen`] – Pair/triplet generation entry points.

use anyhow::{Context, Result};
use camino::Utf8PathBuf;

use fink_fat_engine::engine_config::pair_config::PairConfig;
use fink_fat_engine::spacetime_bucket::healpix_binner::HealpixBinner;
use fink_fat_engine::spacetime_bucket::uniform_time_binner::UniformTimeBinner;
use fink_fat_engine::{MjdTt, Radians};

use crate::FiniteOr;
use crate::dataset::ztf_alerts::{
    AlertStoreWithTruth, ZtfAlertScan, alert_store_with_truth_from_lazyframe, scan_ztf_alerts,
};
use crate::dataset::{ParquetSource, ingest_config::AlertIngestConfig};
use crate::grid::{linspace, logspace};
use crate::seeding::plotting::{
    AngularUnit, PairPlotConfig, extract_pair_features, plot_pairs_dt_hist,
    plot_pairs_scatter_dt_sep, plot_pairs_sep_hist, plot_pairs_tradeoff_vs_sep_threshold,
};
use crate::seeding::seed_gen::generate_pairs_and_triplets_ids_only;

/// Configuration for a post-hoc sweep on pair separation thresholds.
///
/// This configuration is intentionally *portable* across binaries and tests:
/// it contains all inputs needed to ingest a dataset, generate a pair superset,
/// and produce a suite of diagnostic plots.
///
/// Arguments
/// ---------
/// * `parquet` – Input alert Parquet file (UTF-8 path).
/// * `out_dir` – Output directory where plots will be written.
/// * `scan` – Scan configuration applied at Parquet read time (e.g., `nid`,
///   `only_truth`, `minimal` projection).
/// * `ingest` – Ingestion configuration (schema expectations, normalization).
/// * `max_dt_days` – Pair generation maximum Δt (days, TT). This is applied
///   during generation and is **not** swept post-hoc.
/// * `gen_max_sep_rad` – Maximum separation (radians) used to generate the
///   superset of pairs.
/// * `sweep_min_sep_rad` – Minimum separation threshold (radians) for the
///   post-hoc sweep.
/// * `sweep_steps` – Number of thresholds in the sweep grid.
/// * `logspace` – If `true`, use a log-spaced sweep grid; otherwise linear.
/// * `max_flux_difference` – Photometric gating for pair generation.
/// * `allow_same_timebin` – Whether pairs within the same time bin are allowed.
/// * `healpix_depth` – HEALPix depth used for spatial candidate binning.
/// * `time_bin_days` – Uniform time-bin width (days).
/// * `plot` – Plot configuration (size, display angular unit, etc.).
///
/// Return
/// ------
/// * This is a pure configuration container and does not return a value.
///
/// Notes
/// -----
/// * The post-hoc sweep only changes the separation threshold `sep <= thr`.
///   If you want to study the interaction between Δt and separation, you must
///   either re-run generation with different `max_dt_days` values or extend this
///   module to support a 2D sweep.
/// * `gen_max_sep_rad` must be chosen conservatively so the generated superset
///   remains tractable in memory/time while still covering the sweep range.
#[derive(Clone, Debug)]
pub struct PairsPosthocSweepConfig {
    pub parquet: Utf8PathBuf,
    pub out_dir: Utf8PathBuf,

    pub scan: ZtfAlertScan,
    pub ingest: AlertIngestConfig,

    pub max_dt_days: f64,
    pub gen_max_sep_rad: f64,
    pub sweep_min_sep_rad: f64,
    pub sweep_steps: usize,
    pub logspace: bool,

    pub max_flux_difference: f32,
    pub allow_same_timebin: bool,

    pub healpix_depth: u8,
    pub time_bin_days: f64,

    pub plot: PairPlotConfig,
}

impl PairsPosthocSweepConfig {
    /// Validate configuration and fail fast with explicit error messages.
    ///
    /// This method checks numeric sanity constraints that would otherwise
    /// produce misleading plots (or `NaN` grids) later in the pipeline.
    ///
    /// Arguments
    /// ---------
    /// * `self` – Sweep configuration to validate.
    ///
    /// Return
    /// ------
    /// * `Ok(())` if the configuration passes basic validation.
    /// * `Err(anyhow::Error)` if any field violates required constraints.
    ///
    /// Notes
    /// -----
    /// * This routine validates **only** generic invariants (finiteness,
    ///   positivity, ordering). It does not check that the dataset itself is
    ///   consistent with the chosen parameters.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.max_dt_days.is_finite() && self.max_dt_days >= 0.0,
            "max_dt must be finite and >= 0"
        );
        anyhow::ensure!(
            self.gen_max_sep_rad.is_finite() && self.gen_max_sep_rad >= 0.0,
            "gen_max_sep must be finite and >= 0"
        );
        anyhow::ensure!(
            self.sweep_min_sep_rad.is_finite() && self.sweep_min_sep_rad > 0.0,
            "sweep_min_sep must be finite and > 0"
        );
        anyhow::ensure!(
            self.gen_max_sep_rad >= self.sweep_min_sep_rad,
            "gen_max_sep must be >= sweep_min_sep"
        );
        anyhow::ensure!(
            self.time_bin_days.is_finite() && self.time_bin_days > 0.0,
            "time_bin_days must be finite and > 0"
        );
        anyhow::ensure!(self.sweep_steps > 0, "sweep_steps must be > 0");
        Ok(())
    }
}

/// Run the full post-hoc sweep pipeline and write plots to disk.
///
/// Steps
/// -----
/// 1. Scan + ingest alerts into [`AlertStoreWithTruth`].
/// 2. Build spatial/time binners (HEALPix + uniform bins).
/// 3. Generate a superset of pairs with `gen_max_sep_rad`.
/// 4. Extract features and plot diagnostics on the superset.
/// 5. Sweep `sep <= thr` thresholds post-hoc and plot the tradeoff curve.
///
/// Arguments
/// ---------
/// * `cfg` – Pipeline configuration (dataset, generation parameters, sweep grid,
///   and plotting configuration).
///
/// Return
/// ------
/// * `Ok(())` on success (plots written to `cfg.out_dir`).
/// * `Err(anyhow::Error)` if ingestion, generation, feature extraction, or
///   plotting fails.
///
/// Notes
/// -----
/// * The sweep is purely post-hoc: it filters the already-generated superset.
///   If `cfg.gen_max_sep_rad` is too small, the sweep results will be biased
///   because candidate pairs beyond that separation never existed.
/// * Output files are deterministic given a fixed dataset and configuration
///   (subject to floating-point rounding and any nondeterminism upstream).
pub fn run_pairs_posthoc_sweep(cfg: &PairsPosthocSweepConfig) -> Result<()> {
    cfg.validate()?;

    std::fs::create_dir_all(&cfg.out_dir)
        .with_context(|| format!("failed to create output dir: {}", cfg.out_dir))?;

    // 1) Ingest Parquet -> AlertStoreWithTruth
    let source = ParquetSource::new(&cfg.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cfg.parquet))?;

    let lf = scan_ztf_alerts(&source, cfg.scan.clone())?;
    let store = alert_store_with_truth_from_lazyframe(lf, cfg.ingest.clone())?;

    // 2) Build binners (engine implementations)
    let spatial = HealpixBinner::new(cfg.healpix_depth);
    let t0 = infer_t0_mjd_tt(&store);
    let time = UniformTimeBinner::new(t0, cfg.time_bin_days);

    // 3) Generate pairs once with a large `gen_max_sep_rad`
    let pair_cfg = PairConfig {
        max_dt: cfg.max_dt_days,
        max_sep: cfg.gen_max_sep_rad,
        max_flux_difference: cfg.max_flux_difference,
        allow_same_timebin: cfg.allow_same_timebin,
    };

    let (pairs, _triplets, _timings) = generate_pairs_and_triplets_ids_only(
        &store,
        &spatial,
        &time,
        &pair_cfg,
        &Default::default(),
    );

    // 4) Extract features + produce diagnostic plots
    let feats = extract_pair_features(&store, &pairs)?;
    let out_dir_std = cfg.out_dir.as_std_path();

    plot_pairs_dt_hist(&feats, out_dir_std, &cfg.plot)?;
    plot_pairs_sep_hist(&feats, out_dir_std, &cfg.plot)?;
    plot_pairs_scatter_dt_sep(&feats, out_dir_std, &cfg.plot)?;

    // 5) Post-hoc sweep: vary separation threshold without re-running seeding
    let thresholds: Vec<Radians> = if cfg.logspace {
        logspace(cfg.sweep_min_sep_rad, cfg.gen_max_sep_rad, cfg.sweep_steps)
    } else {
        linspace(cfg.sweep_min_sep_rad, cfg.gen_max_sep_rad, cfg.sweep_steps)
    };

    plot_pairs_tradeoff_vs_sep_threshold(
        &store,
        &pairs,
        &feats,
        &thresholds,
        out_dir_std,
        &cfg.plot,
    )?;

    Ok(())
}

/// Infer a robust origin `t0` for uniform time bins from the dataset.
///
/// The uniform time binner computes bin indices as:
///
/// ```text
/// bin(t) = floor((t - t0) / dt)
/// ```
///
/// Choosing `t0 = min(mjd_tt)` makes the binning deterministic for a fixed
/// dataset, which is sufficient for post-hoc studies and diagnostic tools.
///
/// Arguments
/// ---------
/// * `store` – Ingested alerts and truth sidecar.
///
/// Return
/// ------
/// * `MjdTt` – The minimum `mjd_tt` found in the dataset.
///   If the store is empty or only contains non-finite times, returns `0.0`.
///
/// Notes
/// -----
/// * For production pipelines that compare results across datasets, you may
///   prefer a fixed global origin (e.g., a reference MJD) instead of `min(t)`.
pub fn infer_t0_mjd_tt(store: &AlertStoreWithTruth) -> MjdTt {
    store
        .store
        .alerts
        .iter()
        .map(|a| a.mjd_tt)
        .fold(f64::INFINITY, |acc, x| acc.min(x))
        .if_finite_or(0.0)
}

/// Convenience helper to build a [`PairPlotConfig`] with a chosen angular unit and size.
///
/// This is a small ergonomic utility meant for binaries that want consistent
/// default plot styling without repeating boilerplate.
///
/// Arguments
/// ---------
/// * `width` – Plot width in pixels.
/// * `height` – Plot height in pixels.
/// * `angular_unit` – Angular unit used for display (axes labels, legends).
///
/// Return
/// ------
/// * `PairPlotConfig` – A plot configuration initialized from defaults, then
///   overridden with the provided size and angular unit.
///
/// Notes
/// -----
/// * This helper does not modify any other plotting parameters (fonts, margins,
///   etc.). For more customization, start from `PairPlotConfig::default()` and
///   override additional fields.
pub fn default_plot_config(width: u32, height: u32, angular_unit: AngularUnit) -> PairPlotConfig {
    let mut cfg = PairPlotConfig::default();
    cfg.width = width;
    cfg.height = height;
    cfg.angular_unit = angular_unit;
    cfg
}

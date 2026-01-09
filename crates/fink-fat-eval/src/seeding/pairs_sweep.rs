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
//! 2. **Generate pairs once** using a deliberately large maximum angular speed
//!    (`gen_max_angular_speed_rad_per_day`) so the candidate set is a superset.
//! 3. **Extract features** on that fixed superset (Δt, separation, labels).
//! 4. **Plot diagnostics** (histograms, scatter) on the full superset.
//! 5. **Sweep post-hoc kinematic cuts** of the form `sep <= ω_thr * dt` for a
//!    grid of ω thresholds, and plot a tradeoff curve.
//!
//! This approach answers questions like:
//! - Where is the knee point between completeness and contamination?
//! - Do false links concentrate at large Δt or large separation?
//! - How sensitive are metrics to the **angular speed** threshold?
//!
//! Notes
//! -----
//! * You must ensure `gen_max_angular_speed_rad_per_day` is at least as large
//!   as the **largest** ω threshold you want to test.
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

use crate::bin_utils::infer_t0_mjd_tt;
use crate::dataset::ztf_alerts::{
    ZtfAlertScan, alert_store_with_truth_from_lazyframe, scan_ztf_alerts,
};
use crate::dataset::{ParquetSource, ingest_config::AlertIngestConfig};
use crate::grid::{linspace, logspace};
use crate::seeding::plotting::{
    AngularUnit, PairPlotConfig, extract_pair_features, plot_pairs_cost_vs_omega_threshold,
    plot_pairs_dt_hist, plot_pairs_global_tradeoff_vs_omega_threshold, plot_pairs_omega_hist,
    plot_pairs_scatter_dt_sep, plot_pairs_sep_hist, plot_pairs_tradeoff_vs_omega_threshold,
};
use crate::seeding::seed_gen::generate_pairs_and_triplets_ids_only;

/// Configuration for a post-hoc sweep on pair **angular-speed** thresholds.
///
/// Post-hoc sweep cut:
/// -------------------
/// A pair (a,b) is **kept** under threshold ω if:
///
/// `sep(a,b) <= ω * dt(a,b)`
///
/// where:
/// - `sep` is great-circle separation (radians),
/// - `dt` is `mjd_tt(b) - mjd_tt(a)` (days),
/// - ω is in radians/day.
///
/// Generation superset:
/// --------------------
/// Pairs are generated once using `gen_max_angular_speed_rad_per_day` (large),
/// so post-hoc filtering only removes pairs.
#[derive(Clone, Debug)]
pub struct PairsPosthocSweepConfig {
    pub parquet: Utf8PathBuf,
    pub out_dir: Utf8PathBuf,

    pub scan: ZtfAlertScan,
    pub ingest: AlertIngestConfig,

    /// Pair generation max Δt (days, TT). Not swept post-hoc.
    pub max_dt_days: f64,

    /// Max angular speed used to generate the pair superset (rad/day).
    pub gen_max_angular_speed_rad_per_day: f64,

    /// Minimum ω threshold for the post-hoc sweep (rad/day).
    pub sweep_min_angular_speed_rad_per_day: f64,

    /// Number of thresholds in the sweep grid.
    pub sweep_steps: usize,

    /// If true, use a log-spaced ω sweep grid.
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
            self.gen_max_angular_speed_rad_per_day.is_finite()
                && self.gen_max_angular_speed_rad_per_day >= 0.0,
            "gen_max_angular_speed must be finite and >= 0"
        );
        anyhow::ensure!(
            self.sweep_min_angular_speed_rad_per_day.is_finite()
                && self.sweep_min_angular_speed_rad_per_day > 0.0,
            "sweep_min_angular_speed must be finite and > 0"
        );
        anyhow::ensure!(
            self.gen_max_angular_speed_rad_per_day >= self.sweep_min_angular_speed_rad_per_day,
            "gen_max_angular_speed must be >= sweep_min_angular_speed"
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
/// 3. Generate a superset of pairs with `gen_max_angular_speed`.
/// 4. Extract features and plot diagnostics on the superset.
/// 5. Sweep ω thresholds post-hoc and plot the tradeoff curve.
///
/// Notes
/// -----
/// * The sweep is purely post-hoc: it filters the already-generated superset.
///   If `cfg.gen_max_angular_speed_rad_per_day` is too small, results will be biased.
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

    // 3) Generate pairs once with a large `gen_max_angular_speed`
    let pair_cfg = PairConfig {
        max_dt: cfg.max_dt_days,
        max_angular_speed: cfg.gen_max_angular_speed_rad_per_day,
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

    // 4) Extract features + diagnostic plots
    let feats = extract_pair_features(&store, &pairs)?;
    let out_dir_std = cfg.out_dir.as_std_path();

    plot_pairs_dt_hist(&feats, out_dir_std, &cfg.plot)?;
    plot_pairs_sep_hist(&feats, out_dir_std, &cfg.plot)?;
    plot_pairs_scatter_dt_sep(&feats, out_dir_std, &cfg.plot)?;

    // 5) Post-hoc sweep: vary ω threshold without re-running seeding
    let thresholds_omega: Vec<f64> = if cfg.logspace {
        logspace(
            cfg.sweep_min_angular_speed_rad_per_day,
            cfg.gen_max_angular_speed_rad_per_day,
            cfg.sweep_steps,
        )
    } else {
        linspace(
            cfg.sweep_min_angular_speed_rad_per_day,
            cfg.gen_max_angular_speed_rad_per_day,
            cfg.sweep_steps,
        )
    };

    plot_pairs_tradeoff_vs_omega_threshold(
        &store,
        &pairs,
        &feats,
        &thresholds_omega,
        out_dir_std,
        &cfg.plot,
    )?;

    plot_pairs_global_tradeoff_vs_omega_threshold(
        &store,
        &pairs,
        &feats,
        &thresholds_omega,
        out_dir_std,
        &cfg.plot,
    )?;

    plot_pairs_cost_vs_omega_threshold(
        &store,
        &pairs,
        &feats,
        &thresholds_omega,
        out_dir_std,
        &cfg.plot,
    )?;

    plot_pairs_omega_hist(&feats, out_dir_std, &cfg.plot)?;

    Ok(())
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

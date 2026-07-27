//! Domain logic for the Kalman-filter-bank evaluation run.
//!
//! This module owns everything that touches numbers: turning a [`TrajId`]
//! into actual observations, running the topocentric Kalman filter bank on
//! it, reducing the per-step results into a one-row [`TrajSummary`], and
//! aggregating/ranking those summaries across the whole dataset.
//!
//! Printing of *results* is intentionally kept out of this module (see
//! [`crate::reporting`]). The only output produced here is a live progress
//! bar (see [`process_all_trajectories`]) plus one final per-stage timing
//! summary — no periodic/repeated log lines.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use anyhow::Result;
use fink_fat_engine::engine_config::grid_population::GridConfig;
use fink_fat_engine::engine_config::kalman_context::KalmanContext;
use fink_fat_engine::engine_config::kf_bank_config::KFBankConfig;
use fink_fat_engine::engine_config::night_advance_params::NightAdvanceParams;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::Vector6;
use outfit::OrbitalElements;
use rayon::prelude::*;

use fink_fat_engine::error::{EngineError, FinkFatError};
use photom::{
    TrajId,
    observation_dataset::{ObsDataset, iter::MemLayoutObservations, observation::Observation},
};

use crate::ground_truth_state::TruthLookup;
use crate::kalman_traj::{
    NEES_CHI2_2DOF_HIGH, NEES_CHI2_2DOF_LOW, NEES_CHI2_6DOF_HIGH, NEES_CHI2_6DOF_LOW,
    NIS_CHI2_2DOF_HIGH, NIS_CHI2_2DOF_LOW, NIS_CHI2_2DOF_MEDIAN, ObserverGeometryCache,
    StudyOutcome, TrajStopReason, study_kalman_asteroid,
};

// ── Trajectory materialization ──────────────────────────────────────────

/// Fetch the observations of `traj` from `obs_dataset` as a contiguous
/// slice, cloning only if the underlying storage has them split across
/// non-adjacent memory.
pub fn materialize_contiguous_traj<'o>(
    obs_dataset: &'o ObsDataset,
    traj: &TrajId,
) -> Result<Cow<'o, [Observation]>, EngineError> {
    match obs_dataset.materialize_trajectory(traj).ok_or_else(|| {
        EngineError::FinkFat(FinkFatError::Message(format!(
            "failed to metariaze trajectory with id: {}",
            traj
        )))
    })? {
        MemLayoutObservations::Contiguous(slice) => Ok(Cow::Borrowed(slice)),
        MemLayoutObservations::Split(vec_obs) => {
            Ok(Cow::Owned(vec_obs.iter().map(|o| (*o).clone()).collect()))
        }
    }
}

// ── Generic statistics helpers ────────────────────────────────────────────

/// Mean / median / min / max of a metric, computed over any slice of items
/// via an accessor closure. Used both to collapse a single trajectory's
/// per-step [`KFStudyResult`]s into one [`TrajSummary`], and to collapse the
/// whole dataset's [`TrajSummary`]s into one global report (see
/// [`crate::reporting::print_global_aggregate_stats`]).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MetricStats {
    #[serde(with = "finite_f64")]
    pub mean: f64,
    #[serde(with = "finite_f64")]
    pub median: f64,
    #[serde(with = "finite_f64")]
    pub min: f64,
    #[serde(with = "finite_f64")]
    pub max: f64,
}

/// Serde helper (de)serializing `f64`, tolerating `NaN`/`±Infinity`.
///
/// Plain JSON has no representation for these; `serde_json`'s default `f64`
/// impl silently emits `null` for them, which then fails to parse back
/// (`null` isn't a valid `f64`). [`MetricStats`] legitimately produces `NaN`
/// for empty samples (see [`compute_stats`]), so its fields round-trip
/// through this instead of a bare `f64`.
pub(crate) mod finite_f64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() {
            serializer.serialize_f64(*value)
        } else {
            serializer.serialize_str(&value.to_string())
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum NumOrStr {
            Num(f64),
            Str(String),
        }
        match NumOrStr::deserialize(deserializer)? {
            NumOrStr::Num(v) => Ok(v),
            NumOrStr::Str(s) => s.parse().map_err(serde::de::Error::custom),
        }
    }
}

/// Compute [`MetricStats`] over a slice of raw values.
///
/// Returns all-`NAN` stats for an empty slice rather than panicking, since
/// callers may legitimately hand this an empty metric series (e.g. a
/// trajectory that produced zero usable steps).
fn compute_stats(values: &[f64]) -> MetricStats {
    if values.is_empty() {
        return MetricStats {
            mean: f64::NAN,
            median: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        };
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = median_of_sorted(&sorted);

    MetricStats {
        mean,
        median,
        min: sorted[0],
        max: *sorted.last().unwrap(),
    }
}

/// Median of an already-sorted slice (average of the two middle elements
/// when `n` is even).
fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Mean and (population) standard deviation of a slice of values.
///
/// Used only for the final per-stage timing summary in [`log_run_completion`];
/// unlike [`MetricStats`] it doesn't need sorting or median/min/max.
fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

/// Apply `f` to every item and compute [`MetricStats`] over the results.
pub fn metric_stats<T>(items: &[T], f: impl Fn(&T) -> f64) -> MetricStats {
    let values: Vec<f64> = items.iter().map(f).collect();
    compute_stats(&values)
}

/// Like [`metric_stats`], but for ground-truth-derived metrics that are only
/// present for a subset of steps (`None` where no ground truth was
/// available) — those steps are simply excluded rather than counted as NaN.
pub fn metric_stats_opt<T>(items: &[T], f: impl Fn(&T) -> Option<f64>) -> MetricStats {
    let values: Vec<f64> = items
        .iter()
        .filter_map(&f)
        .filter(|v| v.is_finite())
        .collect();
    compute_stats(&values)
}

/// Root-mean-square of a ground-truth-derived error metric, skipping steps
/// with no ground truth (`None`) or non-finite values.
pub fn rmse_opt<T>(items: &[T], f: impl Fn(&T) -> Option<f64>) -> f64 {
    let values: Vec<f64> = items
        .iter()
        .filter_map(&f)
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return f64::NAN;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

/// Percentage of items for which `pred` holds.
fn pct_true<T>(items: &[T], pred: impl Fn(&T) -> bool) -> f64 {
    if items.is_empty() {
        return f64::NAN;
    }
    100.0 * items.iter().filter(|x| pred(x)).count() as f64 / items.len() as f64
}

/// Like [`pct_true`], but for a ground-truth-derived predicate that's only
/// defined for a subset of steps — the denominator is the count of steps
/// with a known ground truth, not the whole trajectory.
fn pct_true_opt<T>(items: &[T], pred: impl Fn(&T) -> Option<bool>) -> f64 {
    let (n_true, n_total) = items
        .iter()
        .filter_map(&pred)
        .fold((0usize, 0usize), |(t, n), b| (t + b as usize, n + 1));
    if n_total == 0 {
        f64::NAN
    } else {
        100.0 * n_true as f64 / n_total as f64
    }
}

/// Format [`MetricStats`] as `mean (med median) [min, max]`, right-aligned
/// for tabular output.
pub fn fmt_stats(stats: &MetricStats) -> String {
    format!(
        "{:>9.4} (med {:>9.4})  [{:>9.4}, {:>9.4}]",
        stats.mean, stats.median, stats.min, stats.max
    )
}

// ── Per-trajectory summary ─────────────────────────────────────────────────

/// One-row digest of a trajectory's [`KFStudyResult`] series, used for
/// dataset-wide aggregation and for ranking trajectories by filter quality.
#[derive(Debug, Clone)]
pub struct TrajSummary {
    pub traj_id: TrajId,
    pub final_kf_state: Vector6<f64>,
    pub estimated_orbit: OrbitalElements,
    /// Raw number of observations in the trajectory (before bootstrap).
    pub n_obs_total: usize,
    /// Number of predict/update steps that actually produced a result.
    pub n_steps: usize,
    /// `n_steps` over [`StudyOutcome::n_processable`] (1.0 means every
    /// observation actually available to the loop — after the bootstrap
    /// pair and any deduplicated epochs — was processed).
    ///
    /// Deliberately *not* `n_obs_total - 2`: a trajectory whose first
    /// same-night pair only appears after some inter-night singletons has
    /// those singletons skipped by [`crate::kalman_traj::init_bank_from_first_pair`]
    /// before the loop even starts — dividing by `n_obs_total - 2` would
    /// then bias `completion_fraction` below 1.0 even on a perfect run.
    pub completion_fraction: f64,
    /// Why the predict/update loop stopped — see
    /// [`crate::kalman_traj::TrajStopReason`].
    pub stop_reason: TrajStopReason,
    /// Observations skipped before the bootstrap pair (inter-night
    /// singletons the first-same-night-pair scan had to pass over) — `0`
    /// unless the trajectory needed more than 2 observations to find a
    /// bootstrap pair.
    pub n_obs_before_bootstrap: usize,
    /// Observations dropped by epoch deduplication (near-identical epoch to
    /// another observation) before bootstrapping — see
    /// [`crate::kalman_traj::TrajStopReason`]'s module doc.
    pub n_obs_deduplicated: usize,
    /// Observations available to the predict/update loop after the
    /// bootstrap pair — the denominator of `completion_fraction`.
    pub n_processable: usize,
    pub pct_within_3sigma: f64,
    pub pct_within_search_radius: f64,
    pub mean_separation_arcsec: f64,
    pub median_separation_arcsec: f64,
    pub mean_nis: f64,
    /// Median NIS over this trajectory's steps — more robust than the mean
    /// to the occasional huge-innovation outlier (see
    /// [`Self::nis_calibration_ratio`]).
    pub nis_median: f64,
    /// Percentage of steps whose NIS falls within the χ²(2) 95% interval
    /// `[NIS_CHI2_2DOF_LOW, NIS_CHI2_2DOF_HIGH]` — the fraction a
    /// well-calibrated filter would put there by construction (95%).
    pub pct_nis_in_chi2_band: f64,
    /// `nis_median / NIS_CHI2_2DOF_MEDIAN` — ≈1.0 for a well-calibrated
    /// filter, `≪ 1` means the filter's predicted covariance is
    /// systematically too large (over-covariant: real residuals are small
    /// compared to what the filter expects), `≫ 1` means it's
    /// over-confident (covariance too small).
    pub nis_calibration_ratio: f64,
    pub mean_mahalanobis: f64,
    pub mean_search_radius_arcsec: f64,
    pub mean_n_hypotheses_after: f64,
    pub mean_effective_sample_size: f64,

    // ── Ground-truth-based metrics (NaN if no ground truth was supplied) ───
    /// Number of steps for which ground truth was available — `0` means the
    /// fields below are all `NaN` (no `--ground-truth` file, or none of this
    /// trajectory's observations matched).
    pub n_steps_with_truth: usize,
    pub rmse_pos_arcsec: f64,
    pub rmse_range_au: f64,
    pub rmse_cart_pos_au: f64,
    pub rmse_cart_vel_au_day: f64,
    pub mean_nees_sky: f64,
    pub pct_nees_sky_in_chi2_band: f64,
    pub mean_nees_cart: f64,
    pub pct_nees_cart_in_chi2_band: f64,

    // ── Fading-memory covariance-inflation diagnostics ─────────────────────
    /// Mean fading-memory inflation factor $\lambda$ over this trajectory's
    /// steps (see [`crate::kalman_traj::KFStudyResult::inflation_lambda`]).
    /// `1.0` if inflation never engaged.
    pub mean_inflation_lambda: f64,
    /// Percentage of steps where $\lambda>1$ (inflation actually active).
    pub pct_steps_inflation_active: f64,
}

/// Reduce one trajectory's [`StudyOutcome`] into a [`TrajSummary`].
///
/// Returns `None` if `outcome.results` is empty (bootstrap failed, the
/// bootstrap pair consumed the whole trajectory, or the very first step
/// already produced a degenerate result) — nothing to summarize numerically,
/// though `outcome.stop_reason` still explains why for the caller's
/// stop-reason histogram (see `RunCounters::stop_reason_counts`).
///
/// `pub`: reused as-is by `kf_calibration::objective`, which runs the same
/// materialize → Kalman filter bank → summarize pipeline as
/// [`process_one_trajectory`] but against explicit, caller-chosen
/// `EngineConfig`/`KalmanContext` variants instead of one fixed config.
pub fn summarize_trajectory(
    traj_id: TrajId,
    n_obs_total: usize,
    outcome: &StudyOutcome,
) -> Option<TrajSummary> {
    let results = &outcome.results;
    if results.is_empty() {
        return None;
    }
    let final_bank = outcome.bank.as_ref()?;

    let separation = metric_stats(results, |r| r.separation_arcsec_from_region);

    let n_steps = results.len();
    let completion_fraction = n_steps as f64 / outcome.n_processable.max(1) as f64;

    let best_final_kf = &final_bank.best().unwrap().kf;
    let final_kf_state = best_final_kf.state;
    let estimated_orbit = best_final_kf.to_orbit();

    let nis_stats = metric_stats(results, |r| r.nis);
    let nis_median = nis_stats.median;

    let n_steps_with_truth = results.iter().filter(|r| r.nees_sky.is_some()).count();

    Some(TrajSummary {
        traj_id,
        final_kf_state,
        estimated_orbit,
        n_obs_total,
        n_steps,
        completion_fraction,
        stop_reason: outcome.stop_reason,
        n_obs_before_bootstrap: outcome.bootstrap_idx.unwrap_or(0),
        n_obs_deduplicated: outcome.n_obs_deduplicated,
        n_processable: outcome.n_processable,
        pct_within_3sigma: pct_true(results, |r| r.obs_within_3sigma_region),
        pct_within_search_radius: pct_true(results, |r| r.obs_within_search_radius),
        mean_separation_arcsec: separation.mean,
        median_separation_arcsec: separation.median,
        mean_nis: nis_stats.mean,
        nis_median,
        pct_nis_in_chi2_band: pct_true(results, |r| {
            r.nis >= NIS_CHI2_2DOF_LOW && r.nis <= NIS_CHI2_2DOF_HIGH
        }),
        nis_calibration_ratio: nis_median / NIS_CHI2_2DOF_MEDIAN,
        mean_mahalanobis: metric_stats(results, |r| r.mahalanobis_distance).mean,
        mean_search_radius_arcsec: metric_stats(results, |r| r.search_region_radius_arcsec).mean,
        mean_n_hypotheses_after: metric_stats(results, |r| r.n_hypotheses_after as f64).mean,
        mean_effective_sample_size: metric_stats(results, |r| r.n_effective).mean,

        n_steps_with_truth,
        rmse_pos_arcsec: rmse_opt(results, |r| r.pos_error_arcsec),
        rmse_range_au: rmse_opt(results, |r| r.range_error_au),
        rmse_cart_pos_au: rmse_opt(results, |r| r.cart_pos_error_au),
        rmse_cart_vel_au_day: rmse_opt(results, |r| r.cart_vel_error_au_day),
        mean_nees_sky: metric_stats_opt(results, |r| r.nees_sky).mean,
        pct_nees_sky_in_chi2_band: pct_true_opt(results, |r| {
            r.nees_sky
                .map(|v| (NEES_CHI2_2DOF_LOW..=NEES_CHI2_2DOF_HIGH).contains(&v))
        }),
        mean_nees_cart: metric_stats_opt(results, |r| r.nees_cart).mean,
        pct_nees_cart_in_chi2_band: pct_true_opt(results, |r| {
            r.nees_cart
                .map(|v| (NEES_CHI2_6DOF_LOW..=NEES_CHI2_6DOF_HIGH).contains(&v))
        }),

        mean_inflation_lambda: metric_stats(results, |r| r.inflation_lambda).mean,
        pct_steps_inflation_active: pct_true(results, |r| r.inflation_lambda > 1.0),
    })
}

// ── Dataset-wide processing ───────────────────────────────────────────────

/// Bookkeeping for trajectories that didn't make it into the final summary
/// list, so the report can explain where they went.
#[derive(Debug, Default)]
pub struct RunCounters {
    pub n_total: usize,
    pub n_materialize_failed: usize,
    pub n_no_result: usize,
    pub n_not_enough_point: usize,
    /// Histogram of [`TrajStopReason`] over every trajectory that reached
    /// [`study_kalman_asteroid`] (i.e. everything except
    /// `n_materialize_failed`, which never gets one — see
    /// [`TrajOutcome::stop_reason`]). Iterate [`TrajStopReason::all`] for a
    /// stable, complete-coverage report order; a missing key means `0`.
    pub stop_reason_counts: AHashMap<TrajStopReason, usize>,
}

/// Intermediate result produced by one parallel worker for one trajectory,
/// carrying both the outcome and the per-stage timings used for throughput
/// diagnostics.
struct TrajOutcome {
    outcome: TrajOutcomeKind,
    /// `None` only for [`TrajOutcomeKind::MaterializeFailed`] — every other
    /// outcome kind reaches at least the "not enough points"/bootstrap
    /// stage and gets a real [`TrajStopReason`].
    stop_reason: Option<TrajStopReason>,
    /// This trajectory's per-step samples, in step order (`by_step[0]` is the
    /// first predict/update step after the bootstrap pair, etc.) — empty
    /// unless the loop produced at least one step. Fed into
    /// [`step_stats_by_bootstrap`]'s dataset-wide bucketing, to check
    /// whether NIS trends toward its χ²(2) expectation as more updates
    /// accumulate (see [`crate::kalman_traj::TrajStopReason`]'s module doc
    /// on the initial angular-rate covariance) or stays low throughout, and
    /// how the search-radius decomposition evolves along the arc.
    by_step: Vec<StepSample>,
    materialize_ms: f64,
    kalman_ms: f64,
    summarize_ms: f64,
    total_ms: f64,
}

enum TrajOutcomeKind {
    MaterializeFailed,
    NoResult,
    Summary(Box<TrajSummary>),
    NotEnoughPoint,
}

/// Run materialize → Kalman filter bank → summarize for a single
/// trajectory, timing each stage in milliseconds.
///
/// Always returns a [`TrajOutcome`] — failures are recorded as
/// [`TrajOutcomeKind::MaterializeFailed`] / [`TrajOutcomeKind::NoResult`]
/// rather than propagated, so a single bad trajectory never aborts the
/// whole parallel scan.
#[allow(clippy::too_many_arguments)]
fn process_one_trajectory(
    traj_id: &TrajId,
    obs_dataset: &ObsDataset,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
    grid_config: &GridConfig,
    advance_params: &NightAdvanceParams,
    geometry_cache: &ObserverGeometryCache,
    progress: &ProgressBar,
    truth_lookup: Option<&TruthLookup>,
) -> TrajOutcome {
    let traj_start = Instant::now();

    // --- materialize ---
    let t0 = Instant::now();
    let traj = match materialize_contiguous_traj(obs_dataset, traj_id) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                traj = %traj_id,
                error = %e,
                "Failed to materialize trajectory, skipping"
            );
            let materialize_ms = t0.elapsed().as_secs_f64() * 1e3;
            let total_ms = traj_start.elapsed().as_secs_f64() * 1e3;
            progress.inc(1);
            return TrajOutcome {
                outcome: TrajOutcomeKind::MaterializeFailed,
                stop_reason: None,
                by_step: Vec::new(),
                materialize_ms,
                kalman_ms: 0.0,
                summarize_ms: 0.0,
                total_ms,
            };
        }
    };
    let materialize_ms = t0.elapsed().as_secs_f64() * 1e3;

    // don't keep trajectory with less than 3 points
    let len_traj = traj.len();
    if len_traj < 3 {
        let total_ms = traj_start.elapsed().as_secs_f64() * 1e3;
        progress.inc(1);
        return TrajOutcome {
            outcome: TrajOutcomeKind::NotEnoughPoint,
            stop_reason: Some(TrajStopReason::NotEnoughPoints),
            by_step: Vec::new(),
            materialize_ms,
            kalman_ms: 0.0,
            summarize_ms: 0.0,
            total_ms,
        };
    }

    // --- kalman filter bank ---
    let t1 = Instant::now();
    let study_outcome = study_kalman_asteroid(
        &traj,
        obs_dataset,
        context,
        bank_config,
        grid_config,
        advance_params,
        geometry_cache,
        None,
        truth_lookup,
    );
    let kalman_ms = t1.elapsed().as_secs_f64() * 1e3;
    let stop_reason = Some(study_outcome.stop_reason);
    // Keep the same "finite NIS" step filter as before (so NIS-by-step is
    // unchanged); the radius decomposition rides along on the surviving steps.
    let by_step: Vec<StepSample> = study_outcome
        .results
        .iter()
        .filter(|r| r.nis.is_finite())
        .map(|r| StepSample {
            nis: r.nis,
            radius_spread_arcsec: r.radius_spread_arcsec,
            radius_component_arcsec: r.radius_component_arcsec,
            map_rho_sigma_au: r.map_rho_sigma_au,
            map_rhodot_sigma: r.map_rhodot_sigma,
        })
        .collect();

    // --- summarize ---
    let t2 = Instant::now();
    let outcome = match summarize_trajectory(traj_id.clone(), len_traj, &study_outcome) {
        Some(summary) => TrajOutcomeKind::Summary(Box::new(summary)),
        None => {
            tracing::debug!(
                traj = %traj_id,
                stop_reason = study_outcome.stop_reason.label(),
                "No usable step produced, skipping"
            );
            TrajOutcomeKind::NoResult
        }
    };
    let summarize_ms = t2.elapsed().as_secs_f64() * 1e3;
    let total_ms = traj_start.elapsed().as_secs_f64() * 1e3;

    progress.inc(1);

    TrajOutcome {
        outcome,
        stop_reason,
        by_step,
        materialize_ms,
        kalman_ms,
        summarize_ms,
        total_ms,
    }
}

/// Build the live progress bar shown while [`process_all_trajectories`]
/// scans the dataset — one line, updated in place (no repeated/scrolling
/// log output), with a running ETA computed by `indicatif` itself.
fn build_progress_bar(nb_traj: usize) -> ProgressBar {
    let progress = ProgressBar::new(nb_traj as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} trajectories ({percent}%)  elapsed {elapsed_precise}  eta {eta_precise}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    progress
}

/// Number of leading steps-since-bootstrap tracked individually by
/// [`step_stats_by_bootstrap`] before folding the rest into one "overflow"
/// bucket — deep enough to see whether NIS trends toward its χ²(2)
/// expectation over the first few updates, without letting a handful of very
/// long arcs blow up the number of (mostly near-empty) buckets.
pub const NIS_STEP_BUCKET_DEPTH: usize = 30;

/// One trajectory's per-step diagnostic sample, in step order. Carries the
/// predictive NIS plus the search-radius decomposition (between-mode spread
/// vs within-mode covariance, arcsec) so [`step_stats_by_bootstrap`] can
/// bucket all three by steps-since-bootstrap.
#[derive(Debug, Clone, Copy)]
pub struct StepSample {
    pub nis: f64,
    pub radius_spread_arcsec: f64,
    pub radius_component_arcsec: f64,
    pub map_rho_sigma_au: f64,
    pub map_rhodot_sigma: f64,
}

/// Dataset-wide distribution for one "steps since the bootstrap pair"
/// bucket — see [`step_stats_by_bootstrap`].
#[derive(Debug, Clone)]
pub struct StepBucketStats {
    /// 1-based step index, e.g. `1` = the first predict/update step right
    /// after the bootstrap pair. [`NIS_STEP_BUCKET_DEPTH`] + 1 means "this
    /// step or later" (the overflow bucket).
    pub step_index: usize,
    /// Number of (trajectory, step) samples folded into this bucket.
    pub n_samples: usize,
    pub nis: MetricStats,
    /// Between-mode spread contribution to the search radius (arcsec).
    pub radius_spread: MetricStats,
    /// Within-mode covariance contribution to the search radius (arcsec).
    pub radius_component: MetricStats,
    /// MAP hypothesis range 1-σ (AU) — range-observability signal.
    pub map_rho_sigma: MetricStats,
    /// MAP hypothesis range-rate 1-σ (AU/day).
    pub map_rhodot_sigma: MetricStats,
}

/// Reduce every trajectory's per-step NIS series into dataset-wide
/// [`StepBucketStats`], one per steps-since-bootstrap value (see
/// [`NIS_STEP_BUCKET_DEPTH`]).
///
/// This is the diagnostic for whether the filter's early-arc over-covariance
/// (see `crate::kalman_traj::TrajStopReason`'s module doc on the
/// finite-difference angular-rate variance) is transient — NIS should climb
/// toward [`crate::kalman_traj::NIS_CHI2_2DOF_MEDIAN`] within the first few
/// buckets as Kalman updates refine the initial velocity estimate — or
/// persistent, which would point at a propagation/update issue instead of
/// the bootstrap's initial covariance.
fn step_stats_by_bootstrap(by_step_all: &[Vec<StepSample>]) -> Vec<StepBucketStats> {
    let mut buckets: Vec<Vec<StepSample>> = vec![Vec::new(); NIS_STEP_BUCKET_DEPTH + 1];

    for by_step in by_step_all {
        for (i, sample) in by_step.iter().enumerate() {
            buckets[i.min(NIS_STEP_BUCKET_DEPTH)].push(*sample);
        }
    }

    // Per-metric finite filter so a NaN in one field (e.g. a degenerate
    // mixture's radius decomposition) can't poison the others' stats.
    let finite =
        |xs: &[f64]| -> Vec<f64> { xs.iter().copied().filter(|v| v.is_finite()).collect() };

    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, samples)| !samples.is_empty())
        .map(|(i, samples)| {
            let nis: Vec<f64> = samples.iter().map(|s| s.nis).collect();
            let spread = finite(
                &samples
                    .iter()
                    .map(|s| s.radius_spread_arcsec)
                    .collect::<Vec<_>>(),
            );
            let component = finite(
                &samples
                    .iter()
                    .map(|s| s.radius_component_arcsec)
                    .collect::<Vec<_>>(),
            );
            let rho_sigma = finite(
                &samples
                    .iter()
                    .map(|s| s.map_rho_sigma_au)
                    .collect::<Vec<_>>(),
            );
            let rhodot_sigma = finite(
                &samples
                    .iter()
                    .map(|s| s.map_rhodot_sigma)
                    .collect::<Vec<_>>(),
            );
            StepBucketStats {
                step_index: i + 1,
                n_samples: samples.len(),
                nis: compute_stats(&nis),
                radius_spread: compute_stats(&spread),
                radius_component: compute_stats(&component),
                map_rho_sigma: compute_stats(&rho_sigma),
                map_rhodot_sigma: compute_stats(&rhodot_sigma),
            }
        })
        .collect()
}

/// Run the Kalman-filter bank study on every trajectory in `obs_dataset` and
/// collapse each one into a [`TrajSummary`].
///
/// Only the lightweight summaries are kept in memory here — the full
/// per-step [`KFStudyResult`] series is discarded after summarizing, so this
/// scales to datasets with many thousands of trajectories. The detailed
/// per-step results for individual trajectories of interest (best/worst) are
/// recomputed on demand by [`crate::reporting::print_detailed_reports`].
///
/// Also returns dataset-wide NIS-by-step-since-bootstrap buckets — see
/// [`StepBucketStats`].
pub fn process_all_trajectories(
    obs_dataset: &ObsDataset,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
    grid_config: &GridConfig,
    advance_params: &NightAdvanceParams,
    truth_lookup: Option<&TruthLookup>,
) -> (Vec<TrajSummary>, RunCounters, Vec<StepBucketStats>) {
    let nb_traj = obs_dataset.iter_traj_id().map(|iter| iter.count()).unwrap();

    let traj_ids: Vec<TrajId> = obs_dataset
        .iter_traj_id()
        .expect("dataset must expose at least one trajectory")
        .cloned()
        .collect();

    let geometry_cache = ObserverGeometryCache::build(obs_dataset, context, &traj_ids);

    let global_start = Instant::now();
    let progress = build_progress_bar(nb_traj);

    // ── Parallel phase ────────────────────────────────────────────────
    let outcomes: Vec<TrajOutcome> = traj_ids
        .par_iter()
        .map(|traj_id| {
            process_one_trajectory(
                traj_id,
                obs_dataset,
                context,
                bank_config,
                grid_config,
                advance_params,
                &geometry_cache,
                &progress,
                truth_lookup,
            )
        })
        .collect();

    progress.finish_and_clear();

    // ── Sequential aggregation ──────────────────────────────────────────
    aggregate_outcomes(outcomes, global_start.elapsed())
}

/// Whole-run timing samples, one push per completed [`TrajOutcome`] — used
/// only for the single per-stage summary [`log_run_completion`] prints at
/// the end, not for any periodic output.
#[derive(Default)]
struct TimingStats {
    total_ms: Vec<f64>,
    materialize_ms: Vec<f64>,
    kalman_ms: Vec<f64>,
    summarize_ms: Vec<f64>,
}

impl TimingStats {
    fn with_capacity(cap: usize) -> Self {
        Self {
            total_ms: Vec::with_capacity(cap),
            materialize_ms: Vec::with_capacity(cap),
            kalman_ms: Vec::with_capacity(cap),
            summarize_ms: Vec::with_capacity(cap),
        }
    }

    fn push(&mut self, outcome: &TrajOutcome) {
        self.total_ms.push(outcome.total_ms);
        self.materialize_ms.push(outcome.materialize_ms);
        self.kalman_ms.push(outcome.kalman_ms);
        self.summarize_ms.push(outcome.summarize_ms);
    }
}

/// Fold the per-trajectory [`TrajOutcome`]s coming out of the parallel phase
/// into the final [`RunCounters`] and the list of [`TrajSummary`]s, then
/// print one final timing summary (see [`log_run_completion`]).
fn aggregate_outcomes(
    outcomes: Vec<TrajOutcome>,
    global_elapsed: Duration,
) -> (Vec<TrajSummary>, RunCounters, Vec<StepBucketStats>) {
    let mut summaries = Vec::with_capacity(outcomes.len());
    let mut counters = RunCounters::default();
    let mut timings = TimingStats::with_capacity(outcomes.len());
    let mut by_step_all: Vec<Vec<StepSample>> = Vec::with_capacity(outcomes.len());

    for outcome in outcomes {
        counters.n_total += 1;

        // Record timings first: this only borrows `outcome`, so the
        // subsequent move of `outcome.outcome` below remains legal.
        timings.push(&outcome);

        if let Some(reason) = outcome.stop_reason {
            *counters.stop_reason_counts.entry(reason).or_insert(0) += 1;
        }

        if !outcome.by_step.is_empty() {
            by_step_all.push(outcome.by_step);
        }

        match outcome.outcome {
            TrajOutcomeKind::MaterializeFailed => counters.n_materialize_failed += 1,
            TrajOutcomeKind::NoResult => counters.n_no_result += 1,
            TrajOutcomeKind::NotEnoughPoint => counters.n_not_enough_point += 1,
            TrajOutcomeKind::Summary(s) => summaries.push(*s),
        }
    }

    log_run_completion(&counters, &timings, global_elapsed);

    let step_buckets = step_stats_by_bootstrap(&by_step_all);

    (summaries, counters, step_buckets)
}

/// Print the single, once-per-run throughput summary to stderr: total
/// elapsed time and a per-stage timing breakdown (materialize / kalman /
/// summarize).
fn log_run_completion(counters: &RunCounters, timings: &TimingStats, global_elapsed: Duration) {
    let (total_mean, total_std) = mean_std(&timings.total_ms);
    let (mat_mean, mat_std) = mean_std(&timings.materialize_ms);
    let (kal_mean, kal_std) = mean_std(&timings.kalman_ms);
    let (sum_mean, sum_std) = mean_std(&timings.summarize_ms);

    eprintln!(
        "  ✓ All {} trajectories processed in {:.2} s ({:.3} ms/traj on average)\n\
         \t| per traj:    {:.2} ± {:.2} ms\n\
         \t| materialize: {:.2} ± {:.2} ms\n\
         \t| kalman:      {:.2} ± {:.2} ms\n\
         \t| summarize:   {:.2} ± {:.2} ms",
        counters.n_total,
        global_elapsed.as_secs_f64(),
        global_elapsed.as_secs_f64() / counters.n_total.max(1) as f64 * 1e3,
        total_mean,
        total_std,
        mat_mean,
        mat_std,
        kal_mean,
        kal_std,
        sum_mean,
        sum_std,
    );
}

// ── Ranking & selection ───────────────────────────────────────────────────

/// Rank trajectories by 3σ predictive-region coverage (descending: the
/// filter is "working" on a trajectory when the true observation actually
/// falls inside its predicted uncertainty ellipse most of the time), then
/// split off the `n` best and `n` worst.
///
/// `completion_fraction` is deliberately *not* part of the ranking: a
/// trajectory that collapsed after one step can look artificially perfect.
/// It is still surfaced in the printed tables so this is visible at a
/// glance, rather than silently filtered out.
pub fn select_extremes(
    summaries: &[TrajSummary],
    n: usize,
) -> (Vec<&TrajSummary>, Vec<&TrajSummary>) {
    let mut ranked: Vec<&TrajSummary> = summaries.iter().collect();
    ranked.sort_by(|a, b| {
        b.pct_within_3sigma
            .partial_cmp(&a.pct_within_3sigma)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_best = n.min(ranked.len());
    let best = ranked[..n_best].to_vec();

    // Avoid double-counting the same trajectories in both tables when the
    // dataset itself has fewer than 2*n entries.
    let n_worst = n.min(ranked.len() - n_best);
    let worst: Vec<&TrajSummary> = ranked.iter().rev().take(n_worst).cloned().collect();

    (best, worst)
}

//! Domain logic for the Kalman-filter-bank evaluation run.
//!
//! This module owns everything that touches numbers: turning a [`TrajId`]
//! into actual observations, running the topocentric Kalman filter bank on
//! it, reducing the per-step results into a one-row [`TrajSummary`], and
//! aggregating/ranking those summaries across the whole dataset.
//!
//! Printing of *results* is intentionally kept out of this module (see
//! [`crate::reporting`]). The only output produced here is lightweight
//! progress/throughput logging on stderr, which is tightly coupled to the
//! timing data computed during processing and would be awkward to extract
//! without duplicating that data.

use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use fink_fat_engine::engine_config::grid_population::GridConfig;
use fink_fat_engine::engine_config::kalman_context::KalmanContext;
use fink_fat_engine::engine_config::kf_bank_config::KFBankConfig;
use fink_fat_engine::topocentric_kf::kalman_bank::KFBank;
use nalgebra::Vector6;
use outfit::OrbitalElements;
use rayon::prelude::*;

use fink_fat_engine::error::{EngineError, FinkFatError};
use photom::{
    TrajId,
    observation_dataset::{ObsDataset, iter::MemLayoutObservations, observation::Observation},
};

use crate::kalman_traj::{KFStudyResult, study_kalman_asteroid};

/// Print a progress notice on stderr every this many trajectories scanned,
/// so a full-dataset run doesn't look stuck.
const PROGRESS_EVERY: usize = 100;

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
/// Used only for lightweight timing diagnostics in [`log_batch_progress`];
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

/// Percentage of items for which `pred` holds.
fn pct_true<T>(items: &[T], pred: impl Fn(&T) -> bool) -> f64 {
    if items.is_empty() {
        return f64::NAN;
    }
    100.0 * items.iter().filter(|x| pred(x)).count() as f64 / items.len() as f64
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
    /// `n_steps` over the number of steps expected if the bank never
    /// collapsed (1.0 means the whole arc was processed).
    pub completion_fraction: f64,
    pub pct_within_3sigma: f64,
    pub pct_within_search_radius: f64,
    pub mean_separation_arcsec: f64,
    pub median_separation_arcsec: f64,
    pub mean_nis: f64,
    pub mean_mahalanobis: f64,
    pub mean_search_radius_arcsec: f64,
    pub mean_n_hypotheses_after: f64,
    pub mean_effective_sample_size: f64,
}

/// Reduce one trajectory's per-step results into a [`TrajSummary`].
///
/// Returns `None` if `results` is empty (bootstrap failed, or the
/// trajectory had too few observations to seed the filter bank).
fn summarize_trajectory(
    traj_id: TrajId,
    n_obs_total: usize,
    results: &[KFStudyResult],
    final_bank: &KFBank,
) -> Option<TrajSummary> {
    if results.is_empty() {
        return None;
    }

    let separation = metric_stats(results, |r| r.separation_arcsec_from_region);

    // The first two observations of every trajectory are consumed by the
    // bootstrap pair, so that's the maximum number of predict/update steps
    // that could ever be produced.
    let n_steps = results.len();
    let max_possible_steps = n_obs_total.saturating_sub(2).max(1);
    let completion_fraction = n_steps as f64 / max_possible_steps as f64;

    let best_final_kf = &final_bank.best().unwrap().kf;
    let final_kf_state = best_final_kf.state;
    let estimated_orbit = best_final_kf.to_orbit();

    Some(TrajSummary {
        traj_id,
        final_kf_state,
        estimated_orbit,
        n_obs_total,
        n_steps,
        completion_fraction,
        pct_within_3sigma: pct_true(results, |r| r.obs_within_3sigma_region),
        pct_within_search_radius: pct_true(results, |r| r.obs_within_search_radius),
        mean_separation_arcsec: separation.mean,
        median_separation_arcsec: separation.median,
        mean_nis: metric_stats(results, |r| r.nis).mean,
        mean_mahalanobis: metric_stats(results, |r| r.mahalanobis_distance).mean,
        mean_search_radius_arcsec: metric_stats(results, |r| r.search_region_radius_arcsec).mean,
        mean_n_hypotheses_after: metric_stats(results, |r| r.n_hypotheses_after as f64).mean,
        mean_effective_sample_size: metric_stats(results, |r| r.n_effective).mean,
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
}

/// Intermediate result produced by one parallel worker for one trajectory,
/// carrying both the outcome and the per-stage timings used for throughput
/// diagnostics.
struct TrajOutcome {
    outcome: TrajOutcomeKind,
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
    completed: &AtomicUsize,
    nb_traj: usize,
    global_start: &Instant,
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
            record_completion(completed, nb_traj, global_start);
            return TrajOutcome {
                outcome: TrajOutcomeKind::MaterializeFailed,
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
        return TrajOutcome {
            outcome: TrajOutcomeKind::NotEnoughPoint,
            materialize_ms,
            kalman_ms: 0.0,
            summarize_ms: 0.0,
            total_ms,
        };
    }

    // --- kalman filter bank ---
    let t1 = Instant::now();
    let (bank_opt, results) =
        study_kalman_asteroid(&traj, obs_dataset, context, bank_config, grid_config);
    let kalman_ms = t1.elapsed().as_secs_f64() * 1e3;

    // --- summarize ---
    let t2 = Instant::now();
    let outcome = match bank_opt {
        None => {
            tracing::debug!(traj = %traj_id, "Bootstrap produced no bank, skipping");
            TrajOutcomeKind::NoResult
        }
        Some(bank) => match summarize_trajectory(traj_id.clone(), len_traj, &results, &bank) {
            Some(summary) => TrajOutcomeKind::Summary(Box::new(summary)),
            None => {
                tracing::debug!(traj = %traj_id, "No usable step produced, skipping");
                TrajOutcomeKind::NoResult
            }
        },
    };
    let summarize_ms = t2.elapsed().as_secs_f64() * 1e3;
    let total_ms = traj_start.elapsed().as_secs_f64() * 1e3;

    record_completion(completed, nb_traj, global_start);

    TrajOutcome {
        outcome,
        materialize_ms,
        kalman_ms,
        summarize_ms,
        total_ms,
    }
}

/// Bump the shared completion counter and emit a progress line if this
/// completion lands on a [`PROGRESS_EVERY`] boundary.
fn record_completion(completed: &AtomicUsize, nb_traj: usize, global_start: &Instant) {
    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
    maybe_print_parallel_progress(done, nb_traj, global_start);
}

/// Print a lightweight progress line every [`PROGRESS_EVERY`] completions
/// during the parallel phase.
///
/// Uses `fetch_add` with `Relaxed` ordering — the exact firing boundary
/// may slip by one under heavy contention, but this is acceptable for a
/// progress indicator.
fn maybe_print_parallel_progress(done: usize, total: usize, global_start: &Instant) {
    if !done.is_multiple_of(PROGRESS_EVERY) {
        return;
    }
    let elapsed = global_start.elapsed().as_secs_f64();
    let mean_secs = elapsed / done as f64;
    let eta = (mean_secs * total as f64 - elapsed).max(0.0);
    eprintln!(
        "  [parallel]    … {}/{} done  |  elapsed: {:.1} s  ETA: {:.1} s  ({:.1} ms/traj)",
        done,
        total,
        elapsed,
        eta,
        mean_secs * 1e3,
    );
}

/// Run the Kalman-filter bank study on every trajectory in `obs_dataset` and
/// collapse each one into a [`TrajSummary`].
///
/// Only the lightweight summaries are kept in memory here — the full
/// per-step [`KFStudyResult`] series is discarded after summarizing, so this
/// scales to datasets with many thousands of trajectories. The detailed
/// per-step results for individual trajectories of interest (best/worst) are
/// recomputed on demand by [`crate::reporting::print_detailed_reports`].
pub fn process_all_trajectories(
    obs_dataset: &ObsDataset,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
    grid_config: &GridConfig,
) -> (Vec<TrajSummary>, RunCounters) {
    let nb_traj = obs_dataset.iter_traj_id().map(|iter| iter.count()).unwrap();

    let traj_ids: Vec<_> = obs_dataset
        .iter_traj_id()
        .expect("dataset must expose at least one trajectory")
        .collect();

    let global_start = Instant::now();
    let completed = AtomicUsize::new(0);

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
                &completed,
                nb_traj,
                &global_start,
            )
        })
        .collect();

    // ── Sequential aggregation + progress reporting ────────────────────
    aggregate_outcomes(outcomes, nb_traj, &global_start)
}

/// Per-batch timing samples collected during the sequential aggregation
/// pass. Used only to print periodic throughput diagnostics, then cleared.
#[derive(Default)]
struct TimingBatch {
    total_ms: Vec<f64>,
    materialize_ms: Vec<f64>,
    kalman_ms: Vec<f64>,
    summarize_ms: Vec<f64>,
}

impl TimingBatch {
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

    fn clear(&mut self) {
        self.total_ms.clear();
        self.materialize_ms.clear();
        self.kalman_ms.clear();
        self.summarize_ms.clear();
    }
}

/// Fold the per-trajectory [`TrajOutcome`]s coming out of the parallel phase
/// into the final [`RunCounters`] and the list of [`TrajSummary`]s, while
/// periodically logging batch-level throughput to stderr.
fn aggregate_outcomes(
    outcomes: Vec<TrajOutcome>,
    nb_traj: usize,
    global_start: &Instant,
) -> (Vec<TrajSummary>, RunCounters) {
    let mut summaries = Vec::with_capacity(nb_traj);
    let mut counters = RunCounters::default();

    let mut batch = TimingBatch::with_capacity(PROGRESS_EVERY);
    let mut batch_start = Instant::now();

    for outcome in outcomes {
        counters.n_total += 1;

        // Record timings first: this only borrows `outcome`, so the
        // subsequent move of `outcome.outcome` below remains legal.
        batch.push(&outcome);

        match outcome.outcome {
            TrajOutcomeKind::MaterializeFailed => counters.n_materialize_failed += 1,
            TrajOutcomeKind::NoResult => counters.n_no_result += 1,
            TrajOutcomeKind::NotEnoughPoint => counters.n_not_enough_point += 1,
            TrajOutcomeKind::Summary(s) => summaries.push(*s),
        }

        if counters.n_total % PROGRESS_EVERY == 0 {
            log_batch_progress(
                &batch,
                &counters,
                nb_traj,
                batch_start.elapsed(),
                global_start.elapsed(),
            );
            batch.clear();
            batch_start = Instant::now();
        }
    }

    log_run_completion(&counters, global_start.elapsed());

    (summaries, counters)
}

/// Print a periodic throughput report (every [`PROGRESS_EVERY`]
/// trajectories) to stderr: batch/global elapsed time, ETA, and a per-stage
/// timing breakdown (materialize / kalman / summarize).
fn log_batch_progress(
    batch: &TimingBatch,
    counters: &RunCounters,
    nb_traj: usize,
    batch_elapsed: Duration,
    global_elapsed: Duration,
) {
    let (total_mean, total_std) = mean_std(&batch.total_ms);
    let (mat_mean, mat_std) = mean_std(&batch.materialize_ms);
    let (kal_mean, kal_std) = mean_std(&batch.kalman_ms);
    let (sum_mean, sum_std) = mean_std(&batch.summarize_ms);

    let mean_secs_per_traj = global_elapsed.as_secs_f64() / counters.n_total as f64;
    let estimated_total_secs = mean_secs_per_traj * nb_traj as f64;
    let eta_secs = (estimated_total_secs - global_elapsed.as_secs_f64()).max(0.0);

    eprintln!(
        "  [aggregation] … {}/{} trajectories\n\
         \t| batch: {:.2} s  elapsed: {:.2} s  est. total: {:.2} s  ETA: {:.2} s\n\
         \t| per traj:    {:.2} ± {:.2} ms\n\
         \t| materialize: {:.2} ± {:.2} ms\n\
         \t| kalman:      {:.2} ± {:.2} ms\n\
         \t| summarize:   {:.2} ± {:.2} ms",
        counters.n_total,
        nb_traj,
        batch_elapsed.as_secs_f64(),
        global_elapsed.as_secs_f64(),
        estimated_total_secs,
        eta_secs,
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

/// Print the final once-per-run throughput summary to stderr.
fn log_run_completion(counters: &RunCounters, global_elapsed: Duration) {
    eprintln!(
        "  ✓ All {} trajectories processed in {:.2} s ({:.3} ms/traj on average)",
        counters.n_total,
        global_elapsed.as_secs_f64(),
        global_elapsed.as_secs_f64() / counters.n_total.max(1) as f64 * 1e3,
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

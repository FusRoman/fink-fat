//! All `stdout` formatting and printing for the Kalman-filter-bank
//! evaluation run: the dataset-wide summary, the best/worst trajectory
//! tables, and the detailed per-step report for individual trajectories.
//!
//! Every function in this module is read-only with respect to
//! [`crate::trajectory_processing`]: it only consumes [`TrajSummary`] /
//! [`KFStudyResult`] data (re-running the Kalman filter on demand for
//! detailed reports), and never mutates any run state.

use fink_fat_engine::{
    engine_config::{
        grid_population::GridConfig, kalman_context::KalmanContext, kf_bank_config::KFBankConfig,
        night_advance_params::NightAdvanceParams,
    },
    topocentric_kf::kalman_bank::KFBank,
};
use photom::{TrajId, observation_dataset::ObsDataset};

use crate::{
    kalman_traj::{
        KFStudyResult, NIS_CHI2_2DOF_MEDIAN, ObserverGeometryCache, TrajStopReason,
        study_kalman_asteroid,
    },
    trajectory_processing::{
        MetricStats, NIS_STEP_BUCKET_DEPTH, RunCounters, StepBucketStats, TrajSummary, fmt_stats,
        materialize_contiguous_traj, metric_stats,
    },
};

// ── Reporting: dataset-wide ────────────────────────────────────────────────

/// Print the headline counters of a run: how many trajectories were
/// scanned, how many failed to materialize, how many produced no usable
/// Kalman result, and how many were successfully summarized.
pub fn print_run_counters(counters: &RunCounters, n_summarized: usize) {
    println!("\n=== Dataset-wide run summary ===");
    println!("  Trajectories scanned             : {}", counters.n_total);
    println!(
        "  Failed to materialize             : {}",
        counters.n_materialize_failed
    );
    println!(
        "  No usable KF result (bootstrap)   : {}",
        counters.n_no_result
    );
    println!("  Not enough points   : {}", counters.n_not_enough_point);
    println!("  Successfully summarised           : {n_summarized}");
}

/// Print a full accounting of *why* every trajectory's predict/update loop
/// stopped where it did — see [`TrajStopReason`]. Complements
/// [`print_run_counters`]: that one only distinguishes materialize/bootstrap/
/// summarize-level failures, this one breaks the KF loop's own stop points
/// down (reached the end vs. collapsed by gating vs. collapsed by a
/// propagation failure vs. a degenerate per-step result), so "trajectories
/// that stopped early" stops being one opaque number.
pub fn print_stop_reason_histogram(counters: &RunCounters) {
    let total: usize = counters.stop_reason_counts.values().sum();
    println!("\n=== Trajectory stop reasons ===");
    if total == 0 {
        println!("  (none reached the Kalman filter loop)");
        return;
    }
    for reason in TrajStopReason::all() {
        let count = counters
            .stop_reason_counts
            .get(&reason)
            .copied()
            .unwrap_or(0);
        println!(
            "  {:<42} {:>8}  ({:>5.1}%)",
            reason.label(),
            count,
            100.0 * count as f64 / total as f64
        );
    }
}

/// Print dataset-wide NIS-calibration diagnostics — see
/// [`TrajSummary::nis_calibration_ratio`]/[`TrajSummary::pct_nis_in_chi2_band`].
/// A `nis_calibration_ratio` median far from 1.0 is a calibration-direction
/// signal, not a bug: `≪ 1` means the filter's predicted covariance is
/// systematically too large relative to the actual residuals (over-covariant
/// — e.g. search regions wider than they need to be), `≫ 1` means the
/// opposite (over-confident).
pub fn print_nis_calibration_summary(summaries: &[TrajSummary]) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] NIS calibration");
    println!("{sep}");
    print_metric_row("NIS median", &metric_stats(summaries, |s| s.nis_median));
    print_metric_row(
        "NIS calibration ratio (target ≈ 1.0)",
        &metric_stats(summaries, |s| s.nis_calibration_ratio),
    );
    print_metric_row(
        "% steps in χ²(2) 95% band (target ≈ 95%)",
        &metric_stats(summaries, |s| s.pct_nis_in_chi2_band),
    );
    let ratio_median = metric_stats(summaries, |s| s.nis_calibration_ratio).median;
    println!("{sep}");
    if ratio_median.is_finite() {
        if ratio_median < 0.5 {
            println!(
                "  ⚠ Dataset-wide ratio median {ratio_median:.4} ≪ 1: the filter looks \
                 over-covariant — predicted uncertainty is systematically larger than the \
                 real residuals."
            );
        } else if ratio_median > 2.0 {
            println!(
                "  ⚠ Dataset-wide ratio median {ratio_median:.4} ≫ 1: the filter looks \
                 over-confident — predicted uncertainty is systematically smaller than the \
                 real residuals."
            );
        } else {
            println!("  Dataset-wide ratio median {ratio_median:.4} — roughly well-calibrated.");
        }
    }
}

/// Print dataset-wide NIS median/mean per steps-since-bootstrap bucket (see
/// [`StepBucketStats`]) — the diagnostic for whether over-covariance is a
/// transient bootstrap effect (NIS should climb toward
/// [`NIS_CHI2_2DOF_MEDIAN`] within the first handful of buckets as updates
/// refine the initial finite-difference angular-rate estimate) or persists
/// across the whole arc (pointing at the propagation/update mechanics
/// instead of the bootstrap's initial covariance).
pub fn print_nis_by_step_since_bootstrap(buckets: &[StepBucketStats]) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] NIS by steps-since-bootstrap (χ²(2) median = {NIS_CHI2_2DOF_MEDIAN:.4})");
    println!("{sep}");
    if buckets.is_empty() {
        println!("  (no per-step data)");
        return;
    }
    println!(
        "  {:>10}  {:>10}  {:>12}  {:>12}",
        "step", "n_samples", "NIS median", "NIS mean"
    );
    for b in buckets {
        let label = if b.step_index > NIS_STEP_BUCKET_DEPTH {
            format!("{NIS_STEP_BUCKET_DEPTH}+")
        } else {
            b.step_index.to_string()
        };
        println!(
            "  {:>10}  {:>10}  {:>12.4}  {:>12.4}",
            label, b.n_samples, b.nis.median, b.nis.mean
        );
    }
    println!("{sep}");
}

/// Print one aligned `name: stats` row, used by [`print_global_aggregate_stats`].
fn print_metric_row(name: &str, stats: &MetricStats) {
    println!("  {name:<32} {}", fmt_stats(stats));
}

/// Print mean/median/min/max for every metric tracked in [`TrajSummary`],
/// aggregated over the whole dataset.
pub fn print_global_aggregate_stats(summaries: &[TrajSummary]) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!(
        "[Global] Aggregated statistics over {} trajectories",
        summaries.len()
    );
    println!("{sep}");

    print_metric_row(
        "Completion fraction",
        &metric_stats(summaries, |s| s.completion_fraction),
    );
    print_metric_row(
        "3σ coverage (%)",
        &metric_stats(summaries, |s| s.pct_within_3sigma),
    );
    print_metric_row(
        "Search-radius coverage (%)",
        &metric_stats(summaries, |s| s.pct_within_search_radius),
    );
    print_metric_row(
        "Mean separation (\")",
        &metric_stats(summaries, |s| s.mean_separation_arcsec),
    );
    print_metric_row(
        "Median separation (\")",
        &metric_stats(summaries, |s| s.median_separation_arcsec),
    );
    print_metric_row("Mean NIS", &metric_stats(summaries, |s| s.mean_nis));
    print_metric_row(
        "Mean Mahalanobis distance",
        &metric_stats(summaries, |s| s.mean_mahalanobis),
    );
    print_metric_row(
        "Mean search radius (\")",
        &metric_stats(summaries, |s| s.mean_search_radius_arcsec),
    );
    print_metric_row(
        "Mean #hypotheses (after step)",
        &metric_stats(summaries, |s| s.mean_n_hypotheses_after),
    );
    print_metric_row(
        "Mean effective sample size",
        &metric_stats(summaries, |s| s.mean_effective_sample_size),
    );

    let n_reached_end = summaries
        .iter()
        .filter(|s| s.stop_reason == TrajStopReason::ReachedEnd)
        .count();
    println!("{sep}");
    println!(
        "  Trajectories that reached the end of their arc: {n_reached_end} / {}\n\
         \t(see the \"Trajectory stop reasons\" section below for why the rest stopped early)",
        summaries.len()
    );
}

/// Print a compact one-row-per-trajectory table. Used for both the
/// best-of and worst-of selections (see
/// [`crate::trajectory_processing::select_extremes`]).
pub fn print_extremes_table(title: &str, items: &[&TrajSummary]) {
    println!("\n--- {title} ---");
    if items.is_empty() {
        println!("  (none)");
        return;
    }
    println!(
        "{:>12}  {:>6}  {:>6}  {:>7}  {:>6}  {:>9}  {:>9}  {:>9}  {:>8}  {:<28}",
        "traj_id",
        "n_obs",
        "n_proc",
        "n_step",
        "cmpl%",
        "3σ_cov%",
        "rad_cov%",
        "sep(\")",
        "NIS",
        "stop_reason"
    );
    for s in items {
        println!(
            "{:>12}  {:>6}  {:>6}  {:>7}  {:>6.1}  {:>9.1}  {:>9.1}  {:>9.3}  {:>8.3}  {:<28}",
            s.traj_id,
            s.n_obs_total,
            s.n_processable,
            s.n_steps,
            s.completion_fraction * 100.0,
            s.pct_within_3sigma,
            s.pct_within_search_radius,
            s.mean_separation_arcsec,
            s.mean_nis,
            s.stop_reason.label(),
        );
    }
}

// ── Reporting: single trajectory deep-dive ─────────────────────────────────
//
// These are reused for both the best and the worst trajectories: results
// are recomputed on demand from their `TrajId` rather than kept around for
// every trajectory in the dataset (see
// [`crate::trajectory_processing::process_all_trajectories`]).

/// Re-materialize and re-run the Kalman filter bank for each trajectory in
/// `traj_ids`, printing a full per-step report for each.
#[allow(clippy::too_many_arguments)]
pub fn print_detailed_reports(
    label: &str,
    traj_ids: &[TrajId],
    obs_dataset: &ObsDataset,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
    grid_config: &GridConfig,
    advance_params: &NightAdvanceParams,
) {
    let geometry_cache = ObserverGeometryCache::build(obs_dataset, context, traj_ids);

    for traj_id in traj_ids {
        match materialize_contiguous_traj(obs_dataset, traj_id) {
            Ok(traj) => {
                let len_traj = traj.len();
                println!("\n############################################################");
                println!("# [{label}] trajectory {traj_id} (size: {len_traj} obs");
                println!("############################################################");

                println!("\n___ Print the whole observation for this trajectory ___\n");
                for obs in traj.iter() {
                    println!("{obs}");
                }
                println!("\n\n =============");

                let study_outcome = study_kalman_asteroid(
                    &traj,
                    obs_dataset,
                    context,
                    bank_config,
                    grid_config,
                    advance_params,
                    &geometry_cache,
                    None,
                );
                println!(
                    "\nStop reason: {} (n_processable={}, n_obs_deduplicated={})",
                    study_outcome.stop_reason.label(),
                    study_outcome.n_processable,
                    study_outcome.n_obs_deduplicated,
                );
                print_single_trajectory_report(
                    study_outcome.bank.as_ref(),
                    &study_outcome.results,
                    len_traj,
                );
            }
            Err(e) => println!("  (failed to re-materialize trajectory: {e})"),
        }
    }
}

/// Print every section of a single trajectory's deep-dive report: the
/// per-step table, then three summary blocks (coverage, residuals/
/// consistency, bank health).
fn print_single_trajectory_report(
    bank: Option<&KFBank>,
    results: &[KFStudyResult],
    len_traj: usize,
) {
    if results.is_empty() || bank.is_none() {
        println!("  (no results to report)");
        return;
    }

    let best_final_kf = &bank.unwrap().best().unwrap().kf;
    let estimated_orbit = best_final_kf.to_orbit();

    println!("== Estimated final best kalman ==");
    println!("{best_final_kf}");

    println!("{estimated_orbit}");

    print_per_step_table(results, len_traj);
    print_search_region_coverage_summary(results);
    print_residuals_and_consistency_summary(results);
    print_bank_health_summary(results);
}

/// Print the raw per-step table: one row per predict/update step, with
/// epoch, residual, predicted-region size, and bank-health columns.
pub fn print_per_step_table(results: &[KFStudyResult], len_traj: usize) {
    // Column widths (data drives the width, header is padded/truncated to match).
    // step | epoch | dt | sep_kf | sep_reg | mahal | 3σ-a | 3σ-b | r_srch | NIS | in_3σ | in_r | n_hyp
    let w = (5, 12, 10, 10, 10, 10, 10, 10, 10, 8, 6, 6, 5);

    let total_width =
        w.0 + w.1 + w.2 + w.3 + w.4 + w.5 + w.6 + w.7 + w.8 + w.9 + w.10 + w.11 + w.12 + 2 * 12; // 2-space separators between 13 columns

    let sep = "=".repeat(total_width);
    let dash = "-".repeat(total_width);

    println!("\n{sep}");
    println!(
        "[Summary] Per-step results ({}/{} observations processed)",
        results.len(),
        len_traj
    );
    println!("{sep}");
    println!(
        "{:>w0$}  {:>w1$}  {:>w2$}  {:>w3$}  {:>w4$}  {:>w5$}  {:>w6$}  {:>w7$}  {:>w8$}  {:>w9$}  {:>w10$}  {:>w11$}  {:>w12$}",
        "step",
        "epoch(MJD)",
        "dt(days)",
        "sep_kf(\")",
        "sep_reg(\")",
        "mahal",
        "3σ-a(\")",
        "3σ-b(\")",
        "r_srch(\")",
        "NIS",
        "in_3σ?",
        "in_r?",
        "n_hyp",
        w0 = w.0,
        w1 = w.1,
        w2 = w.2,
        w3 = w.3,
        w4 = w.4,
        w5 = w.5,
        w6 = w.6,
        w7 = w.7,
        w8 = w.8,
        w9 = w.9,
        w10 = w.10,
        w11 = w.11,
        w12 = w.12,
    );
    println!("{dash}");

    for (i, r) in results.iter().enumerate() {
        println!(
            "{:>w0$}  {:>w1$.4}  {:>w2$.4}  {:>w3$.4}  {:>w4$.4}  {:>w5$.4}  {:>w6$.4}  {:>w7$.4}  {:>w8$.4}  {:>w9$.4}  {:>w10$}  {:>w11$}  {:>w12$}",
            i + 1,
            r.epoch,
            r.dt,
            r.separation_arcsec_from_best_kf,
            r.separation_arcsec_from_region,
            r.mahalanobis_distance,
            r.region_semi_major_3sigma_arcsec,
            r.region_semi_minor_3sigma_arcsec,
            r.search_region_radius_arcsec,
            r.nis,
            if r.obs_within_3sigma_region {
                "YES"
            } else {
                "NO"
            },
            if r.obs_within_search_radius {
                "YES"
            } else {
                "NO"
            },
            r.n_hypotheses_after,
            w0 = w.0,
            w1 = w.1,
            w2 = w.2,
            w3 = w.3,
            w4 = w.4,
            w5 = w.5,
            w6 = w.6,
            w7 = w.7,
            w8 = w.8,
            w9 = w.9,
            w10 = w.10,
            w11 = w.11,
            w12 = w.12,
        );
    }
    println!("{sep}");
}

/// Print how often the true observation fell inside the predicted 3σ
/// ellipse / conservative search radius, plus the size of those regions.
fn print_search_region_coverage_summary(results: &[KFStudyResult]) {
    let n = results.len() as f64;
    let n_within_3sigma = results
        .iter()
        .filter(|r| r.obs_within_3sigma_region)
        .count();
    let n_within_radius = results
        .iter()
        .filter(|r| r.obs_within_search_radius)
        .count();

    println!("\n[Summary] ── Predicted search region coverage ──────────────────────────────────");
    println!(
        "  Obs within best-hyp 3σ ellipse (NIS ≤ 9) : {:>4} / {} ({:.1}%)",
        n_within_3sigma,
        results.len(),
        100.0 * n_within_3sigma as f64 / n
    );
    println!(
        "  Obs within conservative search radius      : {:>4} / {} ({:.1}%)",
        n_within_radius,
        results.len(),
        100.0 * n_within_radius as f64 / n
    );
    println!(
        "  Search radius (\")                          : {}",
        fmt_stats(&metric_stats(results, |r| r.search_region_radius_arcsec))
    );
    println!(
        "  3σ semi-major, best hyp (\")                : {}",
        fmt_stats(&metric_stats(results, |r| r.region_semi_major_3sigma_arcsec))
    );
    println!(
        "  3σ semi-minor, best hyp (\")                : {}",
        fmt_stats(&metric_stats(results, |r| r.region_semi_minor_3sigma_arcsec))
    );
}

/// Print residual/consistency metrics (separation, Mahalanobis distance,
/// NIS, RA/Dec sigmas) for the trajectory.
fn print_residuals_and_consistency_summary(results: &[KFStudyResult]) {
    println!("\n[Summary] ── Residuals & filter consistency ────────────────────────────────────");
    println!(
        "  Separation best kf (\")        : {}",
        fmt_stats(&metric_stats(results, |r| r.separation_arcsec_from_best_kf))
    );
    println!(
        "  Separation region (\")        : {}",
        fmt_stats(&metric_stats(results, |r| r.separation_arcsec_from_region))
    );
    println!(
        "  Mahalanobis distance  : {}",
        fmt_stats(&metric_stats(results, |r| r.mahalanobis_distance))
    );
    println!(
        "  NIS (χ²(2), exp. 2.0) : {}",
        fmt_stats(&metric_stats(results, |r| r.nis))
    );
    println!(
        "  σ_RA (\")              : {}",
        fmt_stats(&metric_stats(results, |r| r.sigma_ra_arcsec))
    );
    println!(
        "  σ_Dec (\")             : {}",
        fmt_stats(&metric_stats(results, |r| r.sigma_dec_arcsec))
    );
}

/// Print hypothesis-bank health metrics (live hypothesis count, effective
/// sample size) for the trajectory.
fn print_bank_health_summary(results: &[KFStudyResult]) {
    println!("\n[Summary] ── Bank health ────────────────────────────────────────────────────────");
    println!(
        "  Live hypotheses (after step) : {}",
        fmt_stats(&metric_stats(results, |r| r.n_hypotheses_after as f64))
    );
    println!(
        "  Effective sample size        : {}",
        fmt_stats(&metric_stats(results, |r| r.n_effective))
    );
}

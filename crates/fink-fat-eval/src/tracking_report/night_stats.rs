//! Per-night multi-hypothesis-tracking statistics: timing, branching
//! volume, gold-trajectory recall/purity/completeness, LLR confidence, and
//! Kalman error-box sizing — everything [`super::report::TrackingReport`]
//! collects one night at a time.

use ahash::AHashSet;
use photom::{
    NightId, TrajId,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::branching::BranchCollection,
};

use crate::{
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
    tracking_report::{
        bank_population::{BankPopulationStats, BankSample},
        contamination_origin::ContaminationOriginStats,
        error_box::{
            BankErrorBox, bank_predictive_error_box, build_next_night_context,
            hypothesis_error_box_radii_arcsec,
        },
        gold_trajectory::GoldTrajectoryTracker,
        lineage_lifecycle::LineageTracker,
    },
    trajectory_processing::{MetricStats, metric_stats},
};

/// Full tracking-quality digest for one night.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NightTrackingStats {
    pub step: usize,
    pub night_id: NightId,
    pub elapsed_ms: f64,

    pub n_observations: usize,
    pub n_observations_consumed: usize,
    pub n_observations_leftover: usize,

    pub n_active_lineages: usize,
    pub n_branches: usize,
    pub branches_per_lineage: MetricStats,
    pub n_new_lineages_seeded: usize,
    pub n_lineages_died: usize,
    pub lineage_survival_nights_of_died: Vec<f64>,

    /// This night's multi-detection recall/purity (narrow, single-night
    /// notion — see [`crate::seed_bank_report::night_stats`]).
    pub n_multi_detection_objects_tonight: usize,
    pub n_objects_recalled_tonight: usize,
    pub n_objects_touched_tonight: usize,
    pub n_branches_pure: usize,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub recall_pct_tonight: f64,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub loose_recall_pct_tonight: f64,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub purity_pct: f64,

    /// Cumulative completeness: of every gold trajectory that is
    /// [`GoldTrajectoryTracker::is_trackable`] (>= 2 observations within a
    /// single night at some point so far — the seeding requirement; see
    /// that method's doc), how many have a pure branch whose association
    /// history exactly matches every observation seen so far (none missing,
    /// none extra).
    pub n_trackable_objects_so_far: usize,
    pub n_objects_complete_so_far: usize,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub completeness_pct_so_far: f64,

    /// Relaxed cumulative completeness: same purity requirement as
    /// [`Self::completeness_pct_so_far`], but a currently-live pure branch
    /// only needs to cover a `>= completeness_coverage_threshold` fraction
    /// of the observations seen so far for its trajectory, rather than an
    /// exact match. Tolerates a live branch that has dropped a few points
    /// along the way as long as it hasn't been contaminated by another
    /// object.
    pub n_objects_complete_relaxed_so_far: usize,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub completeness_relaxed_pct_so_far: f64,

    /// Per-category breakdown of trackable objects'
    /// [`ObjectOutcome`](crate::tracking_report::object_outcome::ObjectOutcome),
    /// as of this night (i.e. `ObjectOutcomeTracker::classify_all` called
    /// with `last_step = step`) — same fixed order as
    /// `ObjectOutcome::all()`. Denominator for percentages is
    /// `n_trackable_objects_so_far` above.
    pub object_outcome_counts: [usize; 7],

    pub cumulative_llr: MetricStats,
    pub effective_sample_size: MetricStats,
    pub hypotheses_per_branch: MetricStats,

    pub hypothesis_error_box_radius_arcsec: MetricStats,
    /// Number of live hypotheses this night whose per-hypothesis error-box
    /// radius exceeded the sanity ceiling and were excluded from
    /// [`Self::hypothesis_error_box_radius_arcsec`] — see
    /// `error_box::hypothesis_error_box_radii_arcsec`'s doc. Should be 0;
    /// nonzero indicates a numerically ill-conditioned hypothesis (e.g. a
    /// long-surviving lineage whose range estimate drifted) still occurred
    /// in the engine this night.
    pub n_hypotheses_excessive_radius: usize,
    pub bank_error_box_radius_arcsec: Option<MetricStats>,
    pub n_observations_in_box_next_night: Option<MetricStats>,
}

/// `100 * numerator / denominator`, or `NaN` when there is nothing to
/// measure — matches [`crate::seed_bank_report::night_stats::percentage`]'s
/// convention (kept as a private copy here to avoid making that function
/// part of `seed_bank_report`'s public surface just for this reuse).
fn percentage(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        100.0 * numerator as f64 / denominator as f64
    }
}

/// Compute the full tracking digest for one night.
///
/// # Arguments
/// * `step`, `night_id` – This night's index/id.
/// * `night_obs` – This night's observations.
/// * `next_night_obs` – The following night's observations, if any (`None`
///   on the last night processed) — used for the predictive error-box
///   metrics.
/// * `prev_collection`, `collection` – The branch collection immediately
///   before and after this night's `advance_one_night` call.
/// * `ground_truth`, `gold_tracker` – Ground-truth lookups;
///   `gold_tracker.observe_night` must already have been called for
///   `night_obs` before this function runs.
/// * `lineage_tracker` – Persists lineage birth/death bookkeeping across
///   the whole run.
#[allow(clippy::too_many_arguments)]
pub fn compute_night_tracking_stats(
    step: usize,
    night_id: NightId,
    night_obs: &[&Observation],
    next_night_obs: Option<&[&Observation]>,
    prev_collection: &BranchCollection,
    collection: &BranchCollection,
    ground_truth: &ObsTrajLookup,
    gold_tracker: &GoldTrajectoryTracker,
    lineage_tracker: &mut LineageTracker,
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
    engine_config: &EngineConfig,
    spatial_binner: &HealpixBinner,
    elapsed_ms: f64,
    completeness_coverage_threshold: f64,
    bank_population: &mut BankPopulationStats,
    contamination_origin: &mut ContaminationOriginStats,
) -> NightTrackingStats {
    // Attribution needs the lineage identities from *before* the advance, so it
    // runs here rather than from the surviving collection alone.
    contamination_origin.observe_night(
        prev_collection,
        collection,
        &collection.last_night_gate_records,
        ground_truth,
    );

    let branches = &collection.branches;

    // ── Branching / lineage churn ───────────────────────────────────────
    let prev_lineage_ids: AHashSet<u64> = prev_collection
        .branches
        .iter()
        .map(|b| b.lineage_id)
        .collect();
    let current_lineage_ids: AHashSet<u64> = branches.iter().map(|b| b.lineage_id).collect();
    let diff = lineage_tracker.advance(step, &prev_lineage_ids, &current_lineage_ids);

    let branches_per_lineage_counts: Vec<f64> = {
        let mut counts: ahash::AHashMap<u64, usize> = ahash::AHashMap::default();
        for b in branches {
            *counts.entry(b.lineage_id).or_insert(0) += 1;
        }
        counts.values().map(|&c| c as f64).collect()
    };

    // ── Observations claimed vs. leftover ────────────────────────────────
    let all_track_ids: AHashSet<ObsId> = branches
        .iter()
        .flat_map(|b| b.track_ids().iter().copied())
        .collect();
    let n_observations_consumed = night_obs
        .iter()
        .filter(|o| all_track_ids.contains(o.id()))
        .count();

    // ── This-night recall/purity (reuse the seeding-report primitive) ───
    let tonight = crate::seed_bank_report::night_stats::compute_night_stats(
        night_id,
        night_obs,
        branches,
        ground_truth,
    );

    // ── Cumulative gold-trajectory completeness ─────────────────────────
    //
    // Counts live branches **and** archived arcs: a lineage retired by the
    // staleness purge keeps its reconstruction (see `ArchivedTrajectory`),
    // and scoring only the live set reported every one of them as though the
    // object had never been reconstructed — which is exactly the accounting
    // error this metric was displaying before.
    //
    // Also keyed by trajectory rather than counted per branch, so an object
    // covered by several reconstructions contributes once (its best
    // coverage) instead of inflating the numerator — previously this could
    // push the reported percentage above 100 %, and folding archives in
    // would have made that worse.
    let mut best_coverage: ahash::AHashMap<TrajId, usize> = ahash::AHashMap::default();
    for track_ids in branches.iter().map(|branch| branch.track_ids()).chain(
        collection
            .archived
            .iter()
            .map(|archived| archived.track_ids.as_slice()),
    ) {
        if let SeedPurity::Pure(traj_id) = ground_truth.classify(track_ids) {
            let best = best_coverage.entry(traj_id).or_insert(0);
            *best = (*best).max(track_ids.len());
        }
    }

    let mut n_objects_complete_so_far = 0;
    let mut n_objects_complete_relaxed_so_far = 0;
    for (traj_id, &n_covered) in &best_coverage {
        let Some(n_seen) = gold_tracker.n_obs_so_far(traj_id) else {
            continue;
        };
        if n_seen < 2 {
            continue;
        }
        if n_covered == n_seen {
            n_objects_complete_so_far += 1;
        }
        if n_covered as f64 / n_seen as f64 >= completeness_coverage_threshold {
            n_objects_complete_relaxed_so_far += 1;
        }
    }
    // Denominator: only objects that ever had >= 2 observations *within a
    // single night* — seeding is strictly intra-night (see
    // `GoldTrajectoryTracker::is_trackable`), so an object whose detections
    // never co-occur on the same night can never be picked up regardless of
    // tracker quality, and including it here would understate completeness
    // for a reason the tracker cannot fix.
    let n_trackable_objects_so_far = gold_tracker.n_trackable_so_far();

    // ── Confidence / mixture health ──────────────────────────────────────
    let cumulative_llr = metric_stats(branches, |b| b.cumulative_llr);
    let effective_sample_size = metric_stats(branches, |b| b.bank.effective_sample_size());
    let hypotheses_per_branch = metric_stats(branches, |b| b.bank.len() as f64);

    // ── Kalman error boxes ────────────────────────────────────────────────
    // Per-bank error-box sizing is independent per branch and dominated by the
    // same per-hypothesis geometry work `advance_one_night` parallelizes —
    // compute it across branches with rayon, merge the (cheap) results serially.
    let per_branch_radii: Vec<(Vec<f64>, usize)> = branches
        .par_iter()
        .map(|b| hypothesis_error_box_radii_arcsec(&b.bank))
        .collect();
    let mut hypothesis_radii: Vec<f64> = Vec::new();
    let mut n_hypotheses_excessive_radius = 0;
    for (radii, n_excessive) in per_branch_radii {
        hypothesis_radii.extend(radii);
        n_hypotheses_excessive_radius += n_excessive;
    }
    let hypothesis_error_box_radius_arcsec = metric_stats(&hypothesis_radii, |&r| r);

    let (bank_error_box_radius_arcsec, n_observations_in_box_next_night) = match next_night_obs {
        Some(next_obs) => {
            match build_next_night_context(obs_dataset, kalman_context, next_obs, spatial_binner) {
                Some(next_night) => {
                    // The dominant cost of this whole function: each bank is
                    // propagated to the next night (Kepler solves) to size its
                    // predictive box — the same embarrassingly-parallel per-bank
                    // work as `advance_one_night`, so run it with rayon too.
                    //
                    // The bank-population sample rides along here rather than in
                    // its own pass: it needs exactly this box radius, and the
                    // propagation that produces it is the expensive part.
                    let boxes: Vec<(Option<BankErrorBox>, BankSample)> = branches
                        .par_iter()
                        .map(|b| {
                            let error_box = bank_predictive_error_box(
                                &b.bank,
                                &next_night,
                                spatial_binner,
                                engine_config,
                            );
                            let sample = BankSample {
                                n_steps: b.bank.n_steps(),
                                n_hypotheses: b.bank.len(),
                                applied_cap: b.bank.last_applied_cap(),
                                effective_sample_size: b.bank.effective_sample_size(),
                                radius_arcsec: error_box.as_ref().map(|bx| bx.radius_arcsec),
                            };
                            (error_box, sample)
                        })
                        .collect();

                    let (boxes, samples): (Vec<_>, Vec<_>) = boxes.into_iter().unzip();
                    bank_population.observe_night(samples);
                    let boxes: Vec<BankErrorBox> = boxes.into_iter().flatten().collect();

                    let radii = metric_stats(&boxes, |bx| bx.radius_arcsec);
                    let in_box = metric_stats(&boxes, |bx| bx.n_observations_in_box as f64);
                    (Some(radii), Some(in_box))
                }
                None => (None, None),
            }
        }
        None => (None, None),
    };

    NightTrackingStats {
        step,
        night_id,
        elapsed_ms,
        n_observations: night_obs.len(),
        n_observations_consumed,
        n_observations_leftover: night_obs.len() - n_observations_consumed,
        n_active_lineages: current_lineage_ids.len(),
        n_branches: branches.len(),
        branches_per_lineage: metric_stats(&branches_per_lineage_counts, |&c| c),
        n_new_lineages_seeded: diff.n_new,
        n_lineages_died: diff.n_died(),
        lineage_survival_nights_of_died: diff.died_survival_nights,
        n_multi_detection_objects_tonight: tonight.n_multi_detection_objects,
        n_objects_recalled_tonight: tonight.n_objects_recalled,
        n_objects_touched_tonight: tonight.n_objects_touched,
        n_branches_pure: tonight.n_pure_branches,
        recall_pct_tonight: tonight.recall_pct(),
        loose_recall_pct_tonight: tonight.loose_recall_pct(),
        purity_pct: tonight.purity_pct(),
        n_trackable_objects_so_far,
        n_objects_complete_so_far,
        completeness_pct_so_far: percentage(n_objects_complete_so_far, n_trackable_objects_so_far),
        n_objects_complete_relaxed_so_far,
        completeness_relaxed_pct_so_far: percentage(
            n_objects_complete_relaxed_so_far,
            n_trackable_objects_so_far,
        ),
        // Filled in by the caller (`tracking_analysis.rs`), which owns the
        // `ObjectOutcomeTracker` and computes this from
        // `ObjectOutcomeTracker::classify_all` at `last_step = step` — kept
        // out of this function so its signature doesn't grow further.
        object_outcome_counts: [0; 7],
        cumulative_llr,
        effective_sample_size,
        hypotheses_per_branch,
        hypothesis_error_box_radius_arcsec,
        n_hypotheses_excessive_radius,
        bank_error_box_radius_arcsec,
        n_observations_in_box_next_night,
    }
}

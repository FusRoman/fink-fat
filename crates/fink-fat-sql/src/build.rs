//! Pure construction of every row this crate loads, from a
//! [`BranchCollection`] (trajectory tables, see [`TrajectoryRows`]), its
//! archived trajectories ([`build_archived_rows`]) and the observation index
//! of its branches ([`ObsIndex`]).
//!
//! Nothing here touches Postgres or the filesystem: nothing in this file
//! does I/O. That is what makes it the part of the SQL conversion pipeline
//! worth unit-testing directly (see the `tests` module below), rather than
//! only indirectly through a real database.

use std::collections::HashMap;

use fink_fat_engine::topocentric_kf::branching::{ArchivedTrajectory, BranchCollection};
use nalgebra::Vector3;
use rayon::prelude::*;

use crate::family::{DynamicalFamily, classify_from_attributable_state};

use crate::rows::{
    ArchivedRow, BranchBestStateRow, BranchObservationRow, BranchRow, HypothesisRow, KfBankRow,
    KfStateFields, KfStateRow,
};

/// Per-observation lookup keyed by `ObservationRow::id`
/// (== `BranchObservationRow::obs_id`), mapping to `(night_id, mjd_tt)`.
/// Built once per load — by [`crate::write_bulk_tables`] from the observation
/// rows read off disk, by the nightly sync from `observations` itself (see
/// [`crate::night_sync::fetch_obs_index`]) — and shared read-only by every
/// branch's [`compute_obs_stats`]
/// call — computing each branch's arc/night-count/inter-night-gap stats once
/// here in Rust, rather than re-deriving them from Postgres on every
/// homepage sort.
pub type ObsIndex = HashMap<i64, (i64, f64)>;

/// Per-branch observation statistics stored on the `branches` row, computed
/// once at conversion time so neither the explorer's homepage nor its bulk
/// fit has to aggregate `branch_observations ⋈ observations` again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObsStats {
    /// `max(mjd_tt) - min(mjd_tt)` across every matched observation, `0.0`
    /// if none matched.
    pub arc_length_days: f64,
    /// Number of distinct `night_id`s among matched observations.
    pub n_nights: i64,
    /// Median gap (in days) between consecutive nights, each night
    /// represented by its earliest observation's `mjd_tt`; `None` when the
    /// branch spans at most one night (there is then no inter-night gap).
    pub median_inter_night_dt_days: Option<f64>,
    /// Number of matched observations, a repeated id counted each time — the
    /// same value as `COUNT(*)` over `branch_observations ⋈ observations`
    /// for this branch, which the orbit-fit eligibility rule thresholds.
    pub n_observations: i64,
    /// Number of nights with at least two matched observations: the
    /// geometric bar the explorer's `WellSampledDiscovery` quality tier adds
    /// on top of `Discovery`.
    pub well_sampled_nights: i64,
}

/// Observation statistics for one branch, see [`ObsStats`].
///
/// # Arguments
///
/// * `track_ids` — the branch's observation ids (`u64`, as stored on the
///   branch's bank), looked up in `obs_index`. An id absent from `obs_index`
///   is silently skipped rather than treated as an error: it just means that
///   particular observation wasn't present in the source parquet this run
///   read, which does not invalidate the rest of the branch's stats.
/// * `obs_index` — see [`ObsIndex`].
///
/// # Returns
///
/// The branch's [`ObsStats`]; all zero (and no median gap) when no id
/// matched.
pub fn compute_obs_stats(track_ids: &[u64], obs_index: &ObsIndex) -> ObsStats {
    // night_id -> (earliest mjd_tt, number of matched observations).
    let mut nights: HashMap<i64, (f64, i64)> = HashMap::new();
    let mut min_mjd = f64::INFINITY;
    let mut max_mjd = f64::NEG_INFINITY;
    let mut n_observations = 0;

    for &obs_id in track_ids {
        let Some(&(night_id, mjd_tt)) = obs_index.get(&(obs_id as i64)) else {
            continue;
        };
        n_observations += 1;
        min_mjd = min_mjd.min(mjd_tt);
        max_mjd = max_mjd.max(mjd_tt);
        nights
            .entry(night_id)
            .and_modify(|(start, count)| {
                *start = start.min(mjd_tt);
                *count += 1;
            })
            .or_insert((mjd_tt, 1));
    }

    if nights.is_empty() {
        return ObsStats {
            arc_length_days: 0.0,
            n_nights: 0,
            median_inter_night_dt_days: None,
            n_observations: 0,
            well_sampled_nights: 0,
        };
    }

    let well_sampled_nights = nights.values().filter(|(_, count)| *count >= 2).count() as i64;
    let n_nights = nights.len() as i64;

    let mut night_times: Vec<f64> = nights.into_values().map(|(start, _)| start).collect();
    night_times.sort_by(|a, b| a.total_cmp(b));

    let mut gaps: Vec<f64> = night_times.windows(2).map(|w| w[1] - w[0]).collect();
    let median_inter_night_dt_days = if gaps.is_empty() {
        None
    } else {
        gaps.sort_by(|a, b| a.total_cmp(b));
        let mid = gaps.len() / 2;
        Some(if gaps.len().is_multiple_of(2) {
            (gaps[mid - 1] + gaps[mid]) / 2.0
        } else {
            gaps[mid]
        })
    };

    ObsStats {
        arc_length_days: max_mjd - min_mjd,
        n_nights,
        median_inter_night_dt_days,
        n_observations,
        well_sampled_nights,
    }
}

/// Index of a bank's best hypothesis: the one with the largest log-weight.
///
/// Matches the `ORDER BY log_weight DESC LIMIT 1` the explorer used to run
/// per branch, NaN included: [`f64::total_cmp`] ranks a (positive) NaN above
/// every number, as Postgres does.
///
/// # Arguments
///
/// * `log_weights` — the bank's hypothesis log-weights, in bank order.
///
/// # Returns
///
/// The index of the largest log-weight, `None` for an empty bank.
pub fn best_hypothesis_index(log_weights: impl IntoIterator<Item = f64>) -> Option<usize> {
    log_weights
        .into_iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}

/// Every row of the 6 trajectory tables for one [`BranchCollection`]: what
/// `convert` loads into `public` and the nightly sync into the staging
/// schema (see [`crate::load_trajectory_tables`]).
///
/// The vectors are aligned by construction (see [`Self::build`]):
/// `hypotheses[i]` and `kf_state[i]` describe the same hypothesis, and each
/// [`BranchBestStateRow::kf_state_index`] points into `kf_state`.
#[derive(Default)]
pub struct TrajectoryRows {
    /// One row per branch.
    pub branches: Vec<BranchRow>,
    /// One row per branch.
    pub kf_bank: Vec<KfBankRow>,
    /// One row per (branch, track position).
    pub branch_observations: Vec<BranchObservationRow>,
    /// One row per hypothesis of every branch's bank.
    pub hypotheses: Vec<HypothesisRow>,
    /// One row per hypothesis, aligned with `hypotheses`.
    pub kf_state: Vec<KfStateRow>,
    /// One row per branch with at least one hypothesis.
    pub best_state: Vec<BranchBestStateRow>,
}

impl TrajectoryRows {
    /// Builds every trajectory-table row by walking `branch_collection` once.
    ///
    /// Walks `&BranchCollection` directly rather than reading back the
    /// Polars `DataFrame`s of `parquet::to_dataframes`: extracting typed
    /// values out of `ChunkedArray`/`ListChunked` just to re-serialize them
    /// for `COPY` would be a pointless extra materialization pass.
    ///
    /// # Arguments
    ///
    /// * `branch_collection` — the collection to convert (only `branches` is
    ///   read).
    /// * `obs_index` — see [`ObsIndex`]; used to compute each branch's
    ///   [`ObsStats`] via [`compute_obs_stats`].
    ///
    /// # Returns
    ///
    /// The rows, in `branch_collection`'s own branch order. Hypothesis ids
    /// are allocated sequentially across every branch, from 0, so
    /// `hypotheses`/`kf_state` are aligned 1:1; `best_state` holds one row
    /// per branch with at least one hypothesis, pointing at its best
    /// hypothesis' `kf_state` row (see [`best_hypothesis_index`]).
    pub fn build(branch_collection: &BranchCollection, obs_index: &ObsIndex) -> Self {
        let n_branches = branch_collection.branches.len();
        let n_track_id_rows: usize = branch_collection
            .branches
            .iter()
            .map(|b| b.bank.track_ids().len())
            .sum();
        let n_hypotheses: usize = branch_collection
            .branches
            .iter()
            .map(|b| b.bank.hypotheses().len())
            .sum();

        let mut branch_rows = Vec::with_capacity(n_branches);
        let mut kf_bank_rows = Vec::with_capacity(n_branches);
        let mut branch_observation_rows = Vec::with_capacity(n_track_id_rows);
        let mut hypothesis_rows = Vec::with_capacity(n_hypotheses);
        let mut kf_state_rows = Vec::with_capacity(n_hypotheses);
        let mut best_state_rows = Vec::with_capacity(n_branches);

        let mut next_hypothesis_id: i64 = 0;

        for branch in &branch_collection.branches {
            let branch_id = branch.branch_id as i64;

            let stats = compute_obs_stats(branch.bank.track_ids(), obs_index);

            branch_rows.push(BranchRow {
                branch_id,
                lineage_id: branch.lineage_id as i64,
                parent_branch_id: branch.parent_branch_id as i64,
                ancestor_at_scan_horizon: branch.ancestor_at_scan_horizon as i64,
                ancestor_creation_step: branch.ancestor_creation_step as i64,
                last_real_update_step: branch.last_real_update_step as i64,
                n_real_updates: branch.n_real_updates as i64,
                cumulative_llr: branch.cumulative_llr,
                lineage_designation: branch.lineage_designation.to_string(),
                designation: branch.designation().to_string(),
                arc_length_days: stats.arc_length_days,
                n_nights: stats.n_nights,
                median_inter_night_dt_days: stats.median_inter_night_dt_days,
                n_observations: stats.n_observations,
                well_sampled_nights: stats.well_sampled_nights,
            });

            let bank_snapshot = branch.bank.to_snapshot();

            kf_bank_rows.push(KfBankRow {
                branch_id,
                n_steps: bank_snapshot.n_steps as i64,
                absolute_magnitude_estimate: bank_snapshot.absolute_magnitude_estimate,
                absolute_magnitude_sample_count: bank_snapshot.absolute_magnitude_sample_count
                    as i32,
            });

            for (position, id) in bank_snapshot.track_ids.iter().enumerate() {
                branch_observation_rows.push(BranchObservationRow {
                    branch_id,
                    position: position as i32,
                    obs_id: *id as i64,
                });
            }

            if let Some(best) =
                best_hypothesis_index(bank_snapshot.hypotheses.iter().map(|h| h.log_weight))
            {
                // Hypothesis ids are allocated sequentially just below, so the
                // best one's `kf_state` row lands at this offset.
                best_state_rows.push(BranchBestStateRow {
                    branch_id,
                    kf_state_index: kf_state_rows.len() + best,
                });
            }

            for hyp_snapshot in bank_snapshot.hypotheses {
                let hypothesis_id = next_hypothesis_id;
                next_hypothesis_id += 1;

                hypothesis_rows.push(HypothesisRow {
                    hypothesis_id,
                    branch_id,
                    local_hyp_id: hyp_snapshot.id as i64,
                    log_weight: hyp_snapshot.log_weight,
                    recent_log_liks: hyp_snapshot.recent_log_liks,
                });

                kf_state_rows.push(KfStateRow {
                    hypothesis_id,
                    fields: KfStateFields::from_snapshot(&hyp_snapshot.kf),
                });
            }
        }

        Self {
            branches: branch_rows,
            kf_bank: kf_bank_rows,
            branch_observations: branch_observation_rows,
            hypotheses: hypothesis_rows,
            kf_state: kf_state_rows,
            best_state: best_state_rows,
        }
    }

    /// Total number of rows over the 6 tables (for progress reporting).
    pub fn len(&self) -> usize {
        self.branches.len()
            + self.kf_bank.len()
            + self.branch_observations.len()
            + self.hypotheses.len()
            + self.kf_state.len()
            + self.best_state.len()
    }

    /// Whether there is no row at all (an empty collection).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Every observation id the branches of `branch_collection` reference,
/// deduplicated — the ids whose `(night_id, mjd_tt)` an [`ObsIndex`] must
/// hold for [`TrajectoryRows::build`] to compute complete statistics.
///
/// # Returns
///
/// The ids, sorted ascending, as `i64` (the `observations.id` type).
pub fn referenced_obs_ids(branch_collection: &BranchCollection) -> Vec<i64> {
    let mut ids: Vec<i64> = branch_collection
        .branches
        .iter()
        .flat_map(|b| b.bank.track_ids().iter().map(|&id| id as i64))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Builds every row of `archived_trajectories` for `archived`.
///
/// # Arguments
///
/// * `archived` — the trajectories to convert: a collection's whole archive
///   for `convert`, one night's batch for the streaming pipeline.
///
/// # Returns
///
/// One [`ArchivedRow`] per trajectory, in input order.
pub fn build_archived_rows(archived: &[ArchivedTrajectory]) -> Vec<ArchivedRow> {
    archived
        .iter()
        .map(|trajectory| ArchivedRow {
            designation: trajectory.designation.to_string(),
            lineage_id: trajectory.lineage_id as i64,
            track_ids: trajectory.track_ids.iter().map(|id| *id as i64).collect(),
            cumulative_llr: trajectory.cumulative_llr,
            n_real_updates: trajectory.n_real_updates as i64,
            last_real_update_step: trajectory.last_real_update_step as i64,
            archived_at_step: trajectory.archived_at_step as i64,
            absolute_magnitude_estimate: trajectory.absolute_magnitude_estimate,
            absolute_magnitude_sample_count: trajectory.absolute_magnitude_sample_count as i32,
            kf_state: KfStateFields::from_snapshot(&trajectory.map_state),
        })
        .collect()
}

/// Classifies every row's dynamical family in parallel with rayon, ahead of
/// the (necessarily sequential) `writer.write()` loop over the single shared
/// `Transaction` in `copy_kf_state`/`copy_archived_trajectories`.
///
/// # Arguments
///
/// * `fields` — one entry per row to classify.
///
/// # Returns
///
/// One `(family, semi_major_axis, eccentricity)` triple per input row, in the
/// same order; a row whose attributable state doesn't convert to a bound
/// orbit falls back to `(DynamicalFamily::Unknown, 0.0, 0.0)` rather than
/// failing the whole conversion over one degenerate hypothesis.
pub fn classify_rows_in_parallel(fields: &[&KfStateFields]) -> Vec<(DynamicalFamily, f64, f64)> {
    fields
        .par_iter()
        .map(|f| {
            classify_from_attributable_state(
                f.ra,
                f.dec,
                f.ra_dot,
                f.dec_dot,
                f.rho,
                f.rho_dot,
                f.epoch,
                Vector3::new(f.r_obs_x, f.r_obs_y, f.r_obs_z),
                Vector3::new(f.v_obs_x, f.v_obs_y, f.v_obs_z),
            )
            .unwrap_or((DynamicalFamily::Unknown, 0., 0.))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `obs_index` fixture: `obs_id -> (night_id, mjd_tt)`.
    fn index(entries: &[(i64, i64, f64)]) -> ObsIndex {
        entries
            .iter()
            .map(|&(obs_id, night_id, mjd_tt)| (obs_id, (night_id, mjd_tt)))
            .collect()
    }

    #[test]
    fn no_matching_observations_returns_zeroed_stats() {
        let obs_index = index(&[(1, 100, 60000.0)]);
        // track_ids references an id that isn't in obs_index at all.
        let stats = compute_obs_stats(&[999], &obs_index);
        assert_eq!(stats.arc_length_days, 0.0);
        assert_eq!(stats.n_nights, 0);
        assert_eq!(stats.median_inter_night_dt_days, None);
        assert_eq!(stats.n_observations, 0);
        assert_eq!(stats.well_sampled_nights, 0);
    }

    #[test]
    fn single_night_has_no_median_gap() {
        let obs_index = index(&[(1, 100, 60000.0), (2, 100, 60000.5)]);
        let stats = compute_obs_stats(&[1, 2], &obs_index);
        assert_eq!(stats.arc_length_days, 0.5);
        assert_eq!(stats.n_nights, 1);
        assert_eq!(stats.median_inter_night_dt_days, None);
    }

    #[test]
    fn two_nights_median_is_the_single_gap() {
        let obs_index = index(&[(1, 100, 60000.0), (2, 101, 60003.0)]);
        let stats = compute_obs_stats(&[1, 2], &obs_index);
        assert_eq!(stats.arc_length_days, 3.0);
        assert_eq!(stats.n_nights, 2);
        assert_eq!(stats.median_inter_night_dt_days, Some(3.0));
    }

    #[test]
    fn odd_number_of_gaps_takes_the_middle_one() {
        // Nights at day 0, 1, 4 -> gaps [1, 3] -> even count, averaged.
        // Add a 4th night at day 10 -> gaps [1, 3, 6] -> odd count, middle = 3.
        let obs_index = index(&[
            (1, 1, 60000.0),
            (2, 2, 60001.0),
            (3, 3, 60004.0),
            (4, 4, 60010.0),
        ]);
        let stats = compute_obs_stats(&[1, 2, 3, 4], &obs_index);
        assert_eq!(stats.n_nights, 4);
        assert_eq!(stats.median_inter_night_dt_days, Some(3.0));
    }

    #[test]
    fn even_number_of_gaps_averages_the_two_middle_ones() {
        // Nights at day 0, 1, 4 -> gaps [1, 3] -> median = (1+3)/2 = 2.
        let obs_index = index(&[(1, 1, 60000.0), (2, 2, 60001.0), (3, 3, 60004.0)]);
        let stats = compute_obs_stats(&[1, 2, 3], &obs_index);
        assert_eq!(stats.n_nights, 3);
        assert_eq!(stats.median_inter_night_dt_days, Some(2.0));
    }

    #[test]
    fn a_nights_representative_time_is_its_earliest_observation() {
        // Night 1 has two observations; the later one must not shift the
        // night's representative time used for the inter-night gap.
        let obs_index = index(&[
            (1, 1, 60000.5), // later observation of night 1
            (2, 1, 60000.0), // earlier observation of night 1
            (3, 2, 60002.0),
        ]);
        let stats = compute_obs_stats(&[1, 2, 3], &obs_index);
        // Arc length still spans every observation, not just night starts.
        assert_eq!(stats.arc_length_days, 2.0);
        assert_eq!(stats.n_nights, 2);
        // Gap measured from night 1's earliest obs (60000.0), not the latest.
        assert_eq!(stats.median_inter_night_dt_days, Some(2.0));
    }

    #[test]
    fn unmatched_ids_are_skipped_without_affecting_matched_ones() {
        let obs_index = index(&[(1, 1, 60000.0), (2, 2, 60002.0)]);
        let with_noise = compute_obs_stats(&[1, 999, 2, 1000], &obs_index);
        let without_noise = compute_obs_stats(&[1, 2], &obs_index);
        assert_eq!(with_noise, without_noise);
    }

    /// `n_observations` mirrors `COUNT(*)` over the SQL join: unmatched ids
    /// are left out, a repeated id is counted each time.
    #[test]
    fn n_observations_counts_matched_ids_including_repeats() {
        let obs_index = index(&[(1, 1, 60000.0), (2, 2, 60002.0)]);
        let stats = compute_obs_stats(&[1, 1, 2, 999], &obs_index);
        assert_eq!(stats.n_observations, 3);
    }

    #[test]
    fn well_sampled_nights_are_the_nights_with_two_or_more_observations() {
        let obs_index = index(&[
            (1, 1, 60000.0),
            (2, 1, 60000.1),
            (3, 2, 60001.0),
            (4, 3, 60002.0),
            (5, 3, 60002.1),
            (6, 3, 60002.2),
        ]);
        let stats = compute_obs_stats(&[1, 2, 3, 4, 5, 6], &obs_index);
        assert_eq!(stats.n_nights, 3);
        assert_eq!(stats.well_sampled_nights, 2);
    }

    #[test]
    fn best_hypothesis_is_the_largest_log_weight() {
        assert_eq!(best_hypothesis_index([-3.0, -0.5, -2.0]), Some(1));
        assert_eq!(best_hypothesis_index(std::iter::empty()), None);
    }

    /// Postgres' `ORDER BY log_weight DESC` puts NaN first; so must we.
    #[test]
    fn nan_log_weight_ranks_first_like_postgres() {
        assert_eq!(best_hypothesis_index([0.0, f64::NAN, 1.0]), Some(1));
        assert_eq!(best_hypothesis_index([f64::NEG_INFINITY, -1.0]), Some(1));
    }
}

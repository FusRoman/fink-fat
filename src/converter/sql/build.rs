//! Pure construction of every row [`super::write_sql_tables`] bulk-loads,
//! from a [`BranchCollection`] and the observation index built from it.
//!
//! Nothing here touches Postgres or the filesystem: nothing in this file
//! does I/O. That is what makes it the part of the SQL conversion pipeline
//! worth unit-testing directly (see the `tests` module below), rather than
//! only indirectly through a real database.

use std::collections::HashMap;

use fink_fat_engine::topocentric_kf::branching::BranchCollection;
use nalgebra::Vector3;
use rayon::prelude::*;

use crate::converter::family::{DynamicalFamily, classify_from_attributable_state};

use super::rows::{
    ArchivedRow, BranchObservationRow, BranchRow, HypothesisRow, KfBankRow, KfStateFields,
    KfStateRow,
};

/// Per-observation lookup keyed by `ObservationRow::id`
/// (== `BranchObservationRow::obs_id`), mapping to `(night_id, mjd_tt)`.
/// Built once in [`super::write_sql_tables`] from the observation rows read
/// off disk, and shared read-only by every branch's [`compute_obs_stats`]
/// call — computing each branch's arc/night-count/inter-night-gap stats once
/// here in Rust, rather than re-deriving them from Postgres on every
/// homepage sort.
pub(super) type ObsIndex = HashMap<i64, (i64, f64)>;

/// Arc length, unique night count, and the median gap between consecutive
/// nights for one branch's observations.
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
/// A triple of:
/// 1. `arc_length_days` — `max(mjd_tt) - min(mjd_tt)` across every matched
///    observation, `0.0` if none matched.
/// 2. `n_nights` — the number of distinct `night_id`s among matched
///    observations.
/// 3. `median_inter_night_dt_days` — the median gap (in days) between
///    consecutive nights, each night represented by its earliest
///    observation's `mjd_tt`; `None` when the branch spans at most one night
///    (there is then no inter-night gap to measure).
pub(super) fn compute_obs_stats(
    track_ids: &[u64],
    obs_index: &ObsIndex,
) -> (f64, i64, Option<f64>) {
    let mut night_start: HashMap<i64, f64> = HashMap::new();
    let mut min_mjd = f64::INFINITY;
    let mut max_mjd = f64::NEG_INFINITY;

    for &obs_id in track_ids {
        let Some(&(night_id, mjd_tt)) = obs_index.get(&(obs_id as i64)) else {
            continue;
        };
        min_mjd = min_mjd.min(mjd_tt);
        max_mjd = max_mjd.max(mjd_tt);
        night_start
            .entry(night_id)
            .and_modify(|t| *t = t.min(mjd_tt))
            .or_insert(mjd_tt);
    }

    if night_start.is_empty() {
        return (0.0, 0, None);
    }

    let arc_length_days = max_mjd - min_mjd;
    let n_nights = night_start.len() as i64;

    let mut night_times: Vec<f64> = night_start.into_values().collect();
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

    (arc_length_days, n_nights, median_inter_night_dt_days)
}

/// Builds every row of `branches`/`kf_bank`/`branch_observations`/
/// `hypotheses`/`kf_state` by walking `branch_collection` once.
///
/// Re-walks `&BranchCollection` with the same shape of loop as
/// `parquet::to_dataframes` rather than reading back the already-built
/// Polars `DataFrame`s: extracting typed values out of `ChunkedArray`/
/// `ListChunked` just to re-serialize them for `COPY` would be a pointless
/// extra materialization pass for a one-shot batch export.
///
/// # Arguments
///
/// * `branch_collection` — the loaded snapshot to convert.
/// * `obs_index` — see [`ObsIndex`]; used to compute each branch's
///   `arc_length_days`/`n_nights`/`median_inter_night_dt_days` via
///   [`compute_obs_stats`].
///
/// # Returns
///
/// The five row vectors, in insertion order matching `branch_collection`'s
/// own branch order; `hypothesis_id`/`kf_state` rows are aligned 1:1 with
/// `hypotheses` rows via a hypothesis id allocated sequentially across every
/// branch.
#[allow(clippy::type_complexity)]
pub(super) fn build_branch_rows(
    branch_collection: &BranchCollection,
    obs_index: &ObsIndex,
) -> (
    Vec<BranchRow>,
    Vec<KfBankRow>,
    Vec<BranchObservationRow>,
    Vec<HypothesisRow>,
    Vec<KfStateRow>,
) {
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

    let mut next_hypothesis_id: i64 = 0;

    for branch in &branch_collection.branches {
        let branch_id = branch.branch_id as i64;

        let (arc_length_days, n_nights, median_inter_night_dt_days) =
            compute_obs_stats(branch.bank.track_ids(), obs_index);

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
            arc_length_days,
            n_nights,
            median_inter_night_dt_days,
        });

        let bank_snapshot = branch.bank.to_snapshot();

        kf_bank_rows.push(KfBankRow {
            branch_id,
            n_steps: bank_snapshot.n_steps as i64,
            absolute_magnitude_estimate: bank_snapshot.absolute_magnitude_estimate,
            absolute_magnitude_sample_count: bank_snapshot.absolute_magnitude_sample_count as i32,
        });

        for (position, id) in bank_snapshot.track_ids.iter().enumerate() {
            branch_observation_rows.push(BranchObservationRow {
                branch_id,
                position: position as i32,
                obs_id: *id as i64,
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

    (
        branch_rows,
        kf_bank_rows,
        branch_observation_rows,
        hypothesis_rows,
        kf_state_rows,
    )
}

/// Builds every row of `archived_trajectories`.
///
/// # Arguments
///
/// * `branch_collection` — the loaded snapshot to convert; only
///   `branch_collection.archived` is read.
///
/// # Returns
///
/// One [`ArchivedRow`] per archived trajectory, in `branch_collection`'s own
/// order.
pub(super) fn build_archived_rows(branch_collection: &BranchCollection) -> Vec<ArchivedRow> {
    branch_collection
        .archived
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
pub(super) fn classify_rows_in_parallel(
    fields: &[&KfStateFields],
) -> Vec<(DynamicalFamily, f64, f64)> {
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
        let (arc_length_days, n_nights, median) = compute_obs_stats(&[999], &obs_index);
        assert_eq!(arc_length_days, 0.0);
        assert_eq!(n_nights, 0);
        assert_eq!(median, None);
    }

    #[test]
    fn single_night_has_no_median_gap() {
        let obs_index = index(&[(1, 100, 60000.0), (2, 100, 60000.5)]);
        let (arc_length_days, n_nights, median) = compute_obs_stats(&[1, 2], &obs_index);
        assert_eq!(arc_length_days, 0.5);
        assert_eq!(n_nights, 1);
        assert_eq!(median, None);
    }

    #[test]
    fn two_nights_median_is_the_single_gap() {
        let obs_index = index(&[(1, 100, 60000.0), (2, 101, 60003.0)]);
        let (arc_length_days, n_nights, median) = compute_obs_stats(&[1, 2], &obs_index);
        assert_eq!(arc_length_days, 3.0);
        assert_eq!(n_nights, 2);
        assert_eq!(median, Some(3.0));
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
        let (_, n_nights, median) = compute_obs_stats(&[1, 2, 3, 4], &obs_index);
        assert_eq!(n_nights, 4);
        assert_eq!(median, Some(3.0));
    }

    #[test]
    fn even_number_of_gaps_averages_the_two_middle_ones() {
        // Nights at day 0, 1, 4 -> gaps [1, 3] -> median = (1+3)/2 = 2.
        let obs_index = index(&[(1, 1, 60000.0), (2, 2, 60001.0), (3, 3, 60004.0)]);
        let (_, n_nights, median) = compute_obs_stats(&[1, 2, 3], &obs_index);
        assert_eq!(n_nights, 3);
        assert_eq!(median, Some(2.0));
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
        let (arc_length_days, n_nights, median) = compute_obs_stats(&[1, 2, 3], &obs_index);
        // Arc length still spans every observation, not just night starts.
        assert_eq!(arc_length_days, 2.0);
        assert_eq!(n_nights, 2);
        // Gap measured from night 1's earliest obs (60000.0), not the latest.
        assert_eq!(median, Some(2.0));
    }

    #[test]
    fn unmatched_ids_are_skipped_without_affecting_matched_ones() {
        let obs_index = index(&[(1, 1, 60000.0), (2, 2, 60002.0)]);
        let with_noise = compute_obs_stats(&[1, 999, 2, 1000], &obs_index);
        let without_noise = compute_obs_stats(&[1, 2], &obs_index);
        assert_eq!(with_noise, without_noise);
    }
}

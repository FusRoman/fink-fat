//! Per-night orchestration: propagate, search, branch, score, prune.
//!
//! Ties together propagation, candidate search, branching, LLR scoring and
//! cross-bank pruning for a whole set of lineages — steps 1–6 of
//! `kalman_update_instruction.md`'s per-night recap. Deliberately **not**
//! implemented here (see the `branching` module doc): inter-bank
//! deduplication ("promotion"), orbital-fit arbitration, cross-run
//! persistence.
//!
//! # Known simplification: two propagations per bank per night
//!
//! [`KFBank::predict_search_region`] and [`KFBank::predict_to`] each
//! propagate every hypothesis independently. The design doc's "propagate
//! once" principle is honoured *within* each call (every branch spawned
//! from one `predict_to`'d bank shares that single propagation), but
//! building the search region is a second, separate propagation. Unifying
//! them would require `predict_search_region` to accept an
//! already-propagated bank; left as a follow-up optimization — both calls
//! are read-only and idempotent, so this affects cost, not correctness.

use nalgebra::{Vector2, Vector3};
use photom::observation_dataset::observation::Observation;

use crate::{
    spacetime_bucket::{
        bucket::{BucketIndex, build_alert_bucket_index},
        clutter_density::local_clutter_density,
        healpix_binner::HealpixBinner,
    },
    topocentric_kf::{
        branching::{
            Branch,
            candidate_search::{SingleBinTimeBinner, find_candidates_for_bank},
            detection_probability::{detection_probability, predicted_apparent_magnitude},
            llr_score::{null_branch_llr_delta, observation_llr_delta},
            pruning::{apply_n_scan_pruning, cap_top_b_per_lineage},
        },
        kalman_bank::{
            KFBank,
            ellipse_region_finder::{radius_strategy::RadiusStrategy, top_k::TopK},
        },
    },
};

/// Advance every lineage by one night: propagate, search for candidates,
/// branch, score in LLR, and prune.
///
/// # Arguments
/// * `lineages` – Surviving branches from the previous night (or the
///   night-0 seed branches, see [`Branch::seed`]).
/// * `night_obs` – This night's observations.
/// * `spatial_binner`, `t_prop`, `r_obs_new`, `v_obs_new`, `obs_noise` –
///   Search-region/candidate-search inputs.
/// * `top_k`, `radius_strategy` – Passed to [`KFBank::predict_search_region`].
/// * `likelihood_threshold` – Minimum mixture likelihood for a candidate to
///   be considered at all (see [`find_candidates_for_bank`]).
/// * `branch_cap`, `n_scan` – Cross-bank pruning parameters (top-B, N-scan).
/// * `limiting_magnitude`, `completeness_width_mag` – Null-branch detection
///   probability parameters (see
///   [`detection_probability`](crate::topocentric_kf::branching::detection_probability)).
/// * `current_step` – Current night index, for N-scan bookkeeping.
///
/// # Returns
/// The surviving branches after this night's cap + N-scan pruning.
#[allow(clippy::too_many_arguments)]
pub fn advance_bank_collection_one_night<'state_lf>(
    lineages: &[Branch<'state_lf>],
    night_obs: &[&Observation],
    spatial_binner: &HealpixBinner,
    t_prop: f64,
    r_obs_new: Vector3<f64>,
    v_obs_new: Vector3<f64>,
    obs_noise: Vector2<f64>,
    top_k: TopK,
    radius_strategy: RadiusStrategy,
    likelihood_threshold: f64,
    branch_cap: usize,
    n_scan: usize,
    limiting_magnitude: f64,
    completeness_width_mag: f64,
    current_step: usize,
) -> Vec<Branch<'state_lf>> {
    let bucket_index = build_alert_bucket_index(
        night_obs.iter().copied(),
        spatial_binner,
        &SingleBinTimeBinner,
    );

    let mut next_branch_id = lineages
        .iter()
        .map(|branch| branch.branch_id)
        .max()
        .map_or(0, |id| id + 1);

    let mut all_branches = Vec::new();
    for lineage in lineages {
        let mut branches = spawn_branches_for_lineage(
            lineage,
            &bucket_index,
            spatial_binner,
            t_prop,
            r_obs_new,
            v_obs_new,
            obs_noise,
            top_k,
            radius_strategy,
            likelihood_threshold,
            limiting_magnitude,
            completeness_width_mag,
            &mut next_branch_id,
        );
        all_branches.append(&mut branches);
    }

    let capped = cap_top_b_per_lineage(all_branches, branch_cap);
    apply_n_scan_pruning(capped, n_scan, current_step)
}

/// Propagate one lineage's bank once, search its candidates, and spawn one
/// observation branch per match plus the mandatory null branch.
///
/// # Returns
/// The branches spawned for this lineage. Empty if the bank fails to
/// propagate at all (dropped, logged at `debug`).
#[allow(clippy::too_many_arguments)]
fn spawn_branches_for_lineage<'state_lf>(
    lineage: &Branch<'state_lf>,
    bucket_index: &BucketIndex<&Observation>,
    spatial_binner: &HealpixBinner,
    t_prop: f64,
    r_obs_new: Vector3<f64>,
    v_obs_new: Vector3<f64>,
    obs_noise: Vector2<f64>,
    top_k: TopK,
    radius_strategy: RadiusStrategy,
    likelihood_threshold: f64,
    limiting_magnitude: f64,
    completeness_width_mag: f64,
    next_branch_id: &mut u64,
) -> Vec<Branch<'state_lf>> {
    let predicted_bank = lineage.bank.predict_to(t_prop, r_obs_new, v_obs_new);

    let Ok(search_region) = lineage.bank.predict_search_region(
        t_prop,
        r_obs_new,
        v_obs_new,
        obs_noise,
        top_k,
        radius_strategy,
    ) else {
        tracing::debug!(
            lineage_id = lineage.lineage_id,
            "Bank failed to propagate for search-region construction, dropping lineage"
        );
        return Vec::new();
    };

    let candidates = find_candidates_for_bank(
        &search_region,
        lineage.bank.track_ids().to_vec(),
        bucket_index,
        spatial_binner,
        lineage.bank.config.gate_chi2,
        likelihood_threshold,
    );

    let mut branches = Vec::with_capacity(candidates.matches.len() + 1);
    for candidate in &candidates.matches {
        let clutter_density = local_clutter_density(
            bucket_index,
            spatial_binner,
            candidate.observation.equ_coord(),
        );
        let llr_delta = observation_llr_delta(candidate.likelihood, clutter_density);

        if let Some(branch) = Branch::from_observation(
            &predicted_bank,
            lineage,
            candidate.observation,
            llr_delta,
            *next_branch_id,
        ) {
            *next_branch_id += 1;
            branches.push(branch);
        }
    }

    let p_detection = null_branch_detection_probability(
        &predicted_bank,
        limiting_magnitude,
        completeness_width_mag,
    );
    branches.push(Branch::from_null(
        &predicted_bank,
        lineage,
        null_branch_llr_delta(p_detection),
        *next_branch_id,
    ));
    *next_branch_id += 1;

    branches
}

/// Detection probability for the null branch, from the bank's running
/// absolute-magnitude estimate and its MAP hypothesis's predicted geometry.
///
/// Falls back to `0.5` (neutral: neither favors nor penalizes the null
/// branch) when the bank has no magnitude history yet — this only happens
/// before the bank's first successful [`KFBank::branch_with`] call.
fn null_branch_detection_probability<'state_lf>(
    predicted_bank: &KFBank<'state_lf>,
    limiting_magnitude: f64,
    completeness_width_mag: f64,
) -> f64 {
    const NO_HISTORY_FALLBACK_P_DETECTION: f64 = 0.5;

    let (Some(absolute_magnitude_estimate), Some(best)) = (
        predicted_bank.absolute_magnitude_estimate(),
        predicted_bank.best(),
    ) else {
        return NO_HISTORY_FALLBACK_P_DETECTION;
    };

    let r_helio_au = best.kf.to_cartesian().pos.norm();
    let delta_topocentric_au = best.kf.state[4];
    let predicted_magnitude = predicted_apparent_magnitude(
        absolute_magnitude_estimate,
        r_helio_au,
        delta_topocentric_au,
    );

    detection_probability(
        predicted_magnitude,
        limiting_magnitude,
        completeness_width_mag,
    )
}

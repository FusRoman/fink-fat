//! Per-night orchestration: propagate, search, branch, score, prune.
//!
//! Ties together propagation, candidate search, branching, LLR scoring and
//! cross-bank pruning for a whole set of lineages — steps 1–6 of
//! `kalman_update_instruction.md`'s per-night recap. Deliberately **not**
//! implemented here (see the `branching` module doc): inter-bank
//! deduplication ("promotion"), orbital-fit arbitration, cross-run
//! persistence. Seeding brand-new lineages from observations no existing
//! lineage claims lives one level up, in
//! [`collection`](super::collection)/[`discovery`](super::discovery) — this
//! module only ever *advances* lineages that already exist.
//!
//! # Why this loops over visits, not the whole night at once
//!
//! A night is not one instant: at LSST cadence a single night holds
//! hundreds of distinct exposure epochs (~30s apart over ~8h). Propagating
//! every lineage to one shared per-night epoch and then updating against
//! candidates from many *different* actual epochs would compare a
//! predicted state against observations it was never propagated to — wrong,
//! not just imprecise. [`Visit`] groups observations that genuinely share
//! one epoch (see [`visit`](super::visit)); within one visit, "propagate
//! once, branch many candidates" ([`KFBank::predict_to`]/
//! [`KFBank::branch_with`]) is exactly correct, since every candidate in a
//! visit really does share that epoch.
//!
//! # Performance: avoid propagating every lineage at every visit
//!
//! Naively calling `predict_to`/`predict_search_region` (two two-body
//! Kepler solves per hypothesis) for *every* lineage at *every* visit is
//! O(lineages × visits) — infeasible at LSST scale (potentially 10⁵–10⁶
//! lineages × ~10³ visits/night). This module gates that cost with a cheap,
//! no-Kepler-solve pre-filter (linear extrapolation from the bank's own
//! stored angular position/rate, checked against the visit's alert index)
//! before paying for the real propagation.
//!
//! # `n_steps` only advances on real updates
//!
//! [`KFBank::branch_null`] does not touch `n_steps` — see its doc for why
//! that matters once null branches can be spawned hundreds of times a
//! night (once per visit with no candidate) instead of once per night.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::Vector3;
use outfit::OutfitError;
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};
use rayon::prelude::*;

use crate::{
    engine_config::{kalman_context::KalmanContext, night_advance_params::NightAdvanceParams},
    spacetime_bucket::{
        bucket::{BucketIndex, BucketKey, build_alert_bucket_index},
        clutter_density::local_clutter_density,
        healpix_binner::HealpixBinner,
        spatial_binner::SpatialBinner,
    },
    topocentric_kf::{
        branching::{
            Branch,
            candidate_search::{SINGLE_TIME_BIN, SingleBinTimeBinner, find_candidates_for_bank},
            detection_probability::{detection_probability, predicted_apparent_magnitude},
            llr_score::{null_branch_llr_delta, observation_llr_delta},
            pruning::{apply_n_scan_pruning, cap_top_b_per_lineage},
            visit::{Visit, group_observations_into_visits},
        },
        kalman_bank::KFBank,
        observer_state::get_observer,
        single_kalman::update::wrap_angle,
    },
};

/// Result of advancing a set of lineages by one night.
pub struct NightAdvanceOutcome<'state_lf, 'bank_config> {
    /// Surviving branches after this night's cap + N-scan pruning.
    pub branches: Vec<Branch<'state_lf, 'bank_config>>,
    /// Ids of every observation that passed the two-stage gate for at least
    /// one lineage's search region this night — regardless of whether the
    /// branch carrying it survived pruning. Consumed by the discovery step
    /// (`seed_new_lineages_from_leftovers`) so an observation plausibly
    /// belonging to a known track doesn't also seed a brand-new one.
    pub consumed_observation_ids: HashSet<ObsId>,
}

/// Advance every lineage by one night, one visit (exposure epoch) at a
/// time: propagate, search for candidates, branch, score in LLR, and
/// top-B-prune — repeated per visit, then N-scan-pruned once for the whole
/// night.
///
/// # Arguments
/// * `lineages` – Branches to advance (from the previous night, or freshly
///   seeded — see [`Branch::seed`]).
/// * `night_obs` – This night's observations, grouped into visits
///   internally (see [`group_observations_into_visits`]).
/// * `obs_dataset`, `kalman_context` – Needed to resolve each visit's
///   observer heliocentric state at its own epoch.
/// * `params` – Tuning parameters, shared across every lineage and visit.
/// * `current_step` – Current night index, for N-scan bookkeeping.
///
/// # Returns
/// The surviving branches after pruning, plus the set of observation ids
/// they consumed (see [`NightAdvanceOutcome`]).
pub fn advance_bank_collection_one_night<'state_lf, 'bank_config>(
    lineages: &[Branch<'state_lf, 'bank_config>],
    night_obs: &[&Observation],
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
    params: &NightAdvanceParams,
    spatial_binner: &HealpixBinner,
    current_step: usize,
) -> NightAdvanceOutcome<'state_lf, 'bank_config> {
    let span = tracing::info_span!("Advance bank collection");
    let _enter = span.enter();

    let visits = group_observations_into_visits(night_obs, params.visit_epoch_tolerance_days);

    tracing::debug!("Number of visit: {}", visits.len());

    let mut branches: Vec<Branch<'state_lf, 'bank_config>> = lineages.to_vec();
    let mut consumed_observation_ids = HashSet::new();
    let next_branch_id = AtomicU64::new(
        lineages
            .iter()
            .map(|branch| branch.branch_id)
            .max()
            .map_or(0, |id| id + 1),
    );

    for visit in &visits {
        let Ok((r_obs, v_obs)) = resolve_observer_state(
            obs_dataset,
            kalman_context,
            visit.representative_obs,
            visit.epoch,
        ) else {
            tracing::debug!(
                epoch = visit.epoch,
                "Failed to resolve observer state for visit, skipping"
            );
            continue;
        };

        // Built once per VISIT, reused by every lineage's cheap pre-filter,
        // candidate search and clutter-density lookup this visit.
        let visit_bucket_index = build_alert_bucket_index(
            visit.observations.iter().copied(),
            spatial_binner,
            &SingleBinTimeBinner,
        );

        // Per-lineage work is independent within a visit (each lineage only
        // ever reads its own bank) and dominated by the two-body Kepler
        // propagation in `spawn_branches_for_lineage` — parallelize across
        // lineages via rayon. `next_branch_id` is a shared atomic counter
        // (branch ids are opaque unique keys, never relied on for ordering
        // — see `branch_id.rs`); `consumed_observation_ids` can't be
        // mutated from parallel closures, so each lineage returns its own
        // consumed ids and they're merged sequentially below (cheap: a
        // `HashSet::extend`, not a bottleneck).
        let outcomes: Vec<LineageOutcome<'state_lf, 'bank_config>> = branches
            .par_iter()
            .map(|lineage| {
                if !lineage_might_be_in_visit(
                    lineage,
                    visit,
                    &visit_bucket_index,
                    params,
                    spatial_binner,
                ) {
                    // No alert anywhere near this lineage's extrapolated
                    // position — skip the two Kepler solves entirely. The
                    // lineage carries over unchanged, picked up again
                    // whenever a later visit's alerts land nearby.
                    return LineageOutcome::Unchanged(lineage.clone());
                }

                let (spawned, consumed) = spawn_branches_for_lineage(
                    lineage,
                    &visit_bucket_index,
                    visit.epoch,
                    r_obs,
                    v_obs,
                    params,
                    spatial_binner,
                    &next_branch_id,
                );
                LineageOutcome::Spawned(spawned, consumed)
            })
            .collect();

        let mut visit_branches = Vec::new();
        for outcome in outcomes {
            match outcome {
                LineageOutcome::Unchanged(branch) => visit_branches.push(branch),
                LineageOutcome::Spawned(spawned, consumed) => {
                    visit_branches.extend(spawned);
                    consumed_observation_ids.extend(consumed);
                }
            }
        }

        // Top-B pruning after every visit — branch counts multiply at every
        // branching event; left unbounded across a night's worth of visits
        // this would explode long before the night ends.
        branches = cap_top_b_per_lineage(visit_branches, params.branch_cap);
    }

    // N-scan stays at night granularity: one call, after every visit this
    // night has been folded in.
    let branches = apply_n_scan_pruning(branches, params.n_scan, current_step);

    NightAdvanceOutcome {
        branches,
        consumed_observation_ids,
    }
}

/// Resolve the observer's heliocentric state at `epoch`.
///
/// `representative_obs` is only used to identify *which* observer
/// (site/instrument) took this visit — any observation in the visit works,
/// they share one exposure.
fn resolve_observer_state(
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
    representative_obs: &Observation,
    epoch: f64,
) -> Result<(Vector3<f64>, Vector3<f64>), OutfitError> {
    let observer = get_observer(obs_dataset, representative_obs)?;
    let helio_state = kalman_context
        .get_ephem()
        .helio_observer_state(observer, epoch)?;
    Ok((helio_state.helio_cart_pos, helio_state.helio_cart_vel))
}

/// Cheap rejection test: does a linear extrapolation of `lineage`'s last
/// known sky position land near any alert in `visit`?
///
/// No two-body solve — just arithmetic on the bank's already-stored angular
/// position/rate (`state[0..=3]`, identical across every hypothesis in the
/// bank: only `ρ`/`ρ̇` differ between them, that's the whole point of the
/// bank — so this extrapolation is exact for the angular part, not an
/// approximation of "which hypothesis to trust") plus a HEALPix neighbor
/// lookup against `visit_bucket_index` (already built for this visit,
/// reused here, not rebuilt).
///
/// `false` means "definitely nothing nearby, skip the real propagation
/// this visit"; `true` still requires
/// [`spawn_branches_for_lineage`]'s real (Mahalanobis-gated) check — this
/// is a coarse, deliberately generous filter, not a replacement for it.
fn lineage_might_be_in_visit(
    lineage: &Branch,
    visit: &Visit,
    visit_bucket_index: &BucketIndex<&Observation>,
    params: &NightAdvanceParams,
    spatial_binner: &HealpixBinner,
) -> bool {
    let Some(best) = lineage.bank.best() else {
        return false;
    };

    let dt = visit.epoch - best.kf.epoch;
    let predicted_ra = wrap_angle(best.kf.state[0] + best.kf.state[2] * dt);
    let predicted_dec = best.kf.state[1] + best.kf.state[3] * dt;
    let predicted_coord = EquCoord::new(predicted_ra, 0.0, predicted_dec, 0.0);

    let predicted_key = spatial_binner.key_for(&predicted_coord);

    spatial_binner
        .neighbors(predicted_key, params.quick_reject_radius_rad)
        .into_iter()
        .any(|space_key| {
            visit_bucket_index.buckets.contains_key(&BucketKey {
                space_key,
                time_bin: SINGLE_TIME_BIN,
            })
        })
}

/// Outcome of checking one lineage against one visit — either it carries
/// over unchanged (no candidate nearby), or it spawned branches plus the
/// observation ids they consumed. Kept separate from mutating shared state
/// directly so the per-lineage work in
/// [`advance_bank_collection_one_night`] can run in parallel via rayon.
enum LineageOutcome<'state_lf, 'bank_config> {
    Unchanged(Branch<'state_lf, 'bank_config>),
    Spawned(Vec<Branch<'state_lf, 'bank_config>>, Vec<ObsId>),
}

/// Propagate one lineage's bank once (to this visit's epoch), search its
/// candidates, and spawn one observation branch per match plus the
/// mandatory null branch.
///
/// # Returns
/// The branches spawned for this lineage, plus the ids of every candidate
/// observation consumed (the caller merges these into the night's
/// `consumed_observation_ids`). Empty if the bank fails to propagate at all
/// (dropped, logged at `debug`).
#[allow(clippy::too_many_arguments)]
fn spawn_branches_for_lineage<'state_lf, 'bank_config>(
    lineage: &Branch<'state_lf, 'bank_config>,
    visit_bucket_index: &BucketIndex<&Observation>,
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
    params: &NightAdvanceParams,
    spatial_binner: &HealpixBinner,
    next_branch_id: &AtomicU64,
) -> (Vec<Branch<'state_lf, 'bank_config>>, Vec<ObsId>) {
    let predicted_bank = lineage.bank.predict_to(epoch, r_obs, v_obs);

    let Ok(search_region) = predicted_bank.search_region(
        params.obs_noise.into(),
        params.top_k,
        params.radius_strategy,
    ) else {
        tracing::debug!(
            lineage_id = lineage.lineage_id,
            "Bank failed to propagate for search-region construction, dropping lineage"
        );
        return (Vec::new(), Vec::new());
    };

    let candidates = find_candidates_for_bank(
        &search_region,
        lineage.bank.track_ids().to_vec(),
        visit_bucket_index,
        spatial_binner,
        lineage.bank.config.gate_chi2,
        params.likelihood_threshold,
    );

    let mut branches = Vec::with_capacity(candidates.matches.len() + 1);
    let mut consumed_observation_ids = Vec::with_capacity(candidates.matches.len());
    for candidate in &candidates.matches {
        consumed_observation_ids.push(*candidate.observation.id());

        let clutter_density = local_clutter_density(
            visit_bucket_index,
            spatial_binner,
            candidate.observation.equ_coord(),
        );
        let llr_delta = observation_llr_delta(candidate.likelihood, clutter_density);

        if let Some(branch) = Branch::from_observation(
            &predicted_bank,
            lineage,
            candidate.observation,
            llr_delta,
            next_branch_id.fetch_add(1, Ordering::Relaxed),
        ) {
            branches.push(branch);
        }
    }

    let p_detection = null_branch_detection_probability(
        &predicted_bank,
        params.limiting_magnitude,
        params.completeness_width_mag,
    );
    branches.push(Branch::from_null(
        &predicted_bank,
        lineage,
        null_branch_llr_delta(p_detection),
        next_branch_id.fetch_add(1, Ordering::Relaxed),
    ));

    (branches, consumed_observation_ids)
}

/// Detection probability for the null branch, from the bank's running
/// absolute-magnitude estimate and its MAP hypothesis's predicted geometry.
///
/// Falls back to `0.5` (neutral: neither favors nor penalizes the null
/// branch) when the bank has no magnitude history yet — this only happens
/// before the bank's first successful [`KFBank::branch_with`] call.
fn null_branch_detection_probability<'state_lf, 'bank_config>(
    predicted_bank: &KFBank<'state_lf, 'bank_config>,
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

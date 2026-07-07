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

use nalgebra::{Vector2, Vector3};
use outfit::OutfitError;
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};

use crate::{
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
        kalman_bank::{
            KFBank,
            ellipse_region_finder::{radius_strategy::RadiusStrategy, top_k::TopK},
        },
        observer_state::get_observer,
        single_kalman::{context::KalmanContext, update::wrap_angle},
    },
};

/// Tuning parameters shared by every lineage advanced in one call to
/// [`advance_bank_collection_one_night`] — everything except the branches
/// being advanced and the current step index, which are the function's
/// primary inputs rather than tuning knobs.
///
/// Grouped in the order they're consumed by the pipeline: visit grouping →
/// the cheap pre-filter → candidate search → cross-bank pruning →
/// null-branch detection probability.
pub struct NightAdvanceParams<'a> {
    /// HEALPix spatial index shared by every step of the night: building
    /// each visit's `BucketIndex`, the cheap pre-filter's neighbor lookup
    /// (see the module-level "Performance" note), candidate search
    /// ([`find_candidates_for_bank`]) and clutter-density estimation
    /// ([`local_clutter_density`]). One instance, reused everywhere — its
    /// `depth` sets the spatial resolution for all of them at once.
    pub spatial_binner: &'a HealpixBinner,

    /// Maximum epoch spread, **in days**, for two observations to be folded
    /// into the same [`Visit`] (see [`group_observations_into_visits`]).
    /// Should be small — a fraction of the exposure/readout time, e.g.
    /// a few seconds expressed as a fraction of a day (`1.0 / 86_400.0` ≈
    /// 1 second) — since its only job is to absorb per-alert timestamp
    /// jitter *within* one exposure, not to merge distinct visits. Too
    /// large would incorrectly treat two different exposures (and thus two
    /// different observer/geometry states) as one epoch.
    pub visit_epoch_tolerance_days: f64,

    /// Angular radius, **in radians**, for the cheap linear-extrapolation
    /// pre-filter (see the module-level "Performance" note). For every
    /// lineage, at every visit, this is the
    /// search radius used to test — via a HEALPix neighbor lookup, no
    /// Kepler solve — whether *any* alert in the visit falls near the
    /// lineage's linearly-extrapolated sky position; a lineage that fails
    /// this test never pays for the real (expensive) propagation this
    /// visit. Must be **generous**: it has to cover both the linear
    /// extrapolation's own error (curvature/eccentricity effects it
    /// ignores) and realistic positional uncertainty growth over the
    /// elapsed time since the lineage's last update. Too small silently
    /// drops real associations (a lineage never gets the chance to match);
    /// too large only costs an occasional wasted full propagation — when
    /// in doubt, err large.
    pub quick_reject_radius_rad: f64,

    /// **A priori** diagonal astrometric noise `[σ_RA², σ_Dec²]`, **in
    /// rad²**, added to each hypothesis's predicted sky covariance solely to
    /// size the search region in [`KFBank::predict_search_region`].
    ///
    /// # Why this is added on top of the sky covariance, not redundant with it
    ///
    /// `kf.sky_covariance()` ($HPH^\top$) is *our* uncertainty about where
    /// the object actually is, accumulated by the filter (process noise,
    /// propagation, prior measurements) — it says nothing about the noise
    /// of a *new* measurement we haven't taken yet. This field is that
    /// second term ($R$): even a perfectly-known position would still show
    /// up scattered by this much in a real observation, due to
    /// instrumental/astrometric noise. The combined innovation covariance
    /// $S = HPH^\top + R$ is the standard Kalman formula — the exact same
    /// additive pattern `Hypothesis::score_and_update` uses downstream for
    /// the real update. Dropping this term would make the search ellipse
    /// too small and miss valid associations.
    ///
    /// # Why "a priori" instead of the real per-alert error
    ///
    /// At search-region time no candidate has been found yet, so there is
    /// no real per-alert error to use — a generic estimate of the survey's
    /// typical astrometric precision stands in instead (e.g. `σ ≈ 0.1"` →
    /// `σ_rad ≈ 4.85e-7`, so `σ² ≈ 2.35e-13`). Once candidates are actually
    /// found, every real scoring/update step downstream —
    /// [`KFBank::branch_with`]'s mixture-likelihood scoring and the Kalman
    /// update itself — ignores this field entirely and instead uses the
    /// candidate observation's *own* `ra_error`/`dec_error` (via
    /// `Observation::equ_coord`), which is always the more accurate value
    /// once it's available.
    pub obs_noise: Vector2<f64>,

    /// Which of a bank's live hypotheses contribute to its predicted
    /// search region this visit — see [`TopK`] for the available policies
    /// (`All`, `Map`, `Best(k)`, `WeightThreshold`). Passed straight
    /// through to [`KFBank::predict_search_region`].
    pub top_k: TopK,

    /// How the search region's bounding radius is computed from the
    /// selected hypotheses' covariances — see [`RadiusStrategy`]
    /// (`MixtureCovariance` vs. `MaxEllipse`). Passed straight through to
    /// [`KFBank::predict_search_region`].
    pub radius_strategy: RadiusStrategy,

    /// Minimum mixture predictive likelihood (unitless, a Gaussian density
    /// value — see
    /// [`SearchRegion::mixture_likelihood`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion::mixture_likelihood))
    /// a candidate observation must reach, *after* passing the coarse
    /// per-component Mahalanobis gate, to be kept by
    /// [`find_candidates_for_bank`]. A second-stage cut on top of the gate:
    /// the gate says "geometrically plausible," this says "and not
    /// negligibly unlikely." `0.0` disables this stage (keep everything the
    /// gate accepts).
    pub likelihood_threshold: f64,

    /// Top-B cap: maximum number of branches kept **per lineage**, applied
    /// via [`cap_top_b_per_lineage`] after *every visit* — not just once
    /// per night, since branch counts multiply at every branching event
    /// (M candidates + 1 null branch) and would explode across a night's
    /// worth of visits otherwise. The design doc recommends `B ≈ 3–5`: wide
    /// enough to carry real ambiguity a visit or two, narrow enough to
    /// bound cost.
    pub branch_cap: usize,

    /// N-scan pruning window, **in nights** (not visits — see
    /// [`apply_n_scan_pruning`]), applied exactly once per call to
    /// [`advance_bank_collection_one_night`], after every visit that night
    /// has been folded in. For every branch-tree node this many nights old,
    /// only the single best-scoring descendant survives; siblings are
    /// discarded. `1` is the design doc's recommendation (association
    /// ambiguity usually resolves by the very next night); `2` is
    /// mentioned as an occasional alternative for slower-resolving cases.
    pub n_scan: usize,

    /// Survey/field limiting magnitude (mag) for this night, used as the
    /// midpoint of the null branch's detection-probability curve — see
    /// [`detection_probability`]. A lineage predicted brighter than this is
    /// very likely to have been detected (so a non-detection weighs heavily
    /// against the null branch); predicted fainter, the opposite.
    pub limiting_magnitude: f64,

    /// Completeness roll-off width (mag) of the survey's detection curve
    /// around `limiting_magnitude` — see [`detection_probability`]. Real
    /// surveys don't have a hard cutoff magnitude; detection probability
    /// decays smoothly over roughly this many magnitudes on either side of
    /// `limiting_magnitude`. Typical values ≈ 0.3–0.5 mag; must be strictly
    /// positive.
    pub completeness_width_mag: f64,
}

/// Result of advancing a set of lineages by one night.
pub struct NightAdvanceOutcome<'state_lf> {
    /// Surviving branches after this night's cap + N-scan pruning.
    pub branches: Vec<Branch<'state_lf>>,
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
pub fn advance_bank_collection_one_night<'state_lf>(
    lineages: &[Branch<'state_lf>],
    night_obs: &[&Observation],
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
    params: &NightAdvanceParams,
    current_step: usize,
) -> NightAdvanceOutcome<'state_lf> {
    let visits = group_observations_into_visits(night_obs, params.visit_epoch_tolerance_days);

    let mut branches: Vec<Branch<'state_lf>> = lineages.to_vec();
    let mut consumed_observation_ids = HashSet::new();
    let mut next_branch_id = lineages
        .iter()
        .map(|branch| branch.branch_id)
        .max()
        .map_or(0, |id| id + 1);

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
            params.spatial_binner,
            &SingleBinTimeBinner,
        );

        let mut visit_branches = Vec::new();
        for lineage in &branches {
            if !lineage_might_be_in_visit(lineage, visit, &visit_bucket_index, params) {
                // No alert anywhere near this lineage's extrapolated
                // position — skip the two Kepler solves entirely. The
                // lineage carries over unchanged, picked up again whenever
                // a later visit's alerts land nearby.
                visit_branches.push(lineage.clone());
                continue;
            }

            let spawned = spawn_branches_for_lineage(
                lineage,
                &visit_bucket_index,
                visit.epoch,
                r_obs,
                v_obs,
                params,
                &mut next_branch_id,
                &mut consumed_observation_ids,
            );
            visit_branches.extend(spawned);
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
) -> bool {
    let Some(best) = lineage.bank.best() else {
        return false;
    };

    let dt = visit.epoch - best.kf.epoch;
    let predicted_ra = wrap_angle(best.kf.state[0] + best.kf.state[2] * dt);
    let predicted_dec = best.kf.state[1] + best.kf.state[3] * dt;
    let predicted_coord = EquCoord::new(predicted_ra, 0.0, predicted_dec, 0.0);

    let predicted_key = params.spatial_binner.key_for(&predicted_coord);
    params
        .spatial_binner
        .neighbors(predicted_key, params.quick_reject_radius_rad)
        .into_iter()
        .any(|space_key| {
            visit_bucket_index.buckets.contains_key(&BucketKey {
                space_key,
                time_bin: SINGLE_TIME_BIN,
            })
        })
}

/// Propagate one lineage's bank once (to this visit's epoch), search its
/// candidates, and spawn one observation branch per match plus the
/// mandatory null branch.
///
/// # Returns
/// The branches spawned for this lineage. Empty if the bank fails to
/// propagate at all (dropped, logged at `debug`).
#[allow(clippy::too_many_arguments)]
fn spawn_branches_for_lineage<'state_lf>(
    lineage: &Branch<'state_lf>,
    visit_bucket_index: &BucketIndex<&Observation>,
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
    params: &NightAdvanceParams,
    next_branch_id: &mut u64,
    consumed_observation_ids: &mut HashSet<ObsId>,
) -> Vec<Branch<'state_lf>> {
    let predicted_bank = lineage.bank.predict_to(epoch, r_obs, v_obs);

    let Ok(search_region) = lineage.bank.predict_search_region(
        epoch,
        r_obs,
        v_obs,
        params.obs_noise,
        params.top_k,
        params.radius_strategy,
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
        visit_bucket_index,
        params.spatial_binner,
        lineage.bank.config.gate_chi2,
        params.likelihood_threshold,
    );

    let mut branches = Vec::with_capacity(candidates.matches.len() + 1);
    for candidate in &candidates.matches {
        consumed_observation_ids.insert(*candidate.observation.id());

        let clutter_density = local_clutter_density(
            visit_bucket_index,
            params.spatial_binner,
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
        params.limiting_magnitude,
        params.completeness_width_mag,
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

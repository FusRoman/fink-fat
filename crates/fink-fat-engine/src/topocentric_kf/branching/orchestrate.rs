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
            ArchivedTrajectory, Branch,
            candidate_filters::apply_candidate_filters,
            candidate_search::{
                SINGLE_TIME_BIN, SingleBinTimeBinner, find_candidates_for_bank,
                find_candidates_for_bank_multi_region,
            },
            detection_probability::{detection_probability, predicted_apparent_magnitude},
            llr_score::{null_branch_llr_delta, observation_llr_delta, photometric_llr_delta},
            pruning::{apply_n_scan_pruning, cap_top_b_per_lineage, purge_stale_lineages},
            visit::{Visit, group_observations_into_visits},
        },
        kalman_bank::{KFBank, ellipse_region_finder::cover_if_clamped},
        observer_state::get_observer,
        single_kalman::update::wrap_angle,
    },
};

use crate::logging::LogTarget;

/// Structured log events for the per-night lineage orchestration
/// (propagate → search → branch → score → prune). See [`crate::logging`]
/// for the `.emit()` pattern.
pub enum OrchestrateEvent {
    VisitSummary {
        n_visits: usize,
    },
    /// Coarse per-night progress through the visit loop — gated modulo at
    /// the call site so a night with hundreds/thousands of visits never
    /// emits more than ~100 lines.
    VisitProgress {
        visit_index: usize,
        n_visits: usize,
    },
    ObserverResolutionFailed {
        epoch: f64,
    },
    LineagePropagationFailed {
        lineage_id: u64,
    },
    NightPruningSummary {
        n_branches_before_n_scan: usize,
        n_branches_after: usize,
    },
}

crate::impl_log_target!(
    OrchestrateEvent,
    "orchestrate",
    "Per-night orchestration of existing lineages: propagate, search, branch, score, prune",
    [tracing::Level::INFO, tracing::Level::DEBUG]
);

impl OrchestrateEvent {
    pub fn emit(&self) {
        use OrchestrateEvent::*;
        match self {
            VisitSummary { n_visits } => tracing::debug!(
                target: OrchestrateEvent::TARGET, n_visits, "Night split into visits"
            ),
            VisitProgress {
                visit_index,
                n_visits,
            } => tracing::info!(
                target: OrchestrateEvent::TARGET, visit_index, n_visits, "Processing visit"
            ),
            ObserverResolutionFailed { epoch } => tracing::debug!(
                target: OrchestrateEvent::TARGET, epoch, "Failed to resolve observer state for visit, skipping"
            ),
            LineagePropagationFailed { lineage_id } => tracing::debug!(
                target: OrchestrateEvent::TARGET, lineage_id,
                "Bank failed to propagate for search-region construction, carrying lineage over unchanged this visit"
            ),
            NightPruningSummary {
                n_branches_before_n_scan,
                n_branches_after,
            } => tracing::debug!(
                target: OrchestrateEvent::TARGET, n_branches_before_n_scan, n_branches_after, "N-scan pruning summary"
            ),
        }
    }

    pub fn span() -> tracing::Span {
        tracing::info_span!(target: OrchestrateEvent::TARGET, "Advance bank collection")
    }
}

/// One existing lineage's cross-night gate outcome for a single visit —
/// passive diagnostics only (the engine's tracking logic ignores it). Lets an
/// evaluator measure gate *selectivity*: how many observations passed a
/// lineage's gate and, against ground truth, how many belonged to a different
/// object (contamination). `nights_since_seed` is `current_step −
/// last_real_update_step`, so `== 1` marks the first cross-night association
/// after seeding — the contamination hotspot.
#[derive(Debug, Clone)]
pub struct GateRecord {
    pub lineage_id: u64,
    pub nights_since_seed: usize,
    pub gated_obs_ids: Vec<ObsId>,
}

/// Result of advancing a set of lineages by one night.
pub struct NightAdvanceOutcome<'state_lf, 'bank_config> {
    /// Surviving branches after this night's cap + N-scan pruning.
    pub branches: Vec<Branch<'state_lf, 'bank_config>>,
    /// One [`GateRecord`] per (existing lineage, visit) that gated at least
    /// one candidate this night — passive diagnostics for gate-selectivity
    /// analysis, ignored by the engine itself.
    pub gate_records: Vec<GateRecord>,
    /// Ids of every observation that passed the two-stage gate for at least
    /// one lineage's search region this night — regardless of whether the
    /// branch carrying it survived pruning. The caller
    /// ([`BranchCollection::advance_one_night`](super::collection::BranchCollection::advance_one_night))
    /// excludes `consumed_observation_ids.difference(&consumed_then_pruned_ids)`
    /// (i.e. observations truly held by a *surviving* branch) from the
    /// discovery step's (`seed_new_lineages_from_leftovers`) candidate pool,
    /// so an observation plausibly belonging to a known track doesn't also
    /// seed a brand-new one.
    pub consumed_observation_ids: HashSet<ObsId>,
    /// The subset of `consumed_observation_ids` that ended up in none of the
    /// surviving `branches`' track histories — i.e. observations claimed by
    /// a candidate extension of an existing lineage that was itself pruned
    /// (by `cap_top_b_per_lineage` or `apply_n_scan_pruning`) before the end
    /// of this night. The caller gives these back to the leftover pool for
    /// discovery (see `consumed_observation_ids`'s doc) instead of
    /// permanently losing them: a candidate extension that didn't survive
    /// pruning was never a real claim, and an observation only ever appears
    /// in one night's `night_obs` — there is no later night where it could
    /// be revisited if not resolved now.
    pub consumed_then_pruned_ids: HashSet<ObsId>,
    /// Reconstructions of the lineages that stopped being propagated this
    /// night — see
    /// [`purge_stale_lineages`]. These are *results*, not state: they carry an
    /// association history but no Kalman bank, and never re-enter `branches`.
    /// The caller accumulates them (see
    /// [`BranchCollection::archived`](super::collection::BranchCollection::archived)),
    /// since a purged lineage's arc is as much an output of the run as a live
    /// branch's.
    pub archived: Vec<ArchivedTrajectory>,
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
    let _enter = OrchestrateEvent::span().entered();

    let visits = group_observations_into_visits(night_obs, params.visit_epoch_tolerance_days);

    OrchestrateEvent::VisitSummary {
        n_visits: visits.len(),
    }
    .emit();
    let visit_progress_modulo = (visits.len() / 100).max(1);

    let mut branches: Vec<Branch<'state_lf, 'bank_config>> = lineages.to_vec();
    let mut consumed_observation_ids = HashSet::new();
    let mut gate_records: Vec<GateRecord> = Vec::new();
    let next_branch_id = AtomicU64::new(
        lineages
            .iter()
            .map(|branch| branch.branch_id)
            .max()
            .map_or(0, |id| id + 1),
    );

    for (visit_index, visit) in visits.iter().enumerate() {
        if visit_index % visit_progress_modulo == 0 {
            OrchestrateEvent::VisitProgress {
                visit_index,
                n_visits: visits.len(),
            }
            .emit();
        }

        let Ok((r_obs, v_obs)) = resolve_observer_state(
            obs_dataset,
            kalman_context,
            visit.representative_obs,
            visit.epoch,
        ) else {
            OrchestrateEvent::ObserverResolutionFailed { epoch: visit.epoch }.emit();
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

                match spawn_branches_for_lineage(
                    lineage,
                    &visit_bucket_index,
                    visit.epoch,
                    r_obs,
                    v_obs,
                    params,
                    spatial_binner,
                    &next_branch_id,
                    current_step,
                ) {
                    Some((spawned, consumed)) => LineageOutcome::Spawned(
                        spawned,
                        GateRecord {
                            lineage_id: lineage.lineage_id,
                            nights_since_seed: current_step
                                .saturating_sub(lineage.last_real_update_step),
                            gated_obs_ids: consumed,
                        },
                    ),
                    // Every hypothesis in the bank failed to propagate this
                    // visit (rare, but not impossible — see
                    // `outfit_propagate_universal_failures.md`). Treat it
                    // like "no candidate nearby": carry the lineage over
                    // unchanged rather than losing its whole track history.
                    None => LineageOutcome::Unchanged(lineage.clone()),
                }
            })
            .collect();

        let mut visit_branches = Vec::new();
        for outcome in outcomes {
            match outcome {
                LineageOutcome::Unchanged(branch) => visit_branches.push(branch),
                LineageOutcome::Spawned(spawned, record) => {
                    visit_branches.extend(spawned);
                    consumed_observation_ids.extend(record.gated_obs_ids.iter().copied());
                    gate_records.push(record);
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
    let n_branches_before_n_scan = branches.len();
    let branches = apply_n_scan_pruning(branches, params.n_scan, current_step);
    let (branches, archived) = purge_stale_lineages(
        branches,
        params.max_lineage_lifetime_nights,
        params.stale_llr_floor,
        params.archive_min_real_updates,
        current_step,
    );
    OrchestrateEvent::NightPruningSummary {
        n_branches_before_n_scan,
        n_branches_after: branches.len(),
    }
    .emit();

    // Deliberately live branches only, not the archived arcs: this set exists
    // to decide which of *this night's* observations go back to the discovery
    // pool, and a lineage purged for staleness consumed none of them (that is
    // what made it stale). Archived observations were claimed on earlier
    // nights, which are long past recycling.
    let surviving_ids: HashSet<ObsId> = branches
        .iter()
        .flat_map(|b| b.track_ids().iter().copied())
        .collect();
    let consumed_then_pruned_ids = consumed_observation_ids
        .difference(&surviving_ids)
        .copied()
        .collect();

    NightAdvanceOutcome {
        branches,
        gate_records,
        consumed_observation_ids,
        consumed_then_pruned_ids,
        archived,
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
/// this visit"; `true` still requires `spawn_branches_for_lineage`'s real
/// (Mahalanobis-gated) check — this is a coarse, deliberately generous
/// filter, not a replacement for it.
pub fn lineage_might_be_in_visit(
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
    Spawned(Vec<Branch<'state_lf, 'bank_config>>, GateRecord),
}

/// Propagate one lineage's bank once (to this visit's epoch), search its
/// candidates, and spawn one observation branch per match plus the
/// mandatory null branch.
///
/// # Returns
/// The branches spawned for this lineage, plus the ids of every candidate
/// observation consumed (the caller merges these into the night's
/// `consumed_observation_ids`). `None` if the bank fails to propagate at all
/// this visit (logged at `debug`) — the caller carries the lineage over
/// unchanged instead, the same as when no candidate is nearby.
#[allow(clippy::too_many_arguments)]
pub fn spawn_branches_for_lineage<'state_lf, 'bank_config>(
    lineage: &Branch<'state_lf, 'bank_config>,
    visit_bucket_index: &BucketIndex<&Observation>,
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
    params: &NightAdvanceParams,
    spatial_binner: &HealpixBinner,
    next_branch_id: &AtomicU64,
    current_step: usize,
) -> Option<(Vec<Branch<'state_lf, 'bank_config>>, Vec<ObsId>)> {
    let predicted_bank = lineage.bank.predict_to(epoch, r_obs, v_obs);

    let Ok(search_region) = predicted_bank.search_region(
        params.obs_noise.into(),
        params.top_k,
        params.radius_strategy,
    ) else {
        OrchestrateEvent::LineagePropagationFailed {
            lineage_id: lineage.lineage_id,
        }
        .emit();
        return None;
    };

    // If the radius came back pinned at `radius_strategy`'s clamp, the single
    // MAP-centered cone is a truncated view of a mixture that is genuinely
    // wider — routinely the case at steps 1–2, where the bank still holds
    // hundreds of (ρ, ρ̇) grid nodes spread over degrees and the MAP node
    // carries ~1 % of the weight. Tile the footprint with cones and search
    // their union instead. Otherwise (the common case) this is one float
    // comparison and we take the single-cone fast path unchanged.
    let candidates = if let Some(cover) = cover_if_clamped(
        &search_region,
        params.radius_strategy,
        params.cone_half_rad(search_region.components.len()),
        params.max_search_cones,
    ) {
        find_candidates_for_bank_multi_region(
            &cover,
            &search_region,
            lineage.bank.track_ids().to_vec(),
            visit_bucket_index,
            spatial_binner,
            lineage.bank.config.gate_chi2,
            params.likelihood_threshold,
        )
    } else {
        find_candidates_for_bank(
            &search_region,
            lineage.bank.track_ids().to_vec(),
            visit_bucket_index,
            spatial_binner,
            lineage.bank.config.gate_chi2,
            params.likelihood_threshold,
        )
    };

    let predicted_magnitude = predicted_apparent_magnitude_for_bank(&predicted_bank);

    // Single per-visit clutter estimate at the region center, used only to
    // rank candidates in the top-K stage below — cheaper than, and
    // deliberately distinct from, the per-candidate density computed inside
    // the loop for each surviving candidate's own branch LLR.
    let region_clutter_density = local_clutter_density(
        visit_bucket_index,
        spatial_binner,
        &EquCoord::new(search_region.center_ra, 0.0, search_region.center_dec, 0.0),
    );
    let filtered_matches = apply_candidate_filters(
        &search_region,
        &predicted_bank,
        predicted_magnitude,
        region_clutter_density,
        candidates.matches,
        params,
    );

    let mut branches = Vec::with_capacity(filtered_matches.len() + 1);
    let mut consumed_observation_ids = Vec::with_capacity(filtered_matches.len());
    for candidate in &filtered_matches {
        consumed_observation_ids.push(*candidate.observation.id());

        let clutter_density = local_clutter_density(
            visit_bucket_index,
            spatial_binner,
            candidate.observation.equ_coord(),
        );
        let llr_delta = observation_llr_delta(candidate.likelihood, clutter_density)
            + photometric_llr_delta(
                predicted_magnitude,
                candidate.observation.photometry().magnitude,
                params.photometric_sigma_mag,
            );

        if let Some(branch) = Branch::from_observation(
            &predicted_bank,
            lineage,
            candidate.observation,
            llr_delta,
            next_branch_id.fetch_add(1, Ordering::Relaxed),
            current_step,
        ) {
            branches.push(branch);
        }
    }

    let p_detection = null_branch_detection_probability(
        predicted_magnitude,
        params.limiting_magnitude,
        params.completeness_width_mag,
    );
    branches.push(Branch::from_null(
        &predicted_bank,
        lineage,
        null_branch_llr_delta(p_detection),
        next_branch_id.fetch_add(1, Ordering::Relaxed),
    ));

    Some((branches, consumed_observation_ids))
}

/// Predicted apparent magnitude of a bank's MAP hypothesis, from its running
/// absolute-magnitude (`H`) estimate and that hypothesis's predicted
/// geometry — `None` before the bank's first successful
/// [`KFBank::branch_with`] call, when there is no magnitude history yet to
/// predict from.
///
/// Shared by [`null_branch_detection_probability`] (predicted magnitude vs.
/// survey depth) and the per-candidate photometric LLR term in
/// [`spawn_branches_for_lineage`] (predicted magnitude vs. a candidate's
/// observed magnitude) — both need the same geometry extraction, just fed
/// into different downstream comparisons.
pub fn predicted_apparent_magnitude_for_bank<'state_lf, 'bank_config>(
    bank: &KFBank<'state_lf, 'bank_config>,
) -> Option<f64> {
    let (Some(absolute_magnitude_estimate), Some(best)) =
        (bank.absolute_magnitude_estimate(), bank.best())
    else {
        return None;
    };

    let r_helio_au = best.kf.to_cartesian().pos.norm();
    let delta_topocentric_au = best.kf.state[4];
    Some(predicted_apparent_magnitude(
        absolute_magnitude_estimate,
        r_helio_au,
        delta_topocentric_au,
    ))
}

/// Detection probability for the null branch, from the bank's predicted
/// apparent magnitude (see [`predicted_apparent_magnitude_for_bank`]).
///
/// Falls back to `0.5` (neutral: neither favors nor penalizes the null
/// branch) when the bank has no magnitude history yet (`predicted_magnitude
/// == None`).
pub fn null_branch_detection_probability(
    predicted_magnitude: Option<f64>,
    limiting_magnitude: f64,
    completeness_width_mag: f64,
) -> f64 {
    const NO_HISTORY_FALLBACK_P_DETECTION: f64 = 0.5;

    let Some(predicted_magnitude) = predicted_magnitude else {
        return NO_HISTORY_FALLBACK_P_DETECTION;
    };

    detection_probability(
        predicted_magnitude,
        limiting_magnitude,
        completeness_width_mag,
    )
}

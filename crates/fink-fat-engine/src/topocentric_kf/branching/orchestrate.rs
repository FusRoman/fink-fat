//! Per-visit orchestration building blocks: propagate, search, branch, score.
//!
//! Provides the per-lineage steps (coarse pre-filter, propagation, candidate
//! search, branching, LLR scoring) that
//! [`NightSession`](super::NightSession) drives visit by visit and wraps with
//! the per-night cap / N-scan / purge — steps 1–6 of
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

use std::collections::HashMap;
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
        bucket::{BucketIndex, BucketKey},
        clutter_density::local_clutter_density,
        healpix_binner::HealpixBinner,
        spatial_binner::SpatialBinner,
    },
    topocentric_kf::{
        branching::{
            Branch,
            candidate_filters::apply_candidate_filters,
            candidate_search::{
                SINGLE_TIME_BIN, find_candidates_for_bank, find_candidates_for_bank_multi_region,
            },
            detection_probability::{
                detection_probability, phase_correction, predicted_apparent_magnitude,
            },
            llr_score::{null_branch_llr_delta, observation_llr_delta, photometric_llr_delta},
            visit::Visit,
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
    /// Live hypotheses in the lineage's bank when the gate ran.
    ///
    /// Carried alongside the gated ids so contamination can be attributed to a
    /// *cause* rather than to age: a bank holding dozens of `(ρ, ρ̇)`
    /// hypotheses projects a search region that is the union of all of them,
    /// and a wide region admits other objects. Correlating the wrong-fraction
    /// with this — rather than with `nights_since_seed` — is what separates
    /// "old lineages contaminate" from "wide lineages contaminate, and here is
    /// why they are wide".
    pub n_hypotheses: usize,
    /// Observations the bank had consumed — the argument the hypothesis cap
    /// schedule decays against, so the count above can be judged against the
    /// cap that should have applied.
    pub n_steps: usize,
}

/// A branch's angular position/rate, freshly propagated to this night's
/// reference epoch, used as the origin of the coarse pre-filter's linear
/// extrapolation.
///
/// # Why this exists
///
/// [`lineage_might_be_in_visit`] extrapolates in a straight line, while the
/// real dynamics is a two-body solve: its error grows as `½·α̈·dt²` against
/// an acceptance radius that does **not** grow with `dt`. A branch that
/// fails the pre-filter is carried over without propagation, so its bank
/// epoch does not advance and `dt` keeps growing on every consecutive miss —
/// a positive feedback loop.
///
/// Measured on a full ZTF-cadence run, the observations lost to the
/// pre-filter sat a median of **45 arcsec** from where a correct two-body
/// propagation put them (0.034 of the search-region radius) — comfortably
/// inside any sane radius — after a median of **22.8 days** without a real
/// update. The prediction was not the problem; extrapolating it linearly
/// over three weeks was. Widening the radius is the wrong lever: it cannot
/// catch a `dt²` error without becoming so large that every lineage passes,
/// which is what makes the pre-filter's cost explode.
///
/// So instead of widening the radius, keep the origin fresh: propagate the
/// MAP hypothesis **once per night per branch** and extrapolate from there.
/// Within a night `dt < ~0.35 day`, where the linear model is excellent.
///
/// Cost: one Kepler solve per branch per night, against
/// [`KFBank::predict_to`]'s `n_hypotheses` solves (≈87 on that same run) per
/// branch per *passing visit*. The anchor pays for itself as soon as it
/// avoids roughly one false positive in a hundred.
#[derive(Debug, Clone, Copy)]
pub struct PrefilterAnchor {
    /// Epoch this anchor was propagated to (MJD TT).
    pub epoch: f64,
    /// Right ascension at `epoch` (rad).
    pub ra: f64,
    /// Declination at `epoch` (rad).
    pub dec: f64,
    /// dRA/dt (rad/day, **not** `cos(dec)`-reduced — see `KFState`'s
    /// state-vector doc).
    pub ra_dot: f64,
    /// dDec/dt (rad/day).
    pub dec_dot: f64,
}

impl PrefilterAnchor {
    /// Age (days) below which a branch's stored angular state is considered
    /// a good enough origin, so no anchor is built.
    ///
    /// A night spans ~8h, so anything updated during the previous night is
    /// already inside this window. The linear model's error over a day is
    /// orders of magnitude below the ~13.7 arcmin HEALPix cell the pre-filter
    /// resolves at; it only becomes a problem over the multi-week gaps
    /// documented on this type.
    pub const FRESH_ENOUGH_DAYS: f64 = 1.0;

    /// Propagate `branch`'s MAP hypothesis to `epoch` — one Kepler solve.
    ///
    /// `None` if the branch has no live hypothesis or that hypothesis fails
    /// to propagate; callers then fall back to the branch's stored state.
    ///
    /// Also `None` — **without** paying the Kepler solve — when the branch's
    /// bank is already within [`Self::FRESH_ENOUGH_DAYS`] of `epoch`: over
    /// such a short span the stored angular rates extrapolate the position to
    /// far better than one HEALPix cell, so re-anchoring would buy nothing.
    /// In production most branches were updated the previous night and take
    /// this path, which is what keeps the anchor essentially free; the ones
    /// that pay for it are exactly the stale branches it exists to rescue.
    ///
    /// Shared by the engine and by `fink-fat-eval`'s `mot_analysis`, which
    /// must reproduce the production pre-filter exactly for its
    /// `no_prefilt%` column to keep measuring the engine.
    pub fn for_branch(
        branch: &Branch<'_, '_>,
        epoch: f64,
        r_obs: Vector3<f64>,
        v_obs: Vector3<f64>,
    ) -> Option<Self> {
        let best = branch.bank.best()?;
        if (epoch - best.kf.epoch).abs() <= Self::FRESH_ENOUGH_DAYS {
            return None;
        }
        let kf = best.kf.predict(epoch, r_obs, v_obs).ok()?;
        Some(Self {
            epoch: kf.epoch,
            ra: kf.state[0],
            dec: kf.state[1],
            ra_dot: kf.state[2],
            dec_dot: kf.state[3],
        })
    }
}

/// Propagate every branch's MAP hypothesis to `epoch`, keyed by `branch_id`,
/// for [`lineage_might_be_in_visit`] to extrapolate from — see
/// [`PrefilterAnchor`].
///
/// Only the incoming (previous-night) branches need this: anything spawned
/// during the night already carries a bank propagated to the visit epoch it
/// was born at, so its own state is fresh by construction.
///
/// A branch whose MAP hypothesis fails to propagate is simply absent from
/// the map; the pre-filter then falls back to that branch's stored state,
/// i.e. exactly the previous behavior.
pub(super) fn compute_prefilter_anchors(
    lineages: &[Branch<'_, '_>],
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
) -> HashMap<u64, PrefilterAnchor> {
    // Collected into a `Vec` first so the map can be built at exactly the
    // right capacity: rayon's `collect()` into a `HashMap` cannot be given
    // one, and rehashing a map this hot is pure waste. Only the stale
    // branches land here (see [`PrefilterAnchor::for_branch`]), so this is
    // typically far smaller than `lineages`.
    let anchors: Vec<(u64, PrefilterAnchor)> = lineages
        .par_iter()
        .filter_map(|branch| {
            PrefilterAnchor::for_branch(branch, epoch, r_obs, v_obs)
                .map(|anchor| (branch.branch_id, anchor))
        })
        .collect();

    let mut by_branch = HashMap::with_capacity(anchors.len());
    by_branch.extend(anchors);
    by_branch
}

/// Resolve the observer's heliocentric state at `epoch`.
///
/// `representative_obs` is only used to identify *which* observer
/// (site/instrument) took this visit — any observation in the visit works,
/// they share one exposure.
pub(super) fn resolve_observer_state(
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
/// No two-body solve — just arithmetic on an angular position/rate, plus a
/// HEALPix neighbor lookup against `visit_bucket_index` (already built for
/// this visit, reused here, not rebuilt).
///
/// `false` means "definitely nothing nearby, skip the real propagation
/// this visit"; `true` still requires `spawn_branches_for_lineage`'s real
/// (Mahalanobis-gated) check — this is a coarse filter, not a replacement
/// for it.
///
/// # Where the extrapolation starts from
///
/// `anchor` is this branch's MAP hypothesis propagated to the night's
/// reference epoch (see [`PrefilterAnchor`] for why that matters — in short,
/// the linear model's error grows as `dt²` and used to be extrapolated over
/// weeks). When it is `None` — a branch spawned during this night, whose own
/// bank is already at a visit epoch of this night, or one whose MAP
/// hypothesis failed to propagate — the branch's own stored state is used
/// instead, which is the historical behavior.
///
/// # The radius is quantized to whole cells — and must stay that way
///
/// [`HealpixBinner::neighbors`](crate::spacetime_bucket::healpix_binner::HealpixBinner)
/// returns whole cells and **ignores the radius entirely** when it is smaller
/// than one cell (at `healpix_depth: 8` that is ~13.7 arcmin): it hands back a
/// fixed 3×3 block, an effective radius of ~20 arcmin, whatever
/// `quick_reject_radius_rad` says. Configuring anything below the cell radius
/// therefore has no effect here.
///
/// **Do not "fix" this by measuring the real angular separation to the alerts
/// in those cells.** That was tried and reverted: it turns an `O(1)`
/// short-circuiting hash lookup into a scan of every alert in nine cells, each
/// costing a Vincenty
/// [`angular_separation`](photom::coordinates::equatorial::EquCoord::angular_separation)
/// (three `sin_cos`, a `hypot`, an `atan2`). Worse, with a radius tighter than
/// the cell block the test almost always fails, so `any` never short-circuits
/// and the full scan runs every time — in the hottest loop of the engine (once
/// per branch per visit; once per *trajectory* per night per visit in
/// `mot_analysis`). Measured: `mot_analysis` went from ~6 minutes to over 36.
///
/// The radius was never the binding constraint anyway — the observations this
/// filter used to lose sat a median of 45 arcsec from a correct propagation.
/// Staleness of the extrapolation origin was the real defect, and that is what
/// [`PrefilterAnchor`] fixes. If sub-cell selectivity is ever genuinely
/// wanted, a flat-sky squared distance (`dx = Δra·cos δ`, `dy = Δδ`) is ~4
/// flops instead of eight transcendentals — but it is still `O(alerts)` where
/// this is `O(1)`, so it needs a measurement first.
pub fn lineage_might_be_in_visit(
    lineage: &Branch,
    anchor: Option<&PrefilterAnchor>,
    visit: &Visit,
    visit_bucket_index: &BucketIndex<&Observation>,
    params: &NightAdvanceParams,
    spatial_binner: &HealpixBinner,
) -> bool {
    // Prefer the freshly propagated anchor; fall back to the branch's own
    // stored angular state (`state[0..=3]`, shared across the bank's
    // hypotheses — only ρ/ρ̇ differ between them).
    let (epoch, ra, dec, ra_dot, dec_dot) = match anchor {
        Some(a) => (a.epoch, a.ra, a.dec, a.ra_dot, a.dec_dot),
        None => {
            let Some(best) = lineage.bank.best() else {
                return false;
            };
            (
                best.kf.epoch,
                best.kf.state[0],
                best.kf.state[1],
                best.kf.state[2],
                best.kf.state[3],
            )
        }
    };

    let dt = visit.epoch - epoch;
    let predicted_ra = wrap_angle(ra + ra_dot * dt);
    let predicted_dec = dec + dec_dot * dt;
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

    let helio = best.kf.to_cartesian().pos;
    let delta_topocentric_au = best.kf.state[4];
    // Put back the phase term `update_absolute_magnitude_estimate` removed, at
    // *this* geometry: the two transformations must stay inverses of each other
    // or the running H and the magnitudes predicted from it drift apart.
    Some(predicted_apparent_magnitude(
        absolute_magnitude_estimate,
        helio.norm(),
        delta_topocentric_au,
        phase_correction(&helio, &best.kf.r_obs, bank.config.slope_parameter_g),
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

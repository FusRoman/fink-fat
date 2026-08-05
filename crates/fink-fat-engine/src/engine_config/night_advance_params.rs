//! # Night-advance tuning configuration (`NightAdvanceParams`)
//!
//! This module defines [`NightAdvanceParams`], the tuning knobs consumed by
//! [`crate::topocentric_kf::branching::orchestrate::advance_bank_collection_one_night`]
//! every time a night's worth of new visits is folded into a collection of
//! tracklet hypothesis banks.
//!
//! The fields are grouped, and documented below, in the order the pipeline
//! consumes them for a single night:
//! 1. Group raw alerts into [`crate::topocentric_kf::branching::visit::Visit`]s
//!    (`visit_epoch_tolerance_days`).
//! 2. Cheaply pre-filter which lineages are even worth propagating this
//!    visit (`quick_reject_radius_rad`).
//! 3. Build a search region for surviving lineages and look up candidate
//!    observations in it (`obs_noise`, `top_k`, `radius_strategy`,
//!    `likelihood_threshold`), then refine the surviving candidates with an
//!    optional post-gate filter cascade (`candidate_photometric_max_delta_mag`,
//!    `candidate_cross_track_k_sigma`, `candidate_cross_track_floor_arcsec`,
//!    `candidate_rel_likelihood_alpha`, `max_candidates_per_visit`) before any
//!    branch is spawned.
//! 4. Prune the resulting branch tree, per-lineage and across the whole
//!    night (`branch_cap`, `n_scan`, `max_lineage_lifetime_nights`), keeping
//!    the reconstruction of every credible lineage that leaves the live set
//!    (`archive_min_real_updates`).
//! 5. Score the null-detection hypothesis for lineages predicted bright
//!    enough to have been seen (`limiting_magnitude`,
//!    `completeness_width_mag`), and score each real candidate's
//!    photometric plausibility against the bank's magnitude history
//!    (`photometric_sigma_mag`).
//!
//! This configuration is `serde`-deserializable (YAML) and uses the
//! project-level unit parsers from [`crate::engine_config::units`] for its
//! time/angle/angular-variance fields.
//!
//! The numeric-range expectations documented on each field (e.g.
//! `completeness_width_mag > 0`, `branch_cap ≥ 1`) are enforced by this
//! type's [`Validate`] implementation.

use serde::{Deserialize, Serialize};

use crate::engine_config::Validate;
use crate::engine_config::error::FieldError;
use crate::engine_config::units::{
    de_angle_arcsec, de_angle_rad, de_angle_var_rad2_pair, de_time_days,
};
use crate::engine_config::validate_helpers::{
    check_finite, check_finite_nonneg, check_finite_positive, check_min_usize,
};
use crate::topocentric_kf::kalman_bank::ellipse_region_finder::{
    radius_strategy::{MixOrMax, RadiusStrategy},
    top_k::TopK,
};

/// Tuning parameters shared by every lineage advanced in one call to
/// [`crate::topocentric_kf::branching::orchestrate::advance_bank_collection_one_night`] — everything except the branches
/// being advanced and the current step index, which are the function's
/// primary inputs rather than tuning knobs.
///
/// Grouped in the order they're consumed by the pipeline: visit grouping →
/// the cheap pre-filter → candidate search → cross-bank pruning →
/// null-branch detection probability.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NightAdvanceParams {
    /// Maximum epoch spread for two observations to be folded into the same
    /// [`crate::topocentric_kf::branching::visit::Visit`] (see
    /// [`crate::topocentric_kf::branching::visit::group_observations_into_visits`]).
    ///
    /// Units
    /// -----
    /// - Canonical: **days**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in days): `1.1574e-5`
    /// - string with units: `"1 sec"`, `"1000 ms"` is not supported (no
    ///   sub-second unit); use fractional seconds instead, e.g. `"0.5 sec"`
    ///
    /// Context
    /// -------
    /// Should be small — a fraction of the exposure/readout time, e.g.
    /// a few seconds expressed as a fraction of a day (`1.0 / 86_400.0` ≈
    /// 1 second) — since its only job is to absorb per-alert timestamp
    /// jitter *within* one exposure, not to merge distinct visits. Too
    /// large would incorrectly treat two different exposures (and thus two
    /// different observer/geometry states) as one epoch.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_time_days`].
    #[serde(deserialize_with = "de_time_days")]
    pub visit_epoch_tolerance_days: f64,

    /// Angular radius for the cheap linear-extrapolation pre-filter (see the
    /// module-level "Performance" note).
    ///
    /// Units
    /// -----
    /// - Canonical: **radians**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in rad): `1e-3`
    /// - string with units: `"3.4 arcmin"`, `"0.057 deg"`
    ///
    /// Context
    /// -------
    /// For every lineage, at every visit, this is the
    /// search radius used to test — no Kepler solve — whether *any* alert in
    /// the visit falls near the lineage's extrapolated sky position; a
    /// lineage that fails this test never pays for the real (expensive)
    /// propagation this visit. Too small silently drops real associations (a
    /// lineage never gets the chance to match); too large costs a wasted
    /// full propagation (`n_hypotheses` Kepler solves) per lineage per visit,
    /// which is the dominant cost of a night.
    ///
    /// # This value is quantized to whole HEALPix cells
    ///
    /// The candidate cells come from
    /// [`HealpixBinner::neighbors`](crate::spacetime_bucket::healpix_binner::HealpixBinner),
    /// which returns **whole cells** and, below one cell radius, ignores the
    /// requested radius entirely: it hands back a fixed 3×3 block. At
    /// `healpix_depth: 8` a cell radius is ≈ 13.7 arcmin, so **anything set
    /// below that has no effect at all** — `1 arcmin`, `3.5 arcmin` and
    /// `13 arcmin` select exactly the same cells, an effective radius of
    /// ~20 arcmin. Tightening this value is not a way to save time.
    ///
    /// Crossing *above* the cell radius switches to a cone coverage whose
    /// cell count grows as `(radius / cell)²`, and — far more costly — many
    /// more lineages pass and pay for a full propagation. Widening is
    /// therefore expensive: measured on a full dataset, 3.5 → 30 arcmin took
    /// `mot_analysis` from ~6 minutes to over 30 for no measurable recall
    /// gain. To change the granularity, change `healpix_depth`, not this.
    ///
    /// # This is no longer the knob that fixes missed associations
    ///
    /// Historically the extrapolation started from the branch's own bank
    /// epoch, which could be weeks stale, so its `dt²` error dwarfed any
    /// sane radius and widening this value was the only (bad) remedy. The
    /// origin is now re-anchored once per night
    /// ([`PrefilterAnchor`](crate::topocentric_kf::branching::orchestrate::PrefilterAnchor)),
    /// keeping `dt` under one night. Set this to cover the residual
    /// intra-night extrapolation error plus astrometric scatter, not
    /// multi-week drift.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_angle_rad`].
    #[serde(deserialize_with = "de_angle_rad")]
    pub quick_reject_radius_rad: f64,

    /// **A priori** diagonal astrometric noise `[σ_RA², σ_Dec²]`, added to
    /// each hypothesis's predicted sky covariance solely to size the search
    /// region in [`crate::topocentric_kf::kalman_bank::KFBank::predict_search_region`].
    ///
    /// Units
    /// -----
    /// - Canonical: **rad²** for each component.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already rad² variances): `[2.35e-13, 2.35e-13]`
    /// - string with an angle unit, interpreted as a **1-sigma** value and
    ///   squared: `["0.1 arcsec", "0.1 arcsec"]`
    ///
    /// See [`crate::engine_config::units`]'s "Angular variance" section for
    /// the sigma-in/variance-out convention.
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
    /// [`crate::topocentric_kf::kalman_bank::KFBank::branch_with`]'s mixture-likelihood scoring and the Kalman
    /// update itself — ignores this field entirely and instead uses the
    /// candidate observation's *own* `ra_error`/`dec_error` (via
    /// `Observation::equ_coord`), which is always the more accurate value
    /// once it's available.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_angle_var_rad2_pair`].
    #[serde(deserialize_with = "de_angle_var_rad2_pair")]
    pub obs_noise: [f64; 2],

    /// Which of a bank's live hypotheses contribute to its predicted
    /// search region this visit — see [`TopK`] for the available policies
    /// (`All`, `Map`, `Best(k)`, `WeightThreshold`). Passed straight
    /// through to [`crate::topocentric_kf::kalman_bank::KFBank::predict_search_region`].
    ///
    /// No dedicated `units.rs` parser applies here: `TopK` is a plain enum
    /// defined outside `engine_config` (in `topocentric_kf`), deserialized
    /// via its own `derive(Deserialize)`.
    pub top_k: TopK,

    /// How the search region's bounding radius is computed from the
    /// selected hypotheses' covariances — see [`RadiusStrategy`]
    /// (`MixtureCovariance` vs. `MaxEllipse`). Passed straight through to
    /// [`crate::topocentric_kf::kalman_bank::KFBank::predict_search_region`].
    ///
    /// No dedicated `units.rs` parser applies here either, for the same
    /// reason as `top_k`; note that `RadiusStrategy::Clamped`'s own
    /// `max_arcsec` field is expressed directly in arcseconds by that type,
    /// independent of this module's angle unit conventions.
    pub radius_strategy: RadiusStrategy,

    /// Maximum number of cones
    /// [`sky_cover_regions`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::sky_cover_regions)
    /// may emit when tiling a lineage's predicted sky footprint for the
    /// coarse candidate search.
    ///
    /// The cover only runs when the single region's radius came back pinned
    /// at `radius_strategy`'s clamp — see
    /// [`SearchRegion::radius_pinned_at_clamp`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion::radius_pinned_at_clamp).
    ///
    /// This cap bounds *both* the number of HEALPix cone queries the coarse
    /// search issues and the greedy cover's own scan (which stops as soon as
    /// it has emitted this many cones). Cones are seeded in descending weight
    /// order, so hitting the cap drops the least likely hypotheses first.
    ///
    /// Dimensionless count. **`0` disables the cover entirely** and is the
    /// current default.
    ///
    /// # Why the default is off
    ///
    /// This cover was built to widen the step-1/2 coarse search after
    /// `mot_analysis` showed 26.3% / 11.3% `not_matched%`. Measured on the
    /// full dataset it did essentially nothing for recall (`found%` 93.6% →
    /// 93.7%) while inflating candidates 6.5M → 27.0M and the bad-candidate
    /// rate 25.1% → 82.1%.
    ///
    /// The reason is that the failure had a different cause entirely:
    /// `TopK::WeightThreshold` was thresholding unnormalized bank weights and
    /// collapsing the mixture to a single hypothesis (see its doc). The cover
    /// was therefore tiling *one* hypothesis — nothing to cover. With that
    /// fixed, re-measure before enabling this: it may have no remaining job.
    ///
    /// Replaces the earlier `n_step1_clusters` (ρ-chunked, step-1 only),
    /// which saturated well short of full recall because ρ̇ also drives
    /// along-track spread, leaving ρ-chunks angularly non-compact.
    pub max_search_cones: usize,

    /// Half-angle of the cones laid down by
    /// [`sky_cover_regions`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::sky_cover_regions),
    /// in **arcseconds**, used only when [`Self::radius_strategy`] is an
    /// unclamped variant and therefore reports no natural tiling scale of its
    /// own (see [`RadiusStrategy::clamp_rad`]). With the clamped strategies
    /// used in practice this value is ignored.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_angle_arcsec`], so `1800.0`, `"30 arcmin"` and
    /// `"0.5 deg"` are all accepted.
    #[serde(deserialize_with = "de_angle_arcsec")]
    pub cone_half_arcsec: f64,

    /// Minimum mixture predictive likelihood (unitless, a Gaussian density
    /// value — see
    /// [`SearchRegion::mixture_likelihood`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion::mixture_likelihood))
    /// a candidate observation must reach, *after* passing the coarse
    /// per-component Mahalanobis gate, to be kept by
    /// [`crate::topocentric_kf::branching::candidate_search::find_candidates_for_bank`]. A second-stage cut on top of the gate:
    /// the gate says "geometrically plausible," this says "and not
    /// negligibly unlikely." Must be `≥ 0.0`; `0.0` disables this stage
    /// (keep everything the gate accepts).
    pub likelihood_threshold: f64,

    /// Max `|observed_magnitude - predicted_magnitude|` a candidate may have
    /// to survive the post-gate filter cascade — see
    /// [`crate::topocentric_kf::branching::candidate_filters::apply_candidate_filters`].
    /// A no-op whenever the lineage has no photometric history yet
    /// (`predicted_magnitude` is `None`), same convention as
    /// [`Self::photometric_sigma_mag`]'s LLR term.
    ///
    /// Units: magnitudes. `0.0` **disables this stage** and is the current
    /// default. Validated (eval, full ZTF-cadence dataset): `2.0` removes
    /// ~9% of remaining bad candidates for 0.56% recall loss, negligible
    /// cost to NEO recall.
    pub candidate_photometric_max_delta_mag: f64,

    /// Cross-track gate: rejects a candidate whose offset perpendicular to
    /// the bank's predicted apparent-motion direction exceeds
    /// `candidate_cross_track_k_sigma · σ_cross` (floored at
    /// [`Self::candidate_cross_track_floor_arcsec`]), where `σ_cross` is the
    /// predicted mixture's cross-track spread — see
    /// [`crate::topocentric_kf::branching::candidate_filters::apply_candidate_filters`].
    /// Motivated by prediction error being along-track dominated (rate/
    /// timing uncertainty) while astrometric clutter lands isotropically in
    /// the search region.
    ///
    /// Units: dimensionless (multiple of σ_cross). `0.0` **disables this
    /// stage** and is the current default. Validated: `5.0` removes ~5% of
    /// remaining bad candidates for a 0.001% recall loss, zero NEO cost.
    pub candidate_cross_track_k_sigma: f64,

    /// Floor (arcsec) applied to σ_cross before scaling by
    /// [`Self::candidate_cross_track_k_sigma`], avoiding an overly tight
    /// gate when the mixture's cross-track spread is numerically tiny (e.g.
    /// a well-converged bank with very few surviving hypotheses). Only used
    /// while the gate above is enabled.
    ///
    /// Units: arcseconds. Validated value: `2.0`.
    pub candidate_cross_track_floor_arcsec: f64,

    /// Relative-likelihood gate ("ambiguity check" — Cao et al., PKF):
    /// within one visit's candidate list, keep only candidates whose
    /// [`CandidateMatch::likelihood`](crate::topocentric_kf::branching::candidate_search::CandidateMatch::likelihood)
    /// is at least this fraction of the visit's best candidate — see
    /// [`crate::topocentric_kf::branching::candidate_filters::apply_candidate_filters`].
    ///
    /// Units: dimensionless fraction in `[0, 1]`. `0.0` **disables this
    /// stage** and is the current default. Validated: `0.01` removes ~41%
    /// of remaining bad candidates for a 0.02% recall loss, zero NEO cost —
    /// the single best-performing gate of the cascade.
    pub candidate_rel_likelihood_alpha: f64,

    /// Hard cap on candidates kept per visit, ranked by the same combined
    /// astrometric+photometric log-likelihood-ratio used downstream to rank
    /// branches (see
    /// [`crate::topocentric_kf::branching::llr_score::observation_llr_delta`]/
    /// [`photometric_llr_delta`](crate::topocentric_kf::branching::llr_score::photometric_llr_delta)),
    /// applied one stage earlier — before any branch exists — via
    /// [`crate::topocentric_kf::branching::candidate_filters::apply_candidate_filters`].
    /// Unlike the gates above, this bounds a visit's surviving candidate
    /// count (and therefore branching cost) *by construction*, regardless
    /// of how many candidates pass the other stages.
    ///
    /// Not to be confused with [`Self::top_k`] (which selects hypotheses
    /// *within* a bank's own mixture) — this field caps *candidates*, a
    /// different stage of the pipeline.
    ///
    /// Dimensionless count. `0` **disables this stage** and is the current
    /// default. Validated: `5` removes ~30% of remaining bad candidates for
    /// a 0.04% recall loss, zero NEO cost.
    pub max_candidates_per_visit: usize,

    /// Top-B cap: maximum number of branches kept **per lineage**, applied
    /// via [`crate::topocentric_kf::branching::pruning::cap_top_b_per_lineage`] after *every visit* — not just once
    /// per night, since branch counts multiply at every branching event
    /// (M candidates + 1 null branch) and would explode across a night's
    /// worth of visits otherwise.
    ///
    /// Dimensionless count, must be `≥ 1`. The design doc recommends
    /// `B ≈ 3–5`: wide enough to carry real ambiguity a visit or two,
    /// narrow enough to bound cost.
    pub branch_cap: usize,

    /// N-scan pruning window, in **nights** (not visits — see
    /// [`crate::topocentric_kf::branching::pruning::apply_n_scan_pruning`]), applied exactly once per call to
    /// [`crate::topocentric_kf::branching::orchestrate::advance_bank_collection_one_night`], after every visit that night
    /// has been folded in. For every branch-tree node this many nights old,
    /// only the single best-scoring descendant survives; siblings are
    /// discarded.
    ///
    /// Dimensionless count of nights, must be `≥ 1`. `1` is the design
    /// doc's recommendation (association ambiguity usually resolves by the
    /// very next night); `2` is mentioned as an occasional alternative for
    /// slower-resolving cases. This is a plain night count, not a
    /// `units.rs`-parsed time quantity (it indexes discrete nightly calls,
    /// not a continuous duration).
    pub n_scan: usize,

    /// Maximum age, in **nights**, since a lineage's last real (non-null)
    /// observation before the whole lineage is purged from
    /// `BranchCollection` — see
    /// [`crate::topocentric_kf::branching::pruning::purge_stale_lineages`],
    /// applied once per night, right after `n_scan` pruning.
    ///
    /// Without this, the number of live lineages is monotone increasing for
    /// the whole run (noise lineages and never-recovered fragments keep
    /// producing a "null" branch forever) — see
    /// `branch_lifetime_and_footprint.md` in the eval reports.
    ///
    /// # What purging does and does not cost
    ///
    /// A purged lineage stops being propagated, which is the entire point:
    /// no more widening error box, no more mis-associations, no more
    /// per-visit `KFBank` propagation. Its already-accumulated reconstruction
    /// is **not** necessarily lost — see
    /// [`Self::archive_min_real_updates`], which decides whether the arc
    /// leaves as an
    /// [`ArchivedTrajectory`](crate::topocentric_kf::branching::ArchivedTrajectory)
    /// or is discarded outright.
    ///
    /// (Earlier revisions of this doc claimed purging "loses no
    /// already-reconstructed trajectory" because a reappearing object gets
    /// re-discovered. That was wrong: re-discovery recovers *future*
    /// detections, never the arc already built. Measured on a 200-night run,
    /// enabling this pass without archiving moved 37 561 trajectories to
    /// "not reconstructed at all".)
    ///
    /// Dimensionless count of nights. `0` **disables this pass entirely**
    /// (no lineage is ever purged for staleness) — the default, and the
    /// historical behavior before this field existed. When enabling it,
    /// validate against `tests/reconstruction.rs`'s reference trajectories
    /// first: a legitimately faint/intermittent object with many
    /// consecutive non-detections could otherwise be purged before it's
    /// fully reconstructed — see `stale_llr_floor`, which guards against
    /// exactly that.
    pub max_lineage_lifetime_nights: usize,

    /// LLR floor gating `max_lineage_lifetime_nights`: a stale lineage is
    /// only purged if its *best* surviving branch's `cumulative_llr` is at
    /// or below this value — see
    /// [`crate::topocentric_kf::branching::pruning::purge_stale_lineages`].
    ///
    /// Protects a lineage that is merely quiet (out of survey footprint, or
    /// predicted too faint to expect a detection — see
    /// [`crate::topocentric_kf::branching::llr_score::null_branch_llr_delta`]/
    /// [`crate::topocentric_kf::branching::detection_probability::detection_probability`],
    /// whose LLR penalty is near-zero or not applied at all in exactly
    /// those cases) from being purged just because it hasn't produced a
    /// real detection in a while: only a lineage whose own LLR judges it
    /// *less plausible than the clutter background* is dropped, regardless
    /// of how long it's been stale.
    ///
    /// Unitless — same scale as `cumulative_llr`, a sum of log-likelihood-
    /// ratio terms against a clutter background (see
    /// [`crate::topocentric_kf::branching::llr_score`]). `0.0` is the
    /// natural default: "at best, no better than pure clutter." Has no
    /// effect while `max_lineage_lifetime_nights == 0` (pass disabled).
    pub stale_llr_floor: f64,

    /// Minimum number of **real** (non-null) observations a lineage purged by
    /// `max_lineage_lifetime_nights` must have consumed for its reconstruction
    /// to be retained as an
    /// [`ArchivedTrajectory`](crate::topocentric_kf::branching::ArchivedTrajectory)
    /// instead of discarded — see
    /// [`crate::topocentric_kf::branching::pruning::purge_stale_lineages`].
    ///
    /// Archiving decouples "stop propagating this lineage" (cheap, and the
    /// reason the purge exists) from "throw away what it reconstructed"
    /// (pure loss). This threshold decides which purged arcs are worth
    /// reporting.
    ///
    /// The test is on the real-update count rather than `cumulative_llr`
    /// because [`Branch::from_null`](crate::topocentric_kf::branching::Branch::from_null)
    /// leaves that counter untouched: a lineage still sitting at `1` was
    /// seeded from a pair and never confirmed by any later night — the
    /// two-point noise tracklet that should not reach the output. An arc
    /// corroborated across several nights is worth keeping whatever coasting
    /// penalty it accumulated on the way out.
    ///
    /// Dimensionless count. `0` archives **every** purged lineage (maximum
    /// recovery, and the value to measure first); raise it (2, 3, …) if the
    /// archive turns out to re-import contaminated arcs. Has no effect while
    /// `max_lineage_lifetime_nights == 0` (pass disabled).
    pub archive_min_real_updates: usize,

    /// Survey/field limiting magnitude for this night, used as the midpoint
    /// of the null branch's detection-probability curve — see
    /// [`crate::topocentric_kf::branching::detection_probability::detection_probability`].
    ///
    /// Units: magnitudes (no `units.rs` parser — a single, unambiguous
    /// photometric scale, unlike angles/time which have many common units).
    /// A lineage predicted brighter than this is very likely to have been
    /// detected (so a non-detection weighs heavily against the null
    /// branch); predicted fainter, the opposite.
    pub limiting_magnitude: f64,

    /// Completeness roll-off width of the survey's detection curve around
    /// `limiting_magnitude` — see [`crate::topocentric_kf::branching::detection_probability::detection_probability`]. Real surveys
    /// don't have a hard cutoff magnitude; detection probability decays
    /// smoothly over roughly this many magnitudes on either side of
    /// `limiting_magnitude`.
    ///
    /// Units: magnitudes. Must be strictly positive (not currently enforced
    /// by any `validate()` method on this struct — see the module-level
    /// note). Typical values ≈ 0.3–0.5 mag.
    pub completeness_width_mag: f64,

    /// Assumed 1-sigma spread (magnitudes) of a candidate observation's
    /// apparent-magnitude residual against a bank's photometric prediction
    /// — see
    /// [`crate::topocentric_kf::branching::llr_score::photometric_llr_delta`].
    /// An *additional* LLR term alongside the astrometric one, used to help
    /// discriminate two lineages whose predicted sky positions/rates are
    /// nearly indistinguishable but whose objects have different absolute
    /// magnitudes.
    ///
    /// Units: magnitudes. Must be strictly positive. Typical values ≈
    /// 0.3–0.5 mag — wide enough to absorb the H-without-phase-term
    /// simplification (see
    /// [`crate::topocentric_kf::branching::detection_probability::implied_absolute_magnitude`])
    /// plus ordinary photometric noise, narrow enough to still discriminate
    /// objects that differ by more than a few tenths of a magnitude.
    pub photometric_sigma_mag: f64,
}

impl NightAdvanceParams {
    /// Cone half-angle (radians) for
    /// [`sky_cover_regions`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::sky_cover_regions).
    ///
    /// Prefers the clamp [`Self::radius_strategy`] already imposes, so the
    /// cover tiles at exactly the granularity a single cone is allowed to
    /// reach — a larger cone would just be clamped away, a smaller one would
    /// multiply queries for nothing. Falls back to [`Self::cone_half_arcsec`]
    /// for the unclamped strategies, which advertise no scale of their own.
    ///
    /// `n_components` selects the clamp that actually applies — see
    /// [`RadiusStrategy::clamp_rad`].
    pub fn cone_half_rad(&self, n_components: usize) -> f64 {
        self.radius_strategy
            .clamp_rad(n_components)
            .unwrap_or((self.cone_half_arcsec / 3600.0_f64).to_radians())
    }
}

impl Default for NightAdvanceParams {
    fn default() -> Self {
        Self {
            visit_epoch_tolerance_days: 1.0 / 86_400.0,
            quick_reject_radius_rad: 1e-3,
            obs_noise: [2.35e-13, 2.35e-13],
            top_k: TopK::WeightThreshold(0.99),
            radius_strategy: RadiusStrategy::Clamped {
                inner: MixOrMax::MixtureCovariance,
                max_arcsec: 30. * 60., // 30 arcminutes
            },
            max_search_cones: 0,
            cone_half_arcsec: 30. * 60., // 30 arcminutes
            likelihood_threshold: 0.0,
            candidate_photometric_max_delta_mag: 0.0,
            candidate_cross_track_k_sigma: 0.0,
            candidate_cross_track_floor_arcsec: 2.0,
            candidate_rel_likelihood_alpha: 0.0,
            max_candidates_per_visit: 0,
            branch_cap: 4,
            n_scan: 1,
            max_lineage_lifetime_nights: 0,
            stale_llr_floor: 0.0,
            archive_min_real_updates: 0,
            limiting_magnitude: 21.0,
            completeness_width_mag: 0.4,
            photometric_sigma_mag: 0.35,
        }
    }
}

impl Validate for NightAdvanceParams {
    /// Validate internal consistency and numeric ranges, accumulating every
    /// failure found instead of stopping at the first one.
    ///
    /// `top_k` and `radius_strategy` are external enums (defined in
    /// `topocentric_kf`, outside `engine_config`) and are not validated here.
    fn validate(&self) -> Result<(), Vec<FieldError>> {
        let mut errors = Vec::new();

        if let Some(e) = check_finite_nonneg(
            "visit_epoch_tolerance_days",
            self.visit_epoch_tolerance_days,
            "set visit_epoch_tolerance_days to a small non-negative duration, e.g. 1.0/86_400.0 (~1 second)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "quick_reject_radius_rad",
            self.quick_reject_radius_rad,
            "set quick_reject_radius_rad to a strictly positive angle, e.g. \"3.4 arcmin\" or 1e-3 (rad)",
        ) {
            errors.push(e);
        }
        for (i, component) in self.obs_noise.iter().enumerate() {
            if let Some(e) = check_finite_nonneg(
                &format!("obs_noise[{i}]"),
                *component,
                "set obs_noise components to non-negative variances, e.g. \"0.1 arcsec\" (squared) or 2.35e-13 (rad^2)",
            ) {
                errors.push(e);
            }
        }
        if let Some(e) = check_finite_nonneg(
            "likelihood_threshold",
            self.likelihood_threshold,
            "set likelihood_threshold to a non-negative density, e.g. 0.0 to disable this stage",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_nonneg(
            "candidate_photometric_max_delta_mag",
            self.candidate_photometric_max_delta_mag,
            "set candidate_photometric_max_delta_mag to a non-negative magnitude, e.g. 0.0 to disable this stage or 2.0",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_nonneg(
            "candidate_cross_track_k_sigma",
            self.candidate_cross_track_k_sigma,
            "set candidate_cross_track_k_sigma to a non-negative multiple of sigma, e.g. 0.0 to disable this stage or 5.0",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_nonneg(
            "candidate_cross_track_floor_arcsec",
            self.candidate_cross_track_floor_arcsec,
            "set candidate_cross_track_floor_arcsec to a non-negative angle, e.g. 2.0",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_nonneg(
            "candidate_rel_likelihood_alpha",
            self.candidate_rel_likelihood_alpha,
            "set candidate_rel_likelihood_alpha to a non-negative fraction, e.g. 0.0 to disable this stage or 0.01",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_min_usize(
            "branch_cap",
            self.branch_cap,
            1,
            "set branch_cap to at least 1 branch per lineage, e.g. 4",
        ) {
            errors.push(e);
        }
        // No lower-bound check: 0 is meaningful here (disables the sky cover).
        if let Some(e) = check_finite_positive(
            "cone_half_arcsec",
            self.cone_half_arcsec,
            "set cone_half_arcsec to a positive angle, e.g. 1800.0 or \"30 arcmin\"",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_min_usize(
            "n_scan",
            self.n_scan,
            1,
            "set n_scan to at least 1 night, e.g. 1",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite(
            "stale_llr_floor",
            self.stale_llr_floor,
            "set stale_llr_floor to a finite cumulative_llr ceiling, e.g. 0.0",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite(
            "limiting_magnitude",
            self.limiting_magnitude,
            "set limiting_magnitude to a finite survey magnitude, e.g. 21.0",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "completeness_width_mag",
            self.completeness_width_mag,
            "set completeness_width_mag to a strictly positive magnitude width, e.g. 0.4",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "photometric_sigma_mag",
            self.photometric_sigma_mag,
            "set photometric_sigma_mag to a strictly positive magnitude width, e.g. 0.35",
        ) {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

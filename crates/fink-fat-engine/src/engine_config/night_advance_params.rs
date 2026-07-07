use serde::{Deserialize, Serialize};

use crate::topocentric_kf::kalman_bank::ellipse_region_finder::{
    radius_strategy::{MixOrMax, RadiusStrategy},
    top_k::TopK,
};

/// Tuning parameters shared by every lineage advanced in one call to
/// [`advance_bank_collection_one_night`] — everything except the branches
/// being advanced and the current step index, which are the function's
/// primary inputs rather than tuning knobs.
///
/// Grouped in the order they're consumed by the pipeline: visit grouping →
/// the cheap pre-filter → candidate search → cross-bank pruning →
/// null-branch detection probability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NightAdvanceParams {
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
    pub obs_noise: [f64; 2],

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
            likelihood_threshold: 0.0,
            branch_cap: 4,
            n_scan: 1,
            limiting_magnitude: 21.0,
            completeness_width_mag: 0.4,
        }
    }
}

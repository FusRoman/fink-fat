//! # Night-advance tuning configuration (`NightAdvanceParams`)
//!
//! This module defines [`NightAdvanceParams`], the tuning knobs consumed by
//! [`advance_bank_collection_one_night`] every time a night's worth of new
//! visits is folded into a collection of tracklet hypothesis banks.
//!
//! The fields are grouped, and documented below, in the order the pipeline
//! consumes them for a single night:
//! 1. Group raw alerts into [`Visit`]s (`visit_epoch_tolerance_days`).
//! 2. Cheaply pre-filter which lineages are even worth propagating this
//!    visit (`quick_reject_radius_rad`).
//! 3. Build a search region for surviving lineages and look up candidate
//!    observations in it (`obs_noise`, `top_k`, `radius_strategy`,
//!    `likelihood_threshold`).
//! 4. Prune the resulting branch tree, per-lineage and across the whole
//!    night (`branch_cap`, `n_scan`).
//! 5. Score the null-detection hypothesis for lineages predicted bright
//!    enough to have been seen (`limiting_magnitude`,
//!    `completeness_width_mag`).
//!
//! This configuration is `serde`-deserializable (YAML) and uses the
//! project-level unit parsers from [`crate::engine_config::units`] for its
//! time/angle/angular-variance fields.
//!
//! Note: unlike `PairConfig`/`TripletConfig`, this struct has no
//! `validate()` method — the numeric-range expectations documented on each
//! field (e.g. `completeness_width_mag > 0`, `branch_cap ≥ 1`) are not
//! currently enforced at load time.

use serde::{Deserialize, Serialize};

use crate::engine_config::units::{de_angle_rad, de_angle_var_rad2_pair, de_time_days};
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
    /// Maximum epoch spread for two observations to be folded into the same
    /// [`Visit`] (see [`group_observations_into_visits`]).
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
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_angle_rad`].
    #[serde(deserialize_with = "de_angle_rad")]
    pub quick_reject_radius_rad: f64,

    /// **A priori** diagonal astrometric noise `[σ_RA², σ_Dec²]`, added to
    /// each hypothesis's predicted sky covariance solely to size the search
    /// region in [`KFBank::predict_search_region`].
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
    /// [`KFBank::branch_with`]'s mixture-likelihood scoring and the Kalman
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
    /// through to [`KFBank::predict_search_region`].
    ///
    /// No dedicated `units.rs` parser applies here: `TopK` is a plain enum
    /// defined outside `engine_config` (in `topocentric_kf`), deserialized
    /// via its own `derive(Deserialize)`.
    pub top_k: TopK,

    /// How the search region's bounding radius is computed from the
    /// selected hypotheses' covariances — see [`RadiusStrategy`]
    /// (`MixtureCovariance` vs. `MaxEllipse`). Passed straight through to
    /// [`KFBank::predict_search_region`].
    ///
    /// No dedicated `units.rs` parser applies here either, for the same
    /// reason as `top_k`; note that `RadiusStrategy::Clamped`'s own
    /// `max_arcsec` field is expressed directly in arcseconds by that type,
    /// independent of this module's angle unit conventions.
    pub radius_strategy: RadiusStrategy,

    /// Minimum mixture predictive likelihood (unitless, a Gaussian density
    /// value — see
    /// [`SearchRegion::mixture_likelihood`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion::mixture_likelihood))
    /// a candidate observation must reach, *after* passing the coarse
    /// per-component Mahalanobis gate, to be kept by
    /// [`find_candidates_for_bank`]. A second-stage cut on top of the gate:
    /// the gate says "geometrically plausible," this says "and not
    /// negligibly unlikely." Must be `≥ 0.0`; `0.0` disables this stage
    /// (keep everything the gate accepts).
    pub likelihood_threshold: f64,

    /// Top-B cap: maximum number of branches kept **per lineage**, applied
    /// via [`cap_top_b_per_lineage`] after *every visit* — not just once
    /// per night, since branch counts multiply at every branching event
    /// (M candidates + 1 null branch) and would explode across a night's
    /// worth of visits otherwise.
    ///
    /// Dimensionless count, must be `≥ 1`. The design doc recommends
    /// `B ≈ 3–5`: wide enough to carry real ambiguity a visit or two,
    /// narrow enough to bound cost.
    pub branch_cap: usize,

    /// N-scan pruning window, in **nights** (not visits — see
    /// [`apply_n_scan_pruning`]), applied exactly once per call to
    /// [`advance_bank_collection_one_night`], after every visit that night
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

    /// Survey/field limiting magnitude for this night, used as the midpoint
    /// of the null branch's detection-probability curve — see
    /// [`detection_probability`].
    ///
    /// Units: magnitudes (no `units.rs` parser — a single, unambiguous
    /// photometric scale, unlike angles/time which have many common units).
    /// A lineage predicted brighter than this is very likely to have been
    /// detected (so a non-detection weighs heavily against the null
    /// branch); predicted fainter, the opposite.
    pub limiting_magnitude: f64,

    /// Completeness roll-off width of the survey's detection curve around
    /// `limiting_magnitude` — see [`detection_probability`]. Real surveys
    /// don't have a hard cutoff magnitude; detection probability decays
    /// smoothly over roughly this many magnitudes on either side of
    /// `limiting_magnitude`.
    ///
    /// Units: magnitudes. Must be strictly positive (not currently enforced
    /// by any `validate()` method on this struct — see the module-level
    /// note). Typical values ≈ 0.3–0.5 mag.
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

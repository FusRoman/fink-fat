//! Fragment linkage: deciding when two arcs are the same object.
//!
//! # Why this exists
//!
//! An object's observations routinely end up split across several lineages:
//! the association breaks at some point (a gate rejection, a cadence gap),
//! the orphaned observations are re-seeded as a brand-new lineage, and
//! nothing ever reunites the two. Measured on a 200-night ZTF-cadence run,
//! **42 % of trackable trajectories came out fragmented, and 40 676 of them
//! were split into pieces that were each individually pure** — reconstructions
//! that are already correct and only need stitching. That is the largest
//! remaining completeness deficit in the pipeline.
//!
//! Since [`ArchivedTrajectory`](super::archive::ArchivedTrajectory) keeps a purged lineage's arc instead of
//! destroying it, both halves of such a split are still available to link.
//!
//! # The hard constraint: never merge two different objects
//!
//! A wrong merge is worse than a missed one — it manufactures a contaminated
//! trajectory out of two clean ones. Every criterion here is therefore built
//! to be *specific* first and sensitive second, and they compose as a
//! cascade, cheapest stage first:
//!
//! 1. [`temporally_disjoint`] — an object has one position per epoch, so
//!    overlapping arcs cannot be the same body. Free.
//! 2. [`photometric_compatible`] — absolute magnitude `H` is intrinsic to the
//!    object (unlike apparent magnitude, which varies with distance and
//!    phase), so two arcs of one body must agree on it. Two scalars.
//! 3. [`sequential_fit`] — walk the converged arc through the other's
//!    observations, scoring each against the prediction *and absorbing it*, so
//!    the covariance tightens as it goes. The expensive, and by far the
//!    strongest, test.
//!
//! # On the photometric threshold
//!
//! [`implied_absolute_magnitude`](super::detection_probability::implied_absolute_magnitude)
//! omits the H-G phase-angle term, so `H` measured on two arcs observed at
//! different phase angles carries a **systematic** offset of a few tenths of
//! a magnitude — not noise that averages out with more observations. A
//! variance-based test (`|ΔH| / σ`) would therefore be confidently wrong: the
//! within-arc scatter it measures says nothing about that between-arc bias.
//! Hence a plain absolute threshold, sized to absorb the missing phase term
//! *plus* the object's lightcurve amplitude (≲ 0.5 mag is typical). Adding
//! the phase term upstream is what would let this tighten.

use ahash::AHashMap;
use nalgebra::{Matrix2, Vector2, Vector3};
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsId, observation::Observation},
};
use rayon::prelude::*;

use crate::{
    spacetime_bucket::{
        healpix_binner::HealpixBinner,
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner},
        uniform_time_binner::UniformTimeBinner,
    },
    topocentric_kf::single_kalman::{KFState, update::wrap_angle},
};

/// Tuning for the linkage cascade and the index that feeds it.
///
/// Every optional stage is disabled by `None` rather than by a numeric
/// sentinel: a `0.0`-means-off convention once inverted the photometric gate
/// in the shadow study, turning the row labelled "off" into the strictest test
/// in the sweep and producing two reports that concluded "zero contamination"
/// from a broken row.
///
/// The defaults are the operating point the sweep retained against ground
/// truth — most recall at ≥ 99 % precision with zero wrong merge on
/// NEO/Centaur/KBO/SDO — not hand-chosen values.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MergeParams {
    /// Minimum number of observations an arc must carry to **predict** — see
    /// [`arc_is_converged`]. `0` disables the gate.
    ///
    /// It acts on the base rate rather than on any one criterion, which is why
    /// it matters more than any threshold. A 2-to-5-point arc has an
    /// unconstrained `ρ`, so propagating it is meaningless; its covariance is
    /// enormous, so the Mahalanobis stage waves it through.
    ///
    /// Note this gates the *predictor* only. Short arcs are still indexed as
    /// **candidates**, because `(α, δ, α̇, δ̇)` at an arc's own epoch is well
    /// determined even from two points — applying the gate to both sides used
    /// to lose 12 090 true pairs against 523 lost to cell sizing.
    pub min_arc_points: usize,

    /// Max `|H_A − H_B|` (magnitudes) for two arcs to be photometrically
    /// compatible. See the module docs for why this is an absolute threshold
    /// and not a σ-scaled one.
    ///
    /// `None` by default because measurement says so: with the H-G phase term
    /// wired, correct pairs sit at `|ΔH|` p50 = 0.35 against wrong pairs at
    /// 1.11 — real separation, but overlapping enough (correct p90 = 1.51)
    /// that any useful cut costs more recall than the precision it buys. The
    /// residual error is dominated by `ρ`, not by phase: `H` carries
    /// `5·log10(r·Δ)`, so a 30 % range error shifts it by 0.57 mag.
    pub max_delta_h: Option<f64>,

    /// Maximum **reduced** chi-square of the sequential fit.
    ///
    /// Reduced, not raw, because it has a physical reference: each point
    /// contributes `d² ~ χ²(2)`, so the statistic should sit near 1 when the
    /// filter is calibrated — a statement about fit quality rather than a
    /// tuned constant. A gate on the raw `d²` never bit at all: thresholds of
    /// 9.21, 23 and 100 rejected exactly the same pairs.
    ///
    /// `None` by default: once the index proposes *dynamically confusable*
    /// pairs (same sky cell, same proper motion) rather than accidental
    /// coincidences, this statistic stops separating them — correct p99 = 53
    /// against wrong p10 = 0.12. It regains value at lower precision targets.
    pub max_reduced_chi2: Option<f64>,

    /// Judge [`Self::max_reduced_chi2`] on the whole absorbed walk rather than
    /// on the first predicted point.
    ///
    /// The first point is the only *pure* prediction test — [`sequential_fit`]
    /// absorbs each observation as it goes, so later terms are partly
    /// self-fulfilling. The whole walk nonetheless measured more discriminating
    /// at equal residual (64.5 % precision against 49.4 %), the extra evidence
    /// outweighing the optimism.
    pub chi2_on_whole_walk: bool,

    /// Absolute cap (arcsec) on the prediction-to-observation miss, applied to
    /// every sampled point.
    ///
    /// The statistical gate alone is not enough: after a long gap the predicted
    /// box spans ~900 arcsec, so a 3σ test accepts anything within half a
    /// degree. An absolute bound asks the physical question the covariance
    /// cannot — *how far off was it, really?* This carries most of the
    /// cascade's precision.
    pub max_residual_arcsec: Option<f64>,

    /// Max disagreement in proper motion between the predictor propagated to
    /// the tested arc's epoch and that arc's own attributable (radians/day,
    /// both components).
    ///
    /// Nearly independent of the position residual — two unrelated arcs can
    /// share a sky cell by coincidence, but sharing a cell *and* a proper
    /// motion is a far stronger statement. This is the criterion that made
    /// ≥ 99 % precision reachable at all.
    pub max_rate_diff_rad_per_day: Option<f64>,

    /// Refuse to link across a gap longer than this (days). `None` disables
    /// the limit.
    ///
    /// Also bounds how far a predictor is carried when searching, so it caps
    /// recall directly: at 30 days it excluded 3391 true pairs, at 60 days
    /// 1032. Long gaps are where two-body propagation drifts and where two
    /// unrelated objects are most likely to look compatible.
    pub max_gap_days: Option<f64>,

    /// Reject any connected component holding more arcs than this.
    ///
    /// Linkage is transitive: A–B and B–C produce {A, B, C}. **A single wrong
    /// link therefore merges two entire components**, which is why pairwise
    /// precision does not translate into component purity. A component of
    /// thirty arcs is not an object fragmented thirty times, it is a chain of
    /// errors — refusing it is cheaper than trying to find the bad link inside
    /// it. `0` disables the cap.
    pub max_component_size: usize,

    /// HEALPix depth of the candidate index's sky cells.
    ///
    /// Depth 5 is ~1.8° across: coarse enough to absorb the propagation error
    /// over a short hop, fine enough that a cell holds few unrelated arcs.
    pub index_healpix_depth: u8,

    /// Width of one index time slice (days).
    ///
    /// One night: arcs are nightly tracklets, and within a night an object
    /// moves far less than a cell, so a finer slice would only multiply
    /// propagations without separating anything.
    pub time_slice_days: f64,

    /// How many observations of the tested arc are walked through.
    ///
    /// Each costs a Kepler propagation plus a Kalman update. A spread sample is
    /// nearly as decisive as the whole arc: the covariance collapses after the
    /// first one or two absorbed points.
    pub max_test_points: usize,
}

impl Default for MergeParams {
    /// The operating point retained by the ground-truth sweep: 4838 correct
    /// merges against 43 wrong, 99.12 % precision, zero wrong merge on
    /// NEO/Centaur/KBO/SDO.
    fn default() -> Self {
        Self {
            min_arc_points: 6,
            max_delta_h: None,
            max_reduced_chi2: None,
            chi2_on_whole_walk: true,
            max_residual_arcsec: Some(300.0),
            // 0.3 arcmin/day.
            max_rate_diff_rad_per_day: Some(8.726_646_259_971_648e-5),
            max_gap_days: Some(60.0),
            max_component_size: 8,
            index_healpix_depth: 5,
            time_slice_days: 1.0,
            max_test_points: 5,
        }
    }
}

impl crate::engine_config::Validate for MergeParams {
    /// Accumulates every failure rather than stopping at the first, matching
    /// the rest of the configuration.
    ///
    /// Only supplied values are range-checked: `None` means "stage disabled"
    /// and is always valid.
    fn validate(&self) -> Result<(), Vec<crate::engine_config::error::FieldError>> {
        use crate::engine_config::validate_helpers::{check_finite_positive, check_min_usize};

        let mut errors = Vec::new();

        for (name, value, hint) in [
            (
                "max_delta_h",
                self.max_delta_h,
                "set max_delta_h to a positive magnitude difference, e.g. 0.6, or null to disable",
            ),
            (
                "max_reduced_chi2",
                self.max_reduced_chi2,
                "set max_reduced_chi2 to a positive reduced chi-square, e.g. 2.0, or null to disable",
            ),
            (
                "max_residual_arcsec",
                self.max_residual_arcsec,
                "set max_residual_arcsec to a positive angle in arcsec, e.g. 300.0, or null to disable",
            ),
            (
                "max_rate_diff_rad_per_day",
                self.max_rate_diff_rad_per_day,
                "set max_rate_diff_rad_per_day to a positive rate in rad/day, or null to disable",
            ),
            (
                "max_gap_days",
                self.max_gap_days,
                "set max_gap_days to a positive number of days, e.g. 60.0, or null to disable",
            ),
        ] {
            if let Some(v) = value
                && let Some(e) = check_finite_positive(name, v, hint)
            {
                errors.push(e);
            }
        }

        if let Some(e) = check_finite_positive(
            "time_slice_days",
            self.time_slice_days,
            "set time_slice_days to a positive slice width, e.g. 1.0 (one night)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_min_usize(
            "max_test_points",
            self.max_test_points,
            1,
            "set max_test_points to at least 1: a link must be demonstrated on some observation",
        ) {
            errors.push(e);
        }
        // Depth 0 is a 12-cell whole-sky tessellation, which would make the
        // index useless rather than merely coarse.
        if self.index_healpix_depth == 0 || self.index_healpix_depth > 29 {
            errors.push(
                crate::engine_config::error::FieldError::new(
                    "index_healpix_depth",
                    format!(
                        "{} is outside the usable HEALPix range",
                        self.index_healpix_depth
                    ),
                )
                .with_hint(
                    "set index_healpix_depth between 1 and 29; 5 (~1.8 deg cells) is the measured default",
                ),
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// One observation of the arc being tested, with the observer geometry at its
/// epoch — everything [`sequential_fit`] needs to both *score* the point and
/// *absorb* it, without reaching back into the observation dataset.
#[derive(Debug, Clone, Copy)]
pub struct LinkTestPoint<'obs> {
    /// The observation itself: its epoch and sky position score the
    /// prediction, and [`KFState::update`] consumes it to tighten the state.
    pub observation: &'obs Observation,
    /// Observer heliocentric position/velocity at the observation's epoch.
    pub r_obs: Vector3<f64>,
    pub v_obs: Vector3<f64>,
}

/// Does this arc carry enough observations for its orbit to mean anything?
///
/// `n_points` is the arc's observation count (`track_ids.len()`). Note the
/// engine's own `n_real_updates` counts one *fewer*: a lineage is seeded from
/// a bootstrap **pair** that registers as a single update, so
/// `n_real_updates == n_points - 1`. Six points is therefore
/// `n_real_updates >= 5`.
///
/// `min_points == 0` disables the gate.
pub fn arc_is_converged(n_points: usize, min_points: usize) -> bool {
    min_points == 0 || n_points >= min_points
}

/// Do these two arcs occupy disjoint time spans?
///
/// An object has exactly one position per epoch, so overlapping arcs are
/// either the same detections counted twice or two different bodies —
/// neither is something to merge. `gap_days` between them must also stay
/// within `max_gap_days` when that limit is enabled.
///
/// Arguments are `(start, end)` epochs in MJD TT.
pub fn temporally_disjoint(earlier: (f64, f64), later: (f64, f64), max_gap_days: f64) -> bool {
    let (earlier_start, earlier_end) = earlier;
    let (later_start, later_end) = later;

    // Well-formed spans only; a NaN epoch must never be read as "disjoint".
    if ![earlier_start, earlier_end, later_start, later_end]
        .iter()
        .all(|e| e.is_finite())
    {
        return false;
    }
    if earlier_end > later_start || earlier_start > earlier_end || later_start > later_end {
        return false;
    }

    let gap = later_start - earlier_end;
    max_gap_days <= 0.0 || gap <= max_gap_days
}

/// Do these two arcs agree on the object's intrinsic brightness?
///
/// Returns `true` when either estimate is missing: an arc that never
/// consumed a real observation carries no photometry, and absence of
/// evidence must not by itself veto a link that the dynamical stage would
/// otherwise confirm.
pub fn photometric_compatible(h_a: Option<f64>, h_b: Option<f64>, max_delta_h: f64) -> bool {
    if max_delta_h <= 0.0 {
        return true;
    }
    match (h_a, h_b) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a - b).abs() <= max_delta_h,
        _ => true,
    }
}

/// Outcome of walking a converged arc through another arc's observations,
/// absorbing each one as it goes.
///
/// # Why sequential, and not one independent test per point
///
/// Testing every point against the *same* un-updated covariance asks "is this
/// consistent with how uncertain I am?" — and after a multi-week gap the
/// predicted sky box is ~900 arcsec across, so almost anything nearby is
/// consistent. Measured on a full run, that test left a 1 % link precision.
///
/// Absorbing each observation instead collapses the covariance: by the second
/// or third point the box has shrunk by orders of magnitude, and the remaining
/// points face a test at the *astrometric noise* level rather than the
/// prediction-uncertainty level. A genuine continuation keeps fitting; a
/// coincidence diverges immediately. That is the question real linkage asks —
/// "does one orbit explain both arcs to within the measurement error?" — and
/// it is three orders of magnitude sharper.
#[derive(Debug, Clone, Default)]
pub struct SequentialFit {
    /// Squared Mahalanobis distance of each point, measured **before** that
    /// point is absorbed, in chronological order.
    pub d2: Vec<f64>,
    /// Angular separation between prediction and observation (arcsec), same
    /// order. Reported alongside `d2` because a huge covariance can make a
    /// large miss look statistically fine; an absolute residual cannot.
    pub separation_arcsec: Vec<f64>,
}

impl SequentialFit {
    /// Reduced chi-square of the **first** point: `d²₁ / 2`.
    ///
    /// The honest prediction test. Each point contributes a `d² ~ χ²(2)` when
    /// the filter is calibrated, so this should sit near 1 for a genuine
    /// continuation — a physical reference rather than a tuned threshold.
    ///
    /// Why the first point specifically: [`sequential_fit`] *absorbs* each
    /// observation as it goes, so from the second point onward the filter has
    /// already adapted to whatever it was fed, including a wrong arc. Later
    /// `d²` values are therefore partly self-fulfilling. Measured on a full
    /// run, gates of 9.21, 23 and 100 on the raw `d²` rejected *exactly the
    /// same* pairs — the statistic had no discriminating power left. The first
    /// point is taken before any of that.
    ///
    /// `None` for an empty fit.
    pub fn reduced_chi2_first(&self) -> Option<f64> {
        self.d2.first().map(|d2| d2 / 2.0)
    }

    /// Reduced chi-square over the whole walk: `Σd²ᵢ / (2n)`.
    ///
    /// Overall consistency of the joint fit. Also expected near 1 when
    /// calibrated, but see [`Self::reduced_chi2_first`] for why the later
    /// terms are optimistic. Note the engine's filter is known to be
    /// over-confident, so a value above 1 measures that over-confidence as
    /// much as it measures the link's quality — read the true-vs-false
    /// distributions before choosing a cut.
    ///
    /// `None` for an empty fit.
    pub fn reduced_chi2(&self) -> Option<f64> {
        if self.d2.is_empty() {
            return None;
        }
        Some(self.d2.iter().sum::<f64>() / (2.0 * self.d2.len() as f64))
    }

    /// Whether the fit clears the configured bounds.
    ///
    /// `None` disables a test. An empty fit never passes: a link must be
    /// demonstrated, not assumed.
    ///
    /// Which chi-square is used is a real choice, not a detail — see
    /// [`MergeParams::chi2_on_whole_walk`]. The absolute residual cannot be
    /// gamed by the sequential absorption, so it is required of every point.
    pub fn passes(
        &self,
        max_reduced_chi2: Option<f64>,
        chi2_on_whole_walk: bool,
        max_residual_arcsec: Option<f64>,
    ) -> bool {
        if self.d2.is_empty() {
            return false;
        }
        if let Some(limit) = max_reduced_chi2 {
            let chi2 = if chi2_on_whole_walk {
                self.reduced_chi2()
            } else {
                self.reduced_chi2_first()
            };
            if !chi2.is_some_and(|c| c <= limit) {
                return false;
            }
        }
        match max_residual_arcsec {
            Some(limit) => self.separation_arcsec.iter().all(|sep| *sep <= limit),
            None => true,
        }
    }
}

/// Radians to arcseconds.
const RAD_TO_ARCSEC: f64 = 206_264.806_247_096_36;

/// Walk `state` through `points`, scoring each observation against the
/// prediction and then absorbing it — see [`SequentialFit`].
///
/// `points` must be in chronological order. Propagation runs in whichever
/// direction each point requires: a converged arc may legitimately have to
/// predict *backwards* onto observations that precede it, which is why
/// process noise is built from `|dt|` (see `build_snc_process_noise`).
///
/// `None` as soon as any step fails to propagate or update — an
/// unjudgeable pair is never a link.
pub fn sequential_fit(state: &KFState<'_>, points: &[LinkTestPoint<'_>]) -> Option<SequentialFit> {
    let mut fit = SequentialFit::default();
    let mut current = state.clone();

    for point in points {
        let coord = point.observation.equ_coord();
        let predicted = current
            .predict(point.observation.mjd_tt(), point.r_obs, point.v_obs)
            .ok()?;

        // Raw (Δα, Δδ) against `H P Hᵀ + R`, no cos(δ) reduction: the state
        // carries α and δ directly and `sky_covariance` uses those same
        // coordinates — the same convention as the candidate-search gate.
        let sky = predicted.sky_covariance().ok()?;
        let r = Matrix2::new(
            coord.ra_error * coord.ra_error,
            0.0,
            0.0,
            coord.dec_error * coord.dec_error,
        );
        let s_inv = (sky + r).try_inverse()?;
        let innovation = Vector2::new(
            wrap_angle(coord.ra - predicted.state[0]),
            coord.dec - predicted.state[1],
        );
        let d2 = (innovation.transpose() * s_inv * innovation)[(0, 0)];
        if !d2.is_finite() {
            return None;
        }

        // Absolute miss on the tangent plane, where cos(δ) *does* apply.
        let dx = innovation[0] * predicted.state[1].cos();
        let separation = (dx * dx + innovation[1] * innovation[1]).sqrt() * RAD_TO_ARCSEC;

        fit.d2.push(d2);
        fit.separation_arcsec.push(separation);

        current = predicted.update(point.observation).ok()?;
    }

    (!fit.d2.is_empty()).then_some(fit)
}

/// Verdict of the full cascade, kept structured so a shadow study can report
/// *which* stage rejected a pair rather than just that it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeVerdict {
    /// The two arcs are the same object; link them.
    Link,
    /// Time spans overlap, or the gap exceeds `max_gap_days`.
    RejectedTemporal,
    /// Absolute magnitudes disagree by more than `max_delta_h`.
    RejectedPhotometric,
    /// Too few of the later arc's observations fall where the earlier arc
    /// predicts.
    RejectedDynamical,
    /// One of the arcs is too short for its orbit to constrain anything —
    /// see [`MergeParams::min_arc_points`].
    RejectedUnconverged,
}

impl MergeVerdict {
    pub fn label(self) -> &'static str {
        match self {
            MergeVerdict::Link => "link",
            MergeVerdict::RejectedTemporal => "rejected (temporal)",
            MergeVerdict::RejectedPhotometric => "rejected (photometric)",
            MergeVerdict::RejectedDynamical => "rejected (dynamical)",
            MergeVerdict::RejectedUnconverged => "rejected (unconverged arc)",
        }
    }
}

/// Run the whole cascade on one pair of arcs.
///
/// # The predictor is the *converged* arc, not the earlier one
///
/// The test is directional — one arc predicts the other's observations — but
/// which arc plays that role is a free choice, and a consequential one.
/// Measured on a full run, **48 % of all true pairs have a short arc first
/// and a long arc second**: a few-point orphan, then the object properly
/// picked up. Insisting the *earlier* arc predict would throw those away.
///
/// So `predictor` is whichever arc actually constrains an orbit, and it
/// predicts the other's observations regardless of which came first. Only the
/// predictor faces the convergence gate: the tested arc contributes raw sky
/// positions, and its own orbit quality never enters the computation.
/// Backward propagation is what makes this possible — see [`sequential_fit`].
#[allow(clippy::too_many_arguments)]
pub fn evaluate_pair(
    predictor_span: (f64, f64),
    predictor_state: &KFState<'_>,
    predictor_h: Option<f64>,
    predictor_n_points: usize,
    tested_span: (f64, f64),
    tested_h: Option<f64>,
    tested_points: &[LinkTestPoint<'_>],
    params: &MergeParams,
) -> MergeVerdict {
    // Cheapest and most decisive stage first — an arc below the threshold has
    // a meaningless orbit and a covariance so wide the dynamical test would
    // wave anything through.
    if !arc_is_converged(predictor_n_points, params.min_arc_points) {
        return MergeVerdict::RejectedUnconverged;
    }
    // Order-independent: the two arcs must simply not overlap in time.
    let max_gap = params.max_gap_days.unwrap_or(0.0);
    let disjoint = temporally_disjoint(predictor_span, tested_span, max_gap)
        || temporally_disjoint(tested_span, predictor_span, max_gap);
    if !disjoint {
        return MergeVerdict::RejectedTemporal;
    }
    if !photometric_compatible(predictor_h, tested_h, params.max_delta_h.unwrap_or(0.0)) {
        return MergeVerdict::RejectedPhotometric;
    }
    if params.max_reduced_chi2.is_some() || params.max_residual_arcsec.is_some() {
        match sequential_fit(predictor_state, tested_points) {
            Some(fit)
                if fit.passes(
                    params.max_reduced_chi2,
                    params.chi2_on_whole_walk,
                    params.max_residual_arcsec,
                ) => {}
            _ => return MergeVerdict::RejectedDynamical,
        }
    }
    MergeVerdict::Link
}

/// The angular half of an attributable: `(α, δ, α̇·cos δ, δ̇)`, radians and
/// radians/day. `None` if any component is non-finite.
///
/// # Why only the angular half
///
/// A topocentric attributable constrains `(α, δ, α̇, δ̇)` well and `(ρ, ρ̇)`
/// poorly. Everything derived from `ρ` — osculating elements above all —
/// inherits that ignorance amplified: measured on a full run, the true pairs an
/// orbital-element index missed differed by a median `Δe = 0.257` (thirteen bin
/// widths) and by `Δi = 68°` at p90. Those are not neighbouring orbits landing
/// either side of a cell boundary; they are orbits with no relationship at all,
/// and no bin width recovers them.
///
/// The angular part, by contrast, is what the observations directly measure —
/// even a two-point tracklet pins it down. That is why an arc too short to be
/// *propagated* can still be *indexed*.
///
/// The RA rate is reduced by `cos δ` so a given value means the same angular
/// speed everywhere on the sky; `state[2]` alone is a coordinate rate and blows
/// up near the poles.
pub fn attributable_at(state: &KFState<'_>) -> Option<(f64, f64, f64, f64)> {
    let (ra, dec, ra_dot, dec_dot) = (
        state.state[0],
        state.state[1],
        state.state[2],
        state.state[3],
    );
    if !(ra.is_finite() && dec.is_finite() && ra_dot.is_finite() && dec_dot.is_finite()) {
        return None;
    }
    Some((wrap_angle(ra), dec, ra_dot * dec.cos(), dec_dot))
}

/// Sky-cell key of a state, for bucketing arcs without an O(N²) sweep.
///
/// HEALPix rather than a lon/lat grid so RA wraparound and the poles are
/// handled correctly.
///
/// # The epoch is the caller's problem
///
/// Sky position only means something at a stated instant — an object is simply
/// somewhere else three weeks later. This function reads whatever state it is
/// handed, so the caller must pair this key with the epoch that state is
/// expressed at (see [`attributable_at`] for why the angular part is the
/// trustworthy half to index on).
///
/// Proper motion deliberately stays *out* of the key: quantising it forces a
/// ±1 probe on two more axes (81 lookups instead of 9) and still cuts arbitrarily
/// at bin edges. Comparing rates exactly, on the short candidate list a cell
/// lookup returns, is both cheaper and sharper — see [`rates_compatible`].
pub fn sky_cell_key(state: &KFState<'_>, spatial_binner: &HealpixBinner) -> Option<SpatialKey> {
    let (ra, dec, _, _) = attributable_at(state)?;
    Some(spatial_binner.key_for(&EquCoord::new(ra, 0.0, dec, 0.0)))
}

/// Whether two attributables agree in proper motion, both components within
/// `max_rate_diff_rad_per_day`.
///
/// Used as an exact post-filter on the candidates a sky-cell lookup returns.
/// Rate agreement is nearly independent of the position residual the dynamical
/// test measures: two unrelated arcs can share a cell by coincidence, but
/// sharing a cell *and* a proper motion is a far stronger statement.
///
/// A non-finite attributable never agrees with anything.
///
/// # Arguments
/// * `a`, `b` – Attributables as returned by [`attributable_at`], **at the same
///   epoch**. Comparing rates across epochs is meaningful only over gaps short
///   enough that the apparent motion has not changed, which is exactly the
///   regime a time-sliced index works in.
pub fn rates_compatible(
    a: (f64, f64, f64, f64),
    b: (f64, f64, f64, f64),
    max_rate_diff_rad_per_day: f64,
) -> bool {
    let (d_ra_rate, d_dec_rate) = ((a.2 - b.2).abs(), (a.3 - b.3).abs());
    d_ra_rate <= max_rate_diff_rad_per_day && d_dec_rate <= max_rate_diff_rad_per_day
}

/// The four indexing elements `(a, e, i, Ω)` of a state's osculating orbit,
/// in AU and radians.
///
/// `None` for anything that is not a bound, finite ellipse — hyperbolic or
/// numerically degenerate — which is also a sign the arc is not worth linking.
///
/// Shared by [`orbit_bucket_key`] and by diagnostics that need the raw
/// elements (e.g. measuring how far apart in element space two arcs of the
/// same object actually land, which is what sizes the bins).
pub fn orbit_elements(state: &KFState<'_>) -> Option<(f64, f64, f64, f64)> {
    use outfit::OrbitalElements;

    let OrbitalElements::Keplerian { elements, .. } = state.to_orbit() else {
        return None;
    };
    let (a, e, i, node) = (
        elements.semi_major_axis,
        elements.eccentricity,
        elements.inclination,
        elements.ascending_node_longitude,
    );
    (a.is_finite()
        && a > 0.0
        && e.is_finite()
        && (0.0..1.0).contains(&e)
        && i.is_finite()
        && node.is_finite())
    .then_some((a, e, i, node))
}

/// Coarse bucket key for pairing candidates without an O(N²) sweep.
///
/// Two arcs of one object share nearly identical orbital elements, so
/// quantized `(a, e, i, Ω)` is a cheap way to bring plausible pairs together —
/// the same idea classical linkage pipelines use. Only a *pairing* aid: every
/// pair it proposes still goes through [`evaluate_pair`], and neighbouring
/// buckets must be probed too, since an object near a cell boundary lands on
/// either side depending on which arc you ask.
///
/// # Why the ascending node is in the key
///
/// `(a, e, i)` alone does not identify a main-belt asteroid among other
/// main-belt asteroids: that region of element space is small and densely
/// populated. Measured on a full run, median bucket occupancy was 1 but the
/// densest held 244 arcs, and those few cells produced most of the 25.7 M
/// candidate pairs. The ascending node is near-uniform across objects and
/// well determined once an arc has converged, so adding it splits precisely
/// the crowded cells while leaving the already-isolated ones untouched.
///
/// `None` when the state yields no bound, finite ellipse (hyperbolic or
/// numerically degenerate), which is also a sign the arc is not worth
/// linking.
pub fn orbit_bucket_key(
    state: &KFState<'_>,
    a_bin_au: f64,
    e_bin: f64,
    i_bin_rad: f64,
    node_bin_rad: f64,
) -> Option<(i64, i64, i64, i64)> {
    debug_assert!(a_bin_au > 0.0 && e_bin > 0.0 && i_bin_rad > 0.0 && node_bin_rad > 0.0);

    let (a, e, i, node) = orbit_elements(state)?;
    Some((
        (a / a_bin_au).floor() as i64,
        (e / e_bin).floor() as i64,
        (i / i_bin_rad).floor() as i64,
        (node / node_bin_rad).floor() as i64,
    ))
}

/// Merge two arcs' association histories into one, chronologically ordered
/// and duplicate-free.
///
/// `epoch_of` resolves an observation's epoch; ids it cannot resolve are kept
/// but sort last, so an unresolvable id degrades ordering rather than losing
/// the observation.
pub fn merged_track_ids(
    earlier: &[ObsId],
    later: &[ObsId],
    epoch_of: impl Fn(ObsId) -> Option<f64>,
) -> Vec<ObsId> {
    let mut ids: Vec<ObsId> = earlier.iter().chain(later).copied().collect();
    ids.sort_unstable_by(|a, b| {
        let ka = epoch_of(*a).unwrap_or(f64::INFINITY);
        let kb = epoch_of(*b).unwrap_or(f64::INFINITY);
        ka.total_cmp(&kb).then_with(|| a.cmp(b))
    });
    ids.dedup();
    ids
}

// ── Whole-run linkage ───────────────────────────────────────────────────────

/// One arc, as the linkage pass sees it.
///
/// Test points are supplied by the caller rather than resolved here: the
/// engine has no observation dataset, and precomputing them is cheaper anyway
/// since every arc is a potential *tested* arc and a pair-by-pair resolution
/// would repeat the same lookups millions of times.
pub struct MergeFragment<'state_lf, 'obs> {
    /// Observations this arc accounts for.
    pub track_ids: &'obs [ObsId],
    /// MAP state at the arc's own epoch.
    pub state: KFState<'state_lf>,
    /// Running absolute-magnitude estimate, if any.
    pub h: Option<f64>,
    /// `(first, last)` observation epoch, MJD TT.
    pub span: (f64, f64),
    /// Sampled observations used when this arc is the *tested* one, in
    /// chronological order — see [`sequential_fit`].
    pub test_points: Vec<LinkTestPoint<'obs>>,
}

/// What a linkage pass produced.
pub struct MergeOutcome {
    /// Connected components of the accepted links, as fragment indices. Every
    /// fragment appears exactly once, so a component of length 1 is an arc
    /// nothing linked to.
    pub components: Vec<Vec<usize>>,
    /// Ordered pairs the index proposed, before any gate.
    pub n_pairs_tested: usize,
    /// Pairs the cascade accepted.
    pub n_links_accepted: usize,
    /// Components refused for exceeding
    /// [`MergeParams::max_component_size`], and the arcs they held. Reported
    /// rather than silently dropped: a component that large is the signature
    /// of a chain of wrong links, and knowing how often it happens is the only
    /// way to tell a tuning problem from a rare accident.
    pub n_components_rejected_oversize: usize,
    pub n_arcs_in_rejected_components: usize,
}

/// Disjoint-set forest over fragment indices, with union by size and path
/// halving.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

/// Canonical ordering of a candidate pair: `(predictor, tested)`.
///
/// The predictor is the **longer** arc, not the earlier one. Measurement
/// showed 48 % of true pairs have the short arc first, so insisting the
/// earlier arc predict would throw those away. `None` when the spans overlap:
/// one object holds one position per epoch, so overlapping arcs are duplicates
/// rather than continuations.
fn order_pair(fragments: &[MergeFragment<'_, '_>], a: usize, b: usize) -> Option<(usize, usize)> {
    let (a_span, b_span) = (fragments[a].span, fragments[b].span);
    if !(a_span.1 <= b_span.0 || b_span.1 <= a_span.0) {
        return None;
    }
    let (na, nb) = (fragments[a].track_ids.len(), fragments[b].track_ids.len());
    Some(if (nb, b) > (na, a) { (b, a) } else { (a, b) })
}

/// One night's worth of indexed candidate arcs.
struct TimeSlice {
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
    /// Sky cell → fragment indices whose own epoch falls in this night.
    cells: AHashMap<u64, Vec<usize>>,
}

/// Ordered candidate pairs proposed by a **time-sliced**, asymmetric index.
///
/// # Why the two sides are treated differently
///
/// Carrying every arc to one shared reference epoch forces a long propagation
/// on both sides, and a propagation is only as good as `ρ` — the
/// badly-determined half of a topocentric attributable. Such an index has to
/// demand convergence of both arcs, and measurement showed the price: of
/// 12 613 true pairs missed, **12 090 were missed because one arc was too
/// short to index**, against 523 genuinely mis-binned.
///
/// So:
/// * **candidates** are indexed at *their own epoch* with no propagation at
///   all — `(α, δ, α̇, δ̇)` is what the observations directly measure, and a
///   two-point tracklet already pins it down, so arc length stops mattering;
/// * **predictors** must be converged, and are carried only to the nightly
///   slices within [`MergeParams::max_gap_days`] of their own span.
///
/// Proper motion stays out of the key deliberately: quantising it would add a
/// ±1 probe on two more axes (81 lookups per slice instead of 9) while still
/// cutting arbitrarily at bin edges. Comparing rates exactly on the short list
/// a cell lookup returns is cheaper and sharper.
fn candidate_pairs(
    fragments: &[MergeFragment<'_, '_>],
    params: &MergeParams,
) -> Vec<(usize, usize)> {
    let binner = HealpixBinner::new(params.index_healpix_depth);
    let max_gap = params.max_gap_days.unwrap_or(f64::INFINITY);

    let attributables: Vec<Option<(f64, f64, f64, f64)>> = fragments
        .iter()
        .map(|f| attributable_at(&f.state))
        .collect();

    let t0 = fragments
        .iter()
        .map(|f| f.state.epoch)
        .filter(|e| e.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !t0.is_finite() {
        return Vec::new();
    }
    let time_binner = UniformTimeBinner::new(t0, params.time_slice_days);

    let mut slices: AHashMap<TimeBin, TimeSlice> = AHashMap::default();
    for (slot, fragment) in fragments.iter().enumerate() {
        let epoch = fragment.state.epoch;
        if attributables[slot].is_none() || !epoch.is_finite() {
            continue;
        }
        let Some(SpatialKey(cell)) = sky_cell_key(&fragment.state, &binner) else {
            continue;
        };
        slices
            .entry(time_binner.bin_for(epoch))
            .or_insert_with(|| TimeSlice {
                epoch,
                r_obs: fragment.state.r_obs,
                v_obs: fragment.state.v_obs,
                cells: AHashMap::default(),
            })
            .cells
            .entry(cell)
            .or_default()
            .push(slot);
    }

    // An arc near a cell edge lands either side depending on which arc you
    // ask, so probing the exact cell alone would lose precisely the pairs of
    // interest.
    let probe_radius = binner.cell_radius() * 1.5;

    let mut pairs: Vec<(usize, usize)> = fragments
        .par_iter()
        .enumerate()
        .filter(|(_, f)| arc_is_converged(f.track_ids.len(), params.min_arc_points))
        .flat_map_iter(|(a, predictor)| {
            let mut local: Vec<(usize, usize)> = Vec::new();
            let (span_start, span_end) = predictor.span;

            // Only outside the predictor's own span: an arc overlapping it is a
            // duplicate, not a continuation.
            for (lo, hi) in [
                (span_start - max_gap, span_start),
                (span_end, span_end + max_gap),
            ] {
                for bin in time_binner.bins_in_range(lo, hi) {
                    let Some(slice) = slices.get(&bin) else {
                        continue;
                    };
                    let Ok(propagated) =
                        predictor
                            .state
                            .predict(slice.epoch, slice.r_obs, slice.v_obs)
                    else {
                        continue;
                    };
                    let Some(attr_pred) = attributable_at(&propagated) else {
                        continue;
                    };
                    let Some(cell) = sky_cell_key(&propagated, &binner) else {
                        continue;
                    };

                    for SpatialKey(neighbour) in binner.neighbors(cell, probe_radius) {
                        let Some(slots) = slice.cells.get(&neighbour) else {
                            continue;
                        };
                        for &b in slots {
                            if a == b {
                                continue;
                            }
                            let Some(attr_b) = attributables[b] else {
                                continue;
                            };
                            if let Some(tol) = params.max_rate_diff_rad_per_day
                                && !rates_compatible(attr_pred, attr_b, tol)
                            {
                                continue;
                            }
                            let Some(pair) = order_pair(fragments, a, b) else {
                                continue; // overlapping: never mergeable
                            };
                            local.push(pair);
                        }
                    }
                }
            }
            local
        })
        .collect();

    // The same pair reaches the list once per slice that proposed it.
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Link fragments of the same object across the whole run.
///
/// # Why components, and why they are capped
///
/// Linkage is transitive: accepting A–B and B–C yields `{A, B, C}`. That is
/// the point — an object broken into three pieces should come back as one —
/// but it also means **a single wrong link merges two entire components**.
/// Pairwise precision therefore does not carry over to component purity, and
/// [`MergeParams::max_component_size`] exists to bound the blast radius: past
/// a certain size a component is a chain of errors rather than a heavily
/// fragmented object, and refusing it wholesale is cheaper and safer than
/// trying to find the bad link inside it.
///
/// Rejected components are split back into their individual arcs, so nothing
/// is ever lost — the worst case is that linkage did nothing for them.
///
/// This does **not** touch any filter state: it consumes arcs and returns
/// groupings, leaving banks, hypotheses and the snapshot schema untouched.
pub fn merge_fragments(fragments: &[MergeFragment<'_, '_>], params: &MergeParams) -> MergeOutcome {
    let pairs = candidate_pairs(fragments, params);

    let accepted: Vec<(usize, usize)> = pairs
        .par_iter()
        .filter(|&&(predictor, tested)| {
            let (p, t) = (&fragments[predictor], &fragments[tested]);
            evaluate_pair(
                p.span,
                &p.state,
                p.h,
                p.track_ids.len(),
                t.span,
                t.h,
                &t.test_points,
                params,
            ) == MergeVerdict::Link
        })
        .copied()
        .collect();

    let mut uf = UnionFind::new(fragments.len());
    for &(a, b) in &accepted {
        uf.union(a, b);
    }

    let mut by_root: AHashMap<usize, Vec<usize>> = AHashMap::default();
    for slot in 0..fragments.len() {
        by_root.entry(uf.find(slot)).or_default().push(slot);
    }

    let mut components = Vec::with_capacity(by_root.len());
    let mut n_components_rejected_oversize = 0usize;
    let mut n_arcs_in_rejected_components = 0usize;
    for (_, members) in by_root {
        if params.max_component_size > 0 && members.len() > params.max_component_size {
            n_components_rejected_oversize += 1;
            n_arcs_in_rejected_components += members.len();
            // Dissolved back into singletons: refusing to merge must never
            // mean losing the arcs.
            components.extend(members.into_iter().map(|slot| vec![slot]));
            continue;
        }
        components.push(members);
    }
    // Deterministic order regardless of hash iteration.
    for component in &mut components {
        component.sort_unstable();
    }
    components.sort_unstable();

    MergeOutcome {
        components,
        n_pairs_tested: pairs.len(),
        n_links_accepted: accepted.len(),
        n_components_rejected_oversize,
        n_arcs_in_rejected_components,
    }
}

/// Observations of one component, chronologically ordered and duplicate-free.
///
/// `epoch_of` resolves an observation's epoch; ids it cannot resolve are kept
/// but sort last, so an unresolvable id degrades ordering rather than losing
/// the observation.
pub fn component_track_ids(
    fragments: &[MergeFragment<'_, '_>],
    component: &[usize],
    epoch_of: impl Fn(ObsId) -> Option<f64>,
) -> Vec<ObsId> {
    let mut ids: Vec<ObsId> = component
        .iter()
        .flat_map(|&slot| fragments[slot].track_ids.iter().copied())
        .collect();
    ids.sort_unstable_by(|a, b| {
        let ka = epoch_of(*a).unwrap_or(f64::INFINITY);
        let kb = epoch_of(*b).unwrap_or(f64::INFINITY);
        ka.total_cmp(&kb).then_with(|| a.cmp(b))
    });
    ids.dedup();
    ids
}

#[cfg(test)]
mod merge_tests {
    use super::*;

    // Note: anything touching `KFState` needs a live `KalmanContext` (JPL
    // ephemeris over the network), so the dynamical stage is not unit
    // testable here — the same limitation documented across this module
    // tree. The pure decision logic below is, and it is where the
    // contamination risk actually lives.

    #[test]
    fn short_arcs_are_not_converged() {
        assert!(!arc_is_converged(5, 6));
        assert!(arc_is_converged(6, 6));
        assert!(arc_is_converged(40, 6));
    }

    #[test]
    fn zero_threshold_disables_the_convergence_gate() {
        assert!(arc_is_converged(2, 0));
    }

    #[test]
    fn overlapping_spans_never_link() {
        assert!(!temporally_disjoint((0.0, 10.0), (5.0, 20.0), 0.0));
    }

    #[test]
    fn touching_spans_are_disjoint() {
        assert!(temporally_disjoint((0.0, 10.0), (10.0, 20.0), 0.0));
    }

    #[test]
    fn later_arc_before_earlier_is_rejected() {
        assert!(!temporally_disjoint((10.0, 20.0), (0.0, 5.0), 0.0));
    }

    #[test]
    fn gap_beyond_limit_is_rejected() {
        assert!(temporally_disjoint((0.0, 10.0), (30.0, 40.0), 25.0));
        assert!(!temporally_disjoint((0.0, 10.0), (40.0, 50.0), 25.0));
    }

    #[test]
    fn zero_max_gap_disables_the_gap_limit() {
        assert!(temporally_disjoint((0.0, 10.0), (10_000.0, 10_001.0), 0.0));
    }

    #[test]
    fn non_finite_epochs_never_link() {
        assert!(!temporally_disjoint((0.0, f64::NAN), (10.0, 20.0), 0.0));
        assert!(!temporally_disjoint(
            (0.0, 10.0),
            (f64::INFINITY, 20.0),
            0.0
        ));
    }

    #[test]
    fn photometric_gate_compares_absolute_magnitudes() {
        assert!(photometric_compatible(Some(17.0), Some(17.4), 0.6));
        assert!(!photometric_compatible(Some(17.0), Some(18.0), 0.6));
    }

    #[test]
    fn missing_photometry_does_not_veto() {
        assert!(photometric_compatible(None, Some(17.0), 0.6));
        assert!(photometric_compatible(Some(17.0), None, 0.6));
        assert!(photometric_compatible(None, None, 0.6));
    }

    #[test]
    fn zero_threshold_disables_the_photometric_gate() {
        assert!(photometric_compatible(Some(10.0), Some(25.0), 0.0));
    }

    #[test]
    fn non_finite_magnitudes_do_not_veto() {
        assert!(photometric_compatible(Some(f64::NAN), Some(17.0), 0.6));
    }

    /// `(α, δ, α̇·cos δ, δ̇)` as [`attributable_at`] returns it.
    fn attr(ra_rate: f64, dec_rate: f64) -> (f64, f64, f64, f64) {
        (1.0, 0.5, ra_rate, dec_rate)
    }

    #[test]
    fn rates_agree_when_both_components_are_within_tolerance() {
        assert!(rates_compatible(
            attr(1e-3, 2e-3),
            attr(1.5e-3, 2.4e-3),
            1e-3
        ));
    }

    #[test]
    fn a_single_disagreeing_component_is_enough_to_reject() {
        // RA rate matches exactly, Dec rate is off by twice the tolerance:
        // agreement is required on both axes, not on their average.
        assert!(!rates_compatible(attr(1e-3, 0.0), attr(1e-3, 2e-3), 1e-3));
        assert!(!rates_compatible(attr(0.0, 1e-3), attr(2e-3, 1e-3), 1e-3));
    }

    #[test]
    fn the_tolerance_is_inclusive() {
        assert!(rates_compatible(attr(0.0, 0.0), attr(1e-3, 1e-3), 1e-3));
    }

    #[test]
    fn opposite_directions_do_not_cancel() {
        // |a - b| = 2e-3, not 0: signed rates must not be compared by magnitude.
        assert!(!rates_compatible(attr(1e-3, 0.0), attr(-1e-3, 0.0), 1e-3));
    }

    #[test]
    fn non_finite_rates_never_agree() {
        assert!(!rates_compatible(attr(f64::NAN, 0.0), attr(0.0, 0.0), 1e-3));
        assert!(!rates_compatible(
            attr(f64::INFINITY, 0.0),
            attr(0.0, 0.0),
            1e-3
        ));
    }

    #[test]
    fn union_find_groups_transitively() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        // 0-1 and 1-2 must place all three in one component: that transitivity
        // is the point of linkage, and also how one wrong link contaminates a
        // whole group.
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(3), uf.find(4));
    }

    #[test]
    fn union_find_is_idempotent() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 1);
        uf.union(0, 1);
        uf.union(1, 0);
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
    }

    #[test]
    fn merged_track_ids_orders_by_epoch_and_dedupes() {
        let epochs = |id: ObsId| match id {
            1 => Some(30.0),
            2 => Some(10.0),
            3 => Some(20.0),
            _ => None,
        };
        assert_eq!(merged_track_ids(&[1, 3], &[2, 3], epochs), vec![2, 3, 1]);
    }
}

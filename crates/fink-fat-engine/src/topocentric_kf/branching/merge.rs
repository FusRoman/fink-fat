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

use nalgebra::{Matrix2, Vector2, Vector3};
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsId, observation::Observation},
};

use crate::{
    spacetime_bucket::{
        healpix_binner::HealpixBinner,
        spatial_binner::{SpatialBinner, SpatialKey},
    },
    topocentric_kf::single_kalman::{KFState, update::wrap_angle},
};

/// Tuning for the linkage cascade. Each stage is disabled by its own
/// sentinel, following the same convention as the rest of the engine's
/// tuning knobs.
#[derive(Debug, Clone, Copy)]
pub struct MergeParams {
    /// Minimum number of observations an arc must carry to be linkable at
    /// all — see [`arc_is_converged`]. `0` disables the gate.
    ///
    /// This is the single most important knob of the cascade, because it acts
    /// on the *base rate* rather than on any one criterion. A 2-to-5-point arc
    /// has an unconstrained orbit: its `(a, e, i)` are noise, so it lands in an
    /// arbitrary index bucket — flooding the candidate space *and* missing its
    /// own true partner. Its covariance is enormous, so the Mahalanobis stage
    /// waves it through. Its absolute magnitude rests on a handful of samples.
    /// Such an arc is not linkable with any confidence, and proposing it costs
    /// precision everywhere.
    ///
    /// Measured on the first shadow run, the candidate space held 39.7 M pairs
    /// for 16 073 true ones — a 0.04 % prevalence at which even a 99.4 %
    /// specific cascade still yields ~232 k false merges. Shrinking the pool is
    /// the only lever that helps, and it is quadratic in the number of pairs.
    pub min_arc_points: usize,
    /// Max `|H_A − H_B|` (magnitudes) for two arcs to be photometrically
    /// compatible. `0.0` disables the stage. See the module docs for why
    /// this is an absolute threshold and not a σ-scaled one.
    pub max_delta_h: f64,
    /// Maximum **reduced** chi-square of the first predicted point — the
    /// primary discriminator. `0.0` disables the dynamical stage (which would
    /// leave linkage resting on photometry alone — never do this outside a
    /// diagnostic).
    ///
    /// Reduced, not raw, because it has a physical reference: each point
    /// contributes a `d² ~ χ²(2)`, so `d²/2` should sit near 1 when the filter
    /// is calibrated. That turns the threshold into a statement about fit
    /// quality rather than a tuned constant.
    ///
    /// Measured on a full run, the separation is total:
    ///
    /// | | p10 | p50 | p90 | p99 |
    /// |---|---|---|---|---|
    /// | correct pairs | 0.000 | 0.000 | 0.071 | **1.634** |
    /// | wrong pairs | **79.4** | 1337 | 4.0e5 | 4.1e7 |
    ///
    /// Any threshold in `[3, 79]` separates the two populations perfectly.
    /// Note the correct pairs sit *well below* 1: over a long propagation the
    /// process noise inflates the covariance far beyond the real error, so the
    /// filter is under-confident here. That is exactly why a gate on the *raw*
    /// `d²` never bit — thresholds of 9.21, 23 and 100 rejected identically —
    /// and why an absolute residual had to do the work instead.
    pub max_reduced_chi2: f64,
    /// Absolute cap (arcsec) on the prediction-to-observation miss, applied
    /// alongside `gate_chi2`. `0.0` disables it.
    ///
    /// The statistical gate alone is not enough: after a long gap the
    /// predicted box spans ~900 arcsec, so a 3σ test accepts anything within
    /// half a degree. An absolute bound asks the physical question the
    /// covariance cannot — *how far off was it, really?*
    pub max_residual_arcsec: f64,
    /// Refuse to link across a gap longer than this (days). `0.0` disables
    /// the limit. Long gaps are where a two-body propagation drifts and
    /// where two unrelated objects are most likely to look compatible.
    pub max_gap_days: f64,
}

impl Default for MergeParams {
    /// Everything off: linkage is opt-in.
    fn default() -> Self {
        Self {
            min_arc_points: 0,
            max_delta_h: 0.0,
            max_reduced_chi2: 0.0,
            max_residual_arcsec: 0.0,
            max_gap_days: 0.0,
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

    /// Whether the fit clears both bounds.
    ///
    /// The reduced chi-square is judged on the **first** point only — see
    /// [`Self::reduced_chi2_first`] for why the later ones are compromised by
    /// the sequential absorption. The absolute residual, which cannot be
    /// gamed that way, is still required of every point.
    ///
    /// `max_reduced_chi2 <= 0.0` disables the statistical test,
    /// `max_residual_arcsec <= 0.0` the absolute one. An empty fit never
    /// passes: a link must be demonstrated, not assumed.
    pub fn passes(&self, max_reduced_chi2: f64, max_residual_arcsec: f64) -> bool {
        if self.d2.is_empty() {
            return false;
        }
        if max_reduced_chi2 > 0.0
            && !self
                .reduced_chi2_first()
                .is_some_and(|chi2| chi2 <= max_reduced_chi2)
        {
            return false;
        }
        max_residual_arcsec <= 0.0
            || self
                .separation_arcsec
                .iter()
                .all(|sep| *sep <= max_residual_arcsec)
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
    let disjoint = temporally_disjoint(predictor_span, tested_span, params.max_gap_days)
        || temporally_disjoint(tested_span, predictor_span, params.max_gap_days);
    if !disjoint {
        return MergeVerdict::RejectedTemporal;
    }
    if !photometric_compatible(predictor_h, tested_h, params.max_delta_h) {
        return MergeVerdict::RejectedPhotometric;
    }
    if params.max_reduced_chi2 > 0.0 || params.max_residual_arcsec > 0.0 {
        match sequential_fit(predictor_state, tested_points) {
            Some(fit) if fit.passes(params.max_reduced_chi2, params.max_residual_arcsec) => {}
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

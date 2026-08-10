//! Candidate-filtering methods for the MOT candidate-contamination study.
//!
//! Every filter takes the candidates found inside a Kalman bank's error box
//! (plus a shared per-visit [`FilterContext`]) and returns a keep-mask over
//! them. `mot_analysis` evaluates each filter in *shadow mode* — every
//! filter sees the same raw candidate set, so their counters are directly
//! comparable — plus one configurable cascade whose surviving candidates
//! feed the before/after per-visit histograms.
//!
//! The hard constraint is recall: a filter that removes true observations
//! (`removed_true`) is a bad filter, whatever its bad-candidate rejection
//! rate. The per-filter counters exist precisely to measure that trade-off
//! before anything is ported into the engine's `filter_candidates`.

use fink_fat_engine::topocentric_kf::{
    branching::{
        candidate_search::CandidateMatch,
        detection_probability::{phase_correction, predicted_apparent_magnitude},
        llr_score::{observation_llr_delta, photometric_llr_delta},
    },
    kalman_bank::{KFBank, ellipse_region_finder::SearchRegion},
    single_kalman::update::wrap_angle,
};
use nalgebra::{Matrix2, Vector2};

pub const RAD_TO_ARCSEC: f64 = 206264.80624709636;

// ── Per-visit shared context ─────────────────────────────────────────────

/// Weighted-mean apparent motion of the predicted bank, reduced to a
/// tangent-plane direction and speed.
#[derive(Debug, Clone, Copy)]
pub struct Motion {
    /// Tangent-plane unit direction of motion `(dRA·cosδ, dDec)`.
    pub ux: f64,
    pub uy: f64,
    /// Tangent-plane angular speed (rad/day).
    pub speed_rad_day: f64,
    /// Weighted-mean declination used for the `cosδ` de-projection.
    pub mean_dec: f64,
}

/// Last real observation the lineage actually consumed — the anchor for the
/// observation-centric (OC-SORT-style) direction and rate filters.
#[derive(Debug, Clone, Copy)]
pub struct LastAccepted {
    pub epoch: f64,
    pub ra: f64,
    pub dec: f64,
}

/// Everything the filters need about one (lineage, visit) pair, computed
/// once per visit and shared by every filter of the suite.
pub struct FilterContext<'a> {
    pub region: &'a SearchRegion,
    pub visit_epoch: f64,
    /// Predicted apparent magnitude from the bank's running `H` estimate and
    /// its MAP hypothesis's predicted geometry — `None` before the first
    /// real update (no magnitude history yet).
    pub predicted_mag: Option<f64>,
    /// `None` when the mixture's motion direction is degenerate (zero norm).
    pub motion: Option<Motion>,
    /// 1-σ cross-track extent of the predicted mixture (arcsec), including
    /// both per-component innovation covariance and inter-component spread.
    pub sigma_cross_arcsec: Option<f64>,
    pub last_accepted: Option<LastAccepted>,
    pub n_hypotheses: usize,
    pub map_weight_fraction: f64,
    pub n_real_updates: usize,
    /// Local alert density (alerts/sr) around the region center — the
    /// clutter background `λ_clutter` used by [`TopKGate`]'s ranking score.
    /// See `local_clutter_density` in `fink-fat-engine`.
    pub clutter_density: f64,
    /// `NightAdvanceParams::photometric_sigma_mag` — same ranking score.
    pub photometric_sigma_mag: f64,
}

impl<'a> FilterContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        region: &'a SearchRegion,
        predicted_bank: &KFBank<'_, '_>,
        last_accepted: Option<LastAccepted>,
        visit_epoch: f64,
        n_real_updates: usize,
        clutter_density: f64,
        photometric_sigma_mag: f64,
    ) -> Self {
        let hypotheses = predicted_bank.hypotheses();
        let n_hypotheses = hypotheses.len();

        // Normalized weights (log-weights are unnormalized — see
        // `geometry_diagnostic`'s note in `mot_analysis`).
        let weights: Vec<f64> = hypotheses.iter().map(|h| h.log_weight.exp()).collect();
        let total_weight: f64 = weights.iter().sum();
        let map_weight_fraction = if total_weight > 0.0 {
            weights.iter().fold(0.0, |a: f64, &w| a.max(w)) / total_weight
        } else {
            0.0
        };

        let motion = (total_weight > 0.0)
            .then(|| {
                let (mut ra_dot, mut dec_dot, mut dec) = (0.0, 0.0, 0.0);
                for (w, h) in weights.iter().zip(hypotheses) {
                    ra_dot += w * h.kf.state[2];
                    dec_dot += w * h.kf.state[3];
                    dec += w * h.kf.state[1];
                }
                let (ra_dot, dec_dot, mean_dec) = (
                    ra_dot / total_weight,
                    dec_dot / total_weight,
                    dec / total_weight,
                );
                let vx = ra_dot * mean_dec.cos();
                let vy = dec_dot;
                let speed = (vx * vx + vy * vy).sqrt();
                (speed > 0.0).then_some(Motion {
                    ux: vx / speed,
                    uy: vy / speed,
                    speed_rad_day: speed,
                    mean_dec,
                })
            })
            .flatten();

        let predicted_mag = predicted_bank
            .absolute_magnitude_estimate()
            .zip(predicted_bank.best())
            .map(|(h_estimate, best)| {
                let helio = best.kf.to_cartesian().pos;
                let delta_topocentric_au = best.kf.state[4];
                // The running H has the phase term removed, so it has to be put
                // back at this geometry — comparing a phase-corrected H against
                // an uncorrected apparent magnitude would inject exactly the
                // systematic the correction exists to remove.
                predicted_apparent_magnitude(
                    h_estimate,
                    helio.norm(),
                    delta_topocentric_au,
                    phase_correction(
                        &helio,
                        &best.kf.r_obs,
                        predicted_bank.config.slope_parameter_g,
                    ),
                )
            });

        let sigma_cross_arcsec = motion.and_then(|m| sigma_cross_of_region(region, &m));

        Self {
            region,
            visit_epoch,
            predicted_mag,
            motion,
            sigma_cross_arcsec,
            last_accepted,
            n_hypotheses,
            map_weight_fraction,
            n_real_updates,
            clutter_density,
            photometric_sigma_mag,
        }
    }

    /// Tangent-plane offset (arcsec) of a sky position from the region
    /// center, using the region center's declination for the `cosδ`
    /// de-projection.
    fn tangent_offset_arcsec(&self, ra: f64, dec: f64) -> (f64, f64) {
        let dx = wrap_angle(ra - self.region.center_ra) * self.region.center_dec.cos();
        let dy = dec - self.region.center_dec;
        (dx * RAD_TO_ARCSEC, dy * RAD_TO_ARCSEC)
    }
}

/// 1-σ cross-track extent of the mixture (arcsec): weighted total of each
/// component's innovation covariance projected on the cross-track direction,
/// plus the cross-track spread of the component centers around the region
/// center (law of total variance).
fn sigma_cross_of_region(region: &SearchRegion, motion: &Motion) -> Option<f64> {
    if region.components.is_empty() {
        return None;
    }
    let cos_dec = region.center_dec.cos();
    // Cross-track projection of a tangent-plane displacement is
    // `-dx·uy + dy·ux` with `dx = ΔRA·cosδ, dy = ΔDec`; applied to a *raw*
    // (ΔRA, ΔDec) covariance that is the row vector `a = (-uy·cosδ, ux)`.
    let a = Vector2::new(-motion.uy * cos_dec, motion.ux);

    let mut total_weight = 0.0;
    let mut var = 0.0;
    for c in &region.components {
        let s: &Matrix2<f64> = &c.s;
        let comp_var = (a.transpose() * s * a)[(0, 0)];
        let cx = wrap_angle(c.center_ra - region.center_ra) * cos_dec;
        let cy = c.center_dec - region.center_dec;
        let comp_cross = -cx * motion.uy + cy * motion.ux;
        var += c.weight * (comp_var + comp_cross * comp_cross);
        total_weight += c.weight;
    }
    // Positive-form test so a NaN anywhere yields `None` (keep everything)
    // rather than a poisoned limit that would silently reject candidates.
    if total_weight > 0.0 && var > 0.0 {
        Some((var / total_weight).sqrt() * RAD_TO_ARCSEC)
    } else {
        None
    }
}

// ── The filter trait ─────────────────────────────────────────────────────

/// One candidate-filtering method: candidates in, keep-mask out.
///
/// A filter must be *conservative by default*: whenever the information it
/// needs is unavailable (no magnitude history, degenerate motion, no last
/// accepted observation…), it keeps everything rather than guessing.
pub trait CandidateFilter: Send + Sync {
    /// Display name, including the operating threshold (e.g. `photometric(1.0mag)`).
    fn name(&self) -> String;

    /// `true` = keep. Must return one entry per candidate.
    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool>;
}

// ── F1: photometric gate ─────────────────────────────────────────────────

/// Reject candidates whose apparent magnitude is further than
/// `max_delta_mag` from the lineage's predicted apparent magnitude
/// (running-`H` estimate + predicted geometry). Physics: an asteroid's
/// brightness varies slowly (distance/phase/lightcurve), while a chance
/// interloper has an unrelated magnitude.
pub struct PhotometricGate {
    pub max_delta_mag: f64,
}

impl CandidateFilter for PhotometricGate {
    fn name(&self) -> String {
        format!("photometric({}mag)", self.max_delta_mag)
    }

    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        let Some(predicted_mag) = ctx.predicted_mag else {
            return vec![true; candidates.len()];
        };
        candidates
            .iter()
            .map(|c| {
                let m = c.observation.photometry().magnitude;
                !m.is_finite() || (m - predicted_mag).abs() <= self.max_delta_mag
            })
            .collect()
    }
}

/// Shared "is the bank converged" test used by the filters that rely on the
/// bank's motion estimate being trustworthy — same thresholds as the
/// `confident` split in `mot_analysis`'s geometry diagnostics.
fn bank_confident(ctx: &FilterContext<'_>) -> bool {
    ctx.n_hypotheses <= 3 && ctx.map_weight_fraction >= 0.9 && ctx.n_real_updates >= 5
}

// ── F2: direction-consistency gate (OC-SORT's OCM) ───────────────────────

/// Reject candidates whose direction from the last accepted observation
/// disagrees with the bank's predicted motion direction. The angular
/// tolerance widens automatically at small separations, where astrometric
/// noise dominates the direction: `θ_eff = max(θ, atan2(tol, sep))`.
pub struct DirectionGate {
    pub max_angle_deg: f64,
    /// Positional tolerance (arcsec) folded into the effective angle — no
    /// direction judgement is possible below this separation.
    pub tol_arcsec: f64,
    /// Only act when [`bank_confident`] holds — the first shadow run showed
    /// the plain variant's `removed_TRUE` concentrated where the bank's
    /// motion estimate isn't trustworthy yet.
    pub confident_only: bool,
}

impl CandidateFilter for DirectionGate {
    fn name(&self) -> String {
        let suffix = if self.confident_only { ",conf" } else { "" };
        format!("direction({}deg{suffix})", self.max_angle_deg)
    }

    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        if self.confident_only && !bank_confident(ctx) {
            return vec![true; candidates.len()];
        }
        let (Some(motion), Some(last)) = (ctx.motion, ctx.last_accepted) else {
            return vec![true; candidates.len()];
        };
        let cos_dec = last.dec.cos();
        candidates
            .iter()
            .map(|c| {
                let coord = c.observation.equ_coord();
                let dx = wrap_angle(coord.ra - last.ra) * cos_dec * RAD_TO_ARCSEC;
                let dy = (coord.dec - last.dec) * RAD_TO_ARCSEC;
                let sep = (dx * dx + dy * dy).sqrt();
                if sep <= self.tol_arcsec {
                    return true;
                }
                let cos_angle = (dx * motion.ux + dy * motion.uy) / sep;
                let angle = cos_angle.clamp(-1.0, 1.0).acos();
                let theta_eff = self
                    .max_angle_deg
                    .to_radians()
                    .max((self.tol_arcsec / sep).atan());
                angle <= theta_eff
            })
            .collect()
    }
}

// ── F3: angular-rate plausibility gate ───────────────────────────────────

/// Reject candidates whose implied angular rate (separation from the last
/// accepted observation over the elapsed time) is incompatible with the
/// bank's predicted rate: keep iff
/// `expected/factor - tol <= sep <= expected*factor + tol`, where
/// `expected = speed·dt`. Complements [`DirectionGate`]: direction vs. norm.
pub struct RateGate {
    pub factor: f64,
    /// Absolute tolerance (arcsec) absorbing astrometric noise at short dt.
    pub tol_arcsec: f64,
    /// See [`DirectionGate::confident_only`] — same rationale.
    pub confident_only: bool,
}

impl CandidateFilter for RateGate {
    fn name(&self) -> String {
        let suffix = if self.confident_only { ",conf" } else { "" };
        format!("rate(x{}{suffix})", self.factor)
    }

    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        if self.confident_only && !bank_confident(ctx) {
            return vec![true; candidates.len()];
        }
        let (Some(motion), Some(last)) = (ctx.motion, ctx.last_accepted) else {
            return vec![true; candidates.len()];
        };
        let dt = ctx.visit_epoch - last.epoch;
        if dt <= 0.0 || !dt.is_finite() {
            return vec![true; candidates.len()];
        }
        let expected_arcsec = motion.speed_rad_day * dt * RAD_TO_ARCSEC;
        let lo = expected_arcsec / self.factor - self.tol_arcsec;
        let hi = expected_arcsec * self.factor + self.tol_arcsec;
        let cos_dec = last.dec.cos();
        candidates
            .iter()
            .map(|c| {
                let coord = c.observation.equ_coord();
                let dx = wrap_angle(coord.ra - last.ra) * cos_dec * RAD_TO_ARCSEC;
                let dy = (coord.dec - last.dec) * RAD_TO_ARCSEC;
                let sep = (dx * dx + dy * dy).sqrt();
                sep >= lo && sep <= hi
            })
            .collect()
    }
}

// ── F4: anisotropic cross-track gate ─────────────────────────────────────

/// Reject candidates further than `k_sigma · σ_cross` (floored at
/// `floor_arcsec`) from the predicted track line, perpendicular to the
/// motion direction. Physics: prediction error is along-track dominated
/// (rate/timing uncertainty), while clutter lands isotropically in the box.
pub struct CrossTrackGate {
    pub k_sigma: f64,
    pub floor_arcsec: f64,
}

impl CandidateFilter for CrossTrackGate {
    fn name(&self) -> String {
        format!("cross_track({}sig)", self.k_sigma)
    }

    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        let (Some(motion), Some(sigma_cross)) = (ctx.motion, ctx.sigma_cross_arcsec) else {
            return vec![true; candidates.len()];
        };
        let limit = (self.k_sigma * sigma_cross).max(self.floor_arcsec);
        candidates
            .iter()
            .map(|c| {
                let coord = c.observation.equ_coord();
                let (dx, dy) = ctx.tangent_offset_arcsec(coord.ra, coord.dec);
                let cross = -dx * motion.uy + dy * motion.ux;
                cross.abs() <= limit
            })
            .collect()
    }
}

// ── F5: best-relative likelihood gate (PKF's ambiguity check) ────────────

/// Keep only candidates whose mixture likelihood is within a factor `alpha`
/// of the visit's best candidate. Directly reduces branching; the known
/// failure mode (an interloper closer to the predicted center than the true
/// point) is exactly what `removed_true` measures.
pub struct RelativeLikelihoodGate {
    pub alpha: f64,
}

impl CandidateFilter for RelativeLikelihoodGate {
    fn name(&self) -> String {
        format!("rel_likelihood({:.0e})", self.alpha)
    }

    fn keep_mask(&self, _ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        // `f64::max` ignores NaN, so `l_max` is 0.0 (not NaN) on an empty or
        // degenerate likelihood set — the early return keeps everything.
        let l_max = candidates.iter().map(|c| c.likelihood).fold(0.0, f64::max);
        if l_max <= 0.0 {
            return vec![true; candidates.len()];
        }
        let threshold = self.alpha * l_max;
        candidates
            .iter()
            .map(|c| c.likelihood >= threshold)
            .collect()
    }
}

// ── F7: rank-and-cap by combined LLR score ───────────────────────────────

/// Hard cap on candidates per visit: score every candidate with the same
/// combined log-likelihood-ratio the engine already uses to rank *branches*
/// downstream (`observation_llr_delta` + `photometric_llr_delta`, see
/// `orchestrate::spawn_branches_for_lineage` and
/// `pruning::cap_top_b_per_lineage`, which sorts by this exact quantity),
/// and keep only the `k` best — applied one stage earlier, on raw
/// candidates, before any branch exists. Unlike the boolean gates above,
/// this guarantees `keep_mask` never returns more than `k` `true`s, so a
/// visit's surviving candidate count is capped by construction once this
/// filter is ANDed into a cascade.
pub struct TopKGate {
    pub k: usize,
}

impl CandidateFilter for TopKGate {
    fn name(&self) -> String {
        format!("top_k({})", self.k)
    }

    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        if candidates.len() <= self.k {
            return vec![true; candidates.len()];
        }
        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let score = observation_llr_delta(c.likelihood, ctx.clutter_density)
                    + photometric_llr_delta(
                        ctx.predicted_mag,
                        c.observation.photometry().magnitude,
                        ctx.photometric_sigma_mag,
                    );
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut mask = vec![false; candidates.len()];
        for &(i, _) in scored.iter().take(self.k) {
            mask[i] = true;
        }
        mask
    }
}

// ── F6: confidence-adaptive chi-square gate ──────────────────────────────

/// Once the bank is converged (few hypotheses, dominant MAP weight, enough
/// real updates), tighten the effective Mahalanobis gate from the engine's
/// generous `gate_chi2` (χ²=23 ≈ 5σ) down to `chi2_confident`. Unconverged
/// banks keep everything — their error boxes are legitimately wide.
pub struct AdaptiveChi2Gate {
    pub chi2_confident: f64,
    pub max_hypotheses: usize,
    pub min_map_weight: f64,
    pub min_real_updates: usize,
}

impl CandidateFilter for AdaptiveChi2Gate {
    fn name(&self) -> String {
        format!("adaptive_chi2({})", self.chi2_confident)
    }

    fn keep_mask(&self, ctx: &FilterContext<'_>, candidates: &[CandidateMatch<'_>]) -> Vec<bool> {
        let confident = ctx.n_hypotheses <= self.max_hypotheses
            && ctx.map_weight_fraction >= self.min_map_weight
            && ctx.n_real_updates >= self.min_real_updates;
        if !confident {
            return vec![true; candidates.len()];
        }
        candidates
            .iter()
            .map(|c| {
                let coord = c.observation.equ_coord();
                ctx.region
                    .any_component_contains(coord.ra, coord.dec, self.chi2_confident)
            })
            .collect()
    }
}

// ── Default suite and cascade ────────────────────────────────────────────

/// The shadow suite, refined after the first dataset-wide run (2026-08-03).
///
/// That run's verdict at the initial thresholds (`true_lost%` /
/// `bad_removed%`): cross-track 5σ 0.001/5.3, rel-likelihood 1e-4 0.003/19.8
/// and 1e-2 0.02/41.2, photometric 2mag 0.56/9.0 — all clear keepers;
/// direction and rate lost 1.5–3.3 % of true observations (concentrated,
/// per hypothesis, in unconverged banks — hence the `confident_only`
/// re-test); adaptive-χ² was rejected outright despite 0.86 % overall
/// because its losses hit NEOs (157 true NEO removals).
///
/// Round 3 (2026-08-04) adds [`TopKGate`]: after round 2's cascade, the
/// per-visit candidate count is much smaller but still has a long tail
/// (up to 40). `TopKGate` bounds it by construction — swept at k ∈
/// {5, 8, 12}, with k=5 (the user's target) placed in the cascade; its
/// `removed_true`/population breakdown will show whether that cap is safe
/// or needs loosening to 8/12.
pub fn default_filter_suite() -> Vec<Box<dyn CandidateFilter>> {
    vec![
        // Photometric: 2mag was the family's best trade-off; sweep upward.
        Box::new(PhotometricGate { max_delta_mag: 1.5 }),
        Box::new(PhotometricGate { max_delta_mag: 2.0 }),
        Box::new(PhotometricGate { max_delta_mag: 2.5 }),
        Box::new(PhotometricGate { max_delta_mag: 3.0 }),
        // Direction/rate: confident-only rescue attempt.
        Box::new(DirectionGate {
            max_angle_deg: 45.0,
            tol_arcsec: 3.0,
            confident_only: true,
        }),
        Box::new(DirectionGate {
            max_angle_deg: 60.0,
            tol_arcsec: 3.0,
            confident_only: true,
        }),
        Box::new(RateGate {
            factor: 2.0,
            tol_arcsec: 3.0,
            confident_only: true,
        }),
        Box::new(RateGate {
            factor: 3.0,
            tol_arcsec: 3.0,
            confident_only: true,
        }),
        // Cross-track: refine between the 3σ and 5σ operating points.
        Box::new(CrossTrackGate {
            k_sigma: 3.0,
            floor_arcsec: 2.0,
        }),
        Box::new(CrossTrackGate {
            k_sigma: 4.0,
            floor_arcsec: 2.0,
        }),
        Box::new(CrossTrackGate {
            k_sigma: 5.0,
            floor_arcsec: 2.0,
        }),
        // Relative likelihood: refine between 1e-4 and 1e-2.
        Box::new(RelativeLikelihoodGate { alpha: 1e-4 }),
        Box::new(RelativeLikelihoodGate { alpha: 1e-3 }),
        Box::new(RelativeLikelihoodGate { alpha: 1e-2 }),
        // Adaptive χ²: kept in shadow only (NEO-costly), at looser levels.
        Box::new(AdaptiveChi2Gate {
            chi2_confident: 13.8,
            max_hypotheses: 3,
            min_map_weight: 0.9,
            min_real_updates: 5,
        }),
        Box::new(AdaptiveChi2Gate {
            chi2_confident: 11.83,
            max_hypotheses: 3,
            min_map_weight: 0.9,
            min_real_updates: 5,
        }),
        // Rank-and-cap: bounds the per-visit candidate count by construction
        // (see TopKGate's doc). k=5 is the user's target and goes in the
        // cascade; 8/12 are shadow-only fallbacks if k=5 costs too much
        // recall.
        Box::new(TopKGate { k: 5 }),
        Box::new(TopKGate { k: 8 }),
        Box::new(TopKGate { k: 12 }),
    ]
}

/// Indices into [`default_filter_suite`] forming the default cascade — the
/// round-2 near-zero-recall-loss winners (photometric(2mag) ∧
/// cross_track(5σ) ∧ rel_likelihood(1e-2)) plus round-3's `top_k(5)`, which
/// caps the surviving count at 5 by construction regardless of how the
/// other three behave. Expected from the shadow table: ≳50 % of bad
/// candidates removed and no visit above 5 candidates after cascade — watch
/// `top_k(5)`'s own `true_lost%`/population breakdown for the actual
/// recall cost of the cap itself.
pub const DEFAULT_CASCADE: &[usize] = &[1, 10, 13, 16];

// ── Shadow-mode accumulation ─────────────────────────────────────────────

/// Dataset-wide counters for one filter (or the cascade).
#[derive(Debug, Default, Clone)]
pub struct FilterCounters {
    pub cand_in: u64,
    pub removed: u64,
    pub removed_bad: u64,
    pub removed_true: u64,
}

impl FilterCounters {
    pub fn absorb(&mut self, other: &FilterCounters) {
        self.cand_in += other.cand_in;
        self.removed += other.removed;
        self.removed_bad += other.removed_bad;
        self.removed_true += other.removed_true;
    }

    fn record(&mut self, mask: &[bool], is_true: &[bool]) {
        self.cand_in += mask.len() as u64;
        for (keep, &truth) in mask.iter().zip(is_true) {
            if !keep {
                self.removed += 1;
                if truth {
                    self.removed_true += 1;
                } else {
                    self.removed_bad += 1;
                }
            }
        }
    }
}

/// Per-trajectory shadow counters — one slot per suite member, plus the
/// cascade. Merged dataset-wide at reporting time.
#[derive(Debug, Default, Clone)]
pub struct ShadowStats {
    pub per_filter: Vec<FilterCounters>,
    pub cascade: FilterCounters,
}

impl ShadowStats {
    pub fn new(n_filters: usize) -> Self {
        Self {
            per_filter: vec![FilterCounters::default(); n_filters],
            cascade: FilterCounters::default(),
        }
    }

    pub fn absorb(&mut self, other: &ShadowStats) {
        if self.per_filter.len() < other.per_filter.len() {
            self.per_filter
                .resize(other.per_filter.len(), FilterCounters::default());
        }
        for (mine, theirs) in self.per_filter.iter_mut().zip(&other.per_filter) {
            mine.absorb(theirs);
        }
        self.cascade.absorb(&other.cascade);
    }
}

/// Evaluate the whole suite in shadow mode on one visit's candidates,
/// accumulate the counters, and return the cascade's keep-mask (logical AND
/// of `cascade` members' masks).
pub fn evaluate_filters(
    suite: &[Box<dyn CandidateFilter>],
    cascade: &[usize],
    ctx: &FilterContext<'_>,
    candidates: &[CandidateMatch<'_>],
    is_true: &[bool],
    shadow: &mut ShadowStats,
) -> Vec<bool> {
    let mut cascade_mask = vec![true; candidates.len()];
    for (i, filter) in suite.iter().enumerate() {
        let mask = filter.keep_mask(ctx, candidates);
        debug_assert_eq!(mask.len(), candidates.len());
        shadow.per_filter[i].record(&mask, is_true);
        if cascade.contains(&i) {
            for (m, keep) in cascade_mask.iter_mut().zip(&mask) {
                *m &= keep;
            }
        }
    }
    shadow.cascade.record(&cascade_mask, is_true);
    cascade_mask
}

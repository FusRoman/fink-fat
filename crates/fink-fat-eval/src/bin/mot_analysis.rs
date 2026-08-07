//! Multi Object Tracking (MOT) candidate-contamination study.
//!
//! For every trajectory the topocentric Kalman filter bank can actually
//! track (bootstrap succeeds, at least one processable observation), this
//! scans the dataset night by night and counts how many observations fall
//! inside each visit's predicted search region — split into the true
//! observation (if present) and "bad candidates" (everything else, i.e.
//! contamination the association step would have to disambiguate).
//!
//! Prints a dataset-wide aggregate report plus a detailed breakdown of the
//! worst trajectories (highest mean bad-candidates-per-visit).

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use ahash::AHashMap;
use anyhow::Result;
use clap::Parser;
use fink_fat_engine::{
    engine_config::{
        EngineConfig, kalman_context::KalmanContext, kf_bank_config::KFBankConfig,
        night_advance_params::NightAdvanceParams,
    },
    error::{EngineError, FinkFatError},
    spacetime_bucket::{
        bucket::{BucketIndex, build_alert_bucket_index},
        clutter_density::local_clutter_density,
        healpix_binner::HealpixBinner,
    },
    topocentric_kf::{
        branching::{
            Branch,
            candidate_search::{
                SingleBinTimeBinner, find_candidates_for_bank,
                find_candidates_for_bank_multi_region,
            },
            orchestrate::{PrefilterAnchor, lineage_might_be_in_visit},
            visit::{Visit, group_observations_into_visits},
        },
        kalman_bank::{KFBank, ellipse_region_finder::cover_if_clamped},
        single_kalman::{KFState, update::wrap_angle},
    },
};
use fink_fat_eval::{
    candidate_filters::{
        CandidateFilter, DEFAULT_CASCADE, FilterContext, LastAccepted, ShadowStats,
        default_filter_suite, evaluate_filters,
    },
    cli::{Cli, load_data},
    kalman_traj::{
        ObserverGeometryCache, compute_predictive_nis, compute_search_region, dedupe_by_epoch,
        init_bank_from_first_pair,
    },
    population::Population,
    trajectory_processing::{MetricStats, fmt_stats, materialize_contiguous_traj, metric_stats},
};
use indicatif::{ProgressBar, ProgressStyle};
use photom::{
    NightId, TrajId,
    coordinates::equatorial::EquCoord,
    observation_dataset::{
        ObsDataset, ObsId, iter::MemLayoutObservations, observation::Observation,
    },
};
use rayon::prelude::*;

/// Number of worst trajectories to report in detail.
const N_WORST: usize = 3;

// ── Generic statistics helpers ───────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<usize>,
}

#[derive(Default, Debug)]
pub struct AggStat<T> {
    pub min: T,
    pub max: T,
    pub median: T,
    pub mean: f64,
    pub std: f64,
    pub histogram: Histogram,
}

impl<T> AggStat<T>
where
    T: Copy + Ord + Into<f64>,
{
    pub fn from_sample(sample: &[T], n_bins: Option<usize>) -> Self {
        let n_bins = n_bins.unwrap_or(50);

        assert!(!sample.is_empty(), "sample must not be empty");
        assert!(n_bins > 0, "n_bins must be > 0");

        let min = *sample.iter().min().unwrap();
        let max = *sample.iter().max().unwrap();

        // mean / std (Welford)
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut count = 0.0;
        for &x in sample {
            let x: f64 = x.into();
            count += 1.0;
            let delta = x - mean;
            mean += delta / count;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        let std = (m2 / count).sqrt();

        // median (quickselect)
        let mut buf = sample.to_vec();
        let n = buf.len();
        let mid = n / 2;
        buf.select_nth_unstable(mid);
        let hi = buf[mid];
        let median = if n % 2 == 1 {
            hi
        } else {
            *buf[..mid].iter().max().unwrap()
        };

        // histogram
        let histogram = Self::histogram(sample, min.into(), max.into(), n_bins);

        Self {
            min,
            max,
            median,
            mean,
            std,
            histogram,
        }
    }

    fn histogram(sample: &[T], lo: f64, hi: f64, n_bins: usize) -> Histogram {
        let width = (hi - lo) / n_bins as f64;

        let edges: Vec<f64> = (0..=n_bins).map(|i| lo + width * i as f64).collect();

        let mut counts = vec![0usize; n_bins];

        if width == 0.0 {
            counts[0] = sample.len();
            return Histogram { edges, counts };
        }

        for &x in sample {
            let x: f64 = x.into();
            let idx = (((x - lo) / width) as usize).min(n_bins - 1);
            counts[idx] += 1;
        }

        Histogram { edges, counts }
    }
}

use std::fmt;

impl fmt::Display for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_count = self.counts.iter().copied().max().unwrap_or(0);
        const BAR_WIDTH: usize = 30;

        for (i, &count) in self.counts.iter().enumerate() {
            let lo = self.edges[i];
            let hi = self.edges[i + 1];
            let bar_len = if max_count == 0 {
                0
            } else {
                count * BAR_WIDTH / max_count
            };
            let bar = "█".repeat(bar_len);
            writeln!(
                f,
                "  [{lo:>8.2}, {hi:>8.2})  {bar:<width$} {count}",
                width = BAR_WIDTH
            )?;
        }
        Ok(())
    }
}

impl<T: fmt::Display> fmt::Display for AggStat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "min={}  max={}  median={}  mean={:.3}  std={:.3}",
            self.min, self.max, self.median, self.mean, self.std
        )?;
        write!(f, "{}", self.histogram)
    }
}

// ── Per-visit / per-trajectory stats ─────────────────────────────────────

#[derive(Debug, Default)]
pub struct VisitStat {
    night_id: NightId,
    visit_index: u32,
    true_in_visit: bool,
    n_candidates: u32,
    n_bad: u32,
    /// Candidates surviving the default filter cascade (see
    /// [`fink_fat_eval::candidate_filters`]) — the "after" side of the
    /// before/after per-visit histograms.
    n_candidates_cascade: u32,
    n_bad_cascade: u32,
    /// Whether the true observation (if `true_in_visit`) survived the
    /// cascade — `false` here on a `true_in_visit` visit is a recall loss.
    true_in_visit_cascade: bool,
}

impl VisitStat {
    /// Build from the per-candidate ground-truth flags and the cascade's
    /// keep-mask (both aligned with the visit's candidate list).
    pub fn from_masks(
        night_id: NightId,
        visit_index: usize,
        is_true: &[bool],
        cascade_mask: &[bool],
    ) -> Self {
        let n_candidates = is_true.len() as u32;
        let n_true = is_true.iter().filter(|&&t| t).count() as u32;
        let mut n_candidates_cascade = 0u32;
        let mut n_true_cascade = 0u32;
        for (&truth, &keep) in is_true.iter().zip(cascade_mask) {
            if keep {
                n_candidates_cascade += 1;
                if truth {
                    n_true_cascade += 1;
                }
            }
        }
        Self {
            night_id,
            visit_index: visit_index as u32,
            true_in_visit: n_true > 0,
            n_candidates,
            n_bad: n_candidates - n_true,
            n_candidates_cascade,
            n_bad_cascade: n_candidates_cascade - n_true_cascade,
            true_in_visit_cascade: n_true_cascade > 0,
        }
    }
}

/// Why a specific real observation of a trajectory was (or wasn't)
/// recovered by the MOT candidate search — see [`study_kalman_asteroid`]'s
/// per-night classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MissReason {
    /// Recovered: its `ObsId` turned up in some visit's `CandidateMatch`es.
    Found,
    /// Every visit of that night failed `lineage_might_be_in_visit`'s coarse
    /// prefilter (fixed-radius extrapolation), so no candidate search was
    /// even attempted for the visit that actually contains this observation.
    NoVisitPassedPrefilter,
    /// The visit containing this observation passed the prefilter and was
    /// searched, but the observation didn't pass `find_candidates_for_bank`'s
    /// gate (`gate_chi2`/`likelihood_threshold`).
    SearchedButNotMatched,
    /// `geometry_cache.get` failed for this observation's night — no
    /// prediction/search could be attempted at all.
    GeometryUnavailable,
    /// Default: the trajectory's walk never reached this observation at all
    /// (the loop ended early — see the `Branch::from_observation` failure
    /// path in [`study_kalman_asteroid`] — or the dataset's own night list
    /// ran out first).
    NotReached,
}

impl MissReason {
    fn label(self) -> &'static str {
        match self {
            MissReason::Found => "found (recovered by candidate search)",
            MissReason::NoVisitPassedPrefilter => "no visit passed the coarse prefilter",
            MissReason::SearchedButNotMatched => "visit searched, candidate not matched",
            MissReason::GeometryUnavailable => "geometry/ephemeris unavailable",
            MissReason::NotReached => "never reached (trajectory truncated)",
        }
    }

    /// All variants, in the fixed order used for reporting.
    fn all() -> [MissReason; 5] {
        [
            MissReason::Found,
            MissReason::NoVisitPassedPrefilter,
            MissReason::SearchedButNotMatched,
            MissReason::GeometryUnavailable,
            MissReason::NotReached,
        ]
    }

    /// Index into a `[T; 5]` bucket aligned with [`Self::all`].
    fn bucket_index(self) -> usize {
        match self {
            MissReason::Found => 0,
            MissReason::NoVisitPassedPrefilter => 1,
            MissReason::SearchedButNotMatched => 2,
            MissReason::GeometryUnavailable => 3,
            MissReason::NotReached => 4,
        }
    }
}

/// Classification of one real observation, plus how the Kalman lineage's
/// *canonical* (`branch`, not the per-visit search copy) prediction compared
/// to the truth at that epoch — the geometric "why" behind [`MissReason`]:
/// was the true point just outside a reasonably-sized box (marginal), or was
/// the prediction wildly off (genuine divergence)?
#[derive(Debug, Clone)]
struct ObsOutcome {
    reason: MissReason,
    /// Days since `branch` was last updated with a real observation. `NaN`
    /// for [`MissReason::NotReached`] (never classified).
    dt_since_last_real_update: f64,
    /// Predictive NIS (χ²(2) if well-calibrated) of the true observation
    /// against `branch`'s predicted mixture at this epoch. `None` when no
    /// prediction could be computed (`MissReason::GeometryUnavailable`/
    /// `NotReached`, or a degenerate mixture).
    nis: Option<f64>,
    /// Days since `branch`'s bootstrap pair (fixed, unlike
    /// `dt_since_last_real_update`) — the length of the *total* two-body
    /// arc propagated so far, as opposed to just the last gap. `NaN` for
    /// `MissReason::NotReached`.
    arc_span_since_bootstrap_days: f64,
    /// Angular separation (arcsec) between the predicted search region's
    /// center and the true observation's actual sky position.
    separation_arcsec: Option<f64>,
    /// The predicted search region's conservative bounding radius (arcsec)
    /// at this epoch — how big the "box" was.
    region_radius_arcsec: Option<f64>,
    /// Truth-vs-prediction error projected along the mixture's weighted
    /// apparent-motion direction — see [`GeometryDiagnostic`].
    along_track_arcsec: Option<f64>,
    /// Truth-vs-prediction error projected perpendicular to the motion
    /// direction — see [`GeometryDiagnostic`].
    cross_track_arcsec: Option<f64>,
    /// Number of surviving hypotheses in the predicted bank — see
    /// [`GeometryDiagnostic::n_hypotheses`].
    n_hypotheses: Option<usize>,
    /// MAP hypothesis's weight fraction — see
    /// [`GeometryDiagnostic::map_weight_fraction`].
    map_weight_fraction: Option<f64>,
    /// How many real observations `branch` had already consumed *before*
    /// this one (`Branch::n_real_updates` at classification time) — lets
    /// the report distinguish "not enough points yet to converge" (misses
    /// concentrated at low values) from "diverges even with an
    /// already-constrained orbit" (misses spread across high values too).
    n_real_updates_so_far: usize,
}

impl ObsOutcome {
    fn not_reached() -> Self {
        Self {
            reason: MissReason::NotReached,
            dt_since_last_real_update: f64::NAN,
            arc_span_since_bootstrap_days: f64::NAN,
            nis: None,
            separation_arcsec: None,
            region_radius_arcsec: None,
            along_track_arcsec: None,
            cross_track_arcsec: None,
            n_hypotheses: None,
            map_weight_fraction: None,
            n_real_updates_so_far: 0,
        }
    }
}

/// Why the trajectory's real (`branch`) lineage walk stopped where it did
/// — mirrors `kalman_traj::TrajStopReason`'s gating/propagation split, but
/// for the `Branch::from_observation` path used here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TruncationReason {
    /// Default: the walk consumed every real observation of the trajectory.
    ReachedEnd,
    /// `Branch::from_observation` collapsed to zero hypotheses because every
    /// one was rejected by the chi-square gate (`n_gated > 0`, `n_failed ==
    /// 0`) — the true observation was a statistical outlier for the bank at
    /// that point (real disagreement), not a numerical failure.
    CollapsedByGating,
    /// ...because every hypothesis failed numerically instead (propagation/
    /// innovation/update failure, `n_failed > 0`, `n_gated == 0`) — a
    /// solver/geometry edge case, not a statistical disagreement.
    CollapsedByPropagation,
    /// Neither counter alone explains the collapse (both gated and failed
    /// contributed, or neither did) — defensive fallback.
    DegenerateState,
}

impl TruncationReason {
    fn label(self) -> &'static str {
        match self {
            TruncationReason::ReachedEnd => "reached end",
            TruncationReason::CollapsedByGating => "collapsed (gating — real outlier)",
            TruncationReason::CollapsedByPropagation => {
                "collapsed (propagation failure — numerical)"
            }
            TruncationReason::DegenerateState => "collapsed (degenerate state)",
        }
    }

    fn all() -> [TruncationReason; 4] {
        [
            TruncationReason::ReachedEnd,
            TruncationReason::CollapsedByGating,
            TruncationReason::CollapsedByPropagation,
            TruncationReason::DegenerateState,
        ]
    }

    /// Index into a `[T; 4]` bucket aligned with [`Self::all`].
    fn bucket_index(self) -> usize {
        match self {
            TruncationReason::ReachedEnd => 0,
            TruncationReason::CollapsedByGating => 1,
            TruncationReason::CollapsedByPropagation => 2,
            TruncationReason::DegenerateState => 3,
        }
    }

    /// Classify a collapsed `Branch::from_observation_diag` call from its
    /// `(n_gated, n_failed)` counts — same priority rule as
    /// `kalman_traj::TrajStopReason`'s classification of `BankStep`.
    fn from_gate_fail_counts(n_gated: usize, n_failed: usize) -> Self {
        match (n_gated, n_failed) {
            (0, f) if f > 0 => TruncationReason::CollapsedByPropagation,
            (g, _) if g > 0 => TruncationReason::CollapsedByGating,
            _ => TruncationReason::DegenerateState,
        }
    }
}

/// JSON-serializable mirror of `KFState`'s existing (`rkyv`-only)
/// `KFStateSnapshot` — same flat fields, built via `.to_snapshot()`, just
/// re-exposed with `serde::Serialize` so a failure case can be dumped to
/// disk without touching `fink-fat-engine`.
#[derive(Debug, Clone, serde::Serialize)]
struct KFStateDump {
    state: [f64; 6],
    /// 6x6 covariance, column-major (36 entries) — `Vec` rather than a
    /// fixed array since `serde`'s array impl doesn't cover this length.
    covariance: Vec<f64>,
    epoch: f64,
    r_obs: [f64; 3],
    v_obs: [f64; 3],
    universal_anomaly: Option<f64>,
    /// 3x4 Kalman gain (12 entries) — see `covariance`'s doc for why `Vec`.
    kalman_gain: Option<Vec<f64>>,
    nis_ema: Option<f64>,
}

impl From<&KFState<'_>> for KFStateDump {
    fn from(kf: &KFState<'_>) -> Self {
        let s = kf.to_snapshot();
        Self {
            state: s.state,
            covariance: s.covariance.to_vec(),
            epoch: s.epoch,
            r_obs: s.r_obs,
            v_obs: s.v_obs,
            universal_anomaly: s.universal_anomaly,
            kalman_gain: s.kalman_gain.map(|k| k.to_vec()),
            nis_ema: s.nis_ema,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct HypothesisDump {
    kf: KFStateDump,
    log_weight: f64,
    id: u64,
}

impl From<&fink_fat_engine::topocentric_kf::kalman_bank::hypothesis::Hypothesis<'_>>
    for HypothesisDump
{
    fn from(h: &fink_fat_engine::topocentric_kf::kalman_bank::hypothesis::Hypothesis<'_>) -> Self {
        Self {
            kf: (&h.kf).into(),
            log_weight: h.log_weight,
            id: h.id,
        }
    }
}

/// Everything needed to reconstruct and replay, standalone, a
/// `predicted_bank.branch_with_diag(obs)` call that collapsed the bank to
/// zero hypotheses — a reproducible case for a regression test. Captured at
/// the moment `Branch::from_observation_diag` returns `None` in the
/// "god-mode" advance block of `study_kalman_asteroid`.
#[derive(Debug, Clone, serde::Serialize)]
struct FailureCaseDump {
    traj_id: String,
    night_id: u32,
    step: usize,
    n_real_updates_so_far: usize,
    n_gated: usize,
    n_failed: usize,
    /// The candidate real observation `branch_with_diag` was called with.
    observation: Observation,
    /// `predicted_bank.hypotheses()` — the bank's full hypothesis
    /// population *before* the failing call. An empty list here means the
    /// bank had already collapsed upstream (e.g. in `predict_to`) rather
    /// than during this call itself — see `pre_predict_hypotheses` below
    /// for the state that actually produced that collapse.
    hypotheses: Vec<HypothesisDump>,
    /// `branch.bank.hypotheses()` just *before* the `predict_to(predict_epoch,
    /// predict_r_obs, predict_v_obs)` call that produced `predicted_bank`
    /// (and, for `DegenerateState`, produced it *empty*). Replaying
    /// `KFState::predict` on each of these against the same epoch/observer
    /// geometry reveals which `PropagateError` variant dropped it (Kepler
    /// solver failure, singular Jacobian, or ρ leaving the sane range) —
    /// `predict_hypotheses` (`KFBank`) silently filters out failures, so
    /// `hypotheses`/hypotheses-after never shows this.
    pre_predict_hypotheses: Vec<HypothesisDump>,
    predict_epoch: f64,
    predict_r_obs: [f64; 3],
    predict_v_obs: [f64; 3],
    bank_config: serde_json::Value,
    /// One entry per real update this trajectory's lineage successfully
    /// consumed *before* the failure, in order — the best (MAP) hypothesis's
    /// state right after each update. Lets a replay tool see the actual
    /// trend (did ρ/ρ̇ drift gradually or jump suddenly? did it start once
    /// the bank had already collapsed to very few hypotheses, i.e. the
    /// chi-square gate's `is_protected` exemption had made every survivor
    /// ungateable?) instead of only the final, already-diverged state.
    state_history: Vec<StateHistoryPoint>,
}

/// One real update's post-update state — see [`FailureCaseDump::state_history`].
#[derive(Debug, Clone, serde::Serialize)]
struct StateHistoryPoint {
    step: usize,
    epoch: f64,
    n_real_updates: usize,
    dt_since_last_real_update: f64,
    /// Number of live hypotheses in the bank right after this update — once
    /// this drops to `min_hypotheses` (often 1), every survivor is
    /// trivially in the gate's protected top-`min_hypotheses` set and can
    /// never be gated out again regardless of statistical consistency.
    n_hypotheses: usize,
    /// Number of live hypotheses in `predicted_bank`, i.e. the population
    /// *before* this update's gate/prune — compare against `n_hypotheses`
    /// (the count *after*) to see whether a collapse happened in this one
    /// step, and against `rho_min_before`/`rho_max_before` to see whether
    /// that population still had a wide ρ spread right before collapsing.
    n_hyp_before: usize,
    /// Min/max ρ (AU) across every hypothesis in `predicted_bank`, i.e.
    /// immediately before this update's gate/prune — the spread the
    /// survivor was drawn from.
    rho_min_before: f64,
    rho_max_before: f64,
    state: [f64; 6],
    /// `covariance[(4,4)]` — Var(ρ) right after this update. A value that's
    /// already tiny before ρ starts diverging points at an underestimated
    /// posterior (mode-collapse-like loss of uncertainty during pruning)
    /// rather than a bad single-step correction.
    p_rho_au2: f64,
    /// `covariance[(5,5)]` — Var(ρ̇), same rationale as `p_rho_au2`.
    p_rho_dot_au2: f64,
    /// Frobenius norm of the Kalman gain used in this update (0.0 if this
    /// state was never updated, which shouldn't happen here since every
    /// entry follows a real update). A norm spiking only at the diverging
    /// step (while `p_rho_au2` stays reasonable) points at a bad gain/
    /// Jacobian computation on that specific step rather than an
    /// underestimated prior.
    kalman_gain_norm: f64,
    /// Exponentially-smoothed NIS carried on the state. A gradual climb
    /// leading up to the jump is consistent with a genuine (if uncorrected)
    /// statistical drift rather than a single-step numerical bug.
    nis_ema: f64,
    /// `covariance[(4,0)]` — Cov(ρ, α). The Kalman gain's ρ row is
    /// `K[4,:] = P[4, 0:2] · S⁻¹`: it depends on this cross-term, not on
    /// `p_rho_au2` (Var(ρ)) alone — a modest `p_rho_au2` next to a huge
    /// `kalman_gain_norm` only makes sense if this (or the ρ↔δ term) is
    /// itself disproportionately large.
    p_cross_rho_ra: f64,
    /// Largest absolute entry of `covariance - covariance.transpose()` —
    /// should be exactly 0 for a mathematically valid covariance. Nothing
    /// in the propagate/update/merge pipeline symmetrizes the full 6×6 `P`
    /// (only the 2×2 innovation covariance `S` is, in
    /// `regularize_covariance_2x2`) — a nonzero value here is a direct
    /// numerical-bug signal, not just a modeling limitation.
    p_asymmetry: f64,
    /// `covariance[(0,0)]` — Var(α). Together with `p_rho_au2` and
    /// `p_cross_rho_ra` lets a consumer compute the actual ρ↔α correlation
    /// coefficient `p_cross_rho_ra / sqrt(p_rho_au2 * p_ra)` — `|corr| > 1`
    /// would mean `P` is not a mathematically valid covariance (violates
    /// Cauchy-Schwarz), a real numerical bug distinct from mere asymmetry.
    p_ra: f64,
    /// `covariance[(1,1)]` — Var(δ), same rationale as `p_ra` for the ρ↔δ pair.
    p_dec: f64,
    /// `covariance[(4,1)]` — Cov(ρ, δ), the ρ↔δ counterpart of `p_cross_rho_ra`.
    p_cross_rho_dec: f64,
    /// `covariance[(2,2)]` — Var(α̇). Checked for the same collapse-below-
    /// astrometric-noise pattern as `p_ra`/`p_dec` — if confirmed, it would
    /// warrant its own floor (`min_angular_rate_variance_ratio`) alongside
    /// `min_angular_variance_ratio`.
    p_ra_dot: f64,
    /// `covariance[(3,3)]` — Var(δ̇), same rationale as `p_ra_dot`.
    p_dec_dot: f64,
}

/// Reproducible snapshot of the worst (largest `|along_track_arcsec|`)
/// `SearchedButNotMatched` case a trajectory hit with
/// `dt_since_last_real_update < SHORT_DT_THRESHOLD_DAYS` — the regime
/// where the confirmed n-body secular-drift mechanism
/// (`crates/fink-fat-engine/tests/two_body_vs_perturbed_drift.rs`) is too
/// small to be the explanation, so these are candidates for manual
/// inspection of whatever else is going on. Carries only the MAP
/// hypothesis's state (not the full bank, which can run to hundreds of
/// hypotheses) to keep the in-memory candidate list — one per trajectory,
/// kept until the dataset-wide top-N is written — bounded.
#[derive(Debug, Clone, serde::Serialize)]
struct ShortDtMissDump {
    traj_id: String,
    night_id: u32,
    dt_since_last_real_update: f64,
    arc_span_since_bootstrap_days: f64,
    n_real_updates_so_far: usize,
    along_track_arcsec: f64,
    cross_track_arcsec: Option<f64>,
    nis: Option<f64>,
    n_hypotheses: Option<usize>,
    map_weight_fraction: Option<f64>,
    /// The true observation the search missed.
    observation: Observation,
    /// The predicted bank's MAP (highest-weight) hypothesis at the time of
    /// the miss.
    best_hypothesis: KFStateDump,
}

/// Full-bank snapshot of a `SearchedButNotMatched` case at step 1 or
/// step 2 post-bootstrap (`step_index` 0 or 1 — position in the
/// trajectory's own `outcomes`, see [`ObsOutcome`]'s doc on why that's
/// not the same thing as `n_real_updates_so_far`). Unlike
/// [`ShortDtMissDump`] (best hypothesis only, picked as the single worst
/// case dataset-wide), this carries **every** surviving hypothesis —
/// deliberately heavier, but only for a capped, targeted sample (see
/// [`write_bootstrap_step_dumps`]) — to let a step-1-vs-step-2 case pair
/// be inspected by hand for how the weight distribution actually shifts
/// across the whole mixture, since the aggregate "MAP weight fraction"
/// metric turned out not to move between step 1 and step 2 even as the
/// along-track bias did (see the investigation plan doc).
#[derive(Debug, Clone, serde::Serialize)]
struct BootstrapStepDump {
    traj_id: String,
    night_id: u32,
    step_index: usize,
    dt_since_last_real_update: f64,
    arc_span_since_bootstrap_days: f64,
    along_track_arcsec: Option<f64>,
    cross_track_arcsec: Option<f64>,
    nis: Option<f64>,
    n_hypotheses: Option<usize>,
    map_weight_fraction: Option<f64>,
    observation: Observation,
    hypotheses: Vec<HypothesisDump>,
}

/// One trajectory's MOT contamination study: how many candidates (true +
/// bad) fell inside the predicted search region of every visit the
/// trajectory's Kalman lineage could plausibly be in.
#[derive(Debug)]
pub struct TrajStat {
    traj_id: TrajId,
    n_obs_total: usize,
    n_obs_true: u32,
    n_true_obs_added: u32,
    coverage: f32,
    visit_stat: Vec<VisitStat>,
    /// One entry per real observation in `observations_to_process` order —
    /// see [`ObsOutcome`].
    outcomes: Vec<ObsOutcome>,
    /// Reproducible dump of the collapse, if `stop_reason` is
    /// `CollapsedByPropagation` or `DegenerateState` — see
    /// [`FailureCaseDump`].
    failure_dump: Option<FailureCaseDump>,
    /// Worst short-dt `SearchedButNotMatched` case this trajectory hit —
    /// see [`ShortDtMissDump`].
    short_dt_dump: Option<ShortDtMissDump>,
    /// Full-bank snapshots of this trajectory's step-1/step-2
    /// `SearchedButNotMatched` cases, if any — see [`BootstrapStepDump`].
    /// At most 2 entries (one per step).
    bootstrap_step_dumps: Vec<BootstrapStepDump>,
    /// Why the real (`branch`) lineage walk stopped — see
    /// [`TruncationReason`].
    stop_reason: TruncationReason,
    /// Shadow-mode counters of every candidate filter (plus the cascade)
    /// over this trajectory's visits — see
    /// [`fink_fat_eval::candidate_filters`].
    shadow: ShadowStats,
}

impl TrajStat {
    pub fn n_visits(&self) -> usize {
        self.visit_stat.len()
    }

    pub fn n_visits_with_true(&self) -> usize {
        self.visit_stat.iter().filter(|v| v.true_in_visit).count()
    }

    pub fn total_candidates(&self) -> u64 {
        self.visit_stat.iter().map(|v| v.n_candidates as u64).sum()
    }

    pub fn total_bad(&self) -> u64 {
        self.visit_stat.iter().map(|v| v.n_bad as u64).sum()
    }

    /// Mean number of bad (non-true) candidates per visit — `NaN` if the
    /// trajectory produced no visit at all.
    pub fn mean_bad_per_visit(&self) -> f64 {
        if self.visit_stat.is_empty() {
            f64::NAN
        } else {
            self.total_bad() as f64 / self.visit_stat.len() as f64
        }
    }

    /// Percentage of candidates that are bad (non-true) — `NaN` if no
    /// candidate was ever found.
    pub fn bad_candidate_rate_pct(&self) -> f64 {
        let total = self.total_candidates();
        if total == 0 {
            f64::NAN
        } else {
            100.0 * self.total_bad() as f64 / total as f64
        }
    }
}

// ── Per-night precomputation ──────────────────────────────────────────────
//
// `group_observations_into_visits` and `build_alert_bucket_index` depend
// only on a night's observations (never on any trajectory's lineage), so
// they're computed once per night here and shared read-only across the
// per-trajectory parallel loop below, instead of being rebuilt from scratch
// by every trajectory that happens to traverse that night.

struct PrecomputedVisit<'o> {
    visit: Visit<'o>,
    bucket_index: BucketIndex<&'o Observation>,
}

struct PrecomputedNight<'o> {
    night_id: NightId,
    visits: Vec<PrecomputedVisit<'o>>,
}

/// Build one [`PrecomputedNight`] per entry of `night_obs_store`, in the
/// same order, so callers can index `precomputed_nights[i]` for
/// `all_nights[i]` / `night_obs_store[i]`.
///
/// Takes `night_obs_store` (rather than materializing nights itself) so the
/// borrowed `Visit`/`BucketIndex` data returned here can outlive this
/// function call: they borrow from `night_obs_store`'s elements, which the
/// caller keeps alive for as long as the returned `Vec<PrecomputedNight>` is
/// used (see `process_all_trajectories_mot`) — two sibling local bindings,
/// not a self-referential struct.
fn precompute_nights<'o>(
    night_obs_store: &'o [Cow<'o, [Observation]>],
    all_nights: &[NightId],
    visit_epoch_tolerance_days: f64,
    spatial_binner: &HealpixBinner,
) -> Vec<PrecomputedNight<'o>> {
    night_obs_store
        .par_iter()
        .zip(all_nights)
        .map(|(night_obs, night_id)| {
            let refs: Vec<&Observation> = night_obs.iter().collect();
            let visits = group_observations_into_visits(&refs, visit_epoch_tolerance_days);
            let visits = visits
                .into_iter()
                .map(|visit| {
                    let bucket_index = build_alert_bucket_index(
                        visit.observations.iter().copied(),
                        spatial_binner,
                        &SingleBinTimeBinner,
                    );
                    PrecomputedVisit {
                        visit,
                        bucket_index,
                    }
                })
                .collect();
            PrecomputedNight {
                night_id: *night_id,
                visits,
            }
        })
        .collect()
}

// ── Dataset-wide driver ──────────────────────────────────────────────────

#[derive(Debug, Default)]
struct MotCounters {
    n_total: usize,
    n_materialize_failed: usize,
    n_too_short: usize,
    n_no_result: usize,
}

enum MotOutcome {
    MaterializeFailed,
    TooShort,
    NoResult,
    Stat(Box<TrajStat>),
}

/// Caps on how many *candidate* heavy dumps (`FailureCaseDump`,
/// `BootstrapStepDump` — both embed a `Vec<HypothesisDump>` that can run to
/// hundreds of entries, since a bank can hold up to `cap_schedule`'s
/// configured start value right after bootstrap) get constructed at all
/// during the parallel trajectory scan, shared across every worker via
/// [`DumpBudgets`].
///
/// Only a small, fixed number of these ever get written to disk in the end
/// (`MAX_DUMPED_CASES_PER_REASON`, `MAX_DUMPED_BOOTSTRAP_PAIRS`), but without
/// this gate every trajectory that hits the relevant condition — which for
/// `DegenerateState` alone can be tens of thousands on a full-survey run —
/// built and retained its own copy of the bank's full hypothesis population
/// in `TrajStat` until the very end of the run, which is what actually
/// exhausted memory. These caps are a generous multiple of the final
/// on-disk caps (not equal to them) so the existing post-hoc selection
/// logic in `write_failure_dump_file`/`write_bootstrap_step_dumps` (which
/// needs a candidate pool larger than what it keeps, e.g. to prefer
/// step-1/step-2 pairs over singles) still has a reasonable pool to choose
/// from — the tradeoff accepted here is an approximately-representative
/// sample rather than an exact global ranking.
const DUMP_BUDGET_MULTIPLIER: usize = 10;

/// Per-reason/per-kind atomic budgets, shared read-only-by-reference across
/// the `rayon` parallel scan in [`process_all_trajectories_mot`] — see
/// [`DUMP_BUDGET_MULTIPLIER`].
#[derive(Default)]
struct DumpBudgets {
    propagation_failures: AtomicUsize,
    degenerate_state: AtomicUsize,
    bootstrap_step_candidates: AtomicUsize,
}

impl DumpBudgets {
    /// Atomically consumes one slot of `counter` if under `cap`; returns
    /// whether the caller may go ahead and build the (expensive) dump.
    fn try_reserve(counter: &AtomicUsize, cap: usize) -> bool {
        counter.fetch_add(1, AtomicOrdering::Relaxed) < cap
    }
}

fn build_progress_bar(n: usize) -> ProgressBar {
    let progress = ProgressBar::new(n as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} trajectories ({percent}%)  elapsed {elapsed_precise}  eta {eta_precise}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    progress
}

#[allow(clippy::too_many_arguments)]
fn process_one_trajectory_mot(
    traj_id: &TrajId,
    obs_dataset: &ObsDataset,
    engine_config: &EngineConfig,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
    night_map: &AHashMap<ObsId, NightId>,
    traj_map: &AHashMap<ObsId, TrajId>,
    all_nights: &[NightId],
    precomputed_nights: &[PrecomputedNight],
    spatial_binner: &HealpixBinner,
    geometry_cache: &ObserverGeometryCache,
    dump_budgets: &DumpBudgets,
    filter_suite: &[Box<dyn CandidateFilter>],
) -> MotOutcome {
    let traj = match materialize_contiguous_traj(obs_dataset, traj_id) {
        Ok(t) => t,
        Err(_) => return MotOutcome::MaterializeFailed,
    };
    if traj.len() < 3 {
        return MotOutcome::TooShort;
    }

    match study_kalman_asteroid(
        &traj,
        traj_id.clone(),
        obs_dataset,
        engine_config,
        context,
        bank_config,
        night_map,
        traj_map,
        all_nights,
        precomputed_nights,
        spatial_binner,
        geometry_cache,
        dump_budgets,
        filter_suite,
    ) {
        Some(stat) => MotOutcome::Stat(Box::new(stat)),
        None => MotOutcome::NoResult,
    }
}

/// Run the MOT contamination study on every trajectory of `obs_dataset` that
/// the Kalman filter bank can actually track (bootstrap succeeds, at least
/// one processable observation afterwards).
fn process_all_trajectories_mot(
    obs_dataset: &ObsDataset,
    engine_config: &EngineConfig,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
) -> (Vec<TrajStat>, MotCounters) {
    let traj_ids: Vec<TrajId> = obs_dataset
        .iter_traj_id()
        .expect("dataset must expose at least one trajectory")
        .cloned()
        .collect();

    let night_map: AHashMap<ObsId, NightId> = obs_dataset
        .iter_full_night()
        .expect("dataset must expose at least one night")
        .map(|(night_id, obs)| (*obs.id(), night_id))
        .collect();

    let traj_map: AHashMap<ObsId, TrajId> = obs_dataset
        .iter_full_trajectory()
        .expect("dataset must expose at least one trajectory")
        .map(|(traj_id, obs)| (*obs.id(), traj_id))
        .collect();

    let mut all_nights: Vec<NightId> = obs_dataset
        .iter_night_id()
        .expect("dataset must expose at least one night")
        .copied()
        .collect();
    all_nights.sort();

    // Materialized once per night and kept alive for the rest of this
    // function — `precomputed_nights` below borrows from these elements, so
    // `night_obs_store` must outlive it (two sibling locals, not a
    // self-referential struct).
    let night_obs_store: Vec<Cow<[Observation]>> = all_nights
        .iter()
        .map(|n| {
            materialize_contiguous_night(obs_dataset, n)
                .expect("night must materialize: night_id was taken from this same dataset")
        })
        .collect();

    let spatial_binner = HealpixBinner::new(engine_config.healpix_depth);

    // `group_observations_into_visits`/`build_alert_bucket_index` depend
    // only on the night, never on the trajectory — computed once here
    // instead of once per (trajectory, night), which used to be the
    // dominant cost of this scan.
    let precomputed_nights = precompute_nights(
        &night_obs_store,
        &all_nights,
        engine_config.advance_params.visit_epoch_tolerance_days,
        &spatial_binner,
    );

    let geometry_cache = ObserverGeometryCache::build(obs_dataset, context, &traj_ids);

    let progress = build_progress_bar(traj_ids.len());

    let dump_budgets = DumpBudgets::default();
    let filter_suite = default_filter_suite();

    let outcomes: Vec<MotOutcome> = traj_ids
        .par_iter()
        .map(|traj_id| {
            let outcome = process_one_trajectory_mot(
                traj_id,
                obs_dataset,
                engine_config,
                context,
                bank_config,
                &night_map,
                &traj_map,
                &all_nights,
                &precomputed_nights,
                &spatial_binner,
                &geometry_cache,
                &dump_budgets,
                &filter_suite,
            );
            progress.inc(1);
            outcome
        })
        .collect();

    progress.finish_and_clear();

    let mut counters = MotCounters::default();
    let mut stats = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        counters.n_total += 1;
        match outcome {
            MotOutcome::MaterializeFailed => counters.n_materialize_failed += 1,
            MotOutcome::TooShort => counters.n_too_short += 1,
            MotOutcome::NoResult => counters.n_no_result += 1,
            MotOutcome::Stat(s) => stats.push(*s),
        }
    }

    (stats, counters)
}

/// Cap on how many reproducible failure cases get written per reason —
/// `DegenerateState` alone can be tens of thousands of trajectories; the
/// dataset-wide histogram already reports the real total, these files are
/// just a sample to isolate/replay concrete cases for a regression test.
const MAX_DUMPED_CASES_PER_REASON: usize = 200;

/// Write reproducible cases for the two collapse reasons worth debugging
/// (`CollapsedByPropagation`: numerical solver failure; `DegenerateState`:
/// the bank had already collapsed to zero hypotheses upstream) to two JSON
/// Lines files — see [`FailureCaseDump`].
fn write_failure_dumps(stats: &[TrajStat]) {
    write_failure_dump_file(
        stats,
        TruncationReason::CollapsedByPropagation,
        "propagation_failures.jsonl",
    );
    write_failure_dump_file(
        stats,
        TruncationReason::DegenerateState,
        "degenerate_state_failures.jsonl",
    );
}

/// How many short-dt `SearchedButNotMatched` cases (dataset-wide top-N by
/// `|along_track_arcsec|`, one candidate per trajectory) to write to
/// `short_dt_not_matched_cases.jsonl` — see [`ShortDtMissDump`].
const MAX_DUMPED_SHORT_DT_CASES: usize = 100;

/// Write the dataset-wide worst short-dt `SearchedButNotMatched` cases —
/// see [`ShortDtMissDump`] and [`print_short_dt_not_matched_diagnostics`].
fn write_short_dt_case_dumps(stats: &[TrajStat]) {
    let mut candidates: Vec<&ShortDtMissDump> = stats
        .iter()
        .filter_map(|s| s.short_dt_dump.as_ref())
        .collect();
    if candidates.is_empty() {
        return;
    }
    candidates.sort_by(|a, b| {
        b.along_track_arcsec
            .abs()
            .partial_cmp(&a.along_track_arcsec.abs())
            .unwrap_or(Ordering::Equal)
    });
    candidates.truncate(MAX_DUMPED_SHORT_DT_CASES);

    let path = "short_dt_not_matched_cases.jsonl";
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            println!("  (failed to create {path}: {e})");
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    let mut n_written = 0usize;
    for dump in &candidates {
        if serde_json::to_writer(&mut writer, dump).is_ok() && writer.write_all(b"\n").is_ok() {
            n_written += 1;
        }
    }
    println!(
        "  Wrote {n_written}/{} short-dt SearchedButNotMatched case(s) to {path}",
        stats.iter().filter(|s| s.short_dt_dump.is_some()).count()
    );
}

/// Cap on trajectories dumped by [`write_bootstrap_step_dumps`] — pairs
/// (both step 1 and step 2 missed, so the same lineage can be compared
/// before/after) are prioritized over singles.
const MAX_DUMPED_BOOTSTRAP_PAIRS: usize = 30;

/// Write full-bank (every hypothesis, not just the MAP one) snapshots of
/// step-1/step-2 `SearchedButNotMatched` cases, prioritizing trajectories
/// where *both* steps were missed (a real same-lineage before/after pair)
/// — for manually inspecting how the hypothesis weight distribution
/// actually shifts between step 1 and step 2, since the aggregate "MAP
/// weight fraction" metric in `print_bootstrap_residual_diagnostics`
/// turned out not to move even as the along-track bias did. See
/// [`BootstrapStepDump`].
fn write_bootstrap_step_dumps(stats: &[TrajStat]) {
    let mut pairs: Vec<&TrajStat> = stats
        .iter()
        .filter(|s| s.bootstrap_step_dumps.len() == 2)
        .collect();
    let mut singles: Vec<&TrajStat> = stats
        .iter()
        .filter(|s| s.bootstrap_step_dumps.len() == 1)
        .collect();
    pairs.truncate(MAX_DUMPED_BOOTSTRAP_PAIRS);
    let n_singles = MAX_DUMPED_BOOTSTRAP_PAIRS.saturating_sub(pairs.len());
    singles.truncate(n_singles);

    let all_dumps: Vec<&BootstrapStepDump> = pairs
        .iter()
        .chain(singles.iter())
        .flat_map(|s| s.bootstrap_step_dumps.iter())
        .collect();
    if all_dumps.is_empty() {
        return;
    }

    let path = "bootstrap_step_dumps.jsonl";
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            println!("  (failed to create {path}: {e})");
            return;
        }
    };
    let mut writer = BufWriter::new(file);
    let mut n_written = 0usize;
    for dump in &all_dumps {
        if serde_json::to_writer(&mut writer, dump).is_ok() && writer.write_all(b"\n").is_ok() {
            n_written += 1;
        }
    }
    let total_trajs = stats
        .iter()
        .filter(|s| !s.bootstrap_step_dumps.is_empty())
        .count();
    println!(
        "  Wrote {n_written} bootstrap step-1/step-2 case(s) ({} paired trajectories, {} single) \
         to {path}, out of {total_trajs} trajectories with such cases",
        pairs.len(),
        singles.len(),
    );
}

fn write_failure_dump_file(stats: &[TrajStat], reason: TruncationReason, path: &str) {
    let total = stats.iter().filter(|s| s.stop_reason == reason).count();
    if total == 0 {
        return;
    }

    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            println!("  (failed to create {path}: {e})");
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    let mut n_written = 0usize;
    for s in stats {
        if s.stop_reason != reason || n_written >= MAX_DUMPED_CASES_PER_REASON {
            continue;
        }
        let Some(dump) = &s.failure_dump else {
            continue;
        };
        if serde_json::to_writer(&mut writer, dump).is_ok() && writer.write_all(b"\n").is_ok() {
            n_written += 1;
        }
    }
    let _ = writer.flush();

    println!(
        "  Wrote {n_written}/{total} \"{}\" case(s) to {path}",
        reason.label()
    );
}

/// Study one trajectory: walk the dataset night by night (starting at the
/// night of its first processable observation, stopping as soon as the
/// trajectory's own observations are exhausted), and for every visit the
/// Kalman lineage might plausibly be in, count how many observations
/// (candidates) fall inside the predicted search region — split into the
/// true observation and everything else ("bad" candidates).
///
/// Returns `None` if the bootstrap failed or left nothing to process.
#[allow(clippy::too_many_arguments)]
fn study_kalman_asteroid(
    traj: &[Observation],
    traj_id: TrajId,
    obs_dataset: &ObsDataset,
    engine_config: &EngineConfig,
    context: &KalmanContext,
    bank_config: &KFBankConfig,
    night_map: &AHashMap<ObsId, NightId>,
    traj_map: &AHashMap<ObsId, TrajId>,
    all_nights: &[NightId],
    precomputed_nights: &[PrecomputedNight],
    spatial_binner: &HealpixBinner,
    geometry_cache: &ObserverGeometryCache,
    dump_budgets: &DumpBudgets,
    filter_suite: &[Box<dyn CandidateFilter>],
) -> Option<TrajStat> {
    let n_obs_total = traj.len();

    let (traj, _n_obs_deduplicated) = dedupe_by_epoch(
        traj,
        engine_config.advance_params.visit_epoch_tolerance_days,
    );
    let traj = traj.as_slice();

    let (idx_first_obs, bank) = init_bank_from_first_pair(
        traj,
        obs_dataset,
        context,
        bank_config,
        &engine_config.seeding_grid_config,
    )?;

    // `KFBank::from_grid` can legitimately return a bank with zero surviving
    // hypotheses (every grid seed's weight gated to ≤ 0) — `Branch::seed`
    // asserts on a live hypothesis and panics otherwise (see
    // `discovery::seed_new_lineages_from_leftovers`, which guards the same
    // way before its own `Branch::seed` calls). Treat it like any other
    // unusable bootstrap.
    if !bank.is_alive() {
        return None;
    }

    let observations_to_process = &traj[idx_first_obs + 2..];
    let n_obs = observations_to_process.len() as u32;
    if n_obs == 0 {
        return None;
    }

    // Reverse lookup for classifying a `CandidateMatch` against this
    // trajectory's own (deduplicated, post-bootstrap) observations — also
    // caps credit to `ObsId`s that are actually members of
    // `observations_to_process` (excludes the bootstrap pair and any
    // near-duplicate-epoch alert `dedupe_by_epoch` folded away but which
    // still exists as its own `Observation` in the raw night data used by
    // candidate search), so `n_found <= n_obs` (coverage <= 100%) is true by
    // construction, not just likely.
    let obs_id_to_slot: ahash::AHashMap<ObsId, usize> = observations_to_process
        .iter()
        .enumerate()
        .map(|(i, o)| (*o.id(), i))
        .collect();

    // One classification per real observation — see `ObsOutcome`. Starts
    // `NotReached` and gets overwritten as the walk below actually reaches
    // (or explicitly skips) each one; anything still `NotReached` at the end
    // means the walk broke off before getting there.
    let mut outcomes: Vec<ObsOutcome> = vec![ObsOutcome::not_reached(); n_obs as usize];

    let mut shadow = ShadowStats::new(filter_suite.len());

    let mut branch = Branch::seed(bank.clone(), 0, 0, 0);
    // Epoch `branch` was last updated with a *real* observation — the basis
    // for `dt_since_last_real_update`, which lets the report show whether
    // misses correlate with how stale the lineage's last real fix is (the
    // coarse prefilter's search radius is fixed, not `dt`-scaled).
    let mut last_real_update_epoch = traj[idx_first_obs + 1].mjd_tt();
    // Sky position/epoch of the last real observation `branch` consumed —
    // the observation-centric anchor for the direction/rate candidate
    // filters (OC-SORT's OCM idea). Starts at the bootstrap pair's second
    // observation, advances with `last_real_update_epoch`.
    let mut last_accepted = {
        let o = &traj[idx_first_obs + 1];
        let c = o.equ_coord();
        LastAccepted {
            epoch: o.mjd_tt(),
            ra: c.ra,
            dec: c.dec,
        }
    };
    // Fixed (unlike `last_real_update_epoch`, which advances at every real
    // update) — the epoch of the bootstrap pair, basis for
    // `arc_span_since_bootstrap_days`: how long the *total* two-body arc
    // has been propagating, as opposed to `dt_since_last_real_update`'s
    // "since the last fix" gap. A trajectory can have a short last-fix gap
    // while still sitting on a long, drift-prone total arc.
    let bootstrap_epoch = last_real_update_epoch;
    let mut stop_reason = TruncationReason::ReachedEnd;
    let mut failure_dump: Option<FailureCaseDump> = None;
    let mut short_dt_dump: Option<ShortDtMissDump> = None;
    let mut bootstrap_step_dumps: Vec<BootstrapStepDump> = Vec::new();
    // Post-update state of the best (MAP) hypothesis after every real
    // update this lineage has successfully consumed so far — see
    // `StateHistoryPoint`. Cheap (a handful of f64s per real update, and
    // real updates are a small fraction of all steps), kept for every
    // trajectory so it's available regardless of which one ends up needing
    // a `failure_dump`.
    let mut state_history: Vec<StateHistoryPoint> = Vec::new();

    // Index-based instead of a plain iterator, so `nb_true_obs_in_night`
    // (below) can be computed by peeking ahead in the trajectory's own
    // (short) observation list rather than scanning the whole night.
    let mut idx = 0usize;
    let mut current_obs = &observations_to_process[idx];

    let mut vec_visit_stat: Vec<VisitStat> = Vec::new();

    let first_night = *night_map.get(current_obs.id())?;
    let start_idx = all_nights.partition_point(|n| *n < first_night);

    'night_loop: for (step, night_data) in precomputed_nights[start_idx..].iter().enumerate() {
        let night_id = night_data.night_id;
        let epoch = current_obs.mjd_tt();

        // O(1) lookup instead of scanning every observation of the night.
        let night_contains_next_true_obs = night_map.get(current_obs.id()) == Some(&night_id);
        // How many of the trajectory's own upcoming observations share this
        // night — a short walk over the trajectory's own observations (not
        // the whole night). Pure function of `night_map`/`idx`, needed by
        // both the geometry-failure path and the normal path below.
        let nb_true_obs_in_night = if night_contains_next_true_obs {
            count_same_night(observations_to_process, night_map, idx, night_id)
        } else {
            0
        };
        let dt_since_last_real_update = epoch - last_real_update_epoch;
        let arc_span_since_bootstrap_days = epoch - bootstrap_epoch;

        if geometry_cache
            .get(obs_dataset, context, current_obs)
            .is_none()
        {
            // No prediction is possible for this night at all. Classify and
            // skip past this night's true observations (if any) instead of
            // leaving `idx`/`current_obs` stuck forever — previously this
            // silently orphaned every later observation of the trajectory.
            for slot in outcomes.iter_mut().skip(idx).take(nb_true_obs_in_night) {
                slot.reason = MissReason::GeometryUnavailable;
                slot.dt_since_last_real_update = dt_since_last_real_update;
                slot.arc_span_since_bootstrap_days = arc_span_since_bootstrap_days;
                slot.n_real_updates_so_far = branch.n_real_updates;
            }
            idx += nb_true_obs_in_night;
            if idx >= observations_to_process.len() {
                break 'night_loop;
            }
            current_obs = &observations_to_process[idx];
            continue;
        }

        let mut tmp_branch_visit = branch.clone();
        let mut visit_prefilter_passed = vec![false; night_data.visits.len()];

        // Mirrors production's per-night anchor (see `PrefilterAnchor`): the
        // coarse prefilter extrapolates from the MAP hypothesis propagated
        // to this night's first resolvable visit, not from a possibly
        // weeks-stale bank epoch. Computed once per night, exactly like
        // `advance_bank_collection_one_night` does.
        //
        // The `find_map` short-circuits on the *geometry*, never on the
        // anchor result: `for_branch` legitimately returns `None` for a
        // branch that is already fresh, and letting that drive the search
        // would re-scan every visit of the night — a per-visit geometry
        // lookup for every fresh (trajectory, night) pair, which is exactly
        // the cost this whole mechanism is supposed to avoid.
        let anchored_branch_id = tmp_branch_visit.branch_id;
        let prefilter_anchor = night_data
            .visits
            .iter()
            .find_map(|pv| {
                geometry_cache
                    .get(obs_dataset, context, pv.visit.representative_obs)
                    .map(|(r_obs, v_obs)| (pv.visit.epoch, r_obs, v_obs))
            })
            .and_then(|(epoch, r_obs, v_obs)| {
                PrefilterAnchor::for_branch(&tmp_branch_visit, epoch, r_obs, v_obs)
            });

        // Each visit gets its own fresh `predict_to`, at *its own* epoch
        // (`pv.visit.epoch`) and observer geometry — never shared across
        // visits. A ZTF night spans ~8h of distinct exposure epochs
        // (`Visit`'s own doc comment); reusing a single per-night epoch
        // here used to propagate the bank to the wrong instant for every
        // visit but the first, silently mismatching the search region
        // against the true observer/object geometry — a spurious
        // along-track offset baked into both the actual candidate search
        // (`not_matched%` itself) and the diagnostics below. Matches
        // production's `orchestrate::advance_bank_collection_one_night`,
        // which predicts fresh per visit for the same reason.
        for (visit_index, pv) in night_data.visits.iter().enumerate() {
            // The anchor belongs to the branch it was computed for; once
            // `tmp_branch_visit` has been replaced by a freshly-updated
            // branch its own bank epoch is already at a visit of this night,
            // so it needs none — same rule as production, where only the
            // incoming lineages are anchored.
            let anchor = (tmp_branch_visit.branch_id == anchored_branch_id)
                .then_some(prefilter_anchor.as_ref())
                .flatten();
            let lineage_in_visit = lineage_might_be_in_visit(
                &tmp_branch_visit,
                anchor,
                &pv.visit,
                &pv.bucket_index,
                &engine_config.advance_params,
                spatial_binner,
            );

            if !lineage_in_visit {
                continue;
            }
            visit_prefilter_passed[visit_index] = true;

            let Some((visit_r_obs, visit_v_obs)) =
                geometry_cache.get(obs_dataset, context, pv.visit.representative_obs)
            else {
                continue;
            };
            let predicted_bank =
                tmp_branch_visit
                    .bank
                    .predict_to(pv.visit.epoch, visit_r_obs, visit_v_obs);

            let Ok(search_region) = predicted_bank.search_region(
                engine_config.advance_params.obs_noise.into(),
                engine_config.advance_params.top_k,
                engine_config.advance_params.radius_strategy,
            ) else {
                continue;
            };

            // Same coarse-search decision as
            // `orchestrate::spawn_branches_for_lineage` — shared via
            // `cover_if_clamped` precisely so the two cannot drift, since any
            // divergence would make `not_matched%` stop measuring the engine.
            let candidates = if let Some(cover) = cover_if_clamped(
                &search_region,
                engine_config.advance_params.radius_strategy,
                engine_config
                    .advance_params
                    .cone_half_rad(search_region.components.len()),
                engine_config.advance_params.max_search_cones,
            ) {
                find_candidates_for_bank_multi_region(
                    &cover,
                    &search_region,
                    tmp_branch_visit.bank.track_ids().to_vec(),
                    &pv.bucket_index,
                    spatial_binner,
                    tmp_branch_visit.bank.config.gate_chi2,
                    engine_config.advance_params.likelihood_threshold,
                )
            } else {
                find_candidates_for_bank(
                    &search_region,
                    tmp_branch_visit.bank.track_ids().to_vec(),
                    &pv.bucket_index,
                    spatial_binner,
                    tmp_branch_visit.bank.config.gate_chi2,
                    engine_config.advance_params.likelihood_threshold,
                )
            };

            // Ground-truth flag per candidate (any observation of the same
            // trajectory counts as true — same rule as the contamination
            // counts) + shadow evaluation of the whole filter suite. The
            // filters themselves never see the ground truth.
            let is_true: Vec<bool> = candidates
                .matches
                .iter()
                .map(|c| traj_map.get(c.observation.id()) == Some(&traj_id))
                .collect();
            // Same clutter background the engine already uses to score
            // spawned branches (`observation_llr_delta` in
            // `orchestrate::spawn_branches_for_lineage`) — reused here by
            // `TopKGate` to rank candidates one stage earlier.
            let region_center = EquCoord::new(
                candidates.search_region.center_ra,
                0.,
                candidates.search_region.center_dec,
                0.,
            );
            let clutter_density =
                local_clutter_density(&pv.bucket_index, spatial_binner, &region_center);
            let filter_ctx = FilterContext::new(
                &candidates.search_region,
                &predicted_bank,
                Some(last_accepted),
                pv.visit.epoch,
                branch.n_real_updates,
                clutter_density,
                engine_config.advance_params.photometric_sigma_mag,
            );
            let cascade_mask = evaluate_filters(
                filter_suite,
                DEFAULT_CASCADE,
                &filter_ctx,
                &candidates.matches,
                &is_true,
                &mut shadow,
            );

            let visit_stats = VisitStat::from_masks(night_id, visit_index, &is_true, &cascade_mask);

            if night_contains_next_true_obs {
                // `candidates`/`find_candidates_for_bank` deliberately do NOT
                // filter by ground truth — a real MOT search has no oracle
                // on identity, which is exactly why "bad candidates" exist.
                // The ground-truth comparison happens here instead, via
                // `obs_id_to_slot` (built only from this trajectory's own
                // real observations — see `VisitStat::from_bank_candidate`,
                // just above, for the equivalent check used for the
                // per-visit contamination counts).
                for cand in candidates.matches.iter() {
                    let obs = cand.observation;

                    if let Some(&slot) = obs_id_to_slot.get(obs.id())
                        && outcomes[slot].reason == MissReason::NotReached
                    {
                        // Diagnostic against this visit's own freshly
                        // predicted bank (`predicted_bank`, already
                        // propagated to `pv.visit.epoch` — the same epoch
                        // `obs` itself belongs to, since it came from this
                        // visit's candidates).
                        let diag = Some(geometry_diagnostic(
                            &predicted_bank,
                            obs,
                            &engine_config.advance_params,
                            bank_config.search_region_chi2,
                        ));
                        outcomes[slot] = ObsOutcome {
                            reason: MissReason::Found,
                            dt_since_last_real_update,
                            arc_span_since_bootstrap_days,
                            nis: diag.as_ref().and_then(|d| d.nis),
                            separation_arcsec: diag.as_ref().and_then(|d| d.separation_arcsec),
                            region_radius_arcsec: diag
                                .as_ref()
                                .and_then(|d| d.region_radius_arcsec),
                            along_track_arcsec: diag.as_ref().and_then(|d| d.along_track_arcsec),
                            cross_track_arcsec: diag.as_ref().and_then(|d| d.cross_track_arcsec),
                            n_hypotheses: diag.as_ref().and_then(|d| d.n_hypotheses),
                            map_weight_fraction: diag.as_ref().and_then(|d| d.map_weight_fraction),
                            n_real_updates_so_far: branch.n_real_updates,
                        };

                        if let Some(new_branch) = Branch::from_observation(
                            &predicted_bank,
                            &tmp_branch_visit,
                            obs,
                            0.,
                            step as u64,
                            step,
                        ) {
                            tmp_branch_visit = new_branch;
                        }
                    }
                }
            }

            vec_visit_stat.push(visit_stats);
        }

        if night_contains_next_true_obs {
            // Classify whatever wasn't found above: did the visit actually
            // containing this observation even pass the coarse prefilter?
            for slot in idx..idx + nb_true_obs_in_night {
                if outcomes[slot].reason != MissReason::NotReached {
                    continue;
                }
                let obs = &observations_to_process[slot];
                let owning_visit = night_data
                    .visits
                    .iter()
                    .position(|pv| pv.visit.observations.iter().any(|o| o.id() == obs.id()));
                let reason = match owning_visit {
                    Some(vi) if visit_prefilter_passed[vi] => MissReason::SearchedButNotMatched,
                    _ => MissReason::NoVisitPassedPrefilter,
                };
                // Diagnostic against a fresh prediction to `obs`'s own
                // epoch/geometry — a multi-observation night must not
                // reuse an earlier observation's epoch here either (same
                // reasoning as the visits loop above).
                let canonical_predicted = geometry_cache
                    .get(obs_dataset, context, obs)
                    .map(|(obs_r, obs_v)| branch.bank.predict_to(obs.mjd_tt(), obs_r, obs_v));
                let diag = canonical_predicted.as_ref().map(|cp| {
                    geometry_diagnostic(
                        cp,
                        obs,
                        &engine_config.advance_params,
                        bank_config.search_region_chi2,
                    )
                });
                outcomes[slot] = ObsOutcome {
                    reason,
                    dt_since_last_real_update,
                    arc_span_since_bootstrap_days,
                    nis: diag.as_ref().and_then(|d| d.nis),
                    separation_arcsec: diag.as_ref().and_then(|d| d.separation_arcsec),
                    region_radius_arcsec: diag.as_ref().and_then(|d| d.region_radius_arcsec),
                    along_track_arcsec: diag.as_ref().and_then(|d| d.along_track_arcsec),
                    cross_track_arcsec: diag.as_ref().and_then(|d| d.cross_track_arcsec),
                    n_hypotheses: diag.as_ref().and_then(|d| d.n_hypotheses),
                    map_weight_fraction: diag.as_ref().and_then(|d| d.map_weight_fraction),
                    n_real_updates_so_far: branch.n_real_updates,
                };

                if reason == MissReason::SearchedButNotMatched
                    && dt_since_last_real_update < SHORT_DT_THRESHOLD_DAYS
                    && let Some(along) = diag.as_ref().and_then(|d| d.along_track_arcsec)
                    && short_dt_dump
                        .as_ref()
                        .is_none_or(|d| along.abs() > d.along_track_arcsec.abs())
                    && let Some(cp) = canonical_predicted.as_ref()
                    && let Some(best) = cp
                        .hypotheses()
                        .iter()
                        .max_by(|a, b| a.log_weight.total_cmp(&b.log_weight))
                {
                    short_dt_dump = Some(ShortDtMissDump {
                        traj_id: traj_id.to_string(),
                        night_id: night_id.0,
                        dt_since_last_real_update,
                        arc_span_since_bootstrap_days,
                        n_real_updates_so_far: branch.n_real_updates,
                        along_track_arcsec: along,
                        cross_track_arcsec: diag.as_ref().and_then(|d| d.cross_track_arcsec),
                        nis: diag.as_ref().and_then(|d| d.nis),
                        n_hypotheses: diag.as_ref().and_then(|d| d.n_hypotheses),
                        map_weight_fraction: diag.as_ref().and_then(|d| d.map_weight_fraction),
                        observation: obs.clone(),
                        best_hypothesis: (&best.kf).into(),
                    });
                }

                // Gated the same way as `FailureCaseDump` — see its comment
                // above and [`DumpBudgets`]'s doc — `hypotheses` below
                // clones the bank's full population, and only
                // `MAX_DUMPED_BOOTSTRAP_PAIRS` trajectories' worth ever get
                // written by `write_bootstrap_step_dumps`.
                if reason == MissReason::SearchedButNotMatched
                    && slot < 2
                    && let Some(cp) = canonical_predicted.as_ref()
                    && DumpBudgets::try_reserve(
                        &dump_budgets.bootstrap_step_candidates,
                        MAX_DUMPED_BOOTSTRAP_PAIRS * 2 * DUMP_BUDGET_MULTIPLIER,
                    )
                {
                    bootstrap_step_dumps.push(BootstrapStepDump {
                        traj_id: traj_id.to_string(),
                        night_id: night_id.0,
                        step_index: slot,
                        dt_since_last_real_update,
                        arc_span_since_bootstrap_days,
                        along_track_arcsec: diag.as_ref().and_then(|d| d.along_track_arcsec),
                        cross_track_arcsec: diag.as_ref().and_then(|d| d.cross_track_arcsec),
                        nis: diag.as_ref().and_then(|d| d.nis),
                        n_hypotheses: diag.as_ref().and_then(|d| d.n_hypotheses),
                        map_weight_fraction: diag.as_ref().and_then(|d| d.map_weight_fraction),
                        observation: obs.clone(),
                        hypotheses: cp.hypotheses().iter().map(Into::into).collect(),
                    });
                }
            }

            for _ in 0..nb_true_obs_in_night {
                // A ZTF night spans ~8h of distinct exposure epochs — when a
                // trajectory has several true observations the same night
                // (a normal tracklet), each one needs *its own* epoch and
                // observer geometry, not the first observation's. Reusing a
                // single per-night epoch/geometry for every observation
                // used to propagate the bank to the wrong instant and score
                // it against the wrong observer position — a spurious
                // mismatch the filter absorbed as if it were real
                // information, mostly into the poorly-observed ρ/ρ̇
                // components (erratic Kalman-gain jumps, eventually ρ
                // crossing zero into the ill-conditioned region and
                // diverging outright).
                let Some((obs_epoch, obs_r, obs_v)) = geometry_cache
                    .get(obs_dataset, context, current_obs)
                    .map(|(r, v)| (current_obs.mjd_tt(), r, v))
                else {
                    // No resolvable geometry for this specific exposure —
                    // skip just this observation rather than corrupt the
                    // branch with another epoch's geometry.
                    idx += 1;
                    if idx >= observations_to_process.len() {
                        break 'night_loop;
                    }
                    current_obs = &observations_to_process[idx];
                    continue;
                };

                // Always a fresh prediction to this specific observation's
                // own epoch — `branch` may have just been updated by a
                // previous iteration of this loop (multiple true
                // observations the same night), and even on the first
                // iteration the bank must be predicted to `current_obs`'s
                // own epoch, not shared across the whole night (see the
                // visits-loop doc above).
                let predicted_bank = branch.bank.predict_to(obs_epoch, obs_r, obs_v);

                let (result, n_gated, n_failed) = Branch::from_observation_diag(
                    &predicted_bank,
                    &branch,
                    current_obs,
                    0.,
                    step as u64,
                    step,
                );
                let Some(new_branch) = result else {
                    stop_reason = TruncationReason::from_gate_fail_counts(n_gated, n_failed);
                    let dump_budget_counter = match stop_reason {
                        TruncationReason::CollapsedByPropagation => {
                            Some(&dump_budgets.propagation_failures)
                        }
                        TruncationReason::DegenerateState => Some(&dump_budgets.degenerate_state),
                        _ => None,
                    };
                    // Gated by a shared budget *before* building the dump —
                    // constructing `FailureCaseDump` clones the bank's full
                    // hypothesis population (up to `cap_schedule`'s start
                    // value, hundreds of entries) plus `state_history`; most
                    // trajectories hitting this path will never have their
                    // dump written to disk (`write_failure_dump_file` only
                    // keeps `MAX_DUMPED_CASES_PER_REASON` of them), so
                    // skipping the allocation once the budget is spent is
                    // what keeps this bounded on a full-survey run instead
                    // of scaling with the (potentially tens-of-thousands)
                    // count of failing trajectories — see [`DumpBudgets`].
                    if let Some(counter) = dump_budget_counter
                        && DumpBudgets::try_reserve(
                            counter,
                            MAX_DUMPED_CASES_PER_REASON * DUMP_BUDGET_MULTIPLIER,
                        )
                    {
                        failure_dump = Some(FailureCaseDump {
                            traj_id: traj_id.to_string(),
                            night_id: night_id.0,
                            step,
                            n_real_updates_so_far: branch.n_real_updates,
                            n_gated,
                            n_failed,
                            observation: current_obs.clone(),
                            hypotheses: predicted_bank
                                .hypotheses()
                                .iter()
                                .map(Into::into)
                                .collect(),
                            pre_predict_hypotheses: branch
                                .bank
                                .hypotheses()
                                .iter()
                                .map(Into::into)
                                .collect(),
                            predict_epoch: obs_epoch,
                            predict_r_obs: [obs_r.x, obs_r.y, obs_r.z],
                            predict_v_obs: [obs_v.x, obs_v.y, obs_v.z],
                            bank_config: serde_json::to_value(bank_config)
                                .unwrap_or(serde_json::Value::Null),
                            state_history: state_history.clone(),
                        });
                    }
                    break 'night_loop;
                };
                branch = new_branch;

                if let Some(best) = branch.bank.best() {
                    let s = best.kf.state;
                    let rho_min_before = predicted_bank
                        .hypotheses()
                        .iter()
                        .map(|h| h.kf.state[4])
                        .fold(f64::INFINITY, f64::min);
                    let rho_max_before = predicted_bank
                        .hypotheses()
                        .iter()
                        .map(|h| h.kf.state[4])
                        .fold(f64::NEG_INFINITY, f64::max);
                    state_history.push(StateHistoryPoint {
                        step,
                        epoch: obs_epoch,
                        n_real_updates: branch.n_real_updates,
                        dt_since_last_real_update: obs_epoch - last_real_update_epoch,
                        n_hypotheses: branch.bank.hypotheses().len(),
                        n_hyp_before: predicted_bank.hypotheses().len(),
                        rho_min_before,
                        rho_max_before,
                        state: [s[0], s[1], s[2], s[3], s[4], s[5]],
                        p_rho_au2: best.kf.covariance[(4, 4)],
                        p_rho_dot_au2: best.kf.covariance[(5, 5)],
                        kalman_gain_norm: best.kf.kalman_gain.map(|k| k.norm()).unwrap_or(0.0),
                        nis_ema: best.kf.nis_ema.unwrap_or(f64::NAN),
                        p_cross_rho_ra: best.kf.covariance[(4, 0)],
                        p_asymmetry: (best.kf.covariance - best.kf.covariance.transpose())
                            .iter()
                            .fold(0.0f64, |acc, v| acc.max(v.abs())),
                        p_ra: best.kf.covariance[(0, 0)],
                        p_dec: best.kf.covariance[(1, 1)],
                        p_cross_rho_dec: best.kf.covariance[(4, 1)],
                        p_ra_dot: best.kf.covariance[(2, 2)],
                        p_dec_dot: best.kf.covariance[(3, 3)],
                    });
                }
                last_real_update_epoch = current_obs.mjd_tt();
                let c = current_obs.equ_coord();
                last_accepted = LastAccepted {
                    epoch: last_real_update_epoch,
                    ra: c.ra,
                    dec: c.dec,
                };

                idx += 1;
                if idx >= observations_to_process.len() {
                    break 'night_loop;
                }
                current_obs = &observations_to_process[idx];
            }
        }
    }

    let n_found = outcomes
        .iter()
        .filter(|o| o.reason == MissReason::Found)
        .count() as u32;
    let traj_coverage = (n_found as f32 / n_obs as f32) * 100.;

    Some(TrajStat {
        traj_id,
        n_obs_total,
        n_obs_true: n_obs,
        n_true_obs_added: n_found,
        coverage: traj_coverage,
        visit_stat: vec_visit_stat,
        outcomes,
        failure_dump,
        short_dt_dump,
        bootstrap_step_dumps,
        stop_reason,
        shadow,
    })
}

/// How far was `obs` (a real observation) from what `predicted_bank`
/// (already predicted to `obs`'s epoch — see callers) expected? Builds the
/// raw `(weight, KFState)` mixture directly from `predicted_bank`'s
/// hypotheses (no re-propagation — `predict_to` already did that) and
/// reuses `kalman_traj`'s own predictive-NIS/search-region reduction, the
/// same ones `kalman_full_traj_eval` uses, so the two binaries' numbers are
/// directly comparable.
///
/// Outcome of [`geometry_diagnostic`] — the geometric "how far off" numbers
/// for one (predicted mixture, true observation) pair.
struct GeometryDiagnostic {
    nis: Option<f64>,
    separation_arcsec: Option<f64>,
    region_radius_arcsec: Option<f64>,
    /// Component of `truth - predicted_center` along the mixture's
    /// weighted-mean apparent-motion direction (tangent-plane, `cos(dec)`
    /// de-projected). Positive = truth is ahead of the prediction along its
    /// own track. `None` whenever `separation_arcsec` is (degenerate region
    /// or zero-norm velocity, e.g. right at bootstrap).
    along_track_arcsec: Option<f64>,
    /// Component of `truth - predicted_center` perpendicular to the
    /// apparent-motion direction. Large `cross_track` relative to
    /// `along_track` points at a mismodeled motion *direction* (bad
    /// `(ρ, ρ̇)` mode, bad orbital-plane orientation); large `along_track`
    /// relative to `cross_track` points at a mismodeled motion *rate*
    /// (process noise / propagation timing) instead.
    cross_track_arcsec: Option<f64>,
    /// Number of surviving hypotheses in `predicted_bank` at the time of
    /// this diagnostic — a proxy for "has the (rho, rho_dot) ambiguity
    /// resolved yet". Used to distinguish a genuine along-track bias that
    /// persists even once the bank is confidently converged (points at
    /// missing two-body physics) from one concentrated in
    /// still-unconverged banks (points at a fit-convergence issue instead).
    n_hypotheses: Option<usize>,
    /// Fraction of the mixture's total weight held by its single
    /// highest-weight hypothesis (`exp(max log_weight) / sum(exp(log_weight))`)
    /// — `1.0` means the bank has fully collapsed onto one mode, low
    /// values mean several `(rho, rho_dot)` modes are still competitive.
    /// See `n_hypotheses`' doc for why this matters.
    map_weight_fraction: Option<f64>,
}

/// Weighted-mean apparent motion `(dRA/dt, dDec/dt)` (rad/day, RA rate
/// **not** `cos(dec)`-reduced — see `KFState`'s state-vector doc) across
/// `mixture`, plus the weighted-mean declination needed to de-project it
/// into a tangent-plane direction.
fn weighted_mean_motion(mixture: &[(f64, KFState<'_>)]) -> Option<(f64, f64, f64)> {
    let total_weight: f64 = mixture.iter().map(|(w, _)| w).sum();
    if !(total_weight > 0.0) {
        return None;
    }
    let (ra_dot, dec_dot, dec) = mixture
        .iter()
        .fold((0.0, 0.0, 0.0), |(a, d, dec), (w, kf)| {
            (
                a + w * kf.state[2],
                d + w * kf.state[3],
                dec + w * kf.state[1],
            )
        });
    Some((
        ra_dot / total_weight,
        dec_dot / total_weight,
        dec / total_weight,
    ))
}

/// Decompose `truth - center` (both in radians) into along-track/cross-track
/// arcsec components relative to the tangent-plane motion direction
/// `(ra_dot * cos(dec), dec_dot)`. `None` if the motion direction is
/// degenerate (zero norm, e.g. right at bootstrap before any rate is
/// constrained).
fn along_cross_track_arcsec(
    center_ra: f64,
    center_dec: f64,
    obs: &Observation,
    ra_dot: f64,
    dec_dot: f64,
    mean_dec: f64,
) -> Option<(f64, f64)> {
    let vx = ra_dot * mean_dec.cos();
    let vy = dec_dot;
    let v_norm = (vx * vx + vy * vy).sqrt();
    if !(v_norm > 0.0) {
        return None;
    }
    let (ux, uy) = (vx / v_norm, vy / v_norm);

    let truth = obs.equ_coord();
    let dx = wrap_angle(truth.ra - center_ra) * center_dec.cos();
    let dy = truth.dec - center_dec;

    let along_rad = dx * ux + dy * uy;
    let cross_rad = -dx * uy + dy * ux;
    const RAD_TO_ARCSEC: f64 = 206264.80624709636;
    Some((along_rad * RAD_TO_ARCSEC, cross_rad * RAD_TO_ARCSEC))
}

fn geometry_diagnostic(
    predicted_bank: &KFBank<'_, '_>,
    obs: &Observation,
    advance_params: &NightAdvanceParams,
    search_region_chi2: f64,
) -> GeometryDiagnostic {
    // Normalized, not raw `exp(log_weight)`: a freshly seeded bank carries
    // unnormalized population priors summing to ~90 (see
    // `TopK::keep_threshold`'s doc). Feeding those raw into
    // `compute_search_region` and `compute_predictive_nis` scales every
    // mixture covariance by that sum, so the reported region radius,
    // separation/radius ratio and NIS were all measuring a distribution that
    // does not integrate to 1.
    let mut mixture: Vec<(f64, KFState<'_>)> = predicted_bank
        .hypotheses()
        .iter()
        .map(|h| (h.log_weight.exp(), h.kf.clone()))
        .collect();
    let total_weight: f64 = mixture.iter().map(|(w, _)| *w).sum();
    if total_weight > 0.0 {
        mixture.iter_mut().for_each(|(w, _)| *w /= total_weight);
    }

    let nis = compute_predictive_nis(&mixture, obs);

    let region = compute_search_region(&mixture, advance_params, search_region_chi2);
    let separation_arcsec = region.as_ref().map(|r| {
        let center = EquCoord::new(r.center_ra, 0., r.center_dec, 0.);
        center.angular_separation(obs.equ_coord()).to_degrees() * 3600.0
    });
    let region_radius_arcsec = region.as_ref().map(|r| r.radius_rad.to_degrees() * 3600.0);

    let (along_track_arcsec, cross_track_arcsec) = region
        .as_ref()
        .zip(weighted_mean_motion(&mixture))
        .and_then(|(r, (ra_dot, dec_dot, mean_dec))| {
            along_cross_track_arcsec(r.center_ra, r.center_dec, obs, ra_dot, dec_dot, mean_dec)
        })
        .map_or((None, None), |(a, c)| (Some(a), Some(c)));

    let n_hypotheses = Some(mixture.len());
    let map_weight_fraction = {
        let total_weight: f64 = mixture.iter().map(|(w, _)| w).sum();
        let max_weight = mixture.iter().map(|(w, _)| *w).fold(0.0, f64::max);
        (total_weight > 0.0).then_some(max_weight / total_weight)
    };

    GeometryDiagnostic {
        nis,
        separation_arcsec,
        region_radius_arcsec,
        along_track_arcsec,
        cross_track_arcsec,
        n_hypotheses,
        map_weight_fraction,
    }
}

/// How many of `observations_to_process[idx..]` share `night_id` — a short
/// walk over the trajectory's own (typically tiny) observation list.
fn count_same_night(
    observations_to_process: &[Observation],
    night_map: &AHashMap<ObsId, NightId>,
    idx: usize,
    night_id: NightId,
) -> usize {
    let mut n = 0usize;
    while idx + n < observations_to_process.len()
        && night_map.get(observations_to_process[idx + n].id()) == Some(&night_id)
    {
        n += 1;
    }
    n
}

// ── Trajectory materialization ──────────────────────────────────────────

/// Fetch the observations of `night_id` from `obs_dataset` as a contiguous
/// slice, cloning only if the underlying storage has them split across
/// non-adjacent memory.
pub fn materialize_contiguous_night<'o>(
    obs_dataset: &'o ObsDataset,
    night_id: &NightId,
) -> Result<Cow<'o, [Observation]>, EngineError> {
    match obs_dataset.materialize_night(night_id).ok_or_else(|| {
        EngineError::FinkFat(FinkFatError::Message(format!(
            "failed to materialize night with id: {}",
            night_id
        )))
    })? {
        MemLayoutObservations::Contiguous(slice) => Ok(Cow::Borrowed(slice)),
        MemLayoutObservations::Split(vec_obs) => {
            Ok(Cow::Owned(vec_obs.iter().map(|o| (*o).clone()).collect()))
        }
    }
}

// ── Reporting ─────────────────────────────────────────────────────────────

fn print_mot_run_counters(counters: &MotCounters, n_trackable: usize) {
    println!("\n=== MOT dataset-wide run summary ===");
    println!(
        "  Trajectories scanned                  : {}",
        counters.n_total
    );
    println!(
        "  Too short (< 3 obs)                    : {}",
        counters.n_too_short
    );
    println!(
        "  Failed to materialize                  : {}",
        counters.n_materialize_failed
    );
    println!(
        "  Bootstrap failed / no processable obs  : {}",
        counters.n_no_result
    );
    println!("  Trackable by the Kalman bank           : {n_trackable}");
}

fn print_metric_row(name: &str, stats: &MetricStats) {
    println!("  {name:<34} {}", fmt_stats(stats));
}

// ── Miss-reason diagnostics: why coverage is low ────────────────────────

/// Print the dataset-wide breakdown of [`MissReason`] over every real
/// observation of every trackable trajectory — answers "where is the MOT
/// re-association actually losing true observations?" as opposed to just
/// the aggregate coverage percentage.
fn print_miss_reason_histogram(stats: &[TrajStat]) {
    let mut counts = [0u64; 5];
    let mut total = 0u64;
    for s in stats {
        for o in &s.outcomes {
            counts[o.reason.bucket_index()] += 1;
            total += 1;
        }
    }

    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] Why true observations were (not) recovered — {total} real observations");
    println!("{sep}");
    if total == 0 {
        println!("  (no real observation processed)");
        return;
    }
    for reason in MissReason::all() {
        let count = counts[reason.bucket_index()];
        println!(
            "  {:<48} {:>10}  ({:>5.1}%)",
            reason.label(),
            count,
            100.0 * count as f64 / total as f64
        );
    }
}

/// Print the dataset-wide breakdown of [`TruncationReason`] over every
/// trackable trajectory — answers "the trajectories that die, why do they
/// die?" (as opposed to [`print_miss_reason_histogram`], which is about
/// individual missed *observations*, not whole trajectories).
fn print_truncation_reason_histogram(stats: &[TrajStat]) {
    let mut counts = [0u64; 4];
    for s in stats {
        counts[s.stop_reason.bucket_index()] += 1;
    }
    let total = stats.len() as u64;

    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] Why trajectories stop early — {total} trackable trajectories");
    println!("{sep}");
    for reason in TruncationReason::all() {
        let count = counts[reason.bucket_index()];
        println!(
            "  {:<48} {:>10}  ({:>5.1}%)",
            reason.label(),
            count,
            100.0 * count as f64 / total.max(1) as f64
        );
    }
}

/// Number of leading steps-since-bootstrap tracked individually by
/// [`print_miss_reason_by_step`] before folding the rest into one overflow
/// bucket — mirrors `trajectory_processing::NIS_STEP_BUCKET_DEPTH`.
const MISS_REASON_STEP_BUCKET_DEPTH: usize = 30;

/// Print [`MissReason`] counts bucketed by "steps since the bootstrap pair"
/// (0-based position within `observations_to_process`) — answers "is the
/// loss concentrated right after bootstrap (bank still a wide, unconverged
/// grid mixture) or spread evenly across the whole arc?".
fn print_miss_reason_by_step(stats: &[TrajStat]) {
    let mut buckets = vec![[0u64; 5]; MISS_REASON_STEP_BUCKET_DEPTH + 1];

    for s in stats {
        for (i, o) in s.outcomes.iter().enumerate() {
            buckets[i.min(MISS_REASON_STEP_BUCKET_DEPTH)][o.reason.bucket_index()] += 1;
        }
    }

    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] Miss reasons by steps-since-bootstrap");
    println!("{sep}");
    println!(
        "  {:>6}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}",
        "step", "n_samples", "found%", "no_prefilt%", "not_matched%", "no_geom%", "not_reach%"
    );
    for (i, counts) in buckets.iter().enumerate() {
        let n: u64 = counts.iter().sum();
        if n == 0 {
            continue;
        }
        let label = if i >= MISS_REASON_STEP_BUCKET_DEPTH {
            format!("{MISS_REASON_STEP_BUCKET_DEPTH}+")
        } else {
            (i + 1).to_string()
        };
        let pct = |c: u64| 100.0 * c as f64 / n as f64;
        println!(
            "  {:>6}  {:>10}  {:>8.1}  {:>10.1}  {:>10.1}  {:>8.1}  {:>10.1}",
            label,
            n,
            pct(counts[MissReason::Found.bucket_index()]),
            pct(counts[MissReason::NoVisitPassedPrefilter.bucket_index()]),
            pct(counts[MissReason::SearchedButNotMatched.bucket_index()]),
            pct(counts[MissReason::GeometryUnavailable.bucket_index()]),
            pct(counts[MissReason::NotReached.bucket_index()]),
        );
    }
    println!("{sep}");
}

/// Print, for each classifiable [`MissReason`], the distribution of:
/// - `dt_since_last_real_update` (days) — if the coarse prefilter's *fixed*
///   search radius is the dominant cause of
///   [`MissReason::NoVisitPassedPrefilter`], its median `dt` should be
///   visibly larger than [`MissReason::Found`]'s;
/// - predictive NIS — χ²(2) if well-calibrated (expected ≈ 2.0); the
///   configured chi-square gate (`gate_chi2`, 23.0 in the active config) is
///   the rough boundary between "marginal" and "the orbit has genuinely
///   diverged";
/// - the angular separation (arcsec) between the true observation and what
///   `branch`'s own prediction expected;
/// - the predicted search region's radius (arcsec) at that epoch;
/// - separation/radius — "how many box-widths away was the truth";
/// - how many real observations `branch` had already consumed before this
///   one — misses concentrated at low values mean "not enough points yet
///   to converge"; misses spread across high values too mean the lineage
///   diverges even once well-constrained.
///
/// This is the geometric "how far off" complement to
/// [`print_miss_reason_histogram`]'s "how often".
fn print_geometry_diagnostics_by_reason(stats: &[TrajStat]) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] Kalman prediction vs. truth, by outcome");
    println!("{sep}");

    for reason in [
        MissReason::Found,
        MissReason::NoVisitPassedPrefilter,
        MissReason::SearchedButNotMatched,
        MissReason::GeometryUnavailable,
    ] {
        println!("\n  ── {} ──", reason.label());

        let outcomes: Vec<&ObsOutcome> = stats
            .iter()
            .flat_map(|s| s.outcomes.iter())
            .filter(|o| o.reason == reason)
            .collect();
        if outcomes.is_empty() {
            println!("    (none)");
            continue;
        }

        let dt: Vec<f64> = outcomes
            .iter()
            .map(|o| o.dt_since_last_real_update)
            .filter(|v| v.is_finite())
            .collect();
        let n_real_updates: Vec<f64> = outcomes
            .iter()
            .map(|o| o.n_real_updates_so_far as f64)
            .collect();
        let nis: Vec<f64> = outcomes.iter().filter_map(|o| o.nis).collect();
        let sep_arcsec: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.separation_arcsec)
            .collect();
        let radius_arcsec: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.region_radius_arcsec)
            .collect();
        let ratio: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| match (o.separation_arcsec, o.region_radius_arcsec) {
                (Some(s), Some(r)) if r > 0.0 => Some(s / r),
                _ => None,
            })
            .collect();
        let along_track: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.along_track_arcsec)
            .collect();
        let cross_track: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.cross_track_arcsec)
            .collect();
        let along_over_cross: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| match (o.along_track_arcsec, o.cross_track_arcsec) {
                (Some(a), Some(c)) if c.abs() > 0.0 => Some(a.abs() / c.abs()),
                _ => None,
            })
            .collect();

        if !dt.is_empty() {
            print_metric_row("Days since last real update", &metric_stats(&dt, |v| *v));
        }
        print_metric_row(
            "Real updates consumed so far",
            &metric_stats(&n_real_updates, |v| *v),
        );
        if !nis.is_empty() {
            print_metric_row(
                "Predictive NIS (χ²(2), exp. ≈2.0)",
                &metric_stats(&nis, |v| *v),
            );
        }
        if !sep_arcsec.is_empty() {
            print_metric_row(
                "Separation truth ↔ predicted center (\")",
                &metric_stats(&sep_arcsec, |v| *v),
            );
        }
        if !radius_arcsec.is_empty() {
            print_metric_row(
                "Predicted region radius (\")",
                &metric_stats(&radius_arcsec, |v| *v),
            );
        }
        if !ratio.is_empty() {
            print_metric_row(
                "Separation / region radius (>1 = outside box)",
                &metric_stats(&ratio, |v| *v),
            );
        }
        if !along_track.is_empty() {
            print_metric_row(
                "Along-track error (\", signed)",
                &metric_stats(&along_track, |v| *v),
            );
        }
        if !cross_track.is_empty() {
            print_metric_row(
                "Cross-track error (\", signed)",
                &metric_stats(&cross_track, |v| *v),
            );
        }
        let n_hyp: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.n_hypotheses)
            .map(|n| n as f64)
            .collect();
        let map_weight: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.map_weight_fraction)
            .collect();
        // High confidence = bank essentially collapsed onto one mode
        // (few surviving hypotheses AND that MAP hypothesis dominates the
        // weight) — see `n_hypotheses`'/`map_weight_fraction`'s doc on
        // `ObsOutcome` for why this discriminates "missing physics" from
        // "fit not converged yet".
        let is_confident = |o: &ObsOutcome| {
            matches!(
                (o.n_hypotheses, o.map_weight_fraction),
                (Some(n), Some(w)) if n <= 3 && w > 0.9
            )
        };
        let along_track_confident: Vec<f64> = outcomes
            .iter()
            .filter(|o| is_confident(o))
            .filter_map(|o| o.along_track_arcsec)
            .collect();
        let along_track_unconverged: Vec<f64> = outcomes
            .iter()
            .filter(|o| !is_confident(o))
            .filter_map(|o| o.along_track_arcsec)
            .collect();

        if !along_over_cross.is_empty() {
            print_metric_row(
                "|Along| / |cross| (>1 = along-track dominated)",
                &metric_stats(&along_over_cross, |v| *v),
            );
        }
        if !n_hyp.is_empty() {
            print_metric_row(
                "Surviving hypotheses (bank size)",
                &metric_stats(&n_hyp, |v| *v),
            );
        }
        if !map_weight.is_empty() {
            print_metric_row(
                "MAP hypothesis weight fraction",
                &metric_stats(&map_weight, |v| *v),
            );
        }
        if !along_track_confident.is_empty() {
            print_metric_row(
                "  Along-track (\"), confident bank (<=3 hyp, MAP>90%)",
                &metric_stats(&along_track_confident, |v| *v),
            );
        }
        if !along_track_unconverged.is_empty() {
            print_metric_row(
                "  Along-track (\"), unconverged bank",
                &metric_stats(&along_track_unconverged, |v| *v),
            );
        }
    }
    println!("{sep}");
}

/// Break `SearchedButNotMatched`'s along-track bias down by orbital-class
/// population (see [`Population`]) — a genuine unmodeled-perturbation
/// secular drift should correlate with dynamical exposure to planetary
/// perturbations (heliocentric distance, eccentricity, resonance
/// proximity) and therefore hit some populations harder than others; a
/// fit-convergence issue has no particular reason to follow this
/// breakdown. Empty (or all-`Unknown`) when `--ground-truth` wasn't
/// passed — `traj_population` is then an empty map and every trajectory
/// classifies as [`Population::Unknown`].
fn print_along_track_bias_by_population(
    stats: &[TrajStat],
    traj_population: &AHashMap<TrajId, Population>,
) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] SearchedButNotMatched along-track bias, by orbital-class population");
    println!("{sep}");

    for population in Population::all() {
        let along_track: Vec<f64> = stats
            .iter()
            .filter(|s| {
                traj_population
                    .get(&s.traj_id)
                    .copied()
                    .unwrap_or(Population::Unknown)
                    == population
            })
            .flat_map(|s| s.outcomes.iter())
            .filter(|o| o.reason == MissReason::SearchedButNotMatched)
            .filter_map(|o| o.along_track_arcsec)
            .collect();

        if along_track.is_empty() {
            continue;
        }
        println!(
            "  {:<24} n={:<8} {}",
            population.label(),
            along_track.len(),
            fmt_stats(&metric_stats(&along_track, |v| *v))
        );
    }
    println!("{sep}");
}

/// `dt_since_last_real_update` threshold (days) below which the n-body
/// secular-drift mechanism confirmed in
/// `crates/fink-fat-engine/tests/two_body_vs_perturbed_drift.rs` is far
/// too small (<2" at 30 days for NEO/MBA) to explain the along-track bias
/// observed in `mot_analysis` — cases under this threshold need a
/// different (or differently-triggered) explanation. See
/// [`print_short_dt_not_matched_diagnostics`].
const SHORT_DT_THRESHOLD_DAYS: f64 = 30.0;

/// Bucket a raw `f64` sample into a [`Histogram`] — like
/// [`AggStat::from_sample`]'s private `histogram` helper, but without its
/// `T: Ord` bound (signed arcsec values can't implement `Ord` due to
/// `NaN`, so `AggStat<f64>` isn't usable here).
/// Bucket a raw `f64` sample into a [`Histogram`] using **robust**
/// (percentile) bounds rather than raw min/max — a handful of degenerate/
/// divergent outliers (this dataset has along-track values up to
/// ±647000″) would otherwise stretch the bin range so far that every bin
/// near the actual median collapses into one giant bar, hiding the shape
/// of the typical-case distribution entirely. Values outside
/// `[lo_percentile, hi_percentile]` are clipped into the first/last bin
/// (every sample is still counted, just resolution is spent where the
/// bulk of the data actually is).
fn histogram_f64(
    sample: &[f64],
    n_bins: usize,
    lo_percentile: f64,
    hi_percentile: f64,
) -> Histogram {
    let mut sorted: Vec<f64> = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let lo = percentile(lo_percentile);
    let hi = percentile(hi_percentile);

    let width = (hi - lo) / n_bins as f64;
    let edges: Vec<f64> = (0..=n_bins).map(|i| lo + width * i as f64).collect();
    let mut counts = vec![0usize; n_bins];
    if width == 0.0 || !width.is_finite() {
        counts[0] = sample.len();
        return Histogram { edges, counts };
    }
    for &x in sample {
        let idx = (((x - lo) / width) as usize).clamp(0, n_bins - 1);
        counts[idx] += 1;
    }
    Histogram { edges, counts }
}

/// Deep-dive on `SearchedButNotMatched` cases whose last real update was
/// less than [`SHORT_DT_THRESHOLD_DAYS`] ago — the regime where the
/// confirmed n-body secular-drift mechanism cannot be the (whole)
/// explanation, so this breaks the along-track bias down along every axis
/// already instrumented to let the data point at whatever else is going
/// on, rather than presupposing an answer.
fn print_short_dt_not_matched_diagnostics(
    stats: &[TrajStat],
    traj_population: &AHashMap<TrajId, Population>,
) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!(
        "[Global] SearchedButNotMatched, dt_since_last_real_update < {SHORT_DT_THRESHOLD_DAYS:.0}d \
         — short-dt regime investigation"
    );
    println!("{sep}");

    struct Row<'a> {
        outcome: &'a ObsOutcome,
        traj_id: &'a TrajId,
    }
    let rows: Vec<Row> = stats
        .iter()
        .flat_map(|s| {
            s.outcomes.iter().map(move |o| Row {
                outcome: o,
                traj_id: &s.traj_id,
            })
        })
        .filter(|r| r.outcome.reason == MissReason::SearchedButNotMatched)
        .filter(|r| r.outcome.dt_since_last_real_update < SHORT_DT_THRESHOLD_DAYS)
        .collect();

    println!("  n = {}", rows.len());
    if rows.is_empty() {
        println!("  (none)");
        println!("{sep}");
        return;
    }

    let along: Vec<f64> = rows
        .iter()
        .filter_map(|r| r.outcome.along_track_arcsec)
        .collect();
    let cross: Vec<f64> = rows
        .iter()
        .filter_map(|r| r.outcome.cross_track_arcsec)
        .collect();
    if !along.is_empty() {
        print_metric_row("Along-track (\", signed)", &metric_stats(&along, |v| *v));
    }
    if !cross.is_empty() {
        print_metric_row("Cross-track (\", signed)", &metric_stats(&cross, |v| *v));
    }
    if !along.is_empty() {
        println!("\n  Along-track error histogram (signed, arcsec) — sign consistency check:");
        print!("{}", histogram_f64(&along, 40, 1.0, 99.0));
    }

    println!("\n  -- By arc span since bootstrap (total two-body arc length) --");
    for (label, lo, hi) in [
        ("<7d", 0.0, 7.0),
        ("7-30d", 7.0, 30.0),
        ("30-90d", 30.0, 90.0),
        ("90d+", 90.0, f64::INFINITY),
    ] {
        let subset: Vec<f64> = rows
            .iter()
            .filter(|r| {
                r.outcome.arc_span_since_bootstrap_days >= lo
                    && r.outcome.arc_span_since_bootstrap_days < hi
            })
            .filter_map(|r| r.outcome.along_track_arcsec)
            .collect();
        if subset.is_empty() {
            continue;
        }
        println!(
            "    {label:<8} n={:<8} {}",
            subset.len(),
            fmt_stats(&metric_stats(&subset, |v| *v))
        );
    }

    println!("\n  -- By real updates consumed so far --");
    for (label, lo, hi) in [
        ("1-5", 1usize, 5usize),
        ("6-15", 6, 15),
        ("16-30", 16, 30),
        ("31+", 31, usize::MAX),
    ] {
        let subset: Vec<f64> = rows
            .iter()
            .filter(|r| {
                r.outcome.n_real_updates_so_far >= lo && r.outcome.n_real_updates_so_far <= hi
            })
            .filter_map(|r| r.outcome.along_track_arcsec)
            .collect();
        if subset.is_empty() {
            continue;
        }
        println!(
            "    {label:<8} n={:<8} {}",
            subset.len(),
            fmt_stats(&metric_stats(&subset, |v| *v))
        );
    }

    println!("\n  -- By orbital-class population --");
    for population in Population::all() {
        let subset: Vec<f64> = rows
            .iter()
            .filter(|r| {
                traj_population
                    .get(r.traj_id)
                    .copied()
                    .unwrap_or(Population::Unknown)
                    == population
            })
            .filter_map(|r| r.outcome.along_track_arcsec)
            .collect();
        if subset.is_empty() {
            continue;
        }
        println!(
            "    {:<24} n={:<8} {}",
            population.label(),
            subset.len(),
            fmt_stats(&metric_stats(&subset, |v| *v))
        );
    }

    println!("\n  -- By bank confidence --");
    let is_confident = |o: &ObsOutcome| {
        matches!(
            (o.n_hypotheses, o.map_weight_fraction),
            (Some(n), Some(w)) if n <= 3 && w > 0.9
        )
    };
    let confident: Vec<f64> = rows
        .iter()
        .filter(|r| is_confident(r.outcome))
        .filter_map(|r| r.outcome.along_track_arcsec)
        .collect();
    let unconverged: Vec<f64> = rows
        .iter()
        .filter(|r| !is_confident(r.outcome))
        .filter_map(|r| r.outcome.along_track_arcsec)
        .collect();
    if !confident.is_empty() {
        println!(
            "    confident   n={:<8} {}",
            confident.len(),
            fmt_stats(&metric_stats(&confident, |v| *v))
        );
    }
    if !unconverged.is_empty() {
        println!(
            "    unconverged n={:<8} {}",
            unconverged.len(),
            fmt_stats(&metric_stats(&unconverged, |v| *v))
        );
    }

    let nis: Vec<f64> = rows.iter().filter_map(|r| r.outcome.nis).collect();
    if !nis.is_empty() {
        print_metric_row("\n  Predictive NIS", &metric_stats(&nis, |v| *v));
    }
    let ratio: Vec<f64> = rows
        .iter()
        .filter_map(
            |r| match (r.outcome.separation_arcsec, r.outcome.region_radius_arcsec) {
                (Some(s), Some(rad)) if rad > 0.0 => Some(s / rad),
                _ => None,
            },
        )
        .collect();
    if !ratio.is_empty() {
        print_metric_row(
            "  Separation / region radius",
            &metric_stats(&ratio, |v| *v),
        );
    }

    println!("{sep}");
}

/// Deep-dive on `SearchedButNotMatched` cases at the very first
/// (`n_real_updates_so_far == 0`, "step 1") and second
/// (`== 1`, "step 2") real update after the bootstrap pair — after fixing
/// the per-visit epoch bug (see `study_kalman_asteroid`'s doc), these two
/// steps dominate the residual `not_matched%` (27.3%/12.8% respectively,
/// vs ≤3% from step 3 onward). Distinguishes:
/// - inherent (ρ, ρ̇) ambiguity right after a 2-point bootstrap (expected,
///   not a bug) from
/// - a still-undiagnosed systematic effect (if along-track stays
///   dominant and/or bank confidence stays high despite the bank being
///   this fresh, that would be surprising and worth a fresh look).
fn print_bootstrap_residual_diagnostics(
    stats: &[TrajStat],
    traj_population: &AHashMap<TrajId, Population>,
) {
    struct Row<'a> {
        outcome: &'a ObsOutcome,
        traj_id: &'a TrajId,
        /// Position within the trajectory's own `outcomes` (0-based —
        /// `step` label `i+1`, same convention as
        /// `print_miss_reason_by_step`). **Not** the same thing as
        /// `outcome.n_real_updates_so_far`: `Branch::seed` counts the
        /// bootstrap pair itself as 1 real update, so the first
        /// post-bootstrap observation (`i == 0`) already has
        /// `n_real_updates_so_far == 1` — and that field only advances on
        /// an actual "Found", so a missed step 1 leaves step 2 at the same
        /// count. Position in the sequence is the only reliable way to
        /// isolate "the Nth post-bootstrap observation".
        step_index: usize,
    }

    fn print_breakdown(label: &str, rows: &[&Row], traj_population: &AHashMap<TrajId, Population>) {
        println!("\n  ── {label} ──");
        println!("    n = {}", rows.len());
        if rows.is_empty() {
            println!("    (none)");
            return;
        }

        let along: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.outcome.along_track_arcsec)
            .collect();
        let cross: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.outcome.cross_track_arcsec)
            .collect();
        if !along.is_empty() {
            print_metric_row(
                "    Along-track (\", signed)",
                &metric_stats(&along, |v| *v),
            );
        }
        if !cross.is_empty() {
            print_metric_row(
                "    Cross-track (\", signed)",
                &metric_stats(&cross, |v| *v),
            );
        }
        if !along.is_empty() {
            println!("\n    Along-track error histogram (signed, arcsec):");
            print!("{}", histogram_f64(&along, 30, 1.0, 99.0));
        }

        println!("\n    -- By orbital-class population --");
        for population in Population::all() {
            let subset: Vec<f64> = rows
                .iter()
                .filter(|r| {
                    traj_population
                        .get(r.traj_id)
                        .copied()
                        .unwrap_or(Population::Unknown)
                        == population
                })
                .filter_map(|r| r.outcome.along_track_arcsec)
                .collect();
            if subset.is_empty() {
                continue;
            }
            println!(
                "      {:<24} n={:<8} {}",
                population.label(),
                subset.len(),
                fmt_stats(&metric_stats(&subset, |v| *v))
            );
        }

        println!("\n    -- By bank confidence --");
        let is_confident = |o: &ObsOutcome| {
            matches!(
                (o.n_hypotheses, o.map_weight_fraction),
                (Some(n), Some(w)) if n <= 3 && w > 0.9
            )
        };
        let confident: Vec<f64> = rows
            .iter()
            .filter(|r| is_confident(r.outcome))
            .filter_map(|r| r.outcome.along_track_arcsec)
            .collect();
        let unconverged: Vec<f64> = rows
            .iter()
            .filter(|r| !is_confident(r.outcome))
            .filter_map(|r| r.outcome.along_track_arcsec)
            .collect();
        if !confident.is_empty() {
            println!(
                "      confident   n={:<8} {}",
                confident.len(),
                fmt_stats(&metric_stats(&confident, |v| *v))
            );
        }
        if !unconverged.is_empty() {
            println!(
                "      unconverged n={:<8} {}",
                unconverged.len(),
                fmt_stats(&metric_stats(&unconverged, |v| *v))
            );
        }

        let n_hyp: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.outcome.n_hypotheses)
            .map(|n| n as f64)
            .collect();
        if !n_hyp.is_empty() {
            print_metric_row(
                "    Surviving hypotheses (bank size)",
                &metric_stats(&n_hyp, |v| *v),
            );
        }
        // Continuous weight concentration, distinct from the binary
        // "confident" split above (<=3 hyp AND MAP>90%, which can stay
        // flat step-to-step even while the underlying weights are
        // visibly sharpening) — see this function's doc for why this
        // matters for telling a weighting effect from a pruning effect.
        let map_weight: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.outcome.map_weight_fraction)
            .collect();
        if !map_weight.is_empty() {
            print_metric_row(
                "    MAP hypothesis weight fraction",
                &metric_stats(&map_weight, |v| *v),
            );
        }
        let dt: Vec<f64> = rows
            .iter()
            .map(|r| r.outcome.dt_since_last_real_update)
            .filter(|v| v.is_finite())
            .collect();
        if !dt.is_empty() {
            print_metric_row(
                "    Days since last real update",
                &metric_stats(&dt, |v| *v),
            );
        }
        let arc_span: Vec<f64> = rows
            .iter()
            .map(|r| r.outcome.arc_span_since_bootstrap_days)
            .filter(|v| v.is_finite())
            .collect();
        if !arc_span.is_empty() {
            print_metric_row(
                "    Arc span since bootstrap (days)",
                &metric_stats(&arc_span, |v| *v),
            );
        }
        let nis: Vec<f64> = rows.iter().filter_map(|r| r.outcome.nis).collect();
        if !nis.is_empty() {
            print_metric_row("    Predictive NIS", &metric_stats(&nis, |v| *v));
        }
        let ratio: Vec<f64> = rows
            .iter()
            .filter_map(
                |r| match (r.outcome.separation_arcsec, r.outcome.region_radius_arcsec) {
                    (Some(s), Some(rad)) if rad > 0.0 => Some(s / rad),
                    _ => None,
                },
            )
            .collect();
        if !ratio.is_empty() {
            print_metric_row(
                "    Separation / region radius",
                &metric_stats(&ratio, |v| *v),
            );
        }
    }

    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] SearchedButNotMatched — bootstrap residual (step 1 / step 2)");
    println!("{sep}");

    let all_rows: Vec<Row> = stats
        .iter()
        .flat_map(|s| {
            s.outcomes
                .iter()
                .enumerate()
                .map(move |(step_index, o)| Row {
                    outcome: o,
                    traj_id: &s.traj_id,
                    step_index,
                })
        })
        .filter(|r| r.outcome.reason == MissReason::SearchedButNotMatched)
        .collect();

    let step1: Vec<&Row> = all_rows.iter().filter(|r| r.step_index == 0).collect();
    let step2: Vec<&Row> = all_rows.iter().filter(|r| r.step_index == 1).collect();

    print_breakdown(
        "step 1 (first post-bootstrap observation)",
        &step1,
        traj_population,
    );
    print_breakdown(
        "step 2 (second post-bootstrap observation)",
        &step2,
        traj_population,
    );
    println!("{sep}");
}

fn print_mot_global_stats(stats: &[TrajStat]) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!(
        "[Global] MOT candidate contamination over {} trackable trajectories",
        stats.len()
    );
    println!("{sep}");

    let total_visits: usize = stats.iter().map(TrajStat::n_visits).sum();
    let total_visits_with_true: usize = stats.iter().map(TrajStat::n_visits_with_true).sum();
    let total_candidates: u64 = stats.iter().map(TrajStat::total_candidates).sum();
    let total_bad: u64 = stats.iter().map(TrajStat::total_bad).sum();

    println!("  Visits examined                       : {total_visits}");
    println!(
        "  Visits containing the true observation : {total_visits_with_true} ({:.1}%)",
        100.0 * total_visits_with_true as f64 / total_visits.max(1) as f64
    );
    println!("  Candidates found in error boxes        : {total_candidates}");
    println!(
        "  Bad candidates (non-true)              : {total_bad} ({:.1}% of all candidates)",
        100.0 * total_bad as f64 / total_candidates.max(1) as f64
    );

    println!("{sep}");
    print_metric_row(
        "Coverage (%) per trajectory",
        &metric_stats(stats, |s| s.coverage as f64),
    );
    print_metric_row(
        "Mean bad candidates / visit (per traj)",
        &metric_stats(stats, TrajStat::mean_bad_per_visit),
    );
    print_metric_row(
        "Bad candidate rate (%) (per traj)",
        &metric_stats(stats, TrajStat::bad_candidate_rate_pct),
    );

    let mut n_cand_with_true: Vec<u32> = Vec::new();
    let mut n_cand_without_true: Vec<u32> = Vec::new();
    for s in stats {
        for v in &s.visit_stat {
            if v.true_in_visit {
                n_cand_with_true.push(v.n_candidates);
            } else {
                n_cand_without_true.push(v.n_candidates);
            }
        }
    }

    println!("\n[Global] Candidates per visit — visits WITH the true observation");
    if n_cand_with_true.is_empty() {
        println!("  (none)");
    } else {
        print!("{}", AggStat::from_sample(&n_cand_with_true, Some(20)));
    }

    println!("\n[Global] Candidates per visit — visits WITHOUT the true observation");
    if n_cand_without_true.is_empty() {
        println!("  (none)");
    } else {
        print!("{}", AggStat::from_sample(&n_cand_without_true, Some(20)));
    }

    // ── Same histograms after the default filter cascade ────────────────
    let mut n_cascade_with_true: Vec<u32> = Vec::new();
    let mut n_cascade_without_true: Vec<u32> = Vec::new();
    let mut n_true_lost_visits = 0usize;
    let mut total_candidates_cascade = 0u64;
    let mut total_bad_cascade = 0u64;
    let mut max_candidates_cascade = 0u32;
    for s in stats {
        for v in &s.visit_stat {
            total_candidates_cascade += v.n_candidates_cascade as u64;
            total_bad_cascade += v.n_bad_cascade as u64;
            max_candidates_cascade = max_candidates_cascade.max(v.n_candidates_cascade);
            if v.true_in_visit {
                n_cascade_with_true.push(v.n_candidates_cascade);
                if !v.true_in_visit_cascade {
                    n_true_lost_visits += 1;
                }
            } else {
                n_cascade_without_true.push(v.n_candidates_cascade);
            }
        }
    }

    println!("\n[Global] AFTER default cascade — summary");
    println!("  Candidates surviving the cascade       : {total_candidates_cascade}");
    println!(
        "  Bad candidates surviving               : {total_bad_cascade} ({:.1}% of survivors)",
        100.0 * total_bad_cascade as f64 / total_candidates_cascade.max(1) as f64
    );
    println!("  Max candidates in a single visit AFTER cascade : {max_candidates_cascade}");
    println!(
        "  ⚠️  Visits where the TRUE obs was removed : {n_true_lost_visits} (recall loss — must be ~0)"
    );

    println!("\n[Global] AFTER cascade — candidates per visit, visits WITH the true observation");
    if n_cascade_with_true.is_empty() {
        println!("  (none)");
    } else {
        print!("{}", AggStat::from_sample(&n_cascade_with_true, Some(20)));
    }

    println!(
        "\n[Global] AFTER cascade — candidates per visit, visits WITHOUT the true observation"
    );
    if n_cascade_without_true.is_empty() {
        println!("  (none)");
    } else {
        print!(
            "{}",
            AggStat::from_sample(&n_cascade_without_true, Some(20))
        );
    }
}

/// Dataset-wide shadow table of every candidate filter (plus the default
/// cascade), and the per-population breakdown of `removed_true` — the recall
/// guard, overall and per orbital class so exotic populations aren't
/// silently sacrificed to MBA-driven aggregates.
fn print_candidate_filter_diagnostics(
    stats: &[TrajStat],
    suite: &[Box<dyn CandidateFilter>],
    traj_population: &AHashMap<TrajId, Population>,
) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Global] Candidate-filter shadow study (each filter sees the same raw candidates)");
    println!("{sep}");

    let mut merged = ShadowStats::new(suite.len());
    for s in stats {
        merged.absorb(&s.shadow);
    }

    let total_candidates: u64 = stats.iter().map(TrajStat::total_candidates).sum();
    let total_bad: u64 = stats.iter().map(TrajStat::total_bad).sum();
    let total_true = total_candidates - total_bad;

    println!(
        "  {:<22} {:>10} {:>9} {:>12} {:>13} {:>13} {:>11}",
        "Filter", "cand_in", "removed", "removed_bad", "removed_TRUE", "bad_removed%", "true_lost%"
    );
    let print_row = |name: &str, c: &fink_fat_eval::candidate_filters::FilterCounters| {
        println!(
            "  {:<22} {:>10} {:>9} {:>12} {:>13} {:>12.2} {:>10.4}",
            name,
            c.cand_in,
            c.removed,
            c.removed_bad,
            c.removed_true,
            100.0 * c.removed_bad as f64 / total_bad.max(1) as f64,
            100.0 * c.removed_true as f64 / total_true.max(1) as f64,
        );
    };
    for (filter, counters) in suite.iter().zip(&merged.per_filter) {
        print_row(&filter.name(), counters);
    }
    let cascade_names: Vec<String> = DEFAULT_CASCADE
        .iter()
        .filter_map(|&i| suite.get(i).map(|f| f.name()))
        .collect();
    print_row("CASCADE", &merged.cascade);
    println!("  (cascade = {})", cascade_names.join(" ∧ "));

    // removed_TRUE per orbital-class population — only for filters that
    // actually removed true observations somewhere.
    let populations = Population::all();
    println!("\n  removed_TRUE by population (filters with removed_TRUE > 0 only):");
    println!(
        "  {:<22} {}",
        "Filter",
        populations
            .iter()
            .map(|p| format!("{:>12}", p.label().split('/').next().unwrap_or(p.label())))
            .collect::<String>()
    );
    let mut any_row = false;
    for (fi, filter) in suite.iter().enumerate() {
        if merged.per_filter[fi].removed_true == 0 {
            continue;
        }
        any_row = true;
        let mut per_pop = [0u64; 7];
        for s in stats {
            let removed_true = s.shadow.per_filter.get(fi).map_or(0, |c| c.removed_true);
            if removed_true == 0 {
                continue;
            }
            let population = traj_population
                .get(&s.traj_id)
                .copied()
                .unwrap_or(Population::Unknown);
            let slot = populations
                .iter()
                .position(|p| *p == population)
                .unwrap_or(6);
            per_pop[slot] += removed_true;
        }
        println!(
            "  {:<22} {}",
            filter.name(),
            per_pop
                .iter()
                .map(|n| format!("{n:>12}"))
                .collect::<String>()
        );
    }
    if !any_row {
        println!("  (no filter removed any true observation)");
    }
    println!("{sep}");
}

/// Rank trajectories by mean bad-candidates-per-visit (descending) and
/// return the `n` worst — trajectories with zero visits are excluded (no
/// contamination signal to rank on).
fn select_worst_by_bad_rate(stats: &[TrajStat], n: usize) -> Vec<&TrajStat> {
    let mut ranked: Vec<&TrajStat> = stats.iter().filter(|s| s.n_visits() > 0).collect();
    ranked.sort_by(|a, b| {
        b.mean_bad_per_visit()
            .partial_cmp(&a.mean_bad_per_visit())
            .unwrap_or(Ordering::Equal)
    });
    ranked.into_iter().take(n).collect()
}

fn print_worst_trajectories_table(items: &[&TrajStat]) {
    println!("\n--- ⚠️  Worst trajectories (highest mean bad candidates / visit) ---");
    if items.is_empty() {
        println!("  (none)");
        return;
    }
    println!(
        "{:>12}  {:>6}  {:>7}  {:>8}  {:>7}  {:>7}  {:>10}  {:>9}  {:>7}",
        "traj_id",
        "n_obs",
        "n_visit",
        "n_v_true",
        "n_cand",
        "n_bad",
        "bad/visit",
        "bad_rate%",
        "cov%"
    );
    for s in items {
        println!(
            "{:>12}  {:>6}  {:>7}  {:>8}  {:>7}  {:>7}  {:>10.3}  {:>9.1}  {:>7.1}",
            s.traj_id,
            s.n_obs_total,
            s.n_visits(),
            s.n_visits_with_true(),
            s.total_candidates(),
            s.total_bad(),
            s.mean_bad_per_visit(),
            s.bad_candidate_rate_pct(),
            s.coverage,
        );
    }
}

fn print_worst_trajectory_detail(s: &TrajStat) {
    println!(
        "\n### Trajectory {} — per-visit detail ({} visits, {} true obs added / {}) ###",
        s.traj_id,
        s.n_visits(),
        s.n_true_obs_added,
        s.n_obs_true
    );
    println!(
        "  {:>10}  {:>6}  {:>9}  {:>8}  {:>6}",
        "night", "visit", "true_obs?", "n_cand", "n_bad"
    );
    for v in &s.visit_stat {
        println!(
            "  {:>10}  {:>6}  {:>9}  {:>8}  {:>6}",
            v.night_id,
            v.visit_index,
            if v.true_in_visit { "YES" } else { "NO" },
            v.n_candidates,
            v.n_bad,
        );
    }
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("FINKFAT_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .without_time()
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    let obs_dataset = load_data(&cli.alerts, cli.override_obs_error_arcsec);

    let engine_config = EngineConfig::load_engine_config_validated(cli.config)?;
    let kalman_ctx = engine_config.build_context();

    println!(
        "Scanning the dataset and running the MOT candidate-contamination study on every trackable trajectory…"
    );

    let (stats, counters) = process_all_trajectories_mot(
        &obs_dataset,
        &engine_config,
        &kalman_ctx,
        &engine_config.kfbank_config,
    );

    print_mot_run_counters(&counters, stats.len());

    if stats.is_empty() {
        println!("\nNo trajectory produced a usable MOT result.");
        return Ok(());
    }

    print_mot_global_stats(&stats);
    print_miss_reason_histogram(&stats);
    print_truncation_reason_histogram(&stats);
    write_failure_dumps(&stats);
    print_miss_reason_by_step(&stats);
    print_geometry_diagnostics_by_reason(&stats);

    let traj_population = match &cli.ground_truth {
        Some(gt_path) => fink_fat_eval::population::load_traj_populations(gt_path)?,
        None => AHashMap::default(),
    };
    print_candidate_filter_diagnostics(&stats, &default_filter_suite(), &traj_population);
    print_along_track_bias_by_population(&stats, &traj_population);
    print_short_dt_not_matched_diagnostics(&stats, &traj_population);
    print_bootstrap_residual_diagnostics(&stats, &traj_population);
    write_short_dt_case_dumps(&stats);
    write_bootstrap_step_dumps(&stats);

    let worst = select_worst_by_bad_rate(&stats, N_WORST);
    print_worst_trajectories_table(&worst);
    for s in &worst {
        print_worst_trajectory_detail(s);
    }

    Ok(())
}

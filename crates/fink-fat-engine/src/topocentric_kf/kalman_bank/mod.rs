//! Multi-hypothesis Kalman filter bank for angles-only initial orbit tracking.
//!
//! A single observation pair (tracklet) only constrains 4 of the 6 state
//! degrees of freedom: the two sky angles and the two angular rates. The
//! topocentric range `ρ` and range-rate `ρ̇` are **unobservable** and must be
//! hypothesized. Rather than committing to one (poor) guess, the bank carries
//! a population of [`KFState`] hypotheses — one per `(ρ, ρ̇)` seed drawn from
//! the admissible region — each with a Bayesian weight.
//!
//! As new observations arrive, every hypothesis is propagated and updated;
//! its weight is multiplied by the predictive likelihood of the observation;
//! clearly-inconsistent hypotheses are gated out and the weakest are pruned.
//! Within a few nights the range ambiguity collapses to a single dominant mode.
//!
//! # Robustness improvements over the original design
//!
//! ## MAP-hypothesis gate protection
//!
//! The chi-square gate now **never discards the highest-weight (MAP) hypothesis**.
//! The gate's purpose is to cull clearly implausible range hypotheses from a large
//! population; applying it to the sole surviving best estimate is counter-productive:
//! it kills the filter right when it should adapt. Instead, a MAP hypothesis that
//! exceeds the gate is logged and passed through; filter inconsistency is tracked
//! via the NIS diagnostic.
//!
//! ## Scheduled hypothesis-count decay
//!
//! Early in the arc many hypotheses are needed to cover the `(ρ, ρ̇)` ambiguity.
//! As more observations constrain the orbit, the bank can be pruned more
//! aggressively. [`HypothesisCapSchedule`] makes this decay explicit and tunable
//! via linear, logarithmic, or exponential schedules, rather than relying on the
//! implicit weight-collapse dynamics alone.
//!
//! A `min_hypotheses` floor is always respected so the bank never collapses below
//! a user-defined minimum.
//!
//! ## Smoothed likelihood pruning
//!
//! The classical pruning criterion (weight floor on the cumulative Bayesian weight)
//! can kill a valid hypothesis after a single outlier observation. The new scheme
//! accumulates a **sliding window of per-step log-likelihoods** for each hypothesis
//! and prunes on the *mean* log-likelihood over that window. This averages out
//! single-epoch astrometric anomalies while still converging to the same result
//! once the window fills with consistent data.
//!
//! Setting `likelihood_window = 0` disables smoothing and falls back to the
//! original weight-floor approach.
//!
//! # Integration point
//!
//! [`KFBank::from_seeds`] takes pre-built `(KFState, weight)` pairs produced by
//! the admissible-region grid generator:
//!
//! ```ignore
//! let seeds = admissible_region_grid(obs_dataset, obs1, obs2, &ctx, &grid_cfg)?;
//! let mut bank = KFBank::from_seeds(seeds, KFBankConfig::default());
//! for obs in observations {
//!     let report = bank.step(obs_dataset, obs);
//!     if report.collapsed { break; }
//! }
//! let best_orbit = bank.best().unwrap().kf.to_orbit();
//! ```
//!
//! # Lifetime
//!
//! [`KFBank`] and [`Hypothesis`] share the `'state_lf` lifetime of [`KFState`]:
//! every hypothesis borrows the same ephemeris context that was used to build its
//! seed.

pub mod config;
pub mod ellipse_region_finder;
pub mod hypothesis;
pub mod hypothesis_cap;
pub mod seed_grid;

use std::collections::VecDeque;

use photom::observation_dataset::{ObsDataset, observation::Observation};
use tracing::{trace, trace_span};

use nalgebra::Vector6;

use crate::{
    error::EngineError,
    topocentric_kf::{
        KalmanContext,
        kalman_bank::{
            config::KFBankConfig,
            hypothesis::{Hypothesis, HypothesisStepResult},
            seed_grid::{GridConfig, admissible_region_grid},
        },
    },
};

// ── Per-step diagnostics ──────────────────────────────────────────────────────

/// Per-step diagnostics returned by [`KFBank::step`].
#[derive(Debug, Clone)]
pub struct BankStep {
    /// Observation epoch (MJD TT) processed in this step.
    pub epoch: f64,
    /// Number of live hypotheses entering the step.
    pub n_before: usize,
    /// Number of live hypotheses after gating / pruning / merging.
    pub n_after: usize,
    /// Hypotheses discarded by the chi-square gate.
    pub n_gated: usize,
    /// Hypotheses lost to a propagation / innovation / update failure.
    pub n_failed: usize,
    /// Effective sample size `1 / Σ wᵢ²` — low values mean one mode dominates.
    pub n_effective: f64,
    /// Weight of the current best (MAP) hypothesis.
    pub best_weight: f64,
    /// `true` if no hypothesis survived (filter lost).
    pub collapsed: bool,
    /// Effective hypothesis cap applied at this step (schedule + min floor).
    ///
    /// Useful for diagnosing whether the cap schedule is driving compression
    /// or whether weight dynamics are doing it faster.
    pub scheduled_cap: usize,
}

// ── Bank ──────────────────────────────────────────────────────────────────────

/// A bank of weighted [`KFState`] hypotheses tracking a single object under
/// range / range-rate ambiguity.
#[derive(Clone)]
pub struct KFBank<'state_lf> {
    pub(crate) hypotheses: Vec<Hypothesis<'state_lf>>,
    pub(crate) config: KFBankConfig,
    /// Number of `step()` calls completed so far.
    ///
    /// Drives the [`HypothesisCapSchedule`] decay: cap = schedule.cap(n_steps).
    n_steps: usize,
}

impl<'state_lf> KFBank<'state_lf> {
    // ── Construction ──────────────────────────────────────────────────────

    /// Build a bank from pre-constructed `(state, weight)` seeds.
    ///
    /// Weights need not be normalized; they are converted to normalized
    /// log-weights internally.  Non-positive weights are silently ignored.
    pub fn from_grid(
        obs_dataset: &ObsDataset,
        first_obs: &Observation,
        second_obs: &Observation,
        state: &'state_lf KalmanContext,
        grid_config: &GridConfig,
        bank_config: KFBankConfig,
    ) -> Result<Self, EngineError> {
        let seeds = admissible_region_grid(obs_dataset, first_obs, second_obs, state, grid_config)?;

        let hypotheses = seeds
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .enumerate()
            .map(|(id, (kf, w))| Hypothesis {
                kf,
                log_weight: w.ln(),
                id: id as u64,
                recent_log_liks: VecDeque::new(),
            })
            .collect();

        let mut bank = Self {
            hypotheses,
            config: bank_config,
            n_steps: 0,
        };
        bank.normalize_weights();
        Ok(bank)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Number of live hypotheses.
    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    /// `true` if at least one hypothesis is alive.
    pub fn is_alive(&self) -> bool {
        !self.hypotheses.is_empty()
    }

    /// Immutable view of the live hypotheses.
    pub fn hypotheses(&self) -> &[Hypothesis<'state_lf>] {
        &self.hypotheses
    }

    /// The maximum-a-posteriori (highest log-weight) hypothesis, if any.
    pub fn best(&self) -> Option<&Hypothesis<'state_lf>> {
        self.hypotheses
            .iter()
            .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
    }

    /// Effective sample size `1 / Σ wᵢ²` (weights assumed normalized).
    ///
    /// A value close to 1 means one hypothesis dominates; a value close to
    /// `len()` means the weights are approximately uniform.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.hypotheses.iter().map(|h| h.weight().powi(2)).sum();
        if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 }
    }

    /// Weighted mean of the 6-D attributable state across all hypotheses.
    ///
    /// **Note:** only meaningful once the mixture is roughly unimodal.  While
    /// several range modes coexist the MAP hypothesis ([`KFBank::best`]) is a
    /// better point estimate.  Averaging angular components $(\alpha, \delta)$
    /// linearly does not account for the $0/2\pi$ wraparound.
    pub fn weighted_mean_state(&self) -> Option<Vector6<f64>> {
        if self.hypotheses.is_empty() {
            return None;
        }
        let mean = self
            .hypotheses
            .iter()
            .fold(Vector6::zeros(), |acc, h| acc + h.kf.state * h.weight());
        Some(mean)
    }

    // ── Main step ─────────────────────────────────────────────────────────

    /// Process one new observation: propagate → score → update → prune → merge.
    ///
    /// The pipeline for each hypothesis is:
    /// 1. **Propagate** to the observation epoch (Kepler + covariance transport).
    /// 2. **Gate** on the Mahalanobis² distance (the MAP hypothesis is exempt).
    /// 3. **Score** with the predictive log-likelihood.
    /// 4. **Update** the Kalman state.
    /// 5. **Push** the log-likelihood into the sliding window.
    ///
    /// After all hypotheses are processed, the bank runs:
    /// - Normalize weights.
    /// - Prune: smoothed-score floor (or classic weight floor if window = 0).
    /// - Apply scheduled cap (decay curve).
    /// - Merge spatially coincident modes.
    /// - Re-normalize.
    pub fn step(&mut self, obs_dataset: &ObsDataset, obs: &Observation) -> BankStep {
        let epoch = obs.mjd_tt();
        let n_before = self.hypotheses.len();

        let span = trace_span!("kf_bank_step", epoch, n_hypotheses = n_before);
        let _enter = span.enter();

        trace!(
            n_hypotheses = n_before,
            "Starting predict/update/score cycle"
        );

        let (survivors, n_gated, n_failed) = self.process_hypotheses(obs_dataset, obs);

        trace!(
            n_survivors = survivors.len(),
            n_gated, n_failed, "Predict/score cycle complete — starting cleanup"
        );

        self.hypotheses = survivors;

        // Compute the scheduled cap *before* cleanup so we can report it.
        let scheduled_cap = self
            .config
            .cap_schedule
            .cap(self.n_steps)
            .max(self.config.min_hypotheses.max(1));

        self.post_step_cleanup();

        // Increment the observation counter *after* the step so that cap(0)
        // applies to the very first observation (full initial population).
        self.n_steps += 1;

        BankStep {
            epoch,
            n_before,
            n_after: self.hypotheses.len(),
            n_gated,
            n_failed,
            n_effective: self.effective_sample_size(),
            best_weight: self.best().map(|h| h.weight()).unwrap_or(0.0),
            collapsed: self.hypotheses.is_empty(),
            scheduled_cap,
        }
    }

    // ── Internal predict/update pipeline ─────────────────────────────────

    /// Run the predict/score/update cycle over all current hypotheses.
    ///
    /// Identifies the top `min_hypotheses` hypotheses **before** the step so
    /// they are marked as exempt from the chi-square gate.
    ///
    /// Protecting only the MAP (single best) was insufficient: if the MAP
    /// survived but the next 4 best hypotheses were gated out on the same
    /// step, the bank collapsed to 1 even though `min_hypotheses = 5`.  The
    /// weight-floor pruning already respects `min_hypotheses`; the gate must
    /// honour the same floor.
    ///
    /// Returns `(survivors, n_gated, n_failed)`.
    fn process_hypotheses(
        &mut self,
        obs_dataset: &ObsDataset,
        obs: &Observation,
    ) -> (Vec<Hypothesis<'state_lf>>, usize, usize) {
        // Compute the protected set *before* weights change.
        // Every hypothesis in the top `min_hypotheses` by current log-weight
        // is immune to the chi-square gate for this step.
        let protected_ids = top_k_ids(&self.hypotheses, self.config.min_hypotheses.max(1));

        let results: Vec<HypothesisStepResult<'state_lf>> = std::mem::take(&mut self.hypotheses)
            .into_iter()
            .map(|hyp| {
                let is_protected = protected_ids.contains(&hyp.id);
                hyp.process(&self.config, obs_dataset, obs, is_protected)
            })
            .collect();

        let n_gated = results
            .iter()
            .filter(|r| matches!(r, HypothesisStepResult::Gated))
            .count();
        let n_failed = results
            .iter()
            .filter(|r| matches!(r, HypothesisStepResult::Failed))
            .count();
        let survivors = results
            .into_iter()
            .filter_map(|r| match r {
                HypothesisStepResult::Survived(h) => Some(h),
                _ => None,
            })
            .collect();

        (survivors, n_gated, n_failed)
    }

    // ── Post-step cleanup pipeline ────────────────────────────────────────

    /// Full post-step cleanup: normalize → prune → cap → merge → normalize.
    fn post_step_cleanup(&mut self) {
        self.normalize_weights();

        let n0 = self.hypotheses.len();
        self.prune_by_smoothed_score();
        let n1 = self.hypotheses.len();
        if n0 != n1 {
            trace!(n_before = n0, n_after = n1, "Pruning phase");
        }

        let n2 = n1;
        self.cap_to_scheduled_max();
        let n3 = self.hypotheses.len();
        if n2 != n3 {
            trace!(n_before = n2, n_after = n3, "Scheduled-cap phase");
        }

        let n4 = n3;
        self.merge_coincident_modes();
        let n5 = self.hypotheses.len();
        if n4 != n5 {
            trace!(
                n_before = n4,
                n_after = n5,
                n_merged = n4 - n5,
                "Merging phase"
            );
        }

        self.normalize_weights();
    }

    /// Prune low-quality hypotheses using a sliding-window likelihood score.
    ///
    /// # Strategy
    ///
    /// When `likelihood_window > 0` (smoothed mode):
    ///
    /// Each hypothesis carries a bounded deque of its last `W` per-step
    /// log-likelihoods.  The **smoothed score** is the mean of this deque.
    /// A hypothesis is pruned if its smoothed score falls more than
    /// `−ln(weight_floor)` below the best smoothed score:
    ///
    /// ```text
    /// prune if: smoothed(h) < smoothed(best) + ln(weight_floor)
    /// ```
    ///
    /// The top `min_hypotheses` by smoothed score are **always protected**,
    /// regardless of the threshold, to prevent the bank from collapsing to
    /// fewer modes than desired.
    ///
    /// When `likelihood_window == 0` (classic mode), falls back to the
    /// original raw weight-floor approach on the cumulative `log_weight`.
    fn prune_by_smoothed_score(&mut self) {
        let window = self.config.likelihood_window;

        if window == 0 {
            // Classic path: weight floor on the cumulative Bayesian weight.
            self.prune_by_weight_floor();
            return;
        }

        let min_keep = self.config.min_hypotheses.max(1);
        if self.hypotheses.len() <= min_keep {
            // Nothing left to prune; protecting min_hypotheses already.
            return;
        }

        // Best smoothed score across all live hypotheses.
        let best_smoothed = self
            .hypotheses
            .iter()
            .map(|h| h.smoothed_log_lik())
            .fold(f64::NEG_INFINITY, f64::max);

        if !best_smoothed.is_finite() {
            // No hypothesis has accumulated any window data yet — skip.
            trace!("Smoothed pruning skipped: window not yet populated");
            return;
        }

        // The score that the min_keep-th best hypothesis achieves.
        // Any hypothesis at or above this floor is in the "protected" group.
        let protection_floor = nth_largest_smoothed_score(&self.hypotheses, min_keep);

        // Relative threshold: a hypothesis must be within weight_floor of the best.
        let log_threshold = best_smoothed + self.config.weight_floor.ln();

        let n_before = self.hypotheses.len();

        // Retain a hypothesis if EITHER:
        //   (a) its smoothed score is above the weight-floor threshold, OR
        //   (b) it is among the top `min_keep` (score >= protection_floor).
        self.hypotheses.retain(|h| {
            let s = h.smoothed_log_lik();
            s >= log_threshold || s >= protection_floor
        });

        let n_removed = n_before - self.hypotheses.len();
        if n_removed > 0 {
            trace!(
                n_removed,
                log_threshold,
                best_smoothed,
                window,
                min_kept = min_keep,
                "Smoothed-score pruning"
            );
        }
    }

    /// Classic weight-floor pruning on the cumulative `log_weight`.
    ///
    /// Used as the fallback when `likelihood_window == 0`, and also respects
    /// the `min_hypotheses` floor.
    fn prune_by_weight_floor(&mut self) {
        let min_keep = self.config.min_hypotheses.max(1);
        if self.hypotheses.len() <= min_keep {
            return;
        }

        // Score of the min_keep-th best hypothesis by cumulative weight.
        let protection_floor = nth_largest_log_weight(&self.hypotheses, min_keep);

        let floor_log = self.config.weight_floor.ln(); // e.g. ln(1e-4) ≈ −9.2
        // After normalization, the best hypothesis has log_weight ≈ 0, so the
        // threshold is just floor_log.
        let n_before = self.hypotheses.len();

        self.hypotheses
            .retain(|h| h.log_weight >= floor_log || h.log_weight >= protection_floor);

        let n_removed = n_before - self.hypotheses.len();
        if n_removed > 0 {
            trace!(
                n_removed,
                weight_floor = self.config.weight_floor,
                min_kept = min_keep,
                "Weight-floor pruning"
            );
        }
    }

    /// Apply the scheduled hypothesis cap, respecting the `min_hypotheses` floor.
    ///
    /// Keeps the `effective_cap` highest-weighted hypotheses (by `log_weight`),
    /// where `effective_cap = max(schedule.cap(n_steps), min_hypotheses)`.
    fn cap_to_scheduled_max(&mut self) {
        let raw_cap = self.config.cap_schedule.cap(self.n_steps);
        let effective_cap = raw_cap.max(self.config.min_hypotheses.max(1));

        if self.hypotheses.len() <= effective_cap {
            return;
        }

        // Sort descending by cumulative weight to keep the most plausible ones.
        self.hypotheses
            .sort_by(|a, b| b.log_weight.partial_cmp(&a.log_weight).unwrap());

        let n_truncated = self.hypotheses.len() - effective_cap;
        self.hypotheses.truncate(effective_cap);

        trace!(
            n_truncated,
            effective_cap,
            raw_cap,
            n_steps = self.n_steps,
            "Scheduled cap truncation"
        );
    }

    /// Greedily merge modes whose mean heliocentric positions are within
    /// `merge_position_au`.
    fn merge_coincident_modes(&mut self) {
        let thresh = self.config.merge_position_au;
        let mut merged: Vec<Hypothesis<'state_lf>> = Vec::with_capacity(self.hypotheses.len());

        for hyp in std::mem::take(&mut self.hypotheses) {
            match merged
                .iter_mut()
                .find(|m| m.kf.position_distance_au(&hyp.kf) < thresh)
            {
                Some(existing) => {
                    trace!(
                        hyp_id_kept = existing.id,
                        hyp_id_merged = hyp.id,
                        threshold_au = thresh,
                        "Merging spatially coincident modes"
                    );
                    *existing = existing.moment_match_merge(&hyp);
                }
                None => merged.push(hyp),
            }
        }

        self.hypotheses = merged;
    }

    /// Normalize log-weights so that `Σ exp(log_weight) = 1`, using the
    /// log-sum-exp trick for numerical stability.
    fn normalize_weights(&mut self) {
        if self.hypotheses.is_empty() {
            return;
        }

        let max = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        if !max.is_finite() {
            // All weights underflowed — fall back to uniform to avoid losing
            // the bank entirely rather than declaring the filter lost.
            let uniform = -(self.hypotheses.len() as f64).ln();
            self.hypotheses
                .iter_mut()
                .for_each(|h| h.log_weight = uniform);
            return;
        }

        let log_norm = max
            + self
                .hypotheses
                .iter()
                .map(|h| (h.log_weight - max).exp())
                .sum::<f64>()
                .ln();

        self.hypotheses
            .iter_mut()
            .for_each(|h| h.log_weight -= log_norm);
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Collect the ids of the `k` highest-weight hypotheses (by `log_weight`).
///
/// Used by [`KFBank::process_hypotheses`] to build the protected set before
/// the predict/gate/update cycle: any hypothesis whose id is in this set is
/// exempt from the chi-square gate for the current step.
///
/// Returns a `Vec` rather than a `HashSet` because `k` (= `min_hypotheses`)
/// is typically small (≤ 10), making a linear `.contains()` search cheaper
/// than the overhead of hashing.
fn top_k_ids(hypotheses: &[Hypothesis<'_>], k: usize) -> Vec<u64> {
    if k == 0 || hypotheses.is_empty() {
        return Vec::new();
    }
    // Partial sort descending by log_weight; no full sort needed.
    let actual_k = k.min(hypotheses.len());
    let mut indexed: Vec<(f64, u64)> = hypotheses.iter().map(|h| (h.log_weight, h.id)).collect();
    // select_nth_unstable_by places the top `actual_k` elements in the first
    // `actual_k` slots (unordered) — O(n) average.
    indexed.select_nth_unstable_by(actual_k - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed[..actual_k].iter().map(|(_, id)| *id).collect()
}

/// Smoothed score of the `k`-th best hypothesis (1-indexed) by smoothed log-likelihood.
///
/// Used to compute the protection floor for `prune_by_smoothed_score`:
/// any hypothesis with a score ≥ this value is in the protected top-k group.
///
/// Returns `f64::NEG_INFINITY` if `k` exceeds the number of hypotheses (all
/// are protected).
fn nth_largest_smoothed_score(hypotheses: &[Hypothesis<'_>], k: usize) -> f64 {
    if k >= hypotheses.len() {
        return f64::NEG_INFINITY;
    }
    let mut scores: Vec<f64> = hypotheses.iter().map(|h| h.smoothed_log_lik()).collect();
    // Partial sort: we only need the k-th element (0-indexed: k-1).
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scores[k - 1]
}

/// Cumulative log-weight of the `k`-th best hypothesis (1-indexed).
///
/// Used by `prune_by_weight_floor` to compute the protection floor.
/// Returns `f64::NEG_INFINITY` if `k` exceeds the number of hypotheses.
fn nth_largest_log_weight(hypotheses: &[Hypothesis<'_>], k: usize) -> f64 {
    if k >= hypotheses.len() {
        return f64::NEG_INFINITY;
    }
    let mut weights: Vec<f64> = hypotheses.iter().map(|h| h.log_weight).collect();
    weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    weights[k - 1]
}

// ── Configuration ─────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

use crate::engine_config::hypothesis_cap::HypothesisCapSchedule;

/// Tuning parameters for the hypothesis bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KFBankConfig {
    /// Chi-square gate on the predictive innovation (2 d.o.f.).
    ///
    /// A hypothesis whose Mahalanobis distance² exceeds this threshold is
    /// discarded.  Typical values: `13.8 ≈ 99.9 %`, `23.0 ≈ 99.999 %`.
    ///
    /// **The MAP (highest-weight) hypothesis is always exempt from this gate.**
    /// When only one hypothesis remains, the gate has no role to play and
    /// gating out the last survivor would destroy the filter.  Inconsistency
    /// should instead be addressed via covariance inflation (see process-noise
    /// tuning) and is tracked by the NIS diagnostic.
    pub gate_chi2: f64,

    /// Relative weight floor for the **smoothed-score pruning** strategy.
    ///
    /// When `likelihood_window > 0` (smoothed mode), a hypothesis is pruned
    /// if its mean per-step log-likelihood satisfies
    ///
    /// ```text
    /// exp(mean_log_lik − best_mean_log_lik) < weight_floor
    /// ```
    ///
    /// When `likelihood_window = 0` (classic mode), falls back to the raw
    /// cumulative-weight floor: `exp(log_weight) < weight_floor`.
    ///
    /// Typical values: `1e-4` (aggressive) to `1e-6` (conservative).
    pub weight_floor: f64,

    /// Minimum number of hypotheses to always keep alive, regardless of
    /// weight or gate outcome.
    ///
    /// Combined with the MAP protection, this prevents premature collapse
    /// before the `(ρ, ρ̇)` ambiguity is truly resolved.  A value of 5–10
    /// is recommended for arcs shorter than ≈15 nights.
    pub min_hypotheses: usize,

    /// Decay schedule for the maximum number of live hypotheses as a function
    /// of the number of observations processed.
    ///
    /// Use [`HypothesisCapSchedule::Fixed`] to reproduce the legacy behaviour.
    /// Use [`HypothesisCapSchedule::Logarithmic`] for fast initial collapse
    /// followed by a stable core.
    pub cap_schedule: HypothesisCapSchedule,

    /// Sliding-window size for the smoothed pruning strategy.
    ///
    /// Each hypothesis accumulates its last `likelihood_window` per-step
    /// log-likelihoods.  Pruning decisions are based on the **mean** of this
    /// window rather than the raw cumulative weight, making the bank robust
    /// to single astrometric outliers.
    ///
    /// Set to `0` to disable smoothing and use the original weight-floor
    /// approach.
    pub likelihood_window: usize,

    /// Chi-square factor used **only** for the search-region radius, decoupled
    /// from the update gate.
    ///
    /// The bounding radius of the predicted [`SearchRegion`] is
    ///
    /// ```text
    /// r = sqrt(search_region_chi2) × sqrt(λ_max(S_mix))
    /// ```
    ///
    /// Setting this independently of `gate_chi2` lets the gate stay tight
    /// (e.g. `gate_chi2 = 23` ≈ 99.999 %) while the search region remains
    /// generous enough to reliably contain the next observation even when the
    /// filter is slightly overconfident.
    ///
    /// Practical guidance
    /// ------------------
    /// - `400.0` (20σ) is a reasonable default: large enough to tolerate mild
    ///   filter inconsistency without saturating the `max_arcsec` clamp for a
    ///   well-converged filter.
    /// - If `in_r` coverage drops, increase this value.  A value large enough
    ///   to always saturate the clamp turns `in_r` into a pure "within 30'"
    ///   indicator, which was the pre-refactor behaviour.
    pub search_region_chi2: f64,

    /// Merge two modes whose mean heliocentric positions are within this
    /// distance (AU).
    ///
    /// Deliberately an *absolute* threshold, not a Mahalanobis one: using
    /// covariance would collapse adjacent range nodes prematurely (their σ is
    /// large at initialisation).  Merging only deduplicates modes that have
    /// physically converged onto the same heliocentric point.
    pub merge_position_au: f64,
}

impl Default for KFBankConfig {
    fn default() -> Self {
        Self {
            gate_chi2: 23.0,
            weight_floor: 1e-4,
            min_hypotheses: 5,
            cap_schedule: HypothesisCapSchedule::default(),
            likelihood_window: 3,
            search_region_chi2: 400.0,
            merge_position_au: 0.02,
        }
    }
}

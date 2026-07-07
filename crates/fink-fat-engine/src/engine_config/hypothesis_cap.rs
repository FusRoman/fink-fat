use serde::{Deserialize, Serialize};

/// Decay schedule controlling how many live hypotheses the bank may hold as a
/// function of the number of observations processed.
///
/// Early in the arc, a large population covers the `(ρ, ρ̇)` ambiguity.  As
/// additional observations constrain the orbit, the true mode emerges and the
/// bank can afford to shed low-weight hypotheses more aggressively.  This enum
/// encodes that intent as a first-class tunable rather than an implicit side
/// effect of the weight dynamics.
///
/// Every variant has a `start` (cap at observation 0) and an `end` (asymptotic
/// minimum cap).  The effective cap is always additionally lower-bounded by
/// [`KFBankConfig::min_hypotheses`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HypothesisCapSchedule {
    /// Fixed cap: the maximum number of live hypotheses never changes.
    ///
    /// Equivalent to the original `max_hypotheses` field.
    Fixed(usize),

    /// Linear decay from `start` down to `end` over `n_obs_full` observations.
    ///
    /// ```text
    /// cap(n) = end + (start − end) × max(0, 1 − n / n_obs_full)
    /// ```
    ///
    /// Reaches `end` at observation `n_obs_full` and stays there.
    /// Predictable and easy to reason about.
    Linear {
        start: usize,
        end: usize,
        /// Number of observations after which the cap saturates at `end`.
        n_obs_full: usize,
    },

    /// Logarithmic decay: fast initial drop that plateaus early.
    ///
    /// ```text
    /// cap(n) = end + (start − end) × (1 − ln(1+n) / ln(1+n_obs_full))
    /// ```
    ///
    /// Useful when most discriminating information arrives in the first few
    /// nights: the bank contracts quickly to a compact core, then stabilises.
    Logarithmic {
        start: usize,
        end: usize,
        /// Number of observations defining the plateau.
        n_obs_full: usize,
    },

    /// Exponential decay with time constant `tau` (in observations).
    ///
    /// ```text
    /// cap(n) = end + (start − end) × exp(−n / tau)
    /// ```
    ///
    /// Never strictly reaches `end` but approaches it asymptotically.
    /// `tau ≈ n_obs_full / 3` gives a ~95 % decay over `n_obs_full` steps.
    Exponential {
        start: usize,
        end: usize,
        /// Decay time constant in units of observations.
        tau: f64,
    },
}

impl HypothesisCapSchedule {
    /// Effective cap after `n_obs` observations have been processed.
    ///
    /// Always returns at least 1.
    pub fn cap(&self, n_obs: usize) -> usize {
        let raw = match self {
            Self::Fixed(cap) => *cap,

            Self::Linear {
                start,
                end,
                n_obs_full,
            } => {
                if n_obs >= *n_obs_full {
                    return (*end).max(1);
                }
                let frac = n_obs as f64 / *n_obs_full as f64;
                (*end as f64 + (*start as f64 - *end as f64) * (1.0 - frac)).round() as usize
            }

            Self::Logarithmic {
                start,
                end,
                n_obs_full,
            } => {
                // ln(1+n) grows from 0 to ln(1+n_obs_full); normalize to [0,1].
                let log_max = (1.0 + *n_obs_full as f64).ln();
                let frac = (1.0 + n_obs as f64).ln() / log_max;
                (*end as f64 + (*start as f64 - *end as f64) * (1.0 - frac)).round() as usize
            }

            Self::Exponential { start, end, tau } => (*end as f64
                + (*start as f64 - *end as f64) * (-(n_obs as f64) / tau).exp())
            .round() as usize,
        };
        raw.max(1)
    }

    /// The initial (maximum) cap, used to pre-size containers.
    pub fn max_cap(&self) -> usize {
        match self {
            Self::Fixed(cap) => *cap,
            Self::Linear { start, .. } => *start,
            Self::Logarithmic { start, .. } => *start,
            Self::Exponential { start, .. } => *start,
        }
    }
}

impl Default for HypothesisCapSchedule {
    /// Defaults to a fixed cap of 1 000, identical to the legacy `max_hypotheses`.
    fn default() -> Self {
        Self::Fixed(1000)
    }
}

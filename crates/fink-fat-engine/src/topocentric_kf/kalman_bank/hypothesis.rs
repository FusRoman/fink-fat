// ── Hypothesis ────────────────────────────────────────────────────────────────

use std::collections::VecDeque;

use nalgebra::{Matrix2, Vector2, Vector6};
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, observation::Observation},
};
use tracing::trace;

use crate::{
    engine_config::{kalman_context::KalmanContext, kf_bank_config::KFBankConfig},
    error::ObservationJacobianError,
    topocentric_kf::{
        constants::MAX_INNOVATION_DET,
        single_kalman::{KFState, KFStateSnapshot, update::wrap_angle},
    },
};

/// A single weighted hypothesis in the bank.
///
/// # Association history lives on the bank, not here
///
/// `Hypothesis` deliberately carries no `track_ids` field. Every hypothesis
/// in a [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank) is propagated,
/// gated and updated against the *same* observation on every step — either
/// it survives and shares the bank's new association id, or it is gated out
/// entirely. That invariant makes a per-hypothesis history redundant: it is
/// tracked once, on the bank, in `KFBank::track_ids`. Consequently
/// [`Self::moment_match_merge`] needs no `track_ids` equality guard between
/// the merged hypotheses — both already belong to the same bank and share
/// the same history by construction.
#[derive(Clone)]
pub struct Hypothesis<'state_lf> {
    /// Filter state for this hypothesis, in attributable coordinates.
    pub kf: KFState<'state_lf>,

    /// Natural-log of the (un-normalized within a step, normalized after) weight.
    ///
    /// Stored in log-space to avoid underflow over many observations.
    /// This is the full cumulative Bayesian posterior: it is used for `best()`
    /// and for weight-based computations, but **not** for the smoothed pruning
    /// decision (see [`Hypothesis::smoothed_log_lik`]).
    pub log_weight: f64,
    /// Stable identifier, useful for tracking a mode across steps.
    pub id: u64,

    // ── Smoothed likelihood window ────────────────────────────────────────
    /// Circular buffer of the last `likelihood_window` per-step log-likelihoods.
    ///
    /// Populated during [`KFBank::step`]; empty until the first observation is
    /// processed.  Used exclusively by the smoothed pruning strategy; has no
    /// effect on `log_weight` or `weight()`.
    pub(crate) recent_log_liks: VecDeque<f64>,
}

/// Owned, borrow-free snapshot of a [`Hypothesis`], for persisting a
/// [`BranchCollection`](crate::topocentric_kf::branching::BranchCollection)
/// to disk across nights (see [`Hypothesis::to_snapshot`]).
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct HypothesisSnapshot {
    pub kf: KFStateSnapshot,
    pub log_weight: f64,
    pub id: u64,
    pub recent_log_liks: Vec<f64>,
}

impl<'state_lf> Hypothesis<'state_lf> {
    /// Convert to an owned, borrow-free snapshot suitable for on-disk
    /// persistence (see [`HypothesisSnapshot`]).
    pub fn to_snapshot(&self) -> HypothesisSnapshot {
        HypothesisSnapshot {
            kf: self.kf.to_snapshot(),
            log_weight: self.log_weight,
            id: self.id,
            recent_log_liks: self.recent_log_liks.iter().copied().collect(),
        }
    }
}

impl HypothesisSnapshot {
    /// Reattach `shared_ctx` (supplied by the caller) to rebuild a full
    /// [`Hypothesis`].
    pub fn into_hypothesis(self, shared_ctx: &KalmanContext) -> Hypothesis<'_> {
        Hypothesis {
            kf: self.kf.into_kf_state(shared_ctx),
            log_weight: self.log_weight,
            id: self.id,
            recent_log_liks: self.recent_log_liks.into_iter().collect(),
        }
    }
}

impl<'state_lf> Hypothesis<'state_lf> {
    /// Linear weight `w = exp(log_weight)`.
    #[inline]
    pub fn weight(&self) -> f64 {
        self.log_weight.exp()
    }

    /// Mean log-likelihood over the recent sliding window.
    ///
    /// Returns `f64::NEG_INFINITY` when the window is empty (e.g. right after
    /// initialisation, before any observation has been processed).
    ///
    /// This is the quantity used by the smoothed pruning strategy.  A higher
    /// value indicates that this hypothesis has been consistently likely under
    /// recent observations.
    pub(crate) fn smoothed_log_lik(&self) -> f64 {
        if self.recent_log_liks.is_empty() {
            return f64::NEG_INFINITY;
        }
        self.recent_log_liks.iter().sum::<f64>() / self.recent_log_liks.len() as f64
    }

    /// Propagate a hypothesis to the observation epoch.
    ///
    /// `pub(crate)` so [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank)
    /// can drive propagation and scoring as two separate phases — needed by
    /// the branch primitives (`predict_to`/`branch_with`), which propagate a
    /// bank once and then apply several candidate updates to the same
    /// propagated state without repeating the (expensive) two-body step.
    pub(crate) fn propagate(
        &self,
        obs_dataset: &ObsDataset,
        obs: &Observation,
    ) -> Result<Self, ()> {
        match self.kf.propagate(obs_dataset, obs) {
            Ok(kf) => {
                trace!(hyp_id = self.id, "Propagation OK");
                Ok(Hypothesis { kf, ..self.clone() })
            }
            Err(e) => {
                trace!(hyp_id = self.id, error = ?e, "Propagation FAILED");
                Err(())
            }
        }
    }

    /// Check whether the Mahalanobis² distance passes the chi-square gate.
    ///
    /// **Protection rule**: when `is_protected` is `true` (hypothesis is among
    /// the pre-step top `min_hypotheses` by weight), it is *never* discarded
    /// even if `d² > gate_chi2`.  The exceedance is logged at trace level.
    ///
    /// Rationale: the gate culls implausible range modes from a large population.
    /// The top `min_hypotheses` must survive so the bank retains the minimum
    /// diversity promised by the config, regardless of a single bad observation.
    fn apply_gate(
        &self,
        config: &KFBankConfig,
        d2: f64,
        is_protected: bool,
    ) -> Result<(), HypothesisStepResult<'state_lf>> {
        if !d2.is_finite() {
            trace!(self.id, "Gate REJECTED: non-finite Mahalanobis²");
            return Err(HypothesisStepResult::Gated);
        }

        if d2 > config.gate_chi2 {
            if is_protected {
                // Protected hypothesis — log the inconsistency and continue.
                trace!(
                    self.id,
                    d2,
                    gate_chi2 = config.gate_chi2,
                    "Protected hypothesis EXEMPT from chi² gate — \
                     high Mahalanobis² recorded but hypothesis preserved"
                );
                return Ok(());
            }

            trace!(
                self.id,
                d2,
                gate_chi2 = config.gate_chi2,
                "Gate REJECTED: Mahalanobis² exceeds threshold"
            );
            return Err(HypothesisStepResult::Gated);
        }

        trace!(self.id, d2, "Gate OK");
        Ok(())
    }

    /// Compute the innovation, apply the gate, score the log-likelihood, and
    /// apply the Kalman measurement update.
    ///
    /// Returns `Ok((log_likelihood, updated_kf))` on success, or a terminal
    /// [`HypothesisStepResult`] variant on failure or gate rejection.
    ///
    /// `is_protected` — if `true`, the gate is applied in advisory mode only:
    /// the hypothesis is never discarded regardless of d².
    pub(crate) fn score_and_update(
        self,
        config: &KFBankConfig,
        obs: &Observation,
        is_protected: bool,
    ) -> Result<(f64, KFState<'state_lf>), HypothesisStepResult<'state_lf>> {
        let (nu, s) = measurement_innovation(&self.kf, obs).map_err(|e| {
            trace!(self.id, error = ?e, "Innovation FAILED");
            HypothesisStepResult::Failed
        })?;

        let s_inv = s.try_inverse().ok_or_else(|| {
            trace!(self.id, "Innovation covariance not invertible");
            HypothesisStepResult::Failed
        })?;

        let d2 = (nu.transpose() * s_inv * nu)[(0, 0)];
        self.apply_gate(config, d2, is_protected)?;

        let log_lik = compute_log_likelihood(&s, d2).ok_or_else(|| {
            trace!(self.id, "Innovation covariance singular (det ≤ 0)");
            HypothesisStepResult::Failed
        })?;

        trace!(self.id, log_lik, "Likelihood scored");

        let updated_kf = self.kf.update(obs).map_err(|e| {
            trace!(self.id, error = ?e, "Measurement update FAILED");
            HypothesisStepResult::Failed
        })?;

        trace!(self.id, "Measurement update OK");
        Ok((log_lik, updated_kf))
    }

    /// Run one full predict/score/update cycle for a single hypothesis.
    ///
    /// `is_protected` — whether this hypothesis belongs to the pre-step top
    /// `min_hypotheses` set.  Protected hypotheses are exempt from the gate.
    pub fn process(
        self,
        config: &KFBankConfig,
        obs_dataset: &ObsDataset,
        obs: &Observation,
        is_protected: bool,
    ) -> HypothesisStepResult<'state_lf> {
        match self.propagate(obs_dataset, obs) {
            Ok(predicted) => predicted.finalize_score_and_update(config, obs, is_protected),
            Err(_) => HypothesisStepResult::Failed,
        }
    }

    /// Gate, score and update an **already-propagated** hypothesis.
    ///
    /// This is the second half of [`Self::process`], extracted so
    /// [`KFBank::branch_with`](crate::topocentric_kf::kalman_bank::KFBank::branch_with)
    /// can reuse it directly on hypotheses that were propagated once (via
    /// [`KFBank::predict_to`](crate::topocentric_kf::kalman_bank::KFBank::predict_to))
    /// and then scored against several candidate observations, without
    /// repeating the two-body propagation for each candidate.
    ///
    /// `is_protected` — if `true`, the gate is applied in advisory mode only:
    /// the hypothesis is never discarded regardless of d².
    pub(crate) fn finalize_score_and_update(
        self,
        config: &KFBankConfig,
        obs: &Observation,
        is_protected: bool,
    ) -> HypothesisStepResult<'state_lf> {
        let id = self.id;
        let prior_log_weight = self.log_weight;

        // Clone the window from the propagated state; we push the new
        // log-likelihood after scoring so it reflects the current observation.
        let mut recent_log_liks = self.recent_log_liks.clone();

        let (log_lik, updated_kf) = match self.score_and_update(config, obs, is_protected) {
            Ok(result) => result,
            Err(outcome) => return outcome,
        };

        // Append the new log-likelihood into the bounded sliding window.
        push_log_lik_to_window(&mut recent_log_liks, log_lik, config.likelihood_window);

        HypothesisStepResult::Survived(Hypothesis {
            log_weight: prior_log_weight + log_lik,
            kf: updated_kf,
            id,
            recent_log_liks,
        })
    }

    /// Merge two weighted Gaussians by moment matching.
    ///
    /// $$w = w_a + w_b, \quad
    ///   \mu = \frac{w_a \mu_a + w_b \mu_b}{w}, \quad
    ///   P = \frac{1}{w} \sum_i w_i \bigl( P_i + (\mu_i - \mu)(\mu_i - \mu)^\top \bigr)$$
    ///
    /// The observer geometry and ephemeris reference are taken from the
    /// higher-weighted hypothesis (both share the same epoch, so their observer
    /// states are identical).  The `recent_log_liks` window is taken from the
    /// dominant hypothesis so the smoothed score reflects its history rather than
    /// a meaningless blend.
    pub fn moment_match_merge(&self, other: &Hypothesis<'state_lf>) -> Self {
        let (wa, wb) = (self.weight(), other.weight());
        let w_total = wa + wb;

        let merged_state = (self.kf.state * wa + other.kf.state * wb) / w_total;
        let merged_covariance = self.moment_match_covariance(other, &merged_state, wa, wb, w_total);

        let dominant = if wa >= wb { self } else { other };
        let mut merged_kf = dominant.kf.clone();
        merged_kf.state = merged_state;
        merged_kf.covariance = merged_covariance;
        merged_kf.kalman_gain = None;

        Hypothesis {
            kf: merged_kf,
            log_weight: w_total.ln(),
            id: dominant.id,
            // Inherit the dominant hypothesis's likelihood window: it carries the
            // most reliable history since it has the higher posterior weight.
            recent_log_liks: dominant.recent_log_liks.clone(),
        }
    }

    /// Compute the moment-matched covariance from two weighted Gaussian components.
    ///
    /// Each component contributes its own covariance plus the outer product of
    /// its mean offset from the merged mean:
    ///
    /// $$P = \frac{1}{w} \sum_i w_i \bigl( P_i + (\mu_i - \mu)(\mu_i - \mu)^\top \bigr)$$
    fn moment_match_covariance(
        &self,
        other: &Hypothesis<'state_lf>,
        merged_mean: &Vector6<f64>,
        wa: f64,
        wb: f64,
        w_total: f64,
    ) -> nalgebra::Matrix6<f64> {
        let spread = |hyp: &Hypothesis, w: f64| {
            let delta = hyp.kf.state - merged_mean;
            (hyp.kf.covariance + delta * delta.transpose()) * w
        };
        (spread(self, wa) + spread(other, wb)) / w_total
    }
}

// ── Step outcome ──────────────────────────────────────────────────────────────

/// Outcome of processing a single hypothesis through one predict/update cycle.
pub enum HypothesisStepResult<'state_lf> {
    /// The hypothesis survived the full cycle with an updated state and new
    /// log-weight.
    Survived(Hypothesis<'state_lf>),
    /// The hypothesis was rejected by the Mahalanobis² gate.
    Gated,
    /// A numerical failure occurred (propagation, innovation, covariance
    /// singularity, or measurement update).
    Failed,
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Predictive innovation $\nu = z - h(x^-)$ and innovation covariance
/// $S = H P^- H^\top + R$ for a single hypothesis at its current epoch.
///
/// In attributable coordinates the predicted sky position is simply the first
/// two state components — no projection through a heliocentric Cartesian
/// position is needed.
fn measurement_innovation(
    kf: &KFState,
    obs: &Observation,
) -> Result<(Vector2<f64>, Matrix2<f64>), ObservationJacobianError> {
    let coord = obs.equ_coord();
    let z = Vector2::new(coord.ra, coord.dec);
    let predicted = Vector2::new(kf.state[0], kf.state[1]);

    let mut nu = z - predicted;
    nu[0] = wrap_angle(nu[0]); // Wrap RA difference into (−π, π].

    let s = kf.sky_covariance()? + observation_noise_matrix(&coord);

    Ok((nu, s))
}

/// Build the diagonal $2\times2$ observation noise matrix $R$ from the
/// per-axis astrometric standard deviations.
fn observation_noise_matrix(coord: &EquCoord) -> Matrix2<f64> {
    Matrix2::from_diagonal(&Vector2::new(
        coord.ra_error * coord.ra_error,
        coord.dec_error * coord.dec_error,
    ))
}

/// Predictive log-likelihood of a 2-D Gaussian innovation.
///
/// $$\log \mathcal{L} = -\frac{1}{2} \bigl( \ln \det S + d^2 \bigr)$$
///
/// The $\ln(2\pi)$ constant is omitted as it cancels during weight
/// normalization.  Returns `None` if $\det S \le 0$ (degenerate covariance).
fn compute_log_likelihood(s: &Matrix2<f64>, d2: f64) -> Option<f64> {
    let det = s.determinant();
    if det <= 0.0 {
        return None;
    }
    Some(-0.5 * (det.min(MAX_INNOVATION_DET).ln() + d2))
}

/// Push a log-likelihood value into a bounded sliding window.
///
/// When `window_size == 0`, this is a no-op (smoothing is disabled).
/// Older entries are evicted from the front once the deque reaches capacity.
fn push_log_lik_to_window(deque: &mut VecDeque<f64>, log_lik: f64, window_size: usize) {
    if window_size == 0 {
        return;
    }
    deque.push_back(log_lik);
    // Evict the oldest entry if we've exceeded the window capacity.
    while deque.len() > window_size {
        deque.pop_front();
    }
}

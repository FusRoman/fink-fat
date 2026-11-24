use crate::logging::LogTarget;

/// Structured log events for the hypothesis-bank predict/score/update/
/// prune/merge cycle (both bank-level, `kalman_bank/mod.rs`, and
/// hypothesis-level, `hypothesis.rs`). See [`crate::logging`] for the
/// `.emit()` pattern.
pub enum BankEvent {
    StepStart {
        n_hypotheses: usize,
    },
    PredictUpdateComplete {
        n_survivors: usize,
        n_gated: usize,
        n_failed: usize,
    },
    HypothesisPropagated {
        hyp_id: u64,
    },
    HypothesisPropagationFailed {
        hyp_id: u64,
        error: String,
    },
    GateRejectedNonFinite {
        hyp_id: u64,
    },
    GateExemptProtected {
        hyp_id: u64,
        d2: f64,
        gate_chi2: f64,
    },
    GateRejected {
        hyp_id: u64,
        d2: f64,
        gate_chi2: f64,
    },
    GateOk {
        hyp_id: u64,
        d2: f64,
    },
    InnovationFailed {
        hyp_id: u64,
        error: String,
    },
    InnovationCovarianceNotInvertible {
        hyp_id: u64,
    },
    LikelihoodSingular {
        hyp_id: u64,
    },
    LikelihoodScored {
        hyp_id: u64,
        log_lik: f64,
    },
    MeasurementUpdateFailed {
        hyp_id: u64,
        error: String,
    },
    MeasurementUpdateOk {
        hyp_id: u64,
    },
    PredictionFailed {
        hyp_id: u64,
        error: String,
    },
    PruningPhase {
        phase: &'static str,
        n_before: usize,
        n_after: usize,
    },
    SmoothedPruningSkipped,
    SmoothedPruning {
        n_removed: usize,
        log_threshold: f64,
        best_smoothed: f64,
        window: usize,
        min_kept: usize,
    },
    WeightFloorPruning {
        n_removed: usize,
        weight_floor: f64,
        min_kept: usize,
    },
    ScheduledCapTruncation {
        n_truncated: usize,
        effective_cap: usize,
        raw_cap: usize,
        n_steps: usize,
    },
    ModesMerged {
        hyp_id_kept: u64,
        hyp_id_merged: u64,
        threshold_au: f64,
    },
}

crate::impl_log_target!(
    BankEvent,
    "bank",
    "Hypothesis-bank predict/score/update/prune/merge cycle for a single tracklet",
    [tracing::Level::TRACE]
);

impl BankEvent {
    pub fn emit(&self) {
        use BankEvent::*;
        match self {
            StepStart { n_hypotheses } => tracing::trace!(
                target: BankEvent::TARGET, n_hypotheses, "Starting predict/update/score cycle"
            ),
            PredictUpdateComplete {
                n_survivors,
                n_gated,
                n_failed,
            } => tracing::trace!(
                target: BankEvent::TARGET, n_survivors, n_gated, n_failed,
                "Predict/score cycle complete — starting cleanup"
            ),
            HypothesisPropagated { hyp_id } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, "Propagation OK"
            ),
            HypothesisPropagationFailed { hyp_id, error } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, error, "Propagation FAILED"
            ),
            GateRejectedNonFinite { hyp_id } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, "Gate REJECTED: non-finite Mahalanobis²"
            ),
            GateExemptProtected {
                hyp_id,
                d2,
                gate_chi2,
            } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, d2, gate_chi2,
                "Protected hypothesis EXEMPT from chi² gate — high Mahalanobis² recorded but hypothesis preserved"
            ),
            GateRejected {
                hyp_id,
                d2,
                gate_chi2,
            } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, d2, gate_chi2, "Gate REJECTED: Mahalanobis² exceeds threshold"
            ),
            GateOk { hyp_id, d2 } => {
                tracing::trace!(target: BankEvent::TARGET, hyp_id, d2, "Gate OK")
            }
            InnovationFailed { hyp_id, error } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, error, "Innovation FAILED"
            ),
            InnovationCovarianceNotInvertible { hyp_id } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, "Innovation covariance not invertible"
            ),
            LikelihoodSingular { hyp_id } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, "Innovation covariance singular (det ≤ 0)"
            ),
            LikelihoodScored { hyp_id, log_lik } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, log_lik, "Likelihood scored"
            ),
            MeasurementUpdateFailed { hyp_id, error } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, error, "Measurement update FAILED"
            ),
            MeasurementUpdateOk { hyp_id } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, "Measurement update OK"
            ),
            PredictionFailed { hyp_id, error } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id, error, "Hypothesis prediction failed, dropping"
            ),
            PruningPhase {
                phase,
                n_before,
                n_after,
            } => tracing::trace!(
                target: BankEvent::TARGET, phase, n_before, n_after, "Post-step cleanup phase"
            ),
            SmoothedPruningSkipped => tracing::trace!(
                target: BankEvent::TARGET, "Smoothed pruning skipped: window not yet populated"
            ),
            SmoothedPruning {
                n_removed,
                log_threshold,
                best_smoothed,
                window,
                min_kept,
            } => tracing::trace!(
                target: BankEvent::TARGET, n_removed, log_threshold, best_smoothed, window, min_kept,
                "Smoothed-score pruning"
            ),
            WeightFloorPruning {
                n_removed,
                weight_floor,
                min_kept,
            } => tracing::trace!(
                target: BankEvent::TARGET, n_removed, weight_floor, min_kept, "Weight-floor pruning"
            ),
            ScheduledCapTruncation {
                n_truncated,
                effective_cap,
                raw_cap,
                n_steps,
            } => tracing::trace!(
                target: BankEvent::TARGET, n_truncated, effective_cap, raw_cap, n_steps, "Scheduled cap truncation"
            ),
            ModesMerged {
                hyp_id_kept,
                hyp_id_merged,
                threshold_au,
            } => tracing::trace!(
                target: BankEvent::TARGET, hyp_id_kept, hyp_id_merged, threshold_au, "Merging spatially coincident modes"
            ),
        }
    }

    pub fn span(epoch: f64, n_hypotheses: usize) -> tracing::Span {
        tracing::trace_span!(target: BankEvent::TARGET, "kf_bank_step", epoch, n_hypotheses)
    }
}

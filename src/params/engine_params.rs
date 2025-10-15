// src/propagation/engine_params.rs

use serde::{Deserialize, Serialize};

use crate::{
    errors::{EngineParamError, ParamError},
    params::{
        min_cost_flow_params::{MinCostFlowConfig, MinCostFlowConfigBuilder},
        propagator_params::{ModelNoise, PredictorParams, PredictorParamsBuilder},
        scoring_params::{ScoreConfig, ScoreConfigBuilder},
    },
};

use std::fmt;

/* -------------------------------------------------------------------------- */
/*  CandidateLimits                                                            */
/* -------------------------------------------------------------------------- */

/// Limits controlling candidate generation and graph size.
///
/// Apply these to keep the bipartite problem **sparse** and the solver fast.
/// These limits are enforced **after** scoring/gating, **per left seed** (Top-K),
/// and optionally on the **global** edge list (max_total_edges).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateLimits {
    /// Keep at most this many **lowest-cost** edges per **left** seed after scoring.
    ///
    /// Tip: 4–16 is often a good range when gating is effective.
    pub top_k_per_left: usize,
    /// Optional cap on the **total** number of edges (after concatenating all Top-K).
    /// Use `None` to disable.
    ///
    /// If exceeded, the global edge list is sorted by cost and truncated.
    #[serde(default)]
    pub max_total_edges: Option<usize>,
    /// Optional **hard cost cutoff**: discard edges with `cost > max_cost`.
    ///
    /// Use this to reject outliers even before Top-K selection.
    pub max_cost: Option<f64>,
}

impl Default for CandidateLimits {
    fn default() -> Self {
        Self {
            top_k_per_left: 8,
            max_total_edges: None,
            max_cost: None,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Top-level config                                                           */
/* -------------------------------------------------------------------------- */

/// End-to-end config for linking **one pair of nights**.
///
/// This aggregates **prediction**, **scoring**, and **graph-size** controls for
/// the `N_left → N_right` pairwise problem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterNightLinkConfig {
    /// Predictor parameters (cone inflation, model noise, and cell-padding).
    ///
    /// See [`PredictorParams`] for details. The `k_sigma` inflation and `pad_cell_radius`
    /// control the **cone robustness** against propagation error and pixelization.
    pub predict: PredictorParams,
    /// Scoring configuration (gates + weights + scales).
    ///
    /// See [`ScoreConfig`] and the `scoring` module for detailed semantics.
    pub scoring: ScoreConfig,
    /// Limits to keep the graph sparse and the solver fast.
    ///
    /// See [`CandidateLimits`].
    pub limits: CandidateLimits,
    /// Min-Cost Flow backend configuration for the global linker.
    /// See [`MinCostFlowConfig`].
    pub mcf: MinCostFlowConfig,

    pub max_speed_rad_per_day: Option<f64>,
}

impl Default for InterNightLinkConfig {
    fn default() -> Self {
        Self {
            predict: PredictorParams {
                k_sigma: 3.0,
                noise: ModelNoise {
                    variance_floor: 1e-12,
                    drift_per_day: 0.0,
                    curvature_per_day2: 5e-14,
                },
                pad_cell_radius: true,
            },
            scoring: ScoreConfig {
                weights: Default::default(),
                gates: Default::default(),
                scales: Default::default(),
            },
            limits: Default::default(),
            mcf: MinCostFlowConfig::default(),
            max_speed_rad_per_day: None,
        }
    }
}

impl InterNightLinkConfig {
    /// Convenience constructor returning project defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

/* -------------------------------------------------------------------------- */
/*  CandidateLimitsBuilder                                                     */
/* -------------------------------------------------------------------------- */

/// Builder for [`CandidateLimits`] with basic validation.
#[derive(Clone, Debug, Default)]
pub struct CandidateLimitsBuilder {
    inner: CandidateLimits,
}

impl CandidateLimitsBuilder {
    /// Create a new limits builder seeded with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of edges kept **per left seed** (Top-K).
    pub fn top_k_per_left(&mut self, v: usize) -> &mut Self {
        self.inner.top_k_per_left = v;
        self
    }

    /// Set a global cap on the total number of edges (after all Top-K), or disable with `None`.
    pub fn max_total_edges(&mut self, v: Option<usize>) -> &mut Self {
        self.inner.max_total_edges = v;
        self
    }

    /// Convenience: disable global edge cap.
    pub fn clear_max_total_edges(&mut self) -> &mut Self {
        self.inner.max_total_edges = None;
        self
    }

    /// Set a hard cost cutoff `max_cost`, or disable with `None`.
    pub fn max_cost(&mut self, v: Option<f64>) -> &mut Self {
        self.inner.max_cost = v;
        self
    }

    /// Convenience: disable hard cost cutoff.
    pub fn clear_max_cost(&mut self) -> &mut Self {
        self.inner.max_cost = None;
        self
    }

    /// Validate and build.
    pub fn build(&self) -> Result<CandidateLimits, ParamError> {
        let l = self.inner;
        if l.top_k_per_left == 0 {
            return Err(ParamError::Engine(EngineParamError::InvalidTopK(
                l.top_k_per_left,
            )));
        }
        if let Some(n) = l.max_total_edges {
            if n == 0 {
                return Err(ParamError::Engine(EngineParamError::InvalidMaxTotal(n)));
            }
        }
        if let Some(c) = l.max_cost {
            if !c.is_finite() || c < 0.0 {
                return Err(ParamError::Engine(EngineParamError::InvalidMaxCost(c)));
            }
        }
        Ok(l)
    }

    /* -------------------------- Practical presets -------------------------- */

    /// Wider fan-out for **recall**, expects strong scoring/gating downstream.
    ///
    /// - `top_k_per_left = 16`, `max_cost = Some(12.0)`, no global cap.
    pub fn preset_recall_wide(&mut self) -> &mut Self {
        self.inner.top_k_per_left = 16;
        self.inner.max_total_edges = None;
        self.inner.max_cost = Some(12.0);
        self
    }

    /// Stricter fan-out for **speed**, good once scoring is calibrated.
    ///
    /// - `top_k_per_left = 8`, `max_cost = Some(9.0)`, optional small global cap.
    pub fn preset_speed_strict(&mut self) -> &mut Self {
        self.inner.top_k_per_left = 8;
        self.inner.max_total_edges = None;
        self.inner.max_cost = Some(9.0);
        self
    }
}

/* -------------------------------------------------------------------------- */
/*  InterNightLinkConfigBuilder                                                */
/* -------------------------------------------------------------------------- */

/// Builder for [`InterNightLinkConfig`].
///
/// Two usage styles are supported:
///
/// 1) **Ergonomic Rust style** with nested builders:
///
/// ```rust
/// use fink_fat::params::engine_params::InterNightLinkConfigBuilder;
/// let cfg = InterNightLinkConfigBuilder::new()
///     .with_predict(|p| p.k_sigma(3.5).set_q0(1e-12).set_q2(5e-14).pad_cell_radius(true))
///     .with_scoring(|s| s.set_w_pos(0.7).set_max_d2_pos(12.0))
///     .with_limits(|l| l.top_k_per_left(12).max_cost(Some(10.0)))
///     .build().unwrap();
/// ```
///
/// 2) **Flat setters** (Python-friendly):
///
/// ```rust
/// use fink_fat::params::engine_params::InterNightLinkConfigBuilder;
/// let cfg = InterNightLinkConfigBuilder::new()
///     .set_k_sigma(3.5)
///     .set_noise_q0(1e-12).set_noise_q2(5e-14)
///     .set_pad_cell_radius(true)
///     .set_w_pos(0.7).set_max_d2_pos(12.0)
///     .set_top_k_per_left(12).set_max_cost(Some(10.0))
///     .build().unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct InterNightLinkConfigBuilder {
    // nested builders (so we can validate each sub-config)
    predict: PredictorParamsBuilder,
    scoring: ScoreConfigBuilder,
    limits: CandidateLimitsBuilder,
    mcf: MinCostFlowConfigBuilder,
    max_speed_rad_per_day: Option<f64>,
}

impl Default for InterNightLinkConfigBuilder {
    fn default() -> Self {
        Self {
            predict: PredictorParamsBuilder::new(), // defaults: k=3.0, pad=true, noise=0
            scoring: ScoreConfigBuilder::new(),     // your defaults for scoring
            limits: CandidateLimitsBuilder::new(),  // top_k=8, etc.
            mcf: MinCostFlowConfigBuilder::default(),
            max_speed_rad_per_day: None,
        }
    }
}

impl InterNightLinkConfigBuilder {
    /// Create a new builder seeded with sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /* ----------------------- Nested (ergonomic Rust) ----------------------- */

    /// Mutate predictor parameters via its own builder (by-value).
    pub fn with_predict<F>(mut self, f: F) -> Self
    where
        F: FnOnce(PredictorParamsBuilder) -> PredictorParamsBuilder,
    {
        self.predict = f(self.predict);
        self
    }

    /// Mutate scoring configuration via its own builder (by-value).
    pub fn with_scoring<F>(mut self, f: F) -> Self
    where
        F: FnOnce(ScoreConfigBuilder) -> ScoreConfigBuilder,
    {
        self.scoring = f(self.scoring);
        self
    }

    /// Mutate candidate limits via its own builder (kept by-ref: its API est &mut).
    pub fn with_limits<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut CandidateLimitsBuilder) -> &mut CandidateLimitsBuilder,
    {
        let _ = f(&mut self.limits);
        self
    }

    /// Mutate min-cost flow configuration via its own builder (by-value).
    pub fn with_mcf<F>(mut self, f: F) -> Self
    where
        F: FnOnce(MinCostFlowConfigBuilder) -> MinCostFlowConfigBuilder,
    {
        self.mcf = f(self.mcf);
        self
    }

    /* ------------------- Flat setters (Python-friendly) -------------------- */
    /* Predictor passthrough */
    pub fn set_k_sigma(mut self, v: f64) -> Self {
        self.predict = self.predict.k_sigma(v);
        self
    }
    pub fn set_noise_q0(mut self, v: f64) -> Self {
        self.predict = self.predict.set_noise_variance_floor(v);
        self
    }
    pub fn set_noise_q1(mut self, v: f64) -> Self {
        self.predict = self.predict.set_noise_drift_per_day(v);
        self
    }
    pub fn set_noise_q2(mut self, v: f64) -> Self {
        self.predict = self.predict.set_noise_curvature_per_day2(v);
        self
    }
    pub fn set_pad_cell_radius(mut self, yes: bool) -> Self {
        self.predict = self.predict.pad_cell_radius(yes);
        self
    }

    /* Scoring passthrough (common setters; add more if you expose them) */
    // Weights
    pub fn set_w_pos(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_w_pos(v);
        self
    }
    pub fn set_w_vel_dir(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_w_vel_dir(v);
        self
    }
    pub fn set_w_vel_norm(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_w_vel_norm(v);
        self
    }
    pub fn set_w_flux(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_w_flux(v);
        self
    }
    pub fn set_w_gap(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_w_gap(v);
        self
    }
    pub fn set_w_band_mismatch(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_w_band_mismatch(v);
        self
    }
    // Gates
    pub fn set_max_d2_pos(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_max_d2_pos(v);
        self
    }
    pub fn set_max_theta_vel(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_max_theta_vel(v);
        self
    }
    pub fn set_max_speed_diff(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_max_speed_diff(v);
        self
    }
    // Scales
    pub fn set_theta0(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_theta0(v);
        self
    }
    pub fn set_v0(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_v0(v);
        self
    }
    pub fn set_flux_sigma_floor(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_flux_sigma_floor(v);
        self
    }
    pub fn set_gap_rho(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_gap_rho(v);
        self
    }
    pub fn set_vel_eps_days(mut self, v: f64) -> Self {
        self.scoring = self.scoring.set_vel_eps_days(v);
        self
    }

    /* Limits (flat) */
    pub fn set_top_k_per_left(mut self, v: usize) -> Self {
        self.limits.top_k_per_left(v);
        self
    }
    pub fn set_max_total_edges(mut self, v: Option<usize>) -> Self {
        self.limits.max_total_edges(v);
        self
    }
    pub fn clear_max_total_edges(mut self) -> Self {
        self.limits.clear_max_total_edges();
        self
    }
    pub fn set_max_cost(mut self, v: Option<f64>) -> Self {
        self.limits.max_cost(v);
        self
    }
    pub fn clear_max_cost(mut self) -> Self {
        self.limits.clear_max_cost();
        self
    }

    pub fn set_max_speed_rad_per_day(mut self, v: Option<f64>) -> Self {
        self.max_speed_rad_per_day = v;
        self
    }

    /* Min-Cost Flow configuration */
    pub fn set_lambda_start(mut self, v: f64) -> Self {
        self.mcf = self.mcf.lambda_start(v);
        self
    }

    pub fn set_lambda_end(mut self, v: f64) -> Self {
        self.mcf = self.mcf.lambda_end(v);
        self
    }

    pub fn set_gap_penalty_weight(mut self, v: f64) -> Self {
        self.mcf = self.mcf.gap_penalty_weight(v);
        self
    }

    pub fn set_max_revisit_gap(mut self, v: u32) -> Self {
        self.mcf = self.mcf.max_revisit_gap(v);
        self
    }

    pub fn set_max_total_flow(mut self, v: Option<u32>) -> Self {
        self.mcf = self.mcf.max_total_flow(v);
        self
    }

    pub fn set_horizon_nights(mut self, v: usize) -> Self {
        self.mcf = self.mcf.horizon_nights(v);
        self
    }

    /// Validate and build the final configuration.
    pub fn build(self) -> Result<InterNightLinkConfig, ParamError> {
        // Predictor + Scoring delegate to their own validation.
        let predict = self
            .predict
            .build()
            .map_err(|e| ParamError::Engine(EngineParamError::InvalidPredictor(e.to_string())))?;

        let scoring = self
            .scoring
            .build()
            .map_err(|e| ParamError::Engine(EngineParamError::InvalidScoring(e.to_string())))?;

        let limits = self.limits.build()?;

        let mcf = self.mcf.build()?;

        Ok(InterNightLinkConfig {
            predict,
            scoring,
            limits,
            mcf,
            max_speed_rad_per_day: self.max_speed_rad_per_day,
        })
    }

    /* ------------------------------- Presets ------------------------------- */

    /// **Recall-focused** preset: wider cones and fan-out; combine with strict gating.
    ///
    /// - Predictor: `k_sigma = 3.5`, conservative noise (pairs), padding on.
    /// - Limits: `top_k_per_left = 16`, `max_cost = Some(12.0)`, no global cap.
    pub fn preset_recall(mut self) -> Self {
        self.predict = self.predict.preset_pairs_conservative();
        let _ = self.limits.preset_recall_wide();
        self
    }

    /// **Speed-focused** preset: tighter fan-out; good once calibrated.
    ///
    /// - Predictor: `k_sigma = 3.0`, padding on (noise defaults ok).
    /// - Limits: `top_k_per_left = 8`, `max_cost = Some(9.0)`.
    pub fn preset_speed(mut self) -> Self {
        // Keep predictor defaults but ensure pad=true (already default).
        self.predict = self.predict.k_sigma(3.0).pad_cell_radius(true);
        let _ = self.limits.preset_speed_strict();
        self
    }

    /// **Triplet-friendly** preset: tighter noise model & standard fan-out.
    pub fn preset_triplets(mut self) -> Self {
        self.predict = self.predict.preset_triplets_tight();
        // limits default (TopK=8) is fine
        self
    }
}

/* -------------------------------------------------------------------------- */
/*  Display helpers                                                            */
/* -------------------------------------------------------------------------- */

impl fmt::Display for CandidateLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CandidateLimits {{ top_k_per_left:{}, max_total_edges:{:?}, max_cost:{:?} }}",
            self.top_k_per_left, self.max_total_edges, self.max_cost
        )
    }
}

impl fmt::Display for InterNightLinkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InterNightLinkConfig {{ predict: {}, scoring: {}, limits: {} }}",
            self.predict, self.scoring, self.limits
        )
    }
}

/* -------------------------------------------------------------------------- */
/*  Tests                                                                      */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_build_ok() {
        let cfg = InterNightLinkConfigBuilder::new().build().unwrap();
        assert!((cfg.predict.k_sigma - 3.0).abs() < 1e-12);
        assert!(cfg.predict.pad_cell_radius);
        assert_eq!(cfg.limits.top_k_per_left, 8);
    }

    #[test]
    fn flat_setters_passthrough() {
        let cfg = InterNightLinkConfigBuilder::new()
            .set_k_sigma(3.7)
            .set_noise_q0(1e-12)
            .set_noise_q2(3e-14)
            .set_pad_cell_radius(false)
            .set_w_pos(0.9)
            .set_max_d2_pos(14.0)
            .set_theta0((4.0f64).to_radians())
            .set_top_k_per_left(12)
            .set_max_cost(Some(10.0))
            .build()
            .unwrap();

        assert!((cfg.predict.k_sigma - 3.7).abs() < 1e-12);
        assert!(!cfg.predict.pad_cell_radius);
        assert!((cfg.scoring.weights.w_pos - 0.9).abs() < 1e-12);
        assert!((cfg.scoring.gates.max_d2_pos - 14.0).abs() < 1e-12);
        assert!((cfg.scoring.scales.theta0 - (4.0f64).to_radians()).abs() < 1e-12);
        assert_eq!(cfg.limits.top_k_per_left, 12);
        assert_eq!(cfg.limits.max_cost, Some(10.0));
    }

    #[test]
    fn nested_builders_passthrough() {
        let cfg = InterNightLinkConfigBuilder::new()
            .with_predict(|p| {
                p.k_sigma(3.7)
                    .set_noise_variance_floor(1e-12)
                    .set_noise_curvature_per_day2(3e-14)
                    .pad_cell_radius(false)
            })
            .with_scoring(|s| {
                s.set_w_pos(0.9)
                    .set_max_d2_pos(14.0)
                    .set_theta0((4.0f64).to_radians())
            })
            .with_limits(|l| l.top_k_per_left(12).max_cost(Some(10.0)))
            .build()
            .unwrap();

        assert!((cfg.predict.k_sigma - 3.7).abs() < 1e-12);
        assert!(!cfg.predict.pad_cell_radius);
        assert!((cfg.scoring.weights.w_pos - 0.9).abs() < 1e-12);
        assert!((cfg.scoring.gates.max_d2_pos - 14.0).abs() < 1e-12);
        assert!((cfg.scoring.scales.theta0 - (4.0f64).to_radians()).abs() < 1e-12);
        assert_eq!(cfg.limits.top_k_per_left, 12);
        assert_eq!(cfg.limits.max_cost, Some(10.0));
    }

    #[test]
    fn limits_validation() {
        let err = InterNightLinkConfigBuilder::new()
            .with_limits(|l| l.top_k_per_left(0))
            .build()
            .unwrap_err();
        matches!(err, ParamError::Engine(EngineParamError::InvalidTopK(0)));

        let err = InterNightLinkConfigBuilder::new()
            .with_limits(|l| l.max_total_edges(Some(0)))
            .build()
            .unwrap_err();
        matches!(
            err,
            ParamError::Engine(EngineParamError::InvalidMaxTotal(0))
        );

        let err = InterNightLinkConfigBuilder::new()
            .with_limits(|l| l.max_cost(Some(-1.0)))
            .build()
            .unwrap_err();
        matches!(err, ParamError::Engine(EngineParamError::InvalidMaxCost(_)));
    }

    #[test]
    fn presets_sanity() {
        let recall = InterNightLinkConfigBuilder::new()
            .preset_recall()
            .build()
            .unwrap();
        assert!(recall.predict.k_sigma >= 3.0);
        assert!(recall.limits.top_k_per_left >= 12);

        let speed = InterNightLinkConfigBuilder::new()
            .preset_speed()
            .build()
            .unwrap();
        assert_eq!(speed.limits.top_k_per_left, 8);

        let trip = InterNightLinkConfigBuilder::new()
            .preset_triplets()
            .build()
            .unwrap();
        assert!(trip.predict.k_sigma >= 3.0); // default/tight
    }
}

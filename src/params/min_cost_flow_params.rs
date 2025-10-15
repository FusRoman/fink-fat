use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::ParamError;

/* -------------------------------------------------------------------------- */
/*  MinCostFlowConfig                                                          */
/* -------------------------------------------------------------------------- */

/// Configurable penalties and structural limits for MCF.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinCostFlowConfig {
    /// Penalty to **start** a new trajectory from Source (λ_start).
    pub lambda_start: f64,
    /// Penalty to **end** a trajectory into Sink (λ_end).
    pub lambda_end: f64,
    /// Optional weight for gap penalty when Δ>1 revisit steps are used.
    pub gap_penalty_weight: f64,
    /// Maximum allowed revisit gap (in integer nights) for link arcs; Δ in [1..=max].
    pub max_revisit_gap: u32,
    /// Optional maximum number of trajectories (total flow units).
    pub max_total_flow: Option<u32>,
    /// How many **previous nights** to connect to the current one (Δ=1..H).
    pub horizon_nights: usize,
}

impl Default for MinCostFlowConfig {
    fn default() -> Self {
        Self {
            lambda_start: 1.0,
            lambda_end: 1.0,
            gap_penalty_weight: 0.0,
            max_revisit_gap: 1,
            max_total_flow: None,
            horizon_nights: 1,
        }
    }
}

impl MinCostFlowConfig {
    /// Create a builder seeded with sane defaults.
    ///
    /// Returns
    /// -------
    /// `MinCostFlowConfigBuilder`
    ///     Builder with defaults:
    ///     `lambda_start=lambda_end=1.0`, `gap_penalty_weight=0.0`,
    ///     `max_revisit_gap=1`, `max_total_flow=None`.
    pub fn builder() -> MinCostFlowConfigBuilder {
        MinCostFlowConfigBuilder::new()
    }
}

impl fmt::Display for MinCostFlowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cap = self
            .max_total_flow
            .map(|n| n.to_string())
            .unwrap_or_else(|| "None".to_string());
        write!(
            f,
            "MCF(l_start={:.3}, l_end={:.3}, w_gap={:.3}, Δ_max={}, flow_cap={})",
            self.lambda_start, self.lambda_end, self.gap_penalty_weight, self.max_revisit_gap, cap
        )
    }
}

/* -------------------------------------------------------------------------- */
/*  Builder                                                                    */
/* -------------------------------------------------------------------------- */

/// Builder for [`MinCostFlowConfig`] with validation.
///
/// Overview
/// --------
/// This builder exposes **flat setters** so it can be driven from Python
/// (via simple keyword arguments). It performs basic sanity checks on
/// penalties and discrete limits when `build()` is called.
///
/// Validation
/// ----------
/// - `lambda_start`, `lambda_end`, `gap_penalty_weight` must be **finite** and `≥ 0`.
/// - `max_revisit_gap` must be `≥ 1`.
/// - `max_total_flow` if `Some(n)` must be `n ≥ 1`.
///
/// Errors are returned as [`ParamError::Inconsistent`] with a descriptive message.
///
/// Examples
/// --------
/// Basic construction:
/// ```rust
/// use fink_fat::params::linking_params::MinCostFlowConfig;
/// let cfg = MinCostFlowConfig::builder()
///     .lambda_start(0.8)
///     .lambda_end(1.2)
///     .gap_penalty_weight(0.25)
///     .max_revisit_gap(3)
///     .max_total_flow(Some(50))
///     .build()
///     .unwrap();
/// ```
///
/// Unlimited total flow:
/// ```rust
/// let cfg = MinCostFlowConfig::builder()
///     .unlimited_total_flow()
///     .build()
///     .unwrap();
/// ```
///
/// Python-friendly setters:
/// ```rust
/// let cfg = MinCostFlowConfigBuilder::new()
///     .set_lambda_start(1.0)
///     .set_lambda_end(1.0)
///     .set_gap_penalty_weight(0.0)
///     .set_max_revisit_gap(1)
///     .set_max_total_flow(None)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Debug, Default)]
pub struct MinCostFlowConfigBuilder {
    inner: MinCostFlowConfig,
}

impl MinCostFlowConfigBuilder {
    /// Create a new builder seeded with defaults.
    pub fn new() -> Self {
        Self {
            inner: MinCostFlowConfig::default(),
        }
    }

    /* ----------------------------- Fluent API ----------------------------- */

    /// Set `lambda_start` (≥ 0, finite).
    pub fn lambda_start(mut self, v: f64) -> Self {
        self.inner.lambda_start = v;
        self
    }

    /// Set `lambda_end` (≥ 0, finite).
    pub fn lambda_end(mut self, v: f64) -> Self {
        self.inner.lambda_end = v;
        self
    }

    /// Set `gap_penalty_weight` (≥ 0, finite).
    pub fn gap_penalty_weight(mut self, v: f64) -> Self {
        self.inner.gap_penalty_weight = v;
        self
    }

    /// Set `max_revisit_gap` (≥ 1).
    pub fn max_revisit_gap(mut self, v: u32) -> Self {
        self.inner.max_revisit_gap = v;
        self
    }

    /// Set `max_total_flow` (`Some(n≥1)` or `None` for unlimited).
    pub fn max_total_flow(mut self, v: Option<u32>) -> Self {
        self.inner.max_total_flow = v;
        self
    }

    /// Convenience: disable flow cap (`max_total_flow=None`).
    pub fn unlimited_total_flow(mut self) -> Self {
        self.inner.max_total_flow = None;
        self
    }

    /// Set `horizon_nights` (≥ 1).
    pub fn horizon_nights(mut self, v: usize) -> Self {
        self.inner.horizon_nights = v;
        self
    }

    /* ------------------------------ Build -------------------------------- */

    /// Build and validate the configuration.
    ///
    /// Returns
    /// -------
    /// * `Ok(MinCostFlowConfig)` if all fields pass sanity checks.
    /// * `Err(ParamError::Inconsistent(_))` otherwise.
    pub fn build(self) -> Result<MinCostFlowConfig, ParamError> {
        let c = self.inner;

        // Helper to return a static message per field (enum expects &'static str).
        fn penalty_err(name: &str) -> ParamError {
            match name {
                "lambda_start" => ParamError::Inconsistent(
                    "MinCostFlowConfig: `lambda_start` must be finite and >= 0.",
                ),
                "lambda_end" => ParamError::Inconsistent(
                    "MinCostFlowConfig: `lambda_end` must be finite and >= 0.",
                ),
                "gap_penalty_weight" => ParamError::Inconsistent(
                    "MinCostFlowConfig: `gap_penalty_weight` must be finite and >= 0.",
                ),
                "horizon_nights" => {
                    ParamError::Inconsistent("MinCostFlowConfig: `horizon_nights` must be >= 1.")
                }
                _ => {
                    ParamError::Inconsistent("MinCostFlowConfig: penalty must be finite and >= 0.")
                }
            }
        }

        // Validate non-negative, finite penalties
        let penalties = [
            ("lambda_start", c.lambda_start),
            ("lambda_end", c.lambda_end),
            ("gap_penalty_weight", c.gap_penalty_weight),
        ];
        for (name, v) in penalties {
            if !v.is_finite() || v < 0.0 {
                return Err(penalty_err(name));
            }
        }

        // Validate discrete limits
        if c.max_revisit_gap < 1 {
            return Err(ParamError::Inconsistent(
                "MinCostFlowConfig: `max_revisit_gap` must be >= 1.",
            ));
        }
        if let Some(n) = c.max_total_flow {
            if n == 0 {
                return Err(ParamError::Inconsistent(
                    "MinCostFlowConfig: `max_total_flow` must be >= 1 when provided.",
                ));
            }
        }

        Ok(c)
    }
}

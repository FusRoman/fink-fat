//! Propagation predictor parameters: kσ cone sizing + additive model noise.
//!
//! # Overview
//! At inter-night linking time, each seed (see [`SeedNode`](crate::propagation::features::SeedNode)) is propagated to a
//! target epoch `t_target` on its **gnomonic tangent plane**. The predicted plane
//! covariance `Σ_p(t)` is turned into a **sky cone** used to retrieve candidates in
//! the target night:
//!
//! 1. Build the **plane covariance** (per axis) with a simple, interpretable rule:
//!    ```text
//!    Σ_p(t) ≈ Σ_pos  +  Δt² Σ_vel  +  Q(Δt)
//!    ```
//!    where the **additive model-noise schedule** is a low-order polynomial
//!    `Q(Δt) = σ²_floor + β_drift · |Δt| + γ_curv · Δt²` (see [`ModelNoise`]).
//! 2. Convert the mean plane position back to the sky (inverse gnomonic).
//! 3. Extract a **single conservative radius** from `Σ_p(t)` as
//!    `r = k_sigma · sqrt(λ_max(Σ_p))` (major-axis 1σ).
//! 4. Optionally **pad** by one spatial cell radius to be robust to **cell coverage**
//!    in approximate cone queries (e.g., HEALPix neighbor unions).
//!
//! # Why `λ_max`?
//! For a 2×2 symmetric covariance, the 1σ contour is an ellipse with semi-axes
//! `sqrt(λ₁) ≥ sqrt(λ₂)`. Using `sqrt(λ_max)` yields a **circle that contains the
//! ellipse** at the same sigma level — safe for candidate retrieval.
//!
//! ```text
//!  ellipse (1σ)              circumscribed circle (1σ)
//!        ^ y                     ^ y
//!        |                       |
//!   ____/ \____                  |¯¯¯¯¯¯¯¯|
//!  /            \                |   •    |   radius = sqrt(λ_max)
//! |     •        |   →   center  |        |   (major axis)
//!  \____    ____/                |________|
//!       \  /                         |
//!        \/                          v x
//! ```
//!
//! # Padding for cell-based coverage
//! When your spatial index uses **discrete cells** to approximate a cone, add one
//! `cell_radius()` to avoid under-coverage at cell boundaries:
//!
//! ```text
//!   true circle (radius r)          cell coverage (r + cell_radius)
//!         _____                                  _______
//!      .-'     '-.                            .-'       '-.
//!    .'    •      '.          vs          _.-'   •         '-._
//!   /               \                    /                     \
//!   |       r        |                  |    r + cell_radius   |
//!   \               /                    \                     /
//!    '.           .'                      '-._             _.-'
//!      '-._____.-'                            '-----------'
//! ```
//!
//! # See also
//! * [`SeedNode::predict_cone`](crate::propagation::features::SeedNode::predict_cone) — Uses these parameters to produce (ra, dec, radius).
//! * [`SeedNode::predict_on_plane`](crate::propagation::features::SeedNode::predict_on_plane) — Mean & diagonal covariance on the plane.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::errors::{ParamError, PredictorParamError};

/* -------------------------------------------------------------------------- */
/*  Core types                                                                */
/* -------------------------------------------------------------------------- */

/// Additive model-noise schedule to cover unmodeled curvature and model mismatch.
///
/// Overview
/// --------
/// `ModelNoise` parameterizes a simple time-dependent **variance** term `Q(Δt)`
/// that is **added per axis** to the predicted plane covariance:
/// `Σ_p(t) ≈ Σ_pos + Δt² Σ_vel + Q(Δt)`. It is especially useful for **pairs**
/// (constant-velocity seeds) where true motion exhibits curvature over multi-night gaps.
///
/// Form
/// ----
/// ```text
/// Q(Δt) = σ²_floor + β_drift · |Δt| + γ_curv · Δt²
/// ```
/// with `Δt = t_target − epoch_mid` in **days**. All coefficients must be **≥ 0**.
///
/// Fields
/// ------
/// - `variance_floor` (σ²_floor) — Static variance floor (rad²). Compensates small
///   systematics and plane/sphere approximations even at `Δt = 0`.
/// - `drift_per_day` (β_drift) — Linear growth (rad²/day). Captures slow drift-like
///   effects (e.g., slight velocity bias).
/// - `curvature_per_day2` (γ_curv) — Quadratic growth (rad²/day²). Covers curvature-like
///   divergence that increases faster with |Δt|.
///
/// Units
/// -----
/// - `variance_floor` in **radians²**,
/// - `drift_per_day` in **radians²/day**,
/// - `curvature_per_day2` in **radians²/day²**.
///
/// Notes
/// -----
/// - Start conservatively for pairs, e.g., `σ²_floor ≈ (0.15″)^2` in rad², small linear term,
///   and a quadratic term tuned on simulation (Sorcha) to reach high recall over 1–2 days.
/// - For triplets (with acceleration), coefficients can be smaller; avoid setting all to zero
///   if you want to absorb residual modeling error.
/// - The schedule is **isotropic** (same on x and y). If later you adopt anisotropic
///   propagation, extend this to a 2D form or inject cross-terms downstream.
///
/// Examples
/// --------
/// ```ignore
/// // 0.15 arcsec in radians, squared:
/// let s2 = (0.15_f64 / 3600.0).to_radians().powi(2);
/// let noise = ModelNoise { variance_floor: s2, drift_per_day: 0.0, curvature_per_day2: 5e-14 };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelNoise {
    /// Static variance floor (rad²).
    pub variance_floor: f64,
    /// Linear variance growth per day (rad²/day).
    pub drift_per_day: f64,
    /// Quadratic variance growth per day² (rad²/day²).
    pub curvature_per_day2: f64,
}

/// Parameters controlling sky-cone prediction for candidate retrieval.
///
/// Overview
/// --------
/// `PredictorParams` governs how the **predicted plane covariance** at a target
/// epoch is converted into a **sky-cone search**:
/// 1) compute `Σ_p(t)` (including [`ModelNoise`]),
/// 2) take a conservative **k-sigma circle** whose radius is
///    `r = k_sigma · sqrt(λ_max(Σ_p))`,
/// 3) optionally **pad** that radius by one spatial-cell radius to ensure coverage
///    with coarse binners (e.g., HEALPix).
///
/// Fields
/// ------
/// - `k_sigma` — Confidence multiplier (e.g., `3.0` for 3σ coverage on the largest
///   principal axis). Larger values increase recall but also the number of candidates.
/// - `noise` — Additive variance schedule `Q(Δt)` plugged into the plane covariance
///   before radius extraction; see [`ModelNoise`].
/// - `pad_cell_radius` — If `true`, add `binner.cell_radius()` to the cone radius
///   to compensate for cell-boundary effects in approximate cone coverage.
///
/// Units
/// -----
/// - `k_sigma` is dimensionless,
/// - cone radius returned by prediction is in **radians**.
///
/// Notes
/// -----
/// - Start with `k_sigma = 3.0`. If recall is low on validation, increase a bit
///   (3.5–4.0). After an IOD confirmation stage, you can tighten it back.
/// - `pad_cell_radius = true` is recommended when your spatial binner performs
///   **cell-based coverage** rather than exact geometric cone slicing.
/// - Excessive inflation increases fan-out; cap downstream candidates (Top-K)
///   and apply strict scoring cuts (Mahalanobis) to keep runtime bounded.
///
/// Examples
/// --------
/// ```ignore
/// let params = PredictorParams {
///     k_sigma: 3.0,
///     noise: ModelNoise { variance_floor: 1e-12, drift_per_day: 0.0, curvature_per_day2: 5e-14 },
///     pad_cell_radius: true,
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictorParams {
    /// k-sigma inflation (e.g., 3.0).
    pub k_sigma: f64,
    /// Additive model noise Q(Δt).
    pub noise: ModelNoise,
    /// If true, add one spatial cell radius to the cone (safety padding).
    pub pad_cell_radius: bool,
}

/* -------------------------------------------------------------------------- */
/*  Sub-builder for ModelNoise                                                */
/* -------------------------------------------------------------------------- */

/// Builder for [`ModelNoise`].
///
/// Validates finiteness and non-negativity for all coefficients.
#[derive(Clone, Debug, Default)]
pub struct ModelNoiseBuilder {
    inner: ModelNoise,
}

impl ModelNoiseBuilder {
    /// Create a new noise builder seeded with defaults (all zeros).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build and validate the noise schedule.
    ///
    /// Return
    /// ------
    /// * `Ok(ModelNoise)` if coefficients are finite and ≥ 0.
    /// * `Err(PredictorParamError)` otherwise.
    pub fn build(self) -> Result<ModelNoise, ParamError> {
        let n = self.inner;
        let coeffs = [
            ("variance_floor", n.variance_floor),
            ("drift_per_day", n.drift_per_day),
            ("curvature_per_day2", n.curvature_per_day2),
        ];
        for (name, v) in coeffs {
            if !v.is_finite() || v < 0.0 {
                return Err(ParamError::Predictor(
                    PredictorParamError::InvalidNoiseCoeff { name, value: v },
                ));
            }
        }
        Ok(n)
    }

    /// Set static variance floor (rad²).
    pub fn variance_floor(&mut self, v: f64) -> &mut Self {
        self.inner.variance_floor = v;
        self
    }
    /// Set linear growth (rad²/day).
    pub fn drift_per_day(&mut self, v: f64) -> &mut Self {
        self.inner.drift_per_day = v;
        self
    }
    /// Set quadratic growth (rad²/day²).
    pub fn curvature_per_day2(&mut self, v: f64) -> &mut Self {
        self.inner.curvature_per_day2 = v;
        self
    }

    /// Convenience: conservative preset for **pairs** (curvature hedging).
    ///
    /// Notes
    /// -----
    /// Uses ~0.15″ floor and a small quadratic term; tune `γ_curv` on Sorcha.
    pub fn preset_pairs_conservative(&mut self) -> &mut Self {
        let s2 = (0.15_f64 / 3600.0).to_radians().powi(2);
        self.inner.variance_floor = s2;
        self.inner.drift_per_day = 0.0;
        self.inner.curvature_per_day2 = 5e-14;
        self
    }

    /// Convenience: tighter preset for **triplets** (with acceleration model).
    pub fn preset_triplets_tight(&mut self) -> &mut Self {
        let s2 = (0.08_f64 / 3600.0).to_radians().powi(2);
        self.inner.variance_floor = s2;
        self.inner.drift_per_day = 0.0;
        self.inner.curvature_per_day2 = 1e-14;
        self
    }
}

/* -------------------------------------------------------------------------- */
/*  Top-level builder for PredictorParams                                     */
/* -------------------------------------------------------------------------- */

/// Builder for [`PredictorParams`].
///
/// Two usage styles are supported:
///
/// 1) **Ergonomic Rust style** with a nested noise builder:
///
/// ```rust
/// use fink_fat::params::propagator_params::{PredictorParamsBuilder, ModelNoiseBuilder};
/// let params = PredictorParamsBuilder::new()
///     .k_sigma(3.0)
///     .with_noise(|n| {
///         n.variance_floor(1.0e-12)
///          .drift_per_day(0.0)
///          .curvature_per_day2(5.0e-14)
///     })
///     .pad_cell_radius(true)
///     .build()
///     .unwrap();
/// ```
///
/// 2) **Flat setters** (Python-friendly, no closures/generics in bindings):
///
/// ```rust
/// use fink_fat::params::propagator_params::PredictorParamsBuilder;
/// let params = PredictorParamsBuilder::new()
///     .k_sigma(3.5)
///     .set_noise_variance_floor(1.0e-12)
///     .set_noise_drift_per_day(0.0)
///     .set_noise_curvature_per_day2(5.0e-14)
///     .pad_cell_radius(true)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct PredictorParamsBuilder {
    k_sigma: f64,
    noise: ModelNoise,
    pad_cell_radius: bool,
}

impl Default for PredictorParamsBuilder {
    fn default() -> Self {
        Self {
            k_sigma: 3.0,
            noise: ModelNoise::default(),
            pad_cell_radius: true,
        }
    }
}

impl PredictorParamsBuilder {
    /// Create a new builder with sensible defaults: `k_sigma=3.0`, zero noise, `pad_cell_radius=true`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the k-sigma inflation (must be finite and > 0).
    ///
    /// Arguments
    /// ---------
    /// * `v` – sigma multiplier for the cone radius (dimensionless).
    pub fn k_sigma(mut self, v: f64) -> Self {
        self.k_sigma = v;
        self
    }

    /// Mutate the noise schedule via a nested builder (ergonomic Rust style).
    ///
    /// Arguments
    /// ---------
    /// * `f` – closure that edits a temporary `ModelNoiseBuilder` and returns it.
    pub fn with_noise<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut ModelNoiseBuilder) -> &mut ModelNoiseBuilder,
    {
        let mut nb = ModelNoiseBuilder { inner: self.noise };
        let nb = f(&mut nb);
        self.noise = nb.inner;
        self
    }

    /// Flat setter: set `variance_floor` (rad²).
    pub fn set_noise_variance_floor(mut self, v: f64) -> Self {
        self.noise.variance_floor = v;
        self
    }
    /// Flat setter: set `drift_per_day` (rad²/day).
    pub fn set_noise_drift_per_day(mut self, v: f64) -> Self {
        self.noise.drift_per_day = v;
        self
    }
    /// Flat setter: set `curvature_per_day2` (rad²/day²).
    pub fn set_noise_curvature_per_day2(mut self, v: f64) -> Self {
        self.noise.curvature_per_day2 = v;
        self
    }

    /* -------- Backward-compat flat setters (deprecated q0/q1/q2 names) -------- */
    /// Deprecated: use `set_noise_variance_floor`.
    #[deprecated(note = "Use `set_noise_variance_floor()` instead of `set_q0()`.")]
    pub fn set_q0(self, v: f64) -> Self {
        self.set_noise_variance_floor(v)
    }
    /// Deprecated: use `set_noise_drift_per_day`.
    #[deprecated(note = "Use `set_noise_drift_per_day()` instead of `set_q1()`.")]
    pub fn set_q1(self, v: f64) -> Self {
        self.set_noise_drift_per_day(v)
    }
    /// Deprecated: use `set_noise_curvature_per_day2`.
    #[deprecated(note = "Use `set_noise_curvature_per_day2()` instead of `set_q2()`.")]
    pub fn set_q2(self, v: f64) -> Self {
        self.set_noise_curvature_per_day2(v)
    }

    /// Enable/disable padding by the spatial cell radius.
    pub fn pad_cell_radius(mut self, yes: bool) -> Self {
        self.pad_cell_radius = yes;
        self
    }

    /// Validate and build the final [`PredictorParams`].
    ///
    /// Return
    /// ------
    /// * `Ok(PredictorParams)` if parameters pass basic sanity checks.
    /// * `Err(ParamError)` on invalid sigma/noise coefficients.
    pub fn build(self) -> Result<PredictorParams, ParamError> {
        // Validate k_sigma
        if !self.k_sigma.is_finite() || self.k_sigma <= 0.0 {
            return Err(ParamError::Predictor(PredictorParamError::InvalidKSigma(
                self.k_sigma,
            )));
        }
        // Validate noise via sub-builder
        let noise = ModelNoiseBuilder { inner: self.noise }.build()?;

        Ok(PredictorParams {
            k_sigma: self.k_sigma,
            noise,
            pad_cell_radius: self.pad_cell_radius,
        })
    }

    /* --------------------------- Preset helpers --------------------------- */

    /// Preset tuned for **pairs**: `k_sigma=3.5`, conservative noise, padding on.
    pub fn preset_pairs_conservative(mut self) -> Self {
        self.k_sigma = 3.5;
        let mut nb = ModelNoiseBuilder::new();
        nb.preset_pairs_conservative();
        self.noise = nb.inner;
        self.pad_cell_radius = true;
        self
    }

    /// Preset tuned for **triplets**: `k_sigma=3.0`, tight noise, padding on.
    pub fn preset_triplets_tight(mut self) -> Self {
        self.k_sigma = 3.0;
        let mut nb = ModelNoiseBuilder::new();
        nb.preset_triplets_tight();
        self.noise = nb.inner;
        self.pad_cell_radius = true;
        self
    }
}

/* -------------------------------------------------------------------------- */
/*  Display helper                                                            */
/* -------------------------------------------------------------------------- */

impl fmt::Display for PredictorParams {
    /// Human-friendly dump for logs and diagnostics.
    ///
    /// See also
    /// --------
    /// * [`ModelNoise`] – details on the noise schedule.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PredictorParams {{ k_sigma:{:.3}, noise: {{ variance_floor:{:.3e}, drift_per_day:{:.3e}, curvature_per_day2:{:.3e} }}, pad_cell_radius:{} }}",
            self.k_sigma, self.noise.variance_floor, self.noise.drift_per_day, self.noise.curvature_per_day2, self.pad_cell_radius
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
        let p = PredictorParamsBuilder::new().build().unwrap();
        assert!((p.k_sigma - 3.0).abs() < 1e-12);
        assert!(p.pad_cell_radius);
        assert_eq!(p.noise.variance_floor, 0.0);
        assert_eq!(p.noise.drift_per_day, 0.0);
        assert_eq!(p.noise.curvature_per_day2, 0.0);
    }

    #[test]
    fn flat_setters_override() {
        let p = PredictorParamsBuilder::new()
            .k_sigma(3.7)
            .set_noise_variance_floor(1e-12)
            .set_noise_drift_per_day(2e-13)
            .set_noise_curvature_per_day2(3e-14)
            .pad_cell_radius(false)
            .build()
            .unwrap();
        assert!((p.k_sigma - 3.7).abs() < 1e-12);
        assert!(!p.pad_cell_radius);
        assert!((p.noise.variance_floor - 1e-12).abs() < 1e-25);
        assert!((p.noise.drift_per_day - 2e-13).abs() < 1e-25);
        assert!((p.noise.curvature_per_day2 - 3e-14).abs() < 1e-25);
    }

    #[test]
    fn invalid_k_sigma() {
        let err = PredictorParamsBuilder::new()
            .k_sigma(0.0)
            .build()
            .unwrap_err();
        matches!(
            err,
            ParamError::Predictor(PredictorParamError::InvalidKSigma(_))
        );
        let err = PredictorParamsBuilder::new()
            .k_sigma(f64::NAN)
            .build()
            .unwrap_err();
        matches!(
            err,
            ParamError::Predictor(PredictorParamError::InvalidKSigma(_))
        );
    }

    #[test]
    fn invalid_noise_coeffs() {
        let err = PredictorParamsBuilder::new()
            .set_noise_variance_floor(-1.0)
            .build()
            .unwrap_err();
        matches!(
            err,
            ParamError::Predictor(PredictorParamError::InvalidNoiseCoeff { .. })
        );
        let err = PredictorParamsBuilder::new()
            .set_noise_drift_per_day(f64::NAN)
            .build()
            .unwrap_err();
        matches!(
            err,
            ParamError::Predictor(PredictorParamError::InvalidNoiseCoeff { .. })
        );
    }

    #[test]
    fn presets_work() {
        let p = PredictorParamsBuilder::new()
            .preset_pairs_conservative()
            .build()
            .unwrap();
        assert!(p.k_sigma >= 3.0);
        assert!(p.pad_cell_radius);

        let q0_pairs = p.noise.variance_floor;

        let p2 = PredictorParamsBuilder::new()
            .preset_triplets_tight()
            .build()
            .unwrap();
        assert!(p2.noise.variance_floor < q0_pairs); // tighter than pairs
    }
}

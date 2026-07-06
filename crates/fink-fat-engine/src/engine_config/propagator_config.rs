//! # Propagation predictor configuration (`PredictorParams`, `ModelNoise`)
//!
//! This module defines the configuration and builder API for the **propagation
//! predictor** used during **inter-night linking**.
//!
//! At edge construction time, each seed is propagated from its internal epoch
//! (typically the mid-time of the seed) to a target epoch `t_target` on the
//! seed’s **gnomonic tangent plane**. The predicted distribution on the plane
//! is then converted into a **sky cone** used to retrieve candidate seeds in the
//! target night.
//!
//! The predictor is intentionally designed to be:
//! - **interpretable** (simple covariance propagation model),
//! - **safe** for retrieval (conservative radius extraction),
//! - **tunable** (kσ inflation, additive model noise, padding heuristics),
//! - **cheap** (no numerical orbit integration in the retrieval stage).
//!
//! -----------------------------------------------------------------------------
//! Overview of the prediction pipeline
//! -----------------------------------------------------------------------------
//!
//! For a given seed and a target epoch `t_target`:
//!
//! 1) Predict the **mean position on the tangent plane** at `t_target`.
//! 2) Predict a **plane covariance** `Σ_p(t_target)` using a simple rule:
//!
//! ```text
//! Σ_p(t) ≈ Σ_pos  +  Δt² Σ_vel  +  Q(Δt)
//! ```
//!
//! where:
//! - `Σ_pos` is the seed’s initial positional covariance on the plane,
//! - `Σ_vel` is the seed’s velocity covariance on the plane,
//! - `Δt = t_target − epoch_mid` in **days (TT)**,
//! - `Q(Δt)` is an **additive model-noise schedule** (see [`ModelNoise`]) that
//!   inflates uncertainty to cover unmodeled curvature and model mismatch.
//!
//! 3) Convert the predicted mean plane position back to the sky (inverse gnomonic).
//! 4) Extract a **single conservative cone radius** from `Σ_p(t)`:
//!
//! ```text
//! r_base = k_sigma · sqrt(λ_max(Σ_p(t)))
//! ```
//!
//! where `λ_max` is the largest eigenvalue of the 2×2 plane covariance.
//!
//! 5) Optionally inflate that radius for index-coverage robustness:
//!    - padding by one spatial cell radius (`pad_cell_radius`),
//!    - optional velocity slack term (`v_slack`) for additional safety.
//!
//! The result is a triplet `(ra, dec, radius)` describing a sky cone to query
//! the target night candidates.
//!
//! -----------------------------------------------------------------------------
//! Why `λ_max`?
//! -----------------------------------------------------------------------------
//!
//! For a 2×2 symmetric covariance, the 1σ contour is an ellipse with semi-axes
//! `sqrt(λ₁) ≥ sqrt(λ₂)`. Using `sqrt(λ_max)` yields a **circle that contains the
//! ellipse** at the same sigma level, which is safe for candidate retrieval.
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
//! -----------------------------------------------------------------------------
//! Padding for cell-based coverage
//! -----------------------------------------------------------------------------
//!
//! Many spatial indices retrieve candidates by covering the query circle with a
//! **set of discrete cells** (e.g., HEALPix neighbor unions). This is an
//! approximation: some points close to the true boundary can be missed if the
//! cell coverage is too tight. To avoid under-coverage, the predictor can add
//! one spatial cell radius:
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
//! Enable this behavior with [`PredictorParams::pad_cell_radius`].
//!
//! -----------------------------------------------------------------------------
//! Additive model noise (`ModelNoise`)
//! -----------------------------------------------------------------------------
//!
//! The plane covariance propagation `Σ_pos + Δt² Σ_vel` is a simple kinematic
//! approximation. Over multi-night gaps, real motion exhibits curvature and
//! other mismatch (especially for constant-velocity seeds built from pairs).
//!
//! To keep recall high in the candidate retrieval stage, this module adds an
//! isotropic, time-dependent **variance schedule** per axis:
//!
//! ```text
//! Q(Δt) = σ²_floor + β_drift · |Δt| + γ_curv · Δt²
//! ```
//!
//! with `Δt` in **days (TT)** and all coefficients **≥ 0**.
//!
//! The schedule is applied **per axis** (x and y) as an additive diagonal term.
//! It does not add cross-covariance, and it is intentionally minimal.
//!
//! -----------------------------------------------------------------------------
//! Serialization and units
//! -----------------------------------------------------------------------------
//!
//! The structs in this module are `serde`-deserializable. Unlike `PairConfig`
//! and `TripletConfig`, the fields here currently use raw `f64` without the
//! `engine_config::units` helpers in the snippet provided.
//!
//! This implies the following when writing YAML:
//! - values are interpreted directly in the documented canonical units,
//! - unit-suffixed strings are **not** accepted unless you wrap these fields
//!   with custom deserializers (similar to `de_time_days`, `de_angle_rad`, etc.).
//!
//! Canonical units used in this module:
//! - angles: **radians**
//! - time: **days (TT)**
//! - angular speed: **radians/day**
//! - variance: **radians²**
//! - variance rates: **radians²/day**, **radians²/day²**
//!
//! -----------------------------------------------------------------------------
//! Typical usage patterns
//! -----------------------------------------------------------------------------
//!
//! - Start with `k_sigma = 3.0` and `pad_cell_radius = true`.
//! - If inter-night recall is low, increase `k_sigma` (e.g., 3.5–4.0) and/or
//!   add a conservative [`ModelNoise`] schedule.
//! - Excessive inflation increases fan-out and runtime; rely on downstream
//!   scoring and Top-K pruning to keep the edge set bounded.
//!
//! The builders provide preset helpers tuned for common seed types:
//! - `preset_pairs_conservative()` inflates more (pairs are less predictive).
//! - `preset_triplets_tight()` inflates less (triplets model acceleration better).
//!
//! -----------------------------------------------------------------------------
//! See also
//! -----------------------------------------------------------------------------
//!
//! - `SeedNode::predict_cone`:
//!   uses these parameters to produce `(ra, dec, radius)`.
//! - `SeedNode::predict_on_plane`:
//!   returns mean & diagonal covariance on the tangent plane.

use std::fmt;

use photom::Radians;
use serde::{Deserialize, Serialize};

use crate::engine_config::units::de_angle_rad_opt;
use crate::error::PredictorParamError;

/* -------------------------------------------------------------------------- */
/*  Core types                                                                */
/* -------------------------------------------------------------------------- */

/// Additive model-noise schedule to cover unmodeled curvature and model mismatch.
///
/// Overview
/// --------
/// `ModelNoise` parameterizes a simple time-dependent **variance** term `Q(Δt)`
/// that is **added per axis** to the predicted plane covariance:
///
/// ```text
/// Σ_p(t) ≈ Σ_pos + Δt² Σ_vel + Q(Δt)
/// Q(Δt) = σ²_floor + β_drift · |Δt| + γ_curv · Δt²
/// ```
///
/// with `Δt = t_target − epoch_mid` in **days (TT)**.
///
/// This schedule is especially useful for **pair-based seeds** (constant velocity)
/// where true motion exhibits curvature over multi-night gaps.
///
/// Fields
/// ------
/// - `variance_floor` (σ²_floor):
///   - static variance floor (rad²),
///   - compensates small systematics and plane/sphere approximation even at `Δt = 0`.
/// - `drift_per_day` (β_drift):
///   - linear variance growth (rad²/day),
///   - captures slow drift-like effects (e.g., slight velocity bias).
/// - `curvature_per_day2` (γ_curv):
///   - quadratic variance growth (rad²/day²),
///   - covers curvature-like divergence increasing faster with |Δt|.
///
/// Units
/// -----
/// - `variance_floor` in **radians²**,
/// - `drift_per_day` in **radians²/day**,
/// - `curvature_per_day2` in **radians²/day²**.
///
/// Validation
/// ----------
/// All coefficients must be finite and **≥ 0**. Validation is performed by:
/// - [`PredictorParams::validate`] when embedded inside [`PredictorParams`],
/// - [`ModelNoiseBuilder::build`] for builder-based construction.
///
/// Notes
/// -----
/// - The schedule is **isotropic**: the same variance term is added to x and y.
/// - If a future propagator models anisotropy, this can be extended to a 2D form
///   (separate coefficients per axis, or full covariance injection).
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
/// `PredictorParams` governs how the predicted state uncertainty at a target
/// epoch is converted into a **sky-cone** used for candidate retrieval.
///
/// The radius construction is intentionally conservative:
///
/// ```text
/// r_base = k_sigma · sqrt(λ_max(Σ_p(t)))
/// ```
///
/// where `Σ_p(t)` is the 2×2 plane covariance at `t_target` including the
/// [`ModelNoise`] schedule.
///
/// Optional inflation steps can then be applied:
/// - add one spatial cell radius (`pad_cell_radius`) to compensate for
///   cell-based coverage approximations,
/// - add a velocity slack term (`v_slack`) if desired (implementation-dependent).
///
/// Fields
/// ------
/// - `k_sigma`:
///   - confidence multiplier (dimensionless),
///   - larger values increase recall but also increase candidate count.
/// - `noise`:
///   - additive variance schedule `Q(Δt)`; see [`ModelNoise`].
/// - `pad_cell_radius`:
///   - if `true`, add `binner.cell_radius()` to the cone radius for robustness.
/// - `time_bin_dt`:
///   - time bin size used when recovering seeds from a time binner (days, TT),
///   - used to ensure consistent epoch handling between prediction and indexing.
/// - `v_slack`:
///   - velocity slack term (rad/day) used to inflate the cone to absorb velocity
///     uncertainty (exact usage depends on the predictor implementation).
///
/// Units
/// -----
/// - `k_sigma` is dimensionless,
/// - `pad_cell_radius` is boolean,
/// - `time_bin_dt` in **days (TT)**,
/// - `v_slack` in **radians/day**,
/// - returned cone radius is in **radians**.
///
/// Validation
/// ----------
/// Validation is performed by [`PredictorParams::validate`]:
/// - `k_sigma` must be finite and strictly **> 0**,
/// - each noise coefficient must be finite and **≥ 0**.
///
/// Notes
/// -----
/// - `pad_cell_radius = true` is recommended when the spatial index performs
///   approximate cone coverage by cell unions.
/// - `v_slack` should be used sparingly: it increases the cone radius even when
///   the covariance is small. Prefer encoding velocity uncertainty in `Σ_vel`
///   and `ModelNoise` when possible.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictorParams {
    /// k-sigma inflation (e.g., 3.0).
    pub k_sigma: f64,
    /// Additive model noise Q(Δt).
    pub noise: ModelNoise,
    /// If true, add one spatial cell radius to the cone (safety padding).
    pub pad_cell_radius: bool,
    /// Time bin size used when recovering seeds from time binner. (Days, TT)
    pub time_bin_dt: f64,
    /// Velocity slack (rad/day) added to cone radius to account for velocity uncertainty.
    pub v_slack: f64,
    /// Hard upper bound on the cone radius (radians).
    ///
    /// When set, any cone radius that would exceed this value is **clamped** to it
    /// before the spatial index is queried.  Seeds whose predicted uncertainty is
    /// very large (e.g. pair seeds over a long inter-night gap) produce huge cones
    /// and therefore many false-positive candidates; capping the radius removes the
    /// worst offenders with no allocation cost.
    ///
    /// `None` (the default) means no cap: the full `k_sigma · √λ_max(Σ_p)` radius
    /// is used.
    ///
    /// In YAML, specify any supported angle unit:
    /// ```yaml
    /// max_cone_radius: "100 arcmin"
    /// max_cone_radius: "1.5 deg"
    /// ```
    /// Omitting the field is equivalent to `null` / no cap.
    ///
    /// Units: **radians** (converted at deserialisation time via `de_angle_rad_opt`).
    #[serde(
        default,
        deserialize_with = "de_angle_rad_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_cone_radius: Option<f64>,

    /// Hard upper bound on the normalised angular offset `δ / cone_radius_base`.
    ///
    /// After a spatial query returns candidates, each candidate's actual angular
    /// separation `δ = ang_sep(predicted_center, candidate)` is divided by
    /// `cone_radius_base = k_sigma · √λ_max(Σ_p(t))` (the uncapped, unpadded
    /// predicted radius).  Candidates whose ratio exceeds this threshold are
    /// **discarded before edge materialisation**.
    ///
    /// This is complementary to [`PredictorParams::max_cone_radius`]: `max_cone_radius` caps the
    /// *physical* query cone (useful when uncertainty is huge), while
    /// `max_norm_offset` caps the *normalised* ratio post-retrieval (useful when
    /// the distribution of FPs beyond a certain sigma multiple has zero TP overlap).
    ///
    /// `None` (the default) means no normalised cut is applied.
    ///
    /// In YAML, specify a plain  dimensionless number:
    /// ```yaml
    /// max_norm_offset: 25.0
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_norm_offset: Option<f64>,

    /// If true, pad the cone radius to ensure coverage of the entire time bin.
    pub pad_time_bin_radius: bool,
}

impl PredictorParams {
    /// Validate numeric ranges and physical sanity.
    ///
    /// This validation is intended to be called after deserialization from YAML/env
    /// or when constructing parameters manually without using the builders.
    ///
    /// Checks performed
    /// ----------------
    /// - `k_sigma` must be finite and strictly **> 0**.
    /// - Noise coefficients (`variance_floor`, `drift_per_day`, `curvature_per_day2`)
    ///   must be finite and **≥ 0**.
    ///
    /// Return
    /// ------
    /// - `Ok(())` if all checks pass.
    /// - `Err(PredictorParamError)` if a parameter is invalid.
    pub fn validate(&self) -> Result<(), PredictorParamError> {
        if !self.k_sigma.is_finite() || self.k_sigma <= 0.0 {
            return Err(PredictorParamError::InvalidKSigma(self.k_sigma));
        }

        let n = self.noise;
        for (name, v) in [
            ("variance_floor", n.variance_floor),
            ("drift_per_day", n.drift_per_day),
            ("curvature_per_day2", n.curvature_per_day2),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(PredictorParamError::InvalidNoiseCoeff { name, value: v });
            }
        }

        if let Some(r) = self.max_cone_radius
            && (!r.is_finite() || r <= 0.0)
        {
            return Err(PredictorParamError::InvalidMaxConeRadius(r));
        }

        if let Some(n) = self.max_norm_offset
            && (!n.is_finite() || n <= 0.0)
        {
            return Err(PredictorParamError::InvalidMaxNormOffset(n));
        }

        Ok(())
    }
}

impl Default for PredictorParams {
    /// Defaults suitable for conservative candidate retrieval.
    ///
    /// Defaults
    /// --------
    /// - `k_sigma = 3.0`
    /// - `noise = {0,0,0}` (no additive inflation)
    /// - `pad_cell_radius = true`
    /// - `time_bin_dt = 0.021` days (~30.24 minutes)
    /// - `v_slack = 0.0`
    fn default() -> Self {
        Self {
            k_sigma: 3.0,
            noise: ModelNoise::default(),
            pad_cell_radius: true,
            pad_time_bin_radius: true,
            time_bin_dt: 0.021, // 30.24 minutes in days
            v_slack: 0.0,
            max_cone_radius: None,
            max_norm_offset: None,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Sub-builder for ModelNoise                                                */
/* -------------------------------------------------------------------------- */

/// Builder for [`ModelNoise`].
///
/// This builder validates finiteness and non-negativity for all coefficients.
///
/// Notes
/// -----
/// - The builder mutates an internal `ModelNoise` instance.
/// - [`ModelNoiseBuilder::build`] performs validation and returns the final schedule.
/// - Preset helpers provide starting points for common seed types (pairs vs triplets).
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
    /// Checks performed
    /// ----------------
    /// - each coefficient must be finite and **≥ 0**.
    ///
    /// Return
    /// ------
    /// - `Ok(ModelNoise)` if coefficients are finite and non-negative.
    /// - `Err(PredictorParamError)` otherwise.
    pub fn build(self) -> Result<ModelNoise, PredictorParamError> {
        let n = self.inner;
        let coeffs = [
            ("variance_floor", n.variance_floor),
            ("drift_per_day", n.drift_per_day),
            ("curvature_per_day2", n.curvature_per_day2),
        ];
        for (name, v) in coeffs {
            if !v.is_finite() || v < 0.0 {
                return Err(PredictorParamError::InvalidNoiseCoeff { name, value: v });
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
    /// Uses ~0.15″ floor and a small quadratic term.
    /// This is a starting point and should be tuned on representative data.
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
/// 1) **Nested noise builder** (ergonomic Rust style):
///
/// ```rust, ignore
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
/// 2) **Flat setters** (binding-friendly, no closures in the public surface):
///
/// ```rust, ignore
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
    pad_time_bin_radius: bool,
    time_bin_dt: f64,
    v_slack: f64,
    max_cone_radius: Option<f64>,
    max_norm_offset: Option<f64>,
}

impl Default for PredictorParamsBuilder {
    fn default() -> Self {
        Self {
            k_sigma: 3.0,
            noise: ModelNoise::default(),
            pad_cell_radius: true,
            pad_time_bin_radius: true,
            time_bin_dt: 0.021, // 30.24 minutes in days
            v_slack: 0.0,
            max_cone_radius: None,
            max_norm_offset: None,
        }
    }
}

impl PredictorParamsBuilder {
    /// Create a new builder with sensible defaults:
    /// `k_sigma=3.0`, zero noise, `pad_cell_radius=true`, `time_bin_dt=0.021`, `v_slack=0.0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the k-sigma inflation (must be finite and > 0).
    ///
    /// Arguments
    /// ---------
    /// - `v`: sigma multiplier for the cone radius (dimensionless).
    pub fn k_sigma(mut self, v: f64) -> Self {
        self.k_sigma = v;
        self
    }

    /// Set the time bin size used when recovering seeds from a time binner (days, TT).
    ///
    /// Arguments
    /// ---------
    /// - `v`: time bin size in days.
    pub fn time_bin_dt(mut self, v: f64) -> Self {
        self.time_bin_dt = v;
        self
    }

    /// Set the velocity slack (rad/day) used to inflate the cone radius.
    ///
    /// Arguments
    /// ---------
    /// - `v`: velocity slack in rad/day.
    pub fn v_slack(mut self, v: f64) -> Self {
        self.v_slack = v;
        self
    }

    /// Mutate the noise schedule via a nested builder (ergonomic Rust style).
    ///
    /// Arguments
    /// ---------
    /// - `f`: closure that edits a temporary [`ModelNoiseBuilder`].
    ///
    /// Notes
    /// -----
    /// This method does not validate immediately; validation occurs in [`build`](Self::build).
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

    /// Enable/disable padding by the spatial cell radius.
    pub fn pad_cell_radius(mut self, yes: bool) -> Self {
        self.pad_cell_radius = yes;
        self
    }

    /// Enable/disable padding by the time bin radius.
    pub fn pad_time_bin_radius(mut self, yes: bool) -> Self {
        self.pad_time_bin_radius = yes;
        self
    }

    /// Set a hard cap on the cone radius (radians).
    ///
    /// Any computed cone radius that exceeds this value is clamped to it before
    /// the spatial index is queried.  Pass `None` to disable the cap (default).
    ///
    /// Use [`crate::astro_math::arcsec_to_rad`] or similar to convert from
    /// human-friendly units when constructing programmatically.
    pub fn max_cone_radius(mut self, r: Option<Radians>) -> Self {
        self.max_cone_radius = r;
        self
    }

    /// Set a hard cap on the normalised offset `δ / cone_radius_base`.
    ///
    /// After the spatial query, each candidate whose actual angular separation
    /// divided by the kσ-inflated predicted radius exceeds this threshold is
    /// discarded.  Pass `None` to disable (default).
    pub fn max_norm_offset(mut self, n: Option<f64>) -> Self {
        self.max_norm_offset = n;
        self
    }

    /// Validate and build the final [`PredictorParams`].
    ///
    /// Return
    /// ------
    /// - `Ok(PredictorParams)` if parameters pass basic sanity checks.
    /// - `Err(PredictorParamError)` if `k_sigma` or noise coefficients are invalid.
    pub fn build(self) -> Result<PredictorParams, PredictorParamError> {
        if !self.k_sigma.is_finite() || self.k_sigma <= 0.0 {
            return Err(PredictorParamError::InvalidKSigma(self.k_sigma));
        }
        let noise = ModelNoiseBuilder { inner: self.noise }.build()?;

        if let Some(r) = self.max_cone_radius
            && (!r.is_finite() || r <= 0.0)
        {
            return Err(PredictorParamError::InvalidMaxConeRadius(r));
        }

        if let Some(n) = self.max_norm_offset
            && (!n.is_finite() || n <= 0.0)
        {
            return Err(PredictorParamError::InvalidMaxNormOffset(n));
        }

        Ok(PredictorParams {
            k_sigma: self.k_sigma,
            noise,
            pad_cell_radius: self.pad_cell_radius,
            pad_time_bin_radius: self.pad_time_bin_radius,
            time_bin_dt: self.time_bin_dt,
            v_slack: self.v_slack,
            max_cone_radius: self.max_cone_radius,
            max_norm_offset: self.max_norm_offset,
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
        self.pad_time_bin_radius = true;
        self
    }

    /// Preset tuned for **triplets**: `k_sigma=3.0`, tight noise, padding on.
    pub fn preset_triplets_tight(mut self) -> Self {
        self.k_sigma = 3.0;
        let mut nb = ModelNoiseBuilder::new();
        nb.preset_triplets_tight();
        self.noise = nb.inner;
        self.pad_cell_radius = true;
        self.pad_time_bin_radius = true;
        self
    }
}

/* -------------------------------------------------------------------------- */
/*  Display helper                                                            */
/* -------------------------------------------------------------------------- */

impl fmt::Display for PredictorParams {
    /// Human-friendly dump for logs and diagnostics.
    ///
    /// Format
    /// ------
    /// The output prints `k_sigma`, the three noise coefficients, and
    /// `pad_cell_radius`. This is intended for logs, not for round-tripping.
    ///
    /// Notes
    /// -----
    /// This display does not currently print `time_bin_dt` or `v_slack`.
    /// If those fields are used operationally, consider extending the output
    /// to include them for easier debugging.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PredictorParams {{ k_sigma:{:.3}, noise: {{ variance_floor:{:.3e}, drift_per_day:{:.3e}, curvature_per_day2:{:.3e} }}, pad_cell_radius:{}, max_cone_radius:{}, max_norm_offset:{} }}",
            self.k_sigma,
            self.noise.variance_floor,
            self.noise.drift_per_day,
            self.noise.curvature_per_day2,
            self.pad_cell_radius,
            match self.max_cone_radius {
                Some(r) => format!("{:.2} arcmin", r * 180.0 * 60.0 / std::f64::consts::PI),
                None => "∞".to_string(),
            },
            match self.max_norm_offset {
                Some(n) => format!("{:.1}", n),
                None => "∞".to_string(),
            }
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
        matches!(err, PredictorParamError::InvalidKSigma(_));
        let err = PredictorParamsBuilder::new()
            .k_sigma(f64::NAN)
            .build()
            .unwrap_err();
        matches!(err, PredictorParamError::InvalidKSigma(_));
    }

    #[test]
    fn invalid_noise_coeffs() {
        let err = PredictorParamsBuilder::new()
            .set_noise_variance_floor(-1.0)
            .build()
            .unwrap_err();
        matches!(err, PredictorParamError::InvalidNoiseCoeff { .. });
        let err = PredictorParamsBuilder::new()
            .set_noise_drift_per_day(f64::NAN)
            .build()
            .unwrap_err();
        matches!(err, PredictorParamError::InvalidNoiseCoeff { .. });
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

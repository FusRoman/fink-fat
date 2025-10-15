use serde::{de, Deserialize, Serialize};
use std::fmt;

use crate::{
    errors::{ParamError, ScoreParamError},
    params::propagator_params::ModelNoise,
};

/* -------------------------------------------------------------------------- */
/*  Core types                                                                 */
/* -------------------------------------------------------------------------- */

/// Additive weights for the edge **cost**.
///
/// The final cost (schematic) is:
///
/// `cost = w_pos * d2_pos
///       + w_vel_dir  * (theta / theta0)
///       + w_vel_norm * (| |v_i| - |v_j| | / v0)
///       + w_flux     * z_flux
///       + w_gap      * gap(Δ)
///       + band_mismatch * w_band_mismatch`
///
/// where missing (uninformative) components contribute `0`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreWeights {
    /// Position term (Mahalanobis d² on the plane). Strongly discriminative.
    pub w_pos: f64,
    /// Direction consistency (angle between `v_i` and `v_j`, normalized by `theta0`).
    pub w_vel_dir: f64,
    /// Speed consistency (absolute speed difference, normalized by `v0`).
    pub w_vel_norm: f64,
    /// Photometry penalty (flux z-score).
    pub w_flux: f64,
    /// Gap penalty weight for Δ>1 revisits.
    pub w_gap: f64,
    /// Small additive penalty when `band(i) != band(j)`. Set to `0.0` to disable.
    pub w_band_mismatch: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            w_pos: 0.5,           // strong — geometry dominates
            w_vel_dir: 0.3,       // moderate — useful when geometry is ambiguous
            w_vel_norm: 0.3,      // moderate — speed mismatch as consistency check
            w_flux: 0.5,          // moderate — helps against confusions
            w_gap: 0.3,           // mild to moderate — encourages short gaps
            w_band_mismatch: 0.2, // small — cross-band matches are possible but less likely
        }
    }
}

/// **Hard** gates: if any is violated, the edge is rejected (`None`).
///
/// Notes
/// -----
/// * `max_theta_vel` is intentionally **private** to force using the in-place setter,
///   keeping the cached cosine `max_theta_vel_cos` in sync.
/// * `max_theta_vel_cos` is **not serialized**; during deserialization we recompute it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ScoreGates {
    /// Max Mahalanobis `d²_pos` (e.g., `~9.0` ≈ 3σ in 2D).
    pub max_d2_pos: f64,
    /// Max angle between velocity directions (**radians**).
    max_theta_vel: f64,
    /// Precomputed cos(max_theta_vel) for efficiency.
    #[serde(skip)]
    max_theta_vel_cos: f64,
    /// Max **absolute** speed difference (**radians/day**).
    pub max_speed_diff: f64,
}

// Manual `Deserialize` to refresh the cosine cache automatically.
impl<'de> Deserialize<'de> for ScoreGates {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct GatesDe {
            max_d2_pos: f64,
            max_theta_vel: f64,
            max_speed_diff: f64,
        }
        let g = GatesDe::deserialize(deserializer)?;
        Ok(ScoreGates::new(
            g.max_d2_pos,
            g.max_theta_vel,
            g.max_speed_diff,
        ))
    }
}

impl Default for ScoreGates {
    fn default() -> Self {
        Self::new(
            9.0,                   // ≈ 3σ in 2D
            10.0_f64.to_radians(), // ~10°
            f64::INFINITY,         // disabled by default
        )
    }
}

impl ScoreGates {
    #[inline]
    pub fn new(max_d2_pos: f64, max_theta_vel: f64, max_speed_diff: f64) -> Self {
        Self {
            max_d2_pos,
            max_theta_vel,
            max_theta_vel_cos: max_theta_vel.cos(),
            max_speed_diff,
        }
    }

    /// Accessor for the configured `max_theta_vel` (radians).
    #[inline]
    pub fn max_theta_vel(&self) -> f64 {
        self.max_theta_vel
    }

    /// Accessor for the precomputed `cos(max_theta_vel)`.
    #[inline]
    pub fn cos_max_theta_vel(&self) -> f64 {
        debug_assert!(
            (self.max_theta_vel_cos - self.max_theta_vel.cos()).abs() < 1e-15,
            "ScoreGates cache out of sync: use `set_max_theta_vel_in_place` for mutations"
        );
        self.max_theta_vel_cos
    }

    /// In-place setter that keeps the cached cosine in sync.
    #[inline]
    pub fn set_max_theta_vel_in_place(&mut self, v: f64) {
        self.max_theta_vel = v;
        self.max_theta_vel_cos = v.cos();
    }

    /// In-place setter for `max_d2_pos`.
    #[inline]
    pub fn set_max_d2_pos_in_place(&mut self, v: f64) {
        self.max_d2_pos = v;
    }

    /// In-place setter for `max_speed_diff`.
    #[inline]
    pub fn set_max_speed_diff_in_place(&mut self, v: f64) {
        self.max_speed_diff = v;
    }

    /// Recompute `max_theta_vel_cos` from `max_theta_vel`.
    #[inline]
    pub fn refresh(&mut self) {
        self.max_theta_vel_cos = self.max_theta_vel.cos();
    }
}

/// Scaling constants and numerical knobs for the score components.
///
/// These parameters set the **natural scales** for the normalized penalties and
/// control the finite-difference step for kinematics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreScales {
    /// Direction reference angle (**radians**) used to scale `theta`.
    pub theta0: f64,
    /// Speed reference (**radians/day**) used to scale `| |v_i| - |v_j| |`.
    pub v0: f64,
    /// Photometry σ floor (**nJy**) for robust pooling:
    /// `σ_pool = sqrt(σ_i² + σ_j² + σ_floor²)`.
    pub flux_sigma_floor: f64,
    /// Gap exponent: `gap(Δ) = (Δ - 1)^rho` if `Δ > 1`, else `0`.
    pub gap_rho: f64,
    /// Symmetric finite-difference step (**days**) for `j`’s plane velocity.
    pub vel_eps_days: f64,
}

impl Default for ScoreScales {
    fn default() -> Self {
        Self {
            theta0: 5.0_f64.to_radians(), // a few degrees
            v0: 0.005,                    // ~0.29 deg/day in rad/day
            flux_sigma_floor: 50.0,       // tune to the survey noise model
            gap_rho: 1.0,                 // linear penalty in (Δ - 1)
            vel_eps_days: 1e-3,           // ~86.4 s; small but safely > integration jitter
        }
    }
}

/// Complete configuration to score edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScoreConfig {
    /// Additive weights for the cost.
    pub weights: ScoreWeights,
    /// Hard gates for pruning.
    pub gates: ScoreGates,
    /// Scaling constants and numerical knobs.
    pub scales: ScoreScales,
}

/* -------------------------------------------------------------------------- */
/*  Sub-builders (weights/gates/scales)                                        */
/* -------------------------------------------------------------------------- */

/// Builder for [`ScoreWeights`].
///
/// Provides **fluent setters** and a `build()` that does basic validation (finite & ≥ 0).
#[derive(Clone, Debug, Default)]
pub struct ScoreWeightsBuilder {
    inner: ScoreWeights,
}

impl ScoreWeightsBuilder {
    /// Create a new builder seeded with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume the builder and return a validated `ScoreWeights`.
    pub fn build(self) -> Result<ScoreWeights, ParamError> {
        let w = self.inner;
        for (name, v) in [
            ("w_pos", w.w_pos),
            ("w_vel_dir", w.w_vel_dir),
            ("w_vel_norm", w.w_vel_norm),
            ("w_flux", w.w_flux),
            ("w_gap", w.w_gap),
            ("w_band_mismatch", w.w_band_mismatch),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(ParamError::Scoring(ScoreParamError::invalid_weight(
                    name, v,
                )));
            }
        }
        Ok(w)
    }

    // --- Setters (return &mut Self for chaining) ---
    pub fn w_pos(&mut self, v: f64) -> &mut Self {
        self.inner.w_pos = v;
        self
    }
    pub fn w_vel_dir(&mut self, v: f64) -> &mut Self {
        self.inner.w_vel_dir = v;
        self
    }
    pub fn w_vel_norm(&mut self, v: f64) -> &mut Self {
        self.inner.w_vel_norm = v;
        self
    }
    pub fn w_flux(&mut self, v: f64) -> &mut Self {
        self.inner.w_flux = v;
        self
    }
    pub fn w_gap(&mut self, v: f64) -> &mut Self {
        self.inner.w_gap = v;
        self
    }
    pub fn w_band_mismatch(&mut self, v: f64) -> &mut Self {
        self.inner.w_band_mismatch = v;
        self
    }
}

/// Builder for [`ScoreGates`].
///
/// Validates finiteness and **non-negativity** for gates, allowing `max_speed_diff = +∞`.
#[derive(Clone, Debug, Default)]
pub struct ScoreGatesBuilder {
    inner: ScoreGates,
}

impl ScoreGatesBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(mut self) -> Result<ScoreGates, ParamError> {
        let g = &mut self.inner;

        // max_d2_pos and max_theta_vel must be finite and >= 0
        for (name, v) in [
            ("max_d2_pos", g.max_d2_pos),
            ("max_theta_vel", g.max_theta_vel),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(ParamError::Scoring(ScoreParamError::invalid_gate(name, v)));
            }
        }
        // max_speed_diff can be finite >= 0 or +∞
        if !(g.max_speed_diff.is_sign_positive() || g.max_speed_diff.is_infinite()) {
            return Err(ParamError::Scoring(ScoreParamError::invalid_gate(
                "max_speed_diff",
                g.max_speed_diff,
            )));
        }

        // Ensure cosine cache is up-to-date even if fields were set directly.
        g.refresh();
        Ok(*g)
    }

    pub fn max_d2_pos(mut self, v: f64) -> Self {
        self.inner.max_d2_pos = v;
        self
    }
    pub fn max_theta_vel(mut self, v: f64) -> Self {
        self.inner.set_max_theta_vel_in_place(v);
        self
    }
    pub fn max_speed_diff(mut self, v: f64) -> Self {
        self.inner.max_speed_diff = v;
        self
    }
}

/// Builder for [`ScoreScales`].
///
/// Validates finiteness and **strict positivity** for all scales.
#[derive(Clone, Debug, Default)]
pub struct ScoreScalesBuilder {
    inner: ScoreScales,
}

impl ScoreScalesBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(self) -> Result<ScoreScales, ParamError> {
        let s = self.inner;
        for (name, v) in [
            ("theta0", s.theta0),
            ("v0", s.v0),
            ("flux_sigma_floor", s.flux_sigma_floor),
            ("gap_rho", s.gap_rho),
            ("vel_eps_days", s.vel_eps_days),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(ParamError::Scoring(ScoreParamError::invalid_scale(name, v)));
            }
        }
        Ok(s)
    }

    pub fn theta0(&mut self, v: f64) -> &mut Self {
        self.inner.theta0 = v;
        self
    }
    pub fn v0(&mut self, v: f64) -> &mut Self {
        self.inner.v0 = v;
        self
    }
    pub fn flux_sigma_floor(&mut self, v: f64) -> &mut Self {
        self.inner.flux_sigma_floor = v;
        self
    }
    pub fn gap_rho(&mut self, v: f64) -> &mut Self {
        self.inner.gap_rho = v;
        self
    }
    pub fn vel_eps_days(&mut self, v: f64) -> &mut Self {
        self.inner.vel_eps_days = v;
        self
    }
}

/* -------------------------------------------------------------------------- */
/*  Top-level builder                                                          */
/* -------------------------------------------------------------------------- */

/// Builder for [`ScoreConfig`].
///
/// Two usage styles are supported: nested closures (ergonomique Rust) ou
/// flat setters (Python-friendly).
#[derive(Clone, Debug, Default)]
pub struct ScoreConfigBuilder {
    noise: Option<ModelNoise>,
    weights: ScoreWeights,
    gates: ScoreGates,
    scales: ScoreScales,
}

impl ScoreConfigBuilder {
    /// Create a new builder seeded with component defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the model noise used for `i`’s prediction covariance at `t_j`.
    pub fn with_noise(mut self, noise: ModelNoise) -> Self {
        self.noise = Some(noise);
        self
    }

    /// Mutate `ScoreWeights` via a nested builder (ergonomic Rust style).
    pub fn with_weights<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut ScoreWeightsBuilder) -> &mut ScoreWeightsBuilder,
    {
        let mut wb = ScoreWeightsBuilder {
            inner: self.weights,
        };
        let wb = f(&mut wb);
        self.weights = wb.inner;
        self
    }

    /// Mutate `ScoreGates` via a nested builder.
    pub fn with_gates<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut ScoreGatesBuilder) -> &mut ScoreGatesBuilder,
    {
        let mut gb = ScoreGatesBuilder { inner: self.gates };
        let gb = f(&mut gb);
        self.gates = gb.inner;
        self
    }

    /// Mutate `ScoreScales` via a nested builder.
    pub fn with_scales<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut ScoreScalesBuilder) -> &mut ScoreScalesBuilder,
    {
        let mut sb = ScoreScalesBuilder { inner: self.scales };
        let sb = f(&mut sb);
        self.scales = sb.inner;
        self
    }

    /* -------------------- Flat setters (Python-friendly) -------------------- */
    // Weights
    pub fn set_w_pos(mut self, v: f64) -> Self {
        self.weights.w_pos = v;
        self
    }
    pub fn set_w_vel_dir(mut self, v: f64) -> Self {
        self.weights.w_vel_dir = v;
        self
    }
    pub fn set_w_vel_norm(mut self, v: f64) -> Self {
        self.weights.w_vel_norm = v;
        self
    }
    pub fn set_w_flux(mut self, v: f64) -> Self {
        self.weights.w_flux = v;
        self
    }
    pub fn set_w_gap(mut self, v: f64) -> Self {
        self.weights.w_gap = v;
        self
    }
    pub fn set_w_band_mismatch(mut self, v: f64) -> Self {
        self.weights.w_band_mismatch = v;
        self
    }

    // Gates
    pub fn set_max_d2_pos(mut self, v: f64) -> Self {
        self.gates.max_d2_pos = v;
        self
    }
    pub fn set_max_theta_vel(mut self, v: f64) -> Self {
        // keep cache in sync
        self.gates.set_max_theta_vel_in_place(v);
        self
    }
    pub fn set_max_speed_diff(mut self, v: f64) -> Self {
        self.gates.max_speed_diff = v;
        self
    }

    // Scales
    pub fn set_theta0(mut self, v: f64) -> Self {
        self.scales.theta0 = v;
        self
    }
    pub fn set_v0(mut self, v: f64) -> Self {
        self.scales.v0 = v;
        self
    }
    pub fn set_flux_sigma_floor(mut self, v: f64) -> Self {
        self.scales.flux_sigma_floor = v;
        self
    }
    pub fn set_gap_rho(mut self, v: f64) -> Self {
        self.scales.gap_rho = v;
        self
    }
    pub fn set_vel_eps_days(mut self, v: f64) -> Self {
        self.scales.vel_eps_days = v;
        self
    }

    /// Validate and build the final [`ScoreConfig`].
    pub fn build(self) -> Result<ScoreConfig, ParamError> {
        let weights = ScoreWeightsBuilder {
            inner: self.weights,
        }
        .build()?;
        let gates = ScoreGatesBuilder { inner: self.gates }.build()?;
        let scales = ScoreScalesBuilder { inner: self.scales }.build()?;

        Ok(ScoreConfig {
            weights,
            gates,
            scales,
        })
    }
}

/* -------------------------------------------------------------------------- */
/*  Display helpers (nice debug / logging)                                     */
/* -------------------------------------------------------------------------- */

impl fmt::Display for ScoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = &self.weights;
        let g = &self.gates;
        let s = &self.scales;
        write!(
            f,
            "ScoreConfig {{ noise: .., weights: {{ w_pos:{:.3}, w_vel_dir:{:.3}, w_vel_norm:{:.3}, w_flux:{:.3}, w_gap:{:.3}, w_band_mismatch:{:.3} }}, \
             gates: {{ max_d2_pos:{:.3}, max_theta_vel:{:.3} rad, max_speed_diff:{:?} rad/day }}, \
             scales: {{ theta0:{:.3} rad, v0:{:.6} rad/day, flux_sigma_floor:{:.3} nJy, gap_rho:{:.3}, vel_eps_days:{:.6} d }} }}",
            w.w_pos, w.w_vel_dir, w.w_vel_norm, w.w_flux, w.w_gap, w.w_band_mismatch,
            g.max_d2_pos, g.max_theta_vel(), g.max_speed_diff,
            s.theta0, s.v0, s.flux_sigma_floor, s.gap_rho, s.vel_eps_days
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
    fn build_defaults_ok() {
        let cfg = ScoreConfigBuilder::new().build().unwrap();
        // Spot-check a couple defaults:
        assert!((cfg.weights.w_pos - 0.5).abs() < 1e-12);
        assert!(cfg.gates.max_speed_diff.is_infinite());
        assert!((cfg.scales.theta0 - (5.0f64).to_radians()).abs() < 1e-12);
        // Cache coherent
        assert!((cfg.gates.cos_max_theta_vel() - cfg.gates.max_theta_vel().cos()).abs() < 1e-15);
    }

    #[test]
    fn flat_setters_override() {
        let cfg = ScoreConfigBuilder::new()
            .set_w_pos(0.9)
            .set_max_d2_pos(16.0)
            .set_theta0((3.0f64).to_radians())
            .build()
            .unwrap();
        assert!((cfg.weights.w_pos - 0.9).abs() < 1e-12);
        assert!((cfg.gates.max_d2_pos - 16.0).abs() < 1e-12);
        assert!((cfg.scales.theta0 - (3.0f64).to_radians()).abs() < 1e-12);
        // Cache coherent
        assert!((cfg.gates.cos_max_theta_vel() - cfg.gates.max_theta_vel().cos()).abs() < 1e-15);
    }

    #[test]
    fn gates_runtime_mutation_keeps_cache_in_sync() {
        let mut gates = ScoreGates::default();
        let c0 = gates.cos_max_theta_vel();
        // mutate with the in-place setter
        gates.set_max_theta_vel_in_place((30.0f64).to_radians());
        let c1 = gates.cos_max_theta_vel();
        assert!(c1 != c0);
        assert!((c1 - gates.max_theta_vel().cos()).abs() < 1e-15);
    }

    #[test]
    fn builder_mutation_keeps_cache_in_sync() {
        let gates = ScoreGatesBuilder::new()
            .max_theta_vel((25.0f64).to_radians())
            .build()
            .unwrap();
        assert!((gates.cos_max_theta_vel() - (25.0f64).to_radians().cos()).abs() < 1e-15);
    }

    #[test]
    fn invalid_negative_weight() {
        let err = ScoreConfigBuilder::new()
            .set_w_flux(-0.1)
            .build()
            .unwrap_err();
        matches!(
            err,
            ParamError::Scoring(ScoreParamError::InvalidWeight { .. })
        );
    }

    #[test]
    fn invalid_zero_scale() {
        let err = ScoreConfigBuilder::new().set_v0(0.0).build().unwrap_err();
        matches!(
            err,
            ParamError::Scoring(ScoreParamError::InvalidScale { .. })
        );
    }

    #[test]
    fn gates_allow_infinite_speed_diff() {
        let cfg = ScoreConfigBuilder::new()
            .set_max_speed_diff(f64::INFINITY)
            .build()
            .unwrap();
        assert!(cfg.gates.max_speed_diff.is_infinite());
    }

    #[test]
    fn serde_toml_roundtrip_refreshes_cache() {
        // Build a TOML string that sets an angle but NOT the cosine cache
        let toml = r#"
            max_d2_pos = 11.0
            max_theta_vel = 0.34906585   # ~20 deg
            max_speed_diff = 0.01
        "#;
        // Deserialize a bare ScoreGates (simulates a section of a larger file)
        let gates: ScoreGates = toml::from_str(toml).unwrap();
        // The implementation of Deserialize recomputes the cosine cache
        assert!((gates.max_theta_vel() - 0.34906585).abs() < 1e-6);
        assert!((gates.cos_max_theta_vel() - gates.max_theta_vel().cos()).abs() < 1e-12);
    }
}

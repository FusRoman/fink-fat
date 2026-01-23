use serde::{Deserialize, Serialize};

use crate::engine_config::{error::ScoringConfigError, propagator_config::ModelNoise};
use crate::engine_config::units::{de_angle_rad, de_ang_speed_rad_per_day, de_time_days};

/// Configuration for inter-night edge scoring.
///
/// Design goals
/// ------------
/// - Keep the YAML surface compact (few nesting levels).
/// - Group related knobs (position, velocity, photometry, gap, band).
/// - Remain deterministic and easy to validate.
///
/// Notes
/// -----
/// - Unknown keys are rejected to catch YAML typos early.
/// - Missing fields are filled from defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScoreConfig {
    /// Propagation-related knobs shared by multiple scoring terms.
    pub predict: PredictConfig,

    /// Numeric guards against pathological values.
    pub numeric: NumericConfig,

    /// Mahalanobis gating + weight on predicted position.
    pub position: PositionScore,

    /// Velocity mismatch (direction + speed): gates + weights + normalizations.
    pub velocity: VelocityScore,

    /// Photometry mismatch: weight + pooled-sigma floor.
    pub photometry: PhotometryScore,

    /// Gap penalty (Δ-1)^rho if Δ>1.
    pub gap: GapScore,

    /// Band mismatch penalty (constant additive).
    pub band: BandScore,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self {
            predict: PredictConfig::default(),
            numeric: NumericConfig::default(),
            position: PositionScore::default(),
            velocity: VelocityScore::default(),
            photometry: PhotometryScore::default(),
            gap: GapScore::default(),
            band: BandScore::default(),
        }
    }
}

impl ScoreConfig {
    pub fn validate(&self) -> Result<(), ScoringConfigError> {
        // numeric
        let mv = self.numeric.min_variance;
        if !mv.is_finite() || mv < 0.0 {
            return Err(ScoringConfigError::InvalidMinVariance { value: mv });
        }

        // position
        let d2 = self.position.max_d2;
        if !d2.is_finite() || d2 <= 0.0 {
            return Err(ScoringConfigError::InvalidMaxD2Pos { value: d2 });
        }
        if !self.position.w_pos.is_finite() {
            return Err(ScoringConfigError::NonFiniteWeight {
                field: "position.w_pos",
                value: self.position.w_pos,
            });
        }

        // velocity
        let dv = self.velocity.max_speed_diff;
        if !dv.is_finite() || dv < 0.0 {
            return Err(ScoringConfigError::InvalidMaxSpeedDiff { value: dv });
        }

        for (field, value) in [
            ("velocity.w_dir", self.velocity.w_dir),
            ("velocity.w_norm", self.velocity.w_norm),
            ("photometry.w_flux", self.photometry.w_flux),
            ("gap.w_gap", self.gap.w_gap),
            ("band.w_band_mismatch", self.band.w_band_mismatch),
        ] {
            if !value.is_finite() {
                return Err(ScoringConfigError::NonFiniteWeight { field, value });
            }
        }

        // scales/normalizations sanity
        if !self.velocity.vel_eps_days.is_finite() || self.velocity.vel_eps_days <= 0.0 {
            return Err(ScoringConfigError::InvalidVelocityScale {
                field: "velocity.vel_eps_days",
                value: self.velocity.vel_eps_days,
            });
        }
        if !self.velocity.theta0.is_finite() || self.velocity.theta0 <= 0.0 {
            return Err(ScoringConfigError::InvalidVelocityScale {
                field: "velocity.theta0",
                value: self.velocity.theta0,
            });
        }
        if !self.velocity.v0.is_finite() || self.velocity.v0 <= 0.0 {
            return Err(ScoringConfigError::InvalidVelocityScale {
                field: "velocity.v0",
                value: self.velocity.v0,
            });
        }

        if !self.photometry.flux_sigma_floor.is_finite() || self.photometry.flux_sigma_floor < 0.0 {
            return Err(ScoringConfigError::InvalidPhotometryScale {
                field: "photometry.flux_sigma_floor",
                value: self.photometry.flux_sigma_floor,
            });
        }

        if !self.gap.rho.is_finite() || self.gap.rho <= 0.0 {
            return Err(ScoringConfigError::InvalidGapScale {
                field: "gap.rho",
                value: self.gap.rho,
            });
        }

        Ok(())
    }
}

/* -------------------------------------------------------------------------- */
/*  Small sub-structs (one per theme)                                          */
/* -------------------------------------------------------------------------- */

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PredictConfig {
    pub noise: ModelNoise,
}

impl Default for PredictConfig {
    fn default() -> Self {
        Self {
            noise: ModelNoise::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NumericConfig {
    /// Treat variances smaller than this as zero (avoid exploding weights).
    pub min_variance: f64,
}

impl Default for NumericConfig {
    fn default() -> Self {
        Self { min_variance: 0.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PositionScore {
    /// Maximum allowed squared Mahalanobis distance.
    pub max_d2: f64,
    /// Weight applied to d2_pos in the final cost.
    pub w_pos: f64,
}

impl Default for PositionScore {
    fn default() -> Self {
        Self {
            max_d2: 25.0,
            w_pos: 1.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VelocityScore {
    /// Maximum allowed absolute speed mismatch (rad/day).
    #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
    pub max_speed_diff: f64,

    /// Weight for direction mismatch term.
    pub w_dir: f64,
    /// Weight for speed mismatch term.
    pub w_norm: f64,

    /// Finite difference half-step around t_j for estimating v_j (days).
    #[serde(deserialize_with = "de_time_days")]
    pub vel_eps_days: f64,
    /// Normalization angle scale for direction mismatch.
    #[serde(deserialize_with = "de_angle_rad")]
    pub theta0: f64,
    /// Normalization speed scale for speed mismatch (rad/day).
    #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
    pub v0: f64,
}

impl Default for VelocityScore {
    fn default() -> Self {
        Self {
            max_speed_diff: 0.02,
            w_dir: 1.0,
            w_norm: 1.0,
            vel_eps_days: 0.01,
            theta0: 0.1,
            v0: 0.01,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhotometryScore {
    pub w_flux: f64,
    /// Noise floor added in quadrature to pooled sigma (nJy).
    pub flux_sigma_floor: f64,
}

impl Default for PhotometryScore {
    fn default() -> Self {
        Self {
            w_flux: 0.2,
            flux_sigma_floor: 5.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GapScore {
    pub w_gap: f64,
    pub rho: f64,
}

impl Default for GapScore {
    fn default() -> Self {
        Self {
            w_gap: 0.1,
            rho: 2.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BandScore {
    pub w_band_mismatch: f64,
}

impl Default for BandScore {
    fn default() -> Self {
        Self {
            w_band_mismatch: 0.5,
        }
    }
}

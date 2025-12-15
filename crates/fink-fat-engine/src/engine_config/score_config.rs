use crate::engine_config::propagator_config::ModelNoise;

/// Top-level configuration for inter-night edge scoring.
///
/// This is a pure configuration container: no mutable state, deterministic.
#[derive(Clone, Debug)]
pub struct InterNightScoreConfig {
    pub predict: PredictConfig,

    pub position: PositionScoreConfig,
    pub velocity: VelocityScoreConfig,
    pub photometry: PhotometryScoreConfig,
    pub gap: GapScoreConfig,
    pub band: BandScoreConfig,

    /// Optional global knobs (rare): e.g., numeric safety, debug switches.
    pub numeric: NumericConfig,
}

/// Prediction / propagation configuration shared by multiple terms.
#[derive(Clone, Debug)]
pub struct PredictConfig {
    pub noise: ModelNoise,
}

/// Guards against numerical pathologies; can be expanded later.
#[derive(Clone, Debug)]
pub struct NumericConfig {
    /// Treat variances smaller than this as zero (avoid exploding weights).
    pub min_variance: f64,
}

impl Default for NumericConfig {
    fn default() -> Self {
        Self { min_variance: 0.0 }
    }
}

/* ------------------------------ POSITION ------------------------------ */

/// Position term configuration: Mahalanobis distance on i's tangent plane.
#[derive(Clone, Debug)]
pub struct PositionScoreConfig {
    pub gate: PositionGates,
    pub weight: PositionWeights,
}

#[derive(Clone, Debug)]
pub struct PositionGates {
    /// Maximum allowed squared Mahalanobis distance.
    pub max_d2_pos: f64,
}

#[derive(Clone, Debug)]
pub struct PositionWeights {
    /// Weight applied to d2_pos in the final cost.
    pub w_pos: f64,
}

/* ------------------------------ VELOCITY ------------------------------ */

/// Velocity term configuration: direction + speed mismatch, both gated.
#[derive(Clone, Debug)]
pub struct VelocityScoreConfig {
    pub gate: VelocityGates,
    pub weight: VelocityWeights,
    pub scale: VelocityScales,
}

#[derive(Clone, Debug)]
pub struct VelocityGates {
    /// Maximum allowed direction mismatch (radians).
    /// Used for gating via cosine to avoid acos in hot path.
    pub max_theta_vel: f64,
    /// Maximum allowed absolute speed mismatch (rad/day).
    pub max_speed_diff: f64,
}

impl VelocityGates {
    #[inline]
    pub fn cos_max_theta_vel(&self) -> f64 {
        self.max_theta_vel.cos()
    }
}

#[derive(Clone, Debug)]
pub struct VelocityWeights {
    /// Weight for direction mismatch term.
    pub w_vel_dir: f64,
    /// Weight for speed mismatch term.
    pub w_vel_norm: f64,
}

#[derive(Clone, Debug)]
pub struct VelocityScales {
    /// Finite difference half-step around t_j for estimating v_j (days).
    pub vel_eps_days: f64,
    /// Normalization angle scale for direction mismatch.
    pub theta0: f64,
    /// Normalization speed scale for speed mismatch (rad/day).
    pub v0: f64,
}

/* ----------------------------- PHOTOMETRY ----------------------------- */

/// Photometry term configuration: robust pooled sigma + floor.
#[derive(Clone, Debug)]
pub struct PhotometryScoreConfig {
    pub weight: PhotometryWeights,
    pub scale: PhotometryScales,
}

#[derive(Clone, Debug)]
pub struct PhotometryWeights {
    pub w_flux: f64,
}

#[derive(Clone, Debug)]
pub struct PhotometryScales {
    /// Noise floor added in quadrature to pooled sigma (nJy).
    pub flux_sigma_floor: f64,
}

/* --------------------------------- GAP -------------------------------- */

/// Gap penalty configuration: (Δ-1)^rho if Δ>1.
#[derive(Clone, Debug)]
pub struct GapScoreConfig {
    pub weight: GapWeights,
    pub scale: GapScales,
}

#[derive(Clone, Debug)]
pub struct GapWeights {
    pub w_gap: f64,
}

#[derive(Clone, Debug)]
pub struct GapScales {
    pub rho: f64,
}

/* -------------------------------- BAND -------------------------------- */

/// Band mismatch penalty (constant additive).
#[derive(Clone, Debug)]
pub struct BandScoreConfig {
    pub weight: BandWeights,
}

#[derive(Clone, Debug)]
pub struct BandWeights {
    pub w_band_mismatch: f64,
}

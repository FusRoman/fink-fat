use outfit::kepler::SolverType;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KalmanConfig {
    /// Baseline acceleration noise PSD (AU² day⁻³).
    pub q0: f64,
    /// Reference interval beyond which perturbation scaling activates (days).
    pub dt_ref: f64,
    /// Solver type for the kepler fitting in two body propagation
    pub solver_type: SolverType,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            q0: 1e-16,
            dt_ref: 1.,
            solver_type: SolverType::default(),
        }
    }
}

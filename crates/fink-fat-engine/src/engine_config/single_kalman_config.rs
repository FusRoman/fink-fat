//! # Per-hypothesis Kalman filter tuning (`KalmanConfig`)
//!
//! This module defines [`KalmanConfig`], the small set of parameters that
//! tune the process-noise model and two-body-propagation solver shared by
//! every Kalman-filter hypothesis in a bank. It is distinct from
//! [`crate::engine_config::kalman_context::KalmanContextConfig`], which
//! bundles this configuration together with the shared ephemeris/UT1 state
//! needed to actually run the filter.

use outfit::kepler::SolverType;
use serde::{Deserialize, Serialize};

use crate::engine_config::units::de_time_days;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KalmanConfig {
    /// Baseline acceleration process-noise power spectral density.
    ///
    /// Units
    /// -----
    /// - Canonical: **AU² day⁻³**.
    ///
    /// Context
    /// -------
    /// Controls how quickly the filter's positional uncertainty grows
    /// between updates absent any perturbation-scaling correction (see
    /// `dt_ref`). Larger values make the filter more tolerant of unmodeled
    /// dynamics at the cost of wider (less informative) predicted search
    /// regions; smaller values assume the two-body/Kepler model is nearly
    /// exact.
    ///
    /// Serialization
    /// -------------
    /// No `units.rs` parser is applied: AU²·day⁻³ is a compound unit outside
    /// the `<value> <unit>` grammar this crate's unit parser supports, so
    /// the value must be supplied directly in canonical units.
    pub q0: f64,

    /// Reference interval beyond which perturbation scaling activates.
    ///
    /// Units
    /// -----
    /// - Canonical: **days**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in days): `1.0`
    /// - string with units: `"24 hour"`, `"1 day"`
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_time_days`].
    #[serde(deserialize_with = "de_time_days")]
    pub dt_ref: f64,

    /// Solver used for the Kepler-equation fit during two-body propagation.
    ///
    /// See [`outfit::kepler::SolverType`] for the accepted YAML variants.
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

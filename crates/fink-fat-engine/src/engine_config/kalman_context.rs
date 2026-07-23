//! # Shared Kalman runtime context configuration (`KalmanContextConfig`)
//!
//! This module bundles the ephemeris/UT1 state shared by every Kalman
//! hypothesis in the engine (loaded once and reference-counted) together
//! with the per-hypothesis tuning knobs from
//! [`crate::engine_config::single_kalman_config::KalmanConfig`], producing a
//! ready-to-use [`KalmanContext`] via [`KalmanContextConfig::build`].
//!
//! Loading the ephemeris (`EphemState::new`) is comparatively expensive, so
//! this configuration is built once at engine startup and shared (via
//! `Arc`) across all banks/hypotheses rather than per-hypothesis.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    engine_config::{
        Validate,
        error::{FieldError, prefix_errors},
        single_kalman_config::KalmanConfig,
        validate_helpers::check_non_empty,
    },
    topocentric_kf::observer_state::EphemState,
};

/// Configuration used to build the shared [`KalmanContext`] at engine startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KalmanContextConfig {
    /// Ephemeris source identifier passed to `EphemState::new`.
    ///
    /// Format
    /// ------
    /// A backend-specific string, e.g. `"horizon:DE440"` selects the JPL
    /// Horizons DE440 planetary ephemeris. No `units.rs` parser applies —
    /// this is an opaque identifier, not a physical quantity.
    pub ephem_file_name: String,

    /// Optional UT1 (Earth-orientation) file version to load alongside the
    /// ephemeris.
    ///
    /// `None` lets `EphemState::new` fall back to its own default UT1
    /// source; `Some(version)` pins a specific version string.
    pub ut1_file_version: Option<String>,

    /// Per-hypothesis Kalman filter tuning shared by every hypothesis built
    /// from this context. See [`KalmanConfig`].
    pub config: KalmanConfig,
}

impl KalmanContextConfig {
    pub fn build(&self) -> KalmanContext {
        KalmanContext {
            ephem_state: Arc::new(EphemState::new(
                &self.ephem_file_name,
                self.ut1_file_version.as_deref(),
            )),
            config: self.config.clone(),
        }
    }
}

impl Default for KalmanContextConfig {
    fn default() -> Self {
        Self {
            ephem_file_name: "horizon:DE440".to_string(),
            ut1_file_version: None,
            config: Default::default(),
        }
    }
}

impl Validate for KalmanContextConfig {
    /// Validate internal consistency, accumulating every failure found
    /// instead of stopping at the first one.
    fn validate(&self) -> Result<(), Vec<FieldError>> {
        let mut errors = Vec::new();

        if let Some(e) = check_non_empty(
            "ephem_file_name",
            &self.ephem_file_name,
            "set ephem_file_name to a valid ephemeris source identifier, e.g. \"horizon:DE440\"",
        ) {
            errors.push(e);
        }

        if let Err(e) = self.config.validate() {
            errors.extend(prefix_errors(e, "config"));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone)]
pub struct KalmanContext {
    pub ephem_state: Arc<EphemState>,
    pub config: KalmanConfig,
}

impl KalmanContext {
    pub fn new(
        config: KalmanConfig,
        ephem_file_name: &str,
        ut1_file_version: Option<&str>,
    ) -> Self {
        Self {
            ephem_state: Arc::new(EphemState::new(ephem_file_name, ut1_file_version)),
            config,
        }
    }

    pub fn get_ephem(&self) -> &EphemState {
        &self.ephem_state
    }

    pub fn get_q0(&self) -> f64 {
        self.config.q0
    }

    pub fn get_dt_ref(&self) -> f64 {
        self.config.dt_ref
    }

    pub fn get_inflation_chi2_threshold(&self) -> f64 {
        self.config.inflation_chi2_threshold
    }

    pub fn get_max_inflation(&self) -> f64 {
        self.config.max_inflation
    }
}

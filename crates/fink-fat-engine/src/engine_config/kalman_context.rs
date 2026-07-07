use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    engine_config::single_kalman_config::KalmanConfig, topocentric_kf::observer_state::EphemState,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanContextConfig {
    pub ephem_file_name: String,
    pub ut1_file_version: Option<String>,
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
}

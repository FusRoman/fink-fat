use crate::topocentric_kf::{config::KalmanConfig, observer_state::EphemState};

pub struct KalmanContext {
    pub ephem_state: EphemState,
    pub config: KalmanConfig,
}

impl KalmanContext {
    pub fn new(
        config: KalmanConfig,
        ephem_file_name: &str,
        ut1_file_version: Option<&str>,
    ) -> Self {
        Self {
            ephem_state: EphemState::new(ephem_file_name, ut1_file_version),
            config,
        }
    }

    pub fn get_ephem(&self) -> &EphemState {
        &self.ephem_state
    }

    pub fn get_mut_ephem(&mut self) -> &mut EphemState {
        &mut self.ephem_state
    }

    pub fn get_q0(&self) -> f64 {
        self.config.q0
    }

    pub fn get_dt_ref(&self) -> f64 {
        self.config.dt_ref
    }
}

//! # Pair Parameters
//!
//! This section defines thresholds for **pair generation**: the minimal
//! seeding unit consisting of two alerts `(a, b)` close in time and brightness,
//! and consistent with a maximum **angular speed**.
//!
//! ## Overview
//! -----------
//! * **Temporal window** (`max_dt`) – how far apart in time two alerts
//!   can be (in days, TT).
//! * **Angular speed** (`max_angular_speed`) – maximum allowed on-sky angular
//!   speed, in **radians per day**. The geometric constraint becomes
//!   `ang_sep(a, b) / Δt ≤ max_angular_speed`.
//! * **Photometric similarity** (`max_flux_difference`) – restricts pairs
//!   to alerts of comparable brightness.
//! * **Time-bin constraints** (`allow_same_timebin`) – whether alerts from
//!   the same temporal bucket may form a pair.
//!
//! ## Derived cap for spatial search
//! -------------------------------
//! The bucket-neighborhood search needs a maximum separation radius. We derive
//! a conservative cap:
//!
//! `sep_cap = max_angular_speed * max_dt`
//!
//! This cap is used only to decide which spatial buckets to visit; the true
//! acceptance criterion remains the per-candidate speed check.
//!
//! ## Typical values
//! -----------------
//! For ZTF/LSST intra-night cadence, defaults are tuned to capture most
//! plausible moving-object pairs while limiting contamination:
//! * `max_dt = 0.06 d` (~86.4 min)
//! * `max_angular_speed ≈ 0.05 rad/d` (~10 arcmin over 0.06 d; order-of-magnitude)
//! * `max_flux_difference = 5.0` (~1.75 mag)
//! * `allow_same_timebin = true`
//!
//! ## Errors
//! ---------
//! Validation can fail with:
//! * [`SeedError::NonFiniteOrNegativeTime`] – invalid `max_dt`.
//! * [`SeedError::NonFiniteOrNegativeAngle`] – invalid `max_angular_speed`.
//! * [`SeedError::NonFiniteOrNegativePhotometry`] – invalid `max_flux_difference`.
//!
//! ## See also
//! -----------
//! * [`BinningParams`](crate::params::binning_params::BinningParams) – controls spatial/temporal bucket sizes.
//! * [`TripletParams`](crate::params::triplet_params::TripletParams) – extends pairs into triplets for initial orbit seeds.

use crate::{MjdTt, error::SeedError};

/// Parameters controlling **pair generation** between alerts `(a, b)`.
///
/// A "pair" is the minimal seed of a possible trajectory, defined by two
/// distinct alerts close in time, consistent with a maximum angular speed,
/// and with compatible photometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairConfig {
    /// Maximum allowed Δt between alerts a and b (days, TT).
    pub max_dt: MjdTt,

    /// Maximum allowed angular speed (radians per day).
    ///
    /// A candidate pair `(a, b)` must satisfy:
    /// `ang_sep(a, b) / (t_b - t_a) ≤ max_angular_speed`.
    pub max_angular_speed: f64,

    /// Maximum allowed photometric difference (e.g. flux units or Δmag).
    pub max_flux_difference: f32,

    /// Whether to allow pairs formed from alerts inside the same time bin.
    pub allow_same_timebin: bool,
}

impl Default for PairConfig {
    /// Provide LSST/ZTF-like defaults for intra-night cadence.
    ///
    /// Defaults
    /// --------
    /// * `max_dt` = 0.06 days (~86.4 min)
    /// * `max_angular_speed` ≈ 0.05 rad/day (order-of-magnitude)
    /// * `max_flux_difference` = 5.0
    /// * `allow_same_timebin` = true
    fn default() -> Self {
        Self {
            max_dt: 0.06,
            max_angular_speed: 5.0e-2,
            max_flux_difference: 5.0,
            allow_same_timebin: true,
        }
    }
}

impl PairConfig {
    /// Validate internal consistency and numeric ranges.
    pub fn validate(&self) -> Result<(), SeedError> {
        if !self.max_dt.is_finite() || self.max_dt < 0.0 {
            return Err(SeedError::NonFiniteOrNegativeTime("pairs.max_dt"));
        }
        if !self.max_angular_speed.is_finite() || self.max_angular_speed < 0.0 {
            // Reuse the "angle" error kind for this angular-rate parameter.
            return Err(SeedError::NonFiniteOrNegativeAngle(
                "pairs.max_angular_speed",
            ));
        }
        if !self.max_flux_difference.is_finite() || self.max_flux_difference < 0.0 {
            return Err(SeedError::NonFiniteOrNegativePhotometry(
                "pairs.max_flux_difference",
            ));
        }
        Ok(())
    }

    /// Derived conservative maximum separation (radians) used for spatial neighborhood search:
    /// `sep_cap = max_angular_speed * max_dt`.
    #[inline]
    pub fn sep_cap(&self) -> f64 {
        (self.max_angular_speed * self.max_dt).max(0.0)
    }

    /// Start building a [`PairConfig`] with chainable setters.
    pub fn builder() -> PairConfigBuilder {
        PairConfigBuilder::default()
    }
}

/// Builder for [`PairConfig`].
#[derive(Clone, Debug, Default)]
pub struct PairConfigBuilder {
    params: PairConfig,
}

impl PairConfigBuilder {
    /// Set maximum Δt between alerts (days, TT).
    pub fn max_dt(mut self, v: MjdTt) -> Self {
        self.params.max_dt = v;
        self
    }

    /// Set maximum angular speed (radians per day).
    pub fn max_angular_speed(mut self, v: f64) -> Self {
        self.params.max_angular_speed = v;
        self
    }

    /// Set maximum allowed photometric difference (dimensionless).
    pub fn max_flux_difference(mut self, v: f32) -> Self {
        self.params.max_flux_difference = v;
        self
    }

    /// Set whether to allow pairs inside the same time bin.
    pub fn allow_same_timebin(mut self, v: bool) -> Self {
        self.params.allow_same_timebin = v;
        self
    }

    /// Finalize builder and validate constraints.
    pub fn build(self) -> Result<PairConfig, SeedError> {
        let p = PairConfig {
            max_dt: self.params.max_dt,
            max_angular_speed: self.params.max_angular_speed,
            max_flux_difference: self.params.max_flux_difference,
            allow_same_timebin: self.params.allow_same_timebin,
        };
        p.validate()?;
        Ok(p)
    }
}

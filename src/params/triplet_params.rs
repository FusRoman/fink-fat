//! # Triplet Parameters
//!
//! This section defines thresholds for **triplet generation**, i.e. sequences
//! of three alerts `(a, b, c)` forming a minimal trajectory seed. Triplets
//! extend the pair-based filtering by enforcing geometric consistency,
//! predictive checks, and stricter photometric similarity.
//!
//! ## Overview
//! -----------
//! * **Temporal consistency** (`max_dt_between`) – maximum separation in time
//!   between consecutive neighbors `(a→b, b→c)`.
//! * **Spatial consistency** (`max_pair_sep`) – maximum on-sky separation
//!   for each consecutive pair `(a↔b, b↔c)`.
//! * **Predictive check** (`max_predicted_residual`) – maximum residual at `c`
//!   when extrapolating linear motion from `a→b`.
//! * **Time ordering** (`enforce_time_order`) – enforce `t(a) < t(b) < t(c)`.
//! * **Photometric similarity** (`max_flux_difference`) – require alerts to be
//!   consistent in brightness.
//!
//! ## Typical values
//! -----------------
//! Defaults are tuned for LSST/ZTF intra-night linking, ensuring good balance
//! between completeness and contamination:
//! * `max_dt_between = 0.04 d` (~57.6 min).
//! * `max_pair_sep = 0.0025 rad` (~8.6 arcmin).
//! * `max_predicted_residual = 8.0e-4 rad` (~2.75 arcmin).
//! * `enforce_time_order = true`.
//! * `max_flux_difference = 5.0` (~1.75 mag).
//!
//! ## Errors
//! ---------
//! Validation can fail with:
//! * [`ParamError::NonFiniteOrNegativeTime`] – invalid `max_dt_between`.
//! * [`ParamError::NonFiniteOrNegativeAngle`] – invalid `max_pair_sep`.
//! * [`ParamError::NonFiniteOrNegativeResidual`] – invalid `max_predicted_residual`.
//! * [`ParamError::NonFiniteOrNegativePhotometry`] – invalid `max_flux_difference`.
//! * [`ParamError::Inconsistent`] – if `max_predicted_residual > max_pair_sep`.
//!
//! ## See also
//! -----------
//! * [`BinningParams`](crate::params::binning_params::BinningParams) – spatial/temporal bucketing configuration.
//! * [`PairParams`](crate::params::pair_params::PairParams) – thresholds for initial pair generation.

/* -------------------------- Triplet Params -------------------------- */

use crate::{errors::ParamError, MjdTt, Radians};

/// Parameters controlling **triplet generation** `(a, b, c)`.
///
/// See module-level docs for overview, defaults, and error conditions.
///
/// Example
/// -------
/// ```rust
/// use fink_fat::params::triplet_params::TripletParams;
/// let params = TripletParams::builder()
///     .max_dt_between(0.03)          // ~43 min
///     .max_pair_sep(0.002)           // ~6.9 arcmin
///     .max_predicted_residual(6e-4)  // ~2.1 arcmin
///     .enforce_time_order(true)      // require strict ordering
///     .max_flux_difference(3.0)      // tighter photometry
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripletParams {
    /// Maximum Δt between consecutive neighbors `(a→b and b→c)`, in days (TT).
    pub max_dt_between: MjdTt,

    /// Maximum angular separation for neighbor pairs `(a↔b and b↔c)`, in radians.
    pub max_pair_sep: Radians,

    /// Maximum linear prediction residual at `c` when extrapolating `a→b`, in radians.
    pub max_predicted_residual: Radians,

    /// Enforce strict time ordering: require `t(a) < t(b) < t(c)`.
    pub enforce_time_order: bool,

    /// Maximum allowed photometric difference (flux units or Δmag).
    pub max_flux_difference: f32,
}

impl Default for TripletParams {
    /// Provide LSST-like defaults for intra-night triplets.
    ///
    /// Defaults
    /// --------
    /// * `max_dt_between` = 0.04 days (~57.6 min)
    /// * `max_pair_sep` = 0.0025 rad (~8.6 arcmin)
    /// * `max_predicted_residual` = 8.0e-4 rad (~2.75 arcmin)
    /// * `enforce_time_order` = true
    /// * `max_flux_difference` = 5.0
    fn default() -> Self {
        Self {
            max_dt_between: 0.04,
            max_pair_sep: 2.5e-3,
            max_predicted_residual: 8.0e-4,
            enforce_time_order: true,
            max_flux_difference: 5.0,
        }
    }
}

impl TripletParams {
    /// Validate internal consistency and numeric ranges.
    ///
    /// Return
    /// ------
    /// * `Ok(())` if valid.
    /// * `Err(ParamError)` if any parameter is invalid.
    pub fn validate(&self) -> Result<(), ParamError> {
        if !self.max_dt_between.is_finite() || self.max_dt_between < 0.0 {
            return Err(ParamError::NonFiniteOrNegativeTime(
                "triplets.max_dt_between",
            ));
        }
        if !self.max_pair_sep.is_finite() || self.max_pair_sep < 0.0 {
            return Err(ParamError::NonFiniteOrNegativeAngle(
                "triplets.max_pair_sep",
            ));
        }
        if !self.max_predicted_residual.is_finite() || self.max_predicted_residual < 0.0 {
            return Err(ParamError::NonFiniteOrNegativeResidual(
                "triplets.max_predicted_residual",
            ));
        }
        if !self.max_flux_difference.is_finite() || self.max_flux_difference < 0.0 {
            return Err(ParamError::NonFiniteOrNegativePhotometry(
                "triplets.max_flux_difference",
            ));
        }
        if self.max_predicted_residual > self.max_pair_sep {
            return Err(ParamError::Inconsistent(
                "triplets.max_predicted_residual > triplets.max_pair_sep",
            ));
        }
        Ok(())
    }

    /// Start building a [`TripletParams`] with chainable setters.
    pub fn builder() -> TripletParamsBuilder {
        TripletParamsBuilder::default()
    }
}

/// Builder for [`TripletParams`].
///
/// Provides a chainable API to construct and validate triplet thresholds.
///
/// Example
/// -------
/// ```rust
/// use fink_fat::params::triplet_params::TripletParamsBuilder;
/// let params = TripletParamsBuilder::default()
///     .max_dt_between(0.02)
///     .max_pair_sep(0.0018)
///     .max_predicted_residual(5e-4)
///     .enforce_time_order(false)
///     .max_flux_difference(4.0)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Debug, Default)]
pub struct TripletParamsBuilder {
    params: TripletParams,
}

impl TripletParamsBuilder {
    /// Set maximum Δt between consecutive neighbors `(days, TT)`.
    pub fn max_dt_between(mut self, v: MjdTt) -> Self {
        self.params.max_dt_between = v;
        self
    }

    /// Set maximum angular separation between neighbor alerts `(radians)`.
    pub fn max_pair_sep(mut self, v: Radians) -> Self {
        self.params.max_pair_sep = v;
        self
    }

    /// Set maximum allowed predicted residual at `c` `(radians)`.
    pub fn max_predicted_residual(mut self, v: Radians) -> Self {
        self.params.max_predicted_residual = v;
        self
    }

    /// Set whether to enforce strict time ordering.
    pub fn enforce_time_order(mut self, v: bool) -> Self {
        self.params.enforce_time_order = v;
        self
    }

    /// Set maximum allowed photometric difference.
    pub fn max_flux_difference(mut self, v: f32) -> Self {
        self.params.max_flux_difference = v;
        self
    }

    /// Finalize builder and validate constraints.
    ///
    /// Return
    /// ------
    /// * `Ok(TripletParams)` if valid.
    /// * `Err(ParamError)` if validation fails.
    pub fn build(self) -> Result<TripletParams, ParamError> {
        let p = TripletParams {
            max_dt_between: self.params.max_dt_between,
            max_pair_sep: self.params.max_pair_sep,
            max_predicted_residual: self.params.max_predicted_residual,
            enforce_time_order: self.params.enforce_time_order,
            max_flux_difference: self.params.max_flux_difference,
        };
        p.validate()?;
        Ok(p)
    }
}

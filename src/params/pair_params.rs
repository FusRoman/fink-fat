//! # Pair Parameters
//!
//! This section defines thresholds for **pair generation**: the minimal
//! seeding unit consisting of two alerts `(a, b)` close in time, on-sky
//! separation, and brightness. Pairs form the first step of trajectory
//! construction in the Fink-FAT pipeline.
//!
//! ## Overview
//! -----------
//! * **Temporal window** (`max_dt`) – how far apart in time two alerts
//!   can be (in days, TT).
//! * **Angular separation** (`max_sep`) – maximum on-sky distance between
//!   alerts, in radians (great-circle distance).
//! * **Photometric similarity** (`max_flux_difference`) – restricts pairs
//!   to alerts of comparable brightness.
//! * **Time-bin constraints** (`allow_same_timebin`) – whether alerts from
//!   the same temporal bucket may form a pair.
//!
//! ## Typical values
//! -----------------
//! For ZTF/LSST intra-night cadence, defaults are tuned to capture most
//! plausible moving-object pairs while limiting contamination:
//! * `max_dt = 0.06 d` (~86.4 min, typical revisit).
//! * `max_sep = 0.003 rad` (~10.3 arcmin).
//! * `max_flux_difference = 5.0` (~1.75 mag).
//! * `allow_same_timebin = true`.
//!
//! ## Errors
//! ---------
//! Validation can fail with:
//! * [`ParamError::NonFiniteOrNegativeTime`] – invalid `max_dt`.
//! * [`ParamError::NonFiniteOrNegativeAngle`] – invalid `max_sep`.
//! * [`ParamError::NonFiniteOrNegativePhotometry`] – invalid `max_flux_difference`.
//!
//! ## See also
//! -----------
//! * [`BinningParams`](crate::params::binning_params::BinningParams) – controls spatial/temporal bucket sizes.
//! * [`TripletParams`](crate::params::triplet_params::TripletParams) – extends pairs into triplets for initial orbit seeds.

use serde::{Deserialize, Serialize};

use crate::{errors::ParamError, MjdTt, Radians};

/// Parameters controlling **pair generation** between alerts `(a, b)`.
///
/// A "pair" is the minimal seed of a possible trajectory, defined by two
/// distinct alerts close in time, space, and photometry. These thresholds
/// filter out unphysical or unlikely combinations while retaining
/// plausible asteroid candidates.
///
/// Overview
/// --------
/// * **Temporal proximity** (`max_dt`) – restricts how far apart in time
///   the two alerts can be.
/// * **Angular separation** (`max_sep`) – maximum allowed on-sky distance
///   between the alerts (great-circle separation).
/// * **Photometric consistency** (`max_flux_difference`) – restricts pairs
///   to alerts of comparable brightness (in flux units or Δmag).
/// * **Bin constraints** (`allow_same_timebin`) – whether two alerts from
///   the same temporal bucket may still form a valid pair.
///
/// Units
/// -----
/// * `max_dt`: **days (TT)**, must be ≥ 0 and finite.
/// * `max_sep`: **radians**, must be ≥ 0 and finite.
/// * `max_flux_difference`: arbitrary flux scale (dimensionless), must be ≥ 0.
/// * `allow_same_timebin`: boolean flag.
///
/// Defaults
/// --------
/// Tuned for **intra-night LSST/ZTF cadence**:
/// * `max_dt` = 0.06 days (~86.4 min, typical LSST revisit window).
/// * `max_sep` = 0.003 rad (~10.3 arcmin, compatible with fast movers).
/// * `max_flux_difference` = 5.0 (~1.75 mag, generous to account for noise).
/// * `allow_same_timebin` = true.
///
/// Errors
/// ------
/// * [`ParamError::NonFiniteOrNegativeTime`] if `max_dt` ≤ 0 or not finite.
/// * [`ParamError::NonFiniteOrNegativeAngle`] if `max_sep` ≤ 0 or not finite.
/// * [`ParamError::NonFiniteOrNegativePhotometry`] if `max_flux_difference` ≤ 0
///   or not finite.
///
/// Example
/// -------
/// ```rust
/// use fink_fat::params::pair_params::PairParams;
///
/// let params = PairParams::builder()
///     .max_dt(0.05)                // 72 min
///     .max_sep(0.0025)             // ~8.6 arcmin
///     .max_flux_difference(3.0)    // stricter photometry
///     .allow_same_timebin(false)   // enforce cross-bin only
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairParams {
    /// Maximum allowed Δt between alerts a and b (days, TT).
    pub max_dt: MjdTt,

    /// Maximum allowed angular separation between alerts (radians).
    pub max_sep: Radians,

    /// Maximum allowed photometric difference (e.g. flux units or Δmag).
    pub max_flux_difference: f32,

    /// Whether to allow pairs formed from alerts inside the same time bin.
    pub allow_same_timebin: bool,
}

impl Default for PairParams {
    /// Provide LSST/ZTF-like defaults for intra-night cadence.
    ///
    /// Defaults
    /// --------
    /// * `max_dt` = 0.06 days (~86.4 min)
    /// * `max_sep` = 0.003 rad (~10.3 arcmin)
    /// * `max_flux_difference` = 5.0
    /// * `allow_same_timebin` = true
    fn default() -> Self {
        Self {
            max_dt: 0.06,
            max_sep: 3.0e-3,
            max_flux_difference: 5.0,
            allow_same_timebin: true,
        }
    }
}

impl PairParams {
    /// Validate internal consistency and numeric ranges.
    ///
    /// Return
    /// ------
    /// * `Ok(())` if valid.
    /// * `Err(ParamError)` if any value is invalid or non-finite.
    pub fn validate(&self) -> Result<(), ParamError> {
        if !self.max_dt.is_finite() || self.max_dt < 0.0 {
            return Err(ParamError::NonFiniteOrNegativeTime("pairs.max_dt"));
        }
        if !self.max_sep.is_finite() || self.max_sep < 0.0 {
            return Err(ParamError::NonFiniteOrNegativeAngle("pairs.max_sep"));
        }
        if !self.max_flux_difference.is_finite() || self.max_flux_difference < 0.0 {
            return Err(ParamError::NonFiniteOrNegativePhotometry(
                "pairs.max_flux_difference",
            ));
        }
        Ok(())
    }

    /// Start building a [`PairParams`] with chainable setters.
    ///
    /// See also
    /// --------
    /// * [`PairParamsBuilder`] for builder pattern usage.
    pub fn builder() -> PairParamsBuilder {
        PairParamsBuilder::default()
    }
}

/// Builder for [`PairParams`].
///
/// Provides a chainable API to construct and validate pair thresholds.
///
/// Example
/// -------
/// ```rust
/// use fink_fat::params::pair_params::PairParamsBuilder;
/// let params = PairParamsBuilder::default()
///     .max_dt(0.04)
///     .max_sep(0.002)
///     .max_flux_difference(2.5)
///     .allow_same_timebin(true)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Debug, Default)]
pub struct PairParamsBuilder {
    params: PairParams,
}

impl PairParamsBuilder {
    /// Set maximum Δt between alerts (days, TT).
    pub fn max_dt(mut self, v: MjdTt) -> Self {
        self.params.max_dt = v;
        self
    }

    /// Set maximum angular separation between alerts (radians).
    pub fn max_sep(mut self, v: Radians) -> Self {
        self.params.max_sep = v;
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
    ///
    /// Return
    /// ------
    /// * `Ok(PairParams)` if all values are valid.
    /// * `Err(ParamError)` if validation fails.
    pub fn build(self) -> Result<PairParams, ParamError> {
        let p = PairParams {
            max_dt: self.params.max_dt,
            max_sep: self.params.max_sep,
            max_flux_difference: self.params.max_flux_difference,
            allow_same_timebin: self.params.allow_same_timebin,
        };
        p.validate()?;
        Ok(p)
    }
}

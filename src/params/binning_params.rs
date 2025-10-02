//! # Binning Parameters
//!
//! This module defines configuration structures for **spatio-temporal bucketing**
//! of astronomical alerts. Bucketing is a preprocessing step to efficiently
//! generate candidate pairs and triplets from massive alert streams (e.g. LSST).
//!
//! ## Overview
//! -----------
//! Bucketing groups alerts into:
//!
//! * **Spatial bins**: using HEALPix tessellation (nested scheme).
//!   - Controlled by `healpix_depth` (NSIDE = 2^depth).
//!   - Higher depth → finer resolution, fewer alerts per bin.
//!   - Lower depth → coarser resolution, more alerts per bin.
//!
//! * **Temporal bins**: using fixed-width slices of MJD (TT).
//!   - Controlled by `time_bin_width_days`.
//!   - Smaller width → stricter contemporaneity.
//!   - Larger width → more inclusive, but may increase contamination.
//!
//! ## Default choices
//! ------------------
//! The defaults are tuned for **LSST-like surveys** (10+ million alerts/night):
//!
//! * `healpix_depth = 10` → NSIDE=1024 (~3–4 arcmin resolution).
//! * `time_bin_width_days = 0.02` → ~28.8 minutes.
//!
//! These values strike a balance between performance (memory and lookup speed)
//! and completeness (capturing most plausible intra-night seeds).
//!
//! ## Components
//! -------------
//! * [`BinningParams`] – immutable validated configuration.
//! * [`BinningParamsBuilder`] – ergonomic builder API.
//!
//! ## Errors
//! ---------
//! Validation can fail with:
//! * [`ParamError::Inconsistent`] if `healpix_depth > 29`.
//! * [`ParamError::NonFiniteOrNegativeTime`] if `time_bin_width_days <= 0`
//!   or not finite.
//!
//! ## See also
//! -----------
//! * [`PairParams`](crate::params::pair_params::PairParams) – thresholds for pair generation.
//! * [`TripletParams`](crate::params::triplet_params::TripletParams) – thresholds for triplet generation.
//!
//! Together, these modules form the **seeding configuration layer** for the
//! Fink-FAT pipeline.

use crate::{errors::ParamError, MjdTt};

/// Global parameters controlling **spatio-temporal bucketing** of alerts.
///
/// This configuration defines how alerts are grouped into *spatial bins*
/// (via HEALPix indexing) and *temporal bins* (via fixed-width MJD segments).
/// Bucketing is the foundation of pair/triplet generation and must be chosen
/// carefully to balance **efficiency** (fast neighbor lookups) and
/// **completeness** (no missed candidate seeds).
///
/// Overview
/// --------
/// * Spatial bucketing uses HEALPix NSIDE = 2^`healpix_depth`, in nested scheme.
///   - Higher depth → smaller pixels (finer spatial resolution).
///   - Lower depth → larger pixels (coarser spatial resolution).
/// * Temporal bucketing slices MJD(TT) into fixed-width bins.
///   - Smaller width → stricter contemporaneity, fewer false matches.
///   - Larger width → looser grouping, higher completeness.
///
/// Typical values for LSST-like alert streams (10M/night):
/// - `healpix_depth = 10` → NSIDE=1024, pixel size ~3–4 arcmin.
/// - `time_bin_width_days = 0.02` → ~28.8 minutes.
///
/// See also
/// --------
/// * [`PairParams`](crate::params::pair_params::PairParams) – pair generation thresholds within bins.
/// * [`TripletParams`](crate::params::triplet_params::TripletParams) – triplet generation thresholds.
///
/// Errors
/// ------
/// * [`ParamError::Inconsistent`] – if `healpix_depth > 29`.
/// * [`ParamError::NonFiniteOrNegativeTime`] – if time bin width is ≤ 0 or NaN/Inf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinningParams {
    /// Spatial resolution parameter for HEALPix (nested).
    ///
    /// Defines NSIDE = 2^depth.
    /// - Valid range: **0..=29** (limited by 64-bit implementations).
    /// - Example: depth=10 → NSIDE=1024 (~3–4 arcmin pixels).
    pub healpix_depth: u8,

    /// Temporal resolution of buckets (days, TT).
    ///
    /// Each alert is assigned to a bin of width `time_bin_width_days`
    /// starting from a reference epoch.
    /// - Units: **days** (TT timescale).
    /// - Must be strictly positive and finite.
    /// - Example: 0.02 days ≈ 28.8 minutes.
    pub time_bin_width_days: MjdTt,
}

impl Default for BinningParams {
    /// Provide LSST-like defaults balancing spatial and temporal resolution.
    ///
    /// Defaults
    /// --------
    /// * `healpix_depth` = 10 → NSIDE=1024 (~3–4 arcmin)
    /// * `time_bin_width_days` = 0.02 days (~28.8 min)
    fn default() -> Self {
        Self {
            healpix_depth: 10,
            time_bin_width_days: 0.02,
        }
    }
}

impl BinningParams {
    /// Validate ranges and consistency of the parameters.
    ///
    /// Return
    /// ------
    /// * `Ok(())` if all constraints are satisfied.
    /// * `Err(ParamError::Inconsistent)` if `healpix_depth > 29`.
    /// * `Err(ParamError::NonFiniteOrNegativeTime)` if `time_bin_width_days <= 0`
    ///   or not finite.
    pub fn validate(&self) -> Result<(), ParamError> {
        // Common CDS HEALPix implementations on u64 support depth up to 29.
        if self.healpix_depth > 29 {
            return Err(ParamError::Inconsistent(
                "binning.healpix_depth out of range [0, 29]",
            ));
        }
        if !self.time_bin_width_days.is_finite() || self.time_bin_width_days <= 0.0 {
            return Err(ParamError::NonFiniteOrNegativeTime(
                "binning.time_bin_width_days (must be > 0)",
            ));
        }
        Ok(())
    }

    /// Start building a [`BinningParams`] using the builder pattern.
    ///
    /// Notes
    /// -----
    /// The builder allows a more ergonomic construction of parameters with
    /// chained setters, enforcing validation at the end.
    ///
    /// Example
    /// -------
    /// ```rust
    /// let params = BinningParams::builder()
    ///     .healpix_depth(12)
    ///     .time_bin_width_days(0.01)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> BinningParamsBuilder {
        BinningParamsBuilder::default()
    }
}

/// Builder for [`BinningParams`].
///
/// Provides a convenient chainable API for setting parameters before
/// final validation.
///
/// Example
/// -------
/// ```rust
/// let params = BinningParamsBuilder::default()
///     .healpix_depth(8)
///     .time_bin_width_days(0.05)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Debug, Default)]
pub struct BinningParamsBuilder {
    params: BinningParams,
}

impl BinningParamsBuilder {
    /// Set the HEALPix depth (NSIDE = 2^depth).
    ///
    /// Arguments
    /// ---------
    /// * `v` – depth, 0..=29, higher means finer resolution.
    pub fn healpix_depth(mut self, v: u8) -> Self {
        self.params.healpix_depth = v;
        self
    }

    /// Set the temporal bin width (days, TT).
    ///
    /// Arguments
    /// ---------
    /// * `v` – bin width in **days** (TT).
    ///   Must be finite and strictly positive.
    pub fn time_bin_width_days(mut self, v: MjdTt) -> Self {
        self.params.time_bin_width_days = v;
        self
    }

    /// Finalize the builder, applying validation.
    ///
    /// Return
    /// ------
    /// * `Ok(BinningParams)` if valid.
    /// * `Err(ParamError)` if validation fails.
    pub fn build(self) -> Result<BinningParams, ParamError> {
        let p = BinningParams {
            healpix_depth: self.params.healpix_depth,
            time_bin_width_days: self.params.time_bin_width_days,
        };
        p.validate()?;
        Ok(p)
    }
}

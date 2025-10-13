//! # Global Parameters for Fink-FAT
//!
//! This module declares the **central configuration** for the Fink-FAT project.
//! It exposes a single top-level parameter bag, [`FinkFatParams`], which
//! aggregates seeding/linking thresholds (binning, pairs, triplets) and
//! **global runtime options** (e.g., progress bars). The intent is to make the
//! configuration **explicit, validated, and extensible** as new components are
//! added to the project (e.g., ML cross-night linking, orbit refinement).
//!
//! ## Overview
//! -----------
//! * **Top-level container:** [`FinkFatParams`] – canonical config passed across
//!   Fink-FAT components.
//! * **Submodules:**
//!   - [`binning_params`] → spatial+temporal bucketing (`BinningParams`).
//!   - [`pair_params`] → pair generation thresholds (`PairParams`).
//!   - [`triplet_params`] → triplet generation thresholds (`TripletParams`).
//!   - [`params_binding`] → Python bindings (thin wrapper; see notes below).
//!
//! ## Defaults
//! -----------
//! Tuned for LSST-like cadence (10M+ alerts/night):
//! * **Binning:** `healpix_depth = 10`, `time_bin_width_days = 0.02` (~28.8 min).
//! * **Pairs:** `max_dt = 0.06 d`, `max_sep = 0.003 rad`, `max_flux_difference = 5.0`.
//! * **Triplets:** `max_dt_between = 0.04 d`, `max_pair_sep = 0.0025 rad`,
//!   `max_predicted_residual = 8e-4 rad`.
//! * **Global:** `show_progress = false`.
//!
//! ## Usage
//! --------
//! Two complementary construction styles are supported.
//!
//! **1) Flat setters** – tweak individual fields quickly:
//! ```rust
//! use fink_fat::params::FinkFatParams;
//!
//! let params = FinkFatParams::builder()
//!     .healpix_depth(12)
//!     .time_bin_width_days(0.03)
//!     .pair_max_dt(0.05)
//!     .triplet_max_predicted_residual(6.0e-4)
//!     .show_progress(true)
//!     .build()
//!     .unwrap();
//! ```
//!
//! **2) Nested setters** – scope related options together with sub-builders:
//! ```rust
//! use fink_fat::params::FinkFatParams;
//!
//! let params = FinkFatParams::builder()
//!     .binning(|b| b.healpix_depth(11).time_bin_width_days(0.02))
//!     .pairs(|p| p.max_sep(0.002).max_flux_difference(3.0))
//!     .triplets(|t| t.enforce_time_order(true).max_dt_between(0.03))
//!     .build()
//!     .unwrap();
//! ```
//!
//! After construction, always validate in contexts where parameters may come
//! from user input or external files:
//! ```rust, ignore
//! params.validate().unwrap();
//! ```
//!
//! ## Python bindings
//! ------------------
//! The Python API **exposes a flat builder** for stability and simplicity
//! (generic closures are not bound). The Rust builder supports both flat and
//! nested styles; Python uses the flat subset only.
//!
//! ## Validation
//! -------------
//! [`FinkFatParams::validate`] calls the validators of each sub-group. Typical
//! failures include out-of-range HEALPix depth, non-finite/negative time or
//! angles, and inconsistent triplet tolerances (e.g., predicted residual >
//! pair separation).
//!
//! ## Extensibility
//! ----------------
//! As the project grows, new global sections (e.g., *linking across nights*,
//! *orbit fitting*, *quality filters*, *I/O policies*) should be added as fields
//! of [`FinkFatParams`] and wired into the builder, while preserving defaults
//! and validation behavior.
//!
//! ## See also
//! -----------
//! * [`binning_params::BinningParams`]
//! * [`pair_params::PairParams`]
//! * [`triplet_params::TripletParams`]

pub mod binning_params;
pub mod engine_params;
pub mod pair_params;
pub mod params_binding;
pub mod propagator_params;
pub mod scoring_params;
pub mod triplet_params;

use camino::Utf8Path;
use serde::{Deserialize, Serialize};

use crate::{
    errors::ParamError,
    params::{
        binning_params::{BinningParams, BinningParamsBuilder},
        engine_params::{InterNightLinkConfig, InterNightLinkConfigBuilder},
        pair_params::{PairParams, PairParamsBuilder},
        triplet_params::{TripletParams, TripletParamsBuilder},
    },
    MjdTt, Radians,
};

/* --------------------------- FinkFatParams --------------------------- */

fn default_schema_version() -> String {
    "1".to_string()
}

/// On-disk representation of Fink-FAT config files.
///
/// Keeps a `schema_version` to allow future migrations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinkFatConfigFile {
    /// Schema version (string to allow semver if needed later).
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    /// Actual parameters.
    #[serde(flatten)]
    pub params: FinkFatParams,
}

/// Top-level parameter bag for the **entire Fink-FAT project**.
///
/// This struct aggregates the sub-parameter groups (binning/pairs/triplets)
/// and **global runtime options** into a single, validated configuration
/// object. It is the canonical entry point for configuring seeding/linking and
/// other modules that will be added over time.
///
/// Example
/// -------
/// ```rust
/// use fink_fat::params::FinkFatParams;
///
/// let params = FinkFatParams::builder()
///     .healpix_depth(12)
///     .pair_max_dt(0.03)
///     .triplet_max_pair_sep(0.002)
///     .show_progress(true)
///     .build()
///     .unwrap();
///
/// params.validate().unwrap();
/// ```
///
/// See also
/// --------
/// * [`PairParams`] – controls pair generation.
/// * [`TripletParams`] – controls triplet generation.
/// * [`BinningParams`] – controls HEALPix depth & time bin width.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinkFatParams {
    /// Global spatial/temporal bucketing parameters.
    #[serde(default)]
    pub binning: BinningParams,
    /// Pair-generation parameters.
    #[serde(default)]
    pub pairs: PairParams,
    /// Triplet-generation parameters.
    #[serde(default)]
    pub triplets: TripletParams,
    /// Inter-night linking configuration (prediction, scoring, graph limits).
    ///
    /// See [`engine_params::InterNightLinkConfig`].
    #[serde(default)]
    pub link: InterNightLinkConfig,
    /// Whether to show progress bars during seeding/linking.
    #[serde(default)]
    pub show_progress: bool,
}

impl Default for FinkFatParams {
    /// Defaults to the defaults of each sub-parameter group and disables progress bars.
    ///
    /// Defaults
    /// --------
    /// * `binning = BinningParams::default()`
    /// * `pairs   = PairParams::default()`
    /// * `triplets= TripletParams::default()`
    /// * `show_progress = false`
    fn default() -> Self {
        Self {
            binning: BinningParams::default(),
            pairs: PairParams::default(),
            triplets: TripletParams::default(),
            link: InterNightLinkConfig::default(),
            show_progress: false,
        }
    }
}

impl FinkFatParams {
    pub fn from_toml_str(s: &str) -> Result<Self, ParamError> {
        let file: FinkFatConfigFile = toml::from_str(s)?;
        let params = file.params;
        params.validate()?;
        Ok(params)
    }

    pub fn to_toml_string_pretty(&self) -> Result<String, ParamError> {
        let file = FinkFatConfigFile {
            schema_version: default_schema_version(),
            params: self.clone(),
        };
        let s = toml::to_string_pretty(&file)?;
        Ok(s)
    }

    pub fn load_toml_file(path: &Utf8Path) -> Result<Self, ParamError> {
        let txt = std::fs::read_to_string(path)?;
        Self::from_toml_str(&txt)
    }

    pub fn save_toml_file(&self, path: &Utf8Path) -> Result<(), ParamError> {
        let s = self.to_toml_string_pretty()?;
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Create a builder for [`FinkFatParams`].
    ///
    /// Example
    /// -------
    /// ```rust
    /// use fink_fat::params::FinkFatParams;
    /// let cfg = FinkFatParams::builder()
    ///     .healpix_depth(11)
    ///     .time_bin_width_days(0.02)
    ///     .pair_max_sep(0.0025)
    ///     .triplet_enforce_time_order(true)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> FinkFatParamsBuilder {
        FinkFatParamsBuilder::default()
    }

    /// Validate the entire parameter set.
    ///
    /// Return
    /// ------
    /// * `Ok(())` if all sub-groups validate successfully.
    /// * `Err(ParamError)` on the first validation failure encountered.
    pub fn validate(&self) -> Result<(), ParamError> {
        self.binning.validate()?;
        self.pairs.validate()?;
        self.triplets.validate()?;
        Ok(())
    }

    /* -------- Optional convenience mutators on the top-level -------- */

    /// Replace the `binning` sub-parameters.
    ///
    /// Notes
    /// -----
    /// This method does **not** call `validate()`. Call it explicitly if needed.
    pub fn with_binning(mut self, binning: BinningParams) -> Self {
        self.binning = binning;
        self
    }

    /// Replace the `pairs` sub-parameters.
    ///
    /// Notes
    /// -----
    /// This method does **not** call `validate()`. Call it explicitly if needed.
    pub fn with_pairs(mut self, pairs: PairParams) -> Self {
        self.pairs = pairs;
        self
    }

    /// Replace the `triplets` sub-parameters.
    ///
    /// Notes
    /// -----
    /// This method does **not** call `validate()`. Call it explicitly if needed.
    pub fn with_triplets(mut self, triplets: TripletParams) -> Self {
        self.triplets = triplets;
        self
    }

    /// Replace the inter-night linking configuration.
    ///
    /// Notes
    /// -----
    /// This method does **not** call `validate()`. Call it explicitly if needed.
    pub fn with_link(mut self, link: InterNightLinkConfig) -> Self {
        self.link = link;
        self
    }
}

/// Builder for [`FinkFatParams`].
///
/// Two complementary styles are supported:
///
/// 1) **Flat setters** (e.g., `pair_max_dt(..)`, `healpix_depth(..)`) to tweak a field quickly.
/// 2) **Nested setters** with closures on sub-builders:
///    `.pairs(|p| p.max_dt(0.03))`, `.binning(|b| b.healpix_depth(12))`.
///
/// Example
/// -------
/// ```rust
/// use fink_fat::params::FinkFatParams;
///
/// // Mixed flat + nested
/// let cfg = FinkFatParams::builder()
///     .show_progress(true)
///     .pair_max_flux_difference(3.0)
///     .binning(|b| b.healpix_depth(11).time_bin_width_days(0.02))
///     .triplets(|t| t.max_predicted_residual(6e-4))
///     .build()
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct FinkFatParamsBuilder {
    binning: BinningParamsBuilder,
    pairs: PairParamsBuilder,
    triplets: TripletParamsBuilder,
    link: InterNightLinkConfigBuilder,
    show_progress: bool,
}

impl Default for FinkFatParamsBuilder {
    /// Initialize sub-builders with their defaults and `show_progress = false`.
    fn default() -> Self {
        Self {
            binning: BinningParamsBuilder::default(),
            pairs: PairParamsBuilder::default(),
            triplets: TripletParamsBuilder::default(),
            link: InterNightLinkConfigBuilder::new(),
            show_progress: false,
        }
    }
}

impl FinkFatParamsBuilder {
    /// Set whether to show progress bars during seeding/linking.
    pub fn show_progress(mut self, v: bool) -> Self {
        self.show_progress = v;
        self
    }

    /* ----------------------- Flat setters (binning) ---------------------- */

    /// Set `binning.healpix_depth` (NSIDE = 2^depth, valid 0..=29).
    pub fn healpix_depth(mut self, v: u8) -> Self {
        self.binning = self.binning.healpix_depth(v);
        self
    }

    /// Set `binning.time_bin_width_days` (days, TT; must be > 0).
    pub fn time_bin_width_days(mut self, v: MjdTt) -> Self {
        self.binning = self.binning.time_bin_width_days(v);
        self
    }

    /* ----------------------- Flat setters (pairs) ----------------------- */

    /// Set `pairs.max_dt` (days, TT; ≥ 0 and finite).
    pub fn pair_max_dt(mut self, v: MjdTt) -> Self {
        self.pairs = self.pairs.max_dt(v);
        self
    }

    /// Set `pairs.max_sep` (radians; ≥ 0 and finite).
    pub fn pair_max_sep(mut self, v: Radians) -> Self {
        self.pairs = self.pairs.max_sep(v);
        self
    }

    /// Set `pairs.max_flux_difference` (dimensionless; ≥ 0 and finite).
    pub fn pair_max_flux_difference(mut self, v: f32) -> Self {
        self.pairs = self.pairs.max_flux_difference(v);
        self
    }

    /// Set `pairs.allow_same_timebin`.
    pub fn pair_allow_same_timebin(mut self, v: bool) -> Self {
        self.pairs = self.pairs.allow_same_timebin(v);
        self
    }

    /* --------------------- Flat setters (triplets) --------------------- */

    /// Set `triplets.max_dt_between` (days, TT; ≥ 0 and finite).
    pub fn triplet_max_dt_between(mut self, v: MjdTt) -> Self {
        self.triplets = self.triplets.max_dt_between(v);
        self
    }

    /// Set `triplets.max_pair_sep` (radians; ≥ 0 and finite).
    pub fn triplet_max_pair_sep(mut self, v: Radians) -> Self {
        self.triplets = self.triplets.max_pair_sep(v);
        self
    }

    /// Set `triplets.max_predicted_residual` (radians; ≥ 0 and finite).
    pub fn triplet_max_predicted_residual(mut self, v: Radians) -> Self {
        self.triplets = self.triplets.max_predicted_residual(v);
        self
    }

    /// Set `triplets.enforce_time_order`.
    pub fn triplet_enforce_time_order(mut self, v: bool) -> Self {
        self.triplets = self.triplets.enforce_time_order(v);
        self
    }

    /// Set `triplets.max_flux_difference` (dimensionless; ≥ 0 and finite).
    pub fn triplet_max_flux_difference(mut self, v: f32) -> Self {
        self.triplets = self.triplets.max_flux_difference(v);
        self
    }

    /* ---------------------- Flat setters (linking) ---------------------- */
    // Predictor passthrough
    pub fn link_k_sigma(mut self, v: f64) -> Self {
        self.link = self.link.set_k_sigma(v);
        self
    }
    pub fn link_noise_q0(mut self, v: f64) -> Self {
        self.link = self.link.set_noise_q0(v);
        self
    }
    pub fn link_noise_q1(mut self, v: f64) -> Self {
        self.link = self.link.set_noise_q1(v);
        self
    }
    pub fn link_noise_q2(mut self, v: f64) -> Self {
        self.link = self.link.set_noise_q2(v);
        self
    }
    pub fn link_pad_cell_radius(mut self, yes: bool) -> Self {
        self.link = self.link.set_pad_cell_radius(yes);
        self
    }

    // Scoring (common knobs)
    pub fn link_w_pos(mut self, v: f64) -> Self {
        self.link = self.link.set_w_pos(v);
        self
    }
    pub fn link_w_vel_dir(mut self, v: f64) -> Self {
        self.link = self.link.set_w_vel_dir(v);
        self
    }
    pub fn link_w_vel_norm(mut self, v: f64) -> Self {
        self.link = self.link.set_w_vel_norm(v);
        self
    }
    pub fn link_w_flux(mut self, v: f64) -> Self {
        self.link = self.link.set_w_flux(v);
        self
    }
    pub fn link_w_gap(mut self, v: f64) -> Self {
        self.link = self.link.set_w_gap(v);
        self
    }
    pub fn link_w_band_mismatch(mut self, v: f64) -> Self {
        self.link = self.link.set_w_band_mismatch(v);
        self
    }

    pub fn link_max_d2_pos(mut self, v: f64) -> Self {
        self.link = self.link.set_max_d2_pos(v);
        self
    }
    pub fn link_max_theta_vel(mut self, v: f64) -> Self {
        self.link = self.link.set_max_theta_vel(v);
        self
    }
    pub fn link_max_speed_diff(mut self, v: f64) -> Self {
        self.link = self.link.set_max_speed_diff(v);
        self
    }

    pub fn link_theta0(mut self, v: f64) -> Self {
        self.link = self.link.set_theta0(v);
        self
    }
    pub fn link_v0(mut self, v: f64) -> Self {
        self.link = self.link.set_v0(v);
        self
    }
    pub fn link_flux_sigma_floor(mut self, v: f64) -> Self {
        self.link = self.link.set_flux_sigma_floor(v);
        self
    }
    pub fn link_gap_rho(mut self, v: f64) -> Self {
        self.link = self.link.set_gap_rho(v);
        self
    }
    pub fn link_vel_eps_days(mut self, v: f64) -> Self {
        self.link = self.link.set_vel_eps_days(v);
        self
    }

    // Limits
    pub fn link_top_k_per_left(mut self, v: usize) -> Self {
        self.link = self.link.set_top_k_per_left(v);
        self
    }
    pub fn link_max_total_edges(mut self, v: Option<usize>) -> Self {
        self.link = self.link.set_max_total_edges(v);
        self
    }
    pub fn link_clear_max_total_edges(mut self) -> Self {
        self.link = self.link.clear_max_total_edges();
        self
    }
    pub fn link_max_cost(mut self, v: Option<f64>) -> Self {
        self.link = self.link.set_max_cost(v);
        self
    }
    pub fn link_clear_max_cost(mut self) -> Self {
        self.link = self.link.clear_max_cost();
        self
    }

    pub fn link_max_speed_rad_per_day(mut self, v: Option<f64>) -> Self {
        self.link = self.link.set_max_speed_rad_per_day(v);
        self
    }

    /* -------------------- Nested setters via closure ------------------- */

    /// Configure `binning` via its builder.
    ///
    /// Notes
    /// -----
    /// This closure-based API is **not** exposed in Python bindings.
    pub fn binning<F>(mut self, f: F) -> Self
    where
        F: FnOnce(BinningParamsBuilder) -> BinningParamsBuilder,
    {
        self.binning = f(self.binning);
        self
    }

    /// Configure `pairs` via its builder.
    ///
    /// Notes
    /// -----
    /// This closure-based API is **not** exposed in Python bindings.
    pub fn pairs<F>(mut self, f: F) -> Self
    where
        F: FnOnce(PairParamsBuilder) -> PairParamsBuilder,
    {
        self.pairs = f(self.pairs);
        self
    }

    /// Configure `triplets` via its builder.
    ///
    /// Notes
    /// -----
    /// This closure-based API is **not** exposed in Python bindings.
    pub fn triplets<F>(mut self, f: F) -> Self
    where
        F: FnOnce(TripletParamsBuilder) -> TripletParamsBuilder,
    {
        self.triplets = f(self.triplets);
        self
    }

    /// Configure inter-night linking via its builder.
    ///
    /// Notes
    /// -----
    /// This closure-based API is **not** exposed in Python bindings.
    pub fn linking<F>(mut self, f: F) -> Self
    where
        F: FnOnce(InterNightLinkConfigBuilder) -> InterNightLinkConfigBuilder,
    {
        self.link = f(self.link);
        self
    }

    /// Build the full [`FinkFatParams`] (apply defaults for unspecified fields) and validate.
    ///
    /// Return
    /// ------
    /// * `Ok(FinkFatParams)` if construction and validation succeed.
    /// * `Err(ParamError)` if any sub-group validation fails.
    pub fn build(self) -> Result<FinkFatParams, ParamError> {
        let binning = self.binning.build()?;
        let pairs = self.pairs.build()?;
        let triplets = self.triplets.build()?;
        let link = self.link.build()?;

        let cfg = FinkFatParams {
            binning,
            pairs,
            triplets,
            link,
            show_progress: self.show_progress,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

impl TryFrom<FinkFatParamsBuilder> for FinkFatParams {
    type Error = ParamError;

    /// Try to convert a builder into a validated [`FinkFatParams`].
    ///
    /// Equivalent to calling [`FinkFatParamsBuilder::build`].
    fn try_from(b: FinkFatParamsBuilder) -> Result<Self, Self::Error> {
        b.build()
    }
}

/* -------------------------------- Tests ------------------------------ */

#[cfg(test)]
mod params_tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        FinkFatParams::default().validate().unwrap();
    }

    #[test]
    fn flat_overrides_work() {
        let p = FinkFatParams::builder()
            .healpix_depth(12)
            .time_bin_width_days(0.03)
            .pair_max_dt(0.02)
            .triplet_max_pair_sep(1.2e-3)
            .link_k_sigma(3.7)
            .link_top_k_per_left(12)
            .build()
            .unwrap();

        assert_eq!(p.binning.healpix_depth, 12);
        assert_eq!(p.binning.time_bin_width_days, 0.03);
        assert_eq!(p.pairs.max_dt, 0.02);
        assert_eq!(p.triplets.max_pair_sep, 1.2e-3);
        assert!((p.link.predict.k_sigma - 3.7).abs() < 1e-12);
        assert_eq!(p.link.limits.top_k_per_left, 12);
    }

    #[test]
    fn nested_overrides_work() {
        let p = FinkFatParams::builder()
            .binning(|b| b.healpix_depth(11).time_bin_width_days(0.015))
            .pairs(|b| b.max_sep(2.1e-3).allow_same_timebin(false))
            .triplets(|b| b.max_predicted_residual(5.0e-4))
            .linking(|l| {
                l.set_k_sigma(3.5)
                    .set_max_d2_pos(12.0)
                    .set_top_k_per_left(10)
            })
            .build()
            .unwrap();

        assert_eq!(p.binning.healpix_depth, 11);
        assert_eq!(p.binning.time_bin_width_days, 0.015);
        assert_eq!(p.pairs.max_sep, 2.1e-3);
        assert!(!p.pairs.allow_same_timebin);
        assert_eq!(p.triplets.max_predicted_residual, 5.0e-4);
        assert!((p.link.predict.k_sigma - 3.5).abs() < 1e-12);
        assert!((p.link.scoring.gates.max_d2_pos - 12.0).abs() < 1e-12);
        assert_eq!(p.link.limits.top_k_per_left, 10);
    }

    #[test]
    fn catches_binning_range_and_triplet_inconsistency() {
        // healpix_depth out of range
        let err = FinkFatParams::builder()
            .healpix_depth(40)
            .build()
            .unwrap_err();
        assert!(matches!(err, ParamError::Inconsistent(_)));

        // residual > pair_sep
        let err = FinkFatParams::builder()
            .triplet_max_pair_sep(1.0e-3)
            .triplet_max_predicted_residual(2.0e-3)
            .build()
            .unwrap_err();
        assert!(matches!(err, ParamError::Inconsistent(_)));
    }

    #[test]
    fn catches_invalid_linking_limits() {
        // top_k_per_left = 0 must fail via EngineParamError mapped to ParamError.
        let err = FinkFatParams::builder()
            .link_top_k_per_left(0)
            .build()
            .unwrap_err();

        assert!(matches!(err, ParamError::Engine(_)));
    }
}

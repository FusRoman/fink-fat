//! # Triplet generation configuration (`TripletConfig`)
//!
//! This module defines the configuration parameters used to generate **triplets**
//! of alerts `(a, b, c)` within a short intra-night time window.
//!
//! A **triplet** is a minimal trajectory seed built from three detections.
//! Compared to pairs, triplets enforce stronger constraints:
//! - local geometric consistency on consecutive neighbors,
//! - a simple predictive check (linear extrapolation from `a→b` to `c`),
//! - stricter time ordering (optionally enforced),
//! - photometric consistency across the three alerts.
//!
//! Triplet generation is typically the last “cheap” stage before building
//! higher-level seed objects (e.g. tangent-plane kinematic models) and before
//! inter-night graph construction.
//!
//! -----------------------------------------------------------------------------
//! Conceptual model
//! -----------------------------------------------------------------------------
//!
//! Triplets are defined as ordered or orderable sequences of three alerts.
//! Depending on `enforce_time_order`, the generator behaves in one of two modes:
//!
//! - **Strict time order enabled** (`enforce_time_order = true`):
//!   - the triplet must satisfy `t(a) < t(b) < t(c)`,
//!   - the two neighbor gaps `Δt_ab = t(b) - t(a)` and `Δt_bc = t(c) - t(b)`
//!     are individually bounded by `max_dt_between`.
//!
//! - **Strict time order disabled** (`enforce_time_order = false`):
//!   - the generator may consider permutations (implementation-defined),
//!   - `max_dt_between` still applies to consecutive neighbors in the chosen
//!     ordering.
//!
//! For a chosen ordering `(a, b, c)`:
//!
//! - Temporal constraints:
//!   - `Δt_ab ≤ max_dt_between` and `Δt_bc ≤ max_dt_between`.
//! - Neighbor spatial constraints:
//!   - `ang_sep(a, b) ≤ max_pair_sep` and `ang_sep(b, c) ≤ max_pair_sep`.
//! - Predictive (linear motion) consistency:
//!   - extrapolate position at `t(c)` using the apparent motion from `(a→b)`,
//!   - compute the angular residual at `c`,
//!   - require `residual ≤ max_predicted_residual`.
//! - Photometric consistency:
//!   - apply the implementation-specific brightness similarity check using
//!     `max_flux_difference`.
//!
//! -----------------------------------------------------------------------------
//! Predictive residual: what is being bounded?
//! -----------------------------------------------------------------------------
//!
//! The predictive check is intentionally simple and local.
//!
//! Let `p(t)` be the on-sky position (in some projection or direct spherical
//! computations depending on the implementation). Using two alerts `(a, b)`,
//! define a constant apparent velocity model and predict the position at `t(c)`.
//! The predicted residual is the angular distance between:
//!
//! - the predicted position at `t(c)` (from `a→b`), and
//! - the observed position of `c`.
//!
//! The threshold `max_predicted_residual` controls how much curvature / noise
//! is tolerated within the short intra-night window.
//!
//! Consistency constraint
//! ----------------------
//! This module enforces a basic consistency requirement:
//!
//! ```text
//! max_predicted_residual ≤ max_pair_sep
//! ```
//!
//! Rationale: the predicted residual is a deviation relative to a neighbor-scale
//! geometry; allowing a residual larger than the maximum neighbor separation is
//! usually not meaningful and tends to admit degenerate / noisy triplets.
//!
//! -----------------------------------------------------------------------------
//! Serialization and units
//! -----------------------------------------------------------------------------
//!
//! This configuration is `serde`-deserializable (YAML / TOML / JSON) and uses
//! project-level unit parsers from `engine_config::units` to allow both canonical
//! numeric values and human-friendly unit strings.
//!
//! ## Numeric vs string quantities
//!
//! For fields that use `deserialize_with = ...` from `engine_config::units`:
//!
//! - A **numeric YAML scalar** is accepted and is interpreted as already being
//!   in **canonical engine units**.
//! - A **string YAML scalar** is accepted and is parsed as a `<value><unit>`
//!   quantity (best-effort whitespace handling).
//!
//! Concretely for [`TripletConfig`]:
//!
//! - `max_dt_between` uses [`de_time_days`] and is stored as **days (TT)**.
//!   - Numeric form: `0.04` means `0.04 days`.
//!   - String form: `"57.6 min"`, `"0.96 h"`, `"240 sec"`, `"0.04 day"`.
//! - `max_pair_sep` uses [`de_angle_rad`] and is stored as **radians**.
//!   - Numeric form: `2.5e-3` means `0.0025 rad`.
//!   - String form: `"8.6 arcmin"`, `"0.143 deg"`, `"515 arcsec"`, `"2.5e-3rad"`.
//! - `max_predicted_residual` uses [`de_angle_rad`] and is stored as **radians**.
//!   - Numeric form: `8.0e-4` means `0.0008 rad`.
//!   - String form: `"2.75 arcmin"`, `"165 arcsec"`, `"8e-4rad"`.
//!
//! As for pairs, `max_flux_difference` is intentionally unit-agnostic at the
//! configuration layer: its meaning must match the triplet photometry kernel.
//!
//! ## Supported units (as implemented in `units.rs`)
//!
//! Time units (case-insensitive):
//! - `day`, `days`, `d`, `jour`, `jours`
//! - `hour`, `hours`, `h`, `heure`, `heures`
//! - `min`, `minute`, `minutes`
//! - `sec`, `second`, `seconds`, `s`, `seconde`, `secondes`
//!
//! Angle units (case-insensitive):
//! - `rad`, `radian`, `radians`
//! - `deg`, `degree`, `degrees`, plus basic French aliases `degre`, `degres`
//! - `arcmin`, `arcminute`, `arcminutes`
//! - `arcsec`, `arcsecond`, `arcseconds`
//!
//! The parser also supports a best-effort “no whitespace” form for `<value><unit>`
//! (e.g. `"1e-3deg"`, `"2.5e-3rad"`). For time quantities, the same applies
//! (e.g. `"90min"`).
//!
//! -----------------------------------------------------------------------------
//! Typical defaults and tuning guidelines
//! -----------------------------------------------------------------------------
//!
//! The provided [`Default`] values are intended for LSST/ZTF-like intra-night
//! linking (order-of-magnitude, conservative):
//!
//! - `max_dt_between = 0.04 d` (~57.6 min)
//! - `max_pair_sep = 2.5e-3 rad` (~8.6 arcmin)
//! - `max_predicted_residual = 8.0e-4 rad` (~2.75 arcmin)
//! - `enforce_time_order = true`
//! - `max_flux_difference = 5.0`
//!
//! Tuning suggestions:
//!
//! - If triplets are too numerous (high contamination / runtime):
//!   - decrease `max_dt_between`,
//!   - decrease `max_pair_sep`,
//!   - decrease `max_predicted_residual` (often the most selective constraint),
//!   - tighten `max_flux_difference`.
//!
//! - If recall is too low:
//!   - increase `max_dt_between` slightly,
//!   - increase `max_pair_sep` if the cadence can produce larger separations,
//!   - increase `max_predicted_residual` if motion is not well approximated as
//!     linear at the time scale considered (or if positional noise is larger).
//!
//! - If your input stream can contain mis-ordered timestamps or duplicated
//!   exposures, consider setting `enforce_time_order = false` only if the
//!   generator implementation explicitly handles such cases safely.
//!
//! -----------------------------------------------------------------------------
//! Configuration examples (YAML)
//! -----------------------------------------------------------------------------
//!
//! Canonical-unit numeric form:
//!
//! ```yaml
//! triplets:
//!   max_dt_between: 0.04            # days (TT)
//!   max_pair_sep: 2.5e-3            # rad
//!   max_predicted_residual: 8.0e-4  # rad
//!   enforce_time_order: true
//!   max_flux_difference: 5.0        # must match triplet photometry metric
//! ```
//!
//! Human-friendly string form:
//!
//! ```yaml
//! triplets:
//!   max_dt_between: "57.6 min"
//!   max_pair_sep: "8.6 arcmin"
//!   max_predicted_residual: "2.75 arcmin"
//!   enforce_time_order: true
//!   max_flux_difference: 5.0
//! ```
//!
//! -----------------------------------------------------------------------------
//! Errors and validation
//! -----------------------------------------------------------------------------
//!
//! [`TripletConfig::validate`] enforces basic numeric validity:
//! - finite values,
//! - non-negative thresholds,
//! - and consistency (`max_predicted_residual ≤ max_pair_sep`).
//!
//! Validation can fail with:
//! - [`SeedError::NonFiniteOrNegativeTime`] for `triplets.max_dt_between`,
//! - [`SeedError::NonFiniteOrNegativeAngle`] for `triplets.max_pair_sep`,
//! - [`SeedError::NonFiniteOrNegativeResidual`] for `triplets.max_predicted_residual`,
//! - [`SeedError::NonFiniteOrNegativePhotometry`] for `triplets.max_flux_difference`,
//! - [`SeedError::Inconsistent`] if `max_predicted_residual > max_pair_sep`.
//!
//! Unit parsing failures (string quantities) are surfaced by serde as
//! deserialization errors with an explicit message from `engine_config::units`
//! (unsupported unit, malformed quantity string, etc.).
//!
//! -----------------------------------------------------------------------------
//! See also
//! -----------------------------------------------------------------------------
//!
//! - `BinningParams` (spatial/temporal bucket sizing) impacts neighborhood search
//!   and the distribution of candidate triplets.
//! - `PairConfig` provides the earlier-stage constraints used to generate
//!   candidate pairs that triplets may build upon.

use photom::{MJDTT, Radians};
use serde::{Deserialize, Serialize};

use crate::engine_config::units::{de_ang_speed_rad_per_day, de_angle_rad, de_time_days};
use crate::error::SeedError;

/// Parameters controlling **triplet generation** `(a, b, c)`.
///
/// A triplet is a local, intra-night seed that should be consistent with
/// a short-timescale, near-linear motion model.
///
/// Behavior
/// --------
/// For a chosen ordering `(a, b, c)`, a candidate triplet is accepted only if:
///
/// - `Δt_ab ≤ max_dt_between` and `Δt_bc ≤ max_dt_between`,
/// - `ang_sep(a, b) ≤ max_pair_sep` and `ang_sep(b, c) ≤ max_pair_sep`,
/// - the linear prediction residual at `c` from `a→b` is
///   `≤ max_predicted_residual`,
/// - the photometry similarity constraints pass using `max_flux_difference`,
/// - and optionally, strict time ordering is enforced (`enforce_time_order`).
///
/// Notes
/// -----
/// This struct is `serde`-deserializable to support robust configuration loading.
/// If you want strict YAML typo checking (recommended), apply:
/// `#[serde(default, deny_unknown_fields)]` as done on other config structs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TripletConfig {
    /// Maximum allowed time separation between consecutive neighbors.
    ///
    /// Units
    /// -----
    /// - Canonical: **days (TT)**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in days): `0.04`
    /// - string with units: `"57.6 min"`, `"0.96 h"`, `"240 sec"`, `"0.04 day"`
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_time_days`].
    #[serde(deserialize_with = "de_time_days")]
    pub max_dt_between: MJDTT,

    /// Maximum angular separation allowed for consecutive neighbors.
    ///
    /// Units
    /// -----
    /// - Canonical: **radians**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in rad): `2.5e-3`
    /// - string with units: `"8.6 arcmin"`, `"515 arcsec"`, `"0.143 deg"`, `"2.5e-3rad"`
    ///
    /// Acceptance test
    /// ---------------
    /// For the chosen ordering `(a, b, c)`:
    /// - `ang_sep(a, b) ≤ max_pair_sep`
    /// - `ang_sep(b, c) ≤ max_pair_sep`
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_angle_rad`].
    #[serde(deserialize_with = "de_angle_rad")]
    pub max_pair_sep: Radians,

    /// Maximum residual at `c` when extrapolating a linear motion model from `a→b`.
    ///
    /// Units
    /// -----
    /// - Canonical: **radians**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in rad): `8.0e-4`
    /// - string with units: `"2.75 arcmin"`, `"165 arcsec"`, `"8e-4rad"`
    ///
    /// Interpretation
    /// --------------
    /// The generator builds a constant apparent motion model from `(a, b)`
    /// and predicts where the object should be at time `t(c)`. This threshold
    /// bounds the angular distance between prediction and observation.
    ///
    /// Consistency
    /// -----------
    /// This parameter must satisfy:
    /// `max_predicted_residual ≤ max_pair_sep`.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_angle_rad`].
    #[serde(deserialize_with = "de_angle_rad")]
    pub max_predicted_residual: Radians,

    /// Enforce strict time ordering: require `t(a) < t(b) < t(c)`.
    ///
    /// When enabled, this prevents degenerate triplets built from mis-ordered
    /// timestamps or identical exposures (depending on upstream data).
    pub enforce_time_order: bool,

    /// Maximum allowed photometric difference across the triplet.
    ///
    /// Important
    /// ---------
    /// This value is **dimensionless at the configuration layer**. Its meaning
    /// depends on the photometry check used by the triplet generator
    /// (flux space, magnitude space, normalized residual, etc.).
    pub max_mag_difference: f64,

    /// Maximum allowed on-sky angular speed.
    ///
    /// Units
    /// -----
    /// - Canonical: **radians per day**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in rad/day): `5.0e-2`
    /// - string with explicit rate: `"35 arcmin/day"`, `"2 arcsec / hour"`, `"0.05 rad/day"`
    ///
    /// Acceptance test
    /// ---------------
    /// A candidate pair `(a, b)` must satisfy:
    ///
    /// ```text
    /// ang_sep(a, b) / (t_b - t_a) ≤ max_angular_speed
    /// ```
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_ang_speed_rad_per_day`].
    #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
    pub max_angular_speed: f64,
}

impl Default for TripletConfig {
    /// Provide LSST-like defaults for intra-night triplets.
    fn default() -> Self {
        Self {
            max_dt_between: 0.04,
            max_pair_sep: 2.5e-3,
            max_predicted_residual: 8.0e-4,
            enforce_time_order: true,
            max_mag_difference: 5.0,
            max_angular_speed: 5.0e-2,
        }
    }
}

impl TripletConfig {
    /// Validate internal consistency and numeric ranges.
    pub fn validate(&self) -> Result<(), SeedError> {
        if !self.max_dt_between.is_finite() || self.max_dt_between < 0.0 {
            return Err(SeedError::NonFiniteOrNegativeTime(
                "triplets.max_dt_between",
            ));
        }
        if !self.max_pair_sep.is_finite() || self.max_pair_sep < 0.0 {
            return Err(SeedError::NonFiniteOrNegativeAngle("triplets.max_pair_sep"));
        }
        if !self.max_predicted_residual.is_finite() || self.max_predicted_residual < 0.0 {
            return Err(SeedError::NonFiniteOrNegativeResidual(
                "triplets.max_predicted_residual",
            ));
        }
        if !self.max_mag_difference.is_finite() || self.max_mag_difference < 0.0 {
            return Err(SeedError::NonFiniteOrNegativePhotometry(
                "triplets.max_mag_difference",
            ));
        }
        if self.max_predicted_residual > self.max_pair_sep {
            return Err(SeedError::Inconsistent(
                "triplets.max_predicted_residual > triplets.max_pair_sep",
            ));
        }
        Ok(())
    }

    /// Start building a [`TripletConfig`] with chainable setters.
    pub fn builder() -> TripletConfigBuilder {
        TripletConfigBuilder::default()
    }
}

/// Builder for [`TripletConfig`].
#[derive(Clone, Debug, Default)]
pub struct TripletConfigBuilder {
    params: TripletConfig,
}

impl TripletConfigBuilder {
    /// Set maximum `Δt` between consecutive neighbors (days, TT).
    pub fn max_dt_between(mut self, v: MJDTT) -> Self {
        self.params.max_dt_between = v;
        self
    }

    /// Set maximum angular separation between neighbor alerts (radians).
    pub fn max_pair_sep(mut self, v: Radians) -> Self {
        self.params.max_pair_sep = v;
        self
    }

    /// Set maximum allowed prediction residual at `c` (radians).
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
    pub fn max_mag_difference(mut self, v: f64) -> Self {
        self.params.max_mag_difference = v;
        self
    }

    /// Set maximum allowed on-sky angular speed.
    pub fn max_angular_speed(mut self, v: f64) -> Self {
        self.params.max_angular_speed = v;
        self
    }

    /// Finalize builder and validate constraints.
    pub fn build(self) -> Result<TripletConfig, SeedError> {
        let p = TripletConfig {
            max_dt_between: self.params.max_dt_between,
            max_pair_sep: self.params.max_pair_sep,
            max_predicted_residual: self.params.max_predicted_residual,
            enforce_time_order: self.params.enforce_time_order,
            max_mag_difference: self.params.max_mag_difference,
            max_angular_speed: self.params.max_angular_speed,
        };
        p.validate()?;
        Ok(p)
    }
}

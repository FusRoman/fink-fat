//! # Pair generation configuration (`PairConfig`)
//!
//! This module defines the configuration parameters used to generate **pairs**
//! of alerts `(a, b)` within a single night (or within a short time window).
//!
//! A **pair** is the smallest seeding unit in the engine: two detections that
//! are close in time, consistent with a maximum on-sky **angular speed**, and
//! compatible in photometry.
//!
//! Pair generation is designed as a **cheap, conservative pre-filter**:
//! it should keep most plausible moving-object candidates while limiting the
//! combinatorial explosion that would occur if every alert could pair with many
//! others.
//!
//! -----------------------------------------------------------------------------
//! Conceptual model
//! -----------------------------------------------------------------------------
//!
//! For two alerts `a` (anchor) and `b` (candidate), with `t_b > t_a`:
//!
//! - Temporal constraint:
//!   - `Δt = t_b - t_a` must be within `max_dt`.
//! - Kinematic constraint (angular speed):
//!   - Let `Δθ = ang_sep(a, b)` be the on-sky angular separation (radians).
//!   - The candidate must satisfy:
//!     `Δθ / Δt ≤ max_angular_speed`.
//! - Photometric constraint:
//!   - The candidate must satisfy a configurable brightness / flux similarity
//!     test controlled by `max_flux_difference`.
//!
//! The exact photometry metric depends on the pairing implementation (flux space,
//! magnitude space, normalized flux difference, etc.). This configuration
//! parameter is intentionally **unit-agnostic** at the config level: it must
//! match what the pairing kernel expects.
//!
//! -----------------------------------------------------------------------------
//! Spatial bucket search and `sep_cap()`
//! -----------------------------------------------------------------------------
//!
//! The pair builder typically queries a spatio-temporal index (bucket grid) to
//! avoid scanning all alerts. To do that, it needs an **upper bound** on the
//! maximum possible separation between two alerts that could pass the kinematic
//! constraint.
//!
//! A conservative bound is:
//!
//! ```text
//! sep_cap = max_angular_speed * max_dt
//! ```
//!
//! This bound is used only for **index traversal** (which buckets to visit).
//! The actual acceptance test remains the per-candidate inequality
//! `Δθ / Δt ≤ max_angular_speed`.
//!
//! -----------------------------------------------------------------------------
//! Serialization and units
//! -----------------------------------------------------------------------------
//!
//! This configuration is `serde`-deserializable (YAML / TOML / JSON) and uses
//! project-level unit parsers to make configuration files human-friendly.
//!
//! ## Numeric vs string quantities
//!
//! For fields that use `deserialize_with = ...` from `engine_config::units`:
//!
//! - A **numeric YAML scalar** is accepted and is interpreted as already being
//!   in **canonical engine units**.
//! - A **string YAML scalar** is accepted and is parsed as a `<value><unit>`
//!   quantity.
//!
//! Concretely for [`PairConfig`]:
//!
//! - `max_dt` uses [`de_time_days`] and is stored as **days (TT)**.
//!   - Numeric form: `0.06` means `0.06 days`.
//!   - String form: `"86.4 min"`, `"1.44 h"`, `"30 sec"`, `"0.06 day"` are accepted.
//! - `max_angular_speed` uses [`de_ang_speed_rad_per_day`] and is stored as
//!   **radians per day**.
//!   - Numeric form: `5.0e-2` means `0.05 rad/day`.
//!   - String form: must be written as an explicit rate `<angle>/<time>`:
//!     `"35 arcmin/day"`, `"2 arcsec / hour"`, `"0.05 rad/day"`.
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
//! (e.g. `"0.05rad"`, `"1e-3deg"`).
//!
//! For rates, the denominator can optionally be prefixed with `"per"`
//! (e.g. `"35 arcmin/per day"`).
//!
//! -----------------------------------------------------------------------------
//! Typical defaults and tuning guidelines
//! -----------------------------------------------------------------------------
//!
//! The provided [`Default`] values are intended to be compatible with LSST/ZTF-like
//! intra-night cadence (order-of-magnitude, conservative):
//!
//! - `max_dt = 0.06 d` (~86.4 min)
//! - `max_angular_speed = 0.05 rad/d`
//! - `max_flux_difference = 5.0`
//! - `allow_same_timebin = true`
//!
//! Tuning suggestions:
//!
//! - If too many pairs are produced (high contamination):
//!   - decrease `max_dt`,
//!   - decrease `max_angular_speed`,
//!   - tighten `max_flux_difference`,
//!   - or set `allow_same_timebin = false` (if your time-binning is coarse and
//!     produces many same-bin candidates).
//! - If too few pairs are produced (low recall):
//!   - increase `max_dt` slightly,
//!   - increase `max_angular_speed` if you target fast movers,
//!   - loosen `max_flux_difference` if photometry is noisy.
//!
//! -----------------------------------------------------------------------------
//! Configuration examples (YAML)
//! -----------------------------------------------------------------------------
//!
//! Canonical-unit numeric form:
//!
//! ```yaml
//! pairs:
//!   max_dt: 0.06                 # days (TT)
//!   max_angular_speed: 5.0e-2    # rad/day
//!   max_flux_difference: 5.0     # must match pairing kernel's photometry metric
//!   allow_same_timebin: true
//! ```
//!
//! Human-friendly string form:
//!
//! ```yaml
//! pairs:
//!   max_dt: "86.4 min"
//!   max_angular_speed: "35 arcmin/day"
//!   max_flux_difference: 5.0
//!   allow_same_timebin: true
//! ```
//!
//! Unknown keys are rejected (`deny_unknown_fields`) to catch YAML typos early.
//!
//! -----------------------------------------------------------------------------
//! Errors and validation
//! -----------------------------------------------------------------------------
//!
//! [`PairConfig::validate`] enforces basic numeric validity:
//! - finite values,
//! - non-negative constraints.
//!
//! Validation can fail with:
//! - [`SeedError::NonFiniteOrNegativeTime`] for `pairs.max_dt`,
//! - [`SeedError::NonFiniteOrNegativeAngle`] for `pairs.max_angular_speed`,
//! - [`SeedError::NonFiniteOrNegativePhotometry`] for `pairs.max_flux_difference`.
//!
//! Unit parsing failures (string quantities) are surfaced by serde as
//! deserialization errors with an explicit message from `engine_config::units`
//! (unsupported unit, malformed `<angle>/<time>` rate, etc.).
//!
//! -----------------------------------------------------------------------------
//! See also
//! -----------------------------------------------------------------------------
//!
//! - `BinningParams` (spatial/temporal bucket sizing) influences the cost/recall
//!   trade-off of neighborhood search.
//! - `TripletConfig` extends validated pairs into higher-quality seeds.

use serde::{Deserialize, Serialize};

use crate::engine_config::units::{de_ang_speed_rad_per_day, de_time_days};
use crate::{MJDTT, error::SeedError};

/// Parameters controlling **pair generation** between alerts `(a, b)`.
///
/// A "pair" is the minimal seed of a possible trajectory, defined by two
/// distinct alerts close in time, consistent with a maximum angular speed,
/// and with compatible photometry.
///
/// Behavior
/// --------
/// A candidate pair `(a, b)` is considered only if it passes:
///
/// - Temporal gating: `t_b > t_a` and `Δt ≤ max_dt`.
/// - Kinematic gating: `ang_sep(a, b) / Δt ≤ max_angular_speed`.
/// - Photometric gating: the implementation-specific brightness similarity
///   test using `max_flux_difference`.
///
/// Notes
/// -----
/// - This struct is `serde`-deserializable to support robust configuration loading
///   (YAML + environment overrides) via the `config` crate.
/// - Unknown keys are rejected (`deny_unknown_fields`) to catch YAML typos early.
/// - Missing fields are filled from [`Default`] (`serde(default)`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PairConfig {
    /// Maximum allowed time separation `Δt` between alerts `a` and `b`.
    ///
    /// Units
    /// -----
    /// - Canonical: **days (TT)**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in days): `0.06`
    /// - string with units: `"86.4 min"`, `"1.44 h"`, `"30 sec"`, `"0.06 day"`
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_time_days`].
    #[serde(deserialize_with = "de_time_days")]
    pub max_dt: MJDTT,

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

    /// Maximum allowed photometric difference between the two alerts.
    ///
    /// Important
    /// ---------
    /// This value is **dimensionless at the configuration layer**. Its meaning
    /// depends on the pair generation kernel:
    /// - raw flux difference threshold,
    /// - magnitude difference threshold,
    /// - normalized residual threshold,
    /// - or any other scalar similarity metric.
    pub max_flux_difference: f64,

    /// Whether to allow pairs formed from alerts inside the same **time bin**.
    ///
    /// Context
    /// -------
    /// Many pipelines pre-bin alerts in time to reduce the neighborhood search.
    /// Depending on the bin width, allowing same-bin pairing can:
    /// - increase recall,
    /// - increase contamination.
    pub allow_same_timebin: bool,
}

impl Default for PairConfig {
    /// Provide LSST/ZTF-like defaults for intra-night cadence.
    ///
    /// Defaults
    /// --------
    /// - `max_dt = 0.06` days (~86.4 minutes)
    /// - `max_angular_speed = 5.0e-2` rad/day (order-of-magnitude)
    /// - `max_flux_difference = 5.0`
    /// - `allow_same_timebin = true`
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

    /// Conservative maximum separation (radians) used for **spatial neighborhood**
    /// traversal in the bucket index:
    ///
    /// ```text
    /// sep_cap = max_angular_speed * max_dt
    /// ```
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
    /// Set maximum allowed `Δt` between alerts (days, TT).
    pub fn max_dt(mut self, v: MJDTT) -> Self {
        self.params.max_dt = v;
        self
    }

    /// Set maximum allowed angular speed (radians per day).
    pub fn max_angular_speed(mut self, v: f64) -> Self {
        self.params.max_angular_speed = v;
        self
    }

    /// Set maximum allowed photometric difference (dimensionless).
    pub fn max_flux_difference(mut self, v: f64) -> Self {
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

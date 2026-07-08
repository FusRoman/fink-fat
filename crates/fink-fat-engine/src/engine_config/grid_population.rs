//! # Admissible-region seeding grid configuration (`GridConfig`, `Population`)
//!
//! This module defines the configuration for the **`(ρ, ρ̇)` admissible-region
//! grid** built by [`admissible_region_grid`], the topocentric-range/range-rate
//! sampling used to seed Kalman hypotheses for a new tracklet before any
//! heliocentric orbit is known.
//!
//! Two pieces make up this configuration:
//! - [`Population`]: a Gaussian dynamical-population prior over semi-major
//!   axis, used to weight grid nodes by dynamical plausibility (see
//!   [`default_populations`] for the built-in Solar System population set).
//! - [`GridConfig`]: the grid's sampling bounds/resolution plus numerical
//!   floors that keep degenerate cells from dominating or vanishing.
//!
//! This configuration is `serde`-deserializable (YAML) and uses the
//! project-level unit parsers from [`crate::engine_config::units`] so that
//! distance/speed fields accept either canonical AU / AU-per-day numbers or
//! human-friendly unit strings (e.g. `"0.02 au"`, `"1 au/day"`).

use serde::{Deserialize, Serialize};

use crate::engine_config::{
    Validate,
    error::{FieldError, prefix_errors},
    units::{de_length_au, de_speed_au_per_day},
    validate_helpers::{
        check_finite_in_range, check_finite_nonneg, check_finite_positive, check_lt,
        check_min_usize,
    },
};

/// A small-body population, modeled as a Gaussian prior over semi-major axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    /// Human-readable label (NEO, MBA, …).
    pub name: String,

    /// Peak semi-major axis of the population.
    ///
    /// Units
    /// -----
    /// - Canonical: **AU**.
    ///
    /// YAML forms
    /// ---------
    /// - numeric (already in AU): `1.5`
    /// - string with units: `"1.5 au"`, `"224 400 000 km"`
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_length_au`].
    #[serde(deserialize_with = "de_length_au")]
    pub a_center: f64,

    /// Spread (standard deviation) of the population in semi-major axis.
    ///
    /// Units
    /// -----
    /// - Canonical: **AU**.
    ///
    /// Must be strictly positive for the Gaussian prior to be well-defined.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_length_au`], same accepted YAML forms as `a_center`.
    #[serde(deserialize_with = "de_length_au")]
    pub a_sigma: f64,

    /// Relative abundance (un-normalized; only ratios between populations'
    /// weights matter — the sampler normalizes internally). Dimensionless,
    /// must be non-negative.
    pub weight: f64,
}

impl Validate for Population {
    /// Validate internal consistency and numeric ranges, accumulating every
    /// failure found instead of stopping at the first one.
    fn validate(&self) -> Result<(), Vec<FieldError>> {
        let mut errors = Vec::new();

        if let Some(e) = check_finite_positive(
            "a_center",
            self.a_center,
            "set a_center to a strictly positive semi-major axis, e.g. \"1.5 au\" or 1.5 (AU)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "a_sigma",
            self.a_sigma,
            "set a_sigma to a strictly positive spread, e.g. \"0.6 au\" or 0.6 (AU)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_nonneg(
            "weight",
            self.weight,
            "set weight to a non-negative relative abundance, e.g. 0.02",
        ) {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Rough default population priors covering the main dynamical families
/// of the Solar System, from Near-Earth Objects out to the distant Kuiper Belt.
///
/// Population model
/// ----------------
/// Each [`Population`] is a Gaussian kernel $\mathcal{N}(a_\text{center},\,
/// a_\sigma^2)$ over semi-major axis $a$ (AU). The `weight` field encodes
/// relative abundance; only ratios between weights matter — the sampler
/// normalises them internally.
///
/// Populations included
/// --------------------
///
/// | Name            | $a_\text{center}$ (AU) | $a_\sigma$ (AU) | Notes                                      |
/// |-----------------|------------------------|------------------|--------------------------------------------|
/// | NEO             | 1.50                   | 0.60             | Amors, Apollos, Atens, Atiras               |
/// | MBA Inner       | 2.20                   | 0.15             | Inside the 3:1 Kirkwood gap                |
/// | MBA Middle      | 2.70                   | 0.15             | Between 3:1 and 5:2 gaps                   |
/// | MBA Outer       | 3.10                   | 0.15             | Between 5:2 and 2:1 gaps                   |
/// | Cybele          | 3.42                   | 0.10             | Just outside the 2:1 resonance             |
/// | Hilda           | 3.95                   | 0.15             | 3:2 mean-motion resonance with Jupiter     |
/// | Trojan          | 5.20                   | 0.30             | Jupiter L4/L5 clouds                       |
/// | Centaur         | 15.00                  | 5.00             | Scattered between Jupiter and Neptune      |
/// | KBO Cold        | 44.00                  | 2.00             | Cold classical Kuiper Belt                 |
/// | KBO Hot         | 45.00                  | 8.00             | Hot classical Kuiper Belt                  |
/// | KBO Resonant    | 39.40                  | 1.00             | Plutinos — 3:2 resonance with Neptune      |
/// | SDO             | 70.00                  | 20.00            | Scattered-disc objects                     |
/// | Detached / ETNO | 90.00                  | 10.00            | Detached / extreme TNOs up to ~100 AU      |
///
/// Design intent
/// -------------
/// These priors are deliberately broad. Within a few nights of observations
/// the data resolves the semi-major axis tightly enough that the prior only
/// needs to seed plausible modes and prevent the hypothesis bank from wasting
/// samples on dynamically empty regions of element space.
///
/// The three MBA sub-populations replace the single broad kernel used
/// previously. This avoids over-smoothing across the major Kirkwood gaps
/// at 2.50 AU (3:1), 2.82 AU (5:2) and 2.95 AU (7:3).
pub fn default_populations() -> Vec<Population> {
    vec![
        // --- Near-Earth Objects ---
        Population {
            name: "NEO".to_string(),
            a_center: 1.50,
            a_sigma: 0.60,
            weight: 0.02,
        },
        // --- Main Belt (three sub-populations separated by Kirkwood gaps) ---
        Population {
            name: "MBA Inner".to_string(),
            a_center: 2.20,
            a_sigma: 0.15,
            weight: 0.60,
        },
        Population {
            name: "MBA Middle".to_string(),
            a_center: 2.70,
            a_sigma: 0.15,
            weight: 1.00,
        },
        Population {
            name: "MBA Outer".to_string(),
            a_center: 3.10,
            a_sigma: 0.15,
            weight: 0.70,
        },
        // --- Cybele group (just outside the 2:1 resonance at 3.28 AU) ---
        Population {
            name: "Cybele".to_string(),
            a_center: 3.42,
            a_sigma: 0.10,
            weight: 0.04,
        },
        // --- Hilda group (3:2 resonance with Jupiter) ---
        Population {
            name: "Hilda".to_string(),
            a_center: 3.95,
            a_sigma: 0.15,
            weight: 0.05,
        },
        // --- Jupiter Trojans (L4 / L5) ---
        Population {
            name: "Trojan".to_string(),
            a_center: 5.20,
            a_sigma: 0.30,
            weight: 0.10,
        },
        // --- Centaurs ---
        Population {
            name: "Centaur".to_string(),
            a_center: 15.00,
            a_sigma: 5.00,
            weight: 0.03,
        },
        // --- Kuiper Belt — cold classical disk ---
        Population {
            name: "KBO Cold".to_string(),
            a_center: 44.00,
            a_sigma: 2.00,
            weight: 0.04,
        },
        // --- Kuiper Belt — hot classical disk ---
        Population {
            name: "KBO Hot".to_string(),
            a_center: 45.00,
            a_sigma: 8.00,
            weight: 0.03,
        },
        // --- Plutinos (3:2 resonance with Neptune at ~39.4 AU) ---
        Population {
            name: "KBO Resonant".to_string(),
            a_center: 39.40,
            a_sigma: 1.00,
            weight: 0.02,
        },
        // --- Scattered-disc objects ---
        Population {
            name: "SDO".to_string(),
            a_center: 70.00,
            a_sigma: 20.00,
            weight: 0.02,
        },
        // --- Detached / extreme TNOs (up to ~100 AU) ---
        Population {
            name: "Detached".to_string(),
            a_center: 90.00,
            a_sigma: 10.00,
            weight: 0.01,
        },
    ]
}

/// Configuration for [`admissible_region_grid`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Minimum topocentric range sampled by the grid.
    ///
    /// Units
    /// -----
    /// - Canonical: **AU**.
    ///
    /// Context
    /// -------
    /// Excludes near-Earth/geocentric objects too close to be handled by the
    /// admissible-region formalism. Must be strictly positive and `< rho_max`.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_length_au`].
    #[serde(deserialize_with = "de_length_au")]
    pub rho_min: f64,

    /// Maximum topocentric range sampled by the grid.
    ///
    /// Units
    /// -----
    /// - Canonical: **AU**.
    ///
    /// Typical values reach out to the outer Solar System (SDO/detached
    /// TNOs); see [`default_populations`] for the population set this bound
    /// should comfortably cover.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_length_au`].
    #[serde(deserialize_with = "de_length_au")]
    pub rho_max: f64,

    /// Number of range nodes, log-spaced between `rho_min` and `rho_max`.
    ///
    /// Dimensionless count. Must be `≥ 1`; in practice a handful of dozens
    /// (e.g. 20-30) balances coverage against the number of hypotheses seeded
    /// per tracklet.
    pub n_rho: usize,

    /// Number of range-rate nodes per range node, linearly spaced inside the
    /// admissible-region's bound interval for that range.
    ///
    /// Dimensionless count. Must be `≥ 1`.
    pub n_rho_dot: usize,

    /// Population priors used to weight each `(ρ, ρ̇)` node by dynamical
    /// plausibility. See [`Population`] and [`default_populations`].
    pub populations: Vec<Population>,

    /// Floor on the range σ, applied when a grid cell's half-width would
    /// otherwise be smaller than this (e.g. near the grid boundary).
    ///
    /// Units
    /// -----
    /// - Canonical: **AU**.
    ///
    /// Must be strictly positive; too small a floor can produce
    /// overconfident (numerically unstable) initial covariances.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_length_au`].
    #[serde(deserialize_with = "de_length_au")]
    pub sigma_pos_au_floor: f64,

    /// Floor on the range-rate σ, applied under the same circumstances as
    /// `sigma_pos_au_floor`.
    ///
    /// Units
    /// -----
    /// - Canonical: **AU/day**.
    ///
    /// Serialization
    /// -------------
    /// Parsed with [`de_speed_au_per_day`].
    #[serde(deserialize_with = "de_speed_au_per_day")]
    pub sigma_rho_dot_floor: f64,

    /// Nodes whose (normalized) population weight falls below this threshold
    /// are dropped from the grid. Dimensionless ratio, expected in `[0, 1)`;
    /// `0.0` disables pruning.
    pub weight_floor: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            rho_min: 0.1,
            rho_max: 60.0,
            n_rho: 28,
            n_rho_dot: 11,
            populations: default_populations(),
            sigma_pos_au_floor: 1e-3,
            sigma_rho_dot_floor: 1e-4,
            weight_floor: 1e-6,
        }
    }
}

impl Validate for GridConfig {
    /// Validate internal consistency and numeric ranges, accumulating every
    /// failure found instead of stopping at the first one.
    fn validate(&self) -> Result<(), Vec<FieldError>> {
        let mut errors = Vec::new();

        if let Some(e) = check_finite_positive(
            "rho_min",
            self.rho_min,
            "set rho_min to a strictly positive topocentric range, e.g. \"0.1 au\" or 0.1 (AU)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "rho_max",
            self.rho_max,
            "set rho_max to a strictly positive topocentric range, e.g. \"60 au\" or 60.0 (AU)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_lt(
            "rho_min",
            self.rho_min,
            "rho_max",
            self.rho_max,
            "raise rho_max above rho_min (or lower rho_min) so the grid samples a non-empty range",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_min_usize(
            "n_rho",
            self.n_rho,
            1,
            "set n_rho to at least 1 range node, e.g. 28",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_min_usize(
            "n_rho_dot",
            self.n_rho_dot,
            1,
            "set n_rho_dot to at least 1 range-rate node per range node, e.g. 11",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "sigma_pos_au_floor",
            self.sigma_pos_au_floor,
            "set sigma_pos_au_floor to a strictly positive distance floor, e.g. 1e-3 (AU)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_positive(
            "sigma_rho_dot_floor",
            self.sigma_rho_dot_floor,
            "set sigma_rho_dot_floor to a strictly positive speed floor, e.g. 1e-4 (AU/day)",
        ) {
            errors.push(e);
        }
        if let Some(e) = check_finite_in_range(
            "weight_floor",
            self.weight_floor,
            0.0,
            1.0 - f64::EPSILON,
            "set weight_floor to a value in [0, 1), e.g. 1e-6 (0.0 disables pruning)",
        ) {
            errors.push(e);
        }

        if self.populations.is_empty() {
            errors.push(
                FieldError::new("populations", "must not be empty").with_hint(
                    "provide at least one Population prior, or use default_populations()",
                ),
            );
        }
        for (i, population) in self.populations.iter().enumerate() {
            if let Err(e) = population.validate() {
                errors.extend(prefix_errors(e, &format!("populations[{i}]")));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

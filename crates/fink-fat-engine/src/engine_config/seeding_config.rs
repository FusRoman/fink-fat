//! Seeding-level policy configuration.
//!
//! This module contains options that control how intra-night seeds are emitted
//! by the [`BuildSeeds`](crate::pipeline::stages::PipelineStage::BuildSeeds)
//! stage.

use serde::{Deserialize, Serialize};

use crate::{
    Radian,
    engine_config::units::{de_ang_speed_rad_per_day, de_angle_rad},
    error::SeedError,
};

/// Seeding strategy used by `BuildSeeds`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedingMethod {
    /// Existing pair/triplet streaming strategy.
    #[default]
    PairTriplet,
    /// Kinematic Hough-transform strategy.
    Hough,
}

/// Parameters for the Hough-transform seeding strategy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HoughSeedingConfig {
    /// Minimum speed norm considered in the velocity grid (rad/day).
    #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
    pub min_angular_speed: f64,

    /// Maximum speed norm considered in the velocity grid (rad/day).
    #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
    pub max_angular_speed: f64,

    /// Number of grid steps on each velocity axis.
    pub velocity_grid_steps: usize,

    /// Spatial bin size for projected `(alpha0, delta0)` in radians.
    #[serde(deserialize_with = "de_angle_rad")]
    pub spatial_bin_size: Radian,

    /// Minimum number of alerts in an accumulator peak.
    pub min_alerts_per_peak: usize,

    /// Maximum number of peaks kept per night (highest score first).
    pub max_peaks_per_night: usize,

    /// Apply photometric consistency filtering on extracted peaks.
    pub photometric_filter: bool,

    /// Maximum allowed magnitude spread inside one band.
    pub photometric_max_mag_diff: f64,

    /// Extra tolerance multiplier on magnitude uncertainties.
    pub photometric_sigma_multiplier: f64,

    /// Weight each vote by photometric uncertainty when possible.
    pub weight_by_photometric_error: bool,
}

impl Default for HoughSeedingConfig {
    fn default() -> Self {
        Self {
            min_angular_speed: 0.0,
            max_angular_speed: 0.08,
            velocity_grid_steps: 21,
            spatial_bin_size: 3.0_f64.to_radians() / 3600.0,
            min_alerts_per_peak: 3,
            max_peaks_per_night: 4_000,
            photometric_filter: true,
            photometric_max_mag_diff: 0.5,
            photometric_sigma_multiplier: 3.0,
            weight_by_photometric_error: true,
        }
    }
}

impl HoughSeedingConfig {
    pub fn validate(&self) -> Result<(), SeedError> {
        if !self.min_angular_speed.is_finite() || self.min_angular_speed < 0.0 {
            return Err(SeedError::NonFiniteOrNegativeAngle(
                "seeding.hough.min_angular_speed",
            ));
        }
        if !self.max_angular_speed.is_finite() || self.max_angular_speed <= 0.0 {
            return Err(SeedError::NonFiniteOrNegativeAngle(
                "seeding.hough.max_angular_speed",
            ));
        }
        if self.min_angular_speed > self.max_angular_speed {
            return Err(SeedError::Inconsistent(
                "seeding.hough.min_angular_speed must be <= seeding.hough.max_angular_speed",
            ));
        }
        if self.velocity_grid_steps < 2 {
            return Err(SeedError::Inconsistent(
                "seeding.hough.velocity_grid_steps must be >= 2",
            ));
        }
        if !self.spatial_bin_size.is_finite() || self.spatial_bin_size <= 0.0 {
            return Err(SeedError::NonFiniteOrNegativeAngle(
                "seeding.hough.spatial_bin_size",
            ));
        }
        if self.min_alerts_per_peak < 2 {
            return Err(SeedError::Inconsistent(
                "seeding.hough.min_alerts_per_peak must be >= 2",
            ));
        }
        if self.max_peaks_per_night == 0 {
            return Err(SeedError::Inconsistent(
                "seeding.hough.max_peaks_per_night must be > 0",
            ));
        }
        if !self.photometric_max_mag_diff.is_finite() || self.photometric_max_mag_diff < 0.0 {
            return Err(SeedError::NonFiniteOrNegativePhotometry(
                "seeding.hough.photometric_max_mag_diff",
            ));
        }
        if !self.photometric_sigma_multiplier.is_finite() || self.photometric_sigma_multiplier < 0.0
        {
            return Err(SeedError::NonFiniteOrNegativePhotometry(
                "seeding.hough.photometric_sigma_multiplier",
            ));
        }
        Ok(())
    }
}

/// Configuration controlling how seeds are emitted during `BuildSeeds`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SeedingConfig {
    /// Seeding strategy used by `BuildSeeds`.
    pub method: SeedingMethod,

    /// If `true`, keep only triplet-derived seeds and drop pair-derived seeds.
    ///
    /// For Hough seeding, this controls whether peaks with only 2 alerts are
    /// allowed to emit pair-derived seeds.
    pub triplet_only: bool,

    /// Parameters for the Hough-transform strategy.
    pub hough: HoughSeedingConfig,
}

impl SeedingConfig {
    pub fn validate(&self) -> Result<(), SeedError> {
        self.hough.validate()
    }
}

#[cfg(test)]
mod seeding_config_tests {
    use super::*;

    #[test]
    fn default_is_pair_triplet() {
        let cfg = SeedingConfig::default();
        assert_eq!(cfg.method, SeedingMethod::PairTriplet);
        assert!(!cfg.triplet_only);
    }

    #[test]
    fn hough_config_validate_ok() {
        HoughSeedingConfig::default()
            .validate()
            .expect("default hough config must validate");
    }

    #[test]
    fn hough_config_invalid_grid_steps() {
        let cfg = HoughSeedingConfig {
            velocity_grid_steps: 1,
            ..HoughSeedingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn serde_method_names() {
        let yaml = r#"
method: hough
triplet_only: true
hough:
  min_angular_speed: "0 arcsec/hour"
  max_angular_speed: "3600 arcsec/hour"
  velocity_grid_steps: 11
  spatial_bin_size: "2 arcsec"
  min_alerts_per_peak: 3
  max_peaks_per_night: 128
  photometric_filter: true
  photometric_max_mag_diff: 0.7
  photometric_sigma_multiplier: 3.0
  weight_by_photometric_error: true
"#;
        let cfg: SeedingConfig = serde_yaml::from_str(yaml).expect("parse seeding config");
        assert_eq!(cfg.method, SeedingMethod::Hough);
        assert!(cfg.triplet_only);
        cfg.validate().expect("config must validate");
    }
}

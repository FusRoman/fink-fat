//! Photometric summaries attached to seed candidates.
//!
//! This module defines [`Photometry`], the compact photometric descriptor stored
//! on [`SeedNode`](crate::seeding::SeedNode) values and consumed by downstream
//! matching, scoring, and diagnostic code.
//!
//! The representation separates two concerns:
//!
//! - `mag_mean` and `mag_std` summarize the magnitude distribution of the
//!   member alerts.
//! - `bands` and `band_mask` describe band coverage for compatibility checks.
//!
//! The `bands` array is intentionally small and primarily intended for display
//! and debugging. The authoritative representation for overlap logic is
//! `band_mask`, which stores the full set of observed band identifiers as a bit
//! mask.
//!
//! ## Scientific interpretation
//!
//! The magnitude spread stored in `mag_std` is a robust mean absolute deviation
//! around the sample mean, not a variance-based estimator. It is therefore a
//! compact heterogeneity indicator for seed membership rather than a statistical
//! uncertainty estimate.
//!
//! ## Main types
//!
//! - [`Photometry`] stores the summary statistics and band coverage for one seed.
//!
//! ## Related behavior
//!
//! - [`Photometry::from_alerts`] aggregates an arbitrary collection of alerts.
//! - [`Photometry::shares_any_band`] performs band-overlap checks using the full
//!   bit mask rather than the display-only `bands` array.

use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use crate::Alert;

/// Photometric summary attached to a seed candidate.
///
/// This structure is intentionally compact: it carries the magnitude summary
/// needed by downstream ranking, together with a band-coverage descriptor used
/// by compatibility tests and diagnostics.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Photometry {
    /// Mean magnitude over the member alerts with finite magnitude values.
    pub mag_mean: f32,

    /// Mean absolute deviation around `mag_mean` over finite magnitudes.
    pub mag_std: f32,

    /// Number of representative bands stored in `bands` (0..=3).
    pub n_bands: u8,

    /// Representative band ids. Unused slots are 0.
    pub bands: [u8; 3],

    /// Bitmask of all bands present in this seed.
    ///
    /// The bitmask is the authoritative representation for band overlap tests.
    #[serde(default)]
    pub band_mask: u32,
}

impl Display for Photometry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let bands = match self.n_bands {
            2 => format!("[{}, {}]", self.bands[0], self.bands[1]),
            3 => format!("[{}, {}, {}]", self.bands[0], self.bands[1], self.bands[2]),
            _ => format!("[{}, {}, {}]", self.bands[0], self.bands[1], self.bands[2]),
        };
        write!(
            f,
            "Photometry {{ mag_mean: {:.6e}, mag_std: {:.6e}, bands: {} }}",
            self.mag_mean, self.mag_std, bands
        )
    }
}

impl Photometry {
    #[inline]
    fn mask_from_bands(bands: &[u8]) -> u32 {
        let mut m = 0u32;
        for &b in bands {
            if b < 32 {
                m |= 1u32 << (b as u32);
            }
        }
        m
    }

    /// Build photometry for a pair seed.
    ///
    /// Arguments
    /// ---------
    /// * `mag_mean` - Mean magnitude of the seed members.
    /// * `mag_std` - Mean absolute deviation of the seed members.
    /// * `band_a` - Band identifier of the first member.
    /// * `band_b` - Band identifier of the second member.
    ///
    /// Return
    /// ------
    /// * `Photometry` - Compact summary with two representative bands and a
    ///   full band bit mask.
    #[inline]
    pub fn from_pair(mag_mean: f32, mag_std: f32, band_a: u8, band_b: u8) -> Self {
        let rep = [band_a, band_b, 0];
        Self {
            mag_mean,
            mag_std,
            n_bands: 2,
            bands: rep,
            band_mask: Self::mask_from_bands(&rep[..2]),
        }
    }

    /// Build photometry for a triplet seed.
    ///
    /// Arguments
    /// ---------
    /// * `mag_mean` - Mean magnitude of the seed members.
    /// * `mag_std` - Mean absolute deviation of the seed members.
    /// * `band_a` - Band identifier of the first member.
    /// * `band_b` - Band identifier of the second member.
    /// * `band_c` - Band identifier of the third member.
    ///
    /// Return
    /// ------
    /// * `Photometry` - Compact summary with three representative bands and a
    ///   full band bit mask.
    #[inline]
    pub fn from_triplet(mag_mean: f32, mag_std: f32, band_a: u8, band_b: u8, band_c: u8) -> Self {
        let rep = [band_a, band_b, band_c];
        Self {
            mag_mean,
            mag_std,
            n_bands: 3,
            bands: rep,
            band_mask: Self::mask_from_bands(&rep[..3]),
        }
    }

    /// Build photometry from an arbitrary number of alerts.
    ///
    /// The numeric summary is aggregated over every alert with a finite
    /// magnitude. The `bands` field keeps only the first three band identifiers
    /// for display, while `band_mask` records the complete band coverage.
    /// Alerts with non-finite magnitudes are ignored for the magnitude
    /// statistics but still contribute to band coverage.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` - Slice of alerts contributing to the summary.
    ///
    /// Return
    /// ------
    /// * `Photometry` - Summary of magnitude central tendency, magnitude
    ///   dispersion, and full band coverage.
    #[inline]
    pub fn from_alerts(alerts: &[&Alert]) -> Self {
        if alerts.is_empty() {
            return Self::default();
        }

        let mut sum = 0.0f64;
        let mut n = 0usize;
        for alert in alerts {
            if alert.mag.is_finite() {
                sum += alert.mag;
                n += 1;
            }
        }

        if n == 0 {
            let mut out = Self::default();
            for alert in alerts {
                if out.n_bands < 3 {
                    out.bands[out.n_bands as usize] = alert.band;
                    out.n_bands += 1;
                }
                if alert.band < 32 {
                    out.band_mask |= 1u32 << (alert.band as u32);
                }
            }
            return out;
        }

        let mean = sum / (n as f64);
        let mut abs_dev_sum = 0.0f64;
        for alert in alerts {
            if alert.mag.is_finite() {
                abs_dev_sum += (alert.mag - mean).abs();
            }
        }
        let std = abs_dev_sum / (n as f64);

        let mut out = Self {
            mag_mean: mean as f32,
            mag_std: std as f32,
            ..Self::default()
        };
        for alert in alerts {
            if out.n_bands < 3 {
                out.bands[out.n_bands as usize] = alert.band;
                out.n_bands += 1;
            }
            if alert.band < 32 {
                out.band_mask |= 1u32 << (alert.band as u32);
            }
        }
        out
    }

    /// True if all detections were observed in the same band.
    ///
    /// Arguments
    /// ---------
    /// None.
    ///
    /// Return
    /// ------
    /// * `true` - The representative bands are identical.
    /// * `false` - At least two different band identifiers are present or the
    ///   summary does not contain enough representative bands to decide.
    #[inline]
    pub fn is_single_band(&self) -> bool {
        match self.n_bands {
            2 => self.bands[0] == self.bands[1],
            3 => self.bands[0] == self.bands[1] && self.bands[1] == self.bands[2],
            _ => false,
        }
    }

    /// If single-band, return that band id.
    ///
    /// Arguments
    /// ---------
    /// None.
    ///
    /// Return
    /// ------
    /// * `Some(u8)` - The single band identifier when the summary is
    ///   monoband.
    /// * `None` - The summary spans multiple bands or the representation does
    ///   not contain enough information.
    #[inline]
    pub fn single_band(&self) -> Option<u8> {
        if self.is_single_band() {
            Some(self.bands[0])
        } else {
            None
        }
    }

    /// True if the seed mixes bands (useful for scoring/debug).
    ///
    /// Arguments
    /// ---------
    /// None.
    ///
    /// Return
    /// ------
    /// * `true` - The seed is not monoband.
    /// * `false` - All representative bands are identical.
    #[inline]
    pub fn band_mismatch(&self) -> bool {
        !self.is_single_band()
    }

    /// Return a compact bitmask of bands present in this seed.
    ///
    /// The stored `band_mask` value is returned when available. Otherwise, the
    /// mask is reconstructed from the representative band list.
    ///
    /// Notes
    /// -----
    /// This assumes band identifiers are small non-negative integers, which is
    /// consistent with the current survey encodings used in the pipeline.
    ///
    /// Arguments
    /// ---------
    /// None.
    ///
    /// Return
    /// ------
    /// * `u32` - Bitmask of all represented bands.
    #[inline]
    pub fn band_mask(&self) -> u32 {
        if self.band_mask != 0 {
            self.band_mask
        } else {
            let n = self.n_bands.min(3) as usize;
            Self::mask_from_bands(&self.bands[..n])
        }
    }

    /// True if this seed shares at least one band with `other`.
    ///
    /// Arguments
    /// ---------
    /// * `other` - Another photometric summary to compare against.
    ///
    /// Return
    /// ------
    /// * `true` - The two summaries overlap in at least one band.
    /// * `false` - The band sets are disjoint.
    #[inline]
    pub fn shares_any_band(&self, other: &Photometry) -> bool {
        (self.band_mask() & other.band_mask()) != 0
    }
}

#[cfg(test)]
mod seeding_photometry_tests {
    use super::*;

    use crate::{AlertKey, night_id::NightId};

    fn mk_alert(source_id: u64, band: u8, mag: f64) -> Alert {
        Alert {
            key: AlertKey {
                night_id: NightId::new(42),
                dia_source_id: source_id,
            },
            band,
            mag,
            ..Default::default()
        }
    }

    #[test]
    fn from_alerts_aggregates_mean_and_member_bands() {
        let alerts = [
            mk_alert(0, 1, 10.0),
            mk_alert(1, 2, 20.0),
            mk_alert(2, 3, 30.0),
            mk_alert(3, 4, 40.0),
        ];
        let refs = [&alerts[0], &alerts[1], &alerts[2], &alerts[3]];

        let p = Photometry::from_alerts(&refs);

        assert!((p.mag_mean as f64 - 25.0).abs() < 1e-9);
        assert_eq!(p.n_bands, 3);
        assert_eq!(p.bands, [1, 2, 3]);

        let mask = p.band_mask();
        assert_ne!(mask & (1u32 << 1), 0);
        assert_ne!(mask & (1u32 << 4), 0);
    }

    #[test]
    fn shares_any_band_uses_full_band_mask() {
        let a = [
            mk_alert(0, 1, 10.0),
            mk_alert(1, 2, 11.0),
            mk_alert(2, 4, 12.0),
        ];
        let b = [
            mk_alert(3, 5, 13.0),
            mk_alert(4, 4, 14.0),
            mk_alert(5, 6, 15.0),
        ];
        let c = [
            mk_alert(6, 7, 16.0),
            mk_alert(7, 8, 17.0),
            mk_alert(8, 9, 18.0),
        ];

        let pa = Photometry::from_alerts(&[&a[0], &a[1], &a[2]]);
        let pb = Photometry::from_alerts(&[&b[0], &b[1], &b[2]]);
        let pc = Photometry::from_alerts(&[&c[0], &c[1], &c[2]]);

        assert!(pa.shares_any_band(&pb));
        assert!(!pa.shares_any_band(&pc));
    }
}

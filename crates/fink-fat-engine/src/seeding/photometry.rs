use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

/// Photometry summary for a seed (pair or triplet).
///
/// Notes
/// -----
/// `bands` stores the per-detection filter id(s) in time order:
/// - pairs: `[b0, b1, 0]`
/// - triplets: `[b0, b1, b2]`
/// with `n_bands` indicating how many entries are valid.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Photometry {
    pub flux_mean: f32,
    pub flux_std: f32,

    /// Number of valid bands stored in `bands` (2 for pairs, 3 for triplets).
    pub n_bands: u8,

    /// Per-detection band ids, in time order. Unused slots are 0.
    pub bands: [u8; 3],
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
            "Photometry {{ flux_mean: {:.6e}, flux_std: {:.6e}, bands: {} }}",
            self.flux_mean, self.flux_std, bands
        )
    }
}

impl Photometry {
    /// Build photometry for a pair seed.
    #[inline]
    pub fn from_pair(flux_mean: f32, flux_std: f32, band_a: u8, band_b: u8) -> Self {
        Self {
            flux_mean,
            flux_std,
            n_bands: 2,
            bands: [band_a, band_b, 0],
        }
    }

    /// Build photometry for a triplet seed.
    #[inline]
    pub fn from_triplet(flux_mean: f32, flux_std: f32, band_a: u8, band_b: u8, band_c: u8) -> Self {
        Self {
            flux_mean,
            flux_std,
            n_bands: 3,
            bands: [band_a, band_b, band_c],
        }
    }

    /// True if all detections were observed in the same band.
    #[inline]
    pub fn is_single_band(&self) -> bool {
        match self.n_bands {
            2 => self.bands[0] == self.bands[1],
            3 => self.bands[0] == self.bands[1] && self.bands[1] == self.bands[2],
            _ => false,
        }
    }

    /// If single-band, return that band id.
    #[inline]
    pub fn single_band(&self) -> Option<u8> {
        if self.is_single_band() {
            Some(self.bands[0])
        } else {
            None
        }
    }

    /// True if the seed mixes bands (useful for scoring/debug).
    #[inline]
    pub fn band_mismatch(&self) -> bool {
        !self.is_single_band()
    }

    /// Return a compact bitmask of bands present in this seed.
    ///
    /// Notes
    /// -----
    /// This assumes band ids are small integers (e.g. ZTF fid 1/2, LSST 0..5).
    #[inline]
    pub fn band_mask(&self) -> u32 {
        let mut m = 0u32;
        let n = self.n_bands.min(3) as usize;
        for &b in &self.bands[..n] {
            if b > 0 && b < 32 {
                m |= 1u32 << (b as u32);
            }
        }
        m
    }

    /// True if this seed shares at least one band with `other`.
    #[inline]
    pub fn shares_any_band(&self, other: &Photometry) -> bool {
        (self.band_mask() & other.band_mask()) != 0
    }
}

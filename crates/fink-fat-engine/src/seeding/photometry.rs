use photom::photometry::Filter;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

/// Photometry summary for a seed (pair or triplet).
///
/// Notes
/// -----
/// `bands` stores the per-detection filter id(s) in time order:
/// - pairs: `[b0, b1, 0]`
/// - triplets: `[b0, b1, b2]`
///   with `n_bands` indicating how many entries are valid.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Photometry {
    pub mag_mean: f32,
    pub mag_std: f32,

    /// Number of valid bands stored in `bands` (2 for pairs, 3 for triplets).
    pub n_bands: u8,

    /// Is the seed observed in multiple bands? True if every detection is in the same band, false otherwise.
    pub share_bands: bool,
}

impl Display for Photometry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Photometry {{ mag_mean: {:.6e}, mag_std: {:.6e}, share_bands: {} }}",
            self.mag_mean, self.mag_std, self.share_bands
        )
    }
}

impl Photometry {
    /// Build photometry for a pair seed.
    #[inline]
    pub fn from_pair(mag_mean: f32, mag_std: f32, band_a: &Filter, band_b: &Filter) -> Self {
        Self {
            mag_mean,
            mag_std,
            n_bands: 2,
            share_bands: band_a == band_b,
        }
    }

    /// Build photometry for a triplet seed.
    #[inline]
    pub fn from_triplet(
        mag_mean: f32,
        mag_std: f32,
        band_a: &Filter,
        band_b: &Filter,
        band_c: &Filter,
    ) -> Self {
        Self {
            mag_mean,
            mag_std,
            n_bands: 3,
            share_bands: band_a == band_b && band_b == band_c,
        }
    }
}

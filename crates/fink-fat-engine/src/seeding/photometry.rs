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
pub struct SeedPhotometry {
    pub mag_mean: f32,
    pub mag_std: f32,

    pub n_bands: u8,
    bands: [Option<Filter>; 3],
}

impl Display for SeedPhotometry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Photometry {{ mag_mean: {:.6e}, mag_std: {:.6e}, bands: {:?} }}",
            self.mag_mean, self.mag_std, self.bands
        )
    }
}

impl SeedPhotometry {
    /// Build photometry for a pair seed.
    #[inline]
    pub fn from_pair(mag_mean: f32, mag_std: f32, band_a: Filter, band_b: Filter) -> Self {
        Self {
            mag_mean,
            mag_std,
            n_bands: 2,
            bands: [Some(band_a), Some(band_b), None],
        }
    }

    /// Build photometry for a triplet seed.
    #[inline]
    pub fn from_triplet(
        mag_mean: f32,
        mag_std: f32,
        band_a: Filter,
        band_b: Filter,
        band_c: Filter,
    ) -> Self {
        Self {
            mag_mean,
            mag_std,
            n_bands: 3,
            bands: [Some(band_a), Some(band_b), Some(band_c)],
        }
    }

    /// Check if this photometry shares any band with another.
    /// Returns `true` if at least one band overlaps, otherwise `false`.
    #[inline]
    pub fn shares_any_band(&self, other: &Self) -> bool {
        let na = (self.n_bands as usize).min(3);
        let nb = (other.n_bands as usize).min(3);

        self.bands[..na]
            .iter()
            .flatten()
            .any(|a| other.bands[..nb].iter().flatten().any(|b| a == b))
    }
}

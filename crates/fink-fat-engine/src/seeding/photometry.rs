use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct Photometry {
    pub flux_mean: f32,
    pub flux_std: f32,
    pub band: u8,
}

impl Display for Photometry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Photometry {{ flux_mean: {:.6e}, flux_std: {:.6e}, band: {} }}",
            self.flux_mean, self.flux_std, self.band
        )
    }
}

impl Photometry {
    #[inline]
    pub fn new(flux_mean: f32, flux_std: f32, band: u8) -> Self {
        Self {
            flux_mean,
            flux_std,
            band,
        }
    }
}

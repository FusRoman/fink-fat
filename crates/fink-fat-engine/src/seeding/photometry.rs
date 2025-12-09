use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct Photometry {
    pub flux_mean: f32,
    pub flux_std: f32,
    pub band: u8,
}

impl Photometry {
    #[inline]
    pub fn new(flux_mean: f32, flux_std: f32, band: u8) -> Self {
        Self { flux_mean, flux_std, band }
    }
}

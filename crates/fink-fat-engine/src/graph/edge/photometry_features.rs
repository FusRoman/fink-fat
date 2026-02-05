use crate::{graph::edge::edge_features::FeatureCore, seeding::seed_node::SeedNode};

/// Photometry features (mostly cadence-invariant).
///
/// These features depend mainly on flux statistics aggregated within each seed.
/// They tend to be more transferable across cadences than raw geometric features,
/// provided photometric calibration is comparable.
#[derive(Clone, Debug)]
pub struct EdgePhotometryFeatures {
    /// Normalized flux difference (z-score):
    /// `|flux_to - flux_from| / sqrt(σ_from² + σ_to² + σ_floor²)`.
    ///
    /// A variance floor is used to avoid infinite z-scores for extremely small
    /// reported uncertainties.
    pub z_flux: f64,

    /// Flux uncertainty ratio `sigma_to / sigma_from` (0 if undefined).
    ///
    /// This can indicate changes in S/N or data quality.
    pub flux_std_ratio: f64,

    /// Band-sharing indicator:
    /// `1.0` if both seeds share at least one photometric band, otherwise `0.0`.
    pub band_shared: f64,
}

impl EdgePhotometryFeatures {
    /// Compute only photometry features.
    #[inline]
    pub fn photometry_features(from: &SeedNode, to: &SeedNode) -> Self {
        let flux_i = from.photom.flux_mean as f64;
        let flux_j = to.photom.flux_mean as f64;
        let flux_abs_diff = (flux_j - flux_i).abs();

        let sigma_i = from.photom.flux_std as f64;
        let sigma_j = to.photom.flux_std as f64;

        // Variance floor is intentionally large-ish (in flux units) to avoid
        // exploding z-scores for tiny reported uncertainties.
        let sigma_floor = 1.0_f64;
        let pooled_var = sigma_i * sigma_i + sigma_j * sigma_j + sigma_floor * sigma_floor;

        let z_flux = if pooled_var.is_finite() && pooled_var > 0.0 {
            flux_abs_diff / pooled_var.sqrt()
        } else {
            0.0
        };

        let flux_std_ratio = if sigma_i.is_finite() && sigma_i > 0.0 {
            sigma_j / sigma_i
        } else {
            0.0
        };

        let band_shared = if from.photom.shares_any_band(&to.photom) {
            1.0
        } else {
            0.0
        };

        Self {
            z_flux: FeatureCore::finite_or_zero(z_flux),
            flux_std_ratio: FeatureCore::finite_or_zero(flux_std_ratio),
            band_shared,
        }
    }
}

//! Edge photometry features
//!
//! This module defines a small set of photometry-based features for an edge
//! between two seeds (`from -> to`).
//!
//! Design goals
//! ------------
//! - Keep features *mostly cadence-invariant*: avoid explicit dependence on `dt`
//!   or geometric propagation, so they generalize better across survey strategies.
//! - Use robust scalar summaries already aggregated inside each `SeedNode`
//!   (mean flux, flux scatter, observed bands).
//! - Ensure numerical stability: avoid NaNs/Infs in exported ML datasets.
//!
//! Notes on fluxes
//! --------------
//! We assume `SeedNode.photom` stores *comparable* flux measurements across seeds.
//! In practice, transferability depends on consistent photometric calibration
//! and bandpass definitions (e.g., same instrument/filter set or well-calibrated
//! cross-instrument mapping).
//!
//! -----------------------------------------------------------------------------
//!
//! Dependencies:
//! - `FeatureCore::finite_or_zero` is used as a shared sanitization policy to
//!   ensure stable feature export (Parquet / ONNX).
//!

use crate::{graph::edge::edge_features::feature_core::FeatureCore, seeding::SeedNode};

/// Photometry features for an edge (mostly cadence-invariant).
///
/// These features depend mainly on flux statistics aggregated within each seed.
/// They tend to be more transferable across cadences than raw geometric features,
/// provided photometric calibration is comparable.
///
/// Attributes
/// ----------
/// * `z_flux` – Normalized absolute flux difference:
///   $z\_f = \frac{|\bar{f}\_{\mathrm{to}} - \bar{f}\_{\mathrm{from}}|}{\sqrt{\sigma\_{\mathrm{from}}^2 + \sigma\_{\mathrm{to}}^2 + \sigma\_{\mathrm{floor}}^2}}$.
/// * `flux_std_ratio` – Ratio of flux standard deviations:
///   $r\_{\sigma} = \sigma\_{\mathrm{to}} / \sigma\_{\mathrm{from}}$.
/// * `band_shared` – Indicator whether seeds share at least one photometric
///   band ($0$ or $1$).
#[derive(Clone, Debug)]
pub struct EdgePhotometryFeatures {
    /// Normalized flux difference (z-score):
    ///
    /// $$z\_f = \frac{|\bar{f}\_{\mathrm{to}} - \bar{f}\_{\mathrm{from}}|}{\sqrt{\sigma\_{\mathrm{from}}^2 + \sigma\_{\mathrm{to}}^2 + \sigma\_{\mathrm{floor}}^2}}$$
    ///
    /// Interpretation
    /// --------------
    /// - Large values suggest an inconsistent brightness evolution between the two seeds.
    /// - Small values suggest photometric compatibility.
    ///
    /// Numerical stability
    /// -------------------
    /// A variance floor $\sigma\_{\mathrm{floor}}^2$ is included to prevent exploding
    /// z-scores when $\sigma\_{\mathrm{from}}$ and/or $\sigma\_{\mathrm{to}}$ are
    /// extremely small or underestimated.
    pub z_flux: f64,

    /// Flux uncertainty ratio:
    /// $r\_{\sigma} = \sigma\_{\mathrm{to}} \,/\, \sigma\_{\mathrm{from}}$
    /// ($0$ if undefined).
    ///
    /// Interpretation
    /// --------------
    /// - $r\_{\sigma} > 1$: the target seed is noisier (lower S/N).
    /// - $r\_{\sigma} < 1$: the target seed is cleaner (higher S/N).
    ///
    /// Notes
    /// -----
    /// This ratio is only meaningful if both $\sigma\_{\mathrm{from}}$ and
    /// $\sigma\_{\mathrm{to}}$ are computed consistently across seeds.
    pub flux_std_ratio: f64,

    /// Band-sharing indicator:
    /// $b\_{\mathrm{shared}} = 1$ if both seeds share at least one photometric band,
    /// otherwise $0$.
    ///
    /// Why this matters
    /// ----------------
    /// Comparing fluxes across different filters can introduce strong systematic
    /// offsets (e.g., color effects). This feature allows an ML model to learn
    /// that a flux mismatch is less informative when bands do not overlap.
    pub band_shared: bool,
}

impl EdgePhotometryFeatures {
    /// Compute photometry features for a directed edge `(from -> to)`.
    ///
    /// Overview
    /// --------
    /// 1. Extract per-seed aggregated flux statistics ($\bar{f}$, $\sigma$).
    /// 2. Compute an absolute flux difference $|\bar{f}\_{\mathrm{to}} - \bar{f}\_{\mathrm{from}}|$.
    /// 3. Normalize by a pooled uncertainty:
    ///    $\sqrt{\sigma\_{\mathrm{from}}^2 + \sigma\_{\mathrm{to}}^2 + \sigma\_{\mathrm{floor}}^2}$.
    /// 4. Compute the uncertainty ratio $r\_{\sigma} = \sigma\_{\mathrm{to}} / \sigma\_{\mathrm{from}}$.
    /// 5. Compute a band-sharing indicator ($0$ or $1$).
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node (older epoch).
    /// * `to` – Target seed node (newer epoch).
    ///
    /// Return
    /// ------
    /// [`EdgePhotometryFeatures`] with stable finite scalars.
    ///
    /// Notes
    /// -----
    /// - This method does not use time separation $\Delta t$; it is intended to be
    ///   relatively robust to cadence changes.
    /// - Non-finite intermediate values are mapped to $0$ via
    ///   `FeatureCore::finite_or_zero`.
    #[inline]
    pub fn photometry_features(from: &SeedNode, to: &SeedNode) -> Self {
        // ---------------------------------------------------------------------
        // 1) Extract aggregated photometry statistics
        // ---------------------------------------------------------------------
        // Mean flux for each seed (cast to f64 for stable numeric operations).
        let flux_i = from.photom.mag_mean as f64;
        let flux_j = to.photom.mag_mean as f64;

        // Absolute difference in mean flux between the two seeds.
        let flux_abs_diff = (flux_j - flux_i).abs();

        // Per-seed flux standard deviation (uncertainty proxy).
        let sigma_i = from.photom.mag_std as f64;
        let sigma_j = to.photom.mag_std as f64;

        // ---------------------------------------------------------------------
        // 2) z_flux: pooled-uncertainty normalized flux difference
        // ---------------------------------------------------------------------
        // Variance floor is intentionally "large-ish" (in flux units) to avoid
        // exploding z-scores for tiny reported uncertainties.
        //
        // Practical intuition:
        // - if sigma_i and sigma_j are unrealistically small, z_flux would become huge
        //   and dominate ML decisions in a brittle way.
        // - the floor limits that effect and makes the feature more robust.
        let sigma_floor = 1.0_f64;

        // Pooled variance: σ_from² + σ_to² + σ_floor²
        let pooled_var = sigma_i * sigma_i + sigma_j * sigma_j + sigma_floor * sigma_floor;

        // Convert pooled variance to pooled stddev and build the z-like score.
        let z_flux = if pooled_var.is_finite() && pooled_var > 0.0 {
            flux_abs_diff / pooled_var.sqrt()
        } else {
            0.0
        };

        // ---------------------------------------------------------------------
        // 3) flux_std_ratio: relative uncertainty proxy
        // ---------------------------------------------------------------------
        // Guard sigma_i to avoid division by zero and invalid ratios.
        let flux_std_ratio = if sigma_i.is_finite() && sigma_i > 0.0 {
            sigma_j / sigma_i
        } else {
            0.0
        };

        // ---------------------------------------------------------------------
        // 4) band_shared: categorical consistency indicator
        // ---------------------------------------------------------------------
        // If seeds share at least one band, direct photometric comparisons are
        // more meaningful (less color-systematic).
        let band_shared = from.photom.shares_any_band(&to.photom);

        // ---------------------------------------------------------------------
        // 5) Sanitize outputs
        // ---------------------------------------------------------------------
        // Keep ML features stable: map NaN/Inf -> 0.0.
        Self {
            z_flux: FeatureCore::finite_or_zero(z_flux),
            flux_std_ratio: FeatureCore::finite_or_zero(flux_std_ratio),
            band_shared,
        }
    }
}

//! Cadence-robust edge features (structured)
//!
//! This module defines a cadence-robust feature set for **inter-night edges**
//! (directed links) between two `SeedNode` objects.
//!
//! Why "cadence-robust"?
//! ---------------------
//! Survey cadence affects the distribution of `dt` (time gaps) between nights.
//! Any feature that depends strongly on `dt` (e.g., raw residual distances) will
//! not generalize well across cadences (ZTF → LSST, rolling cadence, weather gaps).
//!
//! The feature design here favors:
//! - normalized innovations (Mahalanobis / z-scores) rather than raw distances,
//! - along-track / cross-track decomposition rather than axis-aligned residuals,
//! - angular and relative quantities (dimensionless ratios),
//! - robust numerical guards (floors, epsilons, finite checks).
//!
//! Design philosophy
//! -----------------
//! - `FeatureCore` computes expensive shared intermediates once per edge.
//! - Public feature families (`position`, `velocity`, `uncertainty`, `photometry`)
//!   are stable, readable containers used for ML export (Parquet / Arrow / ONNX).
//! - Flat feature access is provided via:
//!   - `EdgeFeatureKey` (type-safe, compile-time checked),
//!   - canonical string paths (`EdgeFeatureKey::path()`),
//!   - canonical ordering (`EDGE_FEATURE_KEYS`) and allocation-free iterators.
//!
//! Any change to the canonical ordering or to the string paths is a breaking
//! change for downstream consumers (Python training code, ONNX export, plots, etc.).
//!
//! -----------------------------------------------------------------------------

use crate::{
    astro_math::safe_ln,
    graph::edge::{
        feature_core::FeatureCore, photometry_features::EdgePhotometryFeatures,
        position_features::EdgePositionFeatures, uncertainty_features::EdgeUncertaintyFeatures,
        velocity_features::EdgeVelocityFeatures,
    },
    seeding::SeedNode,
};

/// Cadence-robust edge feature set.
///
/// This is the **public**, ML-friendly container returned by the feature
/// computation pipeline.
///
/// Design
/// ------
/// The feature vector is intentionally **decomposed** into specialized
/// sub-structures:
/// - [`EdgePositionFeatures`] for innovation geometry on the tangent plane,
/// - [`EdgeVelocityFeatures`] for relative kinematic consistency,
/// - [`EdgeUncertaintyFeatures`] for uncertainty/quality ratios (dimensionless),
/// - [`EdgePhotometryFeatures`] for (mostly cadence-invariant) photometry,
///
/// This provides:
/// - better readability at call sites,
/// - easier iteration and ablation studies per feature family,
/// - cleaner code generation for downstream bindings (PyO3, serde, etc.).
///
/// Attributes
/// ----------
/// * `position` – Innovation geometry on tangent plane (Mahalanobis, whitening, directional z-scores).
/// * `velocity` – Kinematic compatibility (direction, speed ratios, velocity-space χ²).
/// * `uncertainty` – Uncertainty/quality scalars (e.g., covariance trace ratios).
/// * `photometry` – Photometric consistency (flux z-score, band overlap, etc.).
///
/// Notes
/// -----
/// - All scalar leaves are `f64` to keep ML export simple and stable.
/// - The canonical flat ordering is defined by [`EDGE_FEATURE_KEYS`].
#[derive(Clone, Debug)]
pub struct EdgeFeatures {
    /// Innovation geometry on the tangent plane (Mahalanobis, along/cross residuals).
    pub position: EdgePositionFeatures,
    /// Relative velocity consistency (angle, speed ratios).
    pub velocity: EdgeVelocityFeatures,
    /// Uncertainty/quality ratios (covariance trace ratios, anisotropy proxies).
    pub uncertainty: EdgeUncertaintyFeatures,
    /// Photometry consistency (normalized flux differences, band sharing).
    pub photometry: EdgePhotometryFeatures,
}

impl EdgeFeatures {
    // -------------------------------------------------------------------------
    // Feature API (high-level)
    // -------------------------------------------------------------------------

    /// Compute the full cadence-robust feature set for an edge `(from -> to)`.
    ///
    /// Overview
    /// --------
    /// 1. Build a [`FeatureCore`] once (propagation, innovation, covariances, guards).
    /// 2. Extract the structured feature families from the core and seeds:
    ///    - position from `core`,
    ///    - velocity from `core`,
    ///    - uncertainty directly from `(from, to)` (cheap scalar),
    ///    - photometry directly from `(from, to)` (seed-level aggregates).
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node (older epoch).
    /// * `to` – Target seed node (newer epoch).
    ///
    /// Return
    /// ------
    /// [`EdgeFeatures`] – structured feature vector for ML and graph scoring.
    ///
    /// Notes
    /// -----
    /// - This function is the main entrypoint used by edge construction code.
    /// - Numerical stability is primarily handled in [`FeatureCore`].
    #[inline]
    pub fn compute_features(from: &SeedNode, to: &SeedNode) -> Self {
        // Build shared intermediate quantities once (hot-path optimization).
        let core = FeatureCore::from_nodes(from, to);

        // Assemble structured feature families.
        EdgeFeatures {
            position: EdgePositionFeatures::position_features(&core),
            velocity: EdgeVelocityFeatures::velocity_features(&core),
            uncertainty: EdgeUncertaintyFeatures::uncertainty_features(from, to),
            photometry: EdgePhotometryFeatures::photometry_features(from, to),
        }
    }

    /// Return an additive negative-log-likelihood-like cost for graph solvers.
    ///
    /// Motivation
    /// ----------
    /// Many graph solvers (shortest path, min-cost flow, assignment, etc.) operate
    /// on **additive edge costs**. If per-edge costs approximate $-\ln p(\text{edge})$,
    /// summing them along a path approximates $-\ln p(\text{trajectory})$ (up to constants).
    ///
    /// The cost is designed to be:
    /// - physically motivated (Gaussian innovations on the tangent plane),
    /// - mostly parameter-free (only numerical epsilons),
    /// - additive along a trajectory (sum of per-edge costs),
    /// - stable (finite, avoids NaNs/Infs).
    ///
    /// Definition
    /// ----------
    /// Up to additive constants:
    ///
    /// $$\begin{align} c &= \frac{1}{2}\chi^2\_{\mathrm{pos}} \\ &+ \frac{1}{2}\chi^2\_{\mathrm{vel}} \\ &+ \frac{1}{2}z\_{\mathrm{flux}}^{2} \\ &+ \frac{1}{2}\bigl[\ln(|r\_{\sigma}| + \varepsilon)\bigr]^2 \\ &- \ln(\varepsilon\_{\mathrm{band}} + b\_{\mathrm{shared}}) \end{align}$$
    ///
    /// where $r\_{\sigma}$ is `flux_std_ratio` and $b\_{\mathrm{shared}} \in \{0, 1\}$.
    ///
    /// Term interpretation
    /// -------------------
    /// - $\chi^2\_{\mathrm{pos}}$: penalizes geometric inconsistency normalized by
    ///   uncertainties (position-space Mahalanobis distance).
    /// - $\chi^2\_{\mathrm{vel}}$: penalizes velocity inconsistency normalized by
    ///   uncertainties (velocity-space Mahalanobis distance).
    /// - $z\_{\mathrm{flux}}^{\,2}$: penalizes photometric inconsistency (scale-free).
    /// - $[\ln(|r\_{\sigma}| + \varepsilon)]^2$: penalizes large changes in flux
    ///   scatter (quality proxy).
    /// - $-\ln(\varepsilon\_{\mathrm{band}} + b\_{\mathrm{shared}})$: encourages band
    ///   overlap when possible.
    ///
    /// Return
    /// ------
    /// A finite `f64` cost (lower is better) suitable for additive solvers.
    ///
    /// Notes
    /// -----
    /// - $b\_{\mathrm{shared}}$ is expected in $\{0,1\}$ but is clamped defensively.
    /// - `flux_std_ratio` can be 0 if undefined; we guard with $\varepsilon$.
    /// - This is a *heuristic* scoring function. ML ranking may still outperform
    ///   this in practice, but this provides a strong, interpretable baseline.
    #[inline]
    pub fn kinematic_log_likelihood_cost(&self) -> f64 {
        // Numerical epsilons (not tunable model parameters).
        let eps = 1e-12_f64;
        let eps_band = 1e-3_f64;

        // Defensive guards against negative numeric drift.
        let chi2_pos = self.position.chi2_pos.max(0.0);
        let chi2_vel = self.velocity.chi2_vel.max(0.0);

        // Photometry terms.
        let z_flux = self.photometry.z_flux;

        // Guard ratio before log: abs() handles negative drift, eps avoids ln(0).
        let ln_flux_std_ratio = safe_ln(self.photometry.flux_std_ratio.abs() + eps);

        // Encourage band overlap: zero cost when bands are shared, fixed positive
        // penalty otherwise.
        // The previous formulation `-ln(eps_band + band_shared)` yielded a slightly
        // negative value when band_shared = 1 (ln(1.001) > 0), which could make the
        // total cost negative and trigger an `Edge::new` construction error.
        let band_shared = self.photometry.band_shared.clamp(0.0, 1.0);
        let band_term = if band_shared > 0.5 {
            0.0_f64
        } else {
            -safe_ln(eps_band) // ≈ +6.907, penalises absence of shared band
        };

        // Quadratic penalties resemble Gaussian negative log-likelihood terms.
        let cost = 0.5 * (chi2_pos + chi2_vel)
            + 0.5 * (z_flux * z_flux)
            + 0.5 * (ln_flux_std_ratio * ln_flux_std_ratio)
            + band_term;

        // Keep graph solver inputs stable (no NaN/Inf costs).
        FeatureCore::finite_or_zero(cost)
    }

    /// Return the total number of scalar leaf features.
    ///
    /// This is a compile-time constant and corresponds to the length of
    /// [`EDGE_FEATURE_KEYS`].
    ///
    /// Return
    /// ------
    /// Number of scalar features in the canonical flat representation.
    #[inline]
    pub const fn len_flat() -> usize {
        EDGE_FEATURE_KEYS.len()
    }

    /// Return an iterator over all canonical feature names (string paths).
    ///
    /// This iterator is allocation-free and preserves the canonical ordering
    /// defined by [`EDGE_FEATURE_KEYS`].
    ///
    /// Return
    /// ------
    /// Iterator of `&'static str` paths like `"position.chi2_pos"`.
    #[inline]
    pub fn flat_names() -> impl Iterator<Item = &'static str> {
        EDGE_FEATURE_KEYS.into_iter().map(|k| k.path())
    }

    /// Retrieve a feature value using a type-safe [`EdgeFeatureKey`].
    ///
    /// This is the **preferred access method** in core Rust code.
    ///
    /// Arguments
    /// ---------
    /// * `key` – Type-safe identifier of the desired scalar leaf feature.
    ///
    /// Return
    /// ------
    /// The feature value (canonical definition for that key).
    ///
    /// Notes
    /// -----
    /// This match is intentionally explicit (no reflection / string parsing),
    /// which keeps it fast and compiler-checked.
    #[inline]
    pub fn get(&self, key: EdgeFeatureKey) -> f64 {
        match key {
            // Position
            EdgeFeatureKey::PositionChi2Pos => self.position.chi2_pos,
            EdgeFeatureKey::PositionLogChi2Pos => self.position.log_chi2_pos,
            EdgeFeatureKey::PositionZDx => self.position.z_dx,
            EdgeFeatureKey::PositionZDy => self.position.z_dy,
            EdgeFeatureKey::PositionZResidNorm => self.position.z_resid_norm,
            EdgeFeatureKey::PositionZAlong => self.position.z_along,
            EdgeFeatureKey::PositionZCross => self.position.z_cross,
            EdgeFeatureKey::PositionCholZ1 => self.position.chol_z1,
            EdgeFeatureKey::PositionCholZ2 => self.position.chol_z2,
            EdgeFeatureKey::PositionCholZNorm => self.position.chol_z_norm,

            // Velocity
            EdgeFeatureKey::VelocityCosDthetaV => self.velocity.cos_dtheta_v,
            EdgeFeatureKey::VelocityRelSpeedDiff => self.velocity.rel_speed_diff,
            EdgeFeatureKey::VelocityInnovSpeedRatio => self.velocity.innov_speed_ratio,

            // Uncertainty
            EdgeFeatureKey::UncertaintyCovVelRatio => self.uncertainty.cov_vel_ratio(),

            // Photometry
            EdgeFeatureKey::PhotometryZFlux => self.photometry.z_flux,
            EdgeFeatureKey::PhotometryFluxStdRatio => self.photometry.flux_std_ratio,
            EdgeFeatureKey::PhotometryBandShared => self.photometry.band_shared,
        }
    }

    /// Return the `(name, value)` pair of the `idx`-th feature.
    ///
    /// The index refers to the canonical flat ordering defined by
    /// [`EDGE_FEATURE_KEYS`].
    ///
    /// Arguments
    /// ---------
    /// * `idx` – Flat feature index in `[0, len_flat())`.
    ///
    /// Return
    /// ------
    /// `Some((name, value))` if `idx` is in range, otherwise `None`.
    #[inline]
    pub fn flat_at_with_name(&self, idx: usize) -> Option<(&'static str, f64)> {
        let key = *EDGE_FEATURE_KEYS.get(idx)?;
        Some((key.path(), self.get(key)))
    }

    /// Return the value of the `idx`-th feature in canonical order.
    ///
    /// Arguments
    /// ---------
    /// * `idx` – Flat feature index in `[0, len_flat())`.
    ///
    /// Return
    /// ------
    /// `Some(value)` if `idx` is in range, otherwise `None`.
    #[inline]
    pub fn flat_at(&self, idx: usize) -> Option<f64> {
        let key = *EDGE_FEATURE_KEYS.get(idx)?;
        Some(self.get(key))
    }

    /// Iterate over all `(name, value)` feature pairs.
    ///
    /// This is the most convenient iterator for plotting and diagnostics:
    /// - yields stable names and values,
    /// - preserves canonical ordering,
    /// - performs no allocations.
    ///
    /// Return
    /// ------
    /// [`EdgeFeaturesFlatNameIter`] over `(path, value)` pairs.
    #[inline]
    pub fn iter_flat_with_name(&self) -> EdgeFeaturesFlatNameIter<'_> {
        EdgeFeaturesFlatNameIter { f: self, i: 0 }
    }

    /// Iterate over all feature values in canonical order.
    ///
    /// This iterator is preferred in ML export paths:
    /// - allocation-free,
    /// - stable ordering,
    /// - easy to feed into dense arrays / tensors.
    ///
    /// Return
    /// ------
    /// [`EdgeFeaturesIter`] over scalar values.
    #[inline]
    pub fn iter_flat(&self) -> EdgeFeaturesIter<'_> {
        EdgeFeaturesIter { f: self, i: 0 }
    }

    /// Collect all feature values into a `Vec<f64>` (canonical order).
    ///
    /// This is convenient for debugging and small-scale usage, but should be
    /// avoided in hot paths due to allocation.
    ///
    /// Return
    /// ------
    /// Vector of feature values in canonical order.
    #[inline]
    pub fn flat_values_vec(&self) -> Vec<f64> {
        EDGE_FEATURE_KEYS.iter().map(|&k| self.get(k)).collect()
    }

    /// Collect all `(name, value)` pairs into a `Vec`.
    ///
    /// This is convenient for debugging, logging, or ad-hoc reporting, but
    /// should be avoided in hot paths due to allocation.
    ///
    /// Return
    /// ------
    /// Vector of `(path, value)` pairs in canonical order.
    #[inline]
    pub fn flat_named_vec(&self) -> Vec<(&'static str, f64)> {
        EDGE_FEATURE_KEYS
            .iter()
            .map(|&k| (k.path(), self.get(k)))
            .collect()
    }
}

// -----------------------------------------------------------------------------
// Flat feature access helpers (field paths + iterator)
// -----------------------------------------------------------------------------
//
// This section provides ergonomic and *stable* access to all scalar leaf fields
// of [`EdgeFeatures`].
//
// Design goals
// ------------
// - Provide **type-safe** access (via [`EdgeFeatureKey`]) for core logic.
// - Provide **string-path** access (via `EdgeFeatureKey::path`) for configuration,
//   CLI tools, plotting scripts and interactive analysis.
// - Provide a **flat, canonical ordering** of all features for:
//   - ML dataset export,
//   - feature ablation studies,
//   - consistent plotting and diagnostics.
//
// The flat view is guaranteed to be:
// - allocation-free when iterating,
// - stable across versions unless explicitly changed,
// - consistent between keys, paths, and iterators.

/// Type-safe identifier for every scalar leaf feature in [`EdgeFeatures`].
///
/// Each variant corresponds to **exactly one scalar value** (a "leaf") in the
/// nested [`EdgeFeatures`] structure.
///
/// Rationale
/// ---------
/// This enum avoids *stringly-typed* feature access in core logic and ensures:
/// - compiler-checked exhaustiveness,
/// - safe refactors (renames break at compile time),
/// - a single source of truth for feature identity.
///
/// Guarantees
/// ----------
/// - Each variant maps to:
///   - a unique flat index ([`index`](Self::index)),
///   - a unique canonical string path ([`path`](Self::path)).
/// - The mapping is stable and consistent with [`EDGE_FEATURE_KEYS`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EdgeFeatureKey {
    // ---------------------------------------------------------------------
    // Position-related features
    // ---------------------------------------------------------------------
    /// χ² of the positional residual in the tangent plane.
    PositionChi2Pos,
    /// Log-transformed positional χ² (numerical stabilization).
    PositionLogChi2Pos,
    /// Normalized residual along the RA-like axis.
    PositionZDx,
    /// Normalized residual along the DEC-like axis.
    PositionZDy,
    /// Norm of the normalized residual vector.
    PositionZResidNorm,
    /// Residual projected along the motion direction.
    PositionZAlong,
    /// Residual projected orthogonally to the motion direction.
    PositionZCross,
    /// First Cholesky-whitened residual component.
    PositionCholZ1,
    /// Second Cholesky-whitened residual component.
    PositionCholZ2,
    /// Norm of the Cholesky-whitened residual.
    PositionCholZNorm,

    // ---------------------------------------------------------------------
    // Velocity-related features
    // ---------------------------------------------------------------------
    /// Cosine of the angle between predicted and observed velocity vectors.
    VelocityCosDthetaV,
    /// Relative speed difference between the two nodes.
    VelocityRelSpeedDiff,
    /// Innovation-to-expected speed ratio.
    VelocityInnovSpeedRatio,

    // ---------------------------------------------------------------------
    // Uncertainty-related features
    // ---------------------------------------------------------------------
    /// Ratio of velocity covariance traces.
    UncertaintyCovVelRatio,

    // ---------------------------------------------------------------------
    // Photometry-related features
    // ---------------------------------------------------------------------
    /// Flux difference expressed as a Z-score.
    PhotometryZFlux,
    /// Ratio of flux standard deviations.
    PhotometryFluxStdRatio,
    /// Indicator of shared photometric band.
    PhotometryBandShared,
}

impl EdgeFeatureKey {
    /// Return the **flat index** of this feature in canonical order.
    ///
    /// This index is used by:
    /// - flat iterators,
    /// - dense ML array exports,
    /// - column alignment between Rust and Python.
    ///
    /// Return
    /// ------
    /// Canonical flat index in `[0, EdgeFeatures::len_flat())`.
    ///
    /// Guarantees
    /// ----------
    /// - Indices are contiguous.
    /// - Ordering matches [`EDGE_FEATURE_KEYS`].
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            // Position (0..10)
            Self::PositionChi2Pos => 0,
            Self::PositionLogChi2Pos => 1,
            Self::PositionZDx => 2,
            Self::PositionZDy => 3,
            Self::PositionZResidNorm => 4,
            Self::PositionZAlong => 5,
            Self::PositionZCross => 6,
            Self::PositionCholZ1 => 7,
            Self::PositionCholZ2 => 8,
            Self::PositionCholZNorm => 9,

            // Velocity (10..13)
            Self::VelocityCosDthetaV => 10,
            Self::VelocityRelSpeedDiff => 11,
            Self::VelocityInnovSpeedRatio => 12,

            // Uncertainty (13..14)
            Self::UncertaintyCovVelRatio => 13,

            // Photometry (14..17)
            Self::PhotometryZFlux => 14,
            Self::PhotometryFluxStdRatio => 15,
            Self::PhotometryBandShared => 16,
        }
    }

    /// Return the **canonical string path** of this feature.
    ///
    /// These paths are intended for:
    /// - column names in exported datasets,
    /// - plotting labels,
    /// - CLI / YAML configuration.
    ///
    /// Return
    /// ------
    /// Canonical stable string path of the form `<group>.<field>`.
    ///
    /// Guarantees
    /// ----------
    /// The returned path is stable unless explicitly changed as a breaking update.
    #[inline]
    pub const fn path(self) -> &'static str {
        match self {
            // Position
            Self::PositionChi2Pos => "position.chi2_pos",
            Self::PositionLogChi2Pos => "position.log_chi2_pos",
            Self::PositionZDx => "position.z_dx",
            Self::PositionZDy => "position.z_dy",
            Self::PositionZResidNorm => "position.z_resid_norm",
            Self::PositionZAlong => "position.z_along",
            Self::PositionZCross => "position.z_cross",
            Self::PositionCholZ1 => "position.chol_z1",
            Self::PositionCholZ2 => "position.chol_z2",
            Self::PositionCholZNorm => "position.chol_z_norm",

            // Velocity
            Self::VelocityCosDthetaV => "velocity.cos_dtheta_v",
            Self::VelocityRelSpeedDiff => "velocity.rel_speed_diff",
            Self::VelocityInnovSpeedRatio => "velocity.innov_speed_ratio",

            // Uncertainty
            Self::UncertaintyCovVelRatio => "uncertainty.cov_vel_ratio",

            // Photometry
            Self::PhotometryZFlux => "photometry.z_flux",
            Self::PhotometryFluxStdRatio => "photometry.flux_std_ratio",
            Self::PhotometryBandShared => "photometry.band_shared",
        }
    }
}

/// Canonical ordered list of all scalar leaf features.
///
/// This constant defines the **single authoritative ordering** used by:
/// - flat iterators,
/// - index-based access,
/// - ML feature vectors,
/// - exported column names.
///
/// Changing this ordering is a **breaking change** for downstream consumers.
pub const EDGE_FEATURE_KEYS: [EdgeFeatureKey; 17] = [
    // Position
    EdgeFeatureKey::PositionChi2Pos,
    EdgeFeatureKey::PositionLogChi2Pos,
    EdgeFeatureKey::PositionZDx,
    EdgeFeatureKey::PositionZDy,
    EdgeFeatureKey::PositionZResidNorm,
    EdgeFeatureKey::PositionZAlong,
    EdgeFeatureKey::PositionZCross,
    EdgeFeatureKey::PositionCholZ1,
    EdgeFeatureKey::PositionCholZ2,
    EdgeFeatureKey::PositionCholZNorm,
    // Velocity
    EdgeFeatureKey::VelocityCosDthetaV,
    EdgeFeatureKey::VelocityRelSpeedDiff,
    EdgeFeatureKey::VelocityInnovSpeedRatio,
    // Uncertainty
    EdgeFeatureKey::UncertaintyCovVelRatio,
    // Photometry
    EdgeFeatureKey::PhotometryZFlux,
    EdgeFeatureKey::PhotometryFluxStdRatio,
    EdgeFeatureKey::PhotometryBandShared,
];

/// Iterator over `(name, value)` pairs of an [`EdgeFeatures`] instance.
///
/// This iterator:
/// - is exact-sized,
/// - preserves canonical ordering,
/// - performs no allocations.
#[derive(Clone, Debug)]
pub struct EdgeFeaturesFlatNameIter<'a> {
    f: &'a EdgeFeatures,
    i: usize,
}

impl<'a> Iterator for EdgeFeaturesFlatNameIter<'a> {
    type Item = (&'static str, f64);

    /// Return the next `(path, value)` pair in canonical order.
    ///
    /// Return
    /// ------
    /// `Some((path, value))` while features remain, otherwise `None`.
    ///
    /// Notes
    /// -----
    /// This method is allocation-free and delegates indexing to
    /// [`EdgeFeatures::flat_at_with_name`].
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let out = self.f.flat_at_with_name(self.i)?;
        self.i += 1;
        Some(out)
    }

    /// Provide an exact size hint for the remaining number of items.
    ///
    /// Return
    /// ------
    /// `(remaining, Some(remaining))` where `remaining` is the number of
    /// features left to iterate.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = EdgeFeatures::len_flat().saturating_sub(self.i);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EdgeFeaturesFlatNameIter<'a> {}

/// Iterator over feature values of an [`EdgeFeatures`] instance.
///
/// This iterator:
/// - is exact-sized,
/// - allocation-free,
/// - suitable for dense ML pipelines.
#[derive(Clone, Debug)]
pub struct EdgeFeaturesIter<'a> {
    f: &'a EdgeFeatures,
    i: usize,
}

impl<'a> Iterator for EdgeFeaturesIter<'a> {
    type Item = f64;

    /// Return the next feature value in canonical order.
    ///
    /// Return
    /// ------
    /// `Some(value)` while features remain, otherwise `None`.
    ///
    /// Notes
    /// -----
    /// This is the preferred iterator for ML export:
    /// - stable ordering,
    /// - no allocations,
    /// - easy to collect into arrays/tensors when needed.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let out = self.f.flat_at(self.i)?;
        self.i += 1;
        Some(out)
    }

    /// Provide an exact size hint for the remaining number of items.
    ///
    /// Return
    /// ------
    /// `(remaining, Some(remaining))` where `remaining` is the number of
    /// features left to iterate.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = EdgeFeatures::len_flat().saturating_sub(self.i);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EdgeFeaturesIter<'a> {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Alert, AlertKey,
        night_id::NightId,
        seeding::{SeedNode, store::SeedStore},
    };
    use proptest::prelude::*;
    use std::sync::Arc;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Minimal alert factory.
    fn make_alert(id: u64, night: u32, mjd: f64, ra: f64, dec: f64, band: u8, flux: f64) -> Alert {
        let arcsec = std::f64::consts::PI / (180.0 * 3600.0);
        Alert {
            key: AlertKey {
                night_id: NightId::new(night),
                dia_source_id: id,
            },
            ra,
            ra_err: arcsec,
            dec,
            dec_err: arcsec,
            mjd_tt: mjd,
            flux,
            flux_err: flux * 0.05,
            band,
            observer_mpc_code: Arc::new("500".into()),
        }
    }

    /// Build a SeedNode from two alerts using the public `from_pair` constructor.
    fn make_seed_from_pair(
        store: &mut SeedStore,
        night: u32,
        id_a: u64,
        id_b: u64,
        mjd_a: f64,
        ra: f64,
        dec: f64,
        vx_rad_day: f64, // approx angular speed in RA
        band: u8,
        flux: f64,
    ) -> SeedNode {
        // dt = 0.5 h intra-night
        let dt = 0.5 / 24.0;
        let ra_b = ra + vx_rad_day * dt;
        let a = make_alert(id_a, night, mjd_a, ra, dec, band, flux);
        let b = make_alert(id_b, night, mjd_a + dt, ra_b, dec, band, flux);
        SeedNode::from_pair(store, NightId::new(night), &a, &b, None)
            .expect("from_pair should succeed for simple test alerts")
    }

    /// Two seeds separated by ~1 day with consistent kinematics.
    ///
    /// Uses band `1` (shared across both seeds).
    fn seed_pair_consistent() -> (SeedNode, SeedNode) {
        let mut store = SeedStore::new();
        let ra = 0.5_f64;
        let dec = 0.1_f64;
        let vx = 3e-3; // ~0.17 °/day

        let from = make_seed_from_pair(&mut store, 1, 10, 11, 60000.0, ra, dec, vx, 1, 1000.0);
        // To-seed: starts where from ends after ~1 day, same velocity
        let to = make_seed_from_pair(
            &mut store,
            2,
            20,
            21,
            60001.0,
            ra + vx * 1.0,
            dec,
            vx,
            1,
            1000.0,
        );
        (from, to)
    }

    /// Construct an [`EdgeFeatures`] directly from raw scalars, bypassing `SeedNode`.
    ///
    /// Useful for unit-testing `kinematic_log_likelihood_cost` in isolation.
    fn make_features(
        chi2_pos: f64,
        chi2_vel: f64,
        z_flux: f64,
        flux_std_ratio: f64,
        band_shared: f64,
    ) -> EdgeFeatures {
        use crate::graph::edge::{
            photometry_features::EdgePhotometryFeatures, position_features::EdgePositionFeatures,
            uncertainty_features::EdgeUncertaintyFeatures, velocity_features::EdgeVelocityFeatures,
        };
        EdgeFeatures {
            position: EdgePositionFeatures {
                chi2_pos,
                log_chi2_pos: (chi2_pos + 1e-12).ln(),
                z_dx: 0.0,
                z_dy: 0.0,
                z_resid_norm: chi2_pos.sqrt().max(0.0),
                z_along: 0.0,
                z_cross: 0.0,
                chol_z1: 0.0,
                chol_z2: 0.0,
                chol_z_norm: chi2_pos.sqrt().max(0.0),
            },
            velocity: EdgeVelocityFeatures {
                cos_dtheta_v: 1.0,
                rel_speed_diff: 0.0,
                innov_speed_ratio: 0.0,
                chi2_vel,
                log_chi2_vel: (chi2_vel + 1e-12).ln(),
            },
            uncertainty: EdgeUncertaintyFeatures(1.0),
            photometry: EdgePhotometryFeatures {
                z_flux,
                flux_std_ratio,
                band_shared,
            },
        }
    }

    // =========================================================================
    // Unit tests – kinematic_log_likelihood_cost
    // =========================================================================

    /// Regression: band_shared = 1 with near-zero other terms previously gave < 0.
    #[test]
    fn cost_positive_band_shared_near_zero_other_terms() {
        let f = make_features(0.0, 0.0, 0.0, 1.0, 1.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost > 0.0,
            "cost must be > 0 when band_shared=1, got {cost}"
        );
    }

    #[test]
    fn cost_positive_band_not_shared() {
        let f = make_features(0.0, 0.0, 0.0, 1.0, 0.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost > 0.0,
            "cost must be > 0 when band_shared=0, got {cost}"
        );
    }

    /// Sharing a band must give a strictly lower (better) cost than not sharing.
    #[test]
    fn cost_band_shared_lower_than_not_shared() {
        let chi2 = 2.0;
        let shared = make_features(chi2, chi2, 0.5, 1.0, 1.0);
        let not_shared = make_features(chi2, chi2, 0.5, 1.0, 0.0);
        assert!(
            shared.kinematic_log_likelihood_cost() < not_shared.kinematic_log_likelihood_cost(),
            "shared-band cost {} should be < not-shared cost {}",
            shared.kinematic_log_likelihood_cost(),
            not_shared.kinematic_log_likelihood_cost()
        );
    }

    /// When band_shared = 1, band_term = 0.
    /// The cost must equal exactly 0.5*(chi2_pos + chi2_vel) + 0.5*z² + 0.5*ln²(eps).
    #[test]
    fn cost_formula_band_shared() {
        let eps = 1e-12_f64;
        // flux_std_ratio=0 → ln_term = 0.5 * ln(eps)²
        let ln_ratio_sq = (0_f64 + eps).ln().powi(2);
        let expected = 0.5 * ln_ratio_sq; // chi2=0, z_flux=0, band_term=0
        let f = make_features(0.0, 0.0, 0.0, 0.0, 1.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            (cost - expected).abs() < 1e-10,
            "cost {cost} ≠ expected {expected}"
        );
    }

    /// When band not shared, band_term = -ln(eps_band) ≈ +6.907.
    #[test]
    fn cost_formula_band_not_shared() {
        let eps = 1e-12_f64;
        let eps_band = 1e-3_f64;
        let ln_ratio_sq = (0_f64 + eps).ln().powi(2);
        let expected = 0.5 * ln_ratio_sq + (-eps_band.ln());
        let f = make_features(0.0, 0.0, 0.0, 0.0, 0.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            (cost - expected).abs() < 1e-10,
            "cost {cost} ≠ expected {expected}"
        );
    }

    /// Large chi2 must not produce NaN or Inf.
    #[test]
    fn cost_finite_for_large_chi2() {
        let f = make_features(1e8, 1e8, 100.0, 1e6, 0.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost.is_finite(),
            "cost should be finite for large chi2, got {cost}"
        );
        assert!(cost > 0.0, "cost should be > 0 for large chi2, got {cost}");
    }

    /// NaN inputs are sanitized by `finite_or_zero` – cost must remain finite.
    #[test]
    fn cost_finite_for_nan_inputs() {
        let f = make_features(f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.5);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost.is_finite(),
            "cost should be finite even for NaN inputs, got {cost}"
        );
    }

    /// Inf inputs are sanitized to 0 – cost must remain finite.
    #[test]
    fn cost_finite_for_inf_inputs() {
        let f = make_features(f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0, 1.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost.is_finite(),
            "cost should be finite even for Inf inputs, got {cost}"
        );
    }

    /// Negative chi2 (numeric drift) is clamped to 0 internally.
    #[test]
    fn cost_handles_negative_chi2() {
        let f = make_features(-1.0, -5.0, 0.0, 1.0, 1.0);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost > 0.0,
            "cost must be > 0 with negative chi2 inputs, got {cost}"
        );
        assert!(cost.is_finite());
    }

    // =========================================================================
    // Unit tests – EdgeFeatureKey (index stability & path uniqueness)
    // =========================================================================

    #[test]
    fn len_flat_is_17() {
        assert_eq!(EdgeFeatures::len_flat(), 17);
    }

    #[test]
    fn edge_feature_keys_array_len_is_17() {
        assert_eq!(EDGE_FEATURE_KEYS.len(), 17);
    }

    /// Each key's `index()` must match its position in `EDGE_FEATURE_KEYS`.
    #[test]
    fn key_indices_match_position_in_canonical_array() {
        for (pos, &key) in EDGE_FEATURE_KEYS.iter().enumerate() {
            assert_eq!(
                key.index(),
                pos,
                "{key:?}.index() = {} but its position is {pos}",
                key.index()
            );
        }
    }

    /// All string paths must be unique.
    #[test]
    fn key_paths_are_unique() {
        let paths: std::collections::HashSet<&'static str> =
            EDGE_FEATURE_KEYS.iter().map(|k| k.path()).collect();
        assert_eq!(
            paths.len(),
            EDGE_FEATURE_KEYS.len(),
            "duplicate paths in EDGE_FEATURE_KEYS"
        );
    }

    /// All indices must be unique.
    #[test]
    fn key_indices_are_unique() {
        let indices: std::collections::HashSet<usize> =
            EDGE_FEATURE_KEYS.iter().map(|k| k.index()).collect();
        assert_eq!(
            indices.len(),
            EDGE_FEATURE_KEYS.len(),
            "duplicate indices in EDGE_FEATURE_KEYS"
        );
    }

    /// `flat_names()` returns exactly `len_flat()` items.
    #[test]
    fn flat_names_length() {
        assert_eq!(EdgeFeatures::flat_names().count(), 17);
    }

    // =========================================================================
    // Unit tests – compute_features on real SeedNodes
    // =========================================================================

    #[test]
    fn compute_features_all_finite() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        for (name, val) in f.iter_flat_with_name() {
            assert!(val.is_finite(), "feature '{name}' is not finite: {val}");
        }
    }

    #[test]
    fn iter_flat_length_equals_len_flat() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        assert_eq!(f.iter_flat().count(), EdgeFeatures::len_flat());
    }

    #[test]
    fn exact_size_iterator_consistent() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        let mut it = f.iter_flat();
        assert_eq!(it.len(), 17);
        it.next();
        assert_eq!(it.len(), 16);
    }

    #[test]
    fn flat_values_vec_length() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        assert_eq!(f.flat_values_vec().len(), 17);
    }

    /// `get(key)` must match the value returned by `iter_flat` at `key.index()`.
    #[test]
    fn get_consistent_with_iter_flat() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        let vals: Vec<f64> = f.iter_flat().collect();
        for &key in &EDGE_FEATURE_KEYS {
            let via_get = f.get(key);
            let via_iter = vals[key.index()];
            assert_eq!(
                via_get,
                via_iter,
                "get({key:?}) = {via_get}, iter[{}] = {via_iter}",
                key.index()
            );
        }
    }

    /// `flat_at` must return `None` for indices ≥ `len_flat()`.
    #[test]
    fn flat_at_out_of_bounds_returns_none() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        assert!(f.flat_at(17).is_none());
        assert!(f.flat_at(100).is_none());
    }

    /// `iter_flat_with_name` and `flat_named_vec` must agree.
    #[test]
    fn iter_flat_with_name_matches_flat_named_vec() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        let via_iter: Vec<_> = f.iter_flat_with_name().collect();
        let via_vec = f.flat_named_vec();
        assert_eq!(via_iter, via_vec);
    }

    /// `compute_features` is deterministic.
    #[test]
    fn compute_features_is_deterministic() {
        let (from, to) = seed_pair_consistent();
        let f1 = EdgeFeatures::compute_features(&from, &to);
        let f2 = EdgeFeatures::compute_features(&from, &to);
        assert_eq!(f1.flat_values_vec(), f2.flat_values_vec());
    }

    /// `kinematic_log_likelihood_cost` is finite and > 0 on a real consistent pair.
    #[test]
    fn cost_positive_finite_on_consistent_pair() {
        let (from, to) = seed_pair_consistent();
        let f = EdgeFeatures::compute_features(&from, &to);
        let cost = f.kinematic_log_likelihood_cost();
        assert!(
            cost.is_finite(),
            "cost not finite on consistent seed pair: {cost}"
        );
        assert!(cost > 0.0, "cost not > 0 on consistent seed pair: {cost}");
    }

    /// When both seeds use different bands, the cost must be strictly higher
    /// than when they share a band (all other parameters equal).
    #[test]
    fn cost_higher_when_bands_differ() {
        let mut store = SeedStore::new();
        let ra = 0.5;
        let dec = 0.1;
        let vx = 3e-3;
        // Shared band
        let from_shared = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, 1000.0);
        let to_shared =
            make_seed_from_pair(&mut store, 2, 2, 3, 60001.0, ra + vx, dec, vx, 1, 1000.0);
        // Different bands (band 1 vs band 2)
        let from_diff = make_seed_from_pair(&mut store, 1, 4, 5, 60000.0, ra, dec, vx, 1, 1000.0);
        let to_diff =
            make_seed_from_pair(&mut store, 2, 6, 7, 60001.0, ra + vx, dec, vx, 2, 1000.0);

        let cost_shared = EdgeFeatures::compute_features(&from_shared, &to_shared)
            .kinematic_log_likelihood_cost();
        let cost_diff =
            EdgeFeatures::compute_features(&from_diff, &to_diff).kinematic_log_likelihood_cost();

        assert!(
            cost_diff > cost_shared,
            "cost with different bands ({cost_diff}) should be > shared ({cost_shared})"
        );
    }

    // =========================================================================
    // Proptest – kinematic_log_likelihood_cost robustness
    // =========================================================================

    proptest! {
        /// For any combination of well-formed (or slightly degenerate) feature
        /// values, the cost must always be finite and strictly > 0.
        #[test]
        fn prop_cost_always_finite_positive(
            chi2_pos       in  0.0f64..1e6,
            chi2_vel       in  0.0f64..1e6,
            z_flux         in -100.0f64..100.0,
            flux_std_ratio in -1e4f64..1e4,
            band           in 0u8..2,
        ) {
            let f = make_features(chi2_pos, chi2_vel, z_flux, flux_std_ratio, band as f64);
            let cost = f.kinematic_log_likelihood_cost();
            prop_assert!(cost.is_finite(), "cost not finite: {cost}");
            prop_assert!(cost > 0.0,       "cost not > 0: {cost}");
        }

        /// Sharing a band must *never* raise the cost compared to not sharing,
        /// all else being equal.
        #[test]
        fn prop_band_shared_never_raises_cost(
            chi2_pos       in 0.0f64..1e4,
            chi2_vel       in 0.0f64..1e4,
            z_flux         in -50.0f64..50.0,
            flux_std_ratio in 0.0f64..100.0,
        ) {
            let shared     = make_features(chi2_pos, chi2_vel, z_flux, flux_std_ratio, 1.0);
            let not_shared = make_features(chi2_pos, chi2_vel, z_flux, flux_std_ratio, 0.0);
            prop_assert!(
                shared.kinematic_log_likelihood_cost() <= not_shared.kinematic_log_likelihood_cost(),
                "shared {} > not-shared {}",
                shared.kinematic_log_likelihood_cost(),
                not_shared.kinematic_log_likelihood_cost()
            );
        }

        /// A larger chi2 must (weakly) increase the cost.
        #[test]
        fn prop_larger_chi2_increases_cost(
            base  in 0.0f64..1e4,
            delta in 0.0f64..1e4,
        ) {
            let low  = make_features(base, base, 0.0, 1.0, 1.0);
            let high = make_features(base + delta, base + delta, 0.0, 1.0, 1.0);
            prop_assert!(
                high.kinematic_log_likelihood_cost() >= low.kinematic_log_likelihood_cost(),
                "higher chi2 gave lower cost: {} vs {}",
                high.kinematic_log_likelihood_cost(),
                low.kinematic_log_likelihood_cost()
            );
        }

        /// `get` must agree with `iter_flat` for all 17 keys, on physically
        /// plausible seeds built from arbitrary (but valid) kinematic parameters.
        #[test]
        fn prop_get_consistent_with_iter(
            epoch_offset   in 0.1f64..10.0,
            vx             in -1e-2f64..1e-2,
            flux_mean      in 100.0f64..5000.0,
            flux_delta_pct in -0.3f64..0.3,    // flux variation between nights
        ) {
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let flux_to = flux_mean * (1.0 + flux_delta_pct);
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_to.max(1.0));
            let f = EdgeFeatures::compute_features(&from, &to);
            let vals: Vec<f64> = f.iter_flat().collect();
            for &key in &EDGE_FEATURE_KEYS {
                prop_assert_eq!(f.get(key), vals[key.index()]);
            }
        }

        /// All feature values produced by `compute_features` must be finite for
        /// physically reasonable seeds.
        #[test]
        fn prop_compute_features_all_finite(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
        ) {
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let f = EdgeFeatures::compute_features(&from, &to);
            for (name, val) in f.iter_flat_with_name() {
                prop_assert!(val.is_finite(), "feature '{name}' not finite: {val}");
            }
        }
    }
}

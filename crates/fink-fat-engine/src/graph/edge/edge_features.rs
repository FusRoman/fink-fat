// -----------------------------------------------------------------------------
// Cadence-robust edge features (structured)
// -----------------------------------------------------------------------------
//
// This module defines a cadence-robust feature set for **inter-night edges**
// (directed links) between two `SeedNode` objects.
//
// Why "cadence-robust"?
// ---------------------
// Survey cadence affects the distribution of `dt` (time gaps) between nights.
// Any feature that depends strongly on `dt` (e.g., raw residual distances) will
// not generalize well across cadences (ZTF → LSST, rolling cadence, weather gaps).
//
// The feature design here favors:
// - normalized innovations (Mahalanobis / z-scores) rather than raw distances,
// - along-track / cross-track decomposition rather than axis-aligned residuals,
// - angular and relative quantities (dimensionless ratios),
// - robust numerical guards (floors, epsilons, finite checks).
//
// Design philosophy
// -----------------
// - `FeatureCore` computes expensive shared intermediates once per edge.
// - Public feature families (`position`, `velocity`, `uncertainty`, `photometry`)
//   are stable, readable containers used for ML export (Parquet / Arrow / ONNX).
// - Flat feature access is provided via:
//   - `EdgeFeatureKey` (type-safe, compile-time checked),
//   - canonical string paths (`EdgeFeatureKey::path()`),
//   - canonical ordering (`EDGE_FEATURE_KEYS`) and allocation-free iterators.
//
// Any change to the canonical ordering or to the string paths is a breaking
// change for downstream consumers (Python training code, ONNX export, plots, etc.).
//
// -----------------------------------------------------------------------------

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
    /// on **additive edge costs**. If per-edge costs approximate `-log p(edge)`,
    /// summing them along a path approximates `-log p(trajectory)` (up to constants).
    ///
    /// The cost is designed to be:
    /// - physically motivated (Gaussian innovations on the tangent plane),
    /// - mostly parameter-free (only numerical epsilons),
    /// - additive along a trajectory (sum of per-edge costs),
    /// - stable (finite, avoids NaNs/Infs).
    ///
    /// Definition (up to additive constants)
    /// ------------------------------------
    /// ```text
    /// cost =
    ///   0.5 * chi2_pos
    /// + 0.5 * chi2_vel
    /// + 0.5 * z_flux^2
    /// + 0.5 * log(flux_std_ratio)^2
    /// - log(eps_band + band_shared)
    /// ```
    ///
    /// Term interpretation
    /// -------------------
    /// - `chi2_pos`: penalizes geometric inconsistency normalized by uncertainties.
    /// - `chi2_vel`: penalizes velocity inconsistency normalized by uncertainties.
    /// - `z_flux^2`: penalizes photometric inconsistency (scale-free).
    /// - `log(flux_std_ratio)^2`: penalizes large changes in flux scatter (quality proxy).
    /// - `-log(eps_band + band_shared)`: encourages band overlap when possible.
    ///
    /// Return
    /// ------
    /// A finite `f64` cost (lower is better) suitable for additive solvers.
    ///
    /// Notes
    /// -----
    /// - `band_shared` is expected to be in `{0,1}` but we clamp defensively.
    /// - `flux_std_ratio` can be 0 if undefined; we guard with `eps`.
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

        // Encourage band overlap; clamp ensures the input stays in [0,1].
        let band_shared = self.photometry.band_shared.clamp(0.0, 1.0);
        let band_term = -safe_ln(eps_band + band_shared);

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

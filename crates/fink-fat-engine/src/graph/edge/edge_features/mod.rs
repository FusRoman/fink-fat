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
//! Cost computation
//! ----------------
//! The solver-facing scalar cost is computed by [`EdgeFeatures::compute_cost`] in
//! three independent steps:
//!
//! 1. **χ² extraction** — `FeatureCore` covariances are optionally inflated with
//!    Singer CWNA (Continuous White Noise Acceleration) process noise (controlled by `sigma_q`).
//! 2. **Kinematic loss** — maps $(\chi^2_{\mathrm{pos}},\,\chi^2_{\mathrm{vel}})$
//!    to a scalar via the chosen
//!    [`CostVariant`]:
//!    - `GaussianChi2` / `KinematicLogLikelihood` — Gaussian $\frac{1}{2}\chi^2$.
//!    - `SingerCwna` — same Gaussian loss on CWNA-inflated covariances.
//!    - `RobustCauchy` — logarithmic saturation $\ln(1 + \chi^2/\sigma)$.
//!    - `RobustStudentT` — Student-t $\frac{\nu+1}{2}\ln(1 + \chi^2/\nu)$.
//! 3. **Photometry penalty** — variant-independent; a flux z-score term, a
//!    flux-scatter log-ratio term, and a band-sharing penalty.
//!
//! ML features stored in [`EdgeFeatures`] are **not** affected by the cost variant;
//! ONNX ranking always uses baseline covariances for stable feature representations.
//!
//! -----------------------------------------------------------------------------

pub mod feature_core;
pub mod photometry_features;
pub mod position_features;
pub mod seed_features;
pub mod uncertainty_features;
pub mod velocity_features;

use crate::{
    astro_math::safe_ln,
    engine_config::edge_config::{CostConfig, CostVariant},
    graph::edge::edge_features::{
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
    /// 1. Build a `FeatureCore` once (propagation, innovation, covariances, guards).
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
    /// - Numerical stability is primarily handled in `FeatureCore`.
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

    /// Compute a scalar edge cost from raw seed nodes using a configurable cost function.
    ///
    /// The total cost is $c = c_{\mathrm{kin}} + c_{\mathrm{phot}}$ where:
    ///
    /// - $c_{\mathrm{kin}}$ depends on the chosen
    ///   [`CostVariant`]
    ///   (Gaussian ½χ², Cauchy, or Student-t) and optionally on CWNA (Continuous White Noise Acceleration)
    ///   covariance inflation (`sigma_q`).
    /// - $c_{\mathrm{phot}}$ is variant-independent (flux z-score, flux-scatter
    ///   log-ratio, band-sharing penalty).
    ///
    /// The computation proceeds in three steps:
    ///
    /// 1. **χ² extraction** (`chi2_with_cwna`) — builds `FeatureCore` once;
    ///    inflates the innovation covariance with CWNA process noise when
    ///    `sigma_q > 0`.
    /// 2. **Kinematic loss** (`kinematic_loss`) — maps
    ///    $(\chi^2_{\mathrm{pos}},\,\chi^2_{\mathrm{vel}})$ to a scalar
    ///    via the chosen variant.
    /// 3. **Photometry penalty** (`photometry_cost`) — added unconditionally.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node (older epoch).
    /// * `to`   – Target seed node (newer epoch).
    /// * `cfg`  – Cost-function configuration (variant + hyperparameters).
    ///
    /// Return
    /// ------
    /// A finite, strictly-positive `f64` cost suitable for graph solvers.
    ///
    /// Notes
    /// -----
    /// - `KinematicLogLikelihood` is a backward-compatibility alias for `GaussianChi2`
    ///   with `sigma_q = 0`; it always uses baseline covariances regardless of the
    ///   `sigma_q` field.
    /// - ML features in [`EdgeFeatures`] are **not affected** by this function;
    ///   ONNX ranking always uses baseline covariances for stable feature
    ///   representations.
    /// - The result is clamped to $[\varepsilon_{f64},\,+\infty)$ to guarantee
    ///   strict positivity for graph solvers.
    #[inline]
    pub fn compute_cost(from: &SeedNode, to: &SeedNode, cfg: &CostConfig) -> f64 {
        // Build shared intermediates once (propagation + projection are expensive;
        // they are only paid once regardless of the chosen variant).
        let core = FeatureCore::from_nodes(from, to);

        let (chi2_pos, chi2_vel) = from.chi2_with_cwna(&core, to, cfg);
        let kin_cost = Self::kinematic_loss(chi2_pos, chi2_vel, cfg);
        let phot_cost = Self::photometry_cost(from, to);

        FeatureCore::finite_or_zero(kin_cost + phot_cost).max(f64::EPSILON)
    }

    /// Extract positional and velocity $\chi^2$ values, optionally inflated with
    /// CWNA (Continuous White Noise Acceleration) process noise.
    ///
    /// Positional $\chi^2$
    /// -------------------
    /// The positional term uses a **spherical residual** $d$ (see
    /// [`FeatureCore::r_sph`]) instead of the 2-D gnomonic Mahalanobis:
    ///
    /// $$\chi^{2}_{\mathrm{pos}} = \frac{d^{2}}{S_{\mathrm{scalar}}}$$
    ///
    /// where $S_{\mathrm{scalar}} = \operatorname{tr}(\mathbf{S}_{\mathrm{pos}}) / 2$.
    ///
    /// Motivation: the gnomonic residual $\|\mathbf{r}\|$ diverges when the
    /// tangent-plane denominator $\cos c \to 0$ (seed centres separated by
    /// $\gtrsim 45^\circ$), leading to $\chi^{2}_{\mathrm{pos}} \sim 10^{20}$
    /// for otherwise valid edges. The great-circle distance
    /// $d \in [0, \pi]$ is bounded and well-defined for any separation.
    ///
    /// Velocity $\chi^2$
    /// -----------------
    /// The velocity term uses the full 2-D Mahalanobis distance on the
    /// tangent-plane velocity innovation $\delta \mathbf{v}$:
    ///
    /// $$\chi^{2}_{\mathrm{vel}} = \delta \mathbf{v}^{\top}\, \mathbf{S}_{\mathrm{vel}}^{-1}\, \delta \mathbf{v}$$
    ///
    /// Velocity residuals do not suffer from the gnomonic blow-up, so the
    /// standard 2-D form is kept.
    ///
    /// Behavior (two modes)
    /// --------------------
    /// Controlled by `cfg.variant` and `cfg.sigma_q`:
    ///
    /// - If `sigma_q == 0.0` (or `variant == KinematicLogLikelihood`):
    ///   - returns the cached spherical positional $\chi^2$ from `core`,
    ///   - returns the cached velocity $\chi^2$ from `core`,
    ///   - no extra linear algebra is performed.
    ///
    /// - If `sigma_q > 0.0`:
    ///   - recomputes the two $2 \times 2$ innovation covariances with the CWNA
    ///     diagonal term,
    ///   - applies the spherical formula for position,
    ///   - applies the 2-D Mahalanobis form for velocity.
    ///
    /// Propagation and tangent-plane projection are **not** repeated in either
    /// mode — only the covariance inflation and the final $\chi^2$ reduction.
    ///
    /// Arguments
    /// ---------
    /// * `core` – Shared edge intermediates (cached `r_sph`, `s_pos_scalar`,
    ///   `dv`, `chi2_vel`, `dt`, `dt_sq`).
    /// * `from` – Source seed node (provides the measurement covariance in the
    ///   CWNA path).
    /// * `to`   – Target seed node (provides the measurement covariance in the
    ///   CWNA path).
    /// * `cfg`  – Cost configuration; `variant` and `sigma_q` are read here.
    ///
    /// Return
    /// ------
    /// `(chi2_pos, chi2_vel)` — positional and velocity $\chi^2$ values,
    /// guaranteed finite and non-negative.
    #[inline]
    fn chi2_with_cwna(
        core: &FeatureCore,
        from: &SeedNode,
        to: &SeedNode,
        cfg: &CostConfig,
    ) -> (f64, f64) {
        // KinematicLogLikelihood is a baseline alias: always uses sigma_q = 0.
        let sigma_q = match cfg.variant {
            CostVariant::KinematicLogLikelihood => 0.0,
            _ => cfg.sigma_q,
        };

        if sigma_q == 0.0 {
            // Fast path: reuse pre-computed spherical chi2_pos and cached chi2_vel.
            // chi2_pos_sph = r_sph² / s_pos_scalar   (great-circle, no blow-up)
            let chi2_p =
                FeatureCore::finite_or_zero((core.r_sph * core.r_sph / core.s_pos_scalar).max(0.0));
            return (chi2_p, core.chi2_vel);
        }

        // CWNA path: inflate covariances with Singer process noise and recompute.
        //
        // Positional chi2 uses the spherical residual r_sph with an isotropic
        // scalar S (S_scalar = tr(S_cwna) / 2).
        let s_pos = from.innovation_cov_pos_cwna(to, core.dt, core.dt_sq, sigma_q);
        let s_pos_scalar = ((s_pos.xx + s_pos.yy).max(FeatureCore::FLOOR)) / 2.0;
        let chi2_p = FeatureCore::finite_or_zero((core.r_sph * core.r_sph / s_pos_scalar).max(0.0));

        // Velocity chi2: full 2-D Mahalanobis on the velocity innovation.
        let s_vel = from.innovation_cov_vel_cwna(to, core.dt, sigma_q);
        let chi2_v =
            FeatureCore::finite_or_zero(s_vel.mahalanobis_sq(core.dv).unwrap_or(0.0).max(0.0));

        (chi2_p, chi2_v)
    }

    /// Map positional and velocity $\chi^2$ to a scalar kinematic loss.
    ///
    /// Each loss is applied symmetrically to $\chi^2_{\mathrm{pos}}$ and
    /// $\chi^2_{\mathrm{vel}}$:
    ///
    /// - `KinematicLogLikelihood` / `GaussianChi2` / `SingerCwna`:
    ///   $c = \frac{1}{2}(\chi^2_{\mathrm{pos}} + \chi^2_{\mathrm{vel}})$.
    /// - `RobustCauchy`:
    ///   $c = \ln(1 + \chi^2_{\mathrm{pos}} / \sigma) + \ln(1 + \chi^2_{\mathrm{vel}} / \sigma)$
    ///   where $\sigma$ = `cauchy_scale`.
    /// - `RobustStudentT`:
    ///   $c = \frac{\nu+1}{2}\bigl[\ln(1+\chi^2_{\mathrm{pos}}/\nu) + \ln(1+\chi^2_{\mathrm{vel}}/\nu)\bigr]$
    ///   where $\nu$ = `student_nu`.
    ///
    /// Arguments
    /// ---------
    /// * `chi2_pos` – Positional Mahalanobis distance $\chi^2_{\mathrm{pos}}$.
    /// * `chi2_vel` – Velocity Mahalanobis distance $\chi^2_{\mathrm{vel}}$.
    /// * `cfg`      – Cost configuration; `variant`, `cauchy_scale`, and `student_nu`
    ///   are read here.
    ///
    /// Return
    /// ------
    /// Scalar kinematic loss (finite, ≥ 0).
    #[inline]
    fn kinematic_loss(chi2_pos: f64, chi2_vel: f64, cfg: &CostConfig) -> f64 {
        match cfg.variant {
            // KinematicLogLikelihood is a backward-compat alias for GaussianChi2
            // (chi² was already computed with sigma_q=0 in `chi2_with_cwna`).
            CostVariant::KinematicLogLikelihood
            | CostVariant::GaussianChi2
            | CostVariant::SingerCwna => 0.5 * (chi2_pos + chi2_vel),

            // Cauchy: ρ(χ²) = ln(1 + χ²/scale).  Grows logarithmically — bounded
            // penalty for large residuals or wide-gap mismatches.
            CostVariant::RobustCauchy => {
                let scale = cfg.cauchy_scale.max(1e-9);
                (1.0 + chi2_pos / scale).ln() + (1.0 + chi2_vel / scale).ln()
            }

            // Student-t: ρ(χ²) = (ν+1)/2 · ln(1 + χ²/ν).
            // ν=1 recovers Cauchy; ν→∞ converges to Gaussian.
            CostVariant::RobustStudentT => {
                let nu = cfg.student_nu.max(1e-9);
                let half_nu1 = 0.5 * (nu + 1.0);
                half_nu1 * (1.0 + chi2_pos / nu).ln() + half_nu1 * (1.0 + chi2_vel / nu).ln()
            }
        }
    }

    /// Compute the variant-independent photometry penalty.
    ///
    /// The penalty is:
    ///
    /// $$\begin{align} c_{\mathrm{phot}} &= \frac{1}{2} z_{\mathrm{flux}}^{2} + \frac{1}{2}\bigl[\ln(|r_{\sigma}| + \varepsilon)\bigr]^{2} + b_{\mathrm{band}} \end{align}$$
    ///
    /// where $b_{\mathrm{band}} = 0$ when the two seeds share a photometric band,
    /// and $b_{\mathrm{band}} = -\ln(\varepsilon_{\mathrm{band}}) \approx 6.9$ otherwise.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node.
    /// * `to`   – Target seed node.
    ///
    /// Return
    /// ------
    /// Scalar photometry penalty (finite, ≥ 0).
    ///
    /// Notes
    /// -----
    /// - Added to the kinematic loss unconditionally for all
    ///   [`CostVariant`](crate::engine_config::edge_config::CostVariant) choices.
    /// - [`EdgePhotometryFeatures`] is recomputed internally; the caller does not
    ///   need to provide a pre-built feature struct.
    #[inline]
    fn photometry_cost(from: &SeedNode, to: &SeedNode) -> f64 {
        let eps = 1e-12_f64;
        let eps_band = 1e-3_f64;
        let phot = EdgePhotometryFeatures::photometry_features(from, to);
        let ln_ratio = safe_ln(phot.flux_std_ratio.abs() + eps);
        let band_term = if phot.band_shared {
            0.0
        } else {
            -safe_ln(eps_band)
        };
        0.5 * phot.z_flux * phot.z_flux + 0.5 * ln_ratio * ln_ratio + band_term
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
            EdgeFeatureKey::PhotometryBandShared => {
                if self.photometry.band_shared {
                    1.0
                } else {
                    0.0
                }
            }
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
mod edge_feature_tests {
    use super::*;
    use crate::{
        astro_math::arcsec_to_rad,
        night_id::NightId,
        seeding::{SeedNode, store::SeedStore},
    };
    use photom::{
        coordinates::equatorial::EquCoord,
        observation_dataset::observation::Observation,
        photometry::{Filter, Photometry as PhotomPhotometry},
    };
    use proptest::prelude::*;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Minimal observation factory.
    fn make_obs(id: u64, mjd: f64, ra: f64, dec: f64, band: u8, flux: f64) -> Observation {
        let pos_err = arcsec_to_rad(1.0);
        let equ_coord = EquCoord::new(ra, pos_err, dec, pos_err);
        let photometry = PhotomPhotometry {
            magnitude: flux,
            error: flux * 0.05,
            filter: Filter::Int(band as u32),
        };
        Observation::new(id, equ_coord, photometry, mjd, None)
    }

    /// Build a SeedNode from two observations using the public `from_pair` constructor.
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
        let a = make_obs(id_a, mjd_a, ra, dec, band, flux);
        let b = make_obs(id_b, mjd_a + dt, ra_b, dec, band, flux);
        SeedNode::from_pair(store, NightId::new(night), &a, &b, None)
            .expect("from_pair should succeed for simple test observations")
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

    // =========================================================================
    // Unit tests – KinematicLogLikelihood (via compute_cost)
    // =========================================================================

    fn kll_cfg() -> CostConfig {
        use crate::engine_config::edge_config::CostVariant;
        CostConfig {
            variant: CostVariant::KinematicLogLikelihood,
            ..Default::default()
        }
    }

    /// Regression: band_shared = 1 with near-zero other terms previously gave < 0.
    #[test]
    fn cost_positive_band_shared_near_zero_other_terms() {
        let (from, to) = seed_pair_consistent();
        let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
        assert!(cost > 0.0, "cost must be > 0 with shared band, got {cost}");
        assert!(cost.is_finite());
    }

    #[test]
    fn cost_positive_band_not_shared() {
        let mut store = SeedStore::new();
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, 0.5, 0.1, 3e-3, 1, 1000.0);
        let to = make_seed_from_pair(&mut store, 2, 2, 3, 60001.0, 0.503, 0.1, 3e-3, 2, 1000.0);
        let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
        assert!(
            cost > 0.0,
            "cost must be > 0 with different bands, got {cost}"
        );
        assert!(cost.is_finite());
    }

    /// Sharing a band must give a strictly lower (better) cost than not sharing.
    #[test]
    fn cost_band_shared_lower_than_not_shared() {
        let mut store = SeedStore::new();
        let (ra, dec, vx) = (0.5, 0.1, 3e-3);
        let from_s = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, 1000.0);
        let to_s = make_seed_from_pair(&mut store, 2, 2, 3, 60001.0, ra + vx, dec, vx, 1, 1000.0);
        let from_d = make_seed_from_pair(&mut store, 1, 4, 5, 60000.0, ra, dec, vx, 1, 1000.0);
        let to_d = make_seed_from_pair(&mut store, 2, 6, 7, 60001.0, ra + vx, dec, vx, 2, 1000.0);
        let cost_s = EdgeFeatures::compute_cost(&from_s, &to_s, &kll_cfg());
        let cost_d = EdgeFeatures::compute_cost(&from_d, &to_d, &kll_cfg());
        assert!(
            cost_s < cost_d,
            "shared-band cost {cost_s} should be < not-shared cost {cost_d}"
        );
    }

    /// Band shared: cost is finite and > 0.
    #[test]
    fn cost_formula_band_shared() {
        let (from, to) = seed_pair_consistent();
        let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
        assert!(cost.is_finite() && cost > 0.0, "cost={cost}");
    }

    /// Band not shared: cost is finite and strictly greater than the shared-band case.
    #[test]
    fn cost_formula_band_not_shared() {
        let mut store = SeedStore::new();
        let (ra, dec, vx) = (0.5, 0.1, 3e-3);
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, 1000.0);
        let to = make_seed_from_pair(&mut store, 2, 2, 3, 60001.0, ra + vx, dec, vx, 2, 1000.0);
        let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
        assert!(cost.is_finite() && cost > 0.0, "cost={cost}");
    }

    /// Highly inconsistent pair (large residual): cost is still finite and > 0.
    #[test]
    fn cost_finite_for_large_chi2() {
        let mut store = SeedStore::new();
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, 0.5, 0.1, 3e-3, 1, 1000.0);
        let to = make_seed_from_pair(
            &mut store,
            2,
            2,
            3,
            60010.0,
            0.5 + 1.0,
            0.1,
            3e-3,
            1,
            1000.0,
        );
        let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
        assert!(
            cost.is_finite(),
            "cost should be finite for large residual, got {cost}"
        );
        assert!(
            cost > 0.0,
            "cost should be > 0 for large residual, got {cost}"
        );
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

    /// `compute_cost` (KinematicLogLikelihood) is finite and > 0 on a real consistent pair.
    #[test]
    fn cost_positive_finite_on_consistent_pair() {
        let (from, to) = seed_pair_consistent();
        let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
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
        let (ra, dec, vx) = (0.5, 0.1, 3e-3);
        let from_s = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, 1000.0);
        let to_s = make_seed_from_pair(&mut store, 2, 2, 3, 60001.0, ra + vx, dec, vx, 1, 1000.0);
        let from_d = make_seed_from_pair(&mut store, 1, 4, 5, 60000.0, ra, dec, vx, 1, 1000.0);
        let to_d = make_seed_from_pair(&mut store, 2, 6, 7, 60001.0, ra + vx, dec, vx, 2, 1000.0);
        let cost_s = EdgeFeatures::compute_cost(&from_s, &to_s, &kll_cfg());
        let cost_d = EdgeFeatures::compute_cost(&from_d, &to_d, &kll_cfg());
        assert!(
            cost_d > cost_s,
            "cost with different bands ({cost_d}) should be > shared ({cost_s})"
        );
    }

    // =========================================================================
    // Proptest – KinematicLogLikelihood (via compute_cost) robustness
    // =========================================================================

    proptest! {
        /// For physically reasonable seeds, KinematicLogLikelihood cost must always
        /// be finite and strictly > 0.
        #[test]
        fn prop_cost_always_finite_positive(
            epoch_offset   in 0.1f64..10.0,
            vx             in -1e-2f64..1e-2,
            flux_mean      in 100.0f64..5000.0,
            flux_delta_pct in -0.3f64..0.3,
        ) {
            let mut store = SeedStore::new();
            let (ra, dec) = (0.5, 0.1);
            let flux_to = (flux_mean * (1.0 + flux_delta_pct)).max(1.0);
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0,
                                           ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3,
                                           60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_to);
            let cost = EdgeFeatures::compute_cost(&from, &to, &kll_cfg());
            prop_assert!(cost.is_finite(), "cost not finite: {cost}");
            prop_assert!(cost > 0.0,       "cost not > 0: {cost}");
        }

        /// Sharing a band must *never* raise the KLL cost compared to not sharing.
        #[test]
        fn prop_band_shared_never_raises_cost(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
        ) {
            let mut store = SeedStore::new();
            let (ra, dec) = (0.5, 0.1);
            let from_s = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0,
                                             ra, dec, vx, 1, flux_mean);
            let to_s   = make_seed_from_pair(&mut store, 2, 2, 3,
                                             60000.0 + epoch_offset,
                                             ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let from_d = make_seed_from_pair(&mut store, 1, 4, 5, 60000.0,
                                             ra, dec, vx, 1, flux_mean);
            let to_d   = make_seed_from_pair(&mut store, 2, 6, 7,
                                             60000.0 + epoch_offset,
                                             ra + vx * epoch_offset, dec, vx, 2, flux_mean);
            let cost_s = EdgeFeatures::compute_cost(&from_s, &to_s, &kll_cfg());
            let cost_d = EdgeFeatures::compute_cost(&from_d, &to_d, &kll_cfg());
            prop_assert!(cost_s <= cost_d, "shared {cost_s} > not-shared {cost_d}");
        }

        /// Adding an extra position residual must (weakly) increase the KLL cost.
        #[test]
        fn prop_larger_residual_increases_cost(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            delta        in 0.0f64..0.5,
        ) {
            let mut store = SeedStore::new();
            let (ra, dec) = (0.5, 0.1);
            let predicted_ra = ra + vx * epoch_offset;
            let from_lo = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0,
                                              ra, dec, vx, 1, 1000.0);
            let to_lo   = make_seed_from_pair(&mut store, 2, 2, 3,
                                              60000.0 + epoch_offset,
                                              predicted_ra, dec, vx, 1, 1000.0);
            let from_hi = make_seed_from_pair(&mut store, 1, 4, 5, 60000.0,
                                              ra, dec, vx, 1, 1000.0);
            let to_hi   = make_seed_from_pair(&mut store, 2, 6, 7,
                                              60000.0 + epoch_offset,
                                              predicted_ra + delta, dec, vx, 1, 1000.0);
            let cost_lo = EdgeFeatures::compute_cost(&from_lo, &to_lo, &kll_cfg());
            let cost_hi = EdgeFeatures::compute_cost(&from_hi, &to_hi, &kll_cfg());
            prop_assert!(
                cost_hi >= cost_lo,
                "larger residual gave lower cost: lo={cost_lo} hi={cost_hi}"
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

    // =========================================================================
    // Unit tests – compute_cost variants
    // =========================================================================

    fn make_cfg(
        variant: crate::engine_config::edge_config::CostVariant,
        sigma_q: f64,
    ) -> CostConfig {
        CostConfig {
            variant,
            sigma_q,
            cauchy_scale: 2.0,
            student_nu: 3.0,
        }
    }

    /// All cost variants must return a finite, strictly-positive value on a
    /// consistent seed pair.
    #[test]
    fn compute_cost_all_variants_positive_finite() {
        use crate::engine_config::edge_config::CostVariant;
        let (from, to) = seed_pair_consistent();
        let variants = [
            make_cfg(CostVariant::KinematicLogLikelihood, 0.0),
            make_cfg(CostVariant::GaussianChi2, 0.0),
            make_cfg(CostVariant::SingerCwna, 1e-3),
            make_cfg(CostVariant::SingerCwna, 0.0),
            make_cfg(CostVariant::RobustCauchy, 0.0),
            make_cfg(CostVariant::RobustCauchy, 1e-3),
            make_cfg(CostVariant::RobustStudentT, 0.0),
            make_cfg(CostVariant::RobustStudentT, 1e-3),
        ];
        for cfg in &variants {
            let cost = EdgeFeatures::compute_cost(&from, &to, cfg);
            assert!(
                cost.is_finite() && cost > 0.0,
                "variant {:?} sigma_q={}: cost={cost}",
                cfg.variant,
                cfg.sigma_q
            );
        }
    }

    /// `GaussianChi2` with sigma_q=0 must give exactly the same result as
    /// `KinematicLogLikelihood` (same formula, same covariances).
    #[test]
    fn compute_cost_gaussian_chi2_equals_kll() {
        use crate::engine_config::edge_config::CostVariant;
        let (from, to) = seed_pair_consistent();
        let kll = EdgeFeatures::compute_cost(
            &from,
            &to,
            &make_cfg(CostVariant::KinematicLogLikelihood, 0.0),
        );
        let gchi =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
        assert!(
            (kll - gchi).abs() < 1e-10,
            "KinematicLogLikelihood ({kll}) and GaussianChi2 ({gchi}) should match"
        );
    }

    /// `SingerCwna` with sigma_q=0 must collapse to the same result as `GaussianChi2`.
    #[test]
    fn compute_cost_singer_sigma_q_zero_equals_gaussian() {
        use crate::engine_config::edge_config::CostVariant;
        let (from, to) = seed_pair_consistent();
        let singer =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, 0.0));
        let gauss =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
        assert!(
            (singer - gauss).abs() < 1e-10,
            "SingerCwna sigma_q=0 ({singer}) should equal GaussianChi2 ({gauss})"
        );
    }

    /// `SingerCwna` with sigma_q > 0 produces a *different* (generally lower)
    /// kinematic cost compared to `GaussianChi2`, because the inflated covariance
    /// reduces the Mahalanobis distance.
    #[test]
    fn compute_cost_singer_cwna_lowers_kinematic_cost() {
        use crate::engine_config::edge_config::CostVariant;
        let (from, to) = seed_pair_consistent();
        let singer =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, 1e-3));
        let gauss =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
        // CWNA inflates S → S_inv decreases → chi² decreases → kinematic cost ≤ GaussianChi2.
        // (Photometry terms are equal so the comparison holds on total cost too.)
        assert!(
            singer <= gauss + 1e-10,
            "SingerCwna ({singer}) should be ≤ GaussianChi2 ({gauss}) when sigma_q > 0"
        );
    }

    /// For large chi² (inconsistent pair), `RobustCauchy` cost must be strictly
    /// below `GaussianChi2` cost (logarithmic saturation kicks in).
    #[test]
    fn compute_cost_robust_cauchy_bounded_for_inconsistent_pair() {
        use crate::engine_config::edge_config::CostVariant;
        let mut store = SeedStore::new();
        let ra = 0.5_f64;
        let dec = 0.1_f64;
        let vx = 3e-3;
        // "from" expects velocity vx but "to" is far off (large residual).
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, 1000.0);
        let to = make_seed_from_pair(
            &mut store,
            2,
            2,
            3,
            60010.0,
            ra + 10.0 * vx + 1.0,
            dec,
            vx,
            1,
            1000.0,
        );
        let cauchy =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::RobustCauchy, 0.0));
        let gauss =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
        assert!(
            cauchy < gauss,
            "RobustCauchy ({cauchy}) should be < GaussianChi2 ({gauss}) for inconsistent pair"
        );
    }

    /// Same robust-bounding property for `RobustStudentT`.
    #[test]
    fn compute_cost_robust_student_t_bounded_for_inconsistent_pair() {
        use crate::engine_config::edge_config::CostVariant;
        let mut store = SeedStore::new();
        let ra = 0.5_f64;
        let dec = 0.1_f64;
        let vx = 3e-3;
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, 1000.0);
        let to = make_seed_from_pair(
            &mut store,
            2,
            2,
            3,
            60010.0,
            ra + 10.0 * vx + 1.0,
            dec,
            vx,
            1,
            1000.0,
        );
        let student =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::RobustStudentT, 0.0));
        let gauss =
            EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
        assert!(
            student < gauss,
            "RobustStudentT ({student}) should be < GaussianChi2 ({gauss}) for inconsistent pair"
        );
    }

    /// `compute_cost` must be deterministic for every variant.
    #[test]
    fn compute_cost_is_deterministic() {
        use crate::engine_config::edge_config::CostVariant;
        let (from, to) = seed_pair_consistent();
        for cfg in &[
            make_cfg(CostVariant::KinematicLogLikelihood, 0.0),
            make_cfg(CostVariant::GaussianChi2, 0.0),
            make_cfg(CostVariant::SingerCwna, 1e-3),
            make_cfg(CostVariant::RobustCauchy, 0.0),
            make_cfg(CostVariant::RobustStudentT, 0.0),
        ] {
            let c1 = EdgeFeatures::compute_cost(&from, &to, cfg);
            let c2 = EdgeFeatures::compute_cost(&from, &to, cfg);
            assert_eq!(
                c1, c2,
                "compute_cost not deterministic for {:?}",
                cfg.variant
            );
        }
    }

    /// YAML serde round-trip: `gaussian_chi2` deserialises to `GaussianChi2`.
    #[test]
    fn cost_variant_yaml_roundtrip() {
        use crate::engine_config::edge_config::CostVariant;
        let cases = [
            (
                "gaussian_chi2",
                matches!(CostVariant::GaussianChi2, CostVariant::GaussianChi2),
            ),
            (
                "singer_cwna",
                matches!(CostVariant::SingerCwna, CostVariant::SingerCwna),
            ),
            (
                "kinematic_log_likelihood",
                matches!(
                    CostVariant::KinematicLogLikelihood,
                    CostVariant::KinematicLogLikelihood
                ),
            ),
            (
                "robust_cauchy",
                matches!(CostVariant::RobustCauchy, CostVariant::RobustCauchy),
            ),
            (
                "robust_student_t",
                matches!(CostVariant::RobustStudentT, CostVariant::RobustStudentT),
            ),
        ];
        for (yaml_name, _) in &cases {
            let v: CostVariant = serde_yaml::from_str(&format!("\"{}\"", yaml_name))
                .unwrap_or_else(|e| panic!("Failed to parse '{yaml_name}': {e}"));
            let roundtripped = serde_yaml::to_string(&v)
                .unwrap_or_else(|e| panic!("Failed to serialise {yaml_name}: {e}"));
            assert!(
                roundtripped.trim().trim_matches('"') == *yaml_name,
                "round-trip failed for '{yaml_name}': got '{}'",
                roundtripped.trim()
            );
        }
    }

    // =========================================================================
    // Proptest – compute_cost invariants across variants
    // =========================================================================

    proptest! {
        /// For any physically plausible seed pair, all cost variants must return
        /// a finite value strictly greater than zero.
        #[test]
        fn prop_all_variants_positive_finite(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
            sigma_q      in 0.0f64..1e-2,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let cfgs = [
                make_cfg(CostVariant::KinematicLogLikelihood, 0.0),
                make_cfg(CostVariant::GaussianChi2, 0.0),
                make_cfg(CostVariant::SingerCwna, sigma_q),
                make_cfg(CostVariant::RobustCauchy, sigma_q),
                make_cfg(CostVariant::RobustStudentT, sigma_q),
            ];
            for cfg in &cfgs {
                let cost = EdgeFeatures::compute_cost(&from, &to, cfg);
                prop_assert!(
                    cost.is_finite() && cost > 0.0,
                    "variant {:?} sigma_q={}: cost={cost}",
                    cfg.variant, sigma_q
                );
            }
        }

        /// `KinematicLogLikelihood` and `GaussianChi2` (sigma_q=0) must agree to
        /// within floating-point tolerance for any valid seed pair.
        #[test]
        fn prop_kll_matches_gaussian_chi2(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let kll  = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::KinematicLogLikelihood, 0.0));
            let gchi = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
            prop_assert!(
                (kll - gchi).abs() < 1e-9,
                "KLL ({kll}) and GaussianChi2 ({gchi}) disagree by {}",
                (kll - gchi).abs()
            );
        }

        /// `SingerCwna` with sigma_q=0 collapses to `GaussianChi2` for any seed pair.
        #[test]
        fn prop_singer_sigma_q_zero_equals_gaussian(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let singer = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, 0.0));
            let gauss  = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
            prop_assert!(
                (singer - gauss).abs() < 1e-10,
                "SingerCwna sigma_q=0 ({singer}) ≠ GaussianChi2 ({gauss})"
            );
        }

        /// `SingerCwna` with sigma_q > 0 has kinematic cost ≤ `GaussianChi2`
        /// for any seed pair.  This follows from S_cwna ≥ S_baseline (PSD
        /// ordering) ⇒ S_cwna⁻¹ ≤ S_baseline⁻¹ ⇒ χ²_cwna ≤ χ²_baseline.
        #[test]
        fn prop_singer_cwna_le_gaussian_chi2(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
            sigma_q      in 1e-6f64..1e-2,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let singer = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, sigma_q));
            let gauss  = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
            // Allow a tiny floating-point tolerance.
            prop_assert!(
                singer <= gauss + 1e-9,
                "SingerCwna ({singer}) > GaussianChi2 ({gauss}) for sigma_q={sigma_q}"
            );
        }

        /// A larger `sigma_q` produces a weakly smaller (or equal) cost for
        /// `SingerCwna`, because more process noise inflates the covariance
        /// further and reduces χ².
        #[test]
        fn prop_singer_larger_sigma_q_lower_cost(
            epoch_offset in 0.1f64..10.0,
            vx           in -1e-2f64..1e-2,
            flux_mean    in 100.0f64..5000.0,
            sigma_small  in 1e-6f64..1e-3,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            let mut store = SeedStore::new();
            let ra  = 0.5_f64;
            let dec = 0.1_f64;
            let sigma_large = sigma_small * 10.0;
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra, dec, vx, 1, flux_mean);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra + vx * epoch_offset, dec, vx, 1, flux_mean);
            let cost_small = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, sigma_small));
            let cost_large = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, sigma_large));
            prop_assert!(
                cost_large <= cost_small + 1e-9,
                "larger sigma_q gave higher SingerCwna cost: small={cost_small} large={cost_large}"
            );
        }
    }

    // =========================================================================
    // Unit tests – spherical residual stability (r_sph, chi2_pos_sph)
    // =========================================================================

    /// For a near-perfect prediction the spherical residual should be tiny.
    #[test]
    fn r_sph_small_for_consistent_prediction() {
        let (from, to) = seed_pair_consistent();
        let core = FeatureCore::from_nodes(&from, &to);
        assert!(
            core.r_sph.is_finite(),
            "r_sph should be finite for consistent pair, got {}",
            core.r_sph
        );
        assert!(core.r_sph >= 0.0, "r_sph should be ≥ 0, got {}", core.r_sph);
        // Consistent pair: residual should be below 1 arcmin (2.9e-4 rad).
        assert!(
            core.r_sph < 3e-4,
            "r_sph={} should be small for a consistent near-perfect edge",
            core.r_sph
        );
    }

    /// Spherical residual must be finite and in [0, π] even when seed centres
    /// are 45° apart (gnomonic denominator is small but non-zero there).
    #[test]
    fn r_sph_finite_for_45deg_separation() {
        use std::f64::consts::PI;
        let mut store = SeedStore::new();
        // from at (0, 0), to at (PI/4, 0) — 45° apart in RA.
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, 0.0, 0.0, 3e-3, 1, 1000.0);
        let to = make_seed_from_pair(&mut store, 2, 2, 3, 60001.0, PI / 4.0, 0.0, 3e-3, 1, 1000.0);
        let core = FeatureCore::from_nodes(&from, &to);
        assert!(
            core.r_sph.is_finite(),
            "r_sph must be finite at 45º sep, got {}",
            core.r_sph
        );
        assert!(
            core.r_sph >= 0.0 && core.r_sph <= PI + 1e-12,
            "r_sph={} must be in [0, π]",
            core.r_sph
        );
    }

    /// At 90° the gnomonic denominator is zero — the old 2-D Mahalanobis blows
    /// up to ~10^20. The spherical residual must remain finite and in [0, π].
    #[test]
    fn r_sph_finite_for_90deg_separation() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let mut store = SeedStore::new();
        // from at equator, to at north pole (90° away in Dec).
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, 0.0, 0.0, 3e-3, 1, 1000.0);
        let to = make_seed_from_pair(
            &mut store, 2, 2, 3, 60001.0, 0.0, FRAC_PI_2, 3e-3, 1, 1000.0,
        );
        let core = FeatureCore::from_nodes(&from, &to);
        assert!(
            core.r_sph.is_finite(),
            "r_sph must be finite at 90º sep (gnomonic singularity), got {}",
            core.r_sph
        );
        assert!(
            core.r_sph >= 0.0 && core.r_sph <= PI + 1e-12,
            "r_sph={} must be in [0, π] at 90º sep",
            core.r_sph
        );
    }

    /// Near-antipodal seeds (179°): r_sph must still be finite.
    #[test]
    fn r_sph_finite_for_near_antipodal_seeds() {
        use std::f64::consts::PI;
        let mut store = SeedStore::new();
        // from at (0, 0), to at (~π, 0) in RA — nearly 180° apart.
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, 0.0, 0.0, 3e-3, 1, 1000.0);
        let to = make_seed_from_pair(
            &mut store,
            2,
            2,
            3,
            60001.0,
            PI - 0.01,
            0.0,
            3e-3,
            1,
            1000.0,
        );
        let core = FeatureCore::from_nodes(&from, &to);
        assert!(
            core.r_sph.is_finite(),
            "r_sph must be finite near antipodal sep, got {}",
            core.r_sph
        );
        assert!(
            core.r_sph >= 0.0 && core.r_sph <= PI + 1e-12,
            "r_sph={} must be in [0, π]",
            core.r_sph
        );
    }

    /// `compute_cost` must be finite and positive for 90°-separated seeds,
    /// for all cost variants.  This is the primary regression test for the
    /// gnomonic projection blow-up.
    #[test]
    fn cost_finite_for_90deg_separation_all_variants() {
        use crate::engine_config::edge_config::CostVariant;
        use std::f64::consts::FRAC_PI_2;
        let mut store = SeedStore::new();
        let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, 0.0, 0.0, 3e-3, 1, 1000.0);
        let to = make_seed_from_pair(
            &mut store, 2, 2, 3, 60001.0, 0.0, FRAC_PI_2, 3e-3, 1, 1000.0,
        );
        for (label, cfg) in [
            (
                "KinematicLogLikelihood",
                make_cfg(CostVariant::KinematicLogLikelihood, 0.0),
            ),
            ("GaussianChi2", make_cfg(CostVariant::GaussianChi2, 0.0)),
            ("SingerCwna_0", make_cfg(CostVariant::SingerCwna, 0.0)),
            ("SingerCwna_1e3", make_cfg(CostVariant::SingerCwna, 1e-3)),
            ("RobustCauchy", make_cfg(CostVariant::RobustCauchy, 0.0)),
            ("RobustStudentT", make_cfg(CostVariant::RobustStudentT, 0.0)),
        ] {
            let cost = EdgeFeatures::compute_cost(&from, &to, &cfg);
            assert!(
                cost.is_finite(),
                "{label}: cost must be finite at 90º separation, got {cost}"
            );
            assert!(
                cost > 0.0,
                "{label}: cost must be > 0 at 90º separation, got {cost}"
            );
        }
    }

    /// `s_pos_scalar` must always be strictly positive and finite.
    #[test]
    fn s_pos_scalar_positive_finite() {
        let (from, to) = seed_pair_consistent();
        let core = FeatureCore::from_nodes(&from, &to);
        assert!(
            core.s_pos_scalar.is_finite() && core.s_pos_scalar > 0.0,
            "s_pos_scalar must be > 0 and finite, got {}",
            core.s_pos_scalar
        );
    }

    // =========================================================================
    // Proptest – spherical residual invariants across seed separations
    // =========================================================================

    proptest! {
        /// `r_sph` is always in `[0, π]` regardless of seed sky positions.
        #[test]
        fn prop_r_sph_in_0_pi_range(
            ra_from  in 0.0f64..std::f64::consts::TAU,
            dec_from in -1.5f64..1.5,
            ra_to    in 0.0f64..std::f64::consts::TAU,
            dec_to   in -1.5f64..1.5,
            epoch_offset in 0.1f64..10.0,
            vx       in -1e-2f64..1e-2,
        ) {
            use std::f64::consts::PI;
            let mut store = SeedStore::new();
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra_from, dec_from, vx, 1, 1000.0);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra_to, dec_to, vx, 1, 1000.0);
            let core = FeatureCore::from_nodes(&from, &to);
            prop_assert!(
                core.r_sph.is_finite(),
                "r_sph is not finite: {}", core.r_sph
            );
            prop_assert!(
                core.r_sph >= 0.0,
                "r_sph < 0: {}", core.r_sph
            );
            prop_assert!(
                core.r_sph <= PI + 1e-12,
                "r_sph > π: {}", core.r_sph
            );
        }

        /// `s_pos_scalar` is always strictly positive and finite.
        #[test]
        fn prop_s_pos_scalar_positive_finite(
            ra_from  in 0.0f64..std::f64::consts::TAU,
            dec_from in -1.5f64..1.5,
            ra_to    in 0.0f64..std::f64::consts::TAU,
            dec_to   in -1.5f64..1.5,
            epoch_offset in 0.1f64..10.0,
            vx in -1e-2f64..1e-2,
        ) {
            let mut store = SeedStore::new();
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra_from, dec_from, vx, 1, 1000.0);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra_to, dec_to, vx, 1, 1000.0);
            let core = FeatureCore::from_nodes(&from, &to);
            prop_assert!(
                core.s_pos_scalar.is_finite() && core.s_pos_scalar > 0.0,
                "s_pos_scalar must be > 0 and finite, got {}", core.s_pos_scalar
            );
        }

        /// The spherical chi2_pos used in the cost path is always finite,
        /// non-negative, and bounded (no blow-up from gnomonic singularity),
        /// for any combination of seed sky positions and sigma_q.
        #[test]
        fn prop_cost_chi2_pos_sph_finite_nonneg(
            ra_from  in 0.0f64..std::f64::consts::TAU,
            dec_from in -1.5f64..1.5,
            ra_to    in 0.0f64..std::f64::consts::TAU,
            dec_to   in -1.5f64..1.5,
            epoch_offset in 0.1f64..10.0,
            vx       in -1e-2f64..1e-2,
            sigma_q  in 0.0f64..10.0,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            let mut store = SeedStore::new();
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra_from, dec_from, vx, 1, 1000.0);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra_to, dec_to, vx, 1, 1000.0);
            let cost = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::SingerCwna, sigma_q));
            prop_assert!(
                cost.is_finite(),
                "cost not finite for sigma_q={sigma_q}: {cost}"
            );
            prop_assert!(
                cost > 0.0,
                "cost not > 0 for sigma_q={sigma_q}: {cost}"
            );
        }

        /// For any seed pair, the cost from chi2_with_cwna using the spherical
        /// residual must not exceed ~10^10 (no blow-up, bounded by [r_sph ≤ π]).
        ///
        /// Old gnomonic formula could reach ~10^20 for seeds > 45° apart.
        #[test]
        fn prop_cost_bounded_no_gnomonic_blowup(
            ra_from  in 0.0f64..std::f64::consts::TAU,
            dec_from in -1.5f64..1.5,
            ra_to    in 0.0f64..std::f64::consts::TAU,
            dec_to   in -1.5f64..1.5,
            epoch_offset in 0.1f64..10.0,
            vx       in -1e-2f64..1e-2,
        ) {
            use crate::engine_config::edge_config::CostVariant;
            // Maximum chi2_pos_sph = π² / s_pos_scalar_min.
            // With 1-arcsec astrometry, s_pos_scalar_min ~ 2×10^{-11} rad²,
            // giving chi2_pos_sph_max ~ π² / 2e-11 ~ 5e12, <<< 10^20.
            let upper_bound = 1e14_f64;  // conservative
            let mut store = SeedStore::new();
            let from = make_seed_from_pair(&mut store, 1, 0, 1, 60000.0, ra_from, dec_from, vx, 1, 1000.0);
            let to   = make_seed_from_pair(&mut store, 2, 2, 3, 60000.0 + epoch_offset,
                                           ra_to, dec_to, vx, 1, 1000.0);
            let cost = EdgeFeatures::compute_cost(&from, &to, &make_cfg(CostVariant::GaussianChi2, 0.0));
            prop_assert!(
                cost < upper_bound,
                "cost={cost} exceeds upper bound {upper_bound} (gnomonic blow-up?)"
            );
        }
    }
}

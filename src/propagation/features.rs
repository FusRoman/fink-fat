// src/seeding/features.rs

//! Kinematic feature extraction for intra-night seeds (pairs & triplets).
//!
//! # Overview
//! This module converts minimal intra-night seeds—**pairs** `(a, b)` and rarer
//! **triplets** `(a, b, c)` of `Alert`s—into compact, **kinematic features**
//! suitable for **inter-night linking**. For each seed, we:
//!
//! - Center a **gnomonic tangent plane** near the seed's angular centroid.
//! - Estimate a **mid-epoch** and a **sky-plane velocity** (constant-velocity
//!   for pairs; optional **curvature**/acceleration for triplets).
//! - Build simple, interpretable **position/velocity covariances** from the
//!   per-detection astrometric uncertainties and sampling baselines.
//! - Summarize **photometry** using mean and dispersion of **difference PSF
//!   flux** (in nJy), which you can later convert to magnitudes if a zero-point
//!   is available upstream.
//!
//! The output is a list of [`SeedNode`] records—**one per input seed**—that
//! capture the local motion state at the seed's reference epoch. These records
//! are designed to be used by downstream **inter-night graph building** and
//! **assignment/min-cost flow** steps.
//!
//! # Design & Scope
//! - **Stateless & pure**: the public functions operate on an immutable
//!   [`AlertStore`] and slice(s) of seed indices, returning new `Vec<SeedNode>`.
//! - **Pairs-first**: pairs are common; we assume **constant velocity** and
//!   inflate uncertainties to account for unmodeled curvature. Triplets add an
//!   optional **acceleration** term via a quadratic fit in time.
//! - **Tangent plane**: all kinematics are expressed on a local plane in **radians**,
//!   centered at a robust spherical midpoint to minimize projection errors.
//!
//! # Units
//! - Angles (`ra`, `dec`, plane `x`, `y`): **radians**.
//! - Epochs: **MJD (TT)**, days.
//! - Velocities: **radians per day**.
//! - Acceleration (triplets): **radians per day²**.
//! - Flux photometry: **nJy** (difference PSF flux).
//!
//! # See also
//! - [`extract_pair_features`] — constant-velocity features for `(a, b)`.
//! - [`extract_triplet_features`] — quadratic motion features for `(a, b, c)`.
//!
//! # Python
//! These functions are good candidates to expose as a single high-level API
//! with a NumPy-style docstring, e.g. `extract_pair_features(store, pairs, ...)`,
//! returning a list of records or a dict-of-arrays ready for `pandas.DataFrame`.

use ahash::AHashMap;

use crate::{
    alerts::{AlertId, AlertStore},
    seeding::{
        geometrical_seeding::{Pairs, Triplets},
        space_time_bucket::{SpatialBinner, SpatialKey},
    },
};

// --- Small numeric constants reused ---
const INV_COSC_MIN: f64 = 1e-12;
const NORM_MIN: f64 = 1e-16;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

/* --------------------------- Public types --------------------------- */

/// Unified feature record for an intra-night seed, ready for inter-night linking.
///
/// # Overview
/// `SeedNode` encodes the local kinematics of a seed at a **reference epoch**
/// (`epoch_mid`) on a **gnomonic tangent plane** whose center is explicitly
/// stored (`center_ra`, `center_dec`). The state at `epoch_mid` comprises:
/// - `pos_xy` – plane position (rad),
/// - `vel_xy` – plane velocity (rad/day),
/// - `acc_xy` – optional plane acceleration (rad/day²) for triplets,
/// - `cov_pos`, `cov_vel` – simple 2×2 covariances (per-axis on the plane).
///
/// For convenience and fast spatial indexing, the corresponding **sky position**
/// at `epoch_mid` (obtained by inverse gnomonic from `pos_xy`) is cached as
/// (`ra_mid`, `dec_mid`). Photometry is summarized by `flux_mean` and `flux_std`
/// (difference PSF flux in nJy).
///
/// This structure is intentionally compact and cloneable; it can be moved across
/// threads and returned to Python bindings if needed.
///
/// # Fields
/// - `seed_id` – Unique id **within the current extraction batch** (0..N−1).
/// - `night_id` – Night identifier; use `-1` if unknown.
/// - `epoch_mid` – Reference epoch (MJD, TT).
/// - `pos_xy` – Tangent-plane position at `epoch_mid` (rad).
/// - `vel_xy` – Tangent-plane velocity (rad/day).
/// - `acc_xy` – Optional tangent-plane acceleration (rad/day²); `None` for pairs,
///   `Some([ax, ay])` for triplets.
/// - `cov_pos` – 2×2 covariance of position (rad²), diagonal by construction here.
/// - `cov_vel` – 2×2 covariance of velocity ((rad/day)²), diagonal by construction here.
/// - `flux_mean` – Mean difference PSF flux across member detections (nJy).
/// - `flux_std` – Robust dispersion (nJy) across member detections (mean absolute deviation–like).
/// - `band` – Representative photometric band code (forwarded from first member).
/// - `n_obs` – Number of detections in the seed (2 for pairs, 3 for triplets).
/// - `members` – Member `AlertId`s (indexing `AlertStore.alerts`), useful for tracing/debugging.
/// - `center_ra`, `center_dec` – **ICRS** tangent-plane center used during extraction (rad).
/// - `ra_mid`, `dec_mid` – Sky position at `epoch_mid`, obtained from `pos_xy` around
///   `(center_ra, center_dec)` (rad). Typically used for **per-night spatial indexing** (e.g., HEALPix).
///
/// # Units
/// - Angles (`ra`, `dec`, `x`, `y`) in **radians**.
/// - Epochs in **MJD (TT)** days.
/// - Velocities in **radians per day**; accelerations in **radians per day²**.
/// - Fluxes in **nJy** (difference PSF flux).
///
/// # Notes
/// - Covariances are currently **diagonal heuristics**; downstream steps may
///   inflate/adjust them (e.g., process noise for unmodeled curvature).
/// - `ra_mid/dec_mid` are a **cache** derived from `pos_xy` and the plane center;
///   keep them in sync if you modify `pos_xy` or the center (extraction does this once).
/// - For numerical stability, the tangent-plane center is chosen as a spherical
///   midpoint of the seed’s endpoints (pairs) or endpoints of the span (triplets).
///
/// # Methods (step-2 prediction & search)
/// - [`SeedNode::predict_radec`] – Predict sky position (RA, Dec) at a target epoch.
/// - [`SeedNode::cone_candidates`] – Predict a **kσ** sky cone and return **candidate** seeds
///   from a per-night spatial index.
///
/// # See also
/// - [`extract_pair_features`] — constant-velocity features for `(a, b)`.
/// - [`extract_triplet_features`] — quadratic motion features for `(a, b, c)`.
#[derive(Clone, Debug)]
pub struct SeedNode {
    /// Unique seed id **within the current extraction batch** (0..N-1).
    pub seed_id: u64,
    /// Optional night identifier. Use `-1` if unknown.
    pub night_id: i32,
    /// Reference epoch for the kinematics (MJD, TT).
    pub epoch_mid: f64,
    /// Tangent-plane position at `epoch_mid` (radians).
    pub pos_xy: [f64; 2],
    /// Tangent-plane velocity (radians/day).
    pub vel_xy: [f64; 2],
    /// 2×2 covariance of position (rad²), per-axis on the tangent plane.
    pub cov_pos: [[f64; 2]; 2],
    /// 2×2 covariance of velocity ((rad/day)²), per-axis on the tangent plane.
    pub cov_vel: [[f64; 2]; 2],
    /// Optional acceleration (radians/day²). `None` for pairs; `Some([ax, ay])` for triplets.
    pub acc_xy: Option<[f64; 2]>,
    /// Mean difference PSF flux in nJy across member detections.
    pub flux_mean: f32,
    /// Flux dispersion (nJy) across member detections; robust but simple estimate.
    pub flux_std: f32,
    /// Representative photometric band code for the seed (e.g., from the first member).
    pub band: u8,
    /// Number of detections forming the seed (2 for pairs, 3 for triplets).
    pub n_obs: u16,
    /// Member alert indices (into `AlertStore.alerts`), useful for tracing/debugging.
    pub members: Vec<AlertId>,
    /// **ICRS** tangent-plane center used during extraction (rad).
    pub center_ra: f64,
    /// **ICRS** tangent-plane center used during extraction (rad).
    pub center_dec: f64,
    /// Sky position in radians at `epoch_mid`, projected back from `pos_xy` around `(center_ra, center_dec)`.
    pub ra_mid: f64,
    /// Sky position in radians at `epoch_mid`, projected back from `pos_xy` around `(center_ra, center_dec)`.
    pub dec_mid: f64,
}

/// Tuning parameters for feature extraction.
///
/// # Overview
/// Controls the construction of simple, interpretable covariances and optional
/// guardrails. We **do not** inject process noise directly here; rather, we
/// produce base covariances that downstream prediction can inflate with a
/// model-noise schedule (e.g., curvature growing with |Δt|).
///
/// # Fields
/// - `max_speed_rad_per_day` – Optional guardrail. If present, pair seeds whose
///   inferred sky-plane speed exceeds this threshold are **discarded**.
///   Useful to avoid degenerate pairs with tiny Δt or mis-associations.
///
/// # See also
/// - [`extract_pair_features`]
/// - [`extract_triplet_features`]
#[derive(Clone, Copy, Debug)]
pub struct FeatureExtractParams {
    /// Optional maximum sky-plane speed (rad/day). Use `None` to disable.
    pub max_speed_rad_per_day: Option<f64>,
}

/// Additive model-noise schedule to cover unmodeled curvature and model mismatch.
///
/// Overview
/// --------
/// `ModelNoise` parameterizes a simple, time-dependent variance term `Q(Δt)` that
/// is **added per axis** to the predicted plane covariance:
/// `Σ_p(t) ≈ Σ_pos + Δt² Σ_vel + Q(Δt)`. It is mainly useful for **pairs**
/// (constant-velocity seeds) where true motion exhibits curvature between nights.
/// For **triplets**, you can still keep a small `Q` to hedge against residual
/// modeling errors.
///
/// Form
/// ----
/// The schedule is a low-order polynomial of the time gap magnitude:
/// `Q(Δt) = q0 + q1 · |Δt| + q2 · Δt²`,
/// where `Δt = t_target − epoch_mid` in **days**. All coefficients must be
/// non-negative to preserve positive semidefiniteness.
///
/// Fields
/// ------
/// - `q0` — Static variance floor (rad^2). Compensates for small systematics
///   and projection approximations even at `Δt = 0`.
/// - `q1` — Linear growth (rad^2/day). Captures slow drift-like effects that scale
///   approximately with elapsed time (e.g., small bias in velocity).
/// - `q2` — Quadratic growth (rad^2/day^2). Covers curvature-like divergence that
///   increases faster with `|Δt|`.
///
/// Units
/// -----
/// - `q0` in **radians^2**,
/// - `q1` in **radians^2/day** (multiplied by `|Δt|`),
/// - `q2` in **radians^2/day^2** (multiplied by `Δt²`).
///
/// Notes
/// -----
/// - Start conservatively for pairs, e.g. `q0 ≈ (0.15″ in rad)^2`, small `q1`,
///   and a `q2` tuned on simulation (Sorcha) to reach high recall over 1–2 days.
/// - For triplets (with `acc_xy`), you can set `{q0,q1,q2}` smaller, but not
///   strictly zero if you want to absorb residual modeling error.
/// - The polynomial is **isotropic** here (same on x and y). If later you adopt
///   anisotropic propagation, switch to a 2D form or inject cross-terms downstream.
///
/// Examples
/// --------
/// ```ignore
/// // 0.15 arcsec in radians, squared:
/// let q0 = (0.15_f64.to_radians() / 3600.0).powi(2);
/// let noise = ModelNoise { q0, q1: 0.0, q2: 5e-14 };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ModelNoise {
    pub q0: f64,
    pub q1: f64,
    pub q2: f64,
}

/// Parameters controlling sky-cone prediction for candidate retrieval.
///
/// Overview
/// --------
/// `PredictorParams` governs how the **predicted plane covariance** at a target
/// epoch is converted into a **sky-cone search**:
/// 1) compute `Σ_p(t)` (including [`ModelNoise`]),
/// 2) take a conservative **k-sigma circle** whose radius is
///    `r = k_sigma · sqrt(λ_max(Σ_p))`,
/// 3) optionally **pad** that radius by one spatial-cell radius to ensure
///    coverage with coarse binners (e.g., HEALPix).
///
/// Fields
/// ------
/// - `k_sigma` — Confidence multiplier (e.g., `3.0` for 3σ coverage on the largest
///   principal axis). Larger values increase recall but also the number of candidates.
/// - `noise` — Additive variance schedule `Q(Δt)` plugged into the plane covariance
///   before radius extraction. See [`ModelNoise`].
/// - `pad_cell_radius` — If `true`, add `binner.cell_radius()` to the cone radius
///   to compensate for cell-boundary effects in approximate cone coverage.
///
/// Units
/// -----
/// - `k_sigma` dimensionless,
/// - `noise` in squared radians units consistent with covariance,
/// - Cone radius returned by prediction is in **radians**.
///
/// Notes
/// -----
/// - Start with `k_sigma = 3.0`. If recall is low on validation, increase a bit
///   (3.5–4.0). After an IOD confirmation stage, you can tighten it back.
/// - `pad_cell_radius = true` is recommended when your spatial binner performs
///   **cell-based coverage** rather than exact geometric cone slicing.
/// - Excessive inflation increases fan-out; cap downstream candidates (Top-K)
///   and apply strict scoring cuts (Mahalanobis) to keep runtime bounded.
///
/// Examples
/// --------
/// ```ignore
/// let params = PredictorParams {
///     k_sigma: 3.0,
///     noise: ModelNoise { q0: 1e-12, q1: 0.0, q2: 5e-14 },
///     pad_cell_radius: true,
/// };
/// // Predict a cone and query candidates:
/// let (ra, dec, radius) = seed.predict_cone(t_target, &binner, params);
/// let cand: Vec<SeedId> = index.cone_query(&binner, ra, dec, radius).collect();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PredictorParams {
    /// k-sigma inflation (e.g., 3.0).
    pub k_sigma: f64,
    /// Model noise coefficients Q(Δt) = q0 + q1|Δt| + q2 Δt².
    pub noise: ModelNoise,
    /// If true, add one spatial cell radius to the cone (safety padding).
    pub pad_cell_radius: bool,
}

/* --------------------------- Public API --------------------------- */

impl SeedNode {
    /// Predict the **sky position** (RA, Dec) at a target epoch from this seed's local model.
    ///
    /// Overview
    /// --------
    /// Propagates the seed state on its **gnomonic tangent plane** centered at
    /// `(center_ra, center_dec)` using:
    /// - **pairs:** constant-velocity model `p(t) = p0 + v · Δt`,
    /// - **triplets:** small-quadratic model `p(t) = p0 + v · Δt + 0.5 · a · Δt²`
    ///   when `acc_xy` is present.
    ///
    /// The predicted plane coordinates are then converted back to sky coordinates
    /// through the **inverse gnomonic** transform about the same center.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` — Target epoch in **MJD (TT)** days.
    ///
    /// Return
    /// ------
    /// * `(ra, dec)` in **radians** (ICRS). `ra` is normalized to `[0, 2π)`.
    ///
    /// Units
    /// -----
    /// * `pos_xy` in radians, `vel_xy` in radians/day, `acc_xy` in radians/day²,
    ///   `epoch_mid` and `t_target` in days (MJD TT).
    ///
    /// Notes
    /// -----
    /// * This method predicts only the **mean** position. It does **not** return
    ///   an uncertainty; use [`SeedNode::predict_cone`] to obtain a conservative
    ///   search radius that includes model noise growth with |Δt|.
    /// * Accuracy is best when the object remains within a **small cone** around
    ///   the tangent center (a few degrees at most). For large |Δt| or fast movers,
    ///   consider re-centering the tangent plane or switching to a higher-order
    ///   propagation downstream.
    /// * Numerical stability: the gnomonic inverse is well behaved away from the
    ///   90° great-circle from the center; seeds are constructed to keep separations
    ///   small, so this is generally safe.
    ///
    /// Examples
    /// --------
    /// ```ignore
    /// // Predict sky position, then project back to the plane and compare to the
    /// // local kinematic model at Δt = t_target - epoch_mid.
    /// let (ra, dec) = seed.predict_radec(t_target);
    /// let p = radec_to_tangent(ra, dec, seed.center_ra, seed.center_dec);
    /// let dt = t_target - seed.epoch_mid;
    /// let px = seed.pos_xy[0] + seed.vel_xy[0]*dt + 0.5*seed.acc_xy.unwrap_or([0.0,0.0])[0]*dt*dt;
    /// let py = seed.pos_xy[1] + seed.vel_xy[1]*dt + 0.5*seed.acc_xy.unwrap_or([0.0,0.0])[1]*dt*dt;
    /// assert!((p[0] - px).abs() < 1e-10 && (p[1] - py).abs() < 1e-10);
    /// ```
    ///
    /// See also
    /// --------
    /// * [`SeedNode::predict_cone`] — Predicts (RA, Dec) **and** a kσ search radius.
    /// * [`SeedNode::cone_candidates`] — Retrieves candidate seeds by cone search.
    pub fn predict_radec(&self, t_target: f64) -> (f64, f64) {
        let dt = t_target - self.epoch_mid;
        let mut px = self.pos_xy[0] + self.vel_xy[0] * dt;
        let mut py = self.pos_xy[1] + self.vel_xy[1] * dt;
        if let Some(a) = self.acc_xy {
            px += 0.5 * a[0] * dt * dt;
            py += 0.5 * a[1] * dt * dt;
        }
        tangent_to_radec(px, py, self.center_ra, self.center_dec)
    }

    /// Predict a **cone** (center RA/Dec + radius) and return all **candidate SeedId**
    /// from a per-night spatial index via an approximate cone search.
    ///
    /// Overview
    /// --------
    /// - Build predicted plane covariance at `t_target` as:
    ///   `Σ_p(t) ≈ Σ_pos + Δt² Σ_vel + Q(Δt)` with `Q(Δt) = q0 + q1|Δt| + q2 Δt²`.
    /// - Convert the predicted plane position to sky coordinates `(ra, dec)`.
    /// - Cone radius is a conservative **kσ circle** using `k * sqrt(λ_max(Σ_p))`.
    /// - Query the spatial index with that cone; **cell coverage** may return a superset.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Target epoch (MJD TT).
    /// * `index` – Spatial index of seeds for the **target night**.
    /// * `binner` – Sky partitioner (e.g. HEALPix) implementing `SpatialBinner`.
    /// * `params` – Cone inflation and model noise settings.
    ///
    /// Return
    /// ------
    /// * `Vec<SeedId>` – all candidate seed IDs (approximate superset).
    pub fn cone_candidates<Bs: SpatialBinner>(
        &self,
        t_target: f64,
        index: &SeedSpatialIndex,
        binner: &Bs,
        params: PredictorParams,
    ) -> Vec<SeedId> {
        let (ra, dec, radius) = self.predict_cone(t_target, binner, params);
        index.cone_query(binner, ra, dec, radius).collect()
    }

    /// Predict a **sky cone** at `t_target`: center (RA, Dec) and a conservative radius.
    ///
    /// Overview
    /// --------
    /// 1) Propagate the seed state on its **gnomonic tangent plane** (centered at
    ///    `center_ra, center_dec`) to the target epoch `t_target`, obtaining the
    ///    predicted mean plane position `p` and its plane covariance `Σ_p(t)`; see
    ///    private function \[`SeedNode::predict_on_plane`\] for the motion model and model noise.
    /// 2) Convert the plane mean `p = [x, y]` back to sky coordinates `(ra, dec)`
    ///    via **inverse gnomonic** about the same center.
    /// 3) Turn the 2×2 covariance into a **single search radius** by taking a
    ///    conservative **k-sigma circle** that covers the error ellipse:
    ///    `radius = k_sigma × sqrt(λ_max(Σ_p))`.
    ///
    /// Why `λ_max`?
    /// ------------
    /// For a symmetric covariance `Σ = [[a, b], [b, d]]`, the 1σ error contour
    /// on the plane is an ellipse whose semi-axes are `sqrt(λ₁)` and `sqrt(λ₂)`,
    /// where `λ₁ ≥ λ₂ ≥ 0` are the eigenvalues of `Σ`. Using `sqrt(λ_max)` (i.e.,
    /// `sqrt(λ₁)`) as the radius yields a **circle that fully contains the ellipse**
    /// at the same sigma level—safe for cone searches.
    ///
    /// Closed form (2×2 symmetric):
    /// - `trace = a + d`
    /// - `disc  = (a − d)² + 4 b²`
    /// - `λ_max = 0.5 × (trace + sqrt(disc))`
    ///
    /// In the **diagonal** case (`b = 0`), this reduces to `λ_max = max(a, d)`.
    ///
    /// Cell-padding
    /// ------------
    /// If `pad_cell_radius = true`, we add `binner.cell_radius()` to the radius to
    /// compensate for **cell-based cone coverage** (e.g., HEALPix neighbor unions),
    /// ensuring the discrete cover does not under-approximate the true circle.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` — Target epoch in **MJD (TT)** days.
    /// * `binner`   — Spatial partitioner (e.g., HEALPix) exposing `cell_radius()`.
    /// * `params`   — Predictor knobs (`k_sigma`, model noise, padding).
    ///
    /// Return
    /// ------
    /// * `(ra, dec, radius)` in **radians** (ICRS). `ra` is normalized to `[0, 2π)`.
    ///
    /// Units
    /// -----
    /// * The radius is an **angular** value (radians) on the sphere. The ellipse→circle
    ///   reduction is computed in the **tangent plane**; for small cones the spherical
    ///   vs planar difference is negligible.
    ///
    /// Notes
    /// -----
    /// * Complexity is `O(1)`: one inverse gnomonic and a constant-time 2×2 eigenvalue eval.
    /// * Preconditions: `Σ_p(t)` should be PSD. If you later introduce off-diagonal terms
    ///   and observe tiny negative `λ_max` due to round-off, clamp with `lam_max.max(0.0)`
    ///   before `sqrt`.
    ///
    /// See also
    /// --------
    /// * [`SeedNode::predict_radec`]    — mean sky position only.
    /// * [`PredictorParams`]            — `k_sigma`, model noise, padding.
    #[inline]
    pub fn predict_cone<Bs: SpatialBinner>(
        &self,
        t_target: f64,
        binner: &Bs,
        params: PredictorParams,
    ) -> (f64, f64, f64) {
        // 1) Mean & covariance in the tangent plane at t_target
        let (p, cov) = self.predict_on_plane(t_target, params.noise);

        // 2) Cone center on the sky (inverse gnomonic about the same center)
        let (ra, dec) = tangent_to_radec(p[0], p[1], self.center_ra, self.center_dec);

        // 3) Cone radius: k-sigma on the largest principal axis of Σ_p(t)
        //
        // For Σ = [[a, b], [b, d]] (symmetric):
        //   trace = a + d
        //   disc  = (a - d)^2 + 4 b^2
        //   λ_max = 0.5 * (trace + sqrt(disc))
        //
        // The 1σ along the major axis is sqrt(λ_max); multiply by k_sigma for the search circle.
        let lam_max = lambda_max_2x2(cov);
        let mut radius = params.k_sigma * lam_max.sqrt();

        // Optional padding to cover discrete cell neighborhood approximations
        if params.pad_cell_radius {
            radius += binner.cell_radius();
        }

        (ra, dec, radius)
    }

    /// Predict the **tangent-plane mean position** and a **conservative plane covariance**
    /// at the target epoch `t_target`.
    ///
    /// Overview
    /// --------
    /// This helper advances the seed’s local kinematic model on its **gnomonic
    /// tangent plane** from `epoch_mid` to `t_target` and returns:
    /// 1) the predicted plane coordinates `p(t) = [x(t), y(t)]`, and
    /// 2) an approximate 2×2 covariance for that position on the plane.
    ///
    /// Motion model (mean)
    /// -------------------
    /// The mean is propagated with:
    /// - **Pairs:** `p(t) = p0 + v · Δt`
    /// - **Triplets:** `p(t) = p0 + v · Δt + 0.5 · a · Δt²` (if `acc_xy` is present)
    ///   where `Δt = t_target − epoch_mid` (days), `p0 = pos_xy`, `v = vel_xy`,
    ///   and `a = acc_xy`.
    ///
    /// Covariance model (variance per axis)
    /// ------------------------------------
    /// We return a **diagonal** covariance on the plane using a simple,
    /// interpretable heuristic:
    /// ```
    /// Σ_p(t) ≈ Σ_pos  +  Δt² Σ_vel  +  Q(Δt)
    /// ```
    /// where:
    /// - `Σ_pos`   is the 2×2 position covariance at `epoch_mid` (diagonal here),
    /// - `Σ_vel`   is the 2×2 velocity covariance (diagonal here),
    /// - `Q(Δt)`   is an **isotropic** (per-axis) model-noise term that grows with |Δt|
    ///   to cover unmodeled curvature / model mismatch (see [`ModelNoise`]).
    ///
    /// Notes on the approximation
    /// --------------------------
    /// - **Independence:** we **ignore** cross-terms such as `2 Δt · Cov(x, v)` and any
    ///   off-diagonal coupling between X/Y; hence `Σ_p(t)` is diagonal here. This is
    ///   sufficient for **cone sizing** and candidate retrieval.
    /// - **Acceleration uncertainty:** the returned covariance does **not** include an
    ///   explicit `σ_a²` term (e.g., `~ 0.25 Δt⁴ σ_a²`). If you later track an
    ///   acceleration uncertainty, extend the model accordingly.
    /// - **PSD safety:** with non-negative inputs (`Σ_pos ≥ 0`, `Σ_vel ≥ 0`, `Q(Δt) ≥ 0`),
    ///   each returned variance is non-negative. If numerical round-off yields a tiny
    ///   negative, clamp to zero before square roots downstream.
    ///
    /// Return
    /// ------
    /// * `([px, py], [[sxx, 0], [0, syy]])` — mean plane position and a diagonal covariance
    ///   matrix at `t_target` (units below).
    ///
    /// Units
    /// -----
    /// * `px, py` in **radians** (tangent plane),
    /// * `sxx, syy` in **radians²**,
    /// * `Δt` in **days** (MJD TT),
    /// * `vel_xy` in **radians/day**, `acc_xy` in **radians/day²**.
    ///
    /// When to use this
    /// ----------------
    /// - Use this for **cone prediction** (via [`SeedNode::predict_cone`]) and as the
    ///   base covariance for **Mahalanobis** scoring in the plane.
    /// - For orbit-quality filtering or tight residual analysis, prefer a fuller
    ///   propagation that keeps cross-covariances and (optionally) acceleration
    ///   uncertainty.
    #[inline]
    fn predict_on_plane(&self, t_target: f64, noise: ModelNoise) -> ([f64; 2], [[f64; 2]; 2]) {
        let dt = t_target - self.epoch_mid;

        // ---- Mean position on the plane -------------------------------------
        // Start from p0 and advance with v; include the quadratic term if acceleration is present.
        let mut px = self.pos_xy[0] + self.vel_xy[0] * dt;
        let mut py = self.pos_xy[1] + self.vel_xy[1] * dt;
        if let Some(a) = self.acc_xy {
            // Add 0.5 * a * Δt² per axis
            px += 0.5 * a[0] * dt * dt;
            py += 0.5 * a[1] * dt * dt;
        }

        // ---- Variance (diagonal heuristic) ----------------------------------
        // Q(Δt) grows with |Δt| to cover unmodeled curvature / mismatch.
        let q = noise.q0 + noise.q1 * dt.abs() + noise.q2 * dt * dt;

        // Per-axis variance: σ_pos² + (Δt²) σ_vel² + Q(Δt).
        // Off-diagonals are set to zero by design (see doc above).
        let sxx = self.cov_pos[0][0] + dt * dt * self.cov_vel[0][0] + q;
        let syy = self.cov_pos[1][1] + dt * dt * self.cov_vel[1][1] + q;

        ([px, py], [[sxx, 0.0], [0.0, syy]])
    }
}

/// Extract **constant-velocity** kinematic features for **pair seeds** `(a, b)` with `t_b > t_a`.
///
/// # Overview
/// For each pair, we:
/// - Compute a robust **spherical midpoint** of `(a, b)` to define a local
///   **gnomonic tangent plane**.
/// - Project both detections to plane coordinates and estimate:
///   - the **mid-position** (average of endpoints),
///   - the **velocity** `v ≈ (p_b - p_a) / Δt`.
/// - Build simple **position/velocity covariances** from the per-detection
///   astrometric errors (`ra_err`, `dec_err`) and the time baseline `Δt`.
/// - Summarize **flux** photometry (mean & dispersion in nJy).
///
/// The result is a `SeedNode` with `acc_xy = None`. Any **curvature** will be
/// handled downstream by inflating prediction covariances (process noise) or
/// by later promotion to a triplet/longer arc.
///
/// # Arguments
/// - `store` – Immutable alert container (angles in radians, epochs in MJD TT).
/// - `pairs` – Slice of `(AlertId, AlertId)` with strictly increasing times.
/// - `params` – Feature extraction knobs (e.g., optional speed guardrail).
/// - `night_id` – Night identifier to attach to the produced nodes (use `-1` if unknown).
///
/// # Return
/// A vector of [`SeedNode`] (one per input pair). Pairs violating
/// `max_speed_rad_per_day` (if set) are silently skipped.
///
/// # Notes
/// - **Units:** angles in radians, time in **MJD (TT)** days, velocities in rad/day.
/// - **Covariance heuristic:**
///   - position variance per axis uses the **mean squared** of endpoint errors;
///   - velocity variance per axis scales roughly as `2·σ² / Δt²`.
/// - **Band:** we forward the band of the first alert (`a.band`) as a representative.
///
/// # Examples
/// ```ignore
/// let features = extract_pair_features(&store, &pairs, FeatureExtractParams {
///     max_speed_rad_per_day: Some(0.05), // ≈ 2.9 deg/day
/// }, night_id);
/// assert!(!features.is_empty());
/// ```
///
/// # See also
/// - [`extract_triplet_features`]
pub fn extract_pair_features(
    store: &AlertStore,
    pairs: &Pairs,
    params: FeatureExtractParams,
    night_id: i32,
) -> Vec<SeedNode> {
    let mut out = Vec::with_capacity(pairs.len());
    for (seed_id, &(ia, ib)) in pairs.iter().enumerate() {
        let a = &store.alerts[ia as usize];
        let b = &store.alerts[ib as usize];

        let ta = a.mjd_tt;
        let tb = b.mjd_tt;
        let tm = 0.5 * (ta + tb);
        let dt = tb - ta;
        let inv_dt = 1.0 / dt;
        let inv_dt2 = inv_dt * inv_dt;

        let (ra0, dec0) = spherical_midpoint(a.ra, a.dec, b.ra, b.dec);
        let pa = radec_to_tangent(a.ra, a.dec, ra0, dec0);
        let pb = radec_to_tangent(b.ra, b.dec, ra0, dec0);

        let pm = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];

        let (ra0, dec0) = spherical_midpoint(a.ra, a.dec, b.ra, b.dec);
        let pa = radec_to_tangent(a.ra, a.dec, ra0, dec0);
        let pb = radec_to_tangent(b.ra, b.dec, ra0, dec0);

        let (ra_mid, dec_mid) = tangent_to_radec(pm[0], pm[1], ra0, dec0);

        // v = (pb - pa) / dt  -> use inv_dt (one division)
        let vx = (pb[0] - pa[0]) * inv_dt;
        let vy = (pb[1] - pa[1]) * inv_dt;

        // Speed guardrail without sqrt: compare squared norms
        if let Some(vmax) = params.max_speed_rad_per_day {
            let speed2 = vx.mul_add(vx, vy * vy);
            let vmax2 = vmax * vmax;
            if speed2 > vmax2 {
                continue;
            }
        }

        // Covariances: reuse inv_dt²
        let sa = a.ra_err.max(a.dec_err);
        let sb = b.ra_err.max(b.dec_err);
        let s2 = 0.5 * (sa * sa + sb * sb);
        let cov_pos = [[s2, 0.0], [0.0, s2]];
        let vel_var = 2.0 * s2 * inv_dt2;
        let cov_vel = [[vel_var, 0.0], [0.0, vel_var]];

        let flux_mean = (a.flux + b.flux) * 0.5;
        let flux_std = ((a.flux - flux_mean).abs() + (b.flux - flux_mean).abs()) * 0.5;

        out.push(SeedNode {
            seed_id: seed_id as u64,
            night_id,
            epoch_mid: tm,
            pos_xy: pm,
            vel_xy: [vx, vy],
            cov_pos,
            cov_vel,
            acc_xy: None,
            flux_mean,
            flux_std,
            band: a.band,
            n_obs: 2,
            members: vec![ia, ib],
            center_ra: ra0,
            center_dec: dec0,
            ra_mid,
            dec_mid,
        });
    }
    out
}

/// Extract **quadratic** kinematic features for **triplet seeds** `(a, b, c)` with `t_a < t_b < t_c`.
///
/// Overview
/// --------
/// Triplets allow us to estimate a **small curvature** on the tangent plane by
/// fitting the 1D polynomial:
/// `x(t) = p0x + vx·(t−tm) + 0.5·ax·(t−tm)²` (and same for `y`),
/// where `tm` is the average epoch. We still use a gnomonic plane centered at a
/// robust spherical midpoint to reduce projection distortion.
///
/// Arguments
/// ---------
/// - `store` – Immutable alert container (angles in radians, epochs in MJD TT).
/// - `trips` – Slice of `(AlertId, AlertId, AlertId)` with strictly increasing times.
/// - `params` – Feature extraction knobs (e.g., speed guardrail if you reuse it).
/// - `night_id` – Night identifier to attach to the produced nodes.
///
/// Return
/// ------
/// A vector of [`SeedNode`] (one per input triplet) with `acc_xy = Some([ax, ay])`.
///
/// Notes
/// -----
/// - **Covariance heuristic:** compared to pairs, position variance per axis scales
///   like `σ²/3`, and a characteristic `Δt` across the triplet is used to scale
///   velocity variance (`~ σ² / Δt²`). This is intentionally simple and can be
///   replaced by the LS covariance if you later propagate uncertainties properly.
/// - **Band:** forwarded from the first alert (`a.band`) as representative.
/// - **Curvature usage downstream:** `acc_xy` is optional; if you later choose a
///   constant-velocity predictor, consider adding process noise growing with |Δt|.
///
/// # Examples
/// ```ignore
/// let features = extract_triplet_features(&store, &triplets, FeatureExtractParams {
///     max_speed_rad_per_day: None,
/// }, night_id);
/// assert!(features.iter().all(|s| s.acc_xy.is_some()));
/// ```
///
/// See also
/// --------
/// - [`extract_pair_features`]
pub fn extract_triplet_features(
    store: &AlertStore,
    trips: &Triplets,
    night_id: i32,
) -> Vec<SeedNode> {
    let mut out = Vec::with_capacity(trips.len());
    for (seed_id, &(ia, ib, ic)) in trips.iter().enumerate() {
        let a = &store.alerts[ia as usize];
        let b = &store.alerts[ib as usize];
        let c = &store.alerts[ic as usize];

        let (ta, tb, tc) = (a.mjd_tt, b.mjd_tt, c.mjd_tt);
        let tm = (ta + tb + tc) / 3.0;

        let (ra0, dec0) = spherical_midpoint(a.ra, a.dec, c.ra, c.dec);
        let pa = radec_to_tangent(a.ra, a.dec, ra0, dec0);
        let pb = radec_to_tangent(b.ra, b.dec, ra0, dec0);
        let pc = radec_to_tangent(c.ra, c.dec, ra0, dec0);

        let (p0x, vx, ax) = fit_quad_1d([ta - tm, tb - tm, tc - tm], [pa[0], pb[0], pc[0]]);
        let (p0y, vy, ay) = fit_quad_1d([ta - tm, tb - tm, tc - tm], [pa[1], pb[1], pc[1]]);

        let (ra_mid, dec_mid) = tangent_to_radec(p0x, p0y, ra0, dec0);

        let sa = a.ra_err.max(a.dec_err);
        let sb = b.ra_err.max(b.dec_err);
        let sc = c.ra_err.max(c.dec_err);
        let s2 = (sa * sa + sb * sb + sc * sc) / 3.0;

        let dt_char = (tc - ta).max(1e-6);
        let inv_dt2 = 1.0 / (dt_char * dt_char);

        let cov_pos = [[s2 / 3.0, 0.0], [0.0, s2 / 3.0]];
        let vel_var = s2 * inv_dt2;
        let cov_vel = [[vel_var, 0.0], [0.0, vel_var]];

        let flux_mean = (a.flux + b.flux + c.flux) / 3.0;
        let flux_std =
            ((a.flux - flux_mean).abs() + (b.flux - flux_mean).abs() + (c.flux - flux_mean).abs())
                / 3.0;

        out.push(SeedNode {
            seed_id: seed_id as u64,
            night_id,
            epoch_mid: tm,
            pos_xy: [p0x, p0y],
            vel_xy: [vx, vy],
            cov_pos,
            cov_vel,
            acc_xy: Some([ax, ay]),
            flux_mean,
            flux_std,
            band: a.band,
            n_obs: 3,
            members: vec![ia, ib, ic],
            center_ra: ra0,
            center_dec: dec0,
            ra_mid,
            dec_mid,
        });
    }
    out
}

/// Type alias for a seed identifier.
///
/// Overview
/// --------
/// `SeedId` uniquely identifies a `SeedNode` **within a given extraction batch / night**.
/// It is the handle returned by spatial queries and used to reference back to the
/// concrete `SeedNode` record (typically by indexing an external slice or map).
///
/// Notes
/// -----
/// - The identifier is local to the current process/run and does not imply any
///   cross-night/global stability unless you establish one externally (e.g., by
///   storing `(night_id, seed_id)` pairs).
pub type SeedId = u64;

/// Per-night spatial index of seeds with **cell-based cone coverage** (e.g., HEALPix/HTM).
///
/// Overview
/// --------
/// `SeedSpatialIndex` maps a **spatial cell key** to the list of `SeedId`s whose
/// mid-epoch sky positions (`ra_mid`, `dec_mid`) fall inside that cell. A cone
/// query is answered by enumerating all cells that **intersect** the requested
/// circle and streaming their members. This returns a **superset** of the true
/// sky-cone result (some seeds may lie in cells that intersect the cone boundary
/// but are actually outside the circle). Downstream **fine filtering** (e.g.,
/// great-circle distance or Mahalanobis in a local plane) should be applied.
///
/// Design
/// ------
/// - Backing store: `AHashMap<SpatialKey, Vec<SeedId>>` for throughput on large
///   per-night volumes.
/// - One seed → one cell: each `SeedId` is assigned to exactly **one** cell based
///   on `(ra_mid, dec_mid)`. Queries cover multiple cells and **union** their members.
/// - Query semantics: **approximate superset** by construction; no dedup is performed,
///   assuming `neighbors()` returns a unique set of cell keys.
///
/// Units
/// -----
/// - `(ra_mid, dec_mid)` in **radians** (ICRS).
/// - Query radius in **radians**.
///
/// Invariants
/// ----------
/// - The index is **read-only** after `build()`; the internal vectors may be returned
///   in arbitrary order (hash iteration order). Do not rely on iteration stability.
///
/// Complexity
/// ----------
/// - Build: `O(N)` over seeds (single pass).
/// - Query: `O(C + M)` where `C` is the number of covered cells and `M` the total
///   number of members across those cells. This is a **streaming** iterator; collect
///   only if you need ownership.
///
/// Limitations
/// -----------
/// - No **exact geometry**: this is cell coverage, not spherical polygon slicing.
///   Expect **false positives** near cone boundaries.
/// - No **deduplication**: if a `SpatialBinner` ever returns duplicate keys from
///   `neighbors()`, duplicates could leak; callers can wrap results in a set if needed.
///
/// See also
/// --------
/// - [`SpatialBinner`] — provides `key_for`, `neighbors`, and `cell_radius`.
/// - [`SeedNode::predict_cone`] — to compute the (RA, Dec, radius) for a query.
#[derive(Default, Debug)]
pub struct SeedSpatialIndex {
    by_cell: AHashMap<SpatialKey, Vec<SeedId>>,
}

impl SeedSpatialIndex {
    /// Build a per-night seed index using each seed's cached `(ra_mid, dec_mid)`.
    ///
    /// Overview
    /// --------
    /// Computes the spatial cell for every `SeedNode` at its mid-epoch sky position
    /// and appends the `seed_id` to that cell’s membership list. This prepares the
    /// structure for **cell-based cone queries**.
    ///
    /// Arguments
    /// ---------
    /// * `seeds`   — Slice of seeds (must have valid `ra_mid`, `dec_mid`).
    /// * `binner`  — Spatial partitioner (e.g., HEALPix/HTM grid) implementing [`SpatialBinner`].
    ///
    /// Return
    /// ------
    /// * `SeedSpatialIndex` — ready to be queried for this night.
    ///
    /// Complexity
    /// ----------
    /// * Time: `O(N)`; Memory: `O(N)` for the cell → members map.
    ///
    /// Notes
    /// -----
    /// * This function does **not** sort members; iteration order is unspecified.
    /// * Provide a `binner` with a depth that yields cells no larger than your typical
    ///   search cones (overly coarse cells increase false positives).
    pub fn build<Bs: SpatialBinner>(seeds: &[SeedNode], binner: &Bs) -> Self {
        let mut by_cell: AHashMap<SpatialKey, Vec<SeedId>> = AHashMap::new();
        for s in seeds {
            let key = binner.key_for(s.ra_mid, s.dec_mid);
            by_cell.entry(key).or_default().push(s.seed_id);
        }
        Self { by_cell }
    }

    /// Approximate **cone search**: return all `SeedId`s in cells that intersect `(ra, dec, radius)`.
    ///
    /// Overview
    /// --------
    /// Computes the center cell for `(ra, dec)` and fetches all **neighbor cells**
    /// whose union **covers** a circular cone of the requested `radius`, as defined
    /// by `binner.neighbors`. Returns the concatenation of member lists from those
    /// cells as a **lazy iterator**.
    ///
    /// Arguments
    /// ---------
    /// * `binner` — Spatial partitioner that provides `key_for` and `neighbors`.
    /// * `ra`     — Right ascension of cone center (radians, ICRS).
    /// * `dec`    — Declination of cone center (radians, ICRS).
    /// * `radius` — Cone radius (radians).
    ///
    /// Return
    /// ------
    /// * `impl Iterator<Item = SeedId>` — a **superset** of true cone members; may
    ///   include seeds just outside the circle due to cell coverage.
    ///
    /// Semantics & Post-filtering
    /// --------------------------
    /// - The iterator may yield candidates outside the exact cone; downstream code
    ///   should apply a **precise angular filter** (e.g., haversine or dot-product)
    ///   or a **Mahalanobis** test in a tangent plane.
    /// - No **deduplication** is performed. If your `SpatialBinner` can return
    ///   duplicate keys, wrap the iterator in a set or `itertools::unique`.
    ///
    /// Complexity
    /// ----------
    /// - `O(C + M)` where `C` is the number of covered cells and `M` the total
    ///   number of seed memberships across them.
    ///
    /// Examples
    /// --------
    /// ```ignore
    /// // Predict cone for a seed and query candidates in the target night:
    /// let (ra, dec, r) = seed.predict_cone(t_target, &binner, params);
    /// let iter = index.cone_query(&binner, ra, dec, r);
    /// // If you need unique candidates:
    /// use ahash::AHashSet;
    /// let unique: AHashSet<_> = iter.collect();
    /// ```
    pub fn cone_query<'a, Bs: SpatialBinner>(
        &'a self,
        binner: &'a Bs,
        ra: f64,
        dec: f64,
        radius: f64,
    ) -> impl Iterator<Item = SeedId> + 'a {
        let key_center = binner.key_for(ra, dec);
        let cover = binner.neighbors(key_center, radius);
        cover
            .into_iter()
            .filter_map(move |k| self.by_cell.get(&k))
            .flatten()
            .copied()
    }
}

/* --------------------------- Private utilities --------------------------- */

/// Compute a robust spherical midpoint between two directions (ra, dec).
///
/// Overview
/// --------
/// Returns the angular mean of two ICRS directions using **vector averaging**:
/// convert both points to Cartesian unit vectors, add them, normalize, then
/// convert back to spherical coordinates. This is not the geodesic midpoint
/// at fixed arclength along the great circle, but it is stable and well-suited
/// to define a local tangent plane center for small separations.
///
/// Arguments
/// ---------
/// - `ra1`, `dec1` — First direction in radians.
/// - `ra2`, `dec2` — Second direction in radians.
///
/// Return
/// ------
/// - `(ra_mid, dec_mid)` in radians. `ra_mid` is normalized to `[0, 2π)`.
///
/// Notes
/// -----
/// - If the two directions are nearly opposite, the vector sum is ill-defined
///   and the normalization may amplify floating-point noise. In practice we
///   guard the norm with a small floor to avoid division by zero.
/// - Use this midpoint only to minimize gnomonic projection distortion. It is
///   not intended to represent a physical center.
///
/// Units
/// -----
/// - Angles are in radians.
///
/// See also
/// --------
/// - [`radec_to_tangent`] to project around the returned center.
#[inline]
fn spherical_midpoint(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> (f64, f64) {
    let (x1, y1, z1) = sph_to_cart(ra1, dec1);
    let (x2, y2, z2) = sph_to_cart(ra2, dec2);
    let (x, y, z) = (x1 + x2, y1 + y2, z1 + z2);
    let r = (x * x + y * y + z * z).sqrt().max(NORM_MIN);
    cart_to_sph(x / r, y / r, z / r)
}

/// Gnomonic projection of a sky position onto a tangent plane centered at (ra0, dec0).
///
/// Overview
/// --------
/// Projects the ICRS direction `(ra, dec)` onto the plane tangent to the unit
/// sphere at `(ra0, dec0)`, using the exact gnomonic formula. The resulting
/// coordinates `(x, y)` are in radians on the tangent plane and are suitable
/// for small-angle kinematics (e.g., constant-velocity fits).
///
/// Arguments
/// ---------
/// - `ra`, `dec` — Target direction in radians.
/// - `ra0`, `dec0` — Tangent point in radians (projection center).
///
/// Return
/// ------
/// - `[x, y]` in radians on the tangent plane.
///
/// Notes
/// -----
/// - The gnomonic projection has a singularity on the great circle 90° from the
///   center; numerically, when `cosc -> 0` the coordinates diverge. We clamp
///   the inverse by `max(1e-12)` to avoid Inf/NaN, but the values are not
///   meaningful near the singularity. Use this only for small cones around
///   the center (a few degrees).
/// - The formulas follow the standard:
///   `cosc = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra - ra0)`
///   `x = cos(dec) sin(ra - ra0) / cosc`
///   `y = [cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra - ra0)] / cosc`
///
/// Units
/// -----
/// - Angles in radians.
///
/// See also
/// --------
/// - [`spherical_midpoint`] to pick a stable plane center.
///
/// Examples
/// --------
/// ```ignore
/// // Center at object average, then project a and b:
/// let (ra0, dec0) = spherical_midpoint(ra_a, dec_a, ra_b, dec_b);
/// let pa = radec_to_tangent(ra_a, dec_a, ra0, dec0);
/// let pb = radec_to_tangent(ra_b, dec_b, ra0, dec0);
/// ```
#[inline]
fn radec_to_tangent(ra: f64, dec: f64, ra0: f64, dec0: f64) -> [f64; 2] {
    // Precompute sin/cos with a single call per angle
    let (sdec, cdec) = dec.sin_cos();
    let (sdec0, cdec0) = dec0.sin_cos();
    let dra = ra - ra0;
    let (sdra, cdra) = dra.sin_cos();

    // cosc = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(dra)
    let cosc = cdec0 * cdec * cdra + sdec0 * sdec;
    let inv = 1.0 / cosc.max(INV_COSC_MIN);

    // x =  cos(dec) sin(dra) / cosc
    // y = (cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(dra)) / cosc
    let x = cdec * sdra * inv;
    let y = (cdec0 * sdec - sdec0 * cdec * cdra) * inv;
    [x, y]
}

/// Convert spherical angles (ra, dec) to a Cartesian unit vector (x, y, z).
///
/// Overview
/// --------
/// Maps an ICRS direction to its 3D unit vector on the unit sphere, using the
/// convention:
/// `x = cos(dec) cos(ra)`, `y = cos(dec) sin(ra)`, `z = sin(dec)`.
///
/// Arguments
/// ---------
/// - `ra`, `dec` — Angles in radians.
///
/// Return
/// ------
/// - `(x, y, z)` such that `x^2 + y^2 + z^2 = 1` within floating-point error.
///
/// Units
/// -----
/// - Angles in radians.
/// - Output is dimensionless.
#[inline]
fn sph_to_cart(ra: f64, dec: f64) -> (f64, f64, f64) {
    let (sdec, cdec) = dec.sin_cos();
    let (sra, cra) = ra.sin_cos();
    (cdec * cra, cdec * sra, sdec)
}

/// Convert a Cartesian vector (x, y, z) back to spherical angles (ra, dec).
///
/// Overview
/// --------
/// Inverse of [`sph_to_cart`]. The right ascension is returned in `[0, 2π)`
/// via `atan2(y, x).rem_euclid(2π)`. The declination is `asin(z / r)`, where
/// `r = sqrt(x^2 + y^2 + z^2)`.
///
/// Arguments
/// ---------
/// - `x`, `y`, `z` — Cartesian components. They do not need to be normalized.
///
/// Return
/// ------
/// - `(ra, dec)` in radians, with `ra` normalized to `[0, 2π)`.
///
/// Notes
/// -----
/// - If `(x, y, z)` is the zero vector or underflows, `r` becomes very small.
///   We do not clamp here; callers should ensure the input is meaningful.
///   In practice this is fed by normalized sums with a guard in the caller.
///
/// Units
/// -----
/// - Angles in radians.
#[inline]
fn cart_to_sph(x: f64, y: f64, z: f64) -> (f64, f64) {
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let inv_r = 1.0 / r;
    let dec = (z * inv_r).asin();
    let ra = y.atan2(x).rem_euclid(TWO_PI);
    (ra, dec)
}

/// Fit a quadratic through three samples x(t) = p0 + v·t + 0.5·a·t² at given times.
///
/// Overview
/// --------
/// Given three samples `(t_k, x_k)` for `k = 0..2`, solve exactly for the
/// coefficients `(p0, v, a)` of a quadratic in time. This is used to extract a
/// small apparent curvature from intra-night **triplets** on the tangent plane.
/// The times should be passed **relative to a reference** (e.g., `t - t_mid`)
/// to keep coefficients well-scaled.
///
/// Arguments
/// ---------
/// - `dt` — Sample times `[t0, t1, t2]` relative to a chosen origin (e.g., mid-epoch).
/// - `x`  — Sample values `[x0, x1, x2]` at those times.
///
/// Return
/// ------
/// - `(p0, v, a)` where:
///   - `p0` is the value at `t = 0`,
///   - `v` is the first derivative at `t = 0`,
///   - `a` is the second derivative at `t = 0`.
///
/// Notes
/// -----
/// - The method uses finite-difference identities and requires **distinct** times.
///   If any two times are equal or nearly equal, numerical stability degrades.
/// - For robust uncertainty estimates, prefer a least-squares solve and compute
///   the covariance matrix. Here, with exactly three points, the solution is exact.
///
/// Units
/// -----
/// - `t` in days (MJD TT if used with alert times).
/// - `x` in radians if used on a tangent plane axis.
/// - `v` in x-units per day; `a` in x-units per day^2.
///
/// Examples
/// --------
/// ```ignore
/// // Center times around the average to keep coefficients well-scaled:
/// let tm = (ta + tb + tc) / 3.0;
/// let (p0, v, a) = fit_quad_1d([ta - tm, tb - tm, tc - tm], [xa, xb, xc]);
/// ```
#[inline]
fn fit_quad_1d(dt: [f64; 3], x: [f64; 3]) -> (f64, f64, f64) {
    let (t0, t1, t2) = (dt[0], dt[1], dt[2]);
    let inv_01 = 1.0 / (t1 - t0);
    let inv_12 = 1.0 / (t2 - t1);
    let d01 = (x[1] - x[0]) * inv_01;
    let d12 = (x[2] - x[1]) * inv_12;
    let inv_20 = 1.0 / (t2 - t0);
    let a = 2.0 * (d12 - d01) * inv_20;
    let v = d01 - 0.5 * a * (t0 + t1);
    let p0 = x[1] - v * t1 - 0.5 * a * t1 * t1;
    (p0, v, a)
}

/// Return the largest eigenvalue λ_max of a **symmetric 2×2** matrix (robust formulation).
///
/// Overview
/// --------
/// For a symmetric matrix
/// ```text
/// A = [[a11, a12],
///      [a12, a22]],
/// ```
/// the eigenvalues have the closed form:
/// ```text
/// λ_{1,2} = 0.5 * (tr ± sqrt((a11 − a22)^2 + 4 a12^2))
/// where tr = a11 + a22.
/// ```
/// We return `λ_max = max(λ₁, λ₂) = 0.5 * (tr + rad)` with
/// `rad = sqrt((a11 − a22)^2 + (2 a12)^2)`.
///
/// Why this implementation?
/// ------------------------
/// - We compute `rad` via `hypot(a11 − a22, 2*a12)` which is **more stable** than
///   `( (a11 − a22)^2 + (2*a12)^2 ).sqrt()`:
///   it reduces overflow/underflow and avoids tiny negative discriminants from
///   round-off (so **no clamp** is needed).
/// - We **symmetrize** the off-diagonal term (`a12 = 0.5 * (a01 + a10)`) to guard
///   against small asymmetries in the input (e.g., numerical drift). For perfectly
///   symmetric input this is a no-op.
///
/// Assumptions & units
/// -------------------
/// - Input is intended to be **symmetric** and (for covariances) **PSD**. If the
///   matrix is not PSD, `λ_max` may be negative (by definition).
/// - The return value has the same units as the entries of `A` (e.g., **rad²** for
///   a covariance on the tangent plane).
///
/// Usage note (cone sizing)
/// ------------------------
/// When turning a covariance into a search **circle** you will often compute
/// `radius = k_sigma * sqrt(λ_max)`. If you suspect rare negative `λ_max` from
/// non-PSD inputs or round-off, clamp before the square root:
/// `let radius = k_sigma * (lambda_max_2x2(A).max(0.0)).sqrt();`
///
/// Complexity
/// ----------
/// O(1).
///
/// Examples
/// --------
/// ```ignore
/// // Diagonal case → largest diagonal entry
/// let a = [[2.0, 0.0],
///          [0.0, 1.0]];
/// assert_eq!(lambda_max_2x2(a), 2.0);
///
/// // With correlation
/// let a = [[2.0, 0.3],
///          [0.3, 1.0]];
/// let lmax = lambda_max_2x2(a);
/// assert!(lmax > 2.0 && lmax < 2.1);
/// ```
#[inline]
fn lambda_max_2x2(a: [[f64; 2]; 2]) -> f64 {
    // Unpack entries. We also read a[1][0] and average to enforce symmetry
    // in case the caller supplied a matrix with tiny asymmetry.
    let a11 = a[0][0];
    let a22 = a[1][1];
    let a12 = 0.5 * (a[0][1] + a[1][0]); // symmetrize off-diagonal

    let tr = a11 + a22;

    // Compute the “radius” term robustly:
    //   rad = sqrt( (a11 - a22)^2 + (2 a12)^2 ) = hypot(a11 - a22, 2 a12)
    // Using hypot avoids catastrophic cancellation and eliminates the need
    // for clamping a tiny negative discriminant before sqrt.
    let rad = (a11 - a22).hypot(2.0 * a12);

    // Largest eigenvalue: 0.5 * (trace + rad)
    0.5 * (tr + rad)
}

/// Inverse **gnomonic** projection: map plane coordinates `(x, y)` back to
/// sky coordinates `(ra, dec)` around a given tangent point `(ra0, dec0)`.
///
/// Overview
/// --------
/// This inverts the gnomonic projection used by [`radec_to_tangent`]. Given a
/// local tangent plane centered at `(ra0, dec0)`, it converts plane coordinates
/// (in **radians on the plane**) back to the ICRS angles `(ra, dec)`:
///
/// Let:
/// - `ρ = sqrt(x² + y²)`
/// - `c = atan(ρ)`
/// - `(sin c, cos c)` computed from `c`
/// - `(sin dec0, cos dec0)` from `dec0`
///
/// Then:
/// ```text
/// dec = asin( cos c * sin dec0 + (y * sin c * cos dec0) / ρ )
/// ra  = ra0 + atan2( x * sin c,  ρ * cos dec0 * cos c - y * sin dec0 * sin c )
/// ```
/// Finally, `ra` is normalized to `[0, 2π)`.
///
/// Arguments
/// ---------
/// * `x`, `y`  — Plane coordinates (radians) in the tangent plane of `(ra0, dec0)`.
/// * `ra0`     — Tangent point right ascension (radians, ICRS).
/// * `dec0`    — Tangent point declination (radians, ICRS).
///
/// Return
/// ------
/// * `(ra, dec)` — Sky coordinates in **radians** (ICRS), with `ra ∈ [0, 2π)`.
///
/// Units
/// -----
/// * All angles are in **radians**. `(x, y)` are small-angle displacements on the
///   plane (consistent with outputs of `radec_to_tangent`).
///
/// Numerical behavior & edge cases
/// -------------------------------
/// * When `ρ` is extremely small (near the tangent point), we return the center
///   `(ra0, dec0)` to avoid division by very small `ρ` (continuous limit).
/// * The gnomonic model becomes ill-conditioned near the great circle 90° from the
///   center (`cos c → 0`). In practice, seeds and cones are kept **small**, so the
///   inverse is stable.
/// * `atan2` ensures proper quadrant for `ra`. The final `rem_euclid(2π)` normalizes RA.
///
/// Examples
/// --------
/// ```ignore
/// // Round-trip with the forward projection:
/// let (ra0, dec0) = (1.0, 0.2);
/// let (x, y) = radec_to_tangent(ra0 + 1e-4, dec0 - 2e-4, ra0, dec0);
/// let (ra, dec) = tangent_to_radec(x, y, ra0, dec0);
/// assert!((ra - (ra0 + 1e-4)).abs() < 1e-12);
/// assert!((dec - (dec0 - 2e-4)).abs() < 1e-12);
/// ```
#[inline]
fn tangent_to_radec(x: f64, y: f64, ra0: f64, dec0: f64) -> (f64, f64) {
    let rho2 = x * x + y * y;
    if rho2 < 1e-24 {
        return (ra0.rem_euclid(TWO_PI), dec0);
    }
    let rho = rho2.sqrt();
    let c = rho.atan();
    let (sc, cc) = c.sin_cos();
    let (s0, c0) = dec0.sin_cos();

    let dec = (cc * s0 + (y * sc * c0) / rho).asin();
    let denom = rho * c0 * cc - y * s0 * sc;
    let ra = ra0 + (x * sc).atan2(denom);
    (ra.rem_euclid(TWO_PI), dec)
}

#[cfg(test)]
mod feature_extract_tests {
    use crate::alerts::Alert;

    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    /* --------------------------- Test helpers --------------------------- */

    #[allow(clippy::too_many_arguments)]
    fn make_alert(
        id: AlertId,
        ra: f64,
        dec: f64,
        mjd_tt: f64,
        flux: f32,
        flux_err: f32,
        band: u8,
        ra_err: f64,
        dec_err: f64,
    ) -> Alert {
        Alert {
            id,
            dia_source_id: id as u64,
            ra,
            ra_err,
            dec,
            dec_err,
            mjd_tt,
            flux,
            flux_err,
            band,
        }
    }

    fn make_store(mut alerts: Vec<Alert>) -> AlertStore {
        // Ensure IDs are consistent with index.
        for (i, a) in alerts.iter_mut().enumerate() {
            a.id = i as AlertId;
        }
        let start_mjd = alerts
            .iter()
            .map(|a| a.mjd_tt)
            .fold(f64::INFINITY, f64::min)
            .floor();
        AlertStore { start_mjd, alerts }
    }

    /// Inverse gnomonic: from plane (x,y) back to (ra, dec) for a given center (ra0, dec0).
    /// Formulas from Snyder; consistent with radec_to_tangent for small cones.
    fn tangent_to_radec(x: f64, y: f64, ra0: f64, dec0: f64) -> (f64, f64) {
        let rho = (x * x + y * y).sqrt();
        if rho < 1e-18 {
            return (ra0.rem_euclid(2.0 * PI), dec0);
        }
        let c = rho.atan();
        let (sin_c, cos_c) = (c.sin(), c.cos());
        let sin_dec0 = dec0.sin();
        let cos_dec0 = dec0.cos();

        let dec = (cos_c * sin_dec0 + (y * sin_c * cos_dec0) / rho).asin();
        let denom = rho * cos_dec0 * cos_c - y * sin_dec0 * sin_c;
        let ra = ra0 + (x * sin_c).atan2(denom);
        (ra.rem_euclid(2.0 * PI), dec)
    }

    /// Build a synthetic pair from a plane model p(t) = p0 + v * (t - tm),
    /// convert to (ra,dec), create `Alert`s and return store + pair.
    #[allow(clippy::too_many_arguments)]
    fn synthetic_pair_from_plane(
        ra0: f64,
        dec0: f64,
        tm: f64,
        dt: f64,
        p0: [f64; 2],
        v: [f64; 2],
        band: u8,
        ra_err: f64,
        dec_err: f64,
        fluxes: (f32, f32),
    ) -> (AlertStore, Vec<(AlertId, AlertId)>) {
        let t_a = tm - 0.5 * dt;
        let t_b = tm + 0.5 * dt;

        let pa = [p0[0] - 0.5 * v[0] * dt, p0[1] - 0.5 * v[1] * dt];
        let pb = [p0[0] + 0.5 * v[0] * dt, p0[1] + 0.5 * v[1] * dt];

        let (ra_a, dec_a) = tangent_to_radec(pa[0], pa[1], ra0, dec0);
        let (ra_b, dec_b) = tangent_to_radec(pb[0], pb[1], ra0, dec0);

        let a = make_alert(
            0 as AlertId,
            ra_a,
            dec_a,
            t_a,
            fluxes.0,
            0.0,
            band,
            ra_err,
            dec_err,
        );
        let b = make_alert(
            1 as AlertId,
            ra_b,
            dec_b,
            t_b,
            fluxes.1,
            0.0,
            band,
            ra_err,
            dec_err,
        );

        let store = make_store(vec![a, b]);
        let pairs = vec![(0 as AlertId, 1 as AlertId)];
        (store, pairs)
    }

    /// Build a synthetic triplet from a plane quadratic model:
    /// p(t) = p0 + v*(t-tm) + 0.5*a*(t-tm)^2
    #[allow(clippy::too_many_arguments)]
    fn synthetic_triplet_from_plane(
        ra0: f64,
        dec0: f64,
        tm: f64,
        dt_char: f64, // characteristic baseline: we build t = [-Δ, 0, +Δ]
        p0: [f64; 2],
        v: [f64; 2],
        a: [f64; 2],
        band: u8,
        ra_err: f64,
        dec_err: f64,
        fluxes: (f32, f32, f32),
    ) -> (AlertStore, Vec<(AlertId, AlertId, AlertId)>) {
        let t_a = tm - dt_char;
        let t_b = tm;
        let t_c = tm + dt_char;

        let dt_a = -dt_char;
        let dt_b = 0.0;
        let dt_c = dt_char;

        let pa = [
            p0[0] + v[0] * dt_a + 0.5 * a[0] * dt_a * dt_a,
            p0[1] + v[1] * dt_a + 0.5 * a[1] * dt_a * dt_a,
        ];
        let pb = [
            p0[0] + v[0] * dt_b + 0.5 * a[0] * dt_b * dt_b,
            p0[1] + v[1] * dt_b + 0.5 * a[1] * dt_b * dt_b,
        ];
        let pc = [
            p0[0] + v[0] * dt_c + 0.5 * a[0] * dt_c * dt_c,
            p0[1] + v[1] * dt_c + 0.5 * a[1] * dt_c * dt_c,
        ];

        let (ra_a, dec_a) = tangent_to_radec(pa[0], pa[1], ra0, dec0);
        let (ra_b, dec_b) = tangent_to_radec(pb[0], pb[1], ra0, dec0);
        let (ra_c, dec_c) = tangent_to_radec(pc[0], pc[1], ra0, dec0);

        let a = make_alert(
            0 as AlertId,
            ra_a,
            dec_a,
            t_a,
            fluxes.0,
            0.0,
            band,
            ra_err,
            dec_err,
        );
        let b = make_alert(
            1 as AlertId,
            ra_b,
            dec_b,
            t_b,
            fluxes.1,
            0.0,
            band,
            ra_err,
            dec_err,
        );
        let c = make_alert(
            2 as AlertId,
            ra_c,
            dec_c,
            t_c,
            fluxes.2,
            0.0,
            band,
            ra_err,
            dec_err,
        );

        let store = make_store(vec![a, b, c]);
        let trips = vec![(0 as AlertId, 1 as AlertId, 2 as AlertId)];
        (store, trips)
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /* --------------------------- Unit tests --------------------------- */

    #[test]
    fn pair_velocity_recovery_small_motion() {
        let ra0 = 1.0;
        let dec0 = 0.3;
        let tm = 60000.0;
        let dt = 0.05; // 72 min
        let p0 = [1e-5, -5e-6];
        let v = [2e-4, -1e-4]; // rad/day
        let band = 2u8;
        let sigma = 3.0e-6; // ~0.62"
        let (store, pairs) = synthetic_pair_from_plane(
            ra0,
            dec0,
            tm,
            dt,
            p0,
            v,
            band,
            sigma,
            sigma,
            (1000.0, 1200.0),
        );

        let feats = extract_pair_features(
            &store,
            &pairs,
            FeatureExtractParams {
                max_speed_rad_per_day: Some(0.01), // keep it
            },
            3156,
        );
        assert_eq!(feats.len(), 1);
        let s = &feats[0];

        // Velocity should be close to truth (projection center differs slightly)
        assert!(approx_eq(s.vel_xy[0], v[0], 5e-6));
        assert!(approx_eq(s.vel_xy[1], v[1], 5e-6));

        // Covariance formulas
        let sa = sigma;
        let sb = sigma;
        let s2 = 0.5 * (sa * sa + sb * sb);
        let expected_pos = s2;
        let expected_vel = 2.0 * s2 / (dt * dt);

        assert!(approx_eq(s.cov_pos[0][0], expected_pos, 1e-18));
        assert!(approx_eq(s.cov_pos[1][1], expected_pos, 1e-18));
        assert!(approx_eq(s.cov_vel[0][0], expected_vel, 1e-18));
        assert!(approx_eq(s.cov_vel[1][1], expected_vel, 1e-18));

        // Photometry summary
        assert!((s.flux_mean - 1100.0).abs() < 1e-6);
        let expected_mad = ((1000.0_f32 - 1100.0).abs() + (1200.0_f32 - 1100.0).abs()) * 0.5;
        assert!((s.flux_std - expected_mad).abs() < 1e-6);
        assert_eq!(s.n_obs, 2);
        assert_eq!(s.band, band);
    }

    #[test]
    fn speed_guardrail_drops_fast_pairs() {
        let ra0 = 2.0;
        let dec0 = 0.1;
        let tm = 60010.0;
        let dt = 0.02;
        let p0 = [0.0, 0.0];
        let v = [0.2, 0.2]; // very fast
        let (store, pairs) =
            synthetic_pair_from_plane(ra0, dec0, tm, dt, p0, v, 1u8, 2e-6, 2e-6, (10.0, 10.0));

        let feats = extract_pair_features(
            &store,
            &pairs,
            FeatureExtractParams {
                max_speed_rad_per_day: Some(0.05),
            },
            -1,
        );
        assert_eq!(feats.len(), 0, "Fast pair should have been filtered out");
    }

    #[test]
    fn triplet_acceleration_recovery() {
        let ra0 = 1.5;
        let dec0 = -0.2;
        let tm = 60100.0;
        let dt_char = 0.04; // ±57.6 min around mid
        let p0 = [2e-5, -1e-5];
        let v = [1.0e-4, -8.0e-5];
        let a = [3.0e-6, -2.0e-6]; // rad/day^2
        let sigma = 2.5e-6;

        let (store, trips) = synthetic_triplet_from_plane(
            ra0,
            dec0,
            tm,
            dt_char,
            p0,
            v,
            a,
            3u8,
            sigma,
            sigma,
            (500.0, 600.0, 700.0),
        );
        let feats = extract_triplet_features(&store, &trips, 3156);
        assert_eq!(feats.len(), 1);
        let s = &feats[0];

        // Velocity close to true v, acceleration recovered
        let acc = s.acc_xy.expect("Triplet should carry acceleration");
        assert!(approx_eq(s.vel_xy[0], v[0], 5e-6));
        assert!(approx_eq(s.vel_xy[1], v[1], 5e-6));
        assert!(approx_eq(acc[0], a[0], 5e-7));
        assert!(approx_eq(acc[1], a[1], 5e-7));

        // Photometry summary (MAD-like with mean)
        let mean = (500.0 + 600.0 + 700.0) / 3.0;
        let dev = ((500.0_f32 - mean).abs() + (600.0 - mean).abs() + (700.0 - mean).abs()) / 3.0;
        assert!((s.flux_mean - mean).abs() < 1e-6);
        assert!((s.flux_std - dev).abs() < 1e-6);
        assert_eq!(s.n_obs, 3);
    }

    /* --------------------------- Property tests --------------------------- */

    proptest! {
        #[test]
        fn prop_pair_features_finite_and_covariances_hold(
            ra0 in 0.0f64..(2.0*PI),
            dec0 in -1.0..1.0, // stay far from poles
            tm in 59000.0f64..61000.0,
            dt in 0.02f64..0.08,
            p0x in -3e-4f64..3e-4,
            p0y in -3e-4f64..3e-4,
            vx in -3e-4f64..3e-4,
            vy in -3e-4f64..3e-4,
            sigma in 1.0e-6f64..5.0e-6,
            flux_a in 0.0f32..5000.0,
            flux_b in 0.0f32..5000.0,
            band in 0u8..6u8,
        ) {
            let p0 = [p0x, p0y];
            let v = [vx, vy];
            let (store, pairs) = synthetic_pair_from_plane(
                ra0, dec0, tm, dt, p0, v, band, sigma, sigma, (flux_a, flux_b),
            );

            let feats = extract_pair_features(
                &store,
                &pairs,
                FeatureExtractParams { max_speed_rad_per_day: Some(0.05) },
                42,
            );

            prop_assume!(!feats.is_empty());
            let s = &feats[0];

            // Finite checks
            prop_assert!(s.pos_xy.iter().all(|x| x.is_finite()));
            prop_assert!(s.vel_xy.iter().all(|x| x.is_finite()));
            prop_assert!(s.cov_pos.iter().flatten().all(|x| x.is_finite() && *x >= 0.0));
            prop_assert!(s.cov_vel.iter().flatten().all(|x| x.is_finite() && *x >= 0.0));

            // Covariance relation for pair: cov_vel ≈ 2*s2/dt^2
            let sa = sigma;
            let sb = sigma;
            let s2 = 0.5 * (sa*sa + sb*sb);
            let expected_vel = 2.0 * s2 / (dt*dt);
            prop_assert!((s.cov_vel[0][0] - expected_vel).abs() <= 1e-12);
            prop_assert!((s.cov_vel[1][1] - expected_vel).abs() <= 1e-12);

            // Velocity close to truth (absolute tolerance)
            prop_assert!((s.vel_xy[0] - vx).abs() <= 8e-6);
            prop_assert!((s.vel_xy[1] - vy).abs() <= 8e-6);

            // Band and counts
            prop_assert_eq!(s.band, band);
            prop_assert_eq!(s.n_obs, 2);
        }

        #[test]
        fn prop_triplet_acceleration_recovery_small_curvature(
            ra0 in 0.0f64..(2.0*PI),
            dec0 in -0.8f64..0.8,
            tm in 59000.0f64..61000.0,
            dt_char in 0.02f64..0.06,
            p0x in -2e-4f64..2e-4,
            p0y in -2e-4f64..2e-4,
            vx in -2e-4f64..2e-4,
            vy in -2e-4f64..2e-4,
            ax in -3e-6f64..3e-6,
            ay in -3e-6f64..3e-6,
            sigma in 1.0e-6f64..5.0e-6,
            f1 in 0.0f32..5000.0, f2 in 0.0f32..5000.0, f3 in 0.0f32..5000.0,
            band in 0u8..6u8,
        ) {
            let p0 = [p0x, p0y];
            let v = [vx, vy];
            let a = [ax, ay];

            let (store, trips) = synthetic_triplet_from_plane(
                ra0, dec0, tm, dt_char, p0, v, a, band, sigma, sigma, (f1, f2, f3),
            );
            let feats = extract_triplet_features(
                &store,
                &trips,
                7,
            );

            prop_assume!(!feats.is_empty());
            let s = &feats[0];

            // Finite
            prop_assert!(s.pos_xy.iter().all(|x| x.is_finite()));
            prop_assert!(s.vel_xy.iter().all(|x| x.is_finite()));
            prop_assert!(s.cov_pos.iter().flatten().all(|x| x.is_finite() && *x >= 0.0));
            prop_assert!(s.cov_vel.iter().flatten().all(|x| x.is_finite() && *x >= 0.0));

            // Acceleration must be present and close to truth
            let acc = s.acc_xy.expect("triplet must carry acc");
            prop_assert!((acc[0] - ax).abs() <= 8e-7);
            prop_assert!((acc[1] - ay).abs() <= 8e-7);

            // Velocity also close
            prop_assert!((s.vel_xy[0] - vx).abs() <= 1e-5);
            prop_assert!((s.vel_xy[1] - vy).abs() <= 1e-5);

            prop_assert_eq!(s.n_obs, 3);
            prop_assert_eq!(s.band, band);
        }
    }

    #[cfg(test)]
    mod predictions_and_neighbors_tests {
        use crate::Radians;

        use super::*;
        use std::f64::consts::PI;

        // ------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------

        fn approx(a: f64, b: f64, tol: f64) -> bool {
            (a - b).abs() <= tol
        }

        #[allow(clippy::too_many_arguments)]
        fn make_seed(
            seed_id: u64,
            night_id: i32,
            center_ra: f64,
            center_dec: f64,
            epoch_mid: f64,
            pos_xy: [f64; 2],
            vel_xy: [f64; 2],
            acc_xy: Option<[f64; 2]>,
            cov_pos_diag: f64,
            cov_vel_diag: f64,
            band: u8,
        ) -> SeedNode {
            let (ra_mid, dec_mid) =
                super::tangent_to_radec(pos_xy[0], pos_xy[1], center_ra, center_dec);
            SeedNode {
                seed_id,
                night_id,
                epoch_mid,
                pos_xy,
                vel_xy,
                cov_pos: [[cov_pos_diag, 0.0], [0.0, cov_pos_diag]],
                cov_vel: [[cov_vel_diag, 0.0], [0.0, cov_vel_diag]],
                acc_xy,
                flux_mean: 0.0,
                flux_std: 0.0,
                band,
                n_obs: if acc_xy.is_some() { 3 } else { 2 },
                members: vec![],
                center_ra,
                center_dec,
                ra_mid,
                dec_mid,
            }
        }

        // ------------------------------------------------------------
        // Fake spatial binner (equirectangular grid) for deterministic tests
        // ------------------------------------------------------------

        /// Simple RA/Dec grid binner:
        /// - cell size = `cell` radians (same in RA and Dec),
        /// - keys encode (i,j) as (i<<32 | j).
        /// - neighbors: square coverage with margin large enough to fully cover a cone of given radius.
        #[derive(Clone, Debug)]
        struct GridBinner {
            cell: Radians,
            n_ra: i32,
            n_dec: i32,
            cell_radius: Radians,
        }

        impl GridBinner {
            fn new(cell: Radians) -> Self {
                let n_ra = (2.0 * PI / cell).ceil() as i32;
                let n_dec = (PI / cell).ceil() as i32; // from -π/2 to +π/2 -> π span
                                                       // radius a bit larger than half-diagonal to be safe
                let cell_radius = (2.0f64).sqrt() * cell * 0.5;
                GridBinner {
                    cell,
                    n_ra,
                    n_dec,
                    cell_radius,
                }
            }

            fn idx_from_radec(&self, ra: Radians, dec: Radians) -> (i32, i32) {
                let ra_n = ra.rem_euclid(2.0 * PI);
                let i = (ra_n / self.cell).floor() as i32;
                // map dec from [-π/2, +π/2] to [0, π] then index
                let dec_shift = dec + 0.5 * PI;
                let j = (dec_shift / self.cell).floor() as i32;
                (i.rem_euclid(self.n_ra), j.clamp(0, self.n_dec - 1))
            }

            fn key(&self, i: i32, j: i32) -> SpatialKey {
                SpatialKey(((i as u64) << 32) | (j as u64))
            }

            fn ij_from_key(&self, key: SpatialKey) -> (i32, i32) {
                let u = key.0;
                let i = (u >> 32) as i32;
                let j = (u & 0xFFFF_FFFF) as i32;
                (i, j)
            }
        }

        impl SpatialBinner for GridBinner {
            fn key_for(&self, ra: Radians, dec: Radians) -> SpatialKey {
                let (i, j) = self.idx_from_radec(ra, dec);
                self.key(i, j)
            }

            fn neighbors(&self, key: SpatialKey, ang_radius: Radians) -> Vec<SpatialKey> {
                // We cover a square in (i,j) such that every cell whose center might intersect the
                // cone of `ang_radius` around the center cell is included.
                let (ic, jc) = self.ij_from_key(key);
                let steps = ((ang_radius + self.cell_radius) / self.cell).ceil() as i32;
                let mut out = Vec::new();
                for dj in -steps..=steps {
                    let j = (jc + dj).clamp(0, self.n_dec - 1);
                    for di in -steps..=steps {
                        let mut i = ic + di;
                        // wrap RA
                        if i < 0 {
                            i += self.n_ra;
                        }
                        if i >= self.n_ra {
                            i -= self.n_ra;
                        }
                        out.push(self.key(i, j));
                    }
                }
                out
            }

            fn cell_radius(&self) -> Radians {
                self.cell_radius
            }
        }

        // ------------------------------------------------------------
        // Unit tests
        // ------------------------------------------------------------

        #[test]
        fn predict_radec_matches_plane_motion_for_pairs() {
            let center_ra = 1.2;
            let center_dec = 0.3;
            let epoch_mid = 60000.0;
            let pos_xy = [1e-5, -2e-5];
            let vel_xy = [2e-4, -3e-4]; // rad/day
            let seed = make_seed(
                42, 123, center_ra, center_dec, epoch_mid, pos_xy, vel_xy, None, 1e-12, 1e-12, 2,
            );

            let dt = 0.1; // day
            let (ra, dec) = seed.predict_radec(epoch_mid + dt);

            // Re-project to tangent plane and compare to model p = p0 + v*dt
            let p = super::radec_to_tangent(ra, dec, center_ra, center_dec);
            assert!(approx(p[0], pos_xy[0] + vel_xy[0] * dt, 5e-12));
            assert!(approx(p[1], pos_xy[1] + vel_xy[1] * dt, 5e-12));
        }

        #[test]
        fn predict_radec_matches_plane_motion_for_triplets() {
            let center_ra = 2.1;
            let center_dec = -0.2;
            let epoch_mid = 60100.0;
            let pos_xy = [0.0, 0.0];
            let vel_xy = [1.2e-4, -0.8e-4];
            let acc_xy = [3.0e-6, -2.0e-6]; // rad/day^2

            let seed = make_seed(
                7,
                3156,
                center_ra,
                center_dec,
                epoch_mid,
                pos_xy,
                vel_xy,
                Some(acc_xy),
                1e-12,
                1e-12,
                1,
            );

            let dt = 0.2;
            let (ra, dec) = seed.predict_radec(epoch_mid + dt);
            let p = super::radec_to_tangent(ra, dec, center_ra, center_dec);

            let expected_x = pos_xy[0] + vel_xy[0] * dt + 0.5 * acc_xy[0] * dt * dt;
            let expected_y = pos_xy[1] + vel_xy[1] * dt + 0.5 * acc_xy[1] * dt * dt;

            assert!(approx(p[0], expected_x, 5e-12));
            assert!(approx(p[1], expected_y, 5e-12));
        }

        #[test]
        fn cone_radius_grows_with_dt() {
            let center_ra = 0.7;
            let center_dec = 0.1;
            let epoch_mid = 60200.0;
            let pos_xy = [0.0, 0.0];
            let vel_xy = [1e-4, 1e-4];
            // Covariances strictly positive:
            let cov_pos = 1e-12;
            let cov_vel = 5e-10;

            let seed = make_seed(
                1, 0, center_ra, center_dec, epoch_mid, pos_xy, vel_xy, None, cov_pos, cov_vel, 0,
            );

            let binner = GridBinner::new(1e-3);
            let params = PredictorParams {
                k_sigma: 3.0,
                noise: ModelNoise {
                    q0: 0.0,
                    q1: 0.0,
                    q2: 1e-10,
                },
                pad_cell_radius: false,
            };

            let (_, _, r1) = seed.predict_cone(epoch_mid + 0.1, &binner, params);
            let (_, _, r2) = seed.predict_cone(epoch_mid + 0.3, &binner, params);
            assert!(r2 > r1, "radius should increase with |dt|");
        }

        #[test]
        fn cone_candidates_returns_true_target() {
            // Build two seeds on two "nights": seed A (night k) and seed B (night k+1).
            // Choose velocity so that A predicts exactly B at the target epoch.
            let center_ra = 1.0;
            let center_dec = 0.2;

            let epoch_a = 60300.0;
            let epoch_b = epoch_a + 0.2;

            let pos_a = [0.0, 0.0];
            let vel_a = [5e-4, -3e-4]; // so that p(t_b) = v*Δt
            let seed_a = make_seed(
                10, 100, center_ra, center_dec, epoch_a, pos_a, vel_a, None, 1e-12, 1e-12, 2,
            );

            // Build seed B at the predicted position of A at t=epoch_b
            let mut p_b = [0.0, 0.0];
            let dt = epoch_b - epoch_a;
            p_b[0] = pos_a[0] + vel_a[0] * dt;
            p_b[1] = pos_a[1] + vel_a[1] * dt;

            let (ra_b, dec_b) = super::tangent_to_radec(p_b[0], p_b[1], center_ra, center_dec);
            // seed_b pos_xy is at its mid epoch; set velocities arbitrary (not used here)
            let seed_b = SeedNode {
                seed_id: 99,
                night_id: 101,
                epoch_mid: epoch_b,
                pos_xy: p_b,
                vel_xy: [0.0, 0.0],
                cov_pos: [[1e-12, 0.0], [0.0, 1e-12]],
                cov_vel: [[1e-12, 0.0], [0.0, 1e-12]],
                acc_xy: None,
                flux_mean: 0.0,
                flux_std: 0.0,
                band: 2,
                n_obs: 2,
                members: vec![],
                center_ra,
                center_dec,
                ra_mid: ra_b,
                dec_mid: dec_b,
            };

            // Build index for night k+1 containing seed_b
            let index_b =
                SeedSpatialIndex::build(std::slice::from_ref(&seed_b), &GridBinner::new(1e-3));

            let binner = GridBinner::new(1e-3);
            let params = PredictorParams {
                k_sigma: 0.0, // cone from cov may be ~0, but we will pad by cell radius
                noise: ModelNoise {
                    q0: 0.0,
                    q1: 0.0,
                    q2: 0.0,
                },
                pad_cell_radius: true,
            };

            let cand = seed_a.cone_candidates(epoch_b, &index_b, &binner, params);
            assert!(
                cand.contains(&seed_b.seed_id),
                "true target must be included"
            );
        }

        // ------------------------------------------------------------
        // Property tests
        // ------------------------------------------------------------

        // bounded randoms to keep within tangent validity and away from poles
        fn angle_ra() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }
        fn angle_dec() -> impl Strategy<Value = f64> {
            -0.8f64..0.8
        } // avoid projection singularities
        fn small_plane() -> impl Strategy<Value = f64> {
            -5e-3f64..5e-3
        }
        fn small_vel() -> impl Strategy<Value = f64> {
            -3e-3f64..3e-3
        }
        fn small_time() -> impl Strategy<Value = f64> {
            -0.5f64..0.5
        }

        proptest! {
            #[test]
            fn prop_predict_radec_is_finite_and_reversible(
                center_ra in angle_ra(),
                center_dec in angle_dec(),
                epoch in 59000.0f64..61000.0,
                p0x in small_plane(), p0y in small_plane(),
                vx in small_vel(), vy in small_vel(),
                dt in small_time()
            ) {
                // Build pair-like seed (no acceleration)
                let seed = make_seed(
                    1, 0, center_ra, center_dec, epoch,
                    [p0x, p0y], [vx, vy], None,
                    1e-12, 1e-12, 0
                );

                let (ra, dec) = seed.predict_radec(epoch + dt);

                // Finiteness & bounds
                prop_assert!(ra.is_finite() && dec.is_finite());
                prop_assert!((0.0..2.0*PI).contains(&ra));
                prop_assert!(dec > -PI/2.0 && dec < PI/2.0);

                // Round-trip back to plane should agree with model within small epsilon
                let p = super::radec_to_tangent(ra, dec, center_ra, center_dec);
                let ex = p0x + vx*dt;
                let ey = p0y + vy*dt;
                prop_assert!((p[0] - ex).abs() <= 2e-10);
                prop_assert!((p[1] - ey).abs() <= 2e-10);
            }

            #[test]
            fn prop_cone_radius_monotonic_in_dt_abs(
                center_ra in angle_ra(),
                center_dec in angle_dec(),
                epoch in 59000.0f64..61000.0,
                p0x in -1e-4f64..1e-4, p0y in -1e-4f64..1e-4,
                vx in -1e-4f64..1e-4, vy in -1e-4f64..1e-4,
                cov_p in 1e-14f64..1e-10, cov_v in 1e-14f64..1e-10,
                dt1 in 0.05f64..0.2, dt2 in 0.3f64..0.6
            ) {
                let seed = make_seed(
                    11, 0, center_ra, center_dec, epoch,
                    [p0x, p0y], [vx, vy], None,
                    cov_p, cov_v, 1
                );
                let binner = GridBinner::new(5e-3);
                let params = PredictorParams {
                    k_sigma: 3.0,
                    noise: ModelNoise { q0: 0.0, q1: 0.0, q2: 1e-12 }, // non-negative
                    pad_cell_radius: false,
                };
                let (_, _, r1) = seed.predict_cone(epoch + dt1, &binner, params);
                let (_, _, r2) = seed.predict_cone(epoch + dt2, &binner, params);
                prop_assert!(r2 >= r1, "radius must not shrink when |dt| increases");
            }

            #[test]
            fn prop_cone_candidates_includes_center_cell_with_padding(
                center_ra in angle_ra(),
                center_dec in angle_dec(),
                epoch in 59000.0f64..61000.0,
                cell in 1e-3f64..5e-3
            ) {
                let seed = make_seed(
                    5, 0, center_ra, center_dec, epoch,
                    [0.0, 0.0], [0.0, 0.0], None,
                    1e-14, 1e-14, 1
                );
                let binner = GridBinner::new(cell);

                // Target night has a single seed at exactly the predicted sky position at t=epoch
                let seed_target = SeedNode {
                    seed_id: 777,
                    night_id: 1,
                    epoch_mid: epoch,
                    pos_xy: [0.0, 0.0],
                    vel_xy: [0.0, 0.0],
                    cov_pos: [[1e-14, 0.0],[0.0,1e-14]],
                    cov_vel: [[1e-14, 0.0],[0.0,1e-14]],
                    acc_xy: None,
                    flux_mean: 0.0, flux_std: 0.0, band: 0, n_obs: 2,
                    members: vec![],
                    center_ra, center_dec,
                    ra_mid: seed.ra_mid, dec_mid: seed.dec_mid,
                };
                let index = SeedSpatialIndex::build(std::slice::from_ref(&seed_target), &binner);

                let params = PredictorParams {
                    k_sigma: 0.0, // rely on padding to at least cover center cell
                    noise: ModelNoise { q0: 0.0, q1: 0.0, q2: 0.0 },
                    pad_cell_radius: true,
                };

                let cand = seed.cone_candidates(epoch, &index, &binner, params);
                prop_assert!(cand.contains(&seed_target.seed_id));
            }
        }
    }
}

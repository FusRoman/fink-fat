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

use crate::{
    alerts::{AlertId, AlertStore},
    seeding::geometrical_seeding::{Pairs, Triplets},
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
/// (`epoch_mid`) on a **gnomonic tangent plane**:
/// - `pos_xy`: position on the plane (radians),
/// - `vel_xy`: constant sky-plane velocity (radians/day),
/// - `acc_xy`: optional acceleration (radians/day²) for triplets,
/// - `cov_pos`, `cov_vel`: simple 2×2 covariances for position/velocity,
/// - `flux_mean`, `flux_std`: photometry summary (difference PSF flux, nJy).
///
/// This structure is intentionally compact and cloneable; it can be moved across
/// threads and returned to Python bindings if needed.
///
/// # Units
/// See module-level **Units** section.
///
/// # See also
/// - [`extract_pair_features`]
/// - [`extract_triplet_features`]
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

/* --------------------------- Public API --------------------------- */

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
        });
    }
    out
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
}

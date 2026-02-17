//! Local tangent-plane kinematic model for intra-night seeds.
//!
//! This module provides two core data structures:
//!
//! - [`TangentCenter`] — stores the reference sky position of the tangent
//!   plane (α₀, δ₀) together with cached trigonometric terms.
//! - [`TangentPlaneModel`] — compact kinematic model on the tangent plane,
//!   with optional acceleration and simple covariance propagation.
//!
//! The model is deliberately lightweight and designed to be:
//!
//! - cheap to serialise (via `serde` and `bincode`),
//! - stable numerically by delegating projection logic to `astro_math`,
//! - sufficient for fast intra-night prediction and cone generation
//!   (not a full orbit determination).
//!
//! Typical workflow
//! ----------------
//! 1. Build a [`TangentPlaneModel`] from 2 or 3 alerts (see `SeedNode` helpers).
//! 2. Predict position and uncertainty on the tangent plane via
//!    [`TangentPlaneModel::predict_on_plane`].
//! 3. Convert to sky coordinates and cone radius via
//!    [`TangentPlaneModel::predict_cone_base`].
//! 4. Use the cone for spatial indexing or candidate search.

use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use crate::{
    MJDTT, Radian,
    astro_math::{lambda_max_2x2, radec_to_tangent, tangent_to_radec},
    display_format::{fmt_mat2, fmt_vec2},
    engine_config::propagator_config::ModelNoise,
};

/// Reference sky position of the tangent plane.
///
/// This struct holds:
///
/// - the tangent-plane centre `(ra0, dec0)` in radians,
/// - cached trigonometric terms `(sin_dec0, cos_dec0)` for fast projection.
///
/// In this implementation the cached terms are **not** directly used by
/// `radec_to_tangent` / `tangent_to_radec` (we delegate to `astro_math` for
/// robustness), but keeping them here:
///
/// - documents the geometry explicitly,
/// - allows future optimisations if needed.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct TangentCenter {
    /// Tangent-plane centre right ascension α₀ (radians, ICRS).
    pub ra0: Radian,
    /// Tangent-plane centre declination δ₀ (radians, ICRS).
    pub dec0: Radian,
    /// Precomputed `sin(δ₀)` for potential optimisation.
    pub sin_dec0: f64,
    /// Precomputed `cos(δ₀)` for potential optimisation.
    pub cos_dec0: f64,
}

impl Display for TangentCenter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TangentCenter {{")?;

        writeln!(f, "  ra0       : {}", self.ra0)?;
        writeln!(f, "  dec0      : {}", self.dec0)?;

        writeln!(f, "  sin_dec0  : {:.6e}", self.sin_dec0)?;
        writeln!(f, "  cos_dec0  : {:.6e}", self.cos_dec0)?;

        write!(f, "}}")
    }
}

impl TangentCenter {
    /// Construct a new [`TangentCenter`] from a reference sky position.
    ///
    /// The trigonometric terms `(sin_dec0, cos_dec0)` are computed once and
    /// stored alongside the reference coordinates.
    ///
    /// Arguments
    /// ---------
    /// * `ra0` – Right ascension of the tangent-plane centre (radians).
    /// * `dec0` – Declination of the tangent-plane centre (radians).
    ///
    /// Return
    /// ------
    /// * `TangentCenter` – A new instance with cached trigonometric values.
    ///
    /// Notes
    /// -----
    /// * The cached terms are kept even though projection is delegated to
    ///   `astro_math::radec_to_tangent`; this keeps the struct self-describing
    ///   and ready for lower-level optimisations if required.
    #[inline]
    pub fn new(ra0: Radian, dec0: Radian) -> Self {
        let (sin_dec0, cos_dec0) = dec0.sin_cos();
        Self {
            ra0,
            dec0,
            sin_dec0,
            cos_dec0,
        }
    }

    /// Project a sky position `(RA, Dec)` onto the tangent plane.
    ///
    /// This is a thin wrapper around [`radec_to_tangent`] that preserves the
    /// local API while delegating all numerical guards to `astro_math`.
    ///
    /// Arguments
    /// ---------
    /// * `ra` – Right ascension to project (radians).
    /// * `dec` – Declination to project (radians).
    ///
    /// Return
    /// ------
    /// * `[x, y]` – Tangent-plane coordinates (gnomonic projection), in radians.
    ///
    /// Notes
    /// -----
    /// * This method does not use `sin_dec0` and `cos_dec0` directly; it
    ///   simply forwards to the robust implementation in `astro_math`.
    #[inline]
    pub fn radec_to_tangent(&self, ra: Radian, dec: Radian) -> [f64; 2] {
        radec_to_tangent(ra, dec, self.ra0, self.dec0)
    }
}

/// Local kinematic model on a gnomonic tangent plane.
///
/// The model is expressed as:
///
/// ```text
/// x(t) = x₀ + vₓ·Δt + 0.5·aₓ·Δt²
/// y(t) = y₀ + v_y·Δt + 0.5·a_y·Δt²
/// ```
///
/// where:
/// - `(x₀, y₀)` is [`pos_xy`] at [`epoch_mid`],
/// - `(vₓ, v_y)` is [`vel_xy`],
/// - `(aₓ, a_y)` is optional [`acc_xy`].
///
/// Two covariance matrices are stored:
///
/// - [`cov_pos`] — position covariance at `epoch_mid` (rad²),
/// - [`cov_vel`] — velocity covariance (rad²/day²).
///
/// They are propagated in a **simplified, diagonal form** using the
/// [`ModelNoise`] parameters when predicting on the plane.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TangentPlaneModel {
    /// Tangent-plane centre and cached trigonometric terms.
    pub center: TangentCenter,

    /// Reference epoch (typically the middle of the arc), in MJD TT.
    pub epoch_mid: MJDTT,

    /// Position on the tangent plane at `epoch_mid` (radians).
    pub pos_xy: [f64; 2],
    /// Velocity on the tangent plane (radians/day).
    pub vel_xy: [f64; 2],
    /// Optional acceleration on the tangent plane (radians/day²).
    pub acc_xy: Option<[f64; 2]>,

    /// Position covariance on the tangent plane at `epoch_mid` (rad²).
    pub cov_pos: [[f64; 2]; 2],
    /// Velocity covariance on the tangent plane (rad²/day²).
    pub cov_vel: [[f64; 2]; 2],

    /// Mean RA of the arc in ICRS (radians, mostly for QA / debugging).
    pub ra_mid: Radian,
    /// Mean Dec of the arc in ICRS (radians, mostly for QA / debugging).
    pub dec_mid: Radian,
}

impl Display for TangentPlaneModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TangentPlaneModel {{")?;

        writeln!(f, "  center    : {}", self.center)?;
        writeln!(f, "  epoch_mid : {}", self.epoch_mid)?;

        writeln!(f, "  pos_xy    : {}", fmt_vec2(self.pos_xy))?;
        writeln!(f, "  vel_xy    : {}", fmt_vec2(self.vel_xy))?;
        writeln!(
            f,
            "  acc_xy    : {}",
            match self.acc_xy {
                Some(a) => fmt_vec2(a).to_string(),
                None => "None".to_string(),
            }
        )?;

        writeln!(f, "  cov_pos   : {}", fmt_mat2(self.cov_pos))?;
        writeln!(f, "  cov_vel   : {}", fmt_mat2(self.cov_vel))?;

        writeln!(f, "  ra_mid    : {}", self.ra_mid)?;
        writeln!(f, "  dec_mid   : {}", self.dec_mid)?;

        write!(f, "}}")
    }
}

impl TangentPlaneModel {
    /// Build a new [`TangentPlaneModel`] from its components.
    ///
    /// This is a convenience constructor that simply stores all provided
    /// fields without performing additional consistency checks.
    ///
    /// Arguments
    /// ---------
    /// * `center` – Tangent-plane centre and cached trigonometric terms.
    /// * `epoch_mid` – Reference epoch at which `pos_xy` is defined (MJD TT).
    /// * `pos_xy` – Position on the tangent plane at `epoch_mid` (radians).
    /// * `vel_xy` – Velocity on the tangent plane (radians/day).
    /// * `acc_xy` – Optional acceleration on the tangent plane (radians/day²).
    /// * `cov_pos` – Position covariance at `epoch_mid` (rad²).
    /// * `cov_vel` – Velocity covariance (rad²/day²).
    /// * `ra_mid` – Mean RA of the arc (radians, ICRS).
    /// * `dec_mid` – Mean Dec of the arc (radians, ICRS).
    ///
    /// Return
    /// ------
    /// * `TangentPlaneModel` – A fully initialised kinematic model.
    ///
    /// Notes
    /// -----
    /// * Higher-level builders (e.g. from pairs or triplets) are responsible
    ///   for ensuring the physical consistency of the parameters.
    #[inline]
    pub fn new(
        center: TangentCenter,
        epoch_mid: MJDTT,
        pos_xy: [f64; 2],
        vel_xy: [f64; 2],
        acc_xy: Option<[f64; 2]>,
        cov_pos: [[f64; 2]; 2],
        cov_vel: [[f64; 2]; 2],
        ra_mid: Radian,
        dec_mid: Radian,
    ) -> Self {
        Self {
            center,
            epoch_mid,
            pos_xy,
            vel_xy,
            acc_xy,
            cov_pos,
            cov_vel,
            ra_mid,
            dec_mid,
        }
    }

    /// Project `(RA, Dec)` to the tangent plane using the model centre.
    ///
    /// This method mirrors [`TangentCenter::radec_to_tangent`] but is defined
    /// on the model for convenience. It forwards to the implementation
    /// in `astro_math`.
    ///
    /// Arguments
    /// ---------
    /// * `ra` – Right ascension to project (radians).
    /// * `dec` – Declination to project (radians).
    ///
    /// Return
    /// ------
    /// * `[x, y]` – Tangent-plane coordinates (gnomonic projection), in radians.
    ///
    /// Notes
    /// -----
    /// * The name `*_precomp` is historical: previous versions used the
    ///   cached trigonometric terms explicitly; the current implementation
    ///   simply delegates to `radec_to_tangent`.
    #[inline]
    pub fn radec_to_tangent_precomp(&self, ra: Radian, dec: Radian) -> [f64; 2] {
        // Use the projection defined in astro_math.rs.
        radec_to_tangent(ra, dec, self.center.ra0, self.center.dec0)
    }

    /// Predict position and covariance on the tangent plane at a target epoch.
    ///
    /// The prediction uses a polynomial kinematic model and a simple noise
    /// parameterisation:
    ///
    /// ```text
    /// p(t)  = p₀ + v·Δt + 0.5·a·Δt²
    /// q(Δt) = σ_floor² + drift_per_day·|Δt| + curvature_per_day2·Δt²
    ///
    /// cov_pos(t) ≈ cov_pos(0) + Δt²·cov_vel + q(Δt)·I
    /// ```
    ///
    /// where:
    ///
    /// - `Δt = t_target − epoch_mid`,
    /// - the noise term `q(Δt)` is added diagonally to both coordinates,
    /// - off-diagonal covariance terms are ignored in the propagated result.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Epoch at which to predict the state (MJD TT).
    /// * `noise` – Model noise parameters:
    ///   - `variance_floor` – minimum additional variance (rad²),
    ///   - `drift_per_day` – linear growth term vs. |Δt|,
    ///   - `curvature_per_day2` – quadratic growth term vs. Δt².
    ///
    /// Return
    /// ------
    /// * `(pos_xy, cov_pos)` where:
    ///   - `pos_xy` is the predicted `[x, y]` on the tangent plane (radians),
    ///   - `cov_pos` is a diagonal 2×2 covariance matrix for position (rad²).
    ///
    /// Notes
    /// -----
    /// * Acceleration is applied only if [`TangentPlaneModel::acc_xy`] is
    ///   `Some`; otherwise the model is purely linear.
    /// * The covariance propagation is intentionally simplified and assumes
    ///   independence between x and y in the added noise term.
    #[inline]
    pub fn predict_on_plane(
        &self,
        t_target: MJDTT,
        noise: &ModelNoise,
    ) -> ([f64; 2], [[f64; 2]; 2]) {
        let dt = t_target - self.epoch_mid;

        // Kinematic propagation on the tangent plane.
        let (px, py) = self.predict_position(dt, dt * dt);

        // Time-dependent noise model.
        let q = noise.variance_floor
            + noise.drift_per_day * dt.abs()
            + noise.curvature_per_day2 * dt * dt;

        // Diagonal covariance propagation: position + scaled velocity variance + noise.
        let sxx = self.cov_pos[0][0] + dt * dt * self.cov_vel[0][0] + q;
        let syy = self.cov_pos[1][1] + dt * dt * self.cov_vel[1][1] + q;

        ([px, py], [[sxx, 0.0], [0.0, syy]])
    }

    /// Predict position on the tangent plane after dt days.
    ///
    /// Arguments
    /// ---------
    /// * `dt` – Time difference from `epoch_mid` (days).
    /// * `dt_sq` – Square of the time difference (days²).
    ///
    /// Return
    /// ------
    /// * `(x, y)` – Predicted position on the tangent plane (radians).
    #[inline]
    pub fn predict_position(&self, dt: f64, dt_sq: f64) -> (Radian, Radian) {
        // Kinematic propagation on the tangent plane.
        let mut px = self.pos_xy[0] + self.vel_xy[0] * dt;
        let mut py = self.pos_xy[1] + self.vel_xy[1] * dt;
        if let Some(a) = self.acc_xy {
            px += 0.5 * a[0] * dt_sq;
            py += 0.5 * a[1] * dt_sq;
        }
        (px, py)
    }

    /// Predict velocity on the tangent plane after dt days.
    ///
    /// Arguments
    /// ---------
    /// * `dt` – Time difference from `epoch_mid` (days).
    ///
    /// Return
    /// ------
    /// * `(vx, vy)` – Predicted velocity on the tangent plane (radians/day).
    #[inline]
    pub fn predict_velocity(&self, dt: f64) -> (Radian, Radian) {
        // Kinematic propagation on the tangent plane.
        let mut vx = self.vel_xy[0];
        let mut vy = self.vel_xy[1];
        if let Some(a) = self.acc_xy {
            vx += a[0] * dt;
            vy += a[1] * dt;
        }
        (vx, vy)
    }

    /// Predict sky coordinates `(RA, Dec)` at a target epoch.
    ///
    /// This method performs:
    ///
    /// 1. Kinematic propagation on the tangent plane using position, velocity
    ///    and optional acceleration.
    /// 2. Conversion back to sky coordinates via the inverse gnomonic
    ///    projection [`tangent_to_radec`].
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Epoch at which to predict the sky position (MJD TT).
    ///
    /// Return
    /// ------
    /// * `(ra, dec)` – Predicted right ascension and declination (radians).
    ///
    /// Notes
    /// -----
    /// * Uncertainty is **not** returned here; for cone-based searches you
    ///   should use [`TangentPlaneModel::predict_cone_base`] instead.
    /// * As with all tangent-plane models, accuracy degrades as the object
    ///   drifts far from the reference centre or outside the validity time
    ///   range of the fit.
    #[inline]
    pub fn predict_radec(&self, t_target: MJDTT) -> (Radian, Radian) {
        let dt = t_target - self.epoch_mid;

        // Kinematic propagation on the tangent plane.
        let (px, py) = self.predict_position(dt, dt * dt);

        // Convert back to the celestial sphere using the robust inverse
        // gnomonic projection from astro_math.
        tangent_to_radec(px, py, self.center.ra0, self.center.dec0)
    }

    /// Predict a **cone** on the sky (centre + radius) without spatial padding.
    ///
    /// This is the core building block for cone-based candidate searches:
    ///
    /// 1. Predict position and covariance on the tangent plane via
    ///    [`TangentPlaneModel::predict_on_plane`].
    /// 2. Convert the predicted position to `(RA, Dec)` using
    ///    [`tangent_to_radec`].
    /// 3. Extract the maximum eigenvalue of the 2×2 covariance matrix via
    ///    [`lambda_max_2x2`] and convert it to a 1σ angular radius.
    /// 4. Multiply by `k_sigma` to obtain a conservative cone radius.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Epoch at which to predict the cone (MJD TT).
    /// * `noise` – Model noise parameters used in the plane prediction.
    /// * `k_sigma` – Multiplicative factor applied to the 1σ radius derived
    ///   from the largest eigenvalue of the covariance.
    ///
    /// Return
    /// ------
    /// * `(ra_center, dec_center, radius)` where:
    ///   - `ra_center`, `dec_center` – cone centre on the sky (radians),
    ///   - `radius` – angular radius of the cone (radians).
    ///
    /// Notes
    /// -----
    /// * The covariance used here is the result of
    ///   [`TangentPlaneModel::predict_on_plane`]; only its largest eigenvalue
    ///   is considered, making the cone circular even if the underlying
    ///   uncertainty is anisotropic.
    /// * Any additional padding related to spatial indexing cells (e.g. adding
    ///   a HEALPix cell radius) should be applied *outside* this method; see
    ///   `SeedNode::predict_cone` for an example.
    #[inline]
    pub fn predict_cone_base(
        &self,
        t_target: MJDTT,
        noise: &ModelNoise,
        k_sigma: f64,
    ) -> (Radian, Radian, f64) {
        // Predict position and covariance on the tangent plane.
        let (p, cov) = self.predict_on_plane(t_target, noise);

        // Map the predicted plane position back to sky coordinates.
        let (ra, dec) = tangent_to_radec(p[0], p[1], self.center.ra0, self.center.dec0);

        // Use the largest eigenvalue of the covariance as the dominant scale.
        let lam_max = lambda_max_2x2(cov).max(0.0);
        let radius = k_sigma * lam_max.sqrt();

        (ra, dec, radius)
    }
}

#[cfg(test)]
mod tangent_plane_tests {
    use super::*;
    use approx::abs_diff_eq;
    use proptest::prelude::*;

    use crate::{
        astro_math::{arcsec_to_rad, radec_to_tangent},
        engine_config::propagator_config::ModelNoise,
    };

    const EPS: f64 = 1e-12;

    /* ------------------------------ unit tests ------------------------------ */

    #[test]
    fn tangent_center_new_computes_trig_cache() {
        let ra0 = 1.234;
        let dec0 = 0.5;
        let c = TangentCenter::new(ra0, dec0);
        assert!(abs_diff_eq!(c.ra0, ra0, epsilon = EPS));
        assert!(abs_diff_eq!(c.dec0, dec0, epsilon = EPS));
        let (s, c_) = dec0.sin_cos();
        assert!(abs_diff_eq!(c.sin_dec0, s, epsilon = EPS));
        assert!(abs_diff_eq!(c.cos_dec0, c_, epsilon = EPS));
    }

    #[test]
    fn tangent_center_radec_to_tangent_matches_free_function() {
        let ra0 = 1.0;
        let dec0 = 0.3;
        let c = TangentCenter::new(ra0, dec0);

        let ra = 1.01;
        let dec = 0.31;

        let p1 = c.radec_to_tangent(ra, dec);
        let p2 = radec_to_tangent(ra, dec, ra0, dec0);
        assert!(abs_diff_eq!(p1[0], p2[0], epsilon = EPS));
        assert!(abs_diff_eq!(p1[1], p2[1], epsilon = EPS));
    }

    #[test]
    fn model_new_stores_fields() {
        let center = TangentCenter::new(1.0, 0.2);
        let epoch_mid = 60000.0;
        let pos = [0.01, -0.02];
        let vel = [1e-3, -2e-3];
        let acc = Some([5e-6, -3e-6]);
        let cov_pos = [[1e-6, 0.0], [0.0, 2e-6]];
        let cov_vel = [[1e-8, 0.0], [0.0, 2e-8]];
        let ra_mid = 1.1;
        let dec_mid = 0.19;

        let m = TangentPlaneModel::new(
            center, epoch_mid, pos, vel, acc, cov_pos, cov_vel, ra_mid, dec_mid,
        );

        assert!(abs_diff_eq!(m.epoch_mid, epoch_mid, epsilon = EPS));
        assert!(abs_diff_eq!(m.pos_xy[0], pos[0], epsilon = EPS));
        assert!(abs_diff_eq!(m.pos_xy[1], pos[1], epsilon = EPS));
        assert!(abs_diff_eq!(m.vel_xy[0], vel[0], epsilon = EPS));
        assert!(abs_diff_eq!(m.vel_xy[1], vel[1], epsilon = EPS));
        assert!(m.acc_xy.is_some());
        assert!(abs_diff_eq!(m.ra_mid, ra_mid, epsilon = EPS));
        assert!(abs_diff_eq!(m.dec_mid, dec_mid, epsilon = EPS));
    }

    #[test]
    fn predict_on_plane_linear_without_acceleration() {
        let center = TangentCenter::new(1.0, 0.3);
        let epoch_mid = 60000.0;
        let pos = [0.0, 0.0];
        let vel = [1e-3, -2e-3];
        let m = TangentPlaneModel::new(
            center,
            epoch_mid,
            pos,
            vel,
            None,
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            1.0,
            0.3,
        );

        let noise = ModelNoise {
            variance_floor: 0.0,
            drift_per_day: 0.0,
            curvature_per_day2: 0.0,
        };

        let t = epoch_mid + 2.0; // Δt = 2 days
        let (p, cov) = m.predict_on_plane(t, &noise);
        assert!(abs_diff_eq!(p[0], vel[0] * 2.0, epsilon = EPS));
        assert!(abs_diff_eq!(p[1], vel[1] * 2.0, epsilon = EPS));

        assert!(abs_diff_eq!(cov[0][0], 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(cov[1][1], 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(cov[0][1], 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(cov[1][0], 0.0, epsilon = EPS));
    }

    #[test]
    fn predict_on_plane_applies_acceleration_and_noise() {
        let center = TangentCenter::new(1.5, -0.2);
        let epoch_mid = 60000.0;
        let pos = [0.1, -0.1];
        let vel = [4e-3, 3e-3];
        let acc = Some([1e-3, -5e-4]); // large to be measurable in test
        let cov_pos = [[1e-6, 0.0], [0.0, 2e-6]];
        let cov_vel = [[1e-8, 0.0], [0.0, 3e-8]];
        let m = TangentPlaneModel::new(
            center, epoch_mid, pos, vel, acc, cov_pos, cov_vel, 1.5, -0.2,
        );

        let noise = ModelNoise {
            variance_floor: 1e-9,
            drift_per_day: 2e-9,
            curvature_per_day2: 3e-9,
        };

        let dt = 0.5;
        let (p, cov) = m.predict_on_plane(epoch_mid + dt, &noise);

        let px_expected = pos[0] + vel[0] * dt + 0.5 * acc.unwrap()[0] * dt * dt;
        let py_expected = pos[1] + vel[1] * dt + 0.5 * acc.unwrap()[1] * dt * dt;
        assert!(abs_diff_eq!(p[0], px_expected, epsilon = 1e-15));
        assert!(abs_diff_eq!(p[1], py_expected, epsilon = 1e-15));

        let q = noise.variance_floor
            + noise.drift_per_day * dt.abs()
            + noise.curvature_per_day2 * dt * dt;
        let sxx_expected = cov_pos[0][0] + dt * dt * cov_vel[0][0] + q;
        let syy_expected = cov_pos[1][1] + dt * dt * cov_vel[1][1] + q;
        assert!(abs_diff_eq!(cov[0][0], sxx_expected, epsilon = EPS));
        assert!(abs_diff_eq!(cov[1][1], syy_expected, epsilon = EPS));
        assert!(abs_diff_eq!(cov[0][1], 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(cov[1][0], 0.0, epsilon = EPS));
    }

    #[test]
    fn predict_radec_roundtrip_with_plane_propagation() {
        let center = TangentCenter::new(2.0, 0.1);
        let epoch_mid = 61000.0;
        let pos = [0.002, -0.003];
        let vel = [1e-3, 2e-3];
        let m = TangentPlaneModel::new(
            center,
            epoch_mid,
            pos,
            vel,
            None,
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            2.0,
            0.1,
        );

        let t = epoch_mid + 1.0;
        let (ra, dec) = m.predict_radec(t);

        // Project back to tangent plane; should match plane propagation.
        let [x, y] = radec_to_tangent(ra, dec, center.ra0, center.dec0);
        let px = pos[0] + vel[0] * 1.0;
        let py = pos[1] + vel[1] * 1.0;

        assert!(abs_diff_eq!(x, px, epsilon = 5e-12));
        assert!(abs_diff_eq!(y, py, epsilon = 5e-12));
    }

    #[test]
    fn predict_cone_base_produces_non_negative_radius() {
        let center = TangentCenter::new(1.0, 0.3);
        let m = TangentPlaneModel::new(
            center,
            60000.0,
            [0.0, 0.0],
            [0.0, 0.0],
            None,
            [[1e-6, 0.0], [0.0, 2e-6]],
            [[1e-7, 0.0], [0.0, 3e-7]],
            1.0,
            0.3,
        );

        let noise = ModelNoise {
            variance_floor: 1e-9,
            drift_per_day: 0.0,
            curvature_per_day2: 0.0,
        };

        let (ra, dec, r) = m.predict_cone_base(60001.0, &noise, 3.0);
        assert!(ra.is_finite() && dec.is_finite());
        assert!(r >= 0.0);
    }

    #[test]
    fn predict_cone_base_increases_with_k_sigma() {
        let center = TangentCenter::new(1.0, 0.3);
        let m = TangentPlaneModel::new(
            center,
            60000.0,
            [0.0, 0.0],
            [0.0, 0.0],
            None,
            [[1e-6, 0.0], [0.0, 1e-6]],
            [[0.0, 0.0], [0.0, 0.0]],
            1.0,
            0.3,
        );

        let noise = ModelNoise {
            variance_floor: 1e-9,
            drift_per_day: 0.0,
            curvature_per_day2: 0.0,
        };

        let (_, _, r1) = m.predict_cone_base(60000.5, &noise, 1.0);
        let (_, _, r2) = m.predict_cone_base(60000.5, &noise, 3.0);
        assert!(r2 > r1);
    }

    /* --------------------------- property-based tests --------------------------- */

    fn ra_strategy() -> impl Strategy<Value = f64> {
        0.0f64..(2.0 * std::f64::consts::PI)
    }
    fn dec_strategy() -> impl Strategy<Value = f64> {
        // Keep away from poles to avoid extreme gnomonic behaviour in proptests.
        (-0.8f64)..0.8
    }
    fn small_plane() -> impl Strategy<Value = f64> {
        -5e-3f64..5e-3
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 40,
            .. ProptestConfig::default()
        })]

        /// Predict-on-plane must match kinematic propagation formula (up to numerical eps).
        #[test]
        fn prop_predict_on_plane_matches_kinematics(
            ra0 in ra_strategy(),
            dec0 in dec_strategy(),
            epoch_mid in 60000.0f64..60005.0,
            pos_x in small_plane(),
            pos_y in small_plane(),
            vel_x in -5e-3f64..5e-3,
            vel_y in -5e-3f64..5e-3,
            acc_x in -1e-3f64..1e-3,
            acc_y in -1e-3f64..1e-3,
            dt in -2.0f64..2.0,
        ) {
            let center = TangentCenter::new(ra0, dec0);
            let acc = Some([acc_x, acc_y]);
            let m = TangentPlaneModel::new(
                center, epoch_mid, [pos_x, pos_y], [vel_x, vel_y], acc,
                [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
                ra0, dec0,
            );

            let noise = ModelNoise { variance_floor: 0.0, drift_per_day: 0.0, curvature_per_day2: 0.0 };
            let (p, _) = m.predict_on_plane(epoch_mid + dt, &noise);

            let px = pos_x + vel_x * dt + 0.5 * acc_x * dt * dt;
            let py = pos_y + vel_y * dt + 0.5 * acc_y * dt * dt;

            prop_assert!(abs_diff_eq!(p[0], px, epsilon = 5e-12));
            prop_assert!(abs_diff_eq!(p[1], py, epsilon = 5e-12));
        }

        /// Predict_radec composed with radec_to_tangent should recover plane propagation locally.
        #[test]
        fn prop_predict_radec_roundtrip_local(
            ra0 in ra_strategy(),
            dec0 in dec_strategy(),
            epoch_mid in 60000.0f64..60005.0,
            pos_x in small_plane(),
            pos_y in small_plane(),
            vel_x in -5e-3f64..5e-3,
            vel_y in -5e-3f64..5e-3,
            dt in -1.0f64..1.0,
        ) {
            let center = TangentCenter::new(ra0, dec0);
            let m = TangentPlaneModel::new(
                center, epoch_mid, [pos_x, pos_y], [vel_x, vel_y], None,
                [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
                ra0, dec0,
            );

            let (ra, dec) = m.predict_radec(epoch_mid + dt);
            let [x, y] = radec_to_tangent(ra, dec, ra0, dec0);

            let px = pos_x + vel_x * dt;
            let py = pos_y + vel_y * dt;

            prop_assert!(abs_diff_eq!(x, px, epsilon = 5e-12));
            prop_assert!(abs_diff_eq!(y, py, epsilon = 5e-12));
        }

        /// Cone radius grows with plane covariance and k_sigma in a consistent manner.
        #[test]
        fn prop_cone_radius_non_decreasing_with_noise_and_k(
            ra0 in ra_strategy(),
            dec0 in dec_strategy(),
            epoch_mid in 60000.0f64..60005.0,
            pos_x in small_plane(),
            pos_y in small_plane(),
            t in 60000.0f64..60002.0,
        ) {
            let center = TangentCenter::new(ra0, dec0);
            let m = TangentPlaneModel::new(
                center, epoch_mid, [pos_x, pos_y], [0.0, 0.0], None,
                [[1e-8, 0.0], [0.0, 1e-8]], [[0.0, 0.0], [0.0, 0.0]],
                ra0, dec0,
            );

            let n0 = ModelNoise { variance_floor: 0.0, drift_per_day: 0.0, curvature_per_day2: 0.0 };
            let n1 = ModelNoise { variance_floor: arcsec_to_rad(0.1).powi(2), drift_per_day: 0.0, curvature_per_day2: 0.0 };

            let (_, _, r_k1_n0) = m.predict_cone_base(t, &n0, 1.0);
            let (_, _, r_k3_n0) = m.predict_cone_base(t, &n0, 3.0);
            let (_, _, r_k1_n1) = m.predict_cone_base(t, &n1, 1.0);

            prop_assert!(r_k3_n0 >= r_k1_n0 - 1e-20);
            prop_assert!(r_k1_n1 >= r_k1_n0 - 1e-20);
        }
    }
}

//! Local tangent-plane kinematic model for intra-night seeds.
//!
//! This module provides the core data structure for intra-night kinematic
//! modelling:
//!
//! - [`TangentPlaneModel`] — compact kinematic model on a gnomonic tangent
//!   plane, with optional acceleration and simple covariance propagation.
//!
//! The tangent plane reference frame is carried by
//! [`photom::coordinates::gnomonic_projection::TangentPlane`] (from the
//! `photom` crate), which stores the sky reference direction as an
//! [`photom::coordinates::equatorial::EquCoord`] together with its cached
//! trigonometric terms.
//!
//! Supporting types in this module:
//! - [`PosWithCov`] — position on the plane plus a 2×2 covariance.
//! - [`VelWithCov`] — velocity on the plane plus a 2×2 covariance.
//! - [`Acceleration`] — optional acceleration term (rad/day²).
//!
//! The model is deliberately lightweight and designed to be:
//!
//! - cheap to serialise (via `serde` and `bincode`),
//! - stable numerically by delegating projection logic to the `photom` crate,
//! - sufficient for fast intra-night prediction and cone generation
//!   (not a full orbit determination).
//!
//! ## Typical workflow
//!
//! 1. Build a [`TangentPlaneModel`] from 2 or 3 alerts (see `SeedNode` helpers).
//! 2. Predict position and uncertainty on the tangent plane via
//!    [`TangentPlaneModel::predict_on_plane`].
//! 3. Convert to sky coordinates and cone radius via
//!    [`TangentPlaneModel::predict_cone_base`].
//! 4. Use the cone for spatial indexing or candidate search.

use photom::{
    MJDTT,
    coordinates::{
        cov2::Cov2,
        equatorial::EquCoord,
        gnomonic_projection::{TangentPoint, TangentVec},
    },
};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use crate::engine_config::propagator_config::ModelNoise;

/// Position on the tangent plane paired with a 2×2 position covariance (rad²).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PosWithCov {
    pub tangent_point: TangentPoint,
    pub cov: Cov2,
}

/// Velocity on the tangent plane paired with a 2×2 velocity covariance (rad²/day²).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct VelWithCov {
    pub v: TangentVec, // rad/day
    pub cov: Cov2,     // rad²/day²
}

/// Acceleration on the tangent plane (rad/day²).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Acceleration(pub TangentVec); // rad/day²

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
/// - `x₀, y₀` is the position stored in [`TangentPlaneModel::pos`] at `epoch_mid`,
/// - `vₓ, v_y` is the velocity stored in [`TangentPlaneModel::vel`],
/// - `aₓ, a_y` is the optional acceleration stored in [`TangentPlaneModel::acc`].
///
/// Two covariance matrices are stored:
///
/// - `cov_pos` — position covariance at `epoch_mid` (rad²),
/// - `cov_vel` — velocity covariance (rad²/day²).
///
/// They are propagated in a **simplified, diagonal form** using the
/// [`ModelNoise`] parameters when predicting on the plane.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct TangentPlaneModel {
    /// Reference epoch (typically the middle of the arc), in MJD TT.
    pub epoch_mid: MJDTT,

    /// Position on the tangent plane at `epoch_mid`.
    ///
    /// The tangent plane itself is carried by this field (via
    /// `pos.tangent_point.plane`); all other tangent-plane quantities in
    /// this struct (`vel`, `acc`) are expressed in the same frame.
    pub pos: PosWithCov,

    /// Velocity on the tangent plane (radians/day), expressed in
    /// `pos.tangent_point.plane`.
    pub vel: VelWithCov,

    /// Optional acceleration on the tangent plane (radians/day²),
    /// expressed in `pos.tangent_point.plane`.
    pub acc: Option<Acceleration>,
}

impl Display for TangentPlaneModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TangentPlaneModel {{")?;
        writeln!(f, "  epoch_mid : {}", self.epoch_mid)?;
        writeln!(f, "  pos       : {}", self.pos.tangent_point)?;
        writeln!(f, "  vel       : {}", self.vel.v)?;
        match self.acc {
            Some(acc) => writeln!(f, "  acc       : {}", acc.0)?,
            None => writeln!(f, "  acc       : None")?,
        }
        writeln!(f, "  cov_pos   : {}", self.pos.cov)?;
        writeln!(f, "  cov_vel   : {}", self.vel.cov)?;
        write!(f, "}}")
    }
}

impl TangentPlaneModel {
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
    /// # Arguments
    ///
    /// - `t_target` — Epoch at which to predict the state (MJD TT).
    /// - `noise` — Model noise parameters:
    ///   - `variance_floor` — minimum additional variance (rad²),
    ///   - `drift_per_day` — linear growth term vs. |Δt|,
    ///   - `curvature_per_day2` — quadratic growth term vs. Δt².
    ///
    /// # Returns
    ///
    /// `(pos, cov)` where:
    /// - `pos` is the predicted [`TangentPoint`] on the tangent plane,
    /// - `cov` is a 2×2 covariance matrix for position (rad²).
    ///
    /// # Notes
    ///
    /// - Acceleration is applied only if [`TangentPlaneModel::acc`] is
    ///   `Some`; otherwise the model is purely linear.
    /// - The covariance propagation is intentionally simplified and assumes
    ///   independence between x and y in the added noise term.
    #[inline]
    pub fn predict_on_plane(&self, t_target: MJDTT, noise: &ModelNoise) -> (TangentPoint, Cov2) {
        let dt = t_target - self.epoch_mid;

        let p = self.predict_position(dt);

        // Time-dependent isotropic process noise.
        let q = noise.variance_floor
            + noise.drift_per_day * dt.abs()
            + noise.curvature_per_day2 * dt * dt;

        // cov_pos(t) ≈ cov_pos + Δt² · cov_vel + q · I
        let dt2 = dt * dt;
        let cov = (self.pos.cov + self.vel.cov * dt2).inflate_isotropic(q);

        (p, cov)
    }

    /// Predict position on the tangent plane after dt days.
    ///
    /// # Arguments
    ///
    /// - `dt` — Time difference from `epoch_mid` (days).
    ///
    /// # Returns
    ///
    /// Predicted [`TangentPoint`] on the tangent plane.
    #[inline]
    pub fn predict_position(&self, dt: f64) -> TangentPoint {
        let mut p = self.pos.tangent_point + self.vel.v * dt;
        if let Some(acc) = self.acc {
            p = p + acc.0 * (0.5 * dt * dt);
        }
        p
    }

    /// Predict velocity on the tangent plane after dt days.
    ///
    /// # Arguments
    ///
    /// - `dt` — Time difference from `epoch_mid` (days).
    ///
    /// # Returns
    ///
    /// Predicted [`TangentVec`] (velocity on the tangent plane, rad/day).
    #[inline]
    pub fn predict_velocity(&self, dt: f64) -> TangentVec {
        match self.acc {
            Some(acc) => self.vel.v + acc.0 * dt,
            None => self.vel.v,
        }
    }

    /// Predict sky coordinates `(RA, Dec)` at a target epoch.
    ///
    /// This method performs:
    ///
    /// 1. Kinematic propagation on the tangent plane using position, velocity
    ///    and optional acceleration.
    /// 2. Conversion back to sky coordinates via the inverse gnomonic
    ///    projection.
    ///
    /// # Arguments
    ///
    /// - `t_target` — Epoch at which to predict the sky position (MJD TT).
    ///
    /// # Returns
    ///
    /// [`EquCoord`] — Predicted sky position (RA and Dec in radians).
    ///
    /// # Notes
    ///
    /// - Uncertainty is **not** returned here; for cone-based searches use
    ///   [`TangentPlaneModel::predict_cone_base`] instead.
    /// - As with all tangent-plane models, accuracy degrades as the object
    ///   drifts far from the reference centre or outside the validity time
    ///   range of the fit.
    #[inline]
    pub fn predict_radec(&self, t_target: MJDTT) -> EquCoord {
        let dt = t_target - self.epoch_mid;
        self.predict_position(dt).unproject()
    }

    /// Predict a **cone** on the sky (centre + radius) without spatial padding.
    ///
    /// This is the core building block for cone-based candidate searches:
    ///
    /// 1. Predict position and covariance on the tangent plane via
    ///    [`TangentPlaneModel::predict_on_plane`].
    /// 2. Convert the predicted position to `(RA, Dec)` using
    ///    the inverse gnomonic projection.
    /// 3. Extract the maximum eigenvalue of the 2×2 covariance matrix
    ///    and convert it to a 1σ angular radius.
    /// 4. Multiply by `k_sigma` to obtain a conservative cone radius.
    ///
    /// # Arguments
    ///
    /// - `t_target` — Epoch at which to predict the cone (MJD TT).
    /// - `noise` — Model noise parameters used in the plane prediction.
    /// - `k_sigma` — Multiplicative factor applied to the 1σ radius derived
    ///   from the largest eigenvalue of the covariance.
    ///
    /// # Returns
    ///
    /// `(center, radius)` where:
    /// - `center` — cone centre as an [`EquCoord`] on the sky,
    /// - `radius` — angular radius of the cone (radians).
    ///
    /// # Notes
    ///
    /// - The covariance used here is the result of
    ///   [`TangentPlaneModel::predict_on_plane`]; only its largest eigenvalue
    ///   is considered, making the cone circular even if the underlying
    ///   uncertainty is anisotropic.
    /// - Any additional padding related to spatial indexing cells (e.g. adding
    ///   a HEALPix cell radius) should be applied *outside* this method; see
    ///   `SeedNode::predict_cone` for an example.
    #[inline]
    pub fn predict_cone_base(
        &self,
        t_target: MJDTT,
        noise: &ModelNoise,
        k_sigma: f64,
    ) -> (EquCoord, f64) {
        let (p, cov) = self.predict_on_plane(t_target, noise);
        let center = p.unproject();
        let radius = k_sigma * cov.lambda_max().sqrt();
        (center, radius)
    }
}

#[cfg(test)]
mod tangent_plane_tests {
    use super::*;
    use approx::abs_diff_eq;
    use proptest::prelude::*;

    use photom::coordinates::gnomonic_projection::{TangentPlane, TangentPoint, TangentVec};

    use crate::{astro_math::arcsec_to_rad, engine_config::propagator_config::ModelNoise};

    const EPS: f64 = 1e-12;

    /* ------------------------------ helpers ------------------------------ */

    /// Build a [`TangentPlane`] centred at `(ra0, dec0)` with zero errors.
    fn plane_at(ra0: f64, dec0: f64) -> TangentPlane {
        TangentPlane::new(EquCoord::new(ra0, 0.0, dec0, 0.0))
    }

    /// Build a [`TangentPlaneModel`] with the given parameters.
    ///
    /// - `plane` is the gnomonic reference frame.
    /// - `pos_xy` is the initial position on the plane (rad).
    /// - `vel_xy` is the initial velocity on the plane (rad/day).
    /// - `acc_xy` is the optional acceleration (rad/day²).
    /// - `cov_pos` and `cov_vel` are diagonal covariances (only `[0][0]`
    ///   and `[1][1]` are used; off-diagonal is set to zero).
    fn mk_model(
        plane: TangentPlane,
        epoch_mid: f64,
        pos_xy: [f64; 2],
        vel_xy: [f64; 2],
        acc_xy: Option<[f64; 2]>,
        cov_pos: [[f64; 2]; 2],
        cov_vel: [[f64; 2]; 2],
    ) -> TangentPlaneModel {
        TangentPlaneModel {
            epoch_mid,
            pos: PosWithCov {
                tangent_point: TangentPoint::new(plane, pos_xy[0], pos_xy[1]),
                cov: Cov2 {
                    xx: cov_pos[0][0],
                    yy: cov_pos[1][1],
                    xy: cov_pos[0][1],
                },
            },
            vel: VelWithCov {
                v: TangentVec {
                    dx: vel_xy[0],
                    dy: vel_xy[1],
                },
                cov: Cov2 {
                    xx: cov_vel[0][0],
                    yy: cov_vel[1][1],
                    xy: cov_vel[0][1],
                },
            },
            acc: acc_xy.map(|[ax, ay]| Acceleration(TangentVec { dx: ax, dy: ay })),
        }
    }

    /* ------------------------------ unit tests ------------------------------ */

    #[test]
    fn tangent_plane_new_stores_reference_coords() {
        let ra0 = 1.234;
        let dec0 = 0.5;
        let plane = plane_at(ra0, dec0);
        assert!(abs_diff_eq!(plane.equ_ref.ra, ra0, epsilon = EPS));
        assert!(abs_diff_eq!(plane.equ_ref.dec, dec0, epsilon = EPS));
    }

    #[test]
    fn tangent_plane_project_matches_free_function() {
        let ra0 = 1.0;
        let dec0 = 0.3;
        let plane = plane_at(ra0, dec0);

        let ra = 1.01;
        let dec = 0.31;

        // Project onto the plane and unproject back; should recover (ra, dec).
        let p1 = plane.project(&EquCoord::new(ra, 0.0, dec, 0.0));
        let equ = p1.unproject();
        assert!(abs_diff_eq!(equ.ra, ra, epsilon = EPS));
        assert!(abs_diff_eq!(equ.dec, dec, epsilon = EPS));
    }

    #[test]
    fn model_new_stores_fields() {
        let ra0 = 1.0;
        let dec0 = 0.2;
        let plane = plane_at(ra0, dec0);
        let epoch_mid = 60000.0;
        let pos = [0.01, -0.02];
        let vel = [1e-3, -2e-3];
        let acc = Some([5e-6, -3e-6]);
        let cov_pos = [[1e-6, 0.0], [0.0, 2e-6]];
        let cov_vel = [[1e-8, 0.0], [0.0, 2e-8]];

        let m = mk_model(plane, epoch_mid, pos, vel, acc, cov_pos, cov_vel);

        assert!(abs_diff_eq!(m.epoch_mid, epoch_mid, epsilon = EPS));
        assert!(abs_diff_eq!(m.pos.tangent_point.x, pos[0], epsilon = EPS));
        assert!(abs_diff_eq!(m.pos.tangent_point.y, pos[1], epsilon = EPS));
        assert!(abs_diff_eq!(m.vel.v.dx, vel[0], epsilon = EPS));
        assert!(abs_diff_eq!(m.vel.v.dy, vel[1], epsilon = EPS));
        assert!(m.acc.is_some());
        assert!(abs_diff_eq!(
            m.pos.tangent_point.plane.equ_ref.ra,
            ra0,
            epsilon = EPS
        ));
        assert!(abs_diff_eq!(
            m.pos.tangent_point.plane.equ_ref.dec,
            dec0,
            epsilon = EPS
        ));
    }

    #[test]
    fn predict_on_plane_linear_without_acceleration() {
        let plane = plane_at(1.0, 0.3);
        let epoch_mid = 60000.0;
        let pos = [0.0, 0.0];
        let vel = [1e-3, -2e-3];
        let m = mk_model(
            plane,
            epoch_mid,
            pos,
            vel,
            None,
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        );

        let noise = ModelNoise {
            variance_floor: 0.0,
            drift_per_day: 0.0,
            curvature_per_day2: 0.0,
        };

        let t = epoch_mid + 2.0; // Δt = 2 days
        let (p, cov) = m.predict_on_plane(t, &noise);
        assert!(abs_diff_eq!(p.x, vel[0] * 2.0, epsilon = EPS));
        assert!(abs_diff_eq!(p.y, vel[1] * 2.0, epsilon = EPS));

        assert!(abs_diff_eq!(cov.xx, 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(cov.yy, 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(cov.xy, 0.0, epsilon = EPS));
    }

    #[test]
    fn predict_on_plane_applies_acceleration_and_noise() {
        let plane = plane_at(1.5, -0.2);
        let epoch_mid = 60000.0;
        let pos = [0.1, -0.1];
        let vel = [4e-3, 3e-3];
        let acc = Some([1e-3, -5e-4]);
        let cov_pos = [[1e-6, 0.0], [0.0, 2e-6]];
        let cov_vel = [[1e-8, 0.0], [0.0, 3e-8]];
        let m = mk_model(plane, epoch_mid, pos, vel, acc, cov_pos, cov_vel);

        let noise = ModelNoise {
            variance_floor: 1e-9,
            drift_per_day: 2e-9,
            curvature_per_day2: 3e-9,
        };

        let dt = 0.5;
        let (p, cov) = m.predict_on_plane(epoch_mid + dt, &noise);

        let px_expected = pos[0] + vel[0] * dt + 0.5 * acc.unwrap()[0] * dt * dt;
        let py_expected = pos[1] + vel[1] * dt + 0.5 * acc.unwrap()[1] * dt * dt;
        assert!(abs_diff_eq!(p.x, px_expected, epsilon = 1e-15));
        assert!(abs_diff_eq!(p.y, py_expected, epsilon = 1e-15));

        let q = noise.variance_floor
            + noise.drift_per_day * dt.abs()
            + noise.curvature_per_day2 * dt * dt;
        let sxx_expected = cov_pos[0][0] + dt * dt * cov_vel[0][0] + q;
        let syy_expected = cov_pos[1][1] + dt * dt * cov_vel[1][1] + q;
        assert!(abs_diff_eq!(cov.xx, sxx_expected, epsilon = EPS));
        assert!(abs_diff_eq!(cov.yy, syy_expected, epsilon = EPS));
        assert!(abs_diff_eq!(cov.xy, 0.0, epsilon = EPS));
    }

    #[test]
    fn predict_radec_roundtrip_with_plane_propagation() {
        let ra0 = 2.0;
        let dec0 = 0.1;
        let plane = plane_at(ra0, dec0);
        let epoch_mid = 61000.0;
        let pos = [0.002, -0.003];
        let vel = [1e-3, 2e-3];
        let m = mk_model(
            plane,
            epoch_mid,
            pos,
            vel,
            None,
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        );

        let t = epoch_mid + 1.0;
        let equ = m.predict_radec(t);

        // Project back to tangent plane; should match plane propagation.
        let tp = plane.project(&equ);
        let px = pos[0] + vel[0] * 1.0;
        let py = pos[1] + vel[1] * 1.0;

        assert!(abs_diff_eq!(tp.x, px, epsilon = 5e-12));
        assert!(abs_diff_eq!(tp.y, py, epsilon = 5e-12));
    }

    #[test]
    fn predict_cone_base_produces_non_negative_radius() {
        let plane = plane_at(1.0, 0.3);
        let m = mk_model(
            plane,
            60000.0,
            [0.0, 0.0],
            [0.0, 0.0],
            None,
            [[1e-6, 0.0], [0.0, 2e-6]],
            [[1e-7, 0.0], [0.0, 3e-7]],
        );

        let noise = ModelNoise {
            variance_floor: 1e-9,
            drift_per_day: 0.0,
            curvature_per_day2: 0.0,
        };

        let (center, r) = m.predict_cone_base(60001.0, &noise, 3.0);
        assert!(center.ra.is_finite() && center.dec.is_finite());
        assert!(r >= 0.0);
    }

    #[test]
    fn predict_cone_base_increases_with_k_sigma() {
        let plane = plane_at(1.0, 0.3);
        let m = mk_model(
            plane,
            60000.0,
            [0.0, 0.0],
            [0.0, 0.0],
            None,
            [[1e-6, 0.0], [0.0, 1e-6]],
            [[0.0, 0.0], [0.0, 0.0]],
        );

        let noise = ModelNoise {
            variance_floor: 1e-9,
            drift_per_day: 0.0,
            curvature_per_day2: 0.0,
        };

        let (_, r1) = m.predict_cone_base(60000.5, &noise, 1.0);
        let (_, r2) = m.predict_cone_base(60000.5, &noise, 3.0);
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
            let plane = plane_at(ra0, dec0);
            let m = mk_model(
                plane, epoch_mid, [pos_x, pos_y], [vel_x, vel_y],
                Some([acc_x, acc_y]),
                [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
            );

            let noise = ModelNoise { variance_floor: 0.0, drift_per_day: 0.0, curvature_per_day2: 0.0 };
            let (p, _) = m.predict_on_plane(epoch_mid + dt, &noise);

            let px = pos_x + vel_x * dt + 0.5 * acc_x * dt * dt;
            let py = pos_y + vel_y * dt + 0.5 * acc_y * dt * dt;

            prop_assert!(abs_diff_eq!(p.x, px, epsilon = 5e-12));
            prop_assert!(abs_diff_eq!(p.y, py, epsilon = 5e-12));
        }

        /// Predict_radec composed with `TangentPlane::project` should recover plane propagation locally.
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
            let plane = plane_at(ra0, dec0);
            let m = mk_model(
                plane, epoch_mid, [pos_x, pos_y], [vel_x, vel_y], None,
                [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
            );

            let equ = m.predict_radec(epoch_mid + dt);
            let tp = plane.project(&equ);

            let px = pos_x + vel_x * dt;
            let py = pos_y + vel_y * dt;

            prop_assert!(abs_diff_eq!(tp.x, px, epsilon = 5e-12));
            prop_assert!(abs_diff_eq!(tp.y, py, epsilon = 5e-12));
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
            let plane = plane_at(ra0, dec0);
            let m = mk_model(
                plane, epoch_mid, [pos_x, pos_y], [0.0, 0.0], None,
                [[1e-8, 0.0], [0.0, 1e-8]], [[0.0, 0.0], [0.0, 0.0]],
            );

            let n0 = ModelNoise { variance_floor: 0.0, drift_per_day: 0.0, curvature_per_day2: 0.0 };
            let n1 = ModelNoise { variance_floor: arcsec_to_rad(0.1).powi(2), drift_per_day: 0.0, curvature_per_day2: 0.0 };

            let (_, r_k1_n0) = m.predict_cone_base(t, &n0, 1.0);
            let (_, r_k3_n0) = m.predict_cone_base(t, &n0, 3.0);
            let (_, r_k1_n1) = m.predict_cone_base(t, &n1, 1.0);

            prop_assert!(r_k3_n0 >= r_k1_n0 - 1e-20);
            prop_assert!(r_k1_n1 >= r_k1_n0 - 1e-20);
        }
    }
}

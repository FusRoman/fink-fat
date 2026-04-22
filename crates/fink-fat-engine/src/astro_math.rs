//! Lightweight geometric and numerical utilities for sky-coordinate arithmetic.
//!
//! This module provides the low-level building blocks for seed construction,
//! kinematic propagation, and feature computation in the fink-fat pipeline.
//! All functions operate on plain Rust primitives (`f64`, fixed-size arrays)
//! to keep hot paths allocation-free.
//!
//! ## Function groups
//!
//! ### Tangent-plane projection
//! - [`planar_offset_fast`] — small-angle Cartesian offsets around a center.
//!
//! ### Kinematics
//! - [`fit_quad_tangent`] — analytic quadratic-motion fit in a tangent plane.
//! - [`fit_quad_1d`] — analytic quadratic-motion fit through three time samples.
//!
//! ### Numerical guards and unit conversion
//! - [`safe_ln`] — natural logarithm with zero fallback for non-positive inputs.
//! - [`wrap_pm_pi`] — fold an angle into $(-\pi, \pi]$.
//! - [`arcsec_to_rad`] — arcseconds to radians.

use std::f64::consts::PI;

use photom::{
    Arcseconds, Radians,
    coordinates::gnomonic_projection::{TangentPoint, TangentVec},
};

/// Compute a numerically safe natural logarithm.
///
/// This helper returns `ln(x)` only when the input is strictly positive
/// and finite. Otherwise, it returns `0.0`.
///
/// # Arguments
///
/// - `x` – Input scalar value.
///
/// # Returns
///
/// `ln(x)` if `x > 0` and finite, otherwise `0.0`.
///
/// # Notes
///
/// - This function is typically used for log-transformed features
///   (e.g. `log(chi² + ε)`) where negative or non-finite values are
///   not meaningful.
/// - Returning `0.0` instead of `-∞` or `NaN` avoids destabilizing
///   downstream ML pipelines.
#[inline]
pub fn safe_ln(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        x.ln()
    } else {
        0.0
    }
}

/// Fit a quadratic motion in a tangent plane through three projected points.
///
/// Applies [`fit_quad_1d`] independently to the `x` and `y` tangent-plane
/// coordinates, then reassembles the results into typed tangent-plane values.
/// All three input points must lie on the same [`TangentPlane`]; this is
/// verified with a `debug_assert`.
///
/// # Arguments
///
/// - `dt`  – Array of three time offsets `[t0, t1, t2]` (in days), expressed
///   relative to the same origin.
/// - `pts` – Array of three [`TangentPoint`] values corresponding to the times
///   in `dt`. All three must share the same [`TangentPlane`].
///
/// # Return
///
/// `(p0, v, a)` where:
/// - `p0` – [`TangentPoint`] giving the position at $t = 0$, on the same plane
///   as the inputs.
/// - `v`  – [`TangentVec`] velocity at $t = 0$ (tangent-plane units per day).
/// - `a`  – [`TangentVec`] constant acceleration (tangent-plane units per day²).
///
/// # Panics
///
/// Panics (in debug builds) if the three points do not share the same tangent
/// plane. Also panics if any two time samples are equal (see [`fit_quad_1d`]).
pub fn fit_quad_tangent(
    dt: [f64; 3],
    pts: [TangentPoint; 3],
) -> (TangentPoint, TangentVec, TangentVec) {
    debug_assert_eq!(pts[0].plane, pts[1].plane);
    debug_assert_eq!(pts[1].plane, pts[2].plane);

    let (p0x, vx, ax) = fit_quad_1d(dt, [pts[0].x, pts[1].x, pts[2].x]);
    let (p0y, vy, ay) = fit_quad_1d(dt, [pts[0].y, pts[1].y, pts[2].y]);

    (
        TangentPoint::new(pts[0].plane, p0x, p0y),
        TangentVec { dx: vx, dy: vy },
        TangentVec { dx: ax, dy: ay },
    )
}

/// Wrap an angle into the interval $(-\pi, \pi]$.
///
/// Useful when computing RA differences that should be taken modulo $2\pi$.
///
/// # Arguments
///
/// - `x` – Input angle in radians (unbounded).
///
/// # Returns
///
/// Angle in radians, $y \in (-\pi, \pi]$.
#[inline]
pub fn wrap_pm_pi(x: Radians) -> Radians {
    let two_pi = 2.0 * PI;
    let mut y = (x + PI) % two_pi;
    if y < 0.0 {
        y += two_pi;
    }
    y - PI
}

/// Compute tangent-plane offsets around `(ra0, dec0)` using a precomputed `cos(dec0)`.
///
/// This routine projects a target position `(ra, dec)` onto the local tangent
/// plane centered at `(ra0, dec0)`. It is intended for **small angular
/// separations**, where a Cartesian approximation is sufficient.
///
/// The offsets are:
/// - `dx = wrap_pm_pi(ra − ra0) * cos(dec0)`
/// - `dy = dec − dec0`
///
/// # Arguments
///
/// - `ra0` – Center right ascension (radians).
/// - `dec0` – Center declination (radians).
/// - `cos_dec0` – Precomputed cosine of `dec0`, i.e. `dec0.cos()`.
/// - `ra` – Target right ascension (radians).
/// - `dec` – Target declination (radians).
///
/// # Returns
///
/// `(dx, dy)` tangent-plane offsets in **radians**.
#[inline]
pub fn planar_offset_fast(
    ra0: Radians,
    dec0: Radians,
    cos_dec0: f64,
    ra: Radians,
    dec: Radians,
) -> (Radians, Radians) {
    let dx = wrap_pm_pi(ra - ra0) * cos_dec0;
    let dy = dec - dec0;
    (dx, dy)
}

/// Convert an angle from arcseconds to radians.
///
/// Applies $1\text{ arcsec} = \pi / 648000$ radians.
///
/// # Arguments
///
/// - `x` – Angle in arcseconds.
///
/// # Returns
///
/// Angle in radians.
#[inline]
pub fn arcsec_to_rad(x: Arcseconds) -> Radians {
    x * PI / (180.0 * 3600.0)
}

/// Fit a 1D quadratic motion model through three time samples.
///
/// Fits the kinematic model $x(t) = p\_0 + v\, t + \tfrac{1}{2} a\, t^2$
/// through **exactly three samples** $(t\_i, x\_i)$ using an analytic solution.
/// Returns the position $p\_0$, velocity $v$, and acceleration $a$ evaluated
/// at $t = 0$.
///
/// The three time samples must be **distinct** and expressed relative to the
/// same origin.  The motion is assumed to be well approximated by constant
/// acceleration over each interval.
///
/// # Why this formulation?
///
/// Instead of solving a full linear system, this implementation:
/// - first estimates local velocities using finite differences,
/// - then derives the acceleration from the *change in velocity*,
/// - and finally recovers $(p\_0, v)$ consistently.
///
/// This is numerically stable for small time spans, fast (no matrix
/// inversion), and well suited for short-arc astrometric fitting.
///
/// # Arguments
///
/// - `dt` – Array of three time offsets `[t0, t1, t2]` (in days),
///   expressed **relative to the same origin**.
/// - `x`  – Array of three scalar positions `[x0, x1, x2]` corresponding
///   to the times in `dt`.
///
/// # Returns
///
/// `(p0, v, a)` where:
/// - `p0` – position at $t = 0$,
/// - `v`  – velocity at $t = 0$,
/// - `a`  – constant acceleration.
///
/// # Notes
///
/// - The reference time $t = 0$ does **not** need to coincide with any of
///   the sample times.
/// - Choosing $t = 0$ near the middle sample reduces numerical correlations
///   between $p\_0$, $v$, and $a$.
///
/// # Panics
///
/// Panics if any two time samples are equal (division by zero).
#[inline]
pub fn fit_quad_1d(dt: [f64; 3], x: [f64; 3]) -> (f64, f64, f64) {
    // Unpack time samples
    let (t0, t1, t2) = (dt[0], dt[1], dt[2]);

    // Inverse time intervals between consecutive samples
    // (used for finite-difference velocity estimates)
    let inv_01 = 1.0 / (t1 - t0);
    let inv_12 = 1.0 / (t2 - t1);

    // First-order finite-difference velocities on [t0, t1] and [t1, t2]
    let d01 = (x[1] - x[0]) * inv_01;
    let d12 = (x[2] - x[1]) * inv_12;

    // Inverse total time span (t2 - t0)
    let inv_20 = 1.0 / (t2 - t0);

    // Acceleration is twice the slope of the velocity change:
    //
    //   a = 2 · (d12 − d01) / (t2 − t0)
    //
    // This follows from the quadratic model where velocity varies linearly.
    let a = 2.0 * (d12 - d01) * inv_20;

    // Velocity at t = 0.
    //
    // We start from the velocity on [t0, t1] and remove the contribution
    // of acceleration evaluated at the midpoint of the interval.
    let v = d01 - 0.5 * a * (t0 + t1);

    // Position at t = 0, obtained by rearranging:
    //
    //   x(t1) = p0 + v·t1 + 0.5·a·t1²
    let p0 = x[1] - v * t1 - 0.5 * a * t1 * t1;

    (p0, v, a)
}

#[cfg(test)]
mod astro_math_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // -------------------------------------------------------------------------
    // wrap_pm_pi
    // -------------------------------------------------------------------------

    mod wrap_pm_pi_tests {
        use super::*;

        #[test]
        fn output_is_in_range_for_boundary_inputs() {
            for x in [
                0.0,
                PI,
                -PI,
                3.0 * PI,
                -3.0 * PI,
                10.0 * PI,
                -10.0 * PI,
                1e6,
                -1e6,
            ] {
                let y = wrap_pm_pi(x);
                assert!(
                    y >= -PI - 1e-12 && y <= PI + 1e-12,
                    "wrap_pm_pi({x}) = {y} is not in [-π, π]"
                );
            }
        }

        #[test]
        fn simple_cases() {
            assert_abs_diff_eq!(wrap_pm_pi(0.0), 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(wrap_pm_pi(PI), -PI, epsilon = 1e-12);
            assert_abs_diff_eq!(wrap_pm_pi(-PI), -PI, epsilon = 1e-12);
            assert_abs_diff_eq!(wrap_pm_pi(3.0 * PI), -PI, epsilon = 1e-12);
            assert_abs_diff_eq!(wrap_pm_pi(-3.0 * PI), -PI, epsilon = 1e-12);
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

            #[test]
            fn prop_output_always_in_range(x in -1e6f64..1e6f64) {
                let y = wrap_pm_pi(x);
                prop_assert!(y > -PI - 1e-9 && y <= PI + 1e-9);
            }
        }
    }

    // -------------------------------------------------------------------------
    // arcsec_to_rad
    // -------------------------------------------------------------------------

    mod arcsec_to_rad_tests {
        use super::*;

        #[test]
        fn zero_maps_to_zero() {
            assert_abs_diff_eq!(arcsec_to_rad(0.0), 0.0, epsilon = 0.0);
        }

        #[test]
        fn one_degree_equals_pi_over_180() {
            // 3600 arcsec = 1 degree = π/180 rad
            assert_abs_diff_eq!(arcsec_to_rad(3600.0), PI / 180.0, epsilon = 1e-15);
        }

        #[test]
        fn full_circle_is_two_pi() {
            // 360 degrees = 1_296_000 arcsec = 2π rad
            assert_abs_diff_eq!(arcsec_to_rad(1_296_000.0), 2.0 * PI, epsilon = 1e-10);
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

            #[test]
            fn prop_positive_input_gives_positive_output(x in 0.0f64..1e9) {
                prop_assert!(arcsec_to_rad(x) >= 0.0);
            }

            #[test]
            fn prop_linear_scaling(x in 0.0f64..1e6, k in 0.0f64..100.0) {
                let tol = 1e-12 * (arcsec_to_rad(x * k).abs().max(1.0));
                prop_assert!((arcsec_to_rad(x * k) - k * arcsec_to_rad(x)).abs() < tol);
            }
        }
    }

    // -------------------------------------------------------------------------
    // planar_offset_fast
    // -------------------------------------------------------------------------

    mod planar_offset_fast_tests {
        use super::*;

        #[test]
        fn zero_offset_at_center() {
            let ra0 = 1.0;
            let dec0 = 0.4_f64;
            let (dx, dy) = planar_offset_fast(ra0, dec0, dec0.cos(), ra0, dec0);
            assert_abs_diff_eq!(dx, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(dy, 0.0, epsilon = 1e-15);
        }

        #[test]
        fn small_offset_matches_first_order_approximation() {
            let ra0 = 1.0;
            let dec0 = 0.3_f64;
            let cos_dec0 = dec0.cos();
            let d_ra = 1e-6;
            let d_dec = -2e-6;

            let (dx, dy) = planar_offset_fast(ra0, dec0, cos_dec0, ra0 + d_ra, dec0 + d_dec);

            assert_abs_diff_eq!(dx, d_ra * cos_dec0, epsilon = 1e-15);
            assert_abs_diff_eq!(dy, d_dec, epsilon = 1e-15);
        }

        #[test]
        fn dec_offset_is_exact() {
            // dy must always equal dec - dec0 exactly (no cos factor).
            let ra0 = 2.0;
            let dec0 = 0.1_f64;
            let d_dec = 0.05;
            let (_, dy) = planar_offset_fast(ra0, dec0, dec0.cos(), ra0, dec0 + d_dec);
            assert_abs_diff_eq!(dy, d_dec, epsilon = 1e-15);
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

            #[test]
            fn prop_planar_distance_approximates_ang_sep_for_small_offsets(
                ra0  in 0.0f64..(2.0 * PI),
                dec0 in -(PI / 2.0 - 1e-6)..(PI / 2.0 - 1e-6),
                d_ra  in -1e-4f64..1e-4f64,
                d_dec in -1e-4f64..1e-4f64,
            ) {
                let cos_dec0 = dec0.cos();
                let (dx, dy) = planar_offset_fast(ra0, dec0, cos_dec0, ra0 + d_ra, dec0 + d_dec);
                let r_tan = (dx * dx + dy * dy).sqrt();

                let cos_d = dec0.sin() * (dec0 + d_dec).sin()
                    + dec0.cos() * (dec0 + d_dec).cos() * d_ra.cos();
                let d_sph = cos_d.clamp(-1.0, 1.0).acos();

                prop_assert!((r_tan - d_sph).abs() < 1e-6);
            }
        }
    }

    // -------------------------------------------------------------------------
    // safe_ln
    // -------------------------------------------------------------------------

    mod safe_ln_tests {
        use super::*;

        #[test]
        fn known_values() {
            assert_abs_diff_eq!(safe_ln(1.0), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(std::f64::consts::E), 1.0, epsilon = 1e-15);
        }

        #[test]
        fn non_positive_returns_zero() {
            assert_abs_diff_eq!(safe_ln(0.0), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(-1.0), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(-1e10), 0.0, epsilon = 0.0);
        }

        #[test]
        fn non_finite_returns_zero() {
            assert_abs_diff_eq!(safe_ln(f64::NAN), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(f64::INFINITY), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(f64::NEG_INFINITY), 0.0, epsilon = 0.0);
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

            #[test]
            fn prop_output_always_finite(x in any::<f64>()) {
                prop_assert!(safe_ln(x).is_finite());
            }

            #[test]
            fn prop_equals_ln_for_positive_finite(x in 1e-100f64..1e100f64) {
                prop_assert!((safe_ln(x) - x.ln()).abs() <= 0.0);
            }
        }
    }

    // -------------------------------------------------------------------------
    // fit_quad_1d
    // -------------------------------------------------------------------------

    mod fit_quad_1d_tests {
        use super::*;

        #[test]
        fn exact_recovery_for_known_quadratic() {
            let (p0_true, v_true, a_true) = (0.3, -0.2, 0.05);
            let dt = [-1.0, 0.0, 2.0];
            let x = dt.map(|t| p0_true + v_true * t + 0.5 * a_true * t * t);
            let (p0, v, a) = fit_quad_1d(dt, x);
            assert_abs_diff_eq!(p0, p0_true, epsilon = 1e-12);
            assert_abs_diff_eq!(v, v_true, epsilon = 1e-12);
            assert_abs_diff_eq!(a, a_true, epsilon = 1e-12);
        }

        #[test]
        fn pure_linear_motion_gives_zero_acceleration() {
            let (p0_true, v_true) = (1.0, 0.5);
            let dt = [0.0, 1.0, 3.0];
            let x = dt.map(|t| p0_true + v_true * t);
            let (p0, v, a) = fit_quad_1d(dt, x);
            assert_abs_diff_eq!(p0, p0_true, epsilon = 1e-12);
            assert_abs_diff_eq!(v, v_true, epsilon = 1e-12);
            assert_abs_diff_eq!(a, 0.0, epsilon = 1e-12);
        }

        #[test]
        fn constant_position_gives_zero_velocity_and_acceleration() {
            let dt = [-2.0, 0.0, 3.0];
            let x = [7.0_f64; 3];
            let (p0, v, a) = fit_quad_1d(dt, x);
            assert_abs_diff_eq!(p0, 7.0, epsilon = 1e-12);
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(a, 0.0, epsilon = 1e-12);
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

            #[test]
            fn prop_recovers_arbitrary_quadratic(
                t0 in -2.0f64..0.0,
                t1 in 0.1f64..1.0,
                t2 in 1.1f64..3.0,
                p0_true in -1e4f64..1e4f64,
                v_true  in -1e4f64..1e4f64,
                a_true  in -1e4f64..1e4f64,
            ) {
                prop_assume!(t0 < t1 && t1 < t2);
                let dt = [t0, t1, t2];
                let x = dt.map(|t| p0_true + v_true * t + 0.5 * a_true * t * t);
                let (p0, v, a) = fit_quad_1d(dt, x);
                let scale = p0_true.abs().max(v_true.abs()).max(a_true.abs()).max(1.0);
                prop_assert!((p0 - p0_true).abs() < 1e-8 * scale);
                prop_assert!((v  - v_true ).abs() < 1e-8 * scale);
                prop_assert!((a  - a_true ).abs() < 1e-8 * scale);
            }
        }
    }

    // -------------------------------------------------------------------------
    // fit_quad_tangent
    // -------------------------------------------------------------------------

    mod fit_quad_tangent_tests {
        use photom::coordinates::{
            equatorial::EquCoord,
            gnomonic_projection::{TangentPlane, TangentPoint},
        };

        use super::*;

        fn make_plane() -> TangentPlane {
            TangentPlane::new(EquCoord::new(0.1, 0.0, 0.05, 0.0))
        }

        #[test]
        fn linear_motion_gives_zero_acceleration() {
            let plane = make_plane();
            // Three collinear points at t = -1, 0, 1 with v = (0.01, 0.005) rad/day.
            let dt = [-1.0, 0.0, 1.0];
            let vx = 0.01_f64;
            let vy = 0.005_f64;
            let pts = dt.map(|t| TangentPoint::new(plane, vx * t, vy * t));

            let (p0, v, a) = fit_quad_tangent(dt, pts);

            assert_abs_diff_eq!(p0.x, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(p0.y, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(v.dx, vx, epsilon = 1e-12);
            assert_abs_diff_eq!(v.dy, vy, epsilon = 1e-12);
            assert_abs_diff_eq!(a.dx, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(a.dy, 0.0, epsilon = 1e-12);
        }

        #[test]
        fn known_quadratic_is_recovered() {
            let plane = make_plane();
            let (p0x_true, vx_true, ax_true) = (0.02, 0.01, -0.002);
            let (p0y_true, vy_true, ay_true) = (-0.01, 0.005, 0.001);
            let dt = [-1.0, 0.0, 2.0];

            let pts = dt.map(|t| {
                let x = p0x_true + vx_true * t + 0.5 * ax_true * t * t;
                let y = p0y_true + vy_true * t + 0.5 * ay_true * t * t;
                TangentPoint::new(plane, x, y)
            });

            let (p0, v, a) = fit_quad_tangent(dt, pts);

            assert_abs_diff_eq!(p0.x, p0x_true, epsilon = 1e-12);
            assert_abs_diff_eq!(p0.y, p0y_true, epsilon = 1e-12);
            assert_abs_diff_eq!(v.dx, vx_true, epsilon = 1e-12);
            assert_abs_diff_eq!(v.dy, vy_true, epsilon = 1e-12);
            assert_abs_diff_eq!(a.dx, ax_true, epsilon = 1e-12);
            assert_abs_diff_eq!(a.dy, ay_true, epsilon = 1e-12);
        }
    }
}

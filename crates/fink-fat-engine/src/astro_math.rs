//! Small astronomical math helpers.
//!
//! This module provides lightweight geometric utilities on the celestial
//! sphere, tailored for seeding and orbit-linking:
//! - conversion from (RA, Dec) to 3D unit vectors,
//! - fast dot-products in ℝ³,
//! - angle wrapping into (−π, π],
//! - local tangent-plane projections,
//! - angular conversions (arcsec → rad),
//! - great-circle angular separations.

use std::f64::consts::{PI, TAU};

use crate::{Radians, units::Arcsec};

/// Convert equatorial coordinates `(ra, dec)` to a 3D unit vector on the
/// celestial sphere.
///
/// The mapping assumes:
/// - `ra` is a right ascension in **radians**, in `[0, 2π)` (not enforced),
/// - `dec` is a declination in **radians**, in `[-π/2, π/2]`.
///
/// The returned unit vector is:
/// - `x = cos(dec) * cos(ra)`
/// - `y = cos(dec) * sin(ra)`
/// - `z = sin(dec)`
///
/// Arguments
/// ---------
/// * `ra` – Right ascension in radians.
/// * `dec` – Declination in radians.
///
/// Return
/// ------
/// 3D unit vector `[x, y, z]` on the unit sphere.
#[inline]
pub fn unit_vec(ra: Radians, dec: Radians) -> [f64; 3] {
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

/// Compute the dot product of two 3D vectors.
///
/// This is a minimal helper to keep hot paths explicit and branch-free.
///
/// Arguments
/// ---------
/// * `a` – First vector `[x, y, z]`.
/// * `b` – Second vector `[x, y, z]`.
///
/// Return
/// ------
/// Scalar dot product `a · b`.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Wrap an angle into the interval (−π, π].
///
/// This is useful when working with longitudes or right ascensions where
/// differences should be taken modulo `2π`.
///
/// The returned value `y` satisfies:
/// - `-π < y <= π`
///
/// Arguments
/// ---------
/// * `x` – Input angle in radians (unbounded).
///
/// Return
/// ------
/// Angle in radians wrapped to the principal interval (−π, π].
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
/// Arguments
/// ---------
/// * `ra0` – Center right ascension (radians).
/// * `dec0` – Center declination (radians).
/// * `cos_dec0` – Precomputed cosine of `dec0`, i.e. `dec0.cos()`.
/// * `ra` – Target right ascension (radians).
/// * `dec` – Target declination (radians).
///
/// Return
/// ------
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
/// This is a thin helper encoding:
/// `1 arcsec = π / (180 × 3600)` radians.
///
/// Arguments
/// ---------
/// * `x` – Angle in arcseconds.
///
/// Return
/// ------
/// Angle in radians.
#[inline]
pub fn arcsec_to_rad(x: Arcsec) -> Radians {
    x * PI / (180.0 * 3600.0)
}

/// Compute the great-circle angular separation between two sky positions.
///
/// This routine evaluates the spherical law of cosines:
/// `cos(d) = sin(dec1)·sin(dec2) + cos(dec1)·cos(dec2)·cos(Δra)`,
/// where `Δra = wrap_pm_pi(ra2 − ra1)`.  
/// The separation `d` is then obtained as `acos(clamp(cos(d), −1, 1))`.
///
/// The result is always in `[0, π]` and symmetric in its arguments.
///
/// Arguments
/// ---------
/// * `ra1` – First point right ascension (radians).
/// * `dec1` – First point declination (radians).
/// * `ra2` – Second point right ascension (radians).
/// * `dec2` – Second point declination (radians).
///
/// Return
/// ------
/// Great-circle angular distance `d` in **radians**.
#[inline]
pub fn ang_sep(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let s1 = dec1.sin();
    let c1 = dec1.cos();
    let s2 = dec2.sin();
    let c2 = dec2.cos();
    let dlon = wrap_pm_pi(ra2 - ra1);

    let cos_d = s1 * s2 + c1 * c2 * dlon.cos();
    cos_d.clamp(-1.0, 1.0).acos()
}

/// Compute the great-circle angular separation between two sky positions
/// using the **Vincenty formula**.
///
/// This formulation is numerically stable for all separations, including
/// near the poles and antipodal points. It corresponds to the implementation
/// used in Astropy's ``angular_separation``.
///
/// Arguments
/// ---------
/// * `lon1` – Longitude of the first point (radians).
/// * `lat1` – Latitude of the first point (radians).
/// * `lon2` – Longitude of the second point (radians).
/// * `lat2` – Latitude of the second point (radians).
///
/// Return
/// ------
/// Angular separation in **radians**, guaranteed to lie in `[0, π]`.
///
/// Notes
/// -----
/// This uses the Vincenty formula:
/// `d = atan2( sqrt( num1² + num2² ), denom )` where:
/// - `dlon = lon2 − lon1`
/// - `num1 =  cos(lat2)·sin(dlon)`
/// - `num2 =  cos(lat1)·sin(lat2) − sin(lat1)·cos(lat2)·cos(dlon)`
/// - `denom = sin(lat1)·sin(lat2) + cos(lat1)·cos(lat2)·cos(dlon)`
///
/// This formula is stable at all angular distances.
///
/// See also
/// --------
/// * https://en.wikipedia.org/wiki/Great-circle_distance
#[inline]
pub fn angular_separation_vincenty(
    lon1: Radians,
    lat1: Radians,
    lon2: Radians,
    lat2: Radians,
) -> Radians {
    let dlon = lon2 - lon1;

    let (slon, clon) = dlon.sin_cos();
    let (slat1, clat1) = lat1.sin_cos();
    let (slat2, clat2) = lat2.sin_cos();

    let num1 = clat2 * slon;
    let num2 = clat1 * slat2 - slat1 * clat2 * clon;
    let denom = slat1 * slat2 + clat1 * clat2 * clon;

    num1.hypot(num2).atan2(denom)
}

// -----------------------------------------------------------------------------
// Numerical guard constants
// -----------------------------------------------------------------------------

/// Smallest allowed value for the denominator `cosc` in gnomonic projection.
///
/// Context
/// -------
/// In the gnomonic projection:
/// ```text
/// cosc = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra - ra0)
/// x    = cos(dec) sin(ra - ra0) / cosc
/// y    = [cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra - ra0)] / cosc
/// ```
/// When the target direction lies close to **90° from the tangent point**,
/// `cosc → 0` and the exact projection diverges to infinity.
///
/// Why this constant?
/// ------------------
/// Mathematically, divergence is expected and correct. But numerically, when
/// `cosc` reaches values smaller than ~1e-15, floating-point rounding can produce:
/// - divisions by zero,
/// - NaNs,
/// - overflow into `Inf` even for non-pathological inputs.
///
/// We therefore enforce:
/// ```text
/// inv = 1.0 / max(cosc, INV_COSC_MIN)
/// ```
///
/// Choice of value
/// ----------------
/// - `1e-12` keeps enough dynamic range for small-angle work (few degrees),
/// - avoids Inf/NaN for borderline cases,
/// - does *not* distort the projection in the regime where we actually use it:
///   all fink-fat seeds remain well within the validity domain.
///
/// This value is not physically meaningful; it's a **numerical safety floor**.
const INV_COSC_MIN: f64 = 1e-12;

/// Lower bound used to avoid division by a nearly-zero vector norm when
/// averaging or normalizing spherical vectors.
///
/// Context
/// -------
/// - Used in `spherical_midpoint()`: we add two unit vectors and normalize them.
/// - When the directions are **nearly opposite**, the sum vector can approach
///   the zero vector, making its length extremely small.
///
/// Consequences without guard
/// --------------------------
/// A norm `r ≈ 0` produces catastrophic amplification of noise when
/// normalizing `(x/r, y/r, z/r)` → `NaN`, `Inf`, or huge garbage values.
///
/// Why this constant?
/// ------------------
/// We clamp:
/// ```text
/// r = max(r, NORM_MIN)
/// ```
/// before normalizing.
///
/// Choice of value
/// ---------------
/// - `1e-16` is slightly above the smallest meaningful double-precision values
///   for normalized vectors (~1e-308 is too small, causes underflow well before).
/// - It preserves stability without biasing typical use.
/// - It only activates in extreme geometries (nearly antipodal sources) that we
///   *never* use for seed construction anyway.
///
/// This constant is purely a **numerical robustness guard**.
const NORM_MIN: f64 = 1e-16;

/// Compute a robust spherical midpoint between two directions (ra, dec).
///
/// Returns the angular mean using vector averaging. Not the exact geodesic
/// midpoint, but stable and very good to define a tangent-plane center.
#[inline]
pub fn spherical_midpoint(
    ra1: Radians,
    dec1: Radians,
    ra2: Radians,
    dec2: Radians,
) -> (Radians, Radians) {
    let (x1, y1, z1) = sph_to_cart(ra1, dec1);
    let (x2, y2, z2) = sph_to_cart(ra2, dec2);
    let (x, y, z) = (x1 + x2, y1 + y2, z1 + z2);
    let r = (x * x + y * y + z * z).sqrt().max(NORM_MIN);
    cart_to_sph(x / r, y / r, z / r)
}

/// Gnomonic projection of a sky position onto a tangent plane centered at (ra0, dec0).
///
/// Angles in radians. `(x, y)` are in radians on the tangent plane.
#[inline]
pub fn radec_to_tangent(ra: Radians, dec: Radians, ra0: Radians, dec0: Radians) -> [f64; 2] {
    let (sdec, cdec) = dec.sin_cos();
    let (sdec0, cdec0) = dec0.sin_cos();
    let dra = ra - ra0;
    let (sdra, cdra) = dra.sin_cos();

    // cosc = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(dra)
    let cosc = cdec0 * cdec * cdra + sdec0 * sdec;
    let inv = 1.0 / cosc.max(INV_COSC_MIN);

    let x = cdec * sdra * inv;
    let y = (cdec0 * sdec - sdec0 * cdec * cdra) * inv;
    [x, y]
}

/// Inverse gnomonic projection: map plane coordinates `(x, y)` back to `(ra, dec)`.
///
/// All angles in radians. `ra` is normalized to `[0, 2π)`.
#[inline]
pub fn tangent_to_radec(x: f64, y: f64, ra0: Radians, dec0: Radians) -> (Radians, Radians) {
    let rho2 = x * x + y * y;
    if rho2 < 1e-24 {
        return (ra0.rem_euclid(TAU), dec0);
    }
    let rho = rho2.sqrt();
    let c = rho.atan();
    let (sc, cc) = c.sin_cos();
    let (s0, c0) = dec0.sin_cos();

    let dec = (cc * s0 + (y * sc * c0) / rho).asin();
    let denom = rho * c0 * cc - y * s0 * sc;
    let ra = ra0 + (x * sc).atan2(denom);
    (ra.rem_euclid(TAU), dec)
}

/// Fit a quadratic through three samples x(t) = p0 + v·t + 0.5·a·t².
///
/// Times should be given **relative** to some origin (par ex. `t - t_mid`).
#[inline]
pub fn fit_quad_1d(dt: [f64; 3], x: [f64; 3]) -> (f64, f64, f64) {
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

/// Return the largest eigenvalue λ_max of a symmetric 2×2 matrix.
///
/// Useful to turn a covariance into a scalar radius for cone searches.
#[inline]
pub fn lambda_max_2x2(a: [[f64; 2]; 2]) -> f64 {
    let a11 = a[0][0];
    let a22 = a[1][1];
    let a12 = 0.5 * (a[0][1] + a[1][0]); // symmetrize

    let tr = a11 + a22;
    let rad = (a11 - a22).hypot(2.0 * a12);
    0.5 * (tr + rad)
}

/* --------------------------- Private helpers --------------------------- */

/// Spherical → cartesian unit vector.
#[inline]
fn sph_to_cart(ra: Radians, dec: Radians) -> (f64, f64, f64) {
    let (sdec, cdec) = dec.sin_cos();
    let (sra, cra) = ra.sin_cos();
    (cdec * cra, cdec * sra, sdec)
}

/// Cartesian → spherical (ra in [0, 2π)).
#[inline]
fn cart_to_sph(x: f64, y: f64, z: f64) -> (Radians, Radians) {
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let inv_r = 1.0 / r;
    let dec = (z * inv_r).asin();
    let ra = y.atan2(x).rem_euclid(TAU);
    (ra, dec)
}

#[cfg(test)]
mod astro_math_tests {
    use super::*;
    use approx::abs_diff_eq;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-12;
    const SMALL_EPS: f64 = 1e-9;

    /* ------------------------------ unit tests ------------------------------ */

    #[test]
    fn unit_vec_has_unit_norm() {
        // Test a small grid in (ra, dec) and ensure |v| ≈ 1.
        for &ra in &[0.0, PI / 6.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            for &dec in &[-PI / 2.0 + 1e-3, -PI / 3.0, 0.0, PI / 4.0, PI / 2.0 - 1e-3] {
                let v = unit_vec(ra, dec);
                let norm2 = dot3(v, v);
                assert!(
                    abs_diff_eq!(norm2, 1.0, epsilon = 1e-12),
                    "norm² must be ~1, got {norm2} for ra={ra}, dec={dec}"
                );
            }
        }
    }

    #[test]
    fn unit_vec_special_positions() {
        // North pole: (ra arbitrary, dec = +π/2) → (0, 0, 1)
        let v_n = unit_vec(1.234, PI / 2.0);
        assert!(abs_diff_eq!(v_n[0], 0.0, epsilon = 1e-12));
        assert!(abs_diff_eq!(v_n[1], 0.0, epsilon = 1e-12));
        assert!(abs_diff_eq!(v_n[2], 1.0, epsilon = 1e-12));

        // Equator: (ra = 0, dec = 0) → (1, 0, 0)
        let v_e = unit_vec(0.0, 0.0);
        assert!(abs_diff_eq!(v_e[0], 1.0, epsilon = 1e-12));
        assert!(abs_diff_eq!(v_e[1], 0.0, epsilon = 1e-12));
        assert!(abs_diff_eq!(v_e[2], 0.0, epsilon = 1e-12));
    }

    #[test]
    fn dot3_behaves_as_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [-2.0, 0.5, 4.0];
        let d = dot3(a, b);
        let expected = 1.0 * -2.0 + 2.0 * 0.5 + 3.0 * 4.0;
        assert!(abs_diff_eq!(d, expected, epsilon = EPS));
    }

    #[test]
    fn wrap_pm_pi_output_range() {
        let test_values = [
            0.0,
            PI,
            -PI,
            3.0 * PI,
            -3.0 * PI,
            10.0 * PI,
            -10.0 * PI,
            1e6,
            -1e6,
        ];

        for x in test_values {
            let y = wrap_pm_pi(x);
            assert!(
                y >= -PI - 1e-12 && y < PI + 1e-12,
                "wrap_pm_pi({x}) = {y} is not in [-π, π)"
            );
        }
    }

    #[test]
    fn wrap_pm_pi_simple_cases() {
        assert!(abs_diff_eq!(wrap_pm_pi(0.0), 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(wrap_pm_pi(PI), -PI, epsilon = EPS));
        assert!(abs_diff_eq!(wrap_pm_pi(-PI), -PI, epsilon = EPS));
        assert!(abs_diff_eq!(wrap_pm_pi(3.0 * PI), -PI, epsilon = EPS));
        assert!(abs_diff_eq!(wrap_pm_pi(-3.0 * PI), -PI, epsilon = EPS));
    }

    #[test]
    fn planar_offset_fast_zero_at_center() {
        let ra0 = 1.0;
        let dec0: f64 = 0.4;
        let cos_dec0 = dec0.cos();

        let (dx, dy) = planar_offset_fast(ra0, dec0, cos_dec0, ra0, dec0);
        assert!(abs_diff_eq!(dx, 0.0, epsilon = EPS));
        assert!(abs_diff_eq!(dy, 0.0, epsilon = EPS));
    }

    #[test]
    fn planar_offset_fast_small_offset_matches_delta() {
        let ra0 = 1.0;
        let dec0: f64 = 0.3;
        let cos_dec0 = dec0.cos();

        let d_ra = 1e-6;
        let d_dec = -2e-6;

        // For small offsets, dx ≈ d_ra * cos(dec0), dy ≈ d_dec.
        let (dx, dy) = planar_offset_fast(ra0, dec0, cos_dec0, ra0 + d_ra, dec0 + d_dec);

        assert!(abs_diff_eq!(dx, d_ra * cos_dec0, epsilon = 1e-12));
        assert!(abs_diff_eq!(dy, d_dec, epsilon = 1e-12));
    }

    #[test]
    fn arcsec_to_rad_consistency() {
        // 0 arcsec → 0 rad
        assert!(abs_diff_eq!(arcsec_to_rad(0.0), 0.0, epsilon = EPS));

        // 3600 arcsec = 1 degree = π/180 rad
        let one_degree_rad = PI / 180.0;
        let val = arcsec_to_rad(3600.0);
        assert!(abs_diff_eq!(val, one_degree_rad, epsilon = 1e-12));
    }

    #[test]
    fn ang_sep_basic_properties() {
        // Same point → 0
        let d0 = ang_sep(1.0, 0.2, 1.0, 0.2);
        assert!(abs_diff_eq!(d0, 0.0, epsilon = EPS));

        // Antipodal points (0,0) and (π,0) → π
        let d_pi = ang_sep(0.0, 0.0, PI, 0.0);
        assert!(abs_diff_eq!(d_pi, PI, epsilon = 1e-12));

        // Symmetry: sep(p1, p2) == sep(p2, p1)
        let d1 = ang_sep(0.3, -0.1, 1.1, 0.5);
        let d2 = ang_sep(1.1, 0.5, 0.3, -0.1);
        assert!(abs_diff_eq!(d1, d2, epsilon = EPS));
    }

    #[test]
    fn ang_sep_matches_unit_vec_dot_product() {
        // Check consistency between spherical formula and dot3(unit_vec, unit_vec).
        let points = &[
            (0.1, 0.0, 1.0, 0.2),
            (2.0, 0.4, 2.2, 0.45),
            (5.0, -0.3, 0.1, 0.1),
        ];

        for &(ra1, dec1, ra2, dec2) in points {
            let d = ang_sep(ra1, dec1, ra2, dec2);
            let v1 = unit_vec(ra1, dec1);
            let v2 = unit_vec(ra2, dec2);
            let cos_d = d.cos();
            let dot = dot3(v1, v2);
            assert!(
                abs_diff_eq!(cos_d, dot, epsilon = 1e-12),
                "cos(d) and dot product mismatch: cos(d)={cos_d}, dot={dot}"
            );
        }
    }

    /* --------------------------- property-based tests --------------------------- */

    fn ra_strategy() -> impl Strategy<Value = f64> {
        0.0f64..(2.0 * PI)
    }

    fn dec_strategy() -> impl Strategy<Value = f64> {
        // Avoid exactly ±π/2 to reduce numeric pathologies at poles.
        let eps = 1e-6;
        (-(PI / 2.0 - eps))..(PI / 2.0 - eps)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        /// `unit_vec` must always output a vector of unit norm within numerical tolerance.
        #[test]
        fn prop_unit_vec_is_unit_length(ra in ra_strategy(), dec in dec_strategy()) {
            let v = unit_vec(ra, dec);
            let norm2 = dot3(v, v);
            prop_assert!(abs_diff_eq!(norm2, 1.0, epsilon = 1e-12));
        }

        /// `wrap_pm_pi` always returns an angle in (−π, π].
        #[test]
        fn prop_wrap_pm_pi_range(x in -1e6f64..1e6f64) {
            let y = wrap_pm_pi(x);
            prop_assert!(y > -PI - SMALL_EPS && y <= PI + SMALL_EPS);
        }

        /// For small offsets, the tangent-plane distance approximates the great-circle distance.
        #[test]
        fn prop_planar_offset_approximates_ang_sep_small_offsets(
            ra0 in ra_strategy(),
            dec0 in dec_strategy(),
            d_ra in -1e-4f64..1e-4f64,
            d_dec in -1e-4f64..1e-4f64,
        ) {
            let cos_dec0 = dec0.cos();
            let ra = ra0 + d_ra;
            let dec = dec0 + d_dec;

            let (dx, dy) = planar_offset_fast(ra0, dec0, cos_dec0, ra, dec);
            let r_tan = (dx*dx + dy*dy).sqrt();

            let d_sph = ang_sep(ra0, dec0, ra, dec);

            // Small-angle approximation: r_tan ≈ d_sph
            let diff = (r_tan - d_sph).abs();
            prop_assert!(diff < 1e-6);
        }

        /// `ang_sep` must be symmetric, non-negative, and ≤ π.
        #[test]
        fn prop_ang_sep_basic_properties(
            ra1 in ra_strategy(),
            dec1 in dec_strategy(),
            ra2 in ra_strategy(),
            dec2 in dec_strategy(),
        ) {
            let d12 = ang_sep(ra1, dec1, ra2, dec2);
            let d21 = ang_sep(ra2, dec2, ra1, dec1);

            prop_assert!(d12 >= 0.0);
            prop_assert!(d12 <= PI + SMALL_EPS);
            prop_assert!(abs_diff_eq!(d12, d21, epsilon = 1e-12));
        }

        /// `ang_sep` must be consistent with the dot product of the corresponding unit vectors:
        /// cos(d) ≈ unit_vec(p1) · unit_vec(p2).
        #[test]
        fn prop_ang_sep_matches_unit_vec_dot(
            ra1 in ra_strategy(),
            dec1 in dec_strategy(),
            ra2 in ra_strategy(),
            dec2 in dec_strategy(),
        ) {
            let d = ang_sep(ra1, dec1, ra2, dec2);
            let v1 = unit_vec(ra1, dec1);
            let v2 = unit_vec(ra2, dec2);
            let cos_d = d.cos();
            let dot = dot3(v1, v2);

            prop_assert!(abs_diff_eq!(cos_d, dot, epsilon = 1e-12));
        }
    }

    #[cfg(test)]
    mod vincenty_tests {
        use super::*;
        use approx::abs_diff_eq;
        use std::f64::consts::PI;

        const EPS: f64 = 1e-12;
        const SMALL_EPS: f64 = 1e-9;

        /* ------------------------------ unit tests ------------------------------ */

        #[test]
        fn vincenty_zero_separation_same_point() {
            let d = angular_separation_vincenty(1.234, 0.5, 1.234, 0.5);
            assert!(
                abs_diff_eq!(d, 0.0, epsilon = EPS),
                "separation of identical points must be ~0, got {d}"
            );
        }

        #[test]
        fn vincenty_equator_quarter_circle() {
            // (0, 0) to (π/2, 0) → quarter great circle → π/2
            let d = angular_separation_vincenty(0.0, 0.0, PI / 2.0, 0.0);
            assert!(
                abs_diff_eq!(d, PI / 2.0, epsilon = 1e-12),
                "expected π/2, got {d}"
            );
        }

        #[test]
        fn vincenty_pole_quarter_circle() {
            // (0, 0) to (0, π/2) → quarter great circle → π/2
            let d = angular_separation_vincenty(0.0, 0.0, 0.0, PI / 2.0);
            assert!(
                abs_diff_eq!(d, PI / 2.0, epsilon = 1e-12),
                "expected π/2, got {d}"
            );
        }

        #[test]
        fn vincenty_antipodal_points() {
            // (0, 0) and (π, 0) are antipodal → separation = π
            let d = angular_separation_vincenty(0.0, 0.0, PI, 0.0);
            assert!(
                abs_diff_eq!(d, PI, epsilon = 1e-12),
                "antipodal points must be ~π, got {d}"
            );
        }

        #[test]
        fn vincenty_symmetry() {
            let lon1 = 0.7;
            let lat1 = -0.3;
            let lon2 = 2.1;
            let lat2 = 0.4;

            let d12 = angular_separation_vincenty(lon1, lat1, lon2, lat2);
            let d21 = angular_separation_vincenty(lon2, lat2, lon1, lat1);

            assert!(
                abs_diff_eq!(d12, d21, epsilon = EPS),
                "separation must be symmetric, d12={d12}, d21={d21}"
            );
        }

        #[test]
        fn vincenty_in_range_0_to_pi() {
            let cases = &[
                (0.1, -0.2, 1.5, 0.7),
                (2.3, 0.4, 0.5, -0.1),
                (5.8, -1.0, 1.1, 1.0),
                (0.0, 0.0, PI, 0.0),       // antipodal
                (0.0, 0.0, 0.0, PI / 2.0), // pole
            ];

            for &(lon1, lat1, lon2, lat2) in cases {
                let d = angular_separation_vincenty(lon1, lat1, lon2, lat2);
                assert!(
                    d >= 0.0 - 1e-15 && d <= PI + 1e-15,
                    "separation must be in [0, π], got {d}"
                );
            }
        }

        /* --------------------------- property-based tests --------------------------- */

        fn lon_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }

        fn lat_strategy() -> impl Strategy<Value = f64> {
            // Avoid exactement ±π/2 pour réduire les pathologies numériques aux pôles.
            let eps = 1e-9;
            (-(PI / 2.0 - eps))..(PI / 2.0 - eps)
        }

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 64,
                .. ProptestConfig::default()
            })]

            /// Separation is always non-negative, ≤ π, and symmetric.
            #[test]
            fn prop_vincenty_basic_properties(
                lon1 in lon_strategy(),
                lat1 in lat_strategy(),
                lon2 in lon_strategy(),
                lat2 in lat_strategy(),
            ) {
                let d12 = angular_separation_vincenty(lon1, lat1, lon2, lat2);
                let d21 = angular_separation_vincenty(lon2, lat2, lon1, lat1);

                prop_assert!(d12 >= 0.0);
                prop_assert!(d12 <= PI + SMALL_EPS);
                prop_assert!(abs_diff_eq!(d12, d21, epsilon = 1e-12));
            }

            /// Vincenty separation of identical points is ~0.
            #[test]
            fn prop_vincenty_zero_for_identical_points(
                lon in lon_strategy(),
                lat in lat_strategy(),
            ) {
                let d = angular_separation_vincenty(lon, lat, lon, lat);
                prop_assert!(abs_diff_eq!(d, 0.0, epsilon = 1e-12));
            }

            /// For generic points, Vincenty and cosine-law separation should agree
            /// within a small tolerance (except for very small or very large angles).
            #[test]
            fn prop_vincenty_matches_cosine_law_most_of_the_time(
                lon1 in lon_strategy(),
                lat1 in lat_strategy(),
                lon2 in lon_strategy(),
                lat2 in lat_strategy(),
            ) {
                let d_vinc = angular_separation_vincenty(lon1, lat1, lon2, lat2);
                let d_cos  = ang_sep(lon1, lat1, lon2, lat2); // loi des cosinus

                let diff = (d_vinc - d_cos).abs();
                prop_assert!(diff < 1e-9, "Vincenty and cosine-law disagree: diff={diff}, d_vinc={d_vinc}, d_cos={d_cos}");
            }
        }
    }

    // =========================================================================
    // Unit tests — sph_to_cart / cart_to_sph
    // =========================================================================

    #[test]
    fn sph_to_cart_produces_unit_vectors() {
        // Quelques points fixes + check que la norme est ~1.
        let samples = &[
            (0.0, 0.0),
            (PI / 3.0, 0.1),
            (PI, 0.5),
            (1.7 * PI, -0.4),
            (0.0, PI / 2.0 - 1e-6),  // proche du pôle
            (0.0, -PI / 2.0 + 1e-6), // proche de l'autre pôle
        ];

        for &(ra, dec) in samples {
            let (x, y, z) = sph_to_cart(ra, dec);
            let r2 = x * x + y * y + z * z;
            assert!(
                abs_diff_eq!(r2, 1.0, epsilon = 1e-12),
                "r2 = {r2} not close to 1.0"
            );
        }
    }

    #[test]
    fn sph_cart_roundtrip_for_fixed_points() {
        let samples = &[(0.0, 0.0), (PI / 3.0, 0.2), (2.3, -0.7), (5.0, 0.8)];

        for &(ra, dec) in samples {
            let (x, y, z) = sph_to_cart(ra, dec);
            let (ra2, dec2) = cart_to_sph(x, y, z);
            assert!(abs_diff_eq!(ra2, ra.rem_euclid(TAU), epsilon = 1e-12));
            assert!(abs_diff_eq!(dec2, dec, epsilon = 1e-12));
        }
    }

    // =========================================================================
    // Property tests — sph_to_cart / cart_to_sph
    // =========================================================================

    fn any_ra() -> impl Strategy<Value = f64> {
        0.0f64..TAU
    }

    fn any_dec() -> impl Strategy<Value = f64> {
        // on évite les pôles exacts pour garder une marge
        (-PI / 2.0 + 1e-6)..(PI / 2.0 - 1e-6)
    }

    proptest! {
        #[test]
        fn prop_sph_cart_is_roundtrip(ra in any_ra(), dec in any_dec()) {
            let (x, y, z) = sph_to_cart(ra, dec);
            let norm = (x*x + y*y + z*z).sqrt();
            prop_assert!(abs_diff_eq!(norm, 1.0, epsilon = 1e-12));

            let (ra2, dec2) = cart_to_sph(x, y, z);
            // RA est normalisée dans [0, 2π)
            prop_assert!(abs_diff_eq!(ra2, ra.rem_euclid(TAU), epsilon = 1e-12));
            prop_assert!(abs_diff_eq!(dec2, dec, epsilon = 1e-12));
        }
    }

    // =========================================================================
    // Unit tests — spherical_midpoint
    // =========================================================================

    #[test]
    fn spherical_midpoint_of_identical_directions_is_itself() {
        let samples = &[(0.5, 0.2), (2.1, -0.3), (5.9, 0.7)];

        for &(ra, dec) in samples {
            let (ram, decm) = spherical_midpoint(ra, dec, ra, dec);
            assert!(abs_diff_eq!(ram, ra.rem_euclid(TAU), epsilon = 1e-12));
            assert!(abs_diff_eq!(decm, dec, epsilon = 1e-12));
        }
    }

    #[test]
    fn spherical_midpoint_is_symmetric_in_arguments() {
        let a = (0.3, 0.1);
        let b = (1.7, -0.2);

        let (ra_ab, dec_ab) = spherical_midpoint(a.0, a.1, b.0, b.1);
        let (ra_ba, dec_ba) = spherical_midpoint(b.0, b.1, a.0, a.1);

        assert!(abs_diff_eq!(ra_ab, ra_ba, epsilon = 1e-12));
        assert!(abs_diff_eq!(dec_ab, dec_ba, epsilon = 1e-12));
    }

    #[test]
    fn spherical_midpoint_handles_nearly_antipodal_without_nan() {
        // Points ~antipodaux sur l'équateur
        let ra1 = 0.0;
        let dec1 = 0.0;
        let ra2 = PI; // opposé
        let dec2 = 0.0;

        let (ram, decm) = spherical_midpoint(ra1, dec1, ra2, dec2);
        assert!(ram.is_finite());
        assert!(decm.is_finite());

        // Le résultat doit toujours être sur la sphère unité.
        let (x, y, z) = sph_to_cart(ram, decm);
        let r2 = x * x + y * y + z * z;
        assert!(abs_diff_eq!(r2, 1.0, epsilon = 1e-12));
    }

    proptest! {
        #[test]
        fn prop_spherical_midpoint_on_unit_sphere(
            ra1 in any_ra(),
            dec1 in any_dec(),
            ra2 in any_ra(),
            dec2 in any_dec(),
        ) {
            let (ram, decm) = spherical_midpoint(ra1, dec1, ra2, dec2);

            // Coordonnées finies
            prop_assert!(ram.is_finite());
            prop_assert!(decm.is_finite());

            // Sur la sphère unité
            let (x, y, z) = sph_to_cart(ram, decm);
            let r2 = x*x + y*y + z*z;
            prop_assert!(abs_diff_eq!(r2, 1.0, epsilon = 1e-12));
        }
    }

    // =========================================================================
    // Unit tests — gnomonic projection / inverse
    // =========================================================================

    #[test]
    fn gnomonic_projection_of_center_is_zero() {
        let ra0 = 1.2;
        let dec0 = 0.3;

        let [x, y] = radec_to_tangent(ra0, dec0, ra0, dec0);
        assert!(abs_diff_eq!(x, 0.0, epsilon = 1e-15));
        assert!(abs_diff_eq!(y, 0.0, epsilon = 1e-15));

        let (ra2, dec2) = tangent_to_radec(0.0, 0.0, ra0, dec0);
        assert!(abs_diff_eq!(ra2, ra0.rem_euclid(TAU), epsilon = 1e-15));
        assert!(abs_diff_eq!(dec2, dec0, epsilon = 1e-15));
    }

    #[test]
    fn radec_tangent_handles_cosc_zero_without_nan() {
        // Choisir un point ~90° loin du centre pour forcer cosc → 0.
        let ra0 = 0.0;
        let dec0 = 0.0;

        let ra = PI / 2.0;
        let dec = 0.0;

        let [x, y] = radec_to_tangent(ra, dec, ra0, dec0);
        assert!(x.is_finite());
        assert!(y.is_finite());
    }

    #[test]
    fn tangent_to_radec_handles_rho_zero_gracefully() {
        let ra0 = 2.0;
        let dec0 = -0.4;

        let (ra2, dec2) = tangent_to_radec(0.0, 0.0, ra0, dec0);
        assert!(abs_diff_eq!(ra2, ra0.rem_euclid(TAU), epsilon = 1e-15));
        assert!(abs_diff_eq!(dec2, dec0, epsilon = 1e-15));
    }

    fn small_plane() -> impl Strategy<Value = f64> {
        -5e-3f64..5e-3
    }

    proptest! {
        #[test]
        fn prop_gnomonic_roundtrip_small_offsets(
            ra0 in any_ra(),
            dec0 in -0.8f64..0.8,  // on reste loin des pôles pour la gnomonique
            dx in small_plane(),
            dy in small_plane(),
        ) {
            // 1. Part de petites coordonnées planes autour du centre,
            // 2. Va sur la sphère,
            // 3. Re-projette sur le plan.
            let (ra, dec) = tangent_to_radec(dx, dy, ra0, dec0);
            let [x2, y2] = radec_to_tangent(ra, dec, ra0, dec0);

            prop_assert!(x2.is_finite() && y2.is_finite());
            prop_assert!(abs_diff_eq!(x2, dx, epsilon = 5e-12));
            prop_assert!(abs_diff_eq!(y2, dy, epsilon = 5e-12));
        }

        #[test]
        fn prop_gnomonic_inverse_is_locally_stable(
            ra in any_ra(),
            dec in -0.8f64..0.8,
            d_ra in -5e-3f64..5e-3,
            d_dec in -5e-3f64..5e-3,
        ) {
            // Centre proche du point
            let ra0 = (ra + d_ra).rem_euclid(TAU);
            let dec0 = (dec + d_dec).clamp(-0.8, 0.8);

            let [x, y] = radec_to_tangent(ra, dec, ra0, dec0);
            let (ra2, dec2) = tangent_to_radec(x, y, ra0, dec0);

            prop_assert!(ra2.is_finite() && dec2.is_finite());
            prop_assert!(abs_diff_eq!(ra2, ra.rem_euclid(TAU), epsilon = 5e-12));
            prop_assert!(abs_diff_eq!(dec2, dec, epsilon = 5e-12));
        }
    }

    // =========================================================================
    // Unit tests — fit_quad_1d
    // =========================================================================

    #[test]
    fn fit_quad_1d_exact_recovery_for_known_quadratic() {
        // Modèle : x(t) = p0 + v t + 0.5 a t^2
        let p0_true = 0.3;
        let v_true = -0.2;
        let a_true = 0.05;

        let dt = [-1.0, 0.0, 2.0];
        let x = dt.map(|t| p0_true + v_true * t + 0.5 * a_true * t * t);

        let (p0, v, a) = fit_quad_1d(dt, x);

        assert!(abs_diff_eq!(p0, p0_true, epsilon = 1e-12));
        assert!(abs_diff_eq!(v, v_true, epsilon = 1e-12));
        assert!(abs_diff_eq!(a, a_true, epsilon = 1e-12));
    }

    proptest! {
        #[test]
        fn prop_fit_quad_1d_recovers_coeffs(
            // temps strictement croissants pour éviter les divisions par zéro
            t0 in -1.0f64..0.0,
            t1 in 0.0f64..1.0,
            t2 in 1.1f64..2.0,
            p0_true in -1.0f64..1.0,
            v_true in -1.0f64..1.0,
            a_true in -1.0f64..1.0,
        ) {
            prop_assume!(t0 < t1 && t1 < t2);

            let dt = [t0, t1, t2];
            let x = dt.map(|t| p0_true + v_true * t + 0.5 * a_true * t * t);

            let (p0, v, a) = fit_quad_1d(dt, x);

            // La tolérance peut être un peu plus large en prop-test
            prop_assert!(abs_diff_eq!(p0, p0_true, epsilon = 1e-10));
            prop_assert!(abs_diff_eq!(v, v_true, epsilon = 1e-10));
            prop_assert!(abs_diff_eq!(a, a_true, epsilon = 1e-10));
        }
    }

    // =========================================================================
    // Unit tests — lambda_max_2x2
    // =========================================================================

    #[test]
    fn lambda_max_2x2_of_diagonal_is_max_diagonal() {
        let a = [[2.0, 0.0], [0.0, 5.0]];
        let lmax = lambda_max_2x2(a);
        assert!(abs_diff_eq!(lmax, 5.0, epsilon = 1e-15));
    }

    #[test]
    fn lambda_max_2x2_respects_shift_by_identity() {
        let a = [[2.0, 0.3], [0.3, 1.0]];
        let c = 4.0;

        let a_shifted = [[a[0][0] + c, a[0][1]], [a[1][0], a[1][1] + c]];

        let lmax = lambda_max_2x2(a);
        let lmax_shifted = lambda_max_2x2(a_shifted);

        assert!(abs_diff_eq!(lmax_shifted, lmax + c, epsilon = 1e-12));
    }

    #[test]
    fn lambda_max_2x2_non_negative_for_simple_covariances() {
        // matrice de covariance simple PSD
        let a = [[1e-4, 2e-5], [2e-5, 3e-4]];
        let lmax = lambda_max_2x2(a);
        assert!(lmax >= 0.0);
    }

    proptest! {
        #[test]
        fn prop_lambda_max_2x2_scales_with_positive_factor(
            a11 in 0.0f64..1e3,
            a22 in 0.0f64..1e3,
            a12 in -1e2f64..1e2,
            k in 0.0f64..1e3
        ) {
            // matrice symétrique
            let a = [[a11, a12], [a12, a22]];
            let l = lambda_max_2x2(a);

            let a_scaled = [[k * a11, k * a12], [k * a12, k * a22]];
            let l_scaled = lambda_max_2x2(a_scaled);

            if k == 0.0 {
                prop_assert!(abs_diff_eq!(l_scaled, 0.0, epsilon = 1e-12));
            } else {
                prop_assert!(abs_diff_eq!(
                    l_scaled,
                    k * l,
                    epsilon = 1e-9 * (1.0 + k.abs().max(l.abs()))
                ));
            }
        }
    }
}

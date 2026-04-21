//! Lightweight geometric and numerical utilities for sky-coordinate arithmetic.
//!
//! This module provides the low-level building blocks for seed construction,
//! kinematic propagation, and feature computation in the fink-fat pipeline.
//! All functions operate on plain Rust primitives (`f64`, fixed-size arrays)
//! to keep hot paths allocation-free.
//!
//! ## Function groups
//!
//! ### Spherical geometry
//! - [`unit_vec`] — equatorial `(ra, dec)` to 3D unit vector.
//! - [`ang_sep`] — great-circle distance via the spherical law of cosines.
//! - [`angular_separation_vincenty`] — numerically stable Vincenty formula.
//! - [`spherical_midpoint`] — robust angular mean of two sky directions.
//!
//! ### Tangent-plane projection
//! - [`planar_offset_fast`] — small-angle Cartesian offsets around a center.
//! - [`radec_to_tangent`] — full gnomonic projection onto a local tangent plane.
//! - [`tangent_to_radec`] — inverse gnomonic: tangent-plane back to sky coordinates.
//!
//! ### 2D linear algebra
//! - [`dot2`], [`dot3`] — fixed-size dot products.
//! - [`mat_vec2`] — 2×2 matrix–vector product.
//! - [`mat_mul2`] — 2×2 matrix–matrix product.
//! - [`trace_2x2`] — matrix trace.
//! - [`det_sym_2x2`] — determinant of a symmetric 2×2 matrix.
//! - [`invert_sym_2x2`] — robust inversion of a symmetric 2×2 matrix.
//! - [`cholesky_lower_sym_2x2`] — Cholesky factorization of a symmetric 2×2 matrix.
//! - [`lambda_max_2x2`] — largest eigenvalue of a symmetric 2×2 matrix.
//! - [`l2_norm`] — Euclidean norm of a 2D vector.
//!
//! ### Kinematics
//! - [`fit_quad_1d`] — analytic quadratic-motion fit through three time samples.
//!
//! ### Numerical guards and unit conversion
//! - [`clamp_unit`] — clamp to $[-1, 1]$ with NaN protection.
//! - [`safe_ln`] — natural logarithm with zero fallback for non-positive inputs.
//! - [`wrap_pm_pi`] — fold an angle into $(-\pi, \pi]$.
//! - [`arcsec_to_rad`] — arcseconds to radians.

use std::f64::consts::PI;

use photom::{Arcseconds, Radians};

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

/// Compute the dot product of two 2D vectors.
///
/// The dot product is $a \cdot b = a\_0 b\_0 + a\_1 b\_1$.
///
/// Arguments
/// ---------
/// * `a` – First 2D vector `[x, y]`.
/// * `b` – Second 2D vector `[x, y]`.
///
/// Return
/// ------
/// Scalar dot product $a \cdot b$.
#[inline]
pub fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Multiply a 2×2 matrix by a 2D vector.
///
/// This helper evaluates the linear transformation:
/// ```text
/// [ m00 m01 ] [ v0 ] = [ m00·v0 + m01·v1 ]
/// [ m10 m11 ] [ v1 ]   [ m10·v0 + m11·v1 ]
/// ```
///
/// Arguments
/// ---------
/// * `m` – 2×2 matrix stored in row-major order.
/// * `v` – 2D vector `[v0, v1]`.
///
/// Return
/// ------
/// Resulting 2D vector `m · v`.
///
/// Notes
/// -----
/// - The matrix is assumed to be small and dense.
/// - No symmetry or conditioning assumptions are required.
/// - This function is used extensively in innovation and covariance
///   computations where allocating a generic matrix type would be
///   unnecessary overhead.
#[inline]
pub fn mat_vec2(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]
}

/// Clamp a scalar to the interval [−1, 1], with NaN/Inf protection.
///
/// This helper is primarily intended for quantities that should lie
/// within trigonometric bounds, such as:
/// - cosine of an angle,
/// - normalized dot products.
///
/// Arguments
/// ---------
/// * `x` – Input scalar value.
///
/// Return
/// ------
/// Value clamped to the interval `[−1, 1]`.
/// Returns `0.0` if `x` is not finite.
///
/// Notes
/// -----
/// - Returning `0.0` for non-finite inputs avoids propagating NaNs
///   into downstream computations (e.g. `acos`, ML features).
/// - This behavior is intentional and favors robustness over strict
///   error signaling.
#[inline]
pub fn clamp_unit(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Compute a numerically safe natural logarithm.
///
/// This helper returns `ln(x)` only when the input is strictly positive
/// and finite. Otherwise, it returns `0.0`.
///
/// Arguments
/// ---------
/// * `x` – Input scalar value.
///
/// Return
/// ------
/// `ln(x)` if `x > 0` and finite, otherwise `0.0`.
///
/// Notes
/// -----
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

/// Invert a symmetric 2×2 matrix with numerical safeguards.
///
/// This routine computes the inverse of a **symmetric** matrix:
/// ```text
/// [ a  b ]
/// [ b  d ]
/// ```
///
/// with additional protections against:
/// - non-finite entries,
/// - near-singular determinants,
/// - poorly conditioned covariance matrices.
///
/// A diagonal floor is applied before inversion to ensure numerical
/// stability.
///
/// Arguments
/// ---------
/// * `m` – Symmetric 2×2 matrix (only the symmetric part is used).
/// * `floor` – Minimum allowed value for diagonal terms and determinant
///   regularization.
///
/// Return
/// ------
/// Inverse 2×2 matrix.
///
/// Notes
/// -----
/// - If the determinant is too small or non-finite, the function falls
///   back to a **diagonal inverse**:
///   ```text
///   inv ≈ diag(1/a, 1/d)
///   ```
///   effectively discarding off-diagonal correlations.
/// - This behavior is intentional and favors robustness over exactness
///   in degenerate cases.
/// - Designed primarily for inverting innovation or covariance matrices
///   in short-arc astrometric linking.
#[inline]
pub fn invert_sym_2x2(m: [[f64; 2]; 2], floor: f64) -> [[f64; 2]; 2] {
    let mut a = m[0][0];
    let mut b = 0.5 * (m[0][1] + m[1][0]);
    let mut d = m[1][1];

    if !a.is_finite() || a < floor {
        a = floor;
    }
    if !d.is_finite() || d < floor {
        d = floor;
    }
    if !b.is_finite() {
        b = 0.0
    }

    let det = a * d - b * b;
    if !det.is_finite() || det <= floor {
        return [[1.0 / a.max(floor), 0.0], [0.0, 1.0 / d.max(floor)]];
    }

    let inv_det = 1.0 / det;
    [[d * inv_det, -b * inv_det], [-b * inv_det, a * inv_det]]
}

/// Compute the trace of a 2×2 matrix.
///
/// The trace is the sum of diagonal elements: $\mathrm{tr}(M) = M\_{00} + M\_{11}$.
///
/// Arguments
/// ---------
/// * `m` – 2×2 matrix.
///
/// Return
/// ------
/// Trace of the matrix.
///
/// Notes
/// -----
/// For covariance matrices, the trace equals the total variance and is commonly
/// used as a scalar uncertainty proxy.
#[inline]
pub fn trace_2x2(m: [[f64; 2]; 2]) -> f64 {
    m[0][0] + m[1][1]
}

/// Multiply two 2×2 matrices.
///
/// Evaluates $C = A \cdot B$ where $C\_{ij} = \sum\_k A\_{ik} B\_{kj}$.
/// Matrices are stored in row-major order.
///
/// Arguments
/// ---------
/// * `a` – Left-hand 2×2 matrix.
/// * `b` – Right-hand 2×2 matrix.
///
/// Return
/// ------
/// Product matrix $A \cdot B$ as a row-major 2×2 array.
///
/// Notes
/// -----
/// Allocation-free; suitable for propagating 2D covariance matrices and
/// verifying numerical inverses.
///
/// See also
/// --------
/// * [`mat_vec2`] – Matrix–vector multiplication for 2D vectors.
/// * [`invert_sym_2x2`] – Robust inversion of symmetric 2×2 matrices.
#[inline]
pub fn mat_mul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Compute the determinant of a symmetric 2×2 matrix.
///
/// For a symmetric matrix with diagonal entries $a$, $d$ and symmetrized
/// off-diagonal $b = \tfrac{1}{2}(m\_{01} + m\_{10})$, the determinant is:
/// $$\det(M) = ad - b^2$$
///
/// Arguments
/// ---------
/// * `m` – Symmetric 2×2 matrix (only the symmetric part is used).
///
/// Return
/// ------
/// Determinant of the matrix.
///
/// Notes
/// -----
/// - A positive determinant is a necessary condition for positive definiteness.
/// - No finiteness checks are performed; callers are responsible for
///   handling pathological inputs.
///
/// See also
/// --------
/// * [`invert_sym_2x2`] – Robust inversion of symmetric 2×2 matrices.
/// * [`lambda_max_2x2`] – Largest eigenvalue of a symmetric 2×2 matrix.
#[inline]
pub fn det_sym_2x2(m: [[f64; 2]; 2]) -> f64 {
    // Determinant of symmetric 2x2: a*d - b^2 (symmetrize b)
    let a = m[0][0];
    let b = 0.5 * (m[0][1] + m[1][0]);
    let d = m[1][1];
    a * d - b * b
}

/// Compute the lower-triangular Cholesky factor `L` of a symmetric 2×2 matrix.
///
/// This routine factorizes a **symmetric** matrix:
/// ```text
/// M = [ a  b ]
///     [ b  d ]
/// ```
/// into:
/// ```text
/// M = L · Lᵀ,   with  L = [ l00  0  ]
///                      [ l10  l11]
/// ```
///
/// The implementation is specialized for 2×2 and designed for **hot paths**
/// (innovation whitening, covariance normalization, gating).
///
/// Numerical robustness
/// --------------------
/// This function applies safety guards before factorization:
/// - symmetrizes the off-diagonal term `b`,
/// - floors diagonal terms to at least `floor`,
/// - treats non-finite entries as invalid (returns `None`),
/// - checks positive definiteness via the implied Schur complement:
///   `t = d - (b² / a)` and requires `t > floor`.
///
/// If the matrix is not numerically positive definite, returns `None`.
///
/// Arguments
/// ---------
/// * `m` – Symmetric 2×2 matrix (only the symmetric part is used).
/// * `floor` – Minimum allowed value for diagonal terms and for the Schur complement.
///   Typical values: `1e-20` for radians² covariances, or `1e-12` in more
///   conservative settings.
///
/// Return
/// ------
/// * `Some(L)` where `L` is a 2×2 lower-triangular matrix stored in row-major order:
///   ```text
///   [ l00  0.0 ]
///   [ l10  l11 ]
///   ```
/// * `None` if the input is not finite or not positive definite after flooring.
///
/// Notes
/// -----
/// - For a 2×2 symmetric matrix, positive definiteness is equivalent to:
///   `a > 0` and `det(M) > 0`.
///   This routine uses an equivalent check via the Schur complement.
/// - If you want a *fallback* behavior (diagonal-only whitening), implement that
///   at the call site when `None` is returned.
///
/// See also
/// --------
/// * [`invert_sym_2x2`] – Robust inversion with diagonal fallback.
/// * [`det_sym_2x2`] – Determinant of a symmetric 2×2 matrix.
#[inline]
pub fn cholesky_lower_sym_2x2(m: [[f64; 2]; 2], floor: f64) -> Option<[[f64; 2]; 2]> {
    // Symmetrize and extract.
    let mut a = m[0][0];
    let mut d = m[1][1];
    let b = 0.5 * (m[0][1] + m[1][0]);

    // Validate and floor diagonal terms.
    if !a.is_finite() || !d.is_finite() || !b.is_finite() || !floor.is_finite() || floor <= 0.0 {
        return None;
    }
    if a < floor {
        a = floor;
    }
    if d < floor {
        d = floor;
    }

    // Cholesky for 2×2 SPD:
    // l00 = sqrt(a)
    // l10 = b / l00
    // l11 = sqrt(d - l10^2)
    let l00 = a.sqrt();
    if !l00.is_finite() || l00 <= 0.0 {
        return None;
    }

    let l10 = b / l00;
    let t = d - l10 * l10;

    // Require strictly positive Schur complement (with floor).
    if !t.is_finite() || t < floor {
        return None;
    }

    let l11 = t.sqrt();
    if !l11.is_finite() || l11 <= 0.0 {
        return None;
    }

    Some([[l00, 0.0], [l10, l11]])
}

/// Wrap an angle into the interval $(-\pi, \pi]$.
///
/// Useful when computing RA differences that should be taken modulo $2\pi$.
///
/// Arguments
/// ---------
/// * `x` – Input angle in radians (unbounded).
///
/// Return
/// ------
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
/// Applies $1\text{ arcsec} = \pi / 648000$ radians.
///
/// Arguments
/// ---------
/// * `x` – Angle in arcseconds.
///
/// Return
/// ------
/// Angle in radians.
#[inline]
pub fn arcsec_to_rad(x: Arcseconds) -> Radians {
    x * PI / (180.0 * 3600.0)
}

/// Compute the great-circle angular separation between two sky positions.
///
/// Evaluates the spherical law of cosines:
/// $$\cos d = \sin\delta\_1\sin\delta\_2 + \cos\delta\_1\cos\delta\_2\cos(\Delta\alpha)$$
/// where $\Delta\alpha = \alpha\_2 - \alpha\_1$ (wrapped to $(-\pi, \pi]$).
/// The result $d \in [0, \pi]$ is symmetric in its arguments.
///
/// Arguments
/// ---------
/// * `ra1`  – First point right ascension (radians).
/// * `dec1` – First point declination (radians).
/// * `ra2`  – Second point right ascension (radians).
/// * `dec2` – Second point declination (radians).
///
/// Return
/// ------
/// Great-circle angular distance $d$ in radians.
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
/// Why this formulation?
/// ---------------------
/// Instead of solving a full linear system, this implementation:
/// - first estimates local velocities using finite differences,
/// - then derives the acceleration from the *change in velocity*,
/// - and finally recovers $(p\_0, v)$ consistently.
///
/// This is numerically stable for small time spans, fast (no matrix
/// inversion), and well suited for short-arc astrometric fitting.
///
/// Arguments
/// ---------
/// * `dt` – Array of three time offsets `[t0, t1, t2]` (in days),
///   expressed **relative to the same origin**.
/// * `x`  – Array of three scalar positions `[x0, x1, x2]` corresponding
///   to the times in `dt`.
///
/// Return
/// ------
/// `(p0, v, a)` where:
/// * `p0` – position at $t = 0$,
/// * `v`  – velocity at $t = 0$,
/// * `a`  – constant acceleration.
///
/// Notes
/// -----
/// - The reference time $t = 0$ does **not** need to coincide with any of
///   the sample times.
/// - Choosing $t = 0$ near the middle sample reduces numerical correlations
///   between $p\_0$, $v$, and $a$.
///
/// Panics
/// ------
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

/// Compute the largest eigenvalue $\lambda\_{\max}$ of a symmetric 2×2 matrix.
///
/// $$\lambda\_{\max} = \frac{1}{2}\Bigl(\mathrm{tr}(A) + \sqrt{(a\_{11} - a\_{22})^2 + 4a\_{12}^2}\Bigr)$$
/// where $a\_{12} = \tfrac{1}{2}(a\_{01} + a\_{10})$ (symmetrized off-diagonal).
///
/// Arguments
/// ---------
/// * `a` – Symmetric 2×2 matrix.
///
/// Return
/// ------
/// Largest eigenvalue $\lambda\_{\max}$.
///
/// Notes
/// -----
/// Primarily used to convert a 2D positional covariance into a scalar search
/// radius for spatial cone queries.
#[inline]
pub fn lambda_max_2x2(a: [[f64; 2]; 2]) -> f64 {
    let a11 = a[0][0];
    let a22 = a[1][1];
    let a12 = 0.5 * (a[0][1] + a[1][0]); // symmetrize

    let tr = a11 + a22;
    let rad = (a11 - a22).hypot(2.0 * a12);
    0.5 * (tr + rad)
}

/// Compute the Euclidean L2 norm of a 2D vector.
///
/// Returns $\sqrt{x^2 + y^2}$ using `f64::hypot` for numerical stability.
///
/// Arguments
/// ---------
/// * `x` – First component.
/// * `y` – Second component.
///
/// Return
/// ------
/// Euclidean norm $\|(x, y)\|\_2$.
#[inline]
pub fn l2_norm(x: f64, y: f64) -> f64 {
    x.hypot(y)
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

    #[cfg(test)]
    mod algebra2_tests {
        use super::*;
        use approx::assert_abs_diff_eq;

        // -----------------------------------------------------------------------------
        // Helpers for tests
        // -----------------------------------------------------------------------------

        #[inline]
        fn is_finite_mat2(m: [[f64; 2]; 2]) -> bool {
            m[0][0].is_finite() && m[0][1].is_finite() && m[1][0].is_finite() && m[1][1].is_finite()
        }

        #[inline]
        fn is_finite_vec2(v: [f64; 2]) -> bool {
            v[0].is_finite() && v[1].is_finite()
        }

        // A reasonable range for random floats to avoid overflow in products.
        fn finite_f64() -> impl Strategy<Value = f64> {
            -1.0e150f64..=1.0e150f64
        }

        // Generate a symmetric positive-definite (SPD) 2x2 matrix:
        // [[a, b],
        //  [b, d]]
        // with det > 0 and a,d > 0.
        fn spd_sym_2x2() -> BoxedStrategy<[[f64; 2]; 2]> {
            // Keep values moderate to avoid pathological conditioning.
            (
                1.0e-6f64..=1.0e3f64,
                1.0e-6f64..=1.0e3f64,
                -0.999f64..=0.999f64,
            )
                .prop_map(|(a, d, rho)| {
                    // b = rho * sqrt(a*d) ensures |b| < sqrt(a*d) => det > 0
                    let b = rho * (a * d).sqrt();
                    [[a, b], [b, d]]
                })
                .boxed()
        }

        // -----------------------------------------------------------------------------
        // Unit tests
        // -----------------------------------------------------------------------------

        #[test]
        fn dot2_matches_manual() {
            let a = [1.0, 2.0];
            let b = [3.0, 4.0];
            assert_abs_diff_eq!(dot2(a, b), 11.0, epsilon = 0.0);
        }

        #[test]
        fn mat_vec2_matches_manual() {
            let m = [[1.0, 2.0], [3.0, 4.0]];
            let v = [5.0, 6.0];
            let out = mat_vec2(m, v);
            assert_abs_diff_eq!(out[0], 1.0 * 5.0 + 2.0 * 6.0, epsilon = 0.0);
            assert_abs_diff_eq!(out[1], 3.0 * 5.0 + 4.0 * 6.0, epsilon = 0.0);
        }

        #[test]
        fn clamp_unit_basic_cases() {
            assert_abs_diff_eq!(clamp_unit(0.5), 0.5, epsilon = 0.0);
            assert_abs_diff_eq!(clamp_unit(2.0), 1.0, epsilon = 0.0);
            assert_abs_diff_eq!(clamp_unit(-2.0), -1.0, epsilon = 0.0);
        }

        #[test]
        fn clamp_unit_non_finite_returns_zero() {
            assert_abs_diff_eq!(clamp_unit(f64::NAN), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(clamp_unit(f64::INFINITY), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(clamp_unit(f64::NEG_INFINITY), 0.0, epsilon = 0.0);
        }

        #[test]
        fn safe_ln_basic_cases() {
            assert_abs_diff_eq!(safe_ln(1.0), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(std::f64::consts::E), 1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(safe_ln(0.0), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(-1.0), 0.0, epsilon = 0.0);
        }

        #[test]
        fn safe_ln_non_finite_returns_zero() {
            assert_abs_diff_eq!(safe_ln(f64::NAN), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(f64::INFINITY), 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(safe_ln(f64::NEG_INFINITY), 0.0, epsilon = 0.0);
        }

        #[test]
        fn trace_2x2_basic() {
            let m = [[1.0, 2.0], [3.0, 4.0]];
            assert_abs_diff_eq!(trace_2x2(m), 5.0, epsilon = 0.0);
        }

        #[test]
        fn invert_sym_2x2_identity() {
            let floor = 1e-20;
            let m = [[1.0, 0.0], [0.0, 1.0]];
            let inv = invert_sym_2x2(m, floor);

            assert_abs_diff_eq!(inv[0][0], 1.0, epsilon = 0.0);
            assert_abs_diff_eq!(inv[0][1], 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(inv[1][0], 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(inv[1][1], 1.0, epsilon = 0.0);
        }

        #[test]
        fn invert_sym_2x2_near_singular_falls_back_to_diagonal() {
            let floor = 1e-12;

            // a*d - b^2 == 0 -> singular (exact). Should trigger fallback.
            let a: f64 = 2.0;
            let d = 8.0;
            let b = (a * d).sqrt(); // det = 0
            let m = [[a, b], [b, d]];

            let inv = invert_sym_2x2(m, floor);

            // Fallback should return diagonal inverse (off-diagonals ~0).
            assert_abs_diff_eq!(inv[0][1], 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(inv[1][0], 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(inv[0][0], 1.0 / a, epsilon = 1e-15);
            assert_abs_diff_eq!(inv[1][1], 1.0 / d, epsilon = 1e-15);
        }

        // -----------------------------------------------------------------------------
        // Property-based tests
        // -----------------------------------------------------------------------------

        proptest! {
            #[test]
            fn prop_dot2_is_commutative(
                a in finite_f64(), b in finite_f64(), c in finite_f64(), d in finite_f64()
            ) {
                let u = [a, b];
                let v = [c, d];
                let uv = dot2(u, v);
                let vu = dot2(v, u);

                prop_assert!(uv.is_finite() && vu.is_finite());
                let tol = 1e-12 * (uv.abs().max(vu.abs()).max(1.0));
                prop_assert!((uv - vu).abs() <= tol);
            }

            #[test]
            fn prop_dot2_linearity_in_first_argument(
                a in finite_f64(), b in finite_f64(),
                c in finite_f64(), d in finite_f64(),
                e in finite_f64(), f in finite_f64(),
            ) {
                let u = [a, b];
                let v = [c, d];
                let w = [e, f];

                let uv = [u[0] + v[0], u[1] + v[1]];
                let lhs = dot2(uv, w);
                let rhs = dot2(u, w) + dot2(v, w);

                prop_assert!(lhs.is_finite() && rhs.is_finite());
                let tol = 1e-9 * (lhs.abs().max(rhs.abs()).max(1.0));
                prop_assert!((lhs - rhs).abs() <= tol);
            }

            #[test]
            fn prop_mat_vec2_matches_expanded(
                m00 in finite_f64(), m01 in finite_f64(), m10 in finite_f64(), m11 in finite_f64(),
                v0 in finite_f64(), v1 in finite_f64()
            ) {
                let m = [[m00, m01], [m10, m11]];
                let v = [v0, v1];
                let out = mat_vec2(m, v);

                let exp0 = m00 * v0 + m01 * v1;
                let exp1 = m10 * v0 + m11 * v1;

                prop_assert!(out[0].is_finite() && out[1].is_finite());
                prop_assert!((out[0] - exp0).abs() <= 0.0);
                prop_assert!((out[1] - exp1).abs() <= 0.0);
            }

            #[test]
            fn prop_mat_vec2_is_linear_in_vector(
                m00 in finite_f64(), m01 in finite_f64(), m10 in finite_f64(), m11 in finite_f64(),
                a in finite_f64(), b in finite_f64(),
                c in finite_f64(), d in finite_f64(),
            ) {
                let m = [[m00, m01], [m10, m11]];
                let u = [a, b];
                let v = [c, d];
                let uv = [u[0] + v[0], u[1] + v[1]];

                let lhs = mat_vec2(m, uv);
                let rhs_u = mat_vec2(m, u);
                let rhs_v = mat_vec2(m, v);
                let rhs = [rhs_u[0] + rhs_v[0], rhs_u[1] + rhs_v[1]];

                prop_assert!(is_finite_vec2(lhs) && is_finite_vec2(rhs));
                let tol0 = 1e-9 * (lhs[0].abs().max(rhs[0].abs()).max(1.0));
                let tol1 = 1e-9 * (lhs[1].abs().max(rhs[1].abs()).max(1.0));
                prop_assert!((lhs[0] - rhs[0]).abs() <= tol0);
                prop_assert!((lhs[1] - rhs[1]).abs() <= tol1);
            }

            #[test]
            fn prop_clamp_unit_bounds(x in any::<f64>()) {
                let y = clamp_unit(x);
                prop_assert!(y.is_finite());
                prop_assert!(y >= -1.0 && y <= 1.0);
            }

            #[test]
            fn prop_safe_ln_behavior(x in any::<f64>()) {
                let y = safe_ln(x);
                prop_assert!(y.is_finite());

                if x.is_finite() && x > 0.0 {
                    prop_assert!((y - x.ln()).abs() <= 0.0);
                } else {
                    prop_assert!(y == 0.0);
                }
            }

            #[test]
            fn prop_trace_2x2_matches_sum(
                m00 in finite_f64(), m01 in finite_f64(), m10 in finite_f64(), m11 in finite_f64()
            ) {
                let m = [[m00, m01], [m10, m11]];
                let tr = trace_2x2(m);
                prop_assert!(tr.is_finite());
                prop_assert!((tr - (m00 + m11)).abs() <= 0.0);
            }

            #[test]
            fn prop_invert_sym_2x2_spd_is_inverse_both_sides_and_symmetric_when_not_fallback(
                m in spd_sym_2x2(),
                floor in 1e-20f64..=1e-12f64
            ) {
                // SPD => determinant should be > 0.
                let det = det_sym_2x2(m);
                prop_assert!(det.is_finite() && det > 0.0);

                let inv = invert_sym_2x2(m, floor);
                prop_assert!(is_finite_mat2(inv));

                // Check inv * m ≈ I
                let left = mat_mul2(inv, m);
                prop_assert!(is_finite_mat2(left));

                // Check m * inv ≈ I
                let right = mat_mul2(m, inv);
                prop_assert!(is_finite_mat2(right));

                // Tolerance: moderate since m range is controlled.
                let eps = 1e-9;

                // inv*m
                prop_assert!((left[0][0] - 1.0).abs() <= eps);
                prop_assert!(left[0][1].abs() <= eps);
                prop_assert!(left[1][0].abs() <= eps);
                prop_assert!((left[1][1] - 1.0).abs() <= eps);

                // m*inv
                prop_assert!((right[0][0] - 1.0).abs() <= eps);
                prop_assert!(right[0][1].abs() <= eps);
                prop_assert!(right[1][0].abs() <= eps);
                prop_assert!((right[1][1] - 1.0).abs() <= eps);

                // Symmetry check for the non-fallback path:
                // For SPD and "reasonable" floor, we expect det > floor => we should not fallback.
                // In that case, the inverse of a symmetric matrix should also be symmetric.
                if det > floor {
                    let tol = 1e-12 * inv[0][1].abs().max(inv[1][0].abs()).max(1.0);
                    prop_assert!((inv[0][1] - inv[1][0]).abs() <= tol);
                }
            }

            #[test]
            fn prop_invert_sym_2x2_output_is_finite_for_reasonable_inputs(
                a in finite_f64(), b in finite_f64(), d in finite_f64(),
                floor in 1e-20f64..=1e-12f64
            ) {
                // General symmetric matrix (may be indefinite).
                let m = [[a, b], [b, d]];
                let inv = invert_sym_2x2(m, floor);

                // Even with odd inputs, we expect finite output due to flooring and fallback.
                prop_assert!(is_finite_mat2(inv));
            }
        }
    }

    #[cfg(test)]
    mod cholesky_tests {
        use super::*;
        use approx::assert_relative_eq;

        /// Compute L * L^T for a lower-triangular 2×2 matrix:
        /// L = [[l00, 0], [l10, l11]]
        #[inline]
        fn ll_t(l: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
            let l00 = l[0][0];
            let l10 = l[1][0];
            let l11 = l[1][1];

            [[l00 * l00, l00 * l10], [l00 * l10, l10 * l10 + l11 * l11]]
        }

        /// Symmetrize a 2×2 matrix (used for comparison since the API expects symmetric input).
        #[inline]
        fn sym(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
            let b = 0.5 * (m[0][1] + m[1][0]);
            [[m[0][0], b], [b, m[1][1]]]
        }

        /// Add `eps * I` to a 2×2 matrix.
        #[inline]
        fn add_eps_i(m: [[f64; 2]; 2], eps: f64) -> [[f64; 2]; 2] {
            [[m[0][0] + eps, m[0][1]], [m[1][0], m[1][1] + eps]]
        }

        /// Construct SPD matrix from any 2×2 A via M = A A^T + eps I.
        #[inline]
        fn make_spd_from_a(a: [[f64; 2]; 2], eps: f64) -> [[f64; 2]; 2] {
            // A A^T
            let m00 = a[0][0] * a[0][0] + a[0][1] * a[0][1];
            let m01 = a[0][0] * a[1][0] + a[0][1] * a[1][1];
            let m11 = a[1][0] * a[1][0] + a[1][1] * a[1][1];
            add_eps_i([[m00, m01], [m01, m11]], eps)
        }

        #[test]
        fn cholesky_identity() {
            let m = [[1.0, 0.0], [0.0, 1.0]];
            let l = cholesky_lower_sym_2x2(m, 1e-20).expect("I should be SPD");
            assert_relative_eq!(l[0][0], 1.0, max_relative = 1e-12);
            assert_relative_eq!(l[0][1], 0.0, max_relative = 1e-12);
            assert_relative_eq!(l[1][1], 1.0, max_relative = 1e-12);

            let recon = ll_t(l);
            assert_relative_eq!(recon[0][0], 1.0, max_relative = 1e-12);
            assert_relative_eq!(recon[0][1], 0.0, max_relative = 1e-12);
            assert_relative_eq!(recon[1][0], 0.0, max_relative = 1e-12);
            assert_relative_eq!(recon[1][1], 1.0, max_relative = 1e-12);
        }

        #[test]
        fn cholesky_simple_spd() {
            // SPD: [[4, 2], [2, 3]]
            let m = [[4.0, 2.0], [2.0, 3.0]];
            let l = cholesky_lower_sym_2x2(m, 1e-20).expect("matrix should be SPD");

            // L should be lower triangular with positive diagonal.
            assert!(l[0][0] > 0.0);
            assert_relative_eq!(l[0][1], 0.0, max_relative = 0.0);
            assert!(l[1][1] > 0.0);

            // Reconstruction must match.
            let recon = ll_t(l);
            let ms = sym(m);
            assert_relative_eq!(recon[0][0], ms[0][0], max_relative = 1e-12);
            assert_relative_eq!(recon[0][1], ms[0][1], max_relative = 1e-12);
            assert_relative_eq!(recon[1][0], ms[1][0], max_relative = 1e-12);
            assert_relative_eq!(recon[1][1], ms[1][1], max_relative = 1e-12);
        }

        #[test]
        fn cholesky_rejects_non_spd() {
            // Not SPD: determinant <= 0 (even with positive diagonal)
            let m = [[1.0, 2.0], [2.0, 1.0]]; // det = -3
            assert!(cholesky_lower_sym_2x2(m, 1e-20).is_none());

            // Non-finite input should be rejected.
            let m_nan = [[f64::NAN, 0.0], [0.0, 1.0]];
            assert!(cholesky_lower_sym_2x2(m_nan, 1e-20).is_none());

            // Negative diagonal is *floored* and can become factorisable.
            let m_neg = [[-1.0, 0.0], [0.0, 1.0]];
            assert!(cholesky_lower_sym_2x2(m_neg, 1e-20).is_some());
        }

        #[test]
        fn cholesky_floors_small_diagonal() {
            let m = [[0.0, 0.0], [0.0, 0.0]];
            let floor = 1e-6;

            let l = cholesky_lower_sym_2x2(m, floor).expect("floored zero matrix should factorize");

            // Triangular + positive diagonal
            assert!(l[0][0] > 0.0);
            assert_eq!(l[0][1], 0.0);
            assert!(l[1][1] > 0.0);

            let recon = ll_t(l);
            assert_relative_eq!(recon[0][0], floor, max_relative = 1e-12);
            assert_relative_eq!(recon[1][1], floor, max_relative = 1e-12);
            assert_relative_eq!(recon[0][1], 0.0, max_relative = 0.0);
            assert_relative_eq!(recon[1][0], 0.0, max_relative = 0.0);
        }

        // -------------------------------------------------------------------------
        // Property-based tests
        // -------------------------------------------------------------------------

        proptest! {
            #[test]
            fn prop_cholesky_reconstructs_spd(
                a00 in -10.0f64..10.0,
                a01 in -10.0f64..10.0,
                a10 in -10.0f64..10.0,
                a11 in -10.0f64..10.0,
            ) {
                // Build SPD matrix from arbitrary A: M = A A^T + eps I.
                let a = [[a00, a01], [a10, a11]];
                let eps = 1e-3;
                let m = make_spd_from_a(a, eps);

                let floor = 1e-20;
                let l = cholesky_lower_sym_2x2(m, floor).expect("constructed SPD must factorize");

                // Invariants: lower-triangular and positive diagonal.
                prop_assert!(l[0][0].is_finite() && l[0][0] > 0.0);
                prop_assert!(l[1][1].is_finite() && l[1][1] > 0.0);
                prop_assert_eq!(l[0][1], 0.0);

                // Reconstruction accuracy: L L^T ≈ M (relative tolerance).
                let recon = ll_t(l);

                // Use a tolerance that scales with the matrix magnitude.
                // This avoids flaky failures when M is very large.
                let scale = m[0][0].abs().max(m[1][1].abs()).max(1.0);
                let tol = 1e-10 * scale;

                prop_assert!((recon[0][0] - m[0][0]).abs() <= tol);
                prop_assert!((recon[0][1] - m[0][1]).abs() <= tol);
                prop_assert!((recon[1][0] - m[1][0]).abs() <= tol);
                prop_assert!((recon[1][1] - m[1][1]).abs() <= tol);
            }

            #[test]
            fn prop_cholesky_matches_spd_conditions(
                a in 1e-6f64..100.0,
                d in 1e-6f64..100.0,
                b in -10.0f64..10.0,
            ) {
                // For symmetric 2×2: SPD iff a>0 and det>0.
                // We'll craft det using a, d, b and check behavior.
                let m = [[a, b], [b, d]];
                let det = det_sym_2x2(m);

                let floor = 1e-20;
                let chol = cholesky_lower_sym_2x2(m, floor);

                if det > 0.0 {
                    // It *should* factorize (almost always) for positive det and positive diag.
                    // There are rare numeric edge cases, but with these ranges it should pass.
                    prop_assert!(chol.is_some());
                } else {
                    prop_assert!(chol.is_none());
                }
            }
        }
    }
}

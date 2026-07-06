use nalgebra::{Matrix6, Vector3, Vector6};
use outfit::constants::ROT_ECLMJ2000_TO_EQUMJ2000;

use crate::topocentric_kf::init::{unit_los, unit_los_dot};

/// Heliocentric Cartesian state vector in the ecliptic J2000 frame.
///
/// Packs position $\mathbf{r}$ (AU) and velocity $\mathbf{v}$ (AU/day)
/// into a single 6-vector $(\mathbf{r}, \mathbf{v})$.
pub struct CartesianState {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
}

/// Convert an attributable state to a heliocentric Cartesian state.
///
/// The attributable state vector is
/// $\mathbf{x}_{attr} = (\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$,
/// where $(\alpha, \delta)$ are the equatorial J2000 angular coordinates (rad),
/// $(\dot{\alpha}, \dot{\delta})$ the angular rates (rad/day), $\rho$ the
/// topocentric range (AU) and $\dot{\rho}$ the range rate (AU/day).
///
/// The observer heliocentric position $\mathbf{r}_{obs}$ and velocity
/// $\mathbf{v}_{obs}$ must be expressed in the **ecliptic J2000** frame, which
/// is the working frame of Fink-FAT.
///
/// Conversion
/// ----------
///
/// The topocentric position of the object is
///
/// $$\mathbf{r} = \mathbf{r}_{obs} + \rho\,\hat{\rho}$$
///
/// where $\hat{\rho}$ is the unit line-of-sight vector in the ecliptic J2000
/// frame (see [`unit_los`]).
///
/// The topocentric velocity is
///
/// $$\mathbf{v} = \mathbf{v}_{obs} + \dot{\rho}\,\hat{\rho} + \rho\,\dot{\hat{\rho}}$$
///
/// where $\dot{\hat{\rho}}$ is the time derivative of the unit line-of-sight
/// (see [`unit_los_dot`]).
///
/// Arguments
/// ---------
/// * `attr` – Attributable state $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$.
/// * `r_obs` – Observer heliocentric position (AU), ecliptic J2000.
/// * `v_obs` – Observer heliocentric velocity (AU/day), ecliptic J2000.
///
/// Return
/// ------
/// Heliocentric Cartesian state $(\mathbf{r}, \mathbf{v})$ in the ecliptic J2000 frame.
pub fn attributable_to_cartesian(
    attr: &Vector6<f64>,
    r_obs: &Vector3<f64>,
    v_obs: &Vector3<f64>,
) -> CartesianState {
    let (ra, dec, ra_dot, dec_dot, rho, rho_dot) =
        (attr[0], attr[1], attr[2], attr[3], attr[4], attr[5]);

    let los = unit_los(ra, dec);
    let los_dot = unit_los_dot(ra, dec, ra_dot, dec_dot);

    let pos = r_obs + rho * los;
    let vel = v_obs + rho_dot * los + rho * los_dot;

    CartesianState { pos, vel }
}

/// Jacobian of the attributable-to-Cartesian conversion.
///
/// Computes $J = \partial(\mathbf{r}, \mathbf{v}) / \partial(\alpha, \delta,
/// \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$, a $6 \times 6$ matrix.
///
/// Block structure
/// ---------------
///
/// $$J = \begin{pmatrix}
/// \rho\,\partial_\alpha\hat{\rho} & \rho\,\partial_\delta\hat{\rho} & 0 & 0 & \hat{\rho} & 0 \\
/// \rho\,\partial_\alpha\dot{\hat{\rho}} + \dot{\rho}\,\partial_\alpha\hat{\rho}
/// & \rho\,\partial_\delta\dot{\hat{\rho}} + \dot{\rho}\,\partial_\delta\hat{\rho}
/// & \rho\,\partial_{\dot{\alpha}}\dot{\hat{\rho}}
/// & \rho\,\partial_{\dot{\delta}}\dot{\hat{\rho}}
/// & \dot{\hat{\rho}} & \hat{\rho}
/// \end{pmatrix}$$
///
/// where the partial derivatives of $\hat{\rho}$ are
///
/// $$\frac{\partial\hat{\rho}}{\partial\alpha} = \begin{pmatrix} -\cos\delta\sin\alpha \\ \cos\delta\cos\alpha \\ 0 \end{pmatrix}, \quad \frac{\partial\hat{\rho}}{\partial\delta} = \begin{pmatrix} -\sin\delta\cos\alpha \\ -\sin\delta\sin\alpha \\ \cos\delta \end{pmatrix}$$
///
/// and the partial derivatives of $\dot{\hat{\rho}}$ with respect to angular
/// rates are
///
/// $$\frac{\partial\dot{\hat{\rho}}}{\partial\dot{\alpha}} = \frac{\partial\hat{\rho}}{\partial\alpha}, \quad \frac{\partial\dot{\hat{\rho}}}{\partial\dot{\delta}} = \frac{\partial\hat{\rho}}{\partial\delta}$$
///
/// Arguments
/// ---------
/// * `attr` – Attributable state $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$.
///
/// Return
/// ------
/// $6 \times 6$ Jacobian matrix $J_{attr \to cart}$.
pub fn jacobian_attr_to_cart(attr: &Vector6<f64>) -> Matrix6<f64> {
    let (ra, dec, ra_dot, dec_dot, rho, rho_dot) =
        (attr[0], attr[1], attr[2], attr[3], attr[4], attr[5]);

    let cos_dec = dec.cos();
    let sin_dec = dec.sin();
    let cos_ra = ra.cos();
    let sin_ra = ra.sin();

    // ── Partial derivatives of los = rho_hat ─────────────────────────────
    let d_los_d_ra = Vector3::new(-cos_dec * sin_ra, cos_dec * cos_ra, 0.0);
    let d_los_d_dec = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);

    // ── los_dot partials ──────────────────────────────────────────────────
    // los_dot = ra_dot * d_los/d_ra + dec_dot * d_los/d_dec
    // + second-order terms in ra/dec (cross derivatives)
    let d2_los_d_ra2 = Vector3::new(-cos_dec * cos_ra, -cos_dec * sin_ra, 0.0);
    let d2_los_d_dec2 = Vector3::new(-cos_dec * cos_ra, -cos_dec * sin_ra, -sin_dec);
    let d2_los_d_ra_d_dec = Vector3::new(sin_dec * sin_ra, -sin_dec * cos_ra, 0.0);

    let d_losdot_d_ra = ra_dot * d2_los_d_ra2 + dec_dot * d2_los_d_ra_d_dec;
    let d_losdot_d_dec = ra_dot * d2_los_d_ra_d_dec + dec_dot * d2_los_d_dec2;

    // d(los_dot)/d(ra_dot) = d_los/d_ra
    // d(los_dot)/d(dec_dot) = d_los/d_dec

    let los = unit_los(ra, dec);
    let los_dot = unit_los_dot(ra, dec, ra_dot, dec_dot);

    let mut j = Matrix6::zeros();

    // ── Position rows (0..3) ──────────────────────────────────────────────
    // dr/d_ra   = rho * d_los/d_ra
    // dr/d_dec  = rho * d_los/d_dec
    // dr/d_rho  = los
    // dr/d_* = 0 for ra_dot, dec_dot, rho_dot
    for i in 0..3 {
        j[(i, 0)] = rho * d_los_d_ra[i];
        j[(i, 1)] = rho * d_los_d_dec[i];
        // j[(i, 2)] = 0
        // j[(i, 3)] = 0
        j[(i, 4)] = los[i];
        // j[(i, 5)] = 0
    }

    // ── Velocity rows (3..6) ─────────────────────────────────────────────
    // dv/d_ra      = rho * d_losdot/d_ra  + rho_dot * d_los/d_ra
    // dv/d_dec     = rho * d_losdot/d_dec + rho_dot * d_los/d_dec
    // dv/d_ra_dot  = rho * d_los/d_ra
    // dv/d_dec_dot = rho * d_los/d_dec
    // dv/d_rho     = los_dot
    // dv/d_rho_dot = los
    for i in 0..3 {
        j[(3 + i, 0)] = rho * d_losdot_d_ra[i] + rho_dot * d_los_d_ra[i];
        j[(3 + i, 1)] = rho * d_losdot_d_dec[i] + rho_dot * d_los_d_dec[i];
        j[(3 + i, 2)] = rho * d_los_d_ra[i];
        j[(3 + i, 3)] = rho * d_los_d_dec[i];
        j[(3 + i, 4)] = los_dot[i];
        j[(3 + i, 5)] = los[i];
    }

    j
}

/// Convert a heliocentric Cartesian state to an attributable state.
///
/// Inverts [`attributable_to_cartesian`]: given $(\mathbf{r}, \mathbf{v})$
/// and the observer state, recovers
/// $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$.
///
/// Conversion
/// ----------
///
/// The topocentric displacement vector is
///
/// $$\boldsymbol{\Delta} = \mathbf{r} - \mathbf{r}_{obs}$$
///
/// The topocentric range and unit line-of-sight are
///
/// $$\rho = |\boldsymbol{\Delta}|, \quad \hat{\rho} = \boldsymbol{\Delta} / \rho$$
///
/// The equatorial J2000 angles are recovered by rotating $\hat{\rho}$ from
/// ecliptic J2000 to equatorial J2000 and reading off $(\alpha, \delta)$.
///
/// The topocentric velocity is
///
/// $$\boldsymbol{\Delta v} = \mathbf{v} - \mathbf{v}_{obs}$$
///
/// Decomposing along the line of sight and its transverse complement:
///
/// $$\dot{\rho} = \boldsymbol{\Delta v} \cdot \hat{\rho}$$
///
/// $$\rho\,\dot{\hat{\rho}} = \boldsymbol{\Delta v} - \dot{\rho}\,\hat{\rho}$$
///
/// The angular rates $(\dot{\alpha}, \dot{\delta})$ are then obtained by
/// projecting $\dot{\hat{\rho}}$ onto the local sky tangent basis
/// $(\mathbf{e}_\alpha, \mathbf{e}_\delta)$:
///
/// $$\dot{\alpha} = \frac{\dot{\hat{\rho}} \cdot \mathbf{e}_\alpha}{\cos\delta}, \quad \dot{\delta} = \dot{\hat{\rho}} \cdot \mathbf{e}_\delta$$
///
/// where
///
/// $$\mathbf{e}_\alpha = (-\sin\alpha,\,\cos\alpha,\,0)^\top, \quad \mathbf{e}_\delta = (-\sin\delta\cos\alpha,\,-\sin\delta\sin\alpha,\,\cos\delta)^\top$$
///
/// expressed in the **equatorial J2000** frame.
///
/// Arguments
/// ---------
/// * `cart` – Heliocentric Cartesian state $(\mathbf{r}, \mathbf{v})$, ecliptic J2000.
/// * `r_obs` – Observer heliocentric position (AU), ecliptic J2000.
/// * `v_obs` – Observer heliocentric velocity (AU/day), ecliptic J2000.
///
/// Return
/// ------
/// Attributable state $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$.
pub fn cartesian_to_attributable(
    cart: &CartesianState,
    r_obs: &Vector3<f64>,
    v_obs: &Vector3<f64>,
) -> Vector6<f64> {
    // ── Topocentric displacement ──────────────────────────────────────────
    let delta = cart.pos - r_obs;
    let rho = delta.norm();
    let rho_hat_ecl = delta / rho;

    // ── Rotate to equatorial J2000 to read off angles ─────────────────────
    let rho_hat_eq = ROT_ECLMJ2000_TO_EQUMJ2000 * rho_hat_ecl;

    let dec = rho_hat_eq[2].clamp(-1.0, 1.0).asin();
    let ra = rho_hat_eq[1]
        .atan2(rho_hat_eq[0])
        .rem_euclid(std::f64::consts::TAU);

    // ── Topocentric velocity ──────────────────────────────────────────────
    let delta_v = cart.vel - v_obs;
    let rho_dot = delta_v.dot(&rho_hat_ecl);

    // rho * rho_hat_dot in ecliptic, then rotate to equatorial
    let rho_hat_dot_ecl = (delta_v - rho_dot * rho_hat_ecl) / rho;
    let rho_hat_dot_eq = ROT_ECLMJ2000_TO_EQUMJ2000 * rho_hat_dot_ecl;

    // ── Project onto sky tangent basis ───────────────────────────────────
    let cos_dec = dec.cos();
    let sin_dec = dec.sin();
    let cos_ra = ra.cos();
    let sin_ra = ra.sin();

    // e_alpha = d(rho_hat)/d_alpha / cos(dec)  [equatorial]
    let e_alpha = Vector3::new(-sin_ra, cos_ra, 0.0);
    // e_delta = d(rho_hat)/d_delta             [equatorial]
    let e_delta = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);

    let ra_dot = rho_hat_dot_eq.dot(&e_alpha) / cos_dec;
    let dec_dot = rho_hat_dot_eq.dot(&e_delta);

    Vector6::new(ra, dec, ra_dot, dec_dot, rho, rho_dot)
}

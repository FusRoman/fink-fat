use photom::{coordinates::equatorial::EquCoord, observation_dataset::observation::Observation};

/// Compute the total angular speed $\omega$ and its propagated uncertainty
/// $\sigma_\omega$.
///
/// $$\omega = \sqrt{(\dot{\alpha}\cos\delta)^2 + \dot{\delta}^2}$$
///
/// Uncertainty propagation (first-order):
///
/// $$\sigma_\omega = \frac{\sqrt{(\dot{\alpha}\cos\delta\,\sigma_{\dot{\alpha}})^2 + (\dot{\delta}\,\sigma_{\dot{\delta}})^2}}{\omega}$$
///
/// Arguments
/// ---------
/// * `mid_point` – Midpoint coordinates providing $\delta$ (radians).
/// * `mid_speed` – Angular velocity $(\dot{\alpha}, \dot{\delta})$ with errors,
///   in rad/day.
///
/// Return
/// ------
/// `(omega, sigma_omega)` in rad/day.
pub(crate) fn omega_and_sigma(mid_point: &EquCoord, mid_speed: &MidSpeed) -> (f64, f64) {
    let cos_dec = mid_point.dec.cos();
    let alpha_dot = mid_speed.0.ra;
    let delta_dot = mid_speed.0.dec;
    let sigma_alpha_dot = mid_speed.0.ra_error;
    let sigma_delta_dot = mid_speed.0.dec_error;

    let alpha_proj = alpha_dot * cos_dec;
    let omega = alpha_proj.hypot(delta_dot);

    let sigma_omega =
        ((alpha_proj * sigma_alpha_dot).powi(2) + (delta_dot * sigma_delta_dot).powi(2)).sqrt()
            / omega;

    (omega, sigma_omega)
}

/// Midpoint angular coordinates between two observations.
///
/// Computes the arithmetic mean of $(\alpha, \delta)$ and propagates
/// measurement errors in quadrature:
///
/// $$\bar{\alpha} = \frac{\alpha_1 + \alpha_2}{2}, \quad
///   \sigma_{\bar{\alpha}} = \frac{1}{2}\sqrt{\sigma_{\alpha_1}^2 + \sigma_{\alpha_2}^2}$$
///
/// and identically for $\delta$.
///
/// Arguments
/// ---------
/// * `obs1` – First observation.
/// * `obs2` – Second observation.
///
/// Return
/// ------
/// Midpoint [`EquCoord`] with propagated errors, in radians.
pub fn midpoint(obs1: &Observation, obs2: &Observation) -> EquCoord {
    let c1 = obs1.equ_coord();
    let c2 = obs2.equ_coord();

    let mid_ra = (c1.ra + c2.ra) * 0.5;
    let mid_dec = (c1.dec + c2.dec) * 0.5;
    let ra_err = c1.ra_error.hypot(c2.ra_error) * 0.5;
    let dec_err = c1.dec_error.hypot(c2.dec_error) * 0.5;

    EquCoord::new(mid_ra, ra_err, mid_dec, dec_err)
}

#[derive(Debug)]
pub struct MidSpeed(pub EquCoord);

/// Angular velocity between two observations.
///
/// Estimates the mean angular rate $(\dot{\alpha}, \dot{\delta})$ over the
/// interval $\Delta t = t_2 - t_1$:
///
/// $$\dot{\alpha} = \frac{\alpha_2 - \alpha_1}{\Delta t}, \quad
///   \sigma_{\dot{\alpha}} = \frac{\sqrt{\sigma_{\alpha_1}^2 + \sigma_{\alpha_2}^2}}{\Delta t}$$
///
/// and identically for $\dot{\delta}$.
///
/// Arguments
/// ---------
/// * `obs1` – First observation (earlier epoch).
/// * `obs2` – Second observation (later epoch).
///
/// Return
/// ------
/// [`MidSpeed`] wrapping an [`EquCoord`] where the `ra`/`dec` fields
/// hold $(\dot{\alpha}, \dot{\delta})$ in rad/day and the error fields
/// hold their propagated uncertainties.
pub fn mid_speed(obs1: &Observation, obs2: &Observation) -> MidSpeed {
    let c1 = obs1.equ_coord();
    let c2 = obs2.equ_coord();
    let inv_dt = 1. / (obs2.mjd_tt() - obs1.mjd_tt());

    let mid_ra_speed = (c2.ra - c1.ra) * inv_dt;
    let mid_dec_speed = (c2.dec - c1.dec) * inv_dt;
    let ra_err = c1.ra_error.hypot(c2.ra_error) * inv_dt;
    let dec_err = c1.dec_error.hypot(c2.dec_error) * inv_dt;

    MidSpeed(EquCoord::new(mid_ra_speed, ra_err, mid_dec_speed, dec_err))
}

/// Total angular speed $\omega$ on the sky, corrected for declination.
///
/// $$\omega = \sqrt{(\dot{\alpha}\cos\delta)^2 + \dot{\delta}^2}$$
///
/// The $\cos\delta$ factor projects the right-ascension rate onto the
/// great-circle direction, giving a proper angular velocity in rad/day.
///
/// Arguments
/// ---------
/// * `mid_point` – Midpoint coordinates providing $\delta$.
/// * `mid_speed` – Angular rates $(\dot{\alpha}, \dot{\delta})$ in rad/day.
///
/// Return
/// ------
/// Total angular speed $\omega$ in rad/day.
pub fn angular_speed(mid_point: &EquCoord, mid_speed: &MidSpeed) -> f64 {
    omega_and_sigma(mid_point, mid_speed).0
}

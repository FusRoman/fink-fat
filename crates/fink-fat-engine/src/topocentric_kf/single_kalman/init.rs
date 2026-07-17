use nalgebra::{Matrix6, Vector3, Vector6};
use outfit::{
    OutfitError,
    constants::{GAUSS_GRAV_SQUARED, ROT_EQUMJ2000_TO_ECLMJ2000},
};
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, observation::Observation},
};

use crate::{
    engine_config::kalman_context::KalmanContext,
    error::{EngineError, TopocentricRangeError},
    topocentric_kf::{
        observer_state::{EphemState, HelioObsState, get_observer},
        pairs_state::{MidSpeed, mid_speed, midpoint, omega_and_sigma},
        single_kalman::KFState,
    },
};

/// Unit line-of-sight vector $\hat{\rho}$ in the heliocentric ecliptic mean J2000 frame.
///
/// The input angles $(\alpha, \delta)$ are in the equatorial mean J2000 frame
/// (as provided by ZTF/Rubin alerts). The unit vector is first expressed in
/// equatorial J2000:
///
/// $$\hat{\rho}_{eq} = \begin{pmatrix} \cos\delta\cos\alpha \\ \cos\delta\sin\alpha \\ \sin\delta \end{pmatrix}$$
///
/// then rotated into the ecliptic mean J2000 frame:
///
/// $$\hat{\rho}_{ecl} = R_{eq \to ecl}\,\hat{\rho}_{eq}$$
///
/// where $R_{eq \to ecl}$ is [`outfit::constants::ROT_EQUMJ2000_TO_ECLMJ2000`].
pub fn unit_los(ra: f64, dec: f64) -> Vector3<f64> {
    let cos_dec = dec.cos();
    let rho_eq = Vector3::new(cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin());
    ROT_EQUMJ2000_TO_ECLMJ2000 * rho_eq
}

/// Time derivative of the unit line-of-sight vector $\dot{\hat{\rho}}$ in the heliocentric ecliptic mean J2000 frame.
///
/// Given angular rates $(\dot{\alpha}, \dot{\delta})$ in the equatorial mean J2000 frame,
/// the derivative is first computed in equatorial J2000:
///
/// $$\dot{\hat{\rho}}_{eq} = \dot{\alpha}\frac{\partial\hat{\rho}}{\partial\alpha} + \dot{\delta}\frac{\partial\hat{\rho}}{\partial\delta}$$
///
/// where:
///
/// $$\frac{\partial\hat{\rho}}{\partial\alpha} = \begin{pmatrix} -\cos\delta\sin\alpha \\ \cos\delta\cos\alpha \\ 0 \end{pmatrix}, \quad \frac{\partial\hat{\rho}}{\partial\delta} = \begin{pmatrix} -\sin\delta\cos\alpha \\ -\sin\delta\sin\alpha \\ \cos\delta \end{pmatrix}$$
///
/// then rotated into the ecliptic mean J2000 frame:
///
/// $$\dot{\hat{\rho}}_{ecl} = R_{eq \to ecl}\,\dot{\hat{\rho}}_{eq}$$
///
/// where $R_{eq \to ecl}$ is [`outfit::constants::ROT_EQUMJ2000_TO_ECLMJ2000`].
pub fn unit_los_dot(ra: f64, dec: f64, ra_dot: f64, dec_dot: f64) -> Vector3<f64> {
    let cos_dec = dec.cos();
    let sin_dec = dec.sin();
    let cos_ra = ra.cos();
    let sin_ra = ra.sin();

    let d_rho_d_alpha = Vector3::new(-cos_dec * sin_ra, cos_dec * cos_ra, 0.0);
    let d_rho_d_delta = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);

    let rho_dot_eq = ra_dot * d_rho_d_alpha + dec_dot * d_rho_d_delta;
    ROT_EQUMJ2000_TO_ECLMJ2000 * rho_dot_eq
}

/// Reference epoch of an observation pair: the midpoint in time (MJD TT)
/// between the two tracklet observations.
///
/// This single formula anchors every [`KFState`] built from the pair —
/// both [`init_kf_state`]'s single best-guess range and
/// [`crate::topocentric_kf::seed_grid::admissible_region_grid`]'s whole
/// tiling of the admissible region use the same midpoint epoch.
pub(crate) fn pair_midpoint_epoch(first_obs: &Observation, second_obs: &Observation) -> f64 {
    let dt_pair = second_obs.mjd_tt() - first_obs.mjd_tt();
    first_obs.mjd_tt() + dt_pair / 2.0
}

/// Sky-plane geometry of a tracklet (a pair of observations), shared by
/// every initialization strategy built on a single pair.
///
/// Bundles the midpoint angular position, the finite-difference angular
/// rates, and the line-of-sight unit vector together with its time
/// derivative: the four quantities that are fully determined by the
/// tracklet alone, independent of any assumption on the (unobservable)
/// range $\rho$ or range-rate $\dot{\rho}$.
pub(crate) struct TrackletGeometry {
    /// Midpoint sky position $(\alpha, \delta)$ with astrometric errors (rad).
    pub mid_point: EquCoord,
    /// Finite-difference angular velocity $(\dot{\alpha}, \dot{\delta})$ with
    /// propagated errors (rad/day).
    pub mid_speed: MidSpeed,
    /// Unit line-of-sight vector $\hat{\rho}$, ecliptic mean J2000 frame.
    pub los: Vector3<f64>,
    /// Time derivative of the line-of-sight $\dot{\hat{\rho}}$ (rad/day),
    /// ecliptic mean J2000 frame.
    pub los_dot: Vector3<f64>,
}

/// Compute the [`TrackletGeometry`] of an observation pair.
///
/// This is the single entry point used by both [`init_kf_state`] (single
/// best-guess range) and
/// [`crate::topocentric_kf::seed_grid::admissible_region_grid`] (full tiling
/// of the admissible $(\rho, \dot{\rho})$ region) to derive a tracklet's
/// angular geometry. Sharing this function guarantees both initialization
/// strategies start from exactly the same midpoint angles, angular rates,
/// and line-of-sight vectors — there is only one place where this geometry
/// is computed.
pub(crate) fn tracklet_geometry(
    first_obs: &Observation,
    second_obs: &Observation,
) -> TrackletGeometry {
    let mid_point = midpoint(first_obs, second_obs);
    let mid_speed = mid_speed(first_obs, second_obs);
    let los = unit_los(mid_point.ra, mid_point.dec);
    let los_dot = unit_los_dot(mid_point.ra, mid_point.dec, mid_speed.0.ra, mid_speed.0.dec);

    TrackletGeometry {
        mid_point,
        mid_speed,
        los,
        los_dot,
    }
}

/// Topocentric range $\rho$ from a heliocentric distance prior $r = 1/\gamma$.
///
/// Solves the Al-Kashi (law of cosines) equation:
///
/// $$\rho^2 + 2\rho\, r_{obs}\cos\phi + (r_{obs}^2 - r^2) = 0$$
///
/// where $\cos\phi = -\hat{\rho} \cdot \hat{r}_{obs}$ is related to the solar
/// elongation.
///
/// Returns the positive root, or `None` if no positive solution exists.
pub fn topocentric_range(
    los: &Vector3<f64>,
    r_obs: &Vector3<f64>,
    gamma: f64,
) -> Result<f64, TopocentricRangeError> {
    // --- Heliocentric distance prior ---
    let r = 1.0 / gamma;

    // --- Observer distance from Sun ---
    let r_obs_norm = r_obs.norm();

    // --- Solar elongation cosine ---
    let cos_phi = -los.dot(r_obs) / r_obs_norm;
    let phi_deg = cos_phi.acos().to_degrees();

    // --- Al-Kashi quadratic coefficients ---
    let a = 1.0_f64;
    let b = 2.0 * r_obs_norm * cos_phi;
    let c = r_obs_norm * r_obs_norm - r * r;

    // --- Discriminant ---
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return Err(TopocentricRangeError::NegativeDiscriminant {
            discriminant,
            r,
            r_obs_norm,
            phi_deg,
        });
    }

    // --- Roots ---
    let sqrt_disc = discriminant.sqrt();
    let rho1 = (-b + sqrt_disc) / 2.0;
    let rho2 = (-b - sqrt_disc) / 2.0;

    // --- Root selection ---
    match (rho1 > 0.0, rho2 > 0.0) {
        (true, true) => {
            let chosen = rho1.min(rho2);
            Ok(chosen)
        }
        (true, false) => Ok(rho1),
        (false, true) => Ok(rho2),
        (false, false) => Err(TopocentricRangeError::NoPositiveRoot { rho1, rho2, r }),
    }
}

/// Estimate the inverse heliocentric distance $\gamma = 1/r$ from the
/// observed angular speed $\omega$, assuming a circular Keplerian orbit.
///
/// For a circular orbit at heliocentric distance $r$, the transverse velocity
/// equals the circular velocity $v_{circ} = \sqrt{k^2/r}$, and the angular
/// speed satisfies:
///
/// $$\omega = \frac{v_{circ}}{r} = \sqrt{k^2}\,\gamma^{3/2}$$
///
/// Inverting:
///
/// $$\gamma = \left(\frac{\omega}{\sqrt{k^2}}\right)^{2/3}$$
///
/// where $k^2$ is [`GAUSS_GRAV_SQUARED`] (AU³/day²).
///
/// This estimate is exact for circular orbits. For eccentric orbits,
/// $v_\perp$ deviates from $v_{circ}$, introducing a bias that is
/// absorbed by [`estimate_sigma_gamma`].
///
/// Arguments
/// ---------
/// * `mid_point` – Midpoint coordinates providing $\delta$ (radians).
/// * `mid_speed` – Angular velocity $(\dot{\alpha}, \dot{\delta})$ in rad/day.
///
/// Return
/// ------
/// Estimated $\gamma$ in AU$^{-1}$.
pub fn estimate_gamma(mid_point: &EquCoord, mid_speed: &MidSpeed) -> f64 {
    let (omega, _) = omega_and_sigma(mid_point, mid_speed);
    (omega / GAUSS_GRAV_SQUARED.sqrt()).powf(2.0 / 3.0)
}

/// Estimate the uncertainty on the inverse-distance prior $\gamma = 1/r$.
///
/// Two independent contributions are combined in quadrature:
///
/// - **Observational**: propagated from angular rate measurement errors.
///
///   $$\sigma_\gamma^{(obs)} = \gamma \cdot \frac{\sigma_\omega}{\omega}$$
///
///   where $\omega = \sqrt{\dot{\alpha}^2\cos^2\delta + \dot{\delta}^2}$ and
///   $\sigma_\omega$ is its propagated uncertainty.
///
/// - **Prior**: reflects ignorance of the true heliocentric distance within
///   a physically plausible range $[r_{min}, r_{max}]$:
///
///   $$\sigma_\gamma^{(prior)} = \frac{1}{r_{min}} - \frac{1}{r_{max}}$$
///
/// Arguments
/// ---------
/// * `gamma` – Estimated inverse heliocentric distance (AU$^{-1}$).
/// * `mid_point` – Midpoint angular coordinates $(\alpha, \delta)$ in radians.
/// * `mid_speed` – Angular velocity with errors in rad/day.
/// * `r_min` – Minimum plausible heliocentric distance (AU).
/// * `r_max` – Maximum plausible heliocentric distance (AU).
///
/// Return
/// ------
/// * `sigma_gamma` in AU$^{-1}$.
pub fn estimate_sigma_gamma(
    gamma: f64,
    mid_point: &EquCoord,
    mid_speed: &MidSpeed,
    r_min: f64,
    r_max: f64,
) -> f64 {
    let (omega, sigma_omega) = omega_and_sigma(mid_point, mid_speed);

    let sigma_obs = gamma * sigma_omega / omega;
    let sigma_prior = 1.0 / r_min - 1.0 / r_max;

    (sigma_obs.powi(2) + sigma_prior.powi(2)).sqrt()
}

/// Estimate the initial Kalman filter covariance scale factors.
///
/// Derives position and velocity uncertainties from the observational data
/// and the inverse-distance prior $\gamma$.
///
/// Derivation
/// ----------
/// **`sigma_pos`** — propagated from distance uncertainty via $r = 1/\gamma$:
///
/// $$\sigma_r = \frac{\sigma_\gamma}{\gamma^2}$$
///
/// **`sigma_trans`** — propagated from angular rate errors at range $\rho$:
///
/// $$\sigma_{v_\perp} = \rho\sqrt{\sigma_{\dot{\alpha}}^2\cos^2\delta + \sigma_{\dot{\delta}}^2}$$
///
/// **`sigma_rad`** — bounded by the Keplerian circular velocity at $r = 1/\gamma$:
///
/// $$\sigma_{\dot{r}} = \sqrt{\frac{k^2}{r}}$$
///
/// where $k^2$ is [`GAUSS_GRAV_SQUARED`] (AU³/day²). The radial velocity
/// is unobserved at initialization; this bound reflects the maximum
/// physically plausible value for a gravitationally bound orbit.
///
/// Arguments
/// ---------
/// * `rho` – Topocentric range (AU).
/// * `sigma_gamma` – Uncertainty on the inverse-distance prior (AU$^{-1}$).
/// * `gamma` – Inverse heliocentric distance prior $\gamma = 1/r$ (AU$^{-1}$).
/// * `mid_speed` – Angular velocity with propagated errors (rad/day).
/// * `mid_point` – Midpoint coordinates providing $\delta$ (radians).
///
/// Return
/// ------
/// `(sigma_pos, sigma_trans, sigma_rad)` in AU and AU/day respectively.
pub fn estimate_init_sigmas(
    rho: f64,
    gamma: f64,
    mid_speed: &MidSpeed,
    mid_point: &EquCoord,
) -> (f64, f64) {
    // --- Transverse velocity uncertainty ---
    // sigma_trans = rho * sqrt((sigma_ra * cos(dec))^2 + sigma_dec^2)
    let cos_dec = mid_point.dec.cos();
    let ra_term = mid_speed.0.ra_error * cos_dec;
    let dec_term = mid_speed.0.dec_error;
    let angular_speed_err = (ra_term.powi(2) + dec_term.powi(2)).sqrt();
    let sigma_trans = rho * angular_speed_err;

    // --- Radial velocity uncertainty (Keplerian bound) ---
    // sigma_rad = v_circ = sqrt(k^2 / r)
    let r = 1.0 / gamma;
    let v_circ_sq = GAUSS_GRAV_SQUARED / r;
    let sigma_rad = v_circ_sq.sqrt();

    (sigma_trans, sigma_rad)
}

/// Estimate plausible heliocentric distance bounds $[r_{min}, r_{max}]$ from
/// the observed angular speed and its uncertainty.
///
/// Uses the circular Keplerian relation $r = (k^2 / \omega^2)^{1/3}$ to
/// propagate the $\pm n\sigma$ interval on $\omega$ into a distance range.
///
/// $$r_{min} = \left(\frac{k^2}{(\omega + n\,\sigma_\omega)^2}\right)^{1/3}, \quad
///   r_{max} = \left(\frac{k^2}{\max(\omega - n\,\sigma_\omega,\,\epsilon)^2}\right)^{1/3}$$
///
/// Since $r$ is a decreasing function of $\omega$, a higher angular speed
/// implies a closer object.
///
/// The lower bound on $\omega$ is clamped to a small positive value
/// $\epsilon$ to avoid divergence when $\sigma_\omega \geq \omega / n$.
///
/// Arguments
/// ---------
/// * `mid_point` – Midpoint coordinates providing $\delta$ (radians).
/// * `mid_speed` – Angular velocity $(\dot{\alpha}, \dot{\delta})$ with errors,
///   in rad/day.
/// * `n_sigma` – Half-width of the interval in units of $\sigma_\omega$.
///
/// Return
/// ------
/// `(r_min, r_max)` in AU.
pub fn estimate_r_bounds(mid_point: &EquCoord, mid_speed: &MidSpeed, n_sigma: f64) -> (f64, f64) {
    let (omega, sigma_omega) = omega_and_sigma(mid_point, mid_speed);

    let omega_hi = omega + n_sigma * sigma_omega;
    let omega_lo = (omega - n_sigma * sigma_omega).max(f64::EPSILON);

    let r_from_omega = |w: f64| -> f64 { (GAUSS_GRAV_SQUARED / w.powi(2)).powf(1.0 / 3.0) };

    (r_from_omega(omega_hi), r_from_omega(omega_lo))
}

/// Estimate the radial velocity $\dot{\rho}$ under a circular-orbit assumption.
///
/// When the orbit is circular, the heliocentric speed equals the circular
/// velocity:
///
/// $$\|\mathbf{v}\|^2 = v_{\text{circ}}^2 = \frac{k^2}{r}$$
///
/// The heliocentric velocity decomposes as:
///
/// $$\mathbf{v} = \mathbf{v}_{\text{obs}} + \dot{\rho}\,\hat{\mathbf{u}} + \rho\,\dot{\hat{\mathbf{u}}}$$
///
/// Defining the transverse term (the part independent of $\dot{\rho}$):
///
/// $$\mathbf{v}_T = \mathbf{v}_{\text{obs}} + \rho\,\dot{\hat{\mathbf{u}}}$$
///
/// the speed constraint becomes:
///
/// $$\|\mathbf{v}_T + \dot{\rho}\,\hat{\mathbf{u}}\|^2 = v_{\text{circ}}^2$$
///
/// Expanding (and using $\|\hat{\mathbf{u}}\| = 1$):
///
/// $$\dot{\rho}^2 + 2\,(\mathbf{v}_T \cdot \hat{\mathbf{u}})\,\dot{\rho} + \|\mathbf{v}_T\|^2 - v_{\text{circ}}^2 = 0$$
///
/// This is a quadratic in $\dot{\rho}$. When two real roots exist, the one
/// with the smaller absolute value is returned (minimum-energy selection).
///
/// Arguments
/// ---------
/// * `los` – Unit line-of-sight vector $\hat{\mathbf{u}}$ (dimensionless).
/// * `los_dot` – Time derivative of the line-of-sight $\dot{\hat{\mathbf{u}}}$ (rad/day).
/// * `v_obs` – Observer heliocentric velocity (AU/day).
/// * `rho` – Topocentric range $\rho$ (AU).
/// * `r` – Heliocentric distance $r$ (AU).
///
/// Return
/// ------
/// * `Some(rho_dot)` – Estimated $\dot{\rho}$ in AU/day (smallest $|\dot{\rho}|$ root).
/// * `None` – If the discriminant is negative (no real solution: the circular
///   assumption is inconsistent with the geometry).
pub fn estimate_rho_dot_circular(
    los: &Vector3<f64>,
    los_dot: &Vector3<f64>,
    v_obs: &Vector3<f64>,
    rho: f64,
    r: f64,
) -> Option<f64> {
    // --- Transverse velocity ---
    let v_transverse = v_obs + rho * los_dot;

    // --- Circular velocity ---
    let v_circ_sq = GAUSS_GRAV_SQUARED / r;

    // --- Quadratic coefficients ---
    let a = 1.0_f64;
    let b = 2.0 * v_transverse.dot(los);
    let c = v_transverse.norm_squared() - v_circ_sq;

    // --- Discriminant ---
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }

    // --- Roots ---
    let sqrt_disc = disc.sqrt();
    let rho_dot1 = (-b + sqrt_disc) / 2.0;
    let rho_dot2 = (-b - sqrt_disc) / 2.0;

    // --- Root selection ---
    let result = if rho_dot1.abs() <= rho_dot2.abs() {
        rho_dot1
    } else {
        rho_dot2
    };

    Some(result)
}

use crate::logging::LogTarget;

/// Structured log events for initial-orbit-determination bootstrap
/// (tracklet geometry → gamma/rho/rho_dot → initial covariance). See
/// [`crate::logging`] for the `.emit()` pattern.
pub enum InitEvent {
    TrackletGeometry {
        mid_ra_deg: f64,
        mid_dec_deg: f64,
        mid_speed_ra: f64,
        mid_speed_dec: f64,
        sigma_ra_rad: f64,
        sigma_dec_rad: f64,
    },
    ObserverState {
        r_obs: [f64; 3],
        v_obs: [f64; 3],
    },
    LineOfSight {
        los: [f64; 3],
        los_dot: [f64; 3],
        n_sigma: f64,
    },
    GammaEstimate {
        gamma: f64,
        r_helio_au: f64,
    },
    TopocentricRange {
        rho_au: f64,
    },
    DistanceBounds {
        r_min_au: f64,
        r_max_au: f64,
    },
    GammaUncertainty {
        sigma_gamma: f64,
        sigma_gamma_rel_pct: f64,
    },
    InitSigmas {
        sigma_trans_au: f64,
        sigma_rad_au: f64,
    },
    RhoDotFallback {
        rho_au: f64,
        r_helio_au: f64,
    },
    RhoDot {
        rho_dot_au_per_day: f64,
    },
    AngularVariances {
        sigma_ra_rad: f64,
        sigma_dec_rad: f64,
        var_ra: f64,
        var_dec: f64,
    },
    AngularRateVariances {
        sigma_ra_dot: f64,
        sigma_dec_dot: f64,
        var_ra_dot: f64,
        var_dec_dot: f64,
    },
    RangeVariance {
        r_obs_norm_au: f64,
        cos_phi: f64,
        d_rho_d_gamma: f64,
        sigma_rho_au: f64,
        var_rho_au2: f64,
    },
    RangeRateVariance {
        sigma_rho_dot_au_per_day: f64,
        var_rho_dot: f64,
    },
    InitialCovarianceAssembled {
        trace_p0: f64,
    },
    CombinedUncertainty {
        sigma_ang_rad: f64,
    },
    FinalCovarianceTraces {
        cov_pos_trace_au2: f64,
        cov_vel_trace: f64,
    },
    Complete,
}

crate::impl_log_target!(
    InitEvent,
    "init",
    "Initial orbit determination bootstrap from a tracklet (Gauss/topocentric-range)",
    [tracing::Level::TRACE, tracing::Level::WARN]
);

impl InitEvent {
    pub fn emit(&self) {
        use InitEvent::*;
        match self {
            TrackletGeometry {
                mid_ra_deg,
                mid_dec_deg,
                mid_speed_ra,
                mid_speed_dec,
                sigma_ra_rad,
                sigma_dec_rad,
            } => tracing::trace!(
                target: InitEvent::TARGET, mid_ra_deg, mid_dec_deg, mid_speed_ra, mid_speed_dec, sigma_ra_rad, sigma_dec_rad,
                "Tracklet midpoint geometry"
            ),
            ObserverState { r_obs, v_obs } => tracing::trace!(
                target: InitEvent::TARGET, ?r_obs, ?v_obs,
                "Observer heliocentric state, ecliptic J2000 (AU, AU/day)"
            ),
            LineOfSight {
                los,
                los_dot,
                n_sigma,
            } => tracing::trace!(
                target: InitEvent::TARGET, ?los, ?los_dot, n_sigma,
                "Unit line-of-sight and its time derivative, range bound n_sigma"
            ),
            GammaEstimate { gamma, r_helio_au } => tracing::trace!(
                target: InitEvent::TARGET, gamma, r_helio_au, "Estimated gamma and heliocentric distance"
            ),
            TopocentricRange { rho_au } => tracing::trace!(
                target: InitEvent::TARGET, rho_au, "Topocentric range"
            ),
            DistanceBounds { r_min_au, r_max_au } => tracing::trace!(
                target: InitEvent::TARGET, r_min_au, r_max_au, "Heliocentric distance bounds"
            ),
            GammaUncertainty {
                sigma_gamma,
                sigma_gamma_rel_pct,
            } => tracing::trace!(
                target: InitEvent::TARGET, sigma_gamma, sigma_gamma_rel_pct, "Gamma uncertainty"
            ),
            InitSigmas {
                sigma_trans_au,
                sigma_rad_au,
            } => tracing::trace!(
                target: InitEvent::TARGET, sigma_trans_au, sigma_rad_au, "Initial uncertainty decomposition (AU)"
            ),
            RhoDotFallback { rho_au, r_helio_au } => tracing::warn!(
                target: InitEvent::TARGET, rho_au, r_helio_au,
                "rho_dot estimation failed: circular-orbit geometry inconsistent. Falling back to rho_dot = 0."
            ),
            RhoDot { rho_dot_au_per_day } => tracing::trace!(
                target: InitEvent::TARGET, rho_dot_au_per_day, "Range rate rho_dot"
            ),
            AngularVariances {
                sigma_ra_rad,
                sigma_dec_rad,
                var_ra,
                var_dec,
            } => tracing::trace!(
                target: InitEvent::TARGET, sigma_ra_rad, sigma_dec_rad, var_ra, var_dec, "Angular position variances"
            ),
            AngularRateVariances {
                sigma_ra_dot,
                sigma_dec_dot,
                var_ra_dot,
                var_dec_dot,
            } => tracing::trace!(
                target: InitEvent::TARGET, sigma_ra_dot, sigma_dec_dot, var_ra_dot, var_dec_dot, "Angular rate variances"
            ),
            RangeVariance {
                r_obs_norm_au,
                cos_phi,
                d_rho_d_gamma,
                sigma_rho_au,
                var_rho_au2,
            } => tracing::trace!(
                target: InitEvent::TARGET, r_obs_norm_au, cos_phi, d_rho_d_gamma, sigma_rho_au, var_rho_au2,
                "Range variance (propagated from sigma_gamma via Al-Kashi)"
            ),
            RangeRateVariance {
                sigma_rho_dot_au_per_day,
                var_rho_dot,
            } => tracing::trace!(
                target: InitEvent::TARGET, sigma_rho_dot_au_per_day, var_rho_dot,
                "Range-rate variance (Keplerian circular velocity bound)"
            ),
            InitialCovarianceAssembled { trace_p0 } => tracing::trace!(
                target: InitEvent::TARGET, trace_p0, "Initial covariance P_0 assembled (diagonal)"
            ),
            CombinedUncertainty { sigma_ang_rad } => tracing::trace!(
                target: InitEvent::TARGET, sigma_ang_rad, "Combined astrometric uncertainty"
            ),
            FinalCovarianceTraces {
                cov_pos_trace_au2,
                cov_vel_trace,
            } => tracing::trace!(
                target: InitEvent::TARGET, cov_pos_trace_au2, cov_vel_trace, "Initial covariance traces"
            ),
            Complete => {
                tracing::trace!(target: InitEvent::TARGET, "KFState initialised successfully.")
            }
        }
    }

    pub fn span(epoch: f64) -> tracing::Span {
        tracing::trace_span!(target: InitEvent::TARGET, "init_kf_state", epoch)
    }
}

/// Diagonal variances of the angular **position** block $(\alpha, \delta)$.
///
/// These follow directly from the midpoint astrometric errors and do not
/// depend on the (unobservable) range: they are identical for every
/// hypothesis built from the same tracklet, whether there is a single
/// best-guess range ([`init_kf_state`]) or a whole grid of them
/// ([`crate::topocentric_kf::seed_grid::admissible_region_grid`]).
///
/// Return
/// ------
/// `(var_ra, var_dec)` in rad².
pub(crate) fn angular_position_variances(mid_point: &EquCoord) -> (f64, f64) {
    let var_ra = mid_point.ra_error.powi(2);
    let var_dec = mid_point.dec_error.powi(2);

    InitEvent::AngularVariances {
        sigma_ra_rad: mid_point.ra_error,
        sigma_dec_rad: mid_point.dec_error,
        var_ra,
        var_dec,
    }
    .emit();

    (var_ra, var_dec)
}

/// Diagonal variances of the angular **rate** block $(\dot{\alpha}, \dot{\delta})$.
///
/// Propagated from the finite-difference angular rate errors. Like
/// [`angular_position_variances`], this block is shared verbatim by every
/// hypothesis derived from a given tracklet.
///
/// Return
/// ------
/// `(var_ra_dot, var_dec_dot)` in (rad/day)².
pub(crate) fn angular_rate_variances(mid_speed: &MidSpeed) -> (f64, f64) {
    let var_ra_dot = mid_speed.0.ra_error.powi(2);
    let var_dec_dot = mid_speed.0.dec_error.powi(2);

    InitEvent::AngularRateVariances {
        sigma_ra_dot: mid_speed.0.ra_error,
        sigma_dec_dot: mid_speed.0.dec_error,
        var_ra_dot,
        var_dec_dot,
    }
    .emit();

    (var_ra_dot, var_dec_dot)
}

/// Range variance $\sigma_\rho^2$ propagated from the inverse-distance
/// prior's uncertainty $\sigma_\gamma$, via the Al-Kashi relation between
/// $\rho$ and $\gamma = 1/r$.
///
/// $$\sigma_\rho = \left|\frac{d\rho}{d\gamma}\right|\,\sigma_\gamma, \qquad
///   \frac{d\rho}{d\gamma} = \frac{1}{\gamma^3\,(\rho + r_{obs}\cos\phi)}$$
///
/// where $\cos\phi = -\hat{\rho}\cdot\hat{r}_{obs}$ is the solar elongation
/// cosine.
///
/// This is specific to [`init_kf_state`]'s single-range-guess strategy: it
/// is only meaningful when $\rho$ was itself *derived* from `gamma`. The
/// grid-based strategy in
/// [`crate::topocentric_kf::seed_grid::admissible_region_grid`] instead sets
/// each node's range variance from its local grid-cell width, since there
/// `rho` is a free coordinate rather than a quantity derived from a prior
/// on $\gamma$.
pub(crate) fn range_variance_from_gamma_uncertainty(
    los: &Vector3<f64>,
    r_obs: &Vector3<f64>,
    rho: f64,
    gamma: f64,
    sigma_gamma: f64,
) -> f64 {
    // Propagate sigma_gamma through the Al-Kashi solution:
    //   d(rho)/d(gamma) = 1 / (gamma^3 * (rho + r_obs_norm * cos_phi))
    // where cos_phi = -(los . r_obs) / |r_obs|
    let r_obs_norm = r_obs.norm();
    let cos_phi = -los.dot(r_obs) / r_obs_norm;
    let d_rho_d_gamma = 1.0 / (gamma.powi(3) * (rho + r_obs_norm * cos_phi));
    let sigma_rho = d_rho_d_gamma.abs() * sigma_gamma;
    let var_rho = sigma_rho.powi(2);

    InitEvent::RangeVariance {
        r_obs_norm_au: r_obs_norm,
        cos_phi,
        d_rho_d_gamma,
        sigma_rho_au: sigma_rho,
        var_rho_au2: var_rho,
    }
    .emit();

    var_rho
}

/// Range-rate variance $\sigma_{\dot\rho}^2$ bounded by the Keplerian
/// circular velocity at $r = 1/\gamma$.
///
/// $$\sigma_{\dot\rho}^2 = \frac{k^2}{r} = k^2\,\gamma$$
///
/// where $k^2$ is [`GAUSS_GRAV_SQUARED`]. Reflects total ignorance of the
/// radial velocity at initialization. As with
/// [`range_variance_from_gamma_uncertainty`], this is specific to the
/// single-guess strategy; the grid-based strategy derives
/// $\sigma_{\dot\rho}$ from the local extent of the bound-orbit
/// $\dot{\rho}$ interval at each node instead.
pub(crate) fn range_rate_variance_keplerian_bound(gamma: f64) -> f64 {
    let var_rho_dot = GAUSS_GRAV_SQUARED * gamma;

    InitEvent::RangeRateVariance {
        sigma_rho_dot_au_per_day: var_rho_dot.sqrt(),
        var_rho_dot,
    }
    .emit();

    var_rho_dot
}

/// Assemble the $6\times 6$ diagonal initial covariance $P_0$ in attributable
/// coordinates $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$
/// from its six per-component variances.
///
/// All cross-correlations are assumed zero at initialization (see
/// [`initial_covariance`] for the justification of each block). This is the
/// single place where the attributable covariance layout is defined; it is
/// shared by [`initial_covariance`] (used by [`init_kf_state`]) and by
/// [`crate::topocentric_kf::seed_grid::admissible_region_grid`]'s per-node
/// covariance.
pub(crate) fn assemble_diagonal_covariance(
    var_ra: f64,
    var_dec: f64,
    var_ra_dot: f64,
    var_dec_dot: f64,
    var_rho: f64,
    var_rho_dot: f64,
) -> Matrix6<f64> {
    let mut covariance = Matrix6::zeros();
    covariance[(0, 0)] = var_ra;
    covariance[(1, 1)] = var_dec;
    covariance[(2, 2)] = var_ra_dot;
    covariance[(3, 3)] = var_dec_dot;
    covariance[(4, 4)] = var_rho;
    covariance[(5, 5)] = var_rho_dot;

    InitEvent::InitialCovarianceAssembled {
        trace_p0: covariance.trace(),
    }
    .emit();

    covariance
}

/// Build the initial Kalman filter covariance in the attributable state space.
///
/// Constructs the $6 \times 6$ covariance matrix $P_0$ for the state vector
/// $\mathbf{x} = (\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$.
///
/// Block structure
/// ---------------
///
/// **Angular position block** $(\alpha, \delta)$ — diagonal, from midpoint astrometric errors:
///
/// $$P_{\alpha\alpha} = \sigma_\alpha^2, \quad P_{\delta\delta} = \sigma_\delta^2$$
///
/// **Angular rate block** $(\dot{\alpha}, \dot{\delta})$ — diagonal, from propagated finite-difference errors:
///
/// $$P_{\dot{\alpha}\dot{\alpha}} = \sigma_{\dot{\alpha}}^2, \quad P_{\dot{\delta}\dot{\delta}} = \sigma_{\dot{\delta}}^2$$
///
/// **Position–rate cross-correlations** — assumed zero under the hypothesis
/// $\sigma_{\alpha,1} \approx \sigma_{\alpha,2}$ (same survey instrument for both observations).
///
/// **Range block** $(\rho)$ — propagated from the $\gamma$ uncertainty through the
/// Al-Kashi geometry. Differentiating $\rho^2 + 2\rho\,r_{obs}\cos\phi + r_{obs}^2 - \gamma^{-2} = 0$:
///
/// $$\sigma_\rho = \frac{\sigma_\gamma}{\gamma^3\,(\rho + r_{obs}\cos\phi)}$$
///
/// where $\cos\phi = -\hat{\rho} \cdot \hat{r}_{obs}$ is the solar elongation cosine.
///
/// **Range-rate block** $(\dot{\rho})$ — bounded by the circular Keplerian velocity:
///
/// $$\sigma_{\dot{\rho}} = \sqrt{\frac{k^2}{r}} = \sqrt{k^2\,\gamma}$$
///
/// where $k^2$ is [`GAUSS_GRAV_SQUARED`]. This reflects total ignorance of
/// the radial velocity at initialization.
///
/// **Cross-correlations** $(\rho, \dot{\rho})$, $(\rho, \alpha)$, etc. — set to zero.
///
/// Arguments
/// ---------
/// * `mid_point` – Midpoint angular coordinates with astrometric errors $(\sigma_\alpha, \sigma_\delta)$ in radians.
/// * `mid_speed` – Angular rates with propagated errors $(\sigma_{\dot{\alpha}}, \sigma_{\dot{\delta}})$ in rad/day.
/// * `los` – Unit line-of-sight vector $\hat{\rho}$ in the ecliptic J2000 frame.
/// * `r_obs` – Observer heliocentric position vector (AU).
/// * `rho` – Topocentric range (AU).
/// * `gamma` – Inverse heliocentric distance prior $\gamma = 1/r$ (AU$^{-1}$).
/// * `sigma_gamma` – Uncertainty on $\gamma$ (AU$^{-1}$).
///
/// Return
/// ------
/// $P_0$ as a [`Matrix6<f64>`] with units matching the state vector:
/// $(\text{rad}^2,\,\text{rad}^2,\,\text{rad}^2/\text{day}^2,\,\text{rad}^2/\text{day}^2,\,\text{AU}^2,\,\text{AU}^2/\text{day}^2)$.
pub fn initial_covariance(
    mid_point: &EquCoord,
    mid_speed: &MidSpeed,
    los: &Vector3<f64>,
    r_obs: &Vector3<f64>,
    rho: f64,
    gamma: f64,
    sigma_gamma: f64,
) -> Matrix6<f64> {
    let (var_ra, var_dec) = angular_position_variances(mid_point);
    let (var_ra_dot, var_dec_dot) = angular_rate_variances(mid_speed);
    let var_rho = range_variance_from_gamma_uncertainty(los, r_obs, rho, gamma, sigma_gamma);
    let var_rho_dot = range_rate_variance_keplerian_bound(gamma);

    assemble_diagonal_covariance(
        var_ra,
        var_dec,
        var_ra_dot,
        var_dec_dot,
        var_rho,
        var_rho_dot,
    )
}

/// Heliocentric observer state (position and velocity) at the tracklet's
/// midpoint epoch.
///
/// Looks up the observer's heliocentric position/velocity at each
/// observation's own epoch via the ephemeris `state`, then linearly
/// interpolates ([`HelioObsState::mid_state`]) to the midpoint between the
/// two epochs. Shared by [`init_kf_state`] and
/// [`crate::topocentric_kf::seed_grid::admissible_region_grid`], which both
/// need the same observer geometry to resolve a tracklet's range ambiguity.
pub(crate) fn init_observer_state(
    obs_dataset: &ObsDataset,
    obs1: &Observation,
    obs2: &Observation,
    state: &EphemState,
) -> Result<HelioObsState, OutfitError> {
    let observer = get_observer(obs_dataset, obs1)?;
    let helio_state1 = state.helio_observer_state(observer, obs1.mjd_tt())?;

    let observer = get_observer(obs_dataset, obs2)?;
    let helio_state2 = state.helio_observer_state(observer, obs2.mjd_tt())?;

    Ok(helio_state1.mid_state(&helio_state2))
}

pub(crate) fn init_kf_state<'state_lf>(
    obs_dataset: &ObsDataset,
    first_obs: &Observation,
    second_obs: &Observation,
    n_sigma: f64,
    state: &'state_lf KalmanContext,
) -> Result<KFState<'state_lf>, EngineError> {
    let t_mid = pair_midpoint_epoch(first_obs, second_obs);

    let _enter = InitEvent::span(t_mid).entered();

    let HelioObsState {
        helio_cart_pos: r_obs,
        helio_cart_vel: v_obs,
    } = init_observer_state(obs_dataset, first_obs, second_obs, state.get_ephem())?;

    // Tracklet geometry (midpoint angles/rates and line-of-sight vectors),
    // shared with the grid-based initialization in `seed_grid`.
    let TrackletGeometry {
        mid_point,
        mid_speed,
        los,
        los_dot,
    } = tracklet_geometry(first_obs, second_obs);

    InitEvent::TrackletGeometry {
        mid_ra_deg: mid_point.ra.to_degrees(),
        mid_dec_deg: mid_point.dec.to_degrees(),
        mid_speed_ra: mid_speed.0.ra,
        mid_speed_dec: mid_speed.0.dec,
        sigma_ra_rad: mid_point.ra_error,
        sigma_dec_rad: mid_point.dec_error,
    }
    .emit();
    InitEvent::ObserverState {
        r_obs: [r_obs[0], r_obs[1], r_obs[2]],
        v_obs: [v_obs[0], v_obs[1], v_obs[2]],
    }
    .emit();
    InitEvent::LineOfSight {
        los: [los[0], los[1], los[2]],
        los_dot: [los_dot[0], los_dot[1], los_dot[2]],
        n_sigma,
    }
    .emit();

    // ── Gamma (inverse heliocentric distance) ─────────────────────────────
    let gamma = estimate_gamma(&mid_point, &mid_speed);
    let r_helio = 1.0 / gamma;

    InitEvent::GammaEstimate {
        gamma,
        r_helio_au: r_helio,
    }
    .emit();

    // ── Topocentric range ─────────────────────────────────────────────────
    let rho = topocentric_range(&los, &r_obs, gamma)?;

    InitEvent::TopocentricRange { rho_au: rho }.emit();

    // ── Heliocentric distance bounds ──────────────────────────────────────
    let (r_min, r_max) = estimate_r_bounds(&mid_point, &mid_speed, n_sigma);

    InitEvent::DistanceBounds {
        r_min_au: r_min,
        r_max_au: r_max,
    }
    .emit();

    // ── Gamma uncertainty ─────────────────────────────────────────────────
    let sigma_gamma = estimate_sigma_gamma(gamma, &mid_point, &mid_speed, r_min, r_max);
    let sigma_gamma_rel_pct = (sigma_gamma / gamma).abs() * 100.0;

    InitEvent::GammaUncertainty {
        sigma_gamma,
        sigma_gamma_rel_pct,
    }
    .emit();

    // ── Initial position/velocity sigmas ──────────────────────────────────
    let (sigma_trans, sigma_rad) = estimate_init_sigmas(rho, gamma, &mid_speed, &mid_point);

    InitEvent::InitSigmas {
        sigma_trans_au: sigma_trans,
        sigma_rad_au: sigma_rad,
    }
    .emit();

    // ── rho_dot and heliocentric velocity ─────────────────────────────────
    let rho_dot =
        estimate_rho_dot_circular(&los, &los_dot, &v_obs, rho, r_helio).unwrap_or_else(|| {
            InitEvent::RhoDotFallback {
                rho_au: rho,
                r_helio_au: r_helio,
            }
            .emit();
            0.0
        });

    InitEvent::RhoDot {
        rho_dot_au_per_day: rho_dot,
    }
    .emit();

    // let rho = 2.1;
    // let rho_dot = 0.000005;

    // ── State vector ──────────────────────────────────────────────────────
    let x = Vector6::new(
        mid_point.ra,
        mid_point.dec,
        mid_speed.0.ra,
        mid_speed.0.dec,
        rho,
        rho_dot,
    );

    // ── Initial covariance ────────────────────────────────────────────────
    let sigma_ang = mid_point.ra_error.hypot(mid_point.dec_error) / 2_f64.sqrt();

    InitEvent::CombinedUncertainty {
        sigma_ang_rad: sigma_ang,
    }
    .emit();

    let p = initial_covariance(
        &mid_point,
        &mid_speed,
        &los,
        &r_obs,
        rho,
        gamma,
        sigma_gamma,
    );

    InitEvent::FinalCovarianceTraces {
        cov_pos_trace_au2: p.fixed_view::<3, 3>(0, 0).trace(),
        cov_vel_trace: p.fixed_view::<3, 3>(3, 3).trace(),
    }
    .emit();

    InitEvent::Complete.emit();

    Ok(KFState {
        state: x,
        covariance: p,
        epoch: t_mid,

        r_obs,
        v_obs,
        universal_anomaly: None,

        kalman_gain: None,
        nis_ema: None,

        shared_ctx: state,
    })
}

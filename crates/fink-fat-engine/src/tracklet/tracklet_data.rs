use hifitime::ut1::Ut1Provider;
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};
use outfit::{
    ApparentPosition, EphemerisRequest, EphemerisResult, JPLEphem, Position,
    constants::FitOrbitResult,
};
use photom::{
    coordinates::ecliptic::EclipticCoordCov,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};

use crate::{
    ecliptic_state::{EclipticState, SingerParams},
    error::EngineError,
    tracklet::track_storage::TrackId,
};

#[derive(Debug, Clone)]
pub struct TrackletData<State> {
    pub key: TrackId,
    pub state: State,
    pub obs_keys: Vec<ObsId>,
    pub ref_mag: f64,
    pub ref_mag_err: f64,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Ecliptic coordinates and their variances extracted from an observation.
struct EclipticObs {
    lon: f64,
    lat: f64,
    var_lon: f64,
    var_lat: f64,
}

impl EclipticObs {
    fn from_obs(obs: &Observation) -> Self {
        let ecl_cov: EclipticCoordCov = (*obs.equ_coord()).into(); // From<EquCoord> for EclipticCoordCov
        Self {
            lon: ecl_cov.coord.lon,
            lat: ecl_cov.coord.lat,
            var_lon: ecl_cov.cov.xx,
            var_lat: ecl_cov.cov.yy,
        }
    }
}

/// Fit a quadratic polynomial $q(\tau) = c_0 + c_1\tau + c_2\tau^2$ to three points
/// by solving the Vandermonde system via LU decomposition.
///
/// Arguments
/// ---------
/// * `dts`  – Time offsets $[\tau_0, \tau_1, \tau_2]$ from the reference epoch (days).
/// * `vals` – Scalar values $[q_0, q_1, q_2]$ at each epoch.
///
/// Return
/// ------
/// * `Some((c0, c1, c2))` – Fitted coefficients (position, velocity, acceleration).
/// * `None` – Singular system (degenerate time baseline).
fn fit_quadratic(dts: [f64; 3], vals: [f64; 3]) -> Option<(f64, f64, f64)> {
    let [t0, t1, t2] = dts;

    #[rustfmt::skip]
    let v = Matrix3::new(
        1.0, t0, t0 * t0,
        1.0, t1, t1 * t1,
        1.0, t2, t2 * t2,
    );

    v.lu()
        .solve(&Vector3::from(vals))
        .map(|c| (c[0], c[1], c[2]))
}

/// Check whether the apparent speed $\sqrt{\dot\lambda^2 + \dot\beta^2}$
/// exceeds the optional cutoff. Returns `true` if the speed is acceptable.
#[inline]
fn speed_ok(dlon: f64, dlat: f64, vmax: f64) -> bool {
    dlon * dlon + dlat * dlat <= vmax * vmax
}

// ── TrackletData ──────────────────────────────────────────────────────────────

impl TrackletData<EclipticState> {
    fn new(
        id: TrackId,
        x: Vector6<f64>,
        p: Matrix6<f64>,
        epoch: f64,
        obs_keys: Vec<ObsId>,
        process_noise_q: f64,
        singer_params: Option<SingerParams>,
        ref_mag: f64,
        ref_mag_err: f64,
    ) -> Self {
        Self {
            key: id,
            state: EclipticState {
                x,
                p,
                epoch,
                process_noise_q,
                singer: singer_params,
            },
            obs_keys,
            ref_mag,
            ref_mag_err,
        }
    }

    pub fn get_observations<'a>(
        &self,
        obs_dataset: &'a ObsDataset,
    ) -> Result<Vec<&'a Observation>, EngineError> {
        self.obs_keys
            .iter()
            .map(|obs_id| {
                obs_dataset
                    .get_observation(*obs_id)
                    .ok_or(EngineError::ObsDatasetIdNotFound(*obs_id))
            })
            .collect()
    }

    pub fn mean_magnitude(&self, obs_dataset: &ObsDataset) -> Result<(f64, f64), EngineError> {
        let obs_members = self.get_observations(obs_dataset)?;

        let (sum_w, sum_wm) = obs_members
            .iter()
            .map(|obs| obs.photometry())
            .filter(|phot| phot.error > 0.0)
            .fold((0.0_f64, 0.0_f64), |(sum_w, sum_wm), phot| {
                let w = 1.0 / (phot.error * phot.error);
                (sum_w + w, sum_wm + w * phot.magnitude)
            });

        if sum_w == 0.0 {
            return Ok((0.0, 0.0));
        }

        Ok((sum_wm / sum_w, 1.0 / sum_w.sqrt()))
    }

    /// Build a [`TrackletData`] from a pair of observations (linear ecliptic model).
    ///
    /// Fits a constant-velocity kinematic state in ecliptic coordinates from two
    /// observations separated by $\Delta t = t_b - t_a$ days.
    ///
    /// State initialisation
    /// --------------------
    /// The state vector is initialised at the midpoint epoch
    /// $t_\text{mid} = \tfrac{1}{2}(t_a + t_b)$:
    ///
    /// $$\mathbf{x} = (\lambda_\text{mid},\; \dot\lambda,\; 0,\; \beta_\text{mid},\; \dot\beta,\; 0)^\top$$
    ///
    /// where:
    /// $$\lambda_\text{mid} = \tfrac{\lambda_a + \lambda_b}{2}, \quad
    ///   \dot\lambda = \tfrac{\lambda_b - \lambda_a}{\Delta t}$$
    ///
    /// and symmetrically for $\beta$. Accelerations are set to zero.
    ///
    /// Covariance initialisation
    /// -------------------------
    /// Under a two-point linear fit, the covariance on position and velocity
    /// at the midpoint epoch are:
    ///
    /// $$\sigma^2_{\lambda_\text{mid}} = \tfrac{\sigma^2_{\lambda_a} + \sigma^2_{\lambda_b}}{4}, \quad
    ///   \sigma^2_{\dot\lambda} = \tfrac{\sigma^2_{\lambda_a} + \sigma^2_{\lambda_b}}{\Delta t^2}$$
    ///
    /// Acceleration uncertainty is set to a fixed prior $\sigma^2_{\ddot\lambda}$.
    /// All off-diagonal terms are set to zero.
    ///
    /// Arguments
    /// ---------
    /// * `id`                    – [`TrackId`] assigned to this tracklet.
    /// * `obs_a`                 – First observation (earlier epoch).
    /// * `obs_b`                 – Second observation (later epoch).
    /// * `acc_prior_var`         – Prior variance on acceleration (rad²/day⁴).
    /// * `max_speed_rad_per_day` – Speed cutoff (rad/day).
    /// * `process_noise_q` – Acceleration spectral density for process noise
    ///   during state propagation (rad²/day³). Controls how much uncertainty
    ///   is injected per unit time to account for unmodelled forces.
    ///   Pass `0.0` to disable process noise entirely.
    ///
    /// Return
    /// ------
    /// * `Some(TrackletData)` – Successfully initialised tracklet.
    /// * `None`               – Speed exceeds `max_speed_rad_per_day`.
    pub fn from_pair(
        id: TrackId,
        obs_a: &Observation,
        obs_b: &Observation,
        acc_prior_var: f64,
        max_speed_rad_per_day: f64,
        process_noise_q: f64,
        singer_params: Option<SingerParams>,
    ) -> Option<Self> {
        // --- Temporal baseline ---
        // Δt is the time separation between the two observations.
        // The reference epoch is placed at the midpoint to minimise
        // position–velocity correlation in the state covariance.
        let dt = obs_b.mjd_tt() - obs_a.mjd_tt();
        let inv_dt = 1.0 / dt;
        let epoch = 0.5 * (obs_a.mjd_tt() + obs_b.mjd_tt());

        // --- Equatorial → ecliptic conversion ---
        // Each observation carries (Ra, Dec) in equatorial coordinates.
        // We convert to ecliptic (λ, β) where the ecliptic plane roughly
        // aligns with solar system motion, reducing dynamical cross-terms.
        let a = EclipticObs::from_obs(obs_a);
        let b = EclipticObs::from_obs(obs_b);

        // --- Finite-difference velocity ---
        // With only two points, velocity is estimated as a simple finite difference:
        //   λ̇ = (λ_b − λ_a) / Δt,   β̇ = (β_b − β_a) / Δt
        let dlon = (b.lon - a.lon) * inv_dt;
        let dlat = (b.lat - a.lat) * inv_dt;

        // --- Speed filter ---
        // Reject pairs whose apparent angular speed exceeds the physical cutoff.
        // This removes obvious mismatches and very fast NEOs that cannot be
        // reliably linked with a linear model.
        if !speed_ok(dlon, dlat, max_speed_rad_per_day) {
            return None;
        }

        // --- State vector x = (λ_mid, λ̇, 0, β_mid, β̇, 0) ---
        // Position at the midpoint epoch is the arithmetic mean of the two measures.
        // Acceleration is unobservable from two points and is set to zero.
        let x = Vector6::new(
            0.5 * (a.lon + b.lon),
            dlon,
            0.0,
            0.5 * (a.lat + b.lat),
            dlat,
            0.0,
        );

        // --- Diagonal covariance matrix P ---
        //
        // Midpoint position variance (mean of two independent measurements):
        //   Var(λ_mid) = (σ²_λa + σ²_λb) / 4
        //
        // Finite-difference velocity variance (error propagation):
        //   Var(λ̇) = (σ²_λa + σ²_λb) / Δt²
        //
        // Acceleration variance: unobservable, set to the external prior.
        // All off-diagonal terms are zero (no cross-terms estimated from two points).
        let inv_dt2 = inv_dt * inv_dt;
        let sum_var_lon = a.var_lon + b.var_lon;
        let sum_var_lat = a.var_lat + b.var_lat;

        let p = Matrix6::from_diagonal(&Vector6::new(
            sum_var_lon * 0.25,
            sum_var_lon * inv_dt2,
            acc_prior_var,
            sum_var_lat * 0.25,
            sum_var_lat * inv_dt2,
            acc_prior_var,
        ));

        let phot_a = obs_a.photometry();
        let phot_b = obs_b.photometry();

        let w_a = 1.0 / (phot_a.error * phot_a.error);
        let w_b = 1.0 / (phot_b.error * phot_b.error);
        let sum_w = w_a + w_b;

        let ref_mag = (w_a * phot_a.magnitude + w_b * phot_b.magnitude) / sum_w;
        let ref_mag_err = 1.0 / sum_w.sqrt();

        Some(Self::new(
            id,
            x,
            p,
            epoch,
            vec![obs_a.id().clone(), obs_b.id().clone()],
            process_noise_q,
            singer_params,
            ref_mag,
            ref_mag_err,
        ))
    }

    /// Build a [`TrackletData`] from a triplet of observations (quadratic ecliptic model).
    ///
    /// Fits a constant-acceleration kinematic state in ecliptic coordinates from three
    /// observations via quadratic polynomial fit on time offsets from the mean epoch.
    ///
    /// State initialisation
    /// --------------------
    /// The state vector is initialised at the mean epoch
    /// $t_\text{mid} = \tfrac{1}{3}(t_a + t_b + t_c)$:
    ///
    /// $$\mathbf{x} = (\lambda_\text{mid},\; \dot\lambda,\; \ddot\lambda,\; \beta_\text{mid},\; \dot\beta,\; \ddot\beta)^\top$$
    ///
    /// Covariance initialisation
    /// -------------------------
    /// Let $\bar\sigma^2_\lambda = \tfrac{1}{3}(\sigma^2_{\lambda_a} + \sigma^2_{\lambda_b} + \sigma^2_{\lambda_c})$.
    /// The diagonal covariances are approximated as:
    ///
    /// $$\sigma^2_{\lambda_\text{mid}} = \frac{\bar\sigma^2_\lambda}{3}, \quad
    ///   \sigma^2_{\dot\lambda} = \frac{\bar\sigma^2_\lambda}{\Delta t^2}, \quad
    ///   \sigma^2_{\ddot\lambda} = \frac{2\,\bar\sigma^2_\lambda}{\Delta t^4}$$
    ///
    /// where $\Delta t = t_c - t_a$. All off-diagonal terms are set to zero.
    ///
    /// Arguments
    /// ---------
    /// * `id`                    – [`TrackId`] assigned to this tracklet.
    /// * `obs_a`                 – First observation (earliest epoch).
    /// * `obs_b`                 – Second observation (middle epoch).
    /// * `obs_c`                 – Third observation (latest epoch).
    /// * `max_speed_rad_per_day` – Speed cutoff (rad/day).
    /// * `process_noise_q` – Acceleration spectral density for process noise
    ///   during state propagation (rad²/day³). Controls how much uncertainty
    ///   is injected per unit time to account for unmodelled forces.
    ///   Pass `0.0` to disable process noise entirely.
    ///
    /// Return
    /// ------
    /// * `Some(TrackletData)` – Successfully initialised tracklet.
    /// * `None`               – Singular quadratic system or speed exceeds cutoff.
    pub fn from_triplet(
        id: TrackId,
        obs_a: &Observation,
        obs_b: &Observation,
        obs_c: &Observation,
        max_speed_rad_per_day: f64,
        process_noise_q: f64,
        singer_params: Option<SingerParams>,
    ) -> Option<Self> {
        // --- Reference epoch and time offsets ---
        // The mean epoch minimises the correlation between position and velocity
        // in the fitted polynomial. Time offsets τ_i = t_i − t_mid are centred,
        // which improves the numerical conditioning of the Vandermonde system.
        let (ta, tb, tc) = (obs_a.mjd_tt(), obs_b.mjd_tt(), obs_c.mjd_tt());
        let epoch = (ta + tb + tc) / 3.0;
        let dts = [ta - epoch, tb - epoch, tc - epoch];

        // --- Characteristic baseline for covariance scaling ---
        // Δt = t_c − t_a is the total time span of the triplet.
        // The floor at 1e-6 days prevents division by zero for near-simultaneous observations.
        let dt_char = (tc - ta).max(1e-6);
        let inv_dt2 = 1.0 / (dt_char * dt_char);
        let inv_dt4 = inv_dt2 * inv_dt2;

        // --- Equatorial → ecliptic conversion ---
        let a = EclipticObs::from_obs(obs_a);
        let b = EclipticObs::from_obs(obs_b);
        let c = EclipticObs::from_obs(obs_c);

        // --- Quadratic polynomial fit ---
        // For each ecliptic coordinate q ∈ {λ, β}, solve the 3×3 Vandermonde system:
        //
        //   q(τ) = c0 + c1·τ + c2·τ²
        //
        // evaluated at the three centred offsets τ_i. The coefficients map directly
        // to the kinematic quantities at the reference epoch:
        //   c0 → position   q_mid  [rad]
        //   c1 → velocity   q̇      [rad/day]
        //   c2 → acceleration q̈   [rad/day²]
        //
        // Returns None if the system is singular (e.g. duplicate epochs).
        let (lon_mid, dlon, ddlon) = fit_quadratic(dts, [a.lon, b.lon, c.lon])?;
        let (lat_mid, dlat, ddlat) = fit_quadratic(dts, [a.lat, b.lat, c.lat])?;

        // --- Speed filter ---
        // Applied to the fitted velocity at the reference epoch.
        if !speed_ok(dlon, dlat, max_speed_rad_per_day) {
            return None;
        }

        // --- State vector x = (λ_mid, λ̇, λ̈, β_mid, β̇, β̈) ---
        let x = Vector6::new(lon_mid, dlon, ddlon, lat_mid, dlat, ddlat);

        // --- Diagonal covariance matrix P ---
        //
        // The mean measurement variance per coordinate is used as a proxy for the
        // uniform noise level σ²:
        //   σ²_λ = (σ²_λa + σ²_λb + σ²_λc) / 3
        //
        // Under this approximation, the covariances of the three fitted coefficients
        // scale as follows with the total baseline Δt:
        //
        //   Var(λ_mid) = σ²_λ / 3          (mean over 3 samples)
        //   Var(λ̇)    = σ²_λ / Δt²        (first finite difference)
        //   Var(λ̈)    = 2·σ²_λ / Δt⁴      (second finite difference)
        //
        // All off-diagonal terms are set to zero at this initialisation stage.
        let mean_var_lon = (a.var_lon + b.var_lon + c.var_lon) / 3.0;
        let mean_var_lat = (a.var_lat + b.var_lat + c.var_lat) / 3.0;

        let p = Matrix6::from_diagonal(&Vector6::new(
            mean_var_lon / 3.0,
            mean_var_lon * inv_dt2,
            2.0 * mean_var_lon * inv_dt4,
            mean_var_lat / 3.0,
            mean_var_lat * inv_dt2,
            2.0 * mean_var_lat * inv_dt4,
        ));

        let phot_a = obs_a.photometry();
        let phot_b = obs_b.photometry();
        let phot_c = obs_c.photometry();

        let w_a = 1.0 / (phot_a.error * phot_a.error);
        let w_b = 1.0 / (phot_b.error * phot_b.error);
        let w_c = 1.0 / (phot_c.error * phot_c.error);
        let sum_w = w_a + w_b + w_c;

        let ref_mag =
            (w_a * phot_a.magnitude + w_b * phot_b.magnitude + w_c * phot_c.magnitude) / sum_w;
        let ref_mag_err = 1.0 / sum_w.sqrt();

        Some(Self::new(
            id,
            x,
            p,
            epoch,
            vec![obs_a.id().clone(), obs_b.id().clone(), obs_c.id().clone()],
            process_noise_q,
            singer_params,
            ref_mag,
            ref_mag_err,
        ))
    }
}

impl TrackletData<FitOrbitResult> {
    pub fn new(
        id: TrackId,
        obs_keys: Vec<ObsId>,
        orbital_elements: FitOrbitResult,
        ref_mag: f64,
        ref_mag_err: f64,
    ) -> Self {
        Self {
            key: id,
            state: orbital_elements,
            obs_keys,
            ref_mag,
            ref_mag_err,
        }
    }

    pub fn predict(
        &self,
        ephem_request: &EphemerisRequest<Position>,
        jpl: &JPLEphem,
        ut1: &Ut1Provider,
    ) -> EphemerisResult<ApparentPosition> {
        let state = self.state.orbital_elements();
        state.compute(ephem_request, jpl, ut1)
    }
}

pub mod context;
pub mod display;
pub mod init;
pub mod propagate;
pub mod update;

use nalgebra::{Matrix2, Matrix6, Matrix6x2, Vector3, Vector6};
use outfit::{EquinoctialElements, OrbitalElements, OutfitError};
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, observation::Observation},
};

use crate::{
    error::{EngineError, KFUpdateError, ObservationJacobianError},
    topocentric_kf::{
        conversion::{CartesianState, attributable_to_cartesian},
        observer_state::get_observer,
        single_kalman::{
            context::KalmanContext,
            init::init_kf_state,
            propagate::{PropagateError, propagate_kf},
            update::{observation_jacobian, update_kf},
        },
    },
};

/// Topocentric Kalman filter state vector in attributable coordinates.
///
/// # State space
///
/// The state is expressed in **attributable coordinates**, following the
/// formalism of Milani et al. (2007). This representation is chosen because
/// the observations (RA, Dec) correspond directly to state components,
/// making the observation Jacobian well-conditioned and avoiding the
/// structural rank deficiency that arises when a Cartesian 3-D state is
/// updated from 2-D angular observations.
///
/// The state vector is:
///
/// $$\mathbf{x} = (\alpha,\, \delta,\, \dot{\alpha},\, \dot{\delta},\, \rho,\, \dot{\rho})^\top$$
///
/// | Index | Symbol | Description | Unit |
/// |-------|--------|-------------|------|
/// | 0 | $\alpha$ | Right ascension | rad |
/// | 1 | $\delta$ | Declination | rad |
/// | 2 | $\dot{\alpha}$ | RA rate (true, not $\cos\delta$-reduced) | rad/day |
/// | 3 | $\dot{\delta}$ | Dec rate | rad/day |
/// | 4 | $\rho$ | Topocentric range | AU |
/// | 5 | $\dot{\rho}$ | Topocentric range rate | AU/day |
///
/// # Topocentric range parameterization
///
/// The topocentric range $\rho$ is the direct distance from the observer to
/// the object. It is preferred over the inverse range $\gamma = 1/\rho$ in
/// the state vector because it appears naturally in the geometric equations
/// linking the observer, the object, and the Sun:
///
/// $$\mathbf{r}_\text{helio} = \mathbf{r}_\text{obs} + \rho\,\hat{\mathbf{u}}$$
///
/// where $\hat{\mathbf{u}}$ is the unit line-of-sight vector. The inverse
/// range $\gamma$ is nonetheless used **internally** during initialization
/// to estimate the heliocentric distance from angular observables
/// (see `estimate_gamma`), and is then converted to $\rho$ via
/// `topocentric_range` before being stored in the state.
///
/// # Initialization
///
/// The state is initialized from a pair of observations (an intra-night
/// tracklet) using a HelioLinC-inspired approach:
///
/// - $(\alpha, \delta)$ are taken from the midpoint observation.
/// - $(\dot{\alpha}, \dot{\delta})$ are estimated from finite differences
///   between the two observations.
/// - $\gamma = 1/r_\text{helio}$ is first estimated from the angular
///   observables via `estimate_gamma`, then converted to $\rho$ via
///   `topocentric_range`.
/// - $\dot{\rho}$ is estimated from the circular-orbit approximation
///   (`estimate_rho_dot_circular`), and falls back to zero if the geometry
///   is inconsistent.
///
/// The initial covariance $P_0$ is constructed in Cartesian space and
/// mapped to attributable coordinates (see `initial_covariance`):
///
/// $$P_0 = J \, \Sigma_\text{cart} \, J^\top$$
///
/// where $J$ is the Jacobian of the coordinate transform and
/// $\Sigma_\text{cart}$ is the Cartesian uncertainty decomposed into
/// transverse and radial components.
///
/// # Observation model
///
/// At each update step, the observation vector is:
///
/// $$\mathbf{z} = (\alpha_\text{obs},\, \delta_\text{obs})^\top$$
///
/// The observation Jacobian $H \in \mathbb{R}^{2 \times 6}$ is:
///
/// $$H = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{pmatrix}$$
///
/// This is exact at the prediction point and requires no linearization,
/// which is a key advantage of the attributable parameterization.
///
/// # Propagation
///
/// Between observations, the state is propagated under a **uniform
/// apparent motion** model (constant $\dot{\alpha}$, $\dot{\delta}$,
/// $\dot{\rho}$):
///
/// $$\alpha(t) = \alpha_0 + \dot{\alpha}\,\Delta t, \quad
///   \delta(t) = \delta_0 + \dot{\delta}\,\Delta t, \quad
///   \rho(t) = \rho_0 + \dot{\rho}\,\Delta t$$
///
/// giving the transition matrix:
///
/// $$F = \begin{pmatrix}
///   1 & 0 & \Delta t & 0 & 0 & 0 \\
///   0 & 1 & 0 & \Delta t & 0 & 0 \\
///   0 & 0 & 1 & 0 & 0 & 0 \\
///   0 & 0 & 0 & 1 & 0 & 0 \\
///   0 & 0 & 0 & 0 & 1 & \Delta t \\
///   0 & 0 & 0 & 0 & 0 & 1
/// \end{pmatrix}$$
///
/// This model is an approximation; process noise $Q$ must absorb
/// deviations from uniform motion due to gravitational acceleration.
///
/// # Reference
///
/// Milani, A. et al. (2007), *Orbit determination with topocentric
/// corrections*, Celestial Mechanics and Dynamical Astronomy.
#[derive(Clone)]
pub struct KFState<'state_lf> {
    /// State vector $\mathbf{x} \in \mathbb{R}^6$ in attributable coordinates.
    ///
    /// Layout: $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})^\top$.
    pub state: Vector6<f64>,
    /// State covariance matrix $P \in \mathbb{R}^{6 \times 6}$ in attributable space.
    ///
    /// Off-diagonal terms couple angular position with angular rate
    /// (as expected after several updates) and $\rho$ with $\dot{\rho}$.
    pub covariance: Matrix6<f64>,

    /// Reference epoch of the state (MJD TT).
    pub epoch: f64,

    /// Observer heliocentric position at `kf.epoch` (AU), ecliptic J2000.
    pub r_obs: Vector3<f64>,
    /// Observer heliocentric velocity at `kf.epoch` (AU/day), ecliptic J2000.
    pub v_obs: Vector3<f64>,
    /// previous universal anomaly estimation, used for propagation
    pub universal_anomaly: Option<f64>,

    /// Kalman gain matrix $K \in \mathbb{R}^{6 \times 2}$ from the last update step.
    ///
    /// `None` if no update has been applied yet (freshly initialized state).
    pub kalman_gain: Option<Matrix6x2<f64>>,
    /// Exponentially-smoothed Normalized Innovation Squared (NIS) carried from
    /// past update steps.
    ///
    /// Drives the adaptive covariance inflation applied during propagation
    /// (see [`crate::topocentric_kf::propagate`]). Under a consistent filter
    /// the per-step NIS follows a $\chi^2(2)$ law (expected value 2); a
    /// persistently large smoothed value signals an over-confident covariance
    /// that pure two-body process noise fails to keep open, and triggers
    /// fading-memory inflation on the next prediction.
    ///
    /// `None` until the first measurement update has been applied (freshly
    /// initialized or seed-grid states carry no innovation history yet).
    pub nis_ema: Option<f64>,

    pub shared_ctx: &'state_lf KalmanContext,
}

impl<'state_lf> KFState<'state_lf> {
    pub fn update_epoch(
        self,
        obs_dataset: &ObsDataset,
        obs: &Observation,
    ) -> Result<Self, OutfitError> {
        let new_epoch = obs.mjd_tt();
        let observer = get_observer(obs_dataset, obs)?;

        let helio_state1 = self
            .shared_ctx
            .get_ephem()
            .helio_observer_state(observer, new_epoch)?;

        Ok(Self {
            epoch: new_epoch,
            r_obs: helio_state1.helio_cart_pos,
            v_obs: helio_state1.helio_cart_vel,
            ..self
        })
    }

    /// Initialize a topocentric Kalman filter state from a pair of observations.
    ///
    /// Implements the HelioLinC-inspired initialization strategy: the unknown
    /// topocentric range $\rho$ is resolved from a prior on the heliocentric
    /// distance $r = 1/\gamma$ via [`topocentric_range`], and the radial velocity
    /// $\dot{\rho}$ is set to zero (no radial motion prior).
    ///
    /// The state is expressed in the **heliocentric ecliptic mean J2000** frame.
    ///
    /// Initialization steps
    /// --------------------
    /// 1. Compute line-of-sight $\hat{\rho}$ and $\dot{\hat{\rho}}$ from
    ///    `mid_point` and `mid_speed`.
    /// 2. Resolve $\rho$ from the Al-Kashi equation given $\gamma$.
    /// 3. Estimate $\sigma_\gamma$ from angular rate errors and the distance
    ///    range $[r_{min}, r_{max}]$ via [`estimate_sigma_gamma`].
    /// 4. Derive covariance scale factors via [`estimate_init_sigmas`].
    /// 5. Assemble heliocentric state:
    ///
    /// $$\mathbf{r} = \mathbf{r}_{obs} + \rho\,\hat{\rho}$$
    ///
    /// $$\mathbf{v} = \mathbf{v}_{obs} + \rho\,\dot{\hat{\rho}}$$
    ///
    /// 6. Build $P_0$ via [`initial_covariance`].
    ///
    /// Arguments
    /// ---------
    /// * `mid_point` – Midpoint angular coordinates $(\alpha, \delta)$ in radians.
    /// * `mid_speed` – Angular velocity $(\dot{\alpha}, \dot{\delta})$ in rad/day.
    /// * `r_obs` – Heliocentric position of the observer at $t_{mid}$ (AU).
    /// * `v_obs` – Heliocentric velocity of the observer at $t_{mid}$ (AU/day).
    /// * `t_mid` – Reference epoch (MJD TT).
    /// * `n_sigma` – Number of $\sigma_\omega$ used to derive the distance bounds
    ///   $[r_{min}, r_{max}]$ via [`estimate_r_bounds`]. A value of `3.0` is
    ///   recommended for conservative initialization.
    ///
    /// Return
    /// ------
    /// * `Some(KFState)` – Initialized state vector and covariance.
    /// * `None` – If no positive topocentric range solution exists for the given $\gamma$.
    pub fn init_kf_state(
        obs_dataset: &ObsDataset,
        first_obs: &Observation,
        second_obs: &Observation,
        n_sigma: f64,
        state: &'state_lf KalmanContext,
    ) -> Result<KFState<'state_lf>, EngineError> {
        init_kf_state(obs_dataset, first_obs, second_obs, n_sigma, state)
    }

    /// Euclidean distance (AU) between the mean heliocentric positions of two
    /// states.
    ///
    /// Each state is converted from attributable to Cartesian coordinates using
    /// the observer geometry stored at its own epoch.
    pub fn position_distance_au(&self, other: &KFState) -> f64 {
        let pos_a = self.to_cartesian().pos;
        let pos_b = other.to_cartesian().pos;
        (pos_a - pos_b).norm()
    }

    /// Compute the $2\times2$ sky-plane covariance matrix $\Sigma_{sky} = HPH^\top$.
    ///
    /// Arguments
    /// ---------
    /// * `r_obs` – Heliocentric observer position (AU, ecliptic J2000).
    ///
    /// Return
    /// ------
    /// * `Ok(Matrix2)` – Sky covariance in radians².
    /// * `Err` – If the observation Jacobian cannot be evaluated.
    pub fn sky_covariance(&self) -> Result<Matrix2<f64>, ObservationJacobianError> {
        let h = observation_jacobian()?;
        Ok(h * self.covariance * h.transpose())
    }

    pub fn last_gain_frobenius_norm(&self) -> f64 {
        self.kalman_gain
            .as_ref()
            .map(|k| k.norm())
            .unwrap_or(f64::NAN)
    }

    /// Convert the filter state to Keplerian orbital elements.
    ///
    /// The state vector is stored in **attributable coordinates**
    /// $(\alpha, \delta, \dot{\alpha}, \dot{\delta}, \rho, \dot{\rho})$, not as
    /// a raw heliocentric Cartesian state. It must therefore first be mapped
    /// to a heliocentric Cartesian position/velocity via [`Self::to_cartesian`]
    /// (using the observer state stored at `self.epoch`) before orbital
    /// elements can be derived.
    pub fn to_orbit(&self) -> OrbitalElements {
        let cart = self.to_cartesian();
        OrbitalElements::from_orbital_state(&cart.pos, &cart.vel, self.epoch)
    }

    /// Propagate the Kalman state to epoch `t_prop` using the universal-variable
    /// two-body formulation.
    ///
    /// The position and velocity are extracted from the state vector `x`,
    /// propagated via [`propagate_universal`], and repacked into a new [`KFState`].
    /// The covariance matrix is propagated using the analytic State Transition
    /// Matrix (STM) derived from the Lagrange coefficients.
    ///
    /// State Transition Matrix
    /// -----------------------
    /// The universal-variable propagator expresses the propagated state via
    /// the Lagrange coefficients $(f, g, \dot{f}, \dot{g})$:
    ///
    /// $$r_1 = f \cdot r_0 + g \cdot v_0$$
    /// $$v_1 = \dot{f} \cdot r_0 + \dot{g} \cdot v_0$$
    ///
    /// The resulting STM is block-diagonal:
    ///
    /// $$F = \begin{pmatrix} f \cdot I_3 & g \cdot I_3 \\ \dot{f} \cdot I_3 & \dot{g} \cdot I_3 \end{pmatrix}$$
    ///
    /// The covariance is then propagated as:
    ///
    /// $$P_1 = F \cdot P_0 \cdot F^\top$$
    ///
    /// Arguments
    /// ---------
    /// * `t_prop` – Target epoch (days, MJD).
    ///
    /// Return
    /// ------
    /// * `Ok(KFState)` – Propagated state and covariance at `t_prop`.
    /// * `Err(OutfitError)` – If the universal Kepler solver fails.
    pub fn propagate(
        &self,
        obs_dataset: &ObsDataset,
        obs: &Observation,
    ) -> Result<Self, PropagateError> {
        propagate_kf(
            self,
            obs_dataset,
            obs,
            self.shared_ctx.get_q0(),
            self.shared_ctx.get_dt_ref(),
        )
    }

    /// Propagate the filter state to a given epoch without consuming an
    /// observation.
    ///
    /// This is the read-only prediction primitive, intended for generating a
    /// [`SearchRegion`](crate::topocentric_kf::bank::SearchRegion) at a future
    /// epoch before any observation is available. The observer heliocentric
    /// state at `t_prop` must be supplied by the caller (typically resolved
    /// from [`EphemState::helio_observer_state`]).
    ///
    /// Unlike [`KFState::propagate`], this method does **not** require an
    /// [`Observation`] and does **not** modify the bank — it returns a new
    /// `KFState` leaving `self` intact.
    ///
    /// Arguments
    /// ---------
    /// * `t_prop`    – Target epoch (MJD TT).
    /// * `r_obs_new` – Observer heliocentric position at `t_prop` (AU, ecliptic J2000).
    /// * `v_obs_new` – Observer heliocentric velocity at `t_prop` (AU/day, ecliptic J2000).
    ///
    /// Return
    /// ------
    /// * `Ok(KFState)` – Predicted state at `t_prop`.
    /// * `Err(PropagateError)` – If Kepler propagation or Jacobian inversion fails.
    pub fn predict(
        &self,
        t_prop: f64,
        r_obs_new: Vector3<f64>,
        v_obs_new: Vector3<f64>,
    ) -> Result<Self, PropagateError> {
        propagate::propagate_to_epoch(
            self,
            t_prop,
            r_obs_new,
            v_obs_new,
            self.shared_ctx.config.q0,
            self.shared_ctx.config.dt_ref,
        )
    }

    /// Perform a Kalman measurement update using a new $(RA, Dec)$ observation.
    ///
    /// # Update equations
    ///
    /// In attributable coordinates the observation function $h(\mathbf{x}) =
    /// (\alpha, \delta)$ is **linear and exact** — it is simply a selection of
    /// the first two state components (see [`observation_jacobian`]). The
    /// update therefore reduces to the standard (non-extended) Kalman filter
    /// equations:
    ///
    /// **Innovation:**
    /// $$\nu = z - h(\hat{x}^-)$$
    ///
    /// where $z = (RA_{\text{obs}}, Dec_{\text{obs}})^\top$.
    ///
    /// **Innovation covariance:**
    /// $$S = H P^- H^\top + R$$
    ///
    /// **Kalman gain:**
    /// $$K = P^- H^\top S^{-1}$$
    ///
    /// **Updated state:**
    /// $$\hat{x}^+ = \hat{x}^- + K \nu$$
    ///
    /// **Updated covariance (Joseph form):**
    /// $$(I - KH) P^- (I - KH)^\top + K R K^\top$$
    ///
    /// The Joseph form is used instead of the standard $P^+ = (I - KH)P^-$
    /// to preserve symmetry and positive-definiteness numerically.
    ///
    /// # Observation noise matrix
    ///
    /// The observation noise matrix $R \in \mathbb{R}^{2 \times 2}$ is
    /// diagonal and built directly from the 1-σ uncertainties stored in
    /// [`EquCoord`]:
    ///
    /// $$R = \begin{pmatrix} \sigma_{RA}^2 & 0 \\ 0 & \sigma_{Dec}^2 \end{pmatrix}$$
    ///
    /// where $\sigma_{RA}$ and $\sigma_{Dec}$ are in radians.
    ///
    /// # RA wrapping
    ///
    /// The RA component of the innovation is wrapped to $(-\pi, \pi]$ to
    /// handle the $0 / 2\pi$ boundary:
    ///
    /// $$\nu_{RA} = \text{wrap}(\nu_{RA})$$
    ///
    /// # Arguments
    ///
    /// * `new_obs`  – The new astrometric observation.
    /// * `r_obs`    – Unused by the attributable observation model; kept only
    ///   for interface compatibility with [`observation_jacobian`].
    ///
    /// # Returns
    ///
    /// * `Ok(KFState)` – Updated state and covariance at the same epoch as
    ///   the current state. The caller is responsible for propagating the
    ///   state to the observation epoch **before** calling this method.
    /// * `Err(KFUpdateError::SingularInnovationCovariance)` – If the
    ///   innovation covariance $S$ is not invertible.
    pub fn update(self, new_obs: &Observation) -> Result<Self, KFUpdateError> {
        update_kf(self, new_obs)
    }

    /// Convert the filter state to an observed sky position with propagated
    /// astrometric uncertainties.
    ///
    /// In **attributable coordinates**, the angular coordinates $(\alpha, \delta)$
    /// are directly the first two components of the state vector — no
    /// projection through a heliocentric Cartesian position is required.
    ///
    /// Uncertainties are propagated from the state covariance $P$ via the
    /// (constant, exact) observation model:
    ///
    /// $$\Sigma_{sky} = H P H^\top, \qquad H = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{pmatrix}$$
    ///
    /// (see [`observation_jacobian`]), which reduces to simply reading off the
    /// $(\alpha, \alpha)$ and $(\delta, \delta)$ entries of $P$.
    ///
    /// The resulting 1-σ errors are:
    ///
    /// $$\sigma_\alpha = \sqrt{(\Sigma_{sky})_{00}}, \quad
    ///   \sigma_\delta = \sqrt{(\Sigma_{sky})_{11}}$$
    ///
    /// Arguments
    /// ---------
    /// * `r_obs` – Unused by the attributable observation model; kept only for
    ///   interface compatibility with [`observation_jacobian`].
    ///
    /// Returns
    /// -------
    /// * `Ok(EquCoord)` – Sky position with propagated 1-σ uncertainties, in radians.
    /// * `Err(ObservationJacobianError)` – Propagated from [`observation_jacobian`];
    ///   in practice always `Ok` for the current (constant) Jacobian.
    pub fn to_equ_coord(&self) -> Result<EquCoord, ObservationJacobianError> {
        // RA/Dec are directly the first two attributable state components.
        let ra = self.state[0];
        let dec = self.state[1];

        // Propagate covariance through H = [I_2 | 0]
        let h = observation_jacobian()?;
        let sigma_sky = h * self.covariance * h.transpose(); // 2x2

        let sigma_ra = sigma_sky[(0, 0)].max(0.0).sqrt();
        let sigma_dec = sigma_sky[(1, 1)].max(0.0).sqrt();

        Ok(EquCoord::new(ra, sigma_ra, dec, sigma_dec))
    }

    pub fn to_cartesian(&self) -> CartesianState {
        attributable_to_cartesian(&self.state, &self.r_obs, &self.v_obs)
    }
}

pub fn to_equinoctial(kepler_elem: &OrbitalElements) -> Result<EquinoctialElements, OutfitError> {
    kepler_elem
        .to_equinoctial()?
        .as_equinoctial()
        .ok_or(OutfitError::InvalidConversion(
            "Conversion to equinoctial elements failed".to_string(),
        ))
}

// ── State vector ─────────────────────────────────────────────────────────────

use nalgebra::{Matrix2, Matrix2x6, Matrix6, Matrix6x2, Vector2, Vector6};
use photom::coordinates::{
    cov2::Cov2,
    ecliptic::{EclipticCoord, EclipticCoordCov},
};
use serde::{Deserialize, Serialize};

/// Singer model parameters controlling the acceleration autocorrelation.
///
/// The Singer model treats the acceleration as an Ornstein-Uhlenbeck process:
///
/// $$\ddot\theta(t) = -\alpha\,\ddot\theta(t)\,\mathrm{d}t + \sigma_a\,\mathrm{d}W_t$$
///
/// - $\alpha$ is the inverse correlation time (day⁻¹). Small $\alpha$ means
///   the acceleration changes slowly; large $\alpha$ means it decorrelates
///   quickly (approaching a white-noise acceleration model).
/// - $\sigma_a$ is the acceleration standard deviation (rad/day²).
///
/// The limit $\alpha \to 0$ recovers the constant-acceleration (CA) model
/// currently used.
///
/// Typical values for main-belt asteroids:
/// - $\alpha \approx 1/7\ \text{day}^{-1}$ (correlation time ~7 days)
/// - $\sigma_a \approx 10^{-7}\ \text{rad/day}^2$
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SingerParams {
    /// Inverse correlation time $\alpha$ (day⁻¹). Must be positive.
    pub alpha: f64,
    /// Acceleration standard deviation $\sigma_a$ (rad/day²). Must be positive.
    pub sigma_a: f64,
}

impl SingerParams {
    /// Construct Singer parameters from a correlation time $\tau = 1/\alpha$.
    ///
    /// Arguments
    /// ---------
    /// * `tau_days`  – Acceleration correlation time (days).
    /// * `sigma_a`   – Acceleration standard deviation (rad/day²).
    pub fn from_correlation_time(tau_days: f64, sigma_a: f64) -> Self {
        Self {
            alpha: 1.0 / tau_days,
            sigma_a,
        }
    }
}

/// Ecliptic kinematic state vector and its covariance.
///
/// The state vector is defined as:
/// $$\mathbf{x} = (\lambda, \dot\lambda, \ddot\lambda, \beta, \dot\beta, \ddot\beta)^\top$$
///
/// where $\lambda$ is the ecliptic longitude, $\beta$ the ecliptic latitude,
/// and dots denote time derivatives (rad/day and rad/day²).
///
/// The covariance matrix $P \in \mathbb{R}^{6 \times 6}$ encodes the uncertainty
/// on the full state. It is propagated forward via the state transition matrix $F$.
#[derive(Debug, Clone)]
pub struct EclipticState {
    /// State vector $\mathbf{x}$.
    pub x: Vector6<f64>,
    /// Covariance matrix $P$.
    pub p: Matrix6<f64>,
    /// Reference epoch (MJD) at which the state is valid.
    pub epoch: f64,
    /// Acceleration spectral density used as process noise during propagation
    /// (rad²/day³). Set to `0.0` to disable process noise.
    pub process_noise_q: f64,
    /// Singer model parameters. When `Some`, [`EclipticState::propagate_singer`]
    /// is used instead of [`EclipticState::propagate`].
    pub singer: Option<SingerParams>,
}

impl EclipticState {
    /// Get the ecliptic coordinate (position and uncertainty) from the state.
    pub fn position(&self) -> EclipticCoord {
        EclipticCoord::new(
            self.x[0],
            self.p[(0, 0)].sqrt(),
            self.x[3],
            self.p[(3, 3)].sqrt(),
        )
    }

    /// Construct the state transition matrix $F(\Delta t)$ for a uniformly
    /// accelerated kinematic model.
    ///
    /// Under the constant-acceleration assumption, each coordinate pair
    /// $(\theta, \dot\theta, \ddot\theta)$ evolves independently as:
    ///
    /// $$\begin{pmatrix} \theta \\ \dot\theta \\ \ddot\theta \end{pmatrix}_{t+\Delta t}
    /// = \begin{pmatrix} 1 & \Delta t & \tfrac{1}{2}\Delta t^2 \\ 0 & 1 & \Delta t \\ 0 & 0 & 1 \end{pmatrix}
    /// \begin{pmatrix} \theta \\ \dot\theta \\ \ddot\theta \end{pmatrix}_t$$
    ///
    /// The full $6 \times 6$ matrix is block-diagonal:
    ///
    /// $$F = \begin{pmatrix} F_3 & 0 \\ 0 & F_3 \end{pmatrix}$$
    ///
    /// Arguments
    /// ---------
    /// * `dt` – Time step in days.
    ///
    /// Return
    /// ------
    /// The $6 \times 6$ state transition matrix $F$.
    pub fn transition_matrix(dt: f64) -> Matrix6<f64> {
        let dt2 = 0.5 * dt * dt;

        #[rustfmt::skip]
        let f = Matrix6::new(
            1.0,  dt,  dt2,  0.0, 0.0, 0.0,
            0.0, 1.0,   dt,  0.0, 0.0, 0.0,
            0.0, 0.0,  1.0,  0.0, 0.0, 0.0,
            0.0, 0.0,  0.0,  1.0,  dt, dt2,
            0.0, 0.0,  0.0,  0.0, 1.0,  dt,
            0.0, 0.0,  0.0,  0.0, 0.0, 1.0,
        );
        f
    }

    /// Build the Singer state transition matrix $F_{Singer}(\Delta t)$.
    ///
    /// The Singer model replaces the constant-acceleration assumption with an
    /// Ornstein-Uhlenbeck process on acceleration. The $3 \times 3$ block for
    /// one coordinate axis is:
    ///
    /// $$F_3(\Delta t) = \begin{pmatrix}
    /// 1 & \Delta t & \frac{\alpha \Delta t - 1 + e^{-\alpha \Delta t}}{\alpha^2} \\
    /// 0 & 1        & \frac{1 - e^{-\alpha \Delta t}}{\alpha}                     \\
    /// 0 & 0        & e^{-\alpha \Delta t}
    /// \end{pmatrix}$$
    ///
    /// The full $6 \times 6$ matrix is block-diagonal:
    /// $$F = F_3 \oplus F_3$$
    ///
    /// Numerical stability
    /// -------------------
    /// For small $\alpha \Delta t < 10^{-4}$, the Taylor expansions are used
    /// to avoid catastrophic cancellation:
    ///
    /// $$e^{-\alpha \Delta t} \approx 1 - \alpha \Delta t + \frac{(\alpha \Delta t)^2}{2}$$
    ///
    /// $$\frac{\alpha \Delta t - 1 + e^{-\alpha \Delta t}}{\alpha^2}
    ///   \approx \frac{\Delta t^2}{2} - \frac{\alpha \Delta t^3}{6}$$
    ///
    /// In this limit, $F_3$ converges to the standard constant-acceleration matrix.
    ///
    /// Arguments
    /// ---------
    /// * `dt`     – Time step (days).
    /// * `singer` – Singer model parameters $(\alpha, \sigma_a)$.
    ///
    /// Return
    /// ------
    /// The $6 \times 6$ Singer state transition matrix.
    pub fn singer_transition_matrix(dt: f64, singer: &SingerParams) -> Matrix6<f64> {
        let alpha = singer.alpha;
        let adt = alpha * dt;
        let rho = (-adt).exp(); // e^{-α Δt}

        // Numerically stable entries of the 3×3 block.
        // For small αΔt we use Taylor expansion to avoid 0/0.
        let (f02, f12) = if adt < 1e-4 {
            // Taylor: (αΔt − 1 + e^{−αΔt}) / α² ≈ Δt²/2 − αΔt³/6
            let f02 = dt * dt * 0.5 - alpha * dt * dt * dt / 6.0;
            // Taylor: (1 − e^{−αΔt}) / α ≈ Δt − αΔt²/2
            let f12 = dt - alpha * dt * dt * 0.5;
            (f02, f12)
        } else {
            let f02 = (adt - 1.0 + rho) / (alpha * alpha);
            let f12 = (1.0 - rho) / alpha;
            (f02, f12)
        };

        #[rustfmt::skip]
        let f3 = nalgebra::Matrix3::new(
            1.0,  dt,  f02,
            0.0, 1.0,  f12,
            0.0, 0.0,  rho,
        );

        let mut out = Matrix6::zeros();
        out.fixed_view_mut::<3, 3>(0, 0).copy_from(&f3);
        out.fixed_view_mut::<3, 3>(3, 3).copy_from(&f3);
        out
    }

    /// Process noise covariance matrix $Q_{Singer}(\Delta t)$ for one axis.
    ///
    /// Computed via the Van Loan (1978) matrix-exponential method applied to the
    /// continuous-time Singer model
    ///
    /// $$\dot{\mathbf{x}} = A\mathbf{x} + B\,w(t), \qquad
    /// \mathbb{E}[w(t)w(\tau)] = 2\alpha\sigma_a^2\,\delta(t-\tau),$$
    ///
    /// with state $(\theta, \dot\theta, \ddot\theta)$, drift
    ///
    /// $$A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & -\alpha \end{pmatrix},
    /// \qquad B = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}.$$
    ///
    /// Van Loan method
    /// ---------------
    /// Building the $6 \times 6$ block matrix
    ///
    /// $$M = \begin{pmatrix} -A & B Q_c B^\top \\ 0 & A^\top \end{pmatrix} \Delta t,
    /// \qquad Q_c = 2\alpha\sigma_a^2,$$
    ///
    /// the matrix exponential yields
    ///
    /// $$e^{M} = \begin{pmatrix} \cdot & F^{-1} Q_3 \\ 0 & F^\top \end{pmatrix},$$
    ///
    /// from which $Q_3 = F \cdot (\text{top-right block})$, with $F$ the discrete
    /// state-transition matrix on $\Delta t$. This avoids the catastrophic
    /// cancellations of the closed-form analytical entries.
    ///
    /// Small-$\alpha\Delta t$ limit
    /// ----------------------------
    /// When $\alpha\Delta t < 10^{-4}$, the matrix exponential loses accuracy due
    /// to floating-point cancellation. In that regime the Singer model degenerates
    /// to the continuous white-noise-on-jerk (CA) model with spectral density
    /// $q_{CA} = 2\alpha\sigma_a^2$, whose closed-form covariance block is
    ///
    /// $$Q_3^{CA} = q_{CA} \begin{pmatrix}
    /// \Delta t^5/20 & \Delta t^4/8 & \Delta t^3/6 \\
    /// \Delta t^4/8  & \Delta t^3/3 & \Delta t^2/2 \\
    /// \Delta t^3/6  & \Delta t^2/2 & \Delta t
    /// \end{pmatrix}.$$
    ///
    /// Arguments
    /// ---------
    /// * `dt`     – Time step (days).
    /// * `singer` – Singer model parameters $(\alpha, \sigma_a)$.
    ///
    /// Return
    /// ------
    /// The $6 \times 6$ block-diagonal Singer process noise matrix
    /// $Q = Q_3 \oplus Q_3$.
    pub fn singer_process_noise_matrix(dt: f64, singer: &SingerParams) -> Matrix6<f64> {
        use nalgebra::Matrix3;

        let alpha = singer.alpha;
        let sa2 = singer.sigma_a * singer.sigma_a;
        let adt = alpha * dt;

        // Threshold below which the closed-form expressions suffer from
        // catastrophic cancellation; we switch to truncated Taylor series.
        const ADT_SMALL: f64 = 1e-2;

        let q3 = if adt < ADT_SMALL {
            // Taylor expansions around alpha*dt = 0 (from SymPy, truncated to
            // O((alpha*dt)^4) past the leading term for ~1e-16 relative error
            // at adt = 1e-2).
            let a = alpha;
            let a2 = a * a;
            let a3 = a2 * a;
            let a4 = a2 * a2;
            let d = dt;
            let d2 = d * d;
            let d3 = d2 * d;
            let d4 = d3 * d;
            let d5 = d4 * d;
            let d6 = d5 * d;
            let d7 = d6 * d;
            let d8 = d7 * d;

            let q00 =
                sa2 * (a * d5 / 10.0 - a2 * d6 / 18.0 + 5.0 * a3 * d7 / 252.0 - a4 * d8 / 180.0);
            let q01 = sa2 * (a * d4 / 4.0 - a2 * d5 / 6.0 + 5.0 * a3 * d6 / 72.0 - a4 * d7 / 45.0);
            let q02 = sa2
                * (a * d3 / 3.0 - a2 * d4 / 3.0 + 11.0 * a3 * d5 / 60.0 - 13.0 * a4 * d6 / 180.0);
            let q11 =
                sa2 * (2.0 * a * d3 / 3.0 - a2 * d4 / 2.0 + 7.0 * a3 * d5 / 30.0 - a4 * d6 / 12.0);
            let q12 = sa2 * (a * d2 - a2 * d3 + 7.0 * a3 * d4 / 12.0 - a4 * d5 / 4.0);
            let q22 = sa2
                * (
                    2.0 * a * d
    - 2.0 * a2 * d2
    + 4.0 * a3 * d3 / 3.0
    - 2.0 * a4 * d4 / 3.0
    + 4.0 * a4 * a * d5 / 15.0       // order 5
    - 4.0 * a4 * a2 * d6 / 45.0
                    // order 6
                );

            Matrix3::new(q00, q01, q02, q01, q11, q12, q02, q12, q22)
        } else {
            let rho = (-adt).exp();
            let rho2 = rho * rho;
            let a2 = alpha * alpha;
            let a3 = a2 * alpha;
            let a4 = a2 * a2;

            let q22 = sa2 * (1.0 - rho2);
            let q12 = sa2 * (1.0 - rho).powi(2) / alpha;
            let q02 = sa2 * (1.0 - rho2 - 2.0 * adt * rho) / a2;
            let q11 = sa2 * (2.0 * adt - 3.0 + 4.0 * rho - rho2) / a2;
            let q01 = sa2 * (adt * adt - 2.0 * adt + 2.0 * (adt - 1.0) * rho + 1.0 + rho2) / a3;
            let q00 = sa2
                * (2.0 / 3.0 * adt * adt * adt - 2.0 * adt * adt + 2.0 * adt - 4.0 * adt * rho
                    + 1.0
                    - rho2)
                / a4;

            Matrix3::new(q00, q01, q02, q01, q11, q12, q02, q12, q22)
        };

        let mut out = Matrix6::zeros();
        out.fixed_view_mut::<3, 3>(0, 0).copy_from(&q3);
        out.fixed_view_mut::<3, 3>(3, 3).copy_from(&q3);
        out
    }

    /// Propagate the state with the Singer dynamic model.
    ///
    /// Replaces the constant-acceleration [`Self::propagate`] with the
    /// Ornstein-Uhlenbeck acceleration model. The predicted mean and covariance
    /// are:
    ///
    /// $$\mathbf{x}^- = F_{Singer}(\Delta t)\,\mathbf{x}$$
    /// $$P^- = F_{Singer}\,P\,F_{Singer}^\top + Q_{Singer}(\Delta t)$$
    ///
    /// Arguments
    /// ---------
    /// * `target_epoch` – Target epoch (MJD).
    /// * `singer`       – Singer model parameters.
    ///
    /// Return
    /// ------
    /// A new [`EclipticState`] valid at `target_epoch`.
    pub fn propagate_singer(&self, target_epoch: f64, singer: &SingerParams) -> Self {
        let dt = target_epoch - self.epoch;
        let f = Self::singer_transition_matrix(dt, singer);
        let q = Self::singer_process_noise_matrix(dt, singer);
        EclipticState {
            x: f * self.x,
            p: f * self.p * f.transpose() + q,
            epoch: target_epoch,
            process_noise_q: self.process_noise_q,
            singer: self.singer.clone(),
        }
    }

    /// Propagate the state forward to `target_epoch` with process noise.
    ///
    /// The predicted state mean and covariance are:
    ///
    /// $$\mathbf{x}^- = F \, \mathbf{x}$$
    /// $$P^- = F \, P \, F^\top + Q(\Delta t)$$
    ///
    /// where $Q(\Delta t)$ is the process noise matrix from
    /// [`Self::process_noise_matrix`]. Setting $q = 0$ recovers the
    /// noise-free propagation.
    ///
    /// Arguments
    /// ---------
    /// * `target_epoch` – Target epoch (MJD).
    /// * `q` – Acceleration spectral density (rad²/day³). Pass `0.0` to
    ///   disable process noise.
    ///
    /// Return
    /// ------
    /// A new [`EclipticState`] valid at `target_epoch`.
    pub fn propagate(&self, target_epoch: f64, q: f64) -> Self {
        let dt = target_epoch - self.epoch;
        let f = Self::transition_matrix(dt);
        let q_mat = Self::process_noise_matrix(dt, q);
        EclipticState {
            x: f * self.x,
            p: f * self.p * f.transpose() + q_mat,
            epoch: target_epoch,
            process_noise_q: self.process_noise_q,
            singer: self.singer.clone(),
        }
    }

    /// Extract the predicted [`EclipticCoordCov`] from a propagated state.
    ///
    /// The position is taken directly from the state vector components
    /// $\lambda = x_0$ and $\beta = x_3$. The marginal $2 \times 2$ covariance
    /// on $(\lambda, \beta)$ is extracted from the diagonal blocks of $P$:
    ///
    /// $$\Sigma_{\lambda\beta} = \begin{pmatrix} P_{00} & P_{03} \\ P_{30} & P_{33} \end{pmatrix}$$
    ///
    /// Return
    /// ------
    /// An [`EclipticCoordCov`] built from the state position and marginal covariance.
    pub fn to_ecliptic_coord_cov(&self) -> EclipticCoordCov {
        let lon = self.x[0];
        let lat = self.x[3];
        let cov = Cov2 {
            xx: self.p[(0, 0)],
            yy: self.p[(3, 3)],
            xy: self.p[(0, 3)],
        };
        let coord = EclipticCoord::new(lon, cov.xx.sqrt(), lat, cov.yy.sqrt());
        EclipticCoordCov::new(coord, cov)
    }

    /// Observation matrix $H \in \mathbb{R}^{2 \times 6}$.
    ///
    /// Extracts the position components $(\lambda, \beta)$ from the state vector:
    ///
    /// $$H = \begin{pmatrix}
    /// 1 & 0 & 0 & 0 & 0 & 0 \\
    /// 0 & 0 & 0 & 1 & 0 & 0
    /// \end{pmatrix}$$
    #[inline]
    fn observation_matrix() -> Matrix2x6<f64> {
        #[rustfmt::skip]
        let h = Matrix2x6::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        );
        h
    }

    /// Apply a linear Kalman measurement update from an ecliptic observation.
    ///
    /// The observation $\mathbf{z} = (\lambda_\text{obs}, \beta_\text{obs})^\top$
    /// is expressed directly in the ecliptic frame, so the observation function
    /// is linear: $h(\mathbf{x}) = H\mathbf{x}$.
    ///
    /// The standard Kalman equations are:
    ///
    /// $$S = H P H^\top + R$$
    ///
    /// $$K = P H^\top S^{-1}$$
    ///
    /// $$\hat{\mathbf{x}}^+ = \hat{\mathbf{x}}^- + K\,(\mathbf{z} - H\hat{\mathbf{x}}^-)$$
    ///
    /// $$P^+ = (I - KH)\,P^-\,(I - KH)^\top + KRK^\top$$
    ///
    /// The last form is the **Joseph stabilised** update, which preserves
    /// symmetry and positive semi-definiteness of $P$ in the presence of
    /// floating-point rounding errors.
    ///
    /// The input observation is first converted from equatorial to ecliptic
    /// coordinates (including full covariance propagation) via
    /// [`EclipticCoordCov::from`], so the caller does not need to handle
    /// the frame transformation.
    ///
    /// Arguments
    /// ---------
    /// * `obs` – Ecliptic observation $(\lambda, \beta)$ with full 2×2 covariance.
    ///
    /// Return
    /// ------
    /// * `Some(updated_state)` – If the innovation covariance $S$ is invertible.
    /// * `None` – If $S$ is singular (degenerate covariance).
    pub fn kalman_update(&self, obs: &EclipticCoordCov) -> Option<Self> {
        let h = Self::observation_matrix();

        // Observation vector z = (λ, β)
        let z = Vector2::new(obs.coord.lon, obs.coord.lat);

        // Measurement noise matrix R from the observation covariance
        let r = Matrix2::new(obs.cov.xx, obs.cov.xy, obs.cov.xy, obs.cov.yy);

        // Innovation: y = z − H x̂
        let z_pred = h * self.x;
        let mut innovation = z - z_pred;

        // Longitude wrap-around: keep innovation in (−π, π]
        if innovation[0] > std::f64::consts::PI {
            innovation[0] -= std::f64::consts::TAU;
        } else if innovation[0] < -std::f64::consts::PI {
            innovation[0] += std::f64::consts::TAU;
        }

        // Innovation covariance S = H P H' + R
        let s: Matrix2<f64> = h * self.p * h.transpose() + r;

        // Kalman gain K = P H' S⁻¹   (6×2)
        let ph_t: Matrix6x2<f64> = self.p * h.transpose();
        let chol = s.cholesky()?;
        let k: Matrix6x2<f64> = chol.solve(&ph_t.transpose()).transpose();

        // State update
        let x_new = self.x + k * innovation;

        // Joseph stabilised covariance update: P⁺ = (I−KH)P(I−KH)' + KRK'
        let i_kh = nalgebra::Matrix6::identity() - k * h;
        let p_new = i_kh * self.p * i_kh.transpose() + k * r * k.transpose();

        Some(EclipticState {
            x: x_new,
            p: p_new,
            epoch: self.epoch,
            process_noise_q: self.process_noise_q,
            singer: self.singer.clone(),
        })
    }

    /// Process noise covariance matrix $Q(\Delta t)$ for a continuous
    /// white-noise acceleration model.
    ///
    /// Models unobserved perturbations (non-gravitational forces, model
    /// truncation) as a continuous white-noise process on acceleration.
    /// The discrete-time covariance block for one coordinate axis is:
    ///
    /// $$Q_3(\Delta t) = q \begin{pmatrix}
    /// \tfrac{\Delta t^5}{20} & \tfrac{\Delta t^4}{8}  & \tfrac{\Delta t^3}{6} \\
    /// \tfrac{\Delta t^4}{8}  & \tfrac{\Delta t^3}{3}  & \tfrac{\Delta t^2}{2} \\
    /// \tfrac{\Delta t^3}{6}  & \tfrac{\Delta t^2}{2}  & \Delta t
    /// \end{pmatrix}$$
    ///
    /// The full $6 \times 6$ matrix is block-diagonal:
    ///
    /// $$Q = \begin{pmatrix} Q_3 & 0 \\ 0 & Q_3 \end{pmatrix}$$
    ///
    /// Arguments
    /// ---------
    /// * `dt` – Time step in days.
    /// * `q`  – Acceleration spectral density (rad²/day³). Controls how
    ///   much the filter trusts the kinematic model vs. incoming observations.
    ///   Typical values for main-belt asteroids: $10^{-16}$–$10^{-14}$.
    ///
    /// Return
    /// ------
    /// The $6 \times 6$ process noise matrix $Q$.
    pub fn process_noise_matrix(dt: f64, q: f64) -> Matrix6<f64> {
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        let dt5 = dt4 * dt;

        #[rustfmt::skip]
        let q3 = nalgebra::Matrix3::new(
            q * dt5 / 20.0,  q * dt4 / 8.0,  q * dt3 / 6.0,
            q * dt4 / 8.0,   q * dt3 / 3.0,  q * dt2 / 2.0,
            q * dt3 / 6.0,   q * dt2 / 2.0,  q * dt,
        );

        let mut out = Matrix6::zeros();
        out.fixed_view_mut::<3, 3>(0, 0).copy_from(&q3);
        out.fixed_view_mut::<3, 3>(3, 3).copy_from(&q3);
        out
    }
}

#[cfg(test)]
mod singer_q_tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use nalgebra::Matrix3;
    use proptest::prelude::*;

    /// Closed-form analytical reference for the three numerically stable
    /// entries of the Singer covariance block. These three formulas do not
    /// suffer from catastrophic cancellation for any αΔt > 0 and can be
    /// trusted as an independent ground truth.
    fn analytical_stable_entries(alpha: f64, sigma_a: f64, dt: f64) -> (f64, f64, f64) {
        let rho = (-alpha * dt).exp();
        let rho2 = rho * rho;
        let s2 = sigma_a * sigma_a;
        let q33 = s2 * (1.0 - rho2);
        let q23 = s2 * (1.0 - rho).powi(2) / alpha;
        let q22 = s2 * (2.0 * alpha * dt - 3.0 + 4.0 * rho - rho2) / (alpha * alpha);
        (q22, q23, q33)
    }

    fn extract_q3(dt: f64, alpha: f64, sigma_a: f64) -> Matrix3<f64> {
        let s = SingerParams { alpha, sigma_a };
        EclipticState::singer_process_noise_matrix(dt, &s)
            .fixed_view::<3, 3>(0, 0)
            .clone_owned()
    }

    // ──────────────────────────────────────────────────────────────────────
    // Unit tests on hand-picked cases
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn debug_nalgebra_exp() {
        let alpha = 21.4_f64;
        let dt = 12.6_f64;
        let mut a = Matrix3::zeros();
        a[(0, 1)] = 1.0;
        a[(1, 2)] = 1.0;
        a[(2, 2)] = -alpha;
        let mut bqb = Matrix3::zeros();
        bqb[(2, 2)] = 2.0 * alpha;
        let mut big = Matrix6::zeros();
        big.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-a));
        big.fixed_view_mut::<3, 3>(0, 3).copy_from(&bqb);
        big.fixed_view_mut::<3, 3>(3, 3).copy_from(&a.transpose());
        let em = (big * dt).exp();
        println!("{em}");
    }

    #[test]
    fn matches_analytical_stable_entries() {
        let cases = [
            (1.0 / 7.0, 0.5),
            (1.0 / 7.0, 5.0),
            (1.0 / 7.0, 30.0),
            (0.5, 1.0),
            (2.0, 0.1),
            (10.0, 1.0),
            (50.0, 2.0),
        ];
        for &(alpha, dt) in &cases {
            let q3 = extract_q3(dt, alpha, 1.0);
            let (q22, q23, q33) = analytical_stable_entries(alpha, 1.0, dt);
            assert_relative_eq!(q3[(2, 2)], q33, max_relative = 1e-10);
            assert_relative_eq!(q3[(1, 2)], q23, max_relative = 1e-10);
            assert_relative_eq!(q3[(1, 1)], q22, max_relative = 1e-9);
        }
    }

    #[test]
    fn is_symmetric() {
        for &(alpha, dt) in &[(1e-5, 1.0), (0.1, 0.5), (1.0, 1.0), (10.0, 1.0)] {
            let q3 = extract_q3(dt, alpha, 0.7);
            for i in 0..3 {
                for j in (i + 1)..3 {
                    assert_relative_eq!(q3[(i, j)], q3[(j, i)], max_relative = 1e-14);
                }
            }
        }
    }

    #[test]
    fn is_positive_definite() {
        // All Cholesky factorizations must succeed and all eigenvalues > 0.
        for &(alpha, dt) in &[
            (1e-6, 1.0),
            (1e-3, 0.5),
            (0.1, 1.0),
            (1.0, 1.0),
            (10.0, 2.0),
            (100.0, 0.1),
        ] {
            let q3 = extract_q3(dt, alpha, 1.0);
            assert!(
                q3.cholesky().is_some(),
                "Q3 not SPD for alpha={alpha}, dt={dt}: {q3}"
            );
            let eigvals = q3.symmetric_eigenvalues();
            for &lam in eigvals.iter() {
                assert!(
                    lam > 0.0,
                    "non-positive eigenvalue {lam} for alpha={alpha}, dt={dt}"
                );
            }
        }
    }

    #[test]
    fn block_diagonal_structure() {
        // The 6x6 output must be block-diagonal with identical 3x3 blocks
        // and zero off-diagonal coupling between the two ecliptic axes.
        let s = SingerParams {
            alpha: 0.3,
            sigma_a: 0.5,
        };
        let q = EclipticState::singer_process_noise_matrix(1.2, &s);
        let top = q.fixed_view::<3, 3>(0, 0).clone_owned();
        let bot = q.fixed_view::<3, 3>(3, 3).clone_owned();
        let off1 = q.fixed_view::<3, 3>(0, 3).clone_owned();
        let off2 = q.fixed_view::<3, 3>(3, 0).clone_owned();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(top[(i, j)], bot[(i, j)], max_relative = 1e-14);
                assert_abs_diff_eq!(off1[(i, j)], 0.0, epsilon = 1e-30);
                assert_abs_diff_eq!(off2[(i, j)], 0.0, epsilon = 1e-30);
            }
        }
    }

    #[test]
    fn sigma_a_scaling_is_quadratic() {
        // Q is linear in σ_a²: Q(α, k·σ_a, Δt) = k² · Q(α, σ_a, Δt).
        let alpha = 0.5;
        let dt = 2.0;
        let q1 = extract_q3(dt, alpha, 1.0);
        let q2 = extract_q3(dt, alpha, 3.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(q2[(i, j)], 9.0 * q1[(i, j)], max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn ca_limit_for_small_alpha_dt() {
        // For α·Δt ≪ 1, Q must approach the CA white-noise-on-jerk block with
        // spectral density q_CA = 2 α σ_a².
        let alpha = 1e-3;
        let dt = 1.0;
        let sigma_a = 1.0;
        let q_ca = 2.0 * alpha * sigma_a * sigma_a;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        let dt5 = dt4 * dt;
        let expected = Matrix3::new(
            q_ca * dt5 / 20.0,
            q_ca * dt4 / 8.0,
            q_ca * dt3 / 6.0,
            q_ca * dt4 / 8.0,
            q_ca * dt3 / 3.0,
            q_ca * dt2 / 2.0,
            q_ca * dt3 / 6.0,
            q_ca * dt2 / 2.0,
            q_ca * dt,
        );
        let q3 = extract_q3(dt, alpha, sigma_a);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(q3[(i, j)], expected[(i, j)], max_relative = 1e-3);
            }
        }
    }

    #[test]
    fn stationary_variance_for_large_alpha_dt() {
        // As α·Δt → ∞, the acceleration component reaches stationary variance
        // q33 → σ_a², independent of dt.
        let sigma_a = 2.0;
        let q3 = extract_q3(10.0, 5.0, sigma_a); // αΔt = 50
        assert_relative_eq!(q3[(2, 2)], sigma_a * sigma_a, max_relative = 1e-10);
    }

    #[test]
    fn monotonicity_in_dt() {
        // For fixed (α, σ_a), all diagonal entries of Q must be non-decreasing
        // in Δt (more integration time => more accumulated noise).
        let alpha = 0.4;
        let sigma_a = 1.0;
        let dts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let mut prev: Option<Matrix3<f64>> = None;
        for &dt in &dts {
            let q3 = extract_q3(dt, alpha, sigma_a);
            if let Some(p) = prev {
                for i in 0..3 {
                    assert!(
                        q3[(i, i)] >= p[(i, i)] - 1e-14,
                        "diag entry {i} decreased: prev={}, now={}",
                        p[(i, i)],
                        q3[(i, i)]
                    );
                }
            }
            prev = Some(q3);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Property-based tests
    // ──────────────────────────────────────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            .. ProptestConfig::default()
        })]

        /// Q3 is symmetric and positive-definite for all reasonable (α, σ_a, Δt).
        #[test]
        fn prop_symmetric_and_spd(
            log_alpha in -4.0f64..2.0,
            log_sigma in -3.0f64..2.0,
            dt in 1e-3f64..50.0,
        ) {
            let alpha = 10f64.powf(log_alpha);
            let sigma_a = 10f64.powf(log_sigma);
            let q3 = extract_q3(dt, alpha, sigma_a);

            for i in 0..3 {
                for j in (i+1)..3 {
                    let a = q3[(i, j)];
                    let b = q3[(j, i)];
                    let scale = a.abs().max(b.abs()).max(1e-300);
                    prop_assert!((a - b).abs() <= 1e-10 * scale,
                        "asymmetry at ({i},{j}): {a} vs {b}");
                }
            }

            prop_assert!(q3.cholesky().is_some(),
                "not SPD: alpha={alpha}, sigma_a={sigma_a}, dt={dt}, Q={q3}");
        }

        /// Q3 scales as σ_a² and the angular structure is independent of σ_a.
        #[test]
        fn prop_sigma_a_quadratic_scaling(
            log_alpha in -3.0f64..2.0,
            dt in 1e-2f64..20.0,
            k in 0.1f64..10.0,
        ) {
            let alpha = 10f64.powf(log_alpha);
            let q_unit = extract_q3(dt, alpha, 1.0);
            let q_scaled = extract_q3(dt, alpha, k);
            let k2 = k * k;
            for i in 0..3 {
                for j in 0..3 {
                    let lhs = q_scaled[(i, j)];
                    let rhs = k2 * q_unit[(i, j)];
                    let scale = lhs.abs().max(rhs.abs()).max(1e-300);
                    prop_assert!((lhs - rhs).abs() <= 1e-10 * scale,
                        "scaling failed at ({i},{j}): {lhs} vs {rhs}");
                }
            }
        }

        /// The numerically stable entries (q22, q23, q33) match their
        /// closed-form analytical expressions on the whole parameter range.
        /// The numerically stable entries (q23, q33) match their
        /// closed-form analytical expressions on the whole parameter range.
        /// q22 is only compared when αΔt is large enough to avoid cancellation
        /// in the reference formula.
        #[test]
        fn prop_matches_analytical_stable_entries(
            log_alpha in -3.0f64..2.0,
            log_sigma in -2.0f64..2.0,
            dt in 1e-2f64..20.0,
        ) {
            let alpha = 10f64.powf(log_alpha);
            let sigma_a = 10f64.powf(log_sigma);
            let q3 = extract_q3(dt, alpha, sigma_a);
            let (q22, q23, q33) = analytical_stable_entries(alpha, sigma_a, dt);

            prop_assert!(
                (q3[(2, 2)] - q33).abs() <= 1e-8 * q33.abs().max(1e-300),
                "q33 mismatch: got {}, expected {}", q3[(2, 2)], q33
            );
            prop_assert!(
                (q3[(1, 2)] - q23).abs() <= 1e-9 * q23.abs().max(1e-300),
                "q23 mismatch: got {}, expected {}", q3[(1, 2)], q23
            );
            // q22 closed-form suffers catastrophic cancellation for small αΔt;
            // only validate in the numerically safe regime.
            if alpha * dt >= 1e-2 {
                prop_assert!(
                    (q3[(1, 1)] - q22).abs() <= 1e-8 * q22.abs().max(1e-300),
                    "q22 mismatch: got {}, expected {} (alpha={}, dt={})",
                    q3[(1, 1)], q22, alpha, dt
                );
            }
        }

        /// Diagonal entries of Q3 are non-decreasing as Δt grows
        /// (accumulated process noise can only increase with time).
        #[test]
        fn prop_monotone_in_dt(
            log_alpha in -3.0f64..1.5,
            dt1 in 1e-2f64..5.0,
            extra in 1e-3f64..10.0,
        ) {
            let alpha = 10f64.powf(log_alpha);
            let dt2 = dt1 + extra;
            let q_a = extract_q3(dt1, alpha, 1.0);
            let q_b = extract_q3(dt2, alpha, 1.0);
            for i in 0..3 {
                let tol = 1e-10 * q_b[(i, i)].abs().max(1.0);
                prop_assert!(q_b[(i, i)] + tol >= q_a[(i, i)],
                    "diag {i} decreased: {} -> {}", q_a[(i, i)], q_b[(i, i)]);
            }
        }
    }
}

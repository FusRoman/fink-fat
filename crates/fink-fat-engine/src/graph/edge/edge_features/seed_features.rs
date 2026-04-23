use photom::coordinates::cov2::Cov2;

use crate::{
    engine_config::edge_config::{CostConfig, CostVariant},
    graph::edge::edge_features::feature_core::FeatureCore,
    seeding::SeedNode,
};

impl SeedNode {
    /// Numerical floor used across feature computations.
    ///
    /// This constant is applied to:
    /// - stabilize covariance diagonals (avoid singular matrices),
    /// - guard denominators in normalized quantities (`sqrt(S_xx)`),
    /// - prevent whitening from failing due to tiny/negative eigenvalues.
    ///
    /// Design choice
    /// -------------
    /// This is intentionally *very small* to avoid biasing well-behaved cases,
    /// but large enough to avoid division-by-zero and catastrophic numeric blowups.
    pub(crate) const FLOOR: f64 = 1e-20;

    /// Build the innovation covariance matrix $\mathbf{S}$ in position space.
    ///
    /// Definition
    /// ----------
    /// A constant-velocity propagation of uncertainty is used:
    ///
    /// $$\mathbf{C}_{\mathrm{pred}} = \mathbf{C}^{\mathrm{pos}}_{\mathrm{from}} + \Delta t^{2}\, \mathbf{C}^{\mathrm{vel}}_{\mathrm{from}}$$
    ///
    /// $$\mathbf{S} = \mathbf{C}_{\mathrm{pred}} + \mathbf{C}^{\mathrm{pos}}_{\mathrm{to}} + \varepsilon_f\, \mathbf{I}$$
    ///
    /// where $\varepsilon_f$ is a small diagonal floor (see [`Self::FLOOR`]).
    ///
    /// Interpretation
    /// --------------
    /// - $\mathbf{C}^{\mathrm{pos}}_{\mathrm{from}}$: position uncertainty at the `from` epoch.
    /// - $\mathbf{C}^{\mathrm{vel}}_{\mathrm{from}}$: velocity uncertainty at the `from` epoch.
    /// - The $\Delta t^{2}$ factor approximates how velocity uncertainty grows
    ///   into position uncertainty over the time gap $\Delta t$.
    /// - $\mathbf{C}^{\mathrm{pos}}_{\mathrm{to}}$ accounts for measurement uncertainty
    ///   at the `to` epoch.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node providing position and velocity covariances
    ///   (at its own epoch, in tangent-plane coordinates).
    /// * `to` – Target seed node providing position covariance at its epoch.
    /// * `dt_sq` – Precomputed $\Delta t^{2}$ (days²).
    ///
    /// Return
    /// ------
    /// * Innovation covariance [`Cov2`] in the tangent-plane frame of `from`.
    ///
    /// Notes
    /// -----
    /// - An isotropic floor $\varepsilon_f \mathbf{I}$ is added via
    ///   [`Cov2::inflate_isotropic`] to prevent:
    ///   - singular matrices,
    ///   - $\sqrt{0}$ in z-score computations,
    ///   - unstable inversion in Mahalanobis distance,
    ///   - Cholesky failures due to borderline numerical PSD-ness.
    /// - Both input covariances are assumed to be expressed in the same
    ///   tangent-plane frame (that of `from`).
    #[inline]
    pub fn innovation_cov_pos(&self, to: &SeedNode, dt_sq: f64) -> Cov2 {
        // Covariances in tangent-plane coordinates (x, y).
        let cpos_from = self.plane_model.pos.cov;
        let cvel_from = self.plane_model.vel.cov;
        let cpos_to = to.plane_model.pos.cov;

        // Constant-velocity prediction: C_pred = C_pos_from + dt^2 * C_vel_from.
        let cov_pred = cpos_from + cvel_from * dt_sq;

        // Innovation covariance: S = C_pred + C_pos_to + floor * I.
        (cov_pred + cpos_to).inflate_isotropic(Self::FLOOR)
    }

    /// Build the innovation covariance matrix $\mathbf{S}_{\mathrm{vel}}$ in velocity space.
    ///
    /// Definition
    /// ----------
    /// $$\mathbf{S}_{\mathrm{vel}} = \mathbf{C}^{\mathrm{vel}}_{\mathrm{from}} + \mathbf{C}^{\mathrm{vel}}_{\mathrm{to}} + \varepsilon_f\, \mathbf{I}$$
    ///
    /// where $\varepsilon_f$ is a small diagonal floor (see [`Self::FLOOR`]).
    ///
    /// Interpretation
    /// --------------
    /// - $\mathbf{C}^{\mathrm{vel}}_{\mathrm{from}}$: velocity uncertainty at the `from` epoch.
    /// - $\mathbf{C}^{\mathrm{vel}}_{\mathrm{to}}$: velocity uncertainty at the `to` epoch.
    /// - Unlike the position-space innovation, no $\Delta t^{2}$ propagation term
    ///   is required: both covariances are already in velocity units.
    ///
    /// Arguments
    /// ---------
    /// * `to` – Target seed node providing velocity covariance at its epoch.
    ///
    /// Return
    /// ------
    /// * Innovation covariance [`Cov2`] in the tangent-plane velocity frame of `self`.
    ///
    /// Notes
    /// -----
    /// - An isotropic floor $\varepsilon_f \mathbf{I}$ is added via
    ///   [`Cov2::inflate_isotropic`] to prevent:
    ///   - singular matrices,
    ///   - $\sqrt{0}$ in z-score computations,
    ///   - unstable inversion in Mahalanobis distance,
    ///   - Cholesky failures due to borderline numerical PSD-ness.
    /// - Both input covariances are assumed to be expressed in the same
    ///   tangent-plane frame (that of `self`).
    #[inline]
    pub fn innovation_cov_vel(&self, to: &SeedNode) -> Cov2 {
        // Covariances in tangent-plane velocity coordinates (vx, vy).
        let cvel_from = self.plane_model.vel.cov;
        let cvel_to = to.plane_model.vel.cov;

        // Innovation covariance: S_vel = C_vel_from + C_vel_to + floor * I.
        (cvel_from + cvel_to).inflate_isotropic(Self::FLOOR)
    }

    /// Build the positional innovation covariance with optional CWNA
    /// (Continuous White Noise Acceleration) process noise.
    ///
    /// Definition
    /// ----------
    /// Extends [`Self::innovation_cov_pos`] by adding a Singer/CWNA isotropic
    /// diagonal term:
    ///
    /// $$\mathbf{S}_{\mathrm{pos}} \mathrel{+}= \sigma_q^{2}\, \frac{\Delta t^{3}}{3}\, \mathbf{I}$$
    ///
    /// When `sigma_q == 0.0` the result is identical to
    /// [`Self::innovation_cov_pos`].
    ///
    /// Interpretation
    /// --------------
    /// - $\sigma_q$ is the spectral density of an unmodeled white-noise
    ///   acceleration (rad · day$^{-3/2}$).
    /// - The $\Delta t^{3}/3$ scaling is the standard CWNA position-variance
    ///   contribution over a time gap $\Delta t$.
    /// - The term is isotropic in the tangent plane: same variance on $x$ and $y$,
    ///   no cross term.
    ///
    /// Arguments
    /// ---------
    /// * `to` – Target seed node providing position covariance at its epoch.
    /// * `dt` – Time gap $\Delta t$ (days).
    /// * `dt_sq` – Precomputed $\Delta t^{2}$ (days²).
    /// * `sigma_q` – CWNA spectral density (rad · day$^{-3/2}$);
    ///   set to `0.0` to disable the process-noise inflation.
    ///
    /// Return
    /// ------
    /// * Positional innovation covariance [`Cov2`] in the tangent-plane frame
    ///   of `self`.
    ///
    /// Notes
    /// -----
    /// - The isotropic numerical floor from [`Self::innovation_cov_pos`] is
    ///   preserved; the CWNA term is added on top.
    /// - Added via [`Cov2::inflate_isotropic`] to keep the symmetric structure
    ///   and avoid touching off-diagonal terms.
    #[inline]
    pub(crate) fn innovation_cov_pos_cwna(
        &self,
        to: &SeedNode,
        dt: f64,
        dt_sq: f64,
        sigma_q: f64,
    ) -> Cov2 {
        let s = self.innovation_cov_pos(to, dt_sq);
        if sigma_q == 0.0 {
            return s;
        }
        // Q_pos = σ_q² · dt³ / 3  (isotropic in the tangent plane).
        let q = sigma_q * sigma_q * dt * dt_sq / 3.0;
        s.inflate_isotropic(q)
    }

    /// Build the velocity innovation covariance with optional CWNA
    /// (Continuous White Noise Acceleration) process noise.
    ///
    /// Definition
    /// ----------
    /// Extends [`Self::innovation_cov_vel`] by adding a Singer/CWNA isotropic
    /// diagonal term:
    ///
    /// $$\mathbf{S}_{\mathrm{vel}} \mathrel{+}= \sigma_q^{2}\, \Delta t\, \mathbf{I}$$
    ///
    /// When `sigma_q == 0.0` the result is identical to
    /// [`Self::innovation_cov_vel`].
    ///
    /// Interpretation
    /// --------------
    /// - $\sigma_q$ is the spectral density of an unmodeled white-noise
    ///   acceleration (rad · day$^{-3/2}$).
    /// - The $\Delta t$ scaling is the standard CWNA velocity-variance
    ///   contribution over a time gap $\Delta t$.
    /// - The term is isotropic in the tangent plane: same variance on $v_x$ and
    ///   $v_y$, no cross term.
    ///
    /// Arguments
    /// ---------
    /// * `to` – Target seed node providing velocity covariance at its epoch.
    /// * `dt` – Time gap $\Delta t$ (days).
    /// * `sigma_q` – CWNA spectral density (rad · day$^{-3/2}$);
    ///   set to `0.0` to disable the process-noise inflation.
    ///
    /// Return
    /// ------
    /// * Velocity innovation covariance [`Cov2`] in the tangent-plane velocity
    ///   frame of `self`.
    ///
    /// Notes
    /// -----
    /// - The isotropic numerical floor from [`Self::innovation_cov_vel`] is
    ///   preserved; the CWNA term is added on top.
    /// - Added via [`Cov2::inflate_isotropic`] to keep the symmetric structure
    ///   and avoid touching off-diagonal terms.
    #[inline]
    pub(crate) fn innovation_cov_vel_cwna(&self, to: &SeedNode, dt: f64, sigma_q: f64) -> Cov2 {
        let s = self.innovation_cov_vel(to);
        if sigma_q == 0.0 {
            return s;
        }
        // Q_vel = σ_q² · dt  (isotropic in the tangent plane).
        let q = sigma_q * sigma_q * dt;
        s.inflate_isotropic(q)
    }

    /// Extract positional and velocity $\chi^2$ values, optionally inflated with
    /// CWNA (Continuous White Noise Acceleration) process noise.
    ///
    /// Positional $\chi^2$
    /// -------------------
    /// The positional term uses a **spherical residual** $d$ (see
    /// [`FeatureCore::r_sph`]) instead of the 2-D gnomonic Mahalanobis:
    ///
    /// $$\chi^{2}_{\mathrm{pos}} = \frac{d^{2}}{S_{\mathrm{scalar}}}$$
    ///
    /// where $S_{\mathrm{scalar}} = \operatorname{tr}(\mathbf{S}_{\mathrm{pos}}) / 2$.
    ///
    /// Motivation: the gnomonic residual $\|\mathbf{r}\|$ diverges when the
    /// tangent-plane denominator $\cos c \to 0$ (seed centres separated by
    /// $\gtrsim 45^\circ$), leading to $\chi^{2}_{\mathrm{pos}} \sim 10^{20}$
    /// for otherwise valid edges. The great-circle distance
    /// $d \in [0, \pi]$ is bounded and well-defined for any separation.
    ///
    /// Velocity $\chi^2$
    /// -----------------
    /// The velocity term uses the full 2-D Mahalanobis distance on the
    /// tangent-plane velocity innovation $\delta \mathbf{v}$:
    ///
    /// $$\chi^{2}_{\mathrm{vel}} = \delta \mathbf{v}^{\top}\, \mathbf{S}_{\mathrm{vel}}^{-1}\, \delta \mathbf{v}$$
    ///
    /// Velocity residuals do not suffer from the gnomonic blow-up, so the
    /// standard 2-D form is kept.
    ///
    /// Behavior (two modes)
    /// --------------------
    /// Controlled by `cfg.variant` and `cfg.sigma_q`:
    ///
    /// - If `sigma_q == 0.0` (or `variant == KinematicLogLikelihood`):
    ///   - returns the cached spherical positional $\chi^2$ from `core`,
    ///   - returns the cached velocity $\chi^2$ from `core`,
    ///   - no extra linear algebra is performed.
    ///
    /// - If `sigma_q > 0.0`:
    ///   - recomputes the two $2 \times 2$ innovation covariances with the CWNA
    ///     diagonal term (see [`FeatureCore::innovation_cov_cwna`] and
    ///     [`FeatureCore::innovation_cov_vel_cwna`]),
    ///   - applies the spherical formula for position,
    ///   - applies the 2-D Mahalanobis form for velocity,
    ///   - does **not** repeat propagation or tangent-plane projection.
    ///
    /// Arguments
    /// ---------
    /// * `core` – Shared edge intermediates (cached `r_sph`, `s_pos_scalar`,
    ///   `dv`, `chi2_vel`, `dt`, `dt_sq`).
    /// * `to`   – Target seed node; required to rebuild the innovation
    ///   covariances in the CWNA path.
    /// * `cfg`  – Cost configuration; `variant` and `sigma_q` are read here.
    ///
    /// Return
    /// ------
    /// * `(chi2_pos, chi2_vel)` – positional and velocity $\chi^2$ values.
    ///
    /// Notes
    /// -----
    /// - `KinematicLogLikelihood` is a baseline alias: it always forces
    ///   `sigma_q = 0.0`, independent of the configured value.
    /// - Both outputs are passed through [`FeatureCore::finite_or_zero`] and
    ///   clamped to be non-negative to absorb floating-point noise.
    /// - The scalar $S_{\mathrm{scalar}}$ is floored by [`FeatureCore::FLOOR`]
    ///   to prevent division by zero.
    #[inline]
    pub(crate) fn chi2_with_cwna(
        &self,
        core: &FeatureCore,
        to: &SeedNode,
        cfg: &CostConfig,
    ) -> (f64, f64) {
        // KinematicLogLikelihood is a baseline alias: always uses sigma_q = 0.
        let sigma_q = match cfg.variant {
            CostVariant::KinematicLogLikelihood => 0.0,
            _ => cfg.sigma_q,
        };

        if sigma_q == 0.0 {
            // Fast path: reuse pre-computed spherical chi2_pos and cached chi2_vel.
            // chi2_pos_sph = r_sph² / s_pos_scalar   (great-circle, no blow-up)
            let chi2_p =
                FeatureCore::finite_or_zero((core.r_sph * core.r_sph / core.s_pos_scalar).max(0.0));
            return (chi2_p, core.chi2_vel);
        }

        // CWNA path: inflate covariances with Singer process noise and recompute.
        //
        // Positional chi2 uses the spherical residual r_sph with an isotropic
        // scalar S (S_scalar = tr(S_cwna) / 2).
        let s_pos = self.innovation_cov_pos_cwna(to, core.dt, core.dt_sq, sigma_q);
        let s_pos_scalar = ((s_pos.xx + s_pos.yy).max(FeatureCore::FLOOR)) / 2.0;
        let chi2_p = FeatureCore::finite_or_zero((core.r_sph * core.r_sph / s_pos_scalar).max(0.0));

        // Velocity chi2: full 2-D Mahalanobis on the velocity innovation.
        let s_vel = self.innovation_cov_vel_cwna(to, core.dt, sigma_q);
        let chi2_v =
            FeatureCore::finite_or_zero(s_vel.mahalanobis_sq(core.dv).unwrap_or(0.0).max(0.0));

        (chi2_p, chi2_v)
    }
}

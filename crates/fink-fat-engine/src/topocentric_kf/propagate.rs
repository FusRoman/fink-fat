use nalgebra::{Matrix6, Vector3, Vector6};
use outfit::{
    OutfitError,
    kepler::{SolverParams, SolverType, UniversalPropagResult, propagate_universal},
};
use photom::observation_dataset::{ObsDataset, observation::Observation};

use crate::topocentric_kf::{
    KFState,
    conversion::{CartesianState, cartesian_to_attributable, jacobian_attr_to_cart},
    observer_state::{HelioObsState, get_observer},
};

/// Speed of light in astronomical units per day.
///
/// Consistent with the IAU 2009 / DE440 value (`c = 299_792.458 km/s`,
/// `1 AU = 149_597_870.7 km`, `1 day = 86_400 s`). Used to convert the
/// topocentric range `ρ` into a light-time delay `τ = ρ / c`.
const C_AU_PER_DAY: f64 = 173.144_632_674_240_57;

/// χ²(2) upper 95 % quantile — the consistency reference for NIS-driven
/// covariance inflation.
///
/// Used as a **dead-zone** threshold: as long as the smoothed NIS stays below
/// this value the filter is deemed statistically consistent and its covariance
/// is transported unchanged (`λ = 1`). Inflation only engages on genuine
/// inconsistency, so a well-behaved filter is never perturbed.
const CHI2_2DOF_95: f64 = 5.991;

/// Maximum per-step covariance inflation factor.
///
/// Caps how aggressively a single propagation may re-open the covariance. A
/// catastrophic NIS (e.g. 10³) would otherwise inflate `P` by a huge factor in
/// one step (an outlier over-reaction); clamping to `5×` per step spreads the
/// recovery over a few predictions, keeping the transport smooth while still
/// converging quickly back into the consistency dead-zone.
const MAX_INFLATION: f64 = 5.0;

/// Build the universal-variable [`SolverType`] for this filter step, injecting
/// the previous universal anomaly as a warm-start guess.
///
/// Factored out so the light-time iteration below can re-call the propagator
/// several times without duplicating the (verbose) config-splatting.
#[inline]
fn solver_with_guess(kf: &KFState, psi_guess: Option<f64>) -> SolverType {
    SolverType {
        params: SolverParams {
            psi_guess,
            ..kf.shared_ctx.config.solver_type.params
        },
        ..kf.shared_ctx.config.solver_type
    }
}

/// Error type for attributable-space Kalman propagation.
#[derive(Debug, thiserror::Error)]
pub enum PropagateError {
    /// The universal-variable Kepler solver failed to converge.
    #[error("Kepler propagation failed: {0}")]
    Kepler(#[from] OutfitError),
    /// The Jacobian $J_{new}$ at the propagated state is singular.
    #[error("Jacobian inversion failed: degenerate attributable geometry")]
    SingularJacobian,
}

/// Build the $6 \times 6$ Keplerian state transition matrix from Lagrange
/// coefficients.
///
/// Under two-body dynamics the STM has the scalar block structure:
///
/// $$\Phi = \begin{pmatrix} f\,I_3 & g\,I_3 \\ \dot{f}\,I_3 & \dot{g}\,I_3 \end{pmatrix}$$
///
/// where $I_3$ is the $3 \times 3$ identity matrix and $(f, g, \dot{f}, \dot{g})$
/// are the Lagrange coefficients returned by the universal-variable solver.
fn build_keplerian_stm(f: f64, g: f64, f_dot: f64, g_dot: f64) -> Matrix6<f64> {
    let mut stm = Matrix6::zeros();
    let i3 = nalgebra::Matrix3::identity();
    stm.fixed_view_mut::<3, 3>(0, 0).copy_from(&(f * i3));
    stm.fixed_view_mut::<3, 3>(0, 3).copy_from(&(g * i3));
    stm.fixed_view_mut::<3, 3>(3, 0).copy_from(&(f_dot * i3));
    stm.fixed_view_mut::<3, 3>(3, 3).copy_from(&(g_dot * i3));
    stm
}

/// Build the SNC process noise matrix in Cartesian coordinates with
/// adaptive scaling.
///
/// Under the State Noise Compensation (SNC) model, the unmodeled
/// acceleration is treated as a continuous white noise. The baseline
/// power spectral density $q_0$ is scaled by a factor that grows
/// quadratically with the propagation interval $\Delta t$, compensating
/// for the accumulation of unmodeled perturbations (planetary
/// perturbations, non-gravitational forces) on longer arcs:
///
/// $$q_{eff}(\Delta t) = q_0 \left(1 + \left(\frac{\Delta t}{\Delta t_{ref}}\right)^2\right)$$
///
/// The discrete-time process noise covariance is then:
///
/// $$Q_{cart} = q_{eff}(\Delta t)
///   \begin{pmatrix}
///     \frac{\Delta t^3}{3}\,I_3 & \frac{\Delta t^2}{2}\,I_3 \\
///     \frac{\Delta t^2}{2}\,I_3 & \Delta t\,I_3
///   \end{pmatrix}$$
///
/// The quadratic scaling is motivated by the fact that planetary
/// perturbation errors grow roughly as $\Delta t^2$ over short arcs,
/// so $q_{eff} \propto \Delta t^2$ keeps the total position uncertainty
/// $\sim q_{eff} \cdot \Delta t^3 / 3$ growing as $\Delta t^5$, which
/// better matches the actual prediction error budget.
///
/// Arguments
/// ---------
/// * `dt`       – Propagation interval (days).
/// * `q0`       – Baseline acceleration noise PSD (AU² day⁻³).
/// * `dt_ref`   – Reference interval beyond which perturbation scaling
///               activates (days). Typically 1 day.
///
/// Return
/// ------
/// $6 \times 6$ process noise matrix in the Cartesian state space.
fn build_snc_process_noise(dt: f64, q0: f64, dt_ref: f64) -> Matrix6<f64> {
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let i3 = nalgebra::Matrix3::identity();

    // Adaptive scaling: grows quadratically beyond dt_ref.
    let scale = 1.0 + (dt / dt_ref).powi(2);
    let q_eff = q0 * scale;

    let mut q = Matrix6::zeros();
    q.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&((dt3 / 3.0) * i3));
    q.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&((dt2 / 2.0) * i3));
    q.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&((dt2 / 2.0) * i3));
    q.fixed_view_mut::<3, 3>(3, 3).copy_from(&(dt * i3));
    q * q_eff
}

/// Propagate a Kalman filter state to a target epoch without requiring an
/// observation object.
///
/// This is the core propagation primitive. It takes the target epoch and the
/// observer heliocentric state at that epoch directly, making it usable both
/// for filter updates (where an [`Observation`] drives the epoch) and for
/// read-only prediction (where no observation exists yet).
///
/// # Light-time correction (planetary aberration)
///
/// The astrometric positions produced by surveys such as ZTF are **apparent**
/// directions: the light recorded in an exposure taken at reception time
/// `t_prop` physically left the object earlier, at emission time
/// `t_emit = t_prop − τ`, where the light-time delay is
///
/// $$\tau = \frac{\rho}{c}$$
///
/// and `ρ` is the topocentric range. The observed sky angles therefore point
/// to where the object **was** at `t_emit`, seen from where the observer **is**
/// at `t_prop`.
///
/// The Kalman state stores *apparent* angles `(α, δ)` (the initializer reads
/// raw ZTF angles verbatim; see [`crate::topocentric_kf::init::init_kf_state`]),
/// so the predicted measurement must also be apparent. To achieve this we:
///
/// 1. propagate the **object** two-body dynamics only up to the emission epoch
///    `t_emit = t_prop − τ`, and
/// 2. keep the **observer** heliocentric state at the reception epoch `t_prop`
///    (`r_obs_new`, `v_obs_new` are passed in already evaluated at `t_prop`).
///
/// Forming `Δ = r_obj(t_emit) − r_obs(t_prop)` in
/// [`cartesian_to_attributable`] then yields exactly the apparent line of
/// sight the survey measured.
///
/// Without this correction the filter predicts the *geometric* direction
/// `r_obj(t_prop) − r_obs(t_prop)`, which lags the apparent direction by
/// `ω·τ` along the trace (`ω` = on-sky angular speed). For a main-belt object
/// near opposition (`ω ≈ 0.3 °/day`, `ρ ≈ 1.5 AU ⇒ τ ≈ 0.009 day`) this is a
/// systematic ~10″ bias — the dominant error term this filter was suffering
/// from, corrupting the poorly-observed radial components `(ρ, ρ̇)` and driving
/// the NIS far above its `χ²(2)` expectation.
///
/// A single refinement iteration is enough: `ρ` is already well constrained by
/// the incoming state, so `τ` converges immediately once re-evaluated from the
/// propagated range.
///
/// Note: annual (stellar) aberration is **not** applied here. Survey astrometry
/// is calibrated differentially against a star catalogue (Gaia), which already
/// absorbs the observer-velocity aberration common to all field sources; only
/// the object-specific planetary aberration (light-time) survives that
/// calibration and must be modelled explicitly.
///
/// See [`propagate`] for the full mathematical description of the
/// attributable-space covariance transport.
///
/// Arguments
/// ---------
/// * `kf`        – Current filter state.
/// * `t_prop`    – Target (reception) epoch (MJD TT).
/// * `r_obs_new` – Observer heliocentric position at `t_prop` (AU, ecliptic J2000).
/// * `v_obs_new` – Observer heliocentric velocity at `t_prop` (AU/day, ecliptic J2000).
/// * `q0`        – Baseline acceleration noise PSD (AU² day⁻³).
/// * `dt_ref`    – Reference interval beyond which perturbation scaling
///                 activates (days). Typically 1 day.
///
/// Return
/// ------
/// * `Ok(KFState)` – Propagated state at `t_prop` (apparent angles) with
///   updated covariance.
/// * `Err(PropagateError::Kepler)` – If the Kepler solver fails.
/// * `Err(PropagateError::SingularJacobian)` – If $J_{new}$ cannot be inverted.
pub(crate) fn propagate_to_epoch<'state_lf>(
    kf: &KFState<'state_lf>,
    t_prop: f64,
    r_obs_new: Vector3<f64>,
    v_obs_new: Vector3<f64>,
    q0: f64,
    dt_ref: f64,
) -> Result<KFState<'state_lf>, PropagateError> {
    let dt = t_prop - kf.epoch;

    println!("___ Perform Kalman propagation ___");
    println!("dt = {dt}");
    println!("previous universal anomaly: {:?}", kf.universal_anomaly);
    println!(" ______ \n");

    let span = tracing::trace_span!(
        "propagate_to_epoch",
        epoch_from = kf.epoch,
        epoch_to = t_prop,
        dt_days = dt
    );
    let _enter = span.enter();

    let cart = kf.to_cartesian();

    log_initial_cartesian(&cart);

    // ── Light-time correction (planetary aberration) ──────────────────────
    // The stored attributable state holds *apparent* angles: (α, δ) point to
    // where the object WAS at emission time, seen from the observer at the
    // reception epoch. By the apparent-place identity
    //
    //     r_obs(t) + ρ·los_app = r_obj(t − τ),     τ = ρ / c
    //
    // `kf.to_cartesian()` therefore already yields the object's TRUE
    // heliocentric state at the *emission* epoch `kf.epoch − τ_prev`, NOT at
    // `kf.epoch`. The two-body arc must consequently be integrated from one
    // emission epoch to the next:
    //
    //     (kf.epoch − τ_prev)  ──►  (t_prop − τ_new)
    //
    // Integrating instead from `kf.epoch` (as a naive light-time patch would)
    // introduces a −τ_prev epoch error. It is harmless when dt ≫ τ, but for
    // ZTF's intra-night pairs (dt ≈ 0.003 d < τ ≈ 0.009 d) it flips the arc
    // backwards and destabilises the warm-started Kepler solver, collapsing
    // the bank. Anchoring on the emission epoch keeps the object arc equal to
    // the true observation spacing (always forward for forward observations).
    let tau_prev = (kf.state[4] / C_AU_PER_DAY).max(0.0); // ρ_stored / c
    let t0_emit = kf.epoch - tau_prev;

    // First light-time guess for the target uses the current range; one
    // refinement with the propagated range is sufficient (ρ is well
    // constrained, so τ converges immediately).
    let mut tau = tau_prev;
    let mut result = propagate_universal(
        &cart.pos,
        &cart.vel,
        t0_emit,
        t_prop - tau,
        solver_with_guess(kf, kf.universal_anomaly),
    )
    .map_err(PropagateError::Kepler)?;

    {
        // Observer stays at the RECEPTION epoch t_prop; only the object recedes
        // to its emission epoch. Δ = r_obj(t_prop − τ) − r_obs(t_prop) is then
        // exactly the apparent line of sight the survey measured.
        let rho_emit = (result.r1 - r_obs_new).norm();
        tau = (rho_emit / C_AU_PER_DAY).max(0.0);

        result = propagate_universal(
            &cart.pos,
            &cart.vel,
            t0_emit,
            t_prop - tau,
            solver_with_guess(kf, Some(result.psy)),
        )
        .map_err(PropagateError::Kepler)?;
    }

    tracing::trace!(
        target: "propagation",
        tau_prev_days = tau_prev,
        tau_new_days = tau,
        object_arc_days = (t_prop - tau) - t0_emit,
        recept_dt_days = t_prop - kf.epoch,
        "Light-time correction (emission-to-emission object arc)"
    );

    log_kepler_result(&result);

    let stm = build_keplerian_stm(result.f_lag, result.g_lag, result.f_dot, result.g_dot);

    // Object state at emission epoch, differenced against the observer at the
    // reception epoch → apparent attributable state.
    let cart_new = CartesianState {
        pos: result.r1,
        vel: result.v1,
    };
    let attr_new = cartesian_to_attributable(&cart_new, &r_obs_new, &v_obs_new);

    log_propagated_attributable(&attr_new);

    let p_new = propagate_covariance(kf, &stm, &attr_new, dt, q0, dt_ref)?;

    log_covariance_traces(kf, &p_new);

    tracing::trace!(
        target: "propagation",
        "Propagation complete."
    );

    Ok(KFState {
        epoch: t_prop,
        r_obs: r_obs_new,
        v_obs: v_obs_new,
        universal_anomaly: Some(result.psy),
        state: attr_new,
        covariance: p_new,
        kalman_gain: None,
        ..kf.clone()
    })
}

/// Propagate a Kalman filter state to the epoch of a given observation.
///
/// This is the standard filter update entrypoint. The observer heliocentric
/// state at the observation epoch is resolved from `obs_dataset` and
/// forwarded to [`propagate_to_epoch`].
///
/// Arguments
/// ---------
/// * `kf`          – Current filter state.
/// * `obs_dataset` – Dataset used to resolve the observer position at the
///   observation epoch.
/// * `obs`         – Observation whose epoch drives the propagation.
/// * `q0`       – Baseline acceleration noise PSD (AU² day⁻³).
/// * `dt_ref`   – Reference interval beyond which perturbation scaling
///               activates (days). Typically 1 day.
///
/// Return
/// ------
/// * `Ok(KFState)` – Propagated state at `obs.mjd_tt()`.
/// * `Err(PropagateError)` – Propagated from [`propagate_to_epoch`] or from
///   the observer state resolver.
pub(crate) fn propagate<'state_lf>(
    kf: &KFState<'state_lf>,
    obs_dataset: &ObsDataset,
    obs: &Observation,
    q0: f64,
    dt_ref: f64,
) -> Result<KFState<'state_lf>, PropagateError> {
    let t_prop = obs.mjd_tt();
    let observer = get_observer(obs_dataset, obs)?;
    let HelioObsState {
        helio_cart_pos: r_obs_new,
        helio_cart_vel: v_obs_new,
    } = kf
        .shared_ctx
        .get_ephem()
        .helio_observer_state(observer, t_prop)?;

    propagate_to_epoch(kf, t_prop, r_obs_new, v_obs_new, q0, dt_ref)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Propagate and transform the covariance from the current epoch to the
/// propagated epoch.
///
/// The full pipeline is:
///
/// $$P_{cart} = J\,P_{attr}\,J^\top$$
/// $$P_{cart,new} = \Phi\,P_{cart}\,\Phi^\top$$
/// $$P_{attr,new} = J_{new}^{-1}\,P_{cart,new}\,(J_{new}^{-1})^\top + Q_{attr}$$
fn propagate_covariance(
    kf: &KFState,
    stm: &Matrix6<f64>,
    attr_new: &Vector6<f64>,
    dt: f64,
    q0: f64,
    dt_ref: f64,
) -> Result<Matrix6<f64>, PropagateError> {
    let j = jacobian_attr_to_cart(&kf.state);
    tracing::trace!(
        target: "propagation",
        j_det = j.determinant(),
        "Jacobian attr→cart at current epoch"
    );

    let p_cart = j * kf.covariance * j.transpose();

    // ── Adaptive covariance inflation (fading-memory filter) ───────────────
    // Pure two-body propagation cannot represent unmodeled dynamics (planetary
    // perturbations reach tens of arcsec over a multi-month arc), and the tiny
    // baseline process noise lets `P` collapse. Left unchecked, the filter
    // becomes over-confident, the gain on (ρ, ρ̇) closes, and it locks onto a
    // biased two-body orbit — exactly the NIS ≫ χ²(2) regime observed.
    //
    // The fading-memory correction re-opens the covariance in proportion to the
    // *smoothed* NIS carried from past updates:
    //
    //     P⁻ = λ · Φ P Φᵀ + Q,   λ = clamp(NIS_ema / χ²₉₅, 1, λ_max)
    //
    // Using the smoothed NIS (rather than the last raw value) keeps λ robust to
    // isolated outliers; the dead-zone (λ = 1 below χ²₉₅) leaves a consistent
    // filter untouched; the cap bounds the per-step reaction.
    let lambda = kf
        .nis_ema
        .map_or(1.0, |ema| (ema / CHI2_2DOF_95).clamp(1.0, MAX_INFLATION));

    let p_cart_new = lambda * (stm * p_cart * stm.transpose());

    tracing::trace!(
        target: "propagation",
        nis_ema = kf.nis_ema.unwrap_or(f64::NAN),
        lambda,
        inflation_active = lambda > 1.0,
        "Adaptive covariance inflation factor (fading-memory)"
    );

    let j_new = jacobian_attr_to_cart(attr_new);
    tracing::trace!(
        target: "propagation",
        j_new_det = j_new.determinant(),
        "Jacobian attr→cart at propagated epoch"
    );

    let j_new_inv = j_new
        .try_inverse()
        .ok_or(PropagateError::SingularJacobian)?;

    let q_cart = build_snc_process_noise(dt, q0, dt_ref);
    let q_attr = j_new_inv * q_cart * j_new_inv.transpose();

    tracing::trace!(
        target: "propagation",
        q_snc_effective = q0 * (1.0 + (dt / 1.0_f64).powi(2)),
        q_attr_trace = q_attr.trace(),
        dt_days = dt,
        "SNC process noise (adaptive)"
    );

    Ok(j_new_inv * p_cart_new * j_new_inv.transpose() + q_attr)
}

fn log_initial_cartesian(cart: &CartesianState) {
    tracing::trace!(
        target: "propagation",
        pos_norm_au = cart.pos.norm(),
        vel_norm_au_per_day = cart.vel.norm(),
        "Initial heliocentric state (AU, AU/day)"
    );
}

fn log_kepler_result(result: &UniversalPropagResult) {
    tracing::trace!(
        target: "propagation",
        f_lag = result.f_lag,
        g_lag = result.g_lag,
        f_dot = result.f_dot,
        g_dot = result.g_dot,
        r1_norm_au = result.r1.norm(),
        v1_norm_au_per_day = result.v1.norm(),
        "Kepler propagation result"
    );
}

fn log_propagated_attributable(attr: &Vector6<f64>) {
    tracing::trace!(
        target: "propagation",
        ra_deg = attr[0].to_degrees(),
        dec_deg = attr[1].to_degrees(),
        rho_au = attr[4],
        "Propagated attributable state"
    );
}

fn log_covariance_traces(kf: &KFState, p_new: &Matrix6<f64>) {
    tracing::trace!(
        target: "propagation",
        cov_pos_trace_before = kf.covariance.fixed_view::<3, 3>(0, 0).trace(),
        cov_pos_trace_after = p_new.fixed_view::<3, 3>(0, 0).trace(),
        "Covariance traces before/after propagation"
    );
}

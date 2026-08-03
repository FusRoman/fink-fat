use nalgebra::{Matrix6, Vector3, Vector6};
use outfit::{
    OutfitError,
    kepler::{SolverParams, SolverType, propagate_universal},
};
use photom::observation_dataset::{ObsDataset, observation::Observation};

use crate::topocentric_kf::{
    constants::{C_AU_PER_DAY, MAX_RHO_AU, MIN_RHO_AU},
    conversion::{CartesianState, cartesian_to_attributable, jacobian_attr_to_cart},
    observer_state::{HelioObsState, get_observer},
    single_kalman::KFState,
};

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

use crate::logging::LogTarget;

/// Structured log events for the propagation pipeline (light-time correction,
/// Kepler solve, covariance transport). See [`crate::logging`] for the
/// `.emit()` pattern.
pub enum PropagationEvent {
    InitialCartesian {
        pos_norm_au: f64,
        vel_norm_au_per_day: f64,
    },
    LightTimeCorrection {
        tau_prev_days: f64,
        tau_new_days: f64,
        object_arc_days: f64,
        recept_dt_days: f64,
    },
    KeplerResult {
        f_lag: f64,
        g_lag: f64,
        f_dot: f64,
        g_dot: f64,
        r1_norm_au: f64,
        v1_norm_au_per_day: f64,
    },
    PropagatedAttributable {
        ra_deg: f64,
        dec_deg: f64,
        rho_au: f64,
    },
    JacobianAtEpoch {
        label: &'static str,
        det: f64,
    },
    InflationFactor {
        nis_ema: f64,
        lambda: f64,
        inflation_active: bool,
    },
    ProcessNoise {
        q_snc_effective: f64,
        q_attr_trace: f64,
        dt_days: f64,
    },
    CovarianceTraces {
        pos_trace_before: f64,
        pos_trace_after: f64,
    },
    Complete,
}

crate::impl_log_target!(
    PropagationEvent,
    "propagation",
    "Light-time-corrected Keplerian propagation with adaptive covariance inflation",
    [tracing::Level::TRACE]
);

impl PropagationEvent {
    pub fn emit(&self) {
        use PropagationEvent::*;
        match self {
            InitialCartesian {
                pos_norm_au,
                vel_norm_au_per_day,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                pos_norm_au,
                vel_norm_au_per_day,
                "Initial heliocentric state (AU, AU/day)"
            ),
            LightTimeCorrection {
                tau_prev_days,
                tau_new_days,
                object_arc_days,
                recept_dt_days,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                tau_prev_days,
                tau_new_days,
                object_arc_days,
                recept_dt_days,
                "Light-time correction (emission-to-emission object arc)"
            ),
            KeplerResult {
                f_lag,
                g_lag,
                f_dot,
                g_dot,
                r1_norm_au,
                v1_norm_au_per_day,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                f_lag,
                g_lag,
                f_dot,
                g_dot,
                r1_norm_au,
                v1_norm_au_per_day,
                "Kepler propagation result"
            ),
            PropagatedAttributable {
                ra_deg,
                dec_deg,
                rho_au,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                ra_deg,
                dec_deg,
                rho_au,
                "Propagated attributable state"
            ),
            JacobianAtEpoch { label, det } => tracing::trace!(
                target: PropagationEvent::TARGET,
                det,
                "Jacobian attr→cart at {label} epoch"
            ),
            InflationFactor {
                nis_ema,
                lambda,
                inflation_active,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                nis_ema,
                lambda,
                inflation_active,
                "Adaptive covariance inflation factor (fading-memory)"
            ),
            ProcessNoise {
                q_snc_effective,
                q_attr_trace,
                dt_days,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                q_snc_effective,
                q_attr_trace,
                dt_days,
                "SNC process noise (adaptive)"
            ),
            CovarianceTraces {
                pos_trace_before,
                pos_trace_after,
            } => tracing::trace!(
                target: PropagationEvent::TARGET,
                pos_trace_before,
                pos_trace_after,
                "Covariance traces before/after propagation"
            ),
            Complete => tracing::trace!(target: PropagationEvent::TARGET, "Propagation complete."),
        }
    }

    pub fn span(epoch_from: f64, epoch_to: f64, dt_days: f64) -> tracing::Span {
        tracing::trace_span!(
            target: PropagationEvent::TARGET,
            "propagate_to_epoch",
            epoch_from,
            epoch_to,
            dt_days
        )
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
    /// The propagated range estimate ρ left the sane range where
    /// `jacobian_attr_to_cart` stays well-conditioned. Four of its six
    /// columns scale linearly with ρ, so as ρ drifts toward 0 (or diverges),
    /// `try_inverse()` still succeeds (the matrix isn't *exactly* singular)
    /// but returns hugely amplified entries — applied on both sides of the
    /// propagated covariance, this squares the amplification and can
    /// silently inflate a hypothesis's sky-plane covariance by 10+ orders
    /// of magnitude while staying "finite" in the `f64` sense. Rejecting
    /// propagation here lets this hypothesis be dropped through the normal
    /// "failed to propagate" path instead of drifting further, undetected,
    /// on every subsequent night.
    #[error(
        "Range estimate ρ={rho_au} AU outside sane bounds — ill-conditioned attributable↔cartesian Jacobian"
    )]
    IllConditionedRange { rho_au: f64 },
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

/// Maximum number of fixed-point refinements for the light-time delay `τ`
/// (see [`refine_light_time_tau`]). `ρ` is usually well constrained enough
/// that one refinement suffices (the original behaviour, preserved as
/// `max_iters = 2`), but this bounds a real convergence loop instead of
/// assuming it, so poorly-constrained hypotheses (fresh off the seeding
/// grid, or fast NEO-like `ρ̇`) get the extra iterations they need.
const LIGHT_TIME_MAX_ITERS: usize = 5;

/// Convergence tolerance on `τ` (days) for [`refine_light_time_tau`]. Loose
/// relative to `τ`'s own magnitude (minutes, i.e. ~1e-3 day for a typical
/// MBA) but tight enough that the residual along-track bias it leaves
/// behind (`tol_days × angular_rate`) is negligible at any realistic rate.
const LIGHT_TIME_TAU_TOL_DAYS: f64 = 1e-9;

/// Fixed-point refinement of the light-time delay `τ = ρ/c`: propagate the
/// object to the candidate emission epoch `t_prop - τ` via `propagate_at`,
/// measure the resulting range against the (reception-epoch) observer
/// position `r_obs_new`, and update `τ` accordingly — repeating until `τ`
/// stops moving (by more than `tol_days`) or `max_iters` is reached.
///
/// Generic over `propagate_at`/`extract_pos` so it can be driven by the
/// real two-body Kepler solver in production and by closed-form synthetic
/// motion models in tests — decoupling "is the light-time iteration itself
/// correct" from "is the Kepler solver correct".
///
/// Returns `(τ, last propagation result)`, with the invariant that `result`
/// is always `propagate_at(t_prop - τ)` for the returned `τ` — i.e. the
/// pair is always mutually consistent, never a stale `result` paired with
/// an unpropagated `τ` update.
fn refine_light_time_tau<R, E>(
    mut propagate_at: impl FnMut(f64) -> Result<R, E>,
    extract_pos: impl Fn(&R) -> Vector3<f64>,
    r_obs_new: Vector3<f64>,
    t_prop: f64,
    tau_init: f64,
    max_iters: usize,
    tol_days: f64,
) -> Result<(f64, R), E> {
    let mut tau = tau_init;
    let mut result = propagate_at(t_prop - tau)?;

    for _ in 1..max_iters {
        let rho_emit = (extract_pos(&result) - r_obs_new).norm();
        let tau_new = (rho_emit / C_AU_PER_DAY).max(0.0);
        if (tau_new - tau).abs() < tol_days {
            // `result` (propagated at the current `tau`) is already
            // self-consistent with the range it predicts — stop without a
            // wasted extra propagation.
            break;
        }
        tau = tau_new;
        result = propagate_at(t_prop - tau)?;
    }

    Ok((tau, result))
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
///   activates (days). Typically 1 day.
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
/// raw ZTF angles verbatim; see [`crate::topocentric_kf::single_kalman::KFState::init_kf_state`]),
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
///   activates (days). Typically 1 day.
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

    let _enter = PropagationEvent::span(kf.epoch, t_prop, dt).entered();

    let cart = kf.to_cartesian();

    PropagationEvent::InitialCartesian {
        pos_norm_au: cart.pos.norm(),
        vel_norm_au_per_day: cart.vel.norm(),
    }
    .emit();

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

    // Refine τ by fixed-point iteration until it stops moving (bounded —
    // see `refine_light_time_tau`) rather than assuming one refinement
    // always suffices. Observer stays at the RECEPTION epoch t_prop; only
    // the object recedes to its emission epoch. Δ = r_obj(t_prop − τ) −
    // r_obs(t_prop) is then exactly the apparent line of sight the survey
    // measured.
    let mut psi_guess = kf.universal_anomaly;
    let (tau, result) = refine_light_time_tau(
        |t_emit_target| -> Result<outfit::kepler::UniversalPropagResult, PropagateError> {
            let r = propagate_universal(
                &cart.pos,
                &cart.vel,
                t0_emit,
                t_emit_target,
                solver_with_guess(kf, psi_guess),
            )
            .map_err(PropagateError::Kepler)?;
            psi_guess = Some(r.psy);
            Ok(r)
        },
        |r: &outfit::kepler::UniversalPropagResult| r.r1,
        r_obs_new,
        t_prop,
        tau_prev,
        LIGHT_TIME_MAX_ITERS,
        LIGHT_TIME_TAU_TOL_DAYS,
    )?;

    PropagationEvent::LightTimeCorrection {
        tau_prev_days: tau_prev,
        tau_new_days: tau,
        object_arc_days: (t_prop - tau) - t0_emit,
        recept_dt_days: t_prop - kf.epoch,
    }
    .emit();

    PropagationEvent::KeplerResult {
        f_lag: result.f_lag,
        g_lag: result.g_lag,
        f_dot: result.f_dot,
        g_dot: result.g_dot,
        r1_norm_au: result.r1.norm(),
        v1_norm_au_per_day: result.v1.norm(),
    }
    .emit();

    let stm = build_keplerian_stm(result.f_lag, result.g_lag, result.f_dot, result.g_dot);

    // Object state at emission epoch, differenced against the observer at the
    // reception epoch → apparent attributable state.
    let cart_new = CartesianState {
        pos: result.r1,
        vel: result.v1,
    };
    let attr_new = cartesian_to_attributable(&cart_new, &r_obs_new, &v_obs_new);

    PropagationEvent::PropagatedAttributable {
        ra_deg: attr_new[0].to_degrees(),
        dec_deg: attr_new[1].to_degrees(),
        rho_au: attr_new[4],
    }
    .emit();

    let p_new = propagate_covariance(kf, &stm, &attr_new, dt, q0, dt_ref)?;

    PropagationEvent::CovarianceTraces {
        pos_trace_before: kf.covariance.fixed_view::<3, 3>(0, 0).trace(),
        pos_trace_after: p_new.fixed_view::<3, 3>(0, 0).trace(),
    }
    .emit();

    PropagationEvent::Complete.emit();

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
///   activates (days). Typically 1 day.
///
/// Return
/// ------
/// * `Ok(KFState)` – Propagated state at `obs.mjd_tt()`.
/// * `Err(PropagateError)` – Propagated from [`propagate_to_epoch`] or from
///   the observer state resolver.
pub(crate) fn propagate_kf<'state_lf>(
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
    PropagationEvent::JacobianAtEpoch {
        label: "current",
        det: j.determinant(),
    }
    .emit();

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
    let inflation_chi2_threshold = kf.shared_ctx.config.inflation_chi2_threshold;
    let max_inflation = kf.shared_ctx.config.max_inflation;
    let lambda = kf.nis_ema.map_or(1.0, |ema| {
        (ema / inflation_chi2_threshold).clamp(1.0, max_inflation)
    });

    let p_cart_new = lambda * (stm * p_cart * stm.transpose());

    PropagationEvent::InflationFactor {
        nis_ema: kf.nis_ema.unwrap_or(f64::NAN),
        lambda,
        inflation_active: lambda > 1.0,
    }
    .emit();

    // `jacobian_attr_to_cart` has 4 of its 6 columns scaling linearly with ρ
    // (state[4]): as ρ drifts toward 0 or diverges, the Jacobian becomes
    // ill-conditioned without ever being *exactly* singular, so
    // `try_inverse()` below would still succeed but return a hugely
    // amplified inverse — applied on both sides of the propagated
    // covariance, silently inflating it by many orders of magnitude while
    // staying `f64`-finite. Reject propagation before that happens rather
    // than detecting it after the fact.
    let rho_au = attr_new[4];
    if !(MIN_RHO_AU..=MAX_RHO_AU).contains(&rho_au) {
        return Err(PropagateError::IllConditionedRange { rho_au });
    }

    let j_new = jacobian_attr_to_cart(attr_new);
    PropagationEvent::JacobianAtEpoch {
        label: "propagated",
        det: j_new.determinant(),
    }
    .emit();

    let j_new_inv = j_new
        .try_inverse()
        .ok_or(PropagateError::SingularJacobian)?;

    let q_cart = build_snc_process_noise(dt, q0, dt_ref);
    let q_attr = j_new_inv * q_cart * j_new_inv.transpose();

    PropagationEvent::ProcessNoise {
        q_snc_effective: q0 * (1.0 + (dt / 1.0_f64).powi(2)),
        q_attr_trace: q_attr.trace(),
        dt_days: dt,
    }
    .emit();

    Ok(j_new_inv * p_cart_new * j_new_inv.transpose() + q_attr)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod propagate_tests {
    use std::convert::Infallible;

    use proptest::prelude::*;

    use super::*;

    /// Independent oracle for the light-time equation `‖A − v·τ‖ = c·τ`
    /// (with `A = r0 − r_obs`, for an object moving at constant heliocentric
    /// velocity `v` and evaluated at the reception epoch `t_prop`, i.e.
    /// `obj_pos(t) = r0 + v·(t − t_prop)`). Solved by bisection — a
    /// different numerical method from `refine_light_time_tau`'s fixed-point
    /// iteration, so a match between the two isn't just both converging to
    /// the same wrong answer via the same recursion.
    ///
    /// Bracket: `f(0) = |A| > 0`; `f(τ_hi) ≤ 0` at `τ_hi = |A| / (c − |v|)`
    /// by the triangle inequality (`‖A − vτ‖ ≤ |A| + |v|τ`), valid whenever
    /// `|v| < c` (always true for any physical heliocentric velocity).
    fn closed_form_tau(r0: Vector3<f64>, v: Vector3<f64>, r_obs: Vector3<f64>) -> f64 {
        let a = r0 - r_obs;
        let v_norm = v.norm();
        assert!(v_norm < C_AU_PER_DAY, "v must be sub-luminal");

        let f = |tau: f64| (a - v * tau).norm() - C_AU_PER_DAY * tau;

        let mut lo = 0.0;
        let mut hi = a.norm() / (C_AU_PER_DAY - v_norm);
        assert!(f(lo) >= 0.0 && f(hi) <= 0.0, "bisection bracket invalid");

        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) >= 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// `propagate_at` stand-in for a fictional object moving at constant
    /// heliocentric velocity `v`, anchored so that `obj_pos(t_prop) = r0`
    /// (see [`closed_form_tau`]'s doc for the exact convention).
    fn constant_velocity_propagator(
        r0: Vector3<f64>,
        v: Vector3<f64>,
        t_prop: f64,
    ) -> impl FnMut(f64) -> Result<Vector3<f64>, Infallible> {
        move |t_target: f64| Ok(r0 + v * (t_target - t_prop))
    }

    #[test]
    fn static_object_needs_no_refinement() {
        let r_obs = Vector3::new(1.0, 0.0, 0.0);
        let r0 = Vector3::new(2.5, 0.3, 0.0); // rho ~ 1.53 AU from r_obs
        let t_prop = 60_000.0;

        let (tau, pos) = refine_light_time_tau(
            constant_velocity_propagator(r0, Vector3::zeros(), t_prop),
            |p: &Vector3<f64>| *p,
            r_obs,
            t_prop,
            0.0,
            LIGHT_TIME_MAX_ITERS,
            LIGHT_TIME_TAU_TOL_DAYS,
        )
        .unwrap();

        let expected_tau = (r0 - r_obs).norm() / C_AU_PER_DAY;
        assert!((tau - expected_tau).abs() < 1e-12);
        assert_eq!(pos, r0); // static object: emission-epoch position == r0 always
    }

    #[test]
    fn matches_closed_form_oracle_mba_regime() {
        // rho ~ 2 AU, slow apparent motion typical of a well-observed MBA.
        let r_obs = Vector3::new(1.0, 0.0, 0.0);
        let r0 = Vector3::new(3.0, 0.2, 0.05);
        let v = Vector3::new(-0.002, 0.01, 0.0005); // AU/day
        let t_prop = 60_000.0;

        let expected = closed_form_tau(r0, v, r_obs);
        let (tau, _) = refine_light_time_tau(
            constant_velocity_propagator(r0, v, t_prop),
            |p: &Vector3<f64>| *p,
            r_obs,
            t_prop,
            0.0, // deliberately bad initial guess (as if tau_prev were unknown)
            LIGHT_TIME_MAX_ITERS,
            LIGHT_TIME_TAU_TOL_DAYS,
        )
        .unwrap();

        assert!(
            (tau - expected).abs() < 1e-8,
            "tau={tau}, expected={expected}"
        );
    }

    #[test]
    fn matches_closed_form_oracle_neo_regime() {
        // rho ~ 0.2 AU, fast apparent motion typical of a close-approach NEO.
        let r_obs = Vector3::new(1.0, 0.0, 0.0);
        let r0 = Vector3::new(1.15, 0.12, 0.02);
        let v = Vector3::new(-0.15, 0.4, 0.02); // AU/day — much faster than the MBA case
        let t_prop = 60_000.0;

        let expected = closed_form_tau(r0, v, r_obs);
        let (tau, _) = refine_light_time_tau(
            constant_velocity_propagator(r0, v, t_prop),
            |p: &Vector3<f64>| *p,
            r_obs,
            t_prop,
            0.0,
            LIGHT_TIME_MAX_ITERS,
            LIGHT_TIME_TAU_TOL_DAYS,
        )
        .unwrap();

        assert!(
            (tau - expected).abs() < 1e-8,
            "tau={tau}, expected={expected}"
        );
    }

    /// Demonstrates why a single unconditional refinement (the pre-fix
    /// behaviour, `max_iters = 2`) can leave a real residual for a fast NEO
    /// starting from a poor `τ` guess (e.g. a freshly seeded hypothesis
    /// whose stored ρ hasn't converged yet) — while the bounded
    /// convergence loop (`max_iters = 5`, `LIGHT_TIME_MAX_ITERS`) closes it.
    /// The gap is reified in arcsec-equivalent along-track bias
    /// (`Δτ × angular rate`) to connect it to the along-track signature
    /// observed in `mot_analysis`.
    #[test]
    fn single_refinement_can_leave_a_residual_for_fast_neo() {
        let r_obs = Vector3::new(1.0, 0.0, 0.0);
        let r0 = Vector3::new(1.05, 0.25, 0.03); // rho ~ 0.27 AU
        let v = Vector3::new(-0.3, 0.8, 0.05); // AU/day — extreme close-approach NEO
        let t_prop = 60_000.0;
        let tau_init = 0.0; // worst-case bootstrap guess

        let expected = closed_form_tau(r0, v, r_obs);

        let (tau_old, _) = refine_light_time_tau(
            constant_velocity_propagator(r0, v, t_prop),
            |p: &Vector3<f64>| *p,
            r_obs,
            t_prop,
            tau_init,
            2,   // pre-fix behaviour: exactly one refinement, no convergence check
            0.0, // tol=0 disables early-exit, forcing exactly `max_iters` calls
        )
        .unwrap();

        let (tau_new, _) = refine_light_time_tau(
            constant_velocity_propagator(r0, v, t_prop),
            |p: &Vector3<f64>| *p,
            r_obs,
            t_prop,
            tau_init,
            LIGHT_TIME_MAX_ITERS,
            LIGHT_TIME_TAU_TOL_DAYS,
        )
        .unwrap();

        let angular_rate = v.norm() / (r0 - r_obs).norm(); // rad/day, order-of-magnitude
        let rad_to_arcsec = 206_264.80624709636;
        let old_bias_arcsec = (tau_old - expected).abs() * angular_rate * rad_to_arcsec;
        let new_bias_arcsec = (tau_new - expected).abs() * angular_rate * rad_to_arcsec;

        assert!(
            old_bias_arcsec > 1.0,
            "expected the pre-fix single refinement to leave a >1\" residual for this NEO regime, got {old_bias_arcsec}\""
        );
        assert!(
            new_bias_arcsec < 1e-2,
            "the bounded convergence loop should close the residual to <10 mas, got {new_bias_arcsec}\""
        );
    }

    proptest! {
        /// Across the whole MBA→NEO physical range, the production-configured
        /// helper (`LIGHT_TIME_MAX_ITERS`, `LIGHT_TIME_TAU_TOL_DAYS`) always
        /// converges to the independent bisection oracle, regardless of how
        /// bad the initial `τ` guess is (0 to a wildly wrong value) — the
        /// property a fresh seeding-grid hypothesis needs.
        #[test]
        fn converges_to_oracle_across_physical_range(
            rho in 0.05f64..3.5,
            angular_rate_arcsec_per_day in 1.0f64..50_000.0f64, // slow MBA to fast NEO
            radial_frac in -0.3f64..0.3, // rho_dot as a fraction of rho, loosely
            tau_init_frac in 0.0f64..3.0, // 0 = no guess, up to 3x the true tau
        ) {
            let r_obs = Vector3::new(1.0, 0.0, 0.0);
            let r0 = Vector3::new(1.0 + rho, 0.0, 0.0);
            let angular_rate = angular_rate_arcsec_per_day / 206_264.80624709636; // rad/day
            let v_tangential = rho * angular_rate;
            let v = Vector3::new(radial_frac * rho, v_tangential, 0.0);
            let t_prop = 60_000.0;

            prop_assume!(v.norm() < 0.5 * C_AU_PER_DAY); // stay sub-luminal with margin

            let expected = closed_form_tau(r0, v, r_obs);
            let tau_init = (tau_init_frac * expected).max(0.0);

            let (tau, _) = refine_light_time_tau(
                constant_velocity_propagator(r0, v, t_prop),
                |p: &Vector3<f64>| *p,
                r_obs,
                t_prop,
                tau_init,
                LIGHT_TIME_MAX_ITERS,
                LIGHT_TIME_TAU_TOL_DAYS,
            )
            .unwrap();

            prop_assert!(
                (tau - expected).abs() < 1e-7,
                "tau={tau}, expected={expected}, rho={rho}, angular_rate={angular_rate_arcsec_per_day}\"/day"
            );
        }
    }
}

/// Cross-checks `propagate_universal` (the two-body Kepler solver
/// `propagate_to_epoch` relies on) against an independently-derived
/// analytic Keplerian orbit — distinct from `propagate_tests`, which
/// covers the light-time correction *on top of* whatever `propagate_universal`
/// returns. A bug in the solver itself and unmodeled planetary
/// perturbations produce the same along-track-only signature against real
/// astrometry (see the plan doc), so this test exists to rule the solver
/// in or out on its own, before attributing any residual to missing
/// physics.
#[cfg(test)]
mod kepler_solver_tests {
    use proptest::prelude::*;

    use super::*;

    const MU_SUN: f64 = outfit::GAUSS_GRAV * outfit::GAUSS_GRAV; // AU^3/day^2, heliocentric

    /// Solve Kepler's equation `M = E - e·sin(E)` for the eccentric anomaly
    /// `E` via Newton-Raphson — independent of outfit's own solver, which
    /// uses the universal-variable formulation, not this classical
    /// elliptical one.
    fn solve_eccentric_anomaly(mean_anomaly: f64, e: f64) -> f64 {
        let m = mean_anomaly.rem_euclid(std::f64::consts::TAU);
        let mut ecc = if e < 0.8 { m } else { std::f64::consts::PI };
        for _ in 0..100 {
            let f = ecc - e * ecc.sin() - m;
            let f_prime = 1.0 - e * ecc.cos();
            let delta = f / f_prime;
            ecc -= delta;
            if delta.abs() < 1e-14 {
                break;
            }
        }
        ecc
    }

    /// Position and velocity (AU, AU/day) of a pure two-body Keplerian
    /// orbit at eccentric anomaly `E`, confined to the `z = 0` plane
    /// (`i = 0`, periapsis along `+x`) so no 3D rotation is needed — the
    /// solver being tested operates on Cartesian state regardless of
    /// orientation, so this loses no generality for a numerical-agreement
    /// check.
    fn kepler_state_at_eccentric_anomaly(
        a: f64,
        e: f64,
        n: f64,
        ecc: f64,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let (sin_e, cos_e) = ecc.sin_cos();
        let x = a * (cos_e - e);
        let y = a * (1.0 - e * e).sqrt() * sin_e;
        let x_dot = -a * n * sin_e / (1.0 - e * cos_e);
        let y_dot = a * n * (1.0 - e * e).sqrt() * cos_e / (1.0 - e * cos_e);
        (Vector3::new(x, y, 0.0), Vector3::new(x_dot, y_dot, 0.0))
    }

    /// Independently-derived two-body Keplerian position at `t1`, given the
    /// orbit's elements and mean anomaly `m0` at `t0` — never calls
    /// `propagate_universal` or any outfit solver.
    fn analytic_kepler_position(a: f64, e: f64, m0: f64, n: f64, t0: f64, t1: f64) -> Vector3<f64> {
        let m1 = m0 + n * (t1 - t0);
        let ecc1 = solve_eccentric_anomaly(m1, e);
        kepler_state_at_eccentric_anomaly(a, e, n, ecc1).0
    }

    proptest! {
        /// `propagate_universal` must reproduce a pure two-body orbit to
        /// numerical precision (not just "close"), over arc lengths
        /// spanning the ones seen in `mot_analysis`'s `not_matched`
        /// bucket (up to several months). The tolerance (1e-3″-equivalent,
        /// where "arcsec" here is the position error converted via the
        /// orbit's own heliocentric distance — a precision metric, not a
        /// claim about apparent geocentric separation) is three orders of
        /// magnitude tighter than the ~28″ along-track bias observed
        /// against real ZTF astrometry: if this test passes comfortably,
        /// the solver itself cannot be the source of that bias, and it
        /// must come from missing physics (perturbations) or a biased
        /// orbit fit instead.
        #[test]
        fn matches_analytic_two_body_orbit(
            a in 0.3f64..4.0,       // AU: NEO-ish to outer MBA
            e in 0.0f64..0.85,
            m0 in 0.0f64..std::f64::consts::TAU,
            arc_days in prop_oneof![Just(7.0), Just(30.0), Just(90.0), Just(180.0)],
        ) {
            let n = (MU_SUN / a.powi(3)).sqrt();
            let ecc0 = solve_eccentric_anomaly(m0, e);
            let (pos0, vel0) = kepler_state_at_eccentric_anomaly(a, e, n, ecc0);

            let t0 = 60_000.0;
            let t1 = t0 + arc_days;

            let result = propagate_universal(&pos0, &vel0, t0, t1, SolverType::default())
                .expect("propagate_universal should converge for a well-posed elliptical orbit");

            let expected = analytic_kepler_position(a, e, m0, n, t0, t1);
            let error_au = (result.r1 - expected).norm();
            let error_arcsec = (error_au / result.r1.norm()).to_degrees() * 3600.0;

            prop_assert!(
                error_arcsec < 1e-3,
                "solver diverges from the analytic two-body orbit by {error_arcsec}\" \
                 (a={a}, e={e}, m0={m0}, arc_days={arc_days})"
            );
        }
    }
}

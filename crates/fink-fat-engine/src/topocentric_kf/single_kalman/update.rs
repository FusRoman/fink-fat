use std::f64::consts::{PI, TAU};

use nalgebra::{Matrix2, Matrix2x6, Matrix6, Matrix6x2, Vector2, Vector6};
use photom::observation_dataset::observation::Observation;

use crate::{
    error::{KFUpdateError, ObservationJacobianError},
    topocentric_kf::single_kalman::KFState,
};

/// Smoothing factor for the exponential moving average (EMA) of the NIS.
///
/// With `α = 2 / (N + 1)`, the value `0.4` corresponds to an effective window
/// of `N ≈ 4` update steps. The EMA is deliberately **step-based**, not
/// time-based: it smooths over recent *observations* so a single astrometric
/// outlier cannot, on its own, drive the adaptive inflation — while the
/// irregular time gaps of a survey cadence leave the smoothing unaffected.
const NIS_EMA_ALPHA: f64 = 0.4;

/// Wrap an angle to the interval $(-\pi, \pi]$.
///
/// Used to fold the right-ascension component of the innovation across the
/// $0 / 2\pi$ discontinuity so that, e.g., an observed RA of $0.01$ rad and a
/// predicted RA of $6.27$ rad produce a small residual rather than a $\sim 2\pi$
/// one.
#[inline]
pub fn wrap_angle(angle: f64) -> f64 {
    let wrapped = angle.rem_euclid(2.0 * PI);
    if wrapped > PI {
        wrapped - 2.0 * PI
    } else {
        wrapped
    }
}

/// Observation Jacobian $H \in \mathbb{R}^{2 \times 6}$ for the attributable
/// measurement model.
///
/// # Model
///
/// In **attributable coordinates** the state vector is
///
/// $$\mathbf{x} = (\alpha,\, \delta,\, \dot{\alpha},\, \dot{\delta},\, \rho,\, \dot{\rho})^\top$$
///
/// and a single astrometric observation provides $\mathbf{z} = (\alpha, \delta)$.
/// The observation function is therefore the linear projection
///
/// $$h(\mathbf{x}) = (x_0,\, x_1) = (\alpha,\, \delta)$$
///
/// whose Jacobian is the **constant** selection matrix
///
/// $$H = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{pmatrix}$$
///
/// This is the central advantage of the attributable parameterization: the
/// measurement model is exact and linear, so the update step introduces no
/// linearization error and $H$ depends neither on the current state nor on the
/// observer geometry.
///
/// # Arguments
///
/// * `_state` – Current attributable state. **Unused**; retained only for
///   interface compatibility with the previous Cartesian formulation and with
///   the call sites in [`KFState::sky_covariance`] and [`KFState::to_equ_coord`].
/// * `_observer_pos` – Observer heliocentric position. **Unused**, for the same
///   reason.
///
/// # Returns
///
/// Always `Ok(H)` with the constant selection matrix above. The `Result` return
/// type is kept so existing callers using `?` continue to compile unchanged.
pub(crate) fn observation_jacobian() -> Result<Matrix2x6<f64>, ObservationJacobianError> {
    let mut jacobian = Matrix2x6::zeros();
    jacobian[(0, 0)] = 1.0; // ∂α_obs / ∂α
    jacobian[(1, 1)] = 1.0; // ∂δ_obs / ∂δ
    Ok(jacobian)
}

/// Predicted measurement $h(\mathbf{x}^-) = (\alpha,\, \delta)$.
///
/// In attributable coordinates the predicted sky angles are simply the first
/// two state components — no projection through the observer geometry is needed.
#[inline]
fn predicted_observation(state: &Vector6<f64>) -> Vector2<f64> {
    Vector2::new(state[0], state[1]) // (α, δ)
}

/// Measurement innovation $\nu = \mathbf{z} - h(\mathbf{x}^-)$.
///
/// The right-ascension component is wrapped to $(-\pi, \pi]$ to handle the
/// $0 / 2\pi$ boundary. Declination lives in $[-\pi/2, \pi/2]$ and never wraps,
/// so it is left untouched.
#[inline]
fn measurement_innovation(measurement: Vector2<f64>, predicted: Vector2<f64>) -> Vector2<f64> {
    let mut innovation = measurement - predicted;
    innovation[0] = wrap_angle(innovation[0]); // wrap RA residual only
    innovation
}

/// Diagonal measurement-noise covariance $R = \mathrm{diag}(\sigma_\alpha^2,\, \sigma_\delta^2)$
/// in rad².
///
/// Built directly from the 1-σ astrometric uncertainties carried by the
/// observation.
#[inline]
fn observation_noise(ra_error: f64, dec_error: f64) -> Matrix2<f64> {
    Matrix2::from_diagonal(&Vector2::new(ra_error * ra_error, dec_error * dec_error))
}

/// Relative jitter step for [`regularize_covariance_2x2`], as a fraction of
/// the matrix's trace, applied per escalation round.
const SKY_COVARIANCE_JITTER_RATIO: f64 = 1e-6;

/// Absolute jitter floor (rad²) for [`regularize_covariance_2x2`], used when
/// the trace itself is at/near zero (so a purely relative jitter would be
/// zero too). Utterly negligible physically (≪ 1 µas²) — only there so the
/// jitter is never exactly zero.
const SKY_COVARIANCE_JITTER_ABS: f64 = 1e-20;

/// How confidently positive `det(S)` must be, relative to `trace(S)²`,
/// before [`regularize_covariance_2x2`] considers a candidate matrix "safe"
/// — i.e. safely above the floating-point round-off floor of the direct
/// `a·d - b·b` computation any consumer (this module's [`update_kf`],
/// [`super::hypothesis::compute_log_likelihood`]) will redo independently.
const MIN_RELATIVE_DET_RATIO: f64 = 1e-12;

/// Escalation rounds [`regularize_covariance_2x2`] allows before giving up
/// and returning its best attempt — in practice 1-2 rounds always suffice
/// (each multiplies the jitter's effect roughly geometrically relative to
/// the fixed target ratio), this is just a hard safety bound.
const MAX_REGULARIZE_ROUNDS: u32 = 8;

/// Symmetrize a 2×2 sky covariance and, if needed, nudge it towards a
/// comfortably positive-definite matrix — verified by directly
/// recomputing the determinant the same way a consumer will, rather than
/// trusting a one-shot eigenvalue calculation.
///
/// # Why this exists
///
/// `S = H·P·Hᵀ + R` (the innovation covariance used for gating, the
/// predictive log-likelihood, and the Kalman gain) is, mathematically,
/// always symmetric positive semi-definite. But `H` is a pure row-selector
/// (see [`observation_jacobian`]), so `H·P·Hᵀ` is literally the top-left 2×2
/// block of the 6×6 state covariance `P` — and nothing in this crate's
/// predict/update/merge path (`propagate_covariance`, [`update_kf`],
/// `Hypothesis::moment_match_merge`) ever symmetrizes `P` or floors its
/// eigenvalues. After enough real updates (observed in production after
/// ~10-20+), `P` can shrink to a magnitude comparable to `f64` round-off, or
/// become extremely ill-conditioned (RA/Dec correlation near ±1), at which
/// point the extracted 2×2 block can come out **not even symmetric**, with
/// a **negative** diagonal entry, or with a determinant that flickers
/// across zero depending on which formula recomputes it — impossible for a
/// true covariance, and fatal downstream: [`super::hypothesis::compute_log_likelihood`]
/// and `nalgebra`'s `try_inverse()` both require a genuinely
/// positive-definite `S`, so an unregularized `S` collapses the whole
/// hypothesis (and, if it was the bank's only/protected one, the entire
/// lineage) over what is actually just floating-point noise, not a real
/// statistical inconsistency.
///
/// An earlier version of this function computed the eigenvalues once via
/// `mid ± sqrt(half_diff² + b²)` and shifted analytically to a target
/// floor. That is exact in real arithmetic, but for the most
/// ill-conditioned observed cases (`mid` and `delta` both ~1e8-1e9 while
/// the true `lambda_min` is a few hundred) computing `lambda_min = mid -
/// delta` is itself a catastrophic-cancellation subtraction — on exactly
/// the class of matrix this function targets. Worse, even a mathematically
/// correct shift could still read back as non-positive once the *consumer*
/// (`compute_log_likelihood`, `try_inverse()`) independently recomputes
/// the determinant its own way. This version instead escalates a
/// trace-proportional jitter and **directly checks the same `a·d - b·b`
/// formula the consumers use** after each round, so the guarantee is
/// empirical, not just analytical.
///
/// This is called at both places `S = H·P·Hᵀ + R` is computed (this
/// module's [`update_kf`] and `hypothesis::measurement_innovation`) so they
/// stay consistent with each other.
///
/// # What it does *not* do
///
/// It does not touch `P` itself — only the 2×2 `S` derived from it, at the
/// point it's about to be used for a statistical decision. `P` can still
/// carry the same tiny asymmetry into the next propagate/update step; this
/// is a targeted fix for the reported failure mode, not a general
/// numerical overhaul of the covariance pipeline.
pub(crate) fn regularize_covariance_2x2(s: Matrix2<f64>) -> Matrix2<f64> {
    let sym = (s + s.transpose()) * 0.5;

    let a0 = sym[(0, 0)];
    let b0 = sym[(0, 1)];
    let d0 = sym[(1, 1)];

    // Closed-form eigenvalues of the symmetrized input — a *guide* for how
    // much to shift, not the final word: for the most ill-conditioned
    // inputs this `mid - delta` subtraction can itself lose precision (see
    // this function's doc), so any decision it informs is always verified
    // below by directly recomputing the determinant, and escalated
    // geometrically (not just once) if the guide undershot.
    let mid = (a0 + d0) / 2.0;
    let half_diff = (a0 - d0) / 2.0;
    let delta = (half_diff * half_diff + b0 * b0).sqrt();
    let lambda_min_guide = mid - delta;
    let lambda_max_guide = (mid + delta).abs();

    let scale = lambda_max_guide
        .max(a0.abs())
        .max(d0.abs())
        .max(SKY_COVARIANCE_JITTER_ABS);
    let min_det = (scale * scale * MIN_RELATIVE_DET_RATIO).max(SKY_COVARIANCE_JITTER_ABS);

    // Fast path: already comfortably positive-definite by direct
    // recomputation — leave it untouched (no perturbation of an
    // already-sane covariance, and trivially idempotent).
    let det0 = a0 * d0 - b0 * b0;
    if det0 >= min_det && a0 > 0.0 && d0 > 0.0 {
        return sym;
    }

    // Initial guess: enough to bring the (possibly imprecise) `lambda_min`
    // estimate up to a small positive target.
    let target_floor = scale * SKY_COVARIANCE_JITTER_RATIO;
    let mut jitter = (target_floor - lambda_min_guide).max(target_floor);

    for _ in 0..MAX_REGULARIZE_ROUNDS {
        let candidate = sym + Matrix2::identity() * jitter;
        let a = candidate[(0, 0)];
        let b = candidate[(0, 1)];
        let d = candidate[(1, 1)];
        let det = a * d - b * b;

        if det >= min_det && a > 0.0 && d > 0.0 {
            return candidate;
        }

        // The guide undershot (extreme condition number, or its own
        // cancellation error) — escalate geometrically rather than
        // creeping up by a fixed small step, so this converges in a
        // handful of rounds regardless of how far off the guide was.
        jitter = (jitter * 10.0).max(target_floor);
    }

    sym + Matrix2::identity() * jitter
}

/// Closed-form inverse of a 2×2 matrix — exact, no internal
/// invertibility heuristic (unlike `nalgebra`'s generic `try_inverse()`,
/// which rejected some [`regularize_covariance_2x2`]-regularized matrices
/// whose determinant is, in fact, comfortably positive by direct
/// computation — see that function's doc). Only meant to be called on a
/// matrix already known to be safely positive-definite (i.e. straight
/// after [`regularize_covariance_2x2`]); returns `None` only for a
/// genuinely non-positive or non-finite determinant, as a last-resort
/// safety net.
pub(crate) fn try_invert_2x2(s: &Matrix2<f64>) -> Option<Matrix2<f64>> {
    let a = s[(0, 0)];
    let b = s[(0, 1)];
    let c = s[(1, 0)];
    let d = s[(1, 1)];
    let det = a * d - b * c;
    if !det.is_finite() || det <= 0.0 {
        return None;
    }
    Some(Matrix2::new(d, -b, -c, a) / det)
}

use crate::logging::LogTarget;

/// Structured log events for the KF measurement update (innovation, gain,
/// NIS, state/covariance update). See [`crate::logging`] for the `.emit()`
/// pattern.
pub enum UpdateEvent {
    Observed {
        ra_obs_deg: f64,
        dec_obs_deg: f64,
        sigma_ra_rad: f64,
        sigma_dec_rad: f64,
    },
    Innovation {
        innovation_ra_arcsec: f64,
        innovation_dec_arcsec: f64,
    },
    InnovationCovariance {
        s_00: f64,
        s_11: f64,
        s_det: f64,
    },
    SingularInnovationCovariance {
        s_det: f64,
    },
    NisCheck {
        nis: f64,
        nis_expected: f64,
        within_95pct_bounds: bool,
    },
    NisExceedsBound {
        nis: f64,
        innovation_ra_arcsec: f64,
        innovation_dec_arcsec: f64,
    },
    NisEma {
        nis: f64,
        nis_ema: f64,
    },
    KalmanGain {
        k_frobenius: f64,
    },
    UpdatedState {
        ra_deg: f64,
        dec_deg: f64,
        ra_dot_rad_per_d: f64,
        dec_dot_rad_per_d: f64,
        rho_au: f64,
        rho_dot_au_per_d: f64,
    },
    UpdatedCovariance {
        trace_before: f64,
        trace_after: f64,
        var_ra_after: f64,
        var_dec_after: f64,
        var_rho_after: f64,
    },
    Complete,
}

crate::impl_log_target!(
    UpdateEvent,
    "update",
    "Kalman measurement update: innovation, gain, NIS, state/covariance update",
    [tracing::Level::TRACE, tracing::Level::ERROR]
);

impl UpdateEvent {
    pub fn emit(&self) {
        use UpdateEvent::*;
        match self {
            Observed {
                ra_obs_deg,
                dec_obs_deg,
                sigma_ra_rad,
                sigma_dec_rad,
            } => tracing::trace!(
                target: UpdateEvent::TARGET, ra_obs_deg, dec_obs_deg, sigma_ra_rad, sigma_dec_rad,
                "Observed (RA, Dec) and astrometric uncertainties"
            ),
            Innovation {
                innovation_ra_arcsec,
                innovation_dec_arcsec,
            } => tracing::trace!(
                target: UpdateEvent::TARGET, innovation_ra_arcsec, innovation_dec_arcsec,
                "Innovation ν = z − h(x⁻)"
            ),
            InnovationCovariance { s_00, s_11, s_det } => tracing::trace!(
                target: UpdateEvent::TARGET, s_00, s_11, s_det, "Innovation covariance S"
            ),
            SingularInnovationCovariance { s_det } => tracing::error!(
                target: UpdateEvent::TARGET, s_det, "Innovation covariance S is singular — cannot invert"
            ),
            NisCheck {
                nis,
                nis_expected,
                within_95pct_bounds,
            } => tracing::trace!(
                target: UpdateEvent::TARGET, nis, nis_expected, within_95pct_bounds,
                "NIS check — χ²(2) @ 95 %: [0.05, 7.38]"
            ),
            NisExceedsBound {
                nis,
                innovation_ra_arcsec,
                innovation_dec_arcsec,
            } => tracing::trace!(
                target: UpdateEvent::TARGET, nis, innovation_ra_arcsec, innovation_dec_arcsec,
                "Filter inconsistency: NIS exceeds χ²(2) 95% upper bound — \
                 covariance likely too small (overconfident filter)"
            ),
            NisEma { nis, nis_ema } => tracing::trace!(
                target: UpdateEvent::TARGET, nis, nis_ema,
                "NIS exponential moving average (drives adaptive inflation)"
            ),
            KalmanGain { k_frobenius } => tracing::trace!(
                target: UpdateEvent::TARGET, k_frobenius, "Kalman gain K (Frobenius norm)"
            ),
            UpdatedState {
                ra_deg,
                dec_deg,
                ra_dot_rad_per_d,
                dec_dot_rad_per_d,
                rho_au,
                rho_dot_au_per_d,
            } => tracing::trace!(
                target: UpdateEvent::TARGET, ra_deg, dec_deg, ra_dot_rad_per_d, dec_dot_rad_per_d, rho_au, rho_dot_au_per_d,
                "Updated attributable state x⁺"
            ),
            UpdatedCovariance {
                trace_before,
                trace_after,
                var_ra_after,
                var_dec_after,
                var_rho_after,
            } => tracing::trace!(
                target: UpdateEvent::TARGET, trace_before, trace_after, var_ra_after, var_dec_after, var_rho_after,
                "Covariance trace/variances before and after update (Joseph form)"
            ),
            Complete => tracing::trace!(target: UpdateEvent::TARGET, "KF update complete."),
        }
    }

    pub fn span(epoch: f64) -> tracing::Span {
        tracing::trace_span!(target: UpdateEvent::TARGET, "kf_update", epoch)
    }
}

/// Floor the angular-position variances `covariance[(0,0)]`/`[(1,1)]` at
/// `ratio * r_mat[(i,i)]`, in place.
///
/// Nothing else in the update pipeline stops `Var(α)`/`Var(δ)` from
/// shrinking arbitrarily far below the astrometric noise floor `R` — a run
/// of closely-spaced real updates (e.g. several exposures in the same
/// visit) can do this in a handful of steps. Once that happens, `S =
/// P[0:2,0:2] + R` approaches `R` from above, `S⁻¹` grows very large, and
/// even a modest ρ↔angle covariance in `P` gets amplified into a
/// destabilizing correction on ρ (see the `min_angular_variance_ratio` doc
/// on [`crate::engine_config::kalman_context::KalmanContext`]'s
/// `KalmanConfig` for the investigation this responds to).
///
/// Only the diagonal is touched — same philosophy as
/// [`regularize_covariance_2x2`] for `S`: don't touch the cross-terms,
/// just keep the variance itself physically defensible. `ratio <= 0.0` is
/// a no-op (disables the floor).
fn apply_angular_variance_floor(covariance: &mut Matrix6<f64>, r_mat: &Matrix2<f64>, ratio: f64) {
    let min_var_ra = r_mat[(0, 0)] * ratio;
    let min_var_dec = r_mat[(1, 1)] * ratio;
    if covariance[(0, 0)] < min_var_ra {
        covariance[(0, 0)] = min_var_ra;
    }
    if covariance[(1, 1)] < min_var_dec {
        covariance[(1, 1)] = min_var_dec;
    }
}

/// Perform a linear Kalman measurement update in attributable coordinates from
/// a new $(\alpha, \delta)$ observation.
///
/// # Update equations
///
/// With the linear measurement model $h(\mathbf{x}) = (\alpha, \delta)$ and the
/// constant Jacobian $H = [\,I_2 \;|\; 0\,]$ (see [`observation_jacobian`]):
///
/// $$\nu = \mathbf{z} - h(\mathbf{x}^-), \qquad
///   S = H P^- H^\top + R, \qquad
///   K = P^- H^\top S^{-1}$$
///
/// $$\mathbf{x}^+ = \mathbf{x}^- + K\,\nu, \qquad
///   P^+ = (I - KH)\,P^-\,(I - KH)^\top + K R K^\top$$
///
/// The covariance is updated in **Joseph form** to preserve symmetry and
/// positive-definiteness numerically.
///
/// Because the measurement model is exact and linear, this is a standard
/// (non-extended) Kalman update: no Jacobian linearization error is incurred.
///
/// # Arguments
///
/// * `kf` – Predicted Kalman state at the observation epoch. The caller is
///   responsible for propagating to `new_obs`'s epoch **before** calling this.
/// * `new_obs` – The new astrometric observation, providing $(\alpha, \delta)$
///   and their 1-σ uncertainties (equatorial J2000).
/// * `r_obs` – Observer heliocentric position. **Not used** by the attributable
///   update; it is only forwarded to [`observation_jacobian`] for signature
///   compatibility and has no effect on the result.
///
/// # Returns
///
/// * `Ok(KFState)` – Updated state and covariance at the same epoch as `kf`.
/// * `Err(KFUpdateError::SingularInnovationCovariance)` – If $S$ cannot be
///   inverted.
pub(crate) fn update_kf<'state_lf>(
    kf: KFState<'state_lf>,
    new_obs: &Observation,
) -> Result<KFState<'state_lf>, KFUpdateError> {
    let _enter = UpdateEvent::span(kf.epoch).entered();

    // ── Measurement z = (α_obs, δ_obs) in equatorial J2000 ─────────────────
    let coord = new_obs.equ_coord();
    let measurement = Vector2::new(coord.ra, coord.dec);

    UpdateEvent::Observed {
        ra_obs_deg: coord.ra.to_degrees(),
        dec_obs_deg: coord.dec.to_degrees(),
        sigma_ra_rad: coord.ra_error,
        sigma_dec_rad: coord.dec_error,
    }
    .emit();

    // ── Predicted measurement h(x⁻) = (α⁻, δ⁻) ─────────────────────────────
    // The predicted angles ARE the first two state components in attributable
    // coordinates — no Cartesian-to-RA/Dec projection is required.
    let predicted = predicted_observation(&kf.state);

    // ── Innovation ν = z − h(x⁻), with the RA residual wrapped ─────────────
    let innovation = measurement_innovation(measurement, predicted);

    UpdateEvent::Innovation {
        innovation_ra_arcsec: innovation[0].to_degrees() * 3600.0,
        innovation_dec_arcsec: innovation[1].to_degrees() * 3600.0,
    }
    .emit();

    // ── Observation Jacobian H = [I₂ | 0] (constant) ───────────────────────
    let h_mat = observation_jacobian().map_err(KFUpdateError::Jacobian)?;

    // ── Measurement noise R = diag(σ_α², σ_δ²) ─────────────────────────────
    let r_mat = observation_noise(coord.ra_error, coord.dec_error);

    // ── Innovation covariance S = H P⁻ Hᵀ + R ──────────────────────────────
    // Regularized (symmetrized + eigenvalue-floored) — see
    // `regularize_covariance_2x2`'s doc for why: `P` can drift slightly
    // non-symmetric/non-PSD after many updates, purely from f64 round-off,
    // and this is the point that decision is fatal (gate/likelihood/gain).
    let innovation_covariance =
        regularize_covariance_2x2(h_mat * kf.covariance * h_mat.transpose() + r_mat);

    UpdateEvent::InnovationCovariance {
        s_00: innovation_covariance[(0, 0)],
        s_11: innovation_covariance[(1, 1)],
        s_det: innovation_covariance.determinant(),
    }
    .emit();

    // ── Kalman gain K = P⁻ Hᵀ S⁻¹ ──────────────────────────────────────────
    // Closed-form inverse rather than `nalgebra`'s generic `try_inverse()` —
    // see `try_invert_2x2`'s doc: it rejects some already-regularized
    // matrices whose determinant is, in fact, comfortably positive.
    let innovation_covariance_inv = try_invert_2x2(&innovation_covariance).ok_or_else(|| {
        UpdateEvent::SingularInnovationCovariance {
            s_det: innovation_covariance.determinant(),
        }
        .emit();
        KFUpdateError::SingularInnovationCovariance
    })?;

    // ── NIS (Normalized Innovation Squared) ─────────────────────────────────
    // NIS = νᵀ S⁻¹ ν ~ χ²(dim_z) under filter consistency.
    // With dim(z) = 2: E[NIS] = 2, 95% bounds ≈ [0.05, 7.38].
    // NIS >> 7.38 repeatedly → filter overconfident (P too small).
    // NIS << 0.05 repeatedly → filter underconfident (P too large).
    let nis = (innovation.transpose() * innovation_covariance_inv * innovation)[(0, 0)];

    UpdateEvent::NisCheck {
        nis,
        nis_expected: 2.0,
        within_95pct_bounds: (0.05..=7.38).contains(&nis),
    }
    .emit();

    if nis > 7.38 {
        UpdateEvent::NisExceedsBound {
            nis,
            innovation_ra_arcsec: innovation[0].to_degrees() * 3600.0,
            innovation_dec_arcsec: innovation[1].to_degrees() * 3600.0,
        }
        .emit();
    }

    // ── Smoothed NIS (EMA) for adaptive covariance inflation ────────────────
    // Blend the current NIS into the running average carried from prior steps.
    // Seeded with the first raw NIS when no history exists yet. This value is
    // read by `propagate_covariance` on the *next* prediction to decide whether
    // the covariance must be re-inflated (fading-memory filter).
    let nis_ema = match kf.nis_ema {
        Some(prev) => NIS_EMA_ALPHA * nis + (1.0 - NIS_EMA_ALPHA) * prev,
        None => nis,
    };

    UpdateEvent::NisEma { nis, nis_ema }.emit();

    let kalman_gain: Matrix6x2<f64> = kf.covariance * h_mat.transpose() * innovation_covariance_inv;

    UpdateEvent::KalmanGain {
        k_frobenius: kalman_gain.norm(),
    }
    .emit();

    // ── Updated state x⁺ = x⁻ + K ν ────────────────────────────────────────
    let mut updated_state = kf.state + kalman_gain * innovation;

    // Keep RA in [0, 2π) for consistency with the init/propagate conventions
    // (both normalize α via rem_euclid(TAU)).
    updated_state[0] = updated_state[0].rem_euclid(TAU);

    UpdateEvent::UpdatedState {
        ra_deg: updated_state[0].to_degrees(),
        dec_deg: updated_state[1].to_degrees(),
        ra_dot_rad_per_d: updated_state[2],
        dec_dot_rad_per_d: updated_state[3],
        rho_au: updated_state[4],
        rho_dot_au_per_d: updated_state[5],
    }
    .emit();

    // ── Updated covariance (Joseph form) ───────────────────────────────────
    // P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ
    let i_minus_kh = Matrix6::identity() - kalman_gain * h_mat;
    let mut updated_covariance = i_minus_kh * kf.covariance * i_minus_kh.transpose()
        + kalman_gain * r_mat * kalman_gain.transpose();

    apply_angular_variance_floor(
        &mut updated_covariance,
        &r_mat,
        kf.shared_ctx.config.min_angular_variance_ratio,
    );

    UpdateEvent::UpdatedCovariance {
        trace_before: kf.covariance.trace(),
        trace_after: updated_covariance.trace(),
        var_ra_after: updated_covariance[(0, 0)],
        var_dec_after: updated_covariance[(1, 1)],
        var_rho_after: updated_covariance[(4, 4)],
    }
    .emit();

    UpdateEvent::Complete.emit();

    Ok(KFState {
        state: updated_state,
        covariance: updated_covariance,
        kalman_gain: Some(kalman_gain),
        nis_ema: Some(nis_ema),
        ..kf
    })
}

#[cfg(test)]
mod regularize_covariance_2x2_tests {
    use super::*;

    /// Independent (not reusing production code) closed-form eigenvalues of
    /// a symmetric 2x2 matrix `[[a, b], [b, d]]`, returned as `(min, max)`.
    /// Used to check properties of `regularize_covariance_2x2`'s output
    /// without the test being circular with its implementation.
    fn eigenvalues_2x2_symmetric(a: f64, b: f64, d: f64) -> (f64, f64) {
        let mid = (a + d) / 2.0;
        let half_diff = (a - d) / 2.0;
        let delta = (half_diff * half_diff + b * b).sqrt();
        (mid - delta, mid + delta)
    }

    // ── Literal regression cases ────────────────────────────────────────
    //
    // Exact innovation-covariance matrices dumped by `fink-fat-eval`'s
    // `mot_analysis` binary (see `MissReason::SearchedButNotMatched`/
    // `TruncationReason::CollapsedByPropagation` in `mot_analysis.rs`) and
    // replayed via `hypothesis::repro_propagation_failure`'s test harness —
    // all had `determinant() <= 0.0`, collapsing the hypothesis (and, being
    // the bank's only hypothesis in each case, the whole lineage) over pure
    // floating-point round-off rather than a real statistical
    // inconsistency. See `regularize_covariance_2x2`'s doc for the root
    // cause (no symmetrization anywhere in the covariance pipeline).

    #[test]
    fn regression_traj_82383_night_2955_step_2() {
        let s = Matrix2::new(
            -7.126610055882935e-14,
            -1.8155220016834643e-15,
            5.121044093039917e-16,
            1.072809572322692e-12,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_64143_night_2955_step_2() {
        let s = Matrix2::new(
            -1.2837237265074153e-11,
            8.155722712507546e-13,
            -1.9861222820417394e-12,
            1.3103678935782051e-12,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_12756_night_2954_step_23() {
        let s = Matrix2::new(
            -3.655307480843675e-12,
            -4.2840896689616284e-14,
            1.6525853102593324e-13,
            1.0344038646552826e-12,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_17452_night_3107_step_20() {
        let s = Matrix2::new(
            5.0386418790012124e-6,
            -1.7424781242966344e-6,
            -1.7448758172496e-6,
            5.981091795729314e-7,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn well_conditioned_covariance_is_unchanged() {
        let s = Matrix2::new(1.0, 0.1, 0.1, 2.0);
        let r = regularize_covariance_2x2(s);
        assert!((r - s).norm() < 1e-15);
    }

    // ── Literal regression cases, round 2 ────────────────────────────────
    //
    // These 5 survived the *first* version of `regularize_covariance_2x2`
    // (the one-shot analytical eigenvalue shift) — its own `mid - delta`
    // eigenvalue computation is itself a catastrophic-cancellation
    // subtraction for exactly this class of extreme-condition-number
    // matrix (RA/Dec correlation near ±1), and a mathematically-sufficient
    // shift could still read back as non-positive once `nalgebra`'s
    // `try_inverse()`/`compute_log_likelihood`'s own `a·d - b·b`
    // recomputed it independently. Captured from
    // `replay_propagation_failure_case`'s output on the *previous* fix.

    #[test]
    fn regression_traj_17843_night_2955_step_25() {
        let s = Matrix2::new(
            1523691.101822164,
            6069557.690412702,
            6069557.690412702,
            24177820.89374416,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_41272_night_2955_step_15() {
        let s = Matrix2::new(
            39158.23904514313,
            6520445.034503277,
            6520445.034503277,
            1085753713.2588253,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_23880_night_2953_step_22() {
        let s = Matrix2::new(
            91823.34661000967,
            2086934.8362370008,
            2086934.8362370008,
            47431259.82106919,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_4197_night_3128_step_60() {
        let s = Matrix2::new(
            251065.58563286153,
            -78181.24653050026,
            -78181.24653050026,
            24345.460544325702,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    #[test]
    fn regression_traj_18313_night_2954_step_19() {
        let s = Matrix2::new(
            530.9447328238009,
            -560.4428745307805,
            -560.4428745307805,
            591.5798692301181,
        );
        let r = regularize_covariance_2x2(s);
        assert_eq!(r[(0, 1)], r[(1, 0)]);
        assert!(r.determinant() > 0.0);
        assert!(try_invert_2x2(&r).is_some());
    }

    // ── Property-based tests ─────────────────────────────────────────────
    //
    // Same style as `topocentric_kf::branching::llr_score`'s `proptest!`
    // block (`proptest` is already a dev-dependency of this crate).

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn regularized_covariance_is_symmetric(
            a in -1e-3f64..1e-3,
            b in -1e-3f64..1e-3,
            d in -1e-3f64..1e-3,
        ) {
            let s = Matrix2::new(a, b, b + 1e-15, d); // mimics observed near-symmetric noise
            let r = regularize_covariance_2x2(s);
            prop_assert_eq!(r[(0, 1)], r[(1, 0)]);
        }

        #[test]
        fn regularized_covariance_is_positive_definite(
            a in -1e-3f64..1e-3,
            b in -1e-3f64..1e-3,
            d in -1e-3f64..1e-3,
        ) {
            let s = Matrix2::new(a, b, b, d);
            let r = regularize_covariance_2x2(s);
            prop_assert!(r.determinant() > 0.0);
            prop_assert!(r[(0, 0)] > 0.0);
            prop_assert!(r[(1, 1)] > 0.0);
            prop_assert!(try_invert_2x2(&r).is_some());
        }

        /// Extreme condition numbers (RA/Dec correlation approaching ±1, and
        /// a wide dynamic range between the two diagonal entries) — the
        /// class of input that defeated the first (analytical-shift-only)
        /// version of `regularize_covariance_2x2` — must still come out
        /// positive-definite *and* invertible via [`try_invert_2x2`].
        #[test]
        fn extreme_condition_number_is_still_regularized(
            a in 1e-6f64..1e9,
            d in 1e-6f64..1e9,
            corr in -0.999999f64..0.999999,
        ) {
            let b = corr * (a * d).sqrt();
            let s = Matrix2::new(a, b, b, d);
            let r = regularize_covariance_2x2(s);
            prop_assert!(r.determinant() > 0.0);
            prop_assert!(try_invert_2x2(&r).is_some());
        }

        /// Regularization only ever raises eigenvalues (never shrinks them) —
        /// checked against an independent eigenvalue formula, not the
        /// production code's own internals.
        #[test]
        fn regularization_never_shrinks_eigenvalues(
            a in -1e-3f64..1e-3,
            b in -1e-3f64..1e-3,
            d in -1e-3f64..1e-3,
        ) {
            let (lo_before, hi_before) = eigenvalues_2x2_symmetric(a, b, d);
            let s = Matrix2::new(a, b, b, d);
            let r = regularize_covariance_2x2(s);
            let (lo_after, hi_after) = eigenvalues_2x2_symmetric(r[(0, 0)], r[(0, 1)], r[(1, 1)]);
            prop_assert!(lo_after >= lo_before - 1e-18);
            prop_assert!(hi_after >= hi_before - 1e-18);
        }

        #[test]
        fn regularization_is_idempotent(
            a in -1e-3f64..1e-3,
            b in -1e-3f64..1e-3,
            d in -1e-3f64..1e-3,
        ) {
            let s = Matrix2::new(a, b, b, d);
            let once = regularize_covariance_2x2(s);
            let twice = regularize_covariance_2x2(once);
            prop_assert!((once - twice).norm() < 1e-9 * once.norm().max(1.0));
        }

        /// A comfortably positive-definite matrix (`|b| < 0.9*sqrt(a*d)`,
        /// `a`/`d` well above the round-off floor) must pass through
        /// unchanged — regularization should never perturb an already-sane
        /// covariance.
        #[test]
        fn well_conditioned_covariance_passes_through_unchanged(
            a in 1e-6f64..1e6,
            d in 1e-6f64..1e6,
            b_frac in -0.9f64..0.9,
        ) {
            let b = b_frac * (a * d).sqrt();
            let s = Matrix2::new(a, b, b, d);
            let r = regularize_covariance_2x2(s);
            prop_assert!((r - s).norm() < 1e-6 * s.norm().max(1.0));
        }
    }
}

#[cfg(test)]
mod angular_variance_floor_tests {
    use super::*;

    fn identity_r(sigma_ra2: f64, sigma_dec2: f64) -> Matrix2<f64> {
        Matrix2::new(sigma_ra2, 0.0, 0.0, sigma_dec2)
    }

    #[test]
    fn floors_a_collapsed_variance() {
        let mut p = Matrix6::identity() * 1e-20;
        let r = identity_r(1e-12, 1e-12);
        apply_angular_variance_floor(&mut p, &r, 1e-4);
        assert_eq!(p[(0, 0)], 1e-12 * 1e-4);
        assert_eq!(p[(1, 1)], 1e-12 * 1e-4);
    }

    #[test]
    fn leaves_a_comfortably_wide_variance_unchanged() {
        let mut p = Matrix6::identity() * 1e-3;
        let r = identity_r(1e-12, 1e-12);
        apply_angular_variance_floor(&mut p, &r, 1e-4);
        assert_eq!(p[(0, 0)], 1e-3);
        assert_eq!(p[(1, 1)], 1e-3);
    }

    #[test]
    fn zero_ratio_disables_the_floor() {
        let mut p = Matrix6::zeros();
        let r = identity_r(1e-12, 1e-12);
        apply_angular_variance_floor(&mut p, &r, 0.0);
        assert_eq!(p[(0, 0)], 0.0);
        assert_eq!(p[(1, 1)], 0.0);
    }

    #[test]
    fn does_not_touch_other_entries() {
        let mut p =
            Matrix6::from_diagonal(&nalgebra::Vector6::new(1e-10, 1e-10, 5.0, 6.0, 7.0, 8.0));
        let r = identity_r(1e-12, 1e-12);
        apply_angular_variance_floor(&mut p, &r, 1e-4);
        assert_eq!(p[(2, 2)], 5.0);
        assert_eq!(p[(3, 3)], 6.0);
        assert_eq!(p[(4, 4)], 7.0);
        assert_eq!(p[(5, 5)], 8.0);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn floored_variance_is_never_below_the_floor(
            var_ra in 0.0f64..1e-2,
            var_dec in 0.0f64..1e-2,
            sigma_ra2 in 1e-14f64..1e-8,
            sigma_dec2 in 1e-14f64..1e-8,
            ratio in 0.0f64..1.0,
        ) {
            let mut p = Matrix6::identity();
            p[(0, 0)] = var_ra;
            p[(1, 1)] = var_dec;
            let r = identity_r(sigma_ra2, sigma_dec2);
            apply_angular_variance_floor(&mut p, &r, ratio);
            prop_assert!(p[(0, 0)] >= sigma_ra2 * ratio - 1e-30);
            prop_assert!(p[(1, 1)] >= sigma_dec2 * ratio - 1e-30);
        }

        #[test]
        fn floor_never_lowers_an_already_wider_variance(
            var_ra in 0.0f64..1e-2,
            var_dec in 0.0f64..1e-2,
            sigma_ra2 in 1e-14f64..1e-8,
            sigma_dec2 in 1e-14f64..1e-8,
            ratio in 0.0f64..1.0,
        ) {
            let mut p = Matrix6::identity();
            p[(0, 0)] = var_ra;
            p[(1, 1)] = var_dec;
            let r = identity_r(sigma_ra2, sigma_dec2);
            apply_angular_variance_floor(&mut p, &r, ratio);
            prop_assert!(p[(0, 0)] >= var_ra);
            prop_assert!(p[(1, 1)] >= var_dec);
        }
    }
}

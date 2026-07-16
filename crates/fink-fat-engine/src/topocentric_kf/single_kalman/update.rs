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
    let innovation_covariance = h_mat * kf.covariance * h_mat.transpose() + r_mat;

    UpdateEvent::InnovationCovariance {
        s_00: innovation_covariance[(0, 0)],
        s_11: innovation_covariance[(1, 1)],
        s_det: innovation_covariance.determinant(),
    }
    .emit();

    // ── Kalman gain K = P⁻ Hᵀ S⁻¹ ──────────────────────────────────────────
    let innovation_covariance_inv = innovation_covariance.try_inverse().ok_or_else(|| {
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
    let updated_covariance = i_minus_kh * kf.covariance * i_minus_kh.transpose()
        + kalman_gain * r_mat * kalman_gain.transpose();

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

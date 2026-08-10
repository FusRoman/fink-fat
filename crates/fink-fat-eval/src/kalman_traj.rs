use ahash::AHashMap;
use fink_fat_engine::{
    engine_config::{
        grid_population::GridConfig, kalman_context::KalmanContext, kf_bank_config::KFBankConfig,
        night_advance_params::NightAdvanceParams,
    },
    topocentric_kf::{
        kalman_bank::{
            BankStep, KFBank,
            ellipse_region_finder::{SearchRegion, search_region_from_mixture},
        },
        single_kalman::{KFState, update::wrap_angle},
    },
};
use nalgebra::{Matrix2, Vector2, Vector3, Vector6};
use outfit::constants::ROT_EQUMJ2000_TO_ECLMJ2000;
use photom::{
    TrajId,
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};
use rayon::prelude::*;

use crate::ground_truth_state::{TruthLookup, TruthState};

const RAD_TO_ARCSEC: f64 = 3600.0 * 180.0 / std::f64::consts::PI;
const ARCSEC_TO_RAD: f64 = 1.0 / RAD_TO_ARCSEC;

/// χ²(2) quantiles/median, used to judge NIS calibration dataset-wide (see
/// `crate::trajectory_processing::TrajSummary::nis_calibration_ratio`). A
/// well-calibrated filter's NIS follows χ²(2); a median far below
/// [`NIS_CHI2_2DOF_MEDIAN`] means the filter's predicted covariance is
/// systematically too large (over-covariant — the actual residuals are
/// small compared to what the filter expects), far above means it's
/// over-confident (too small a covariance).
///
/// Closed form for k=2 degrees of freedom (χ²(2) is `Exponential(rate =
/// 1/2)`): `P(X ≤ x) = 1 - exp(-x/2)`, so `x_p = -2·ln(1-p)`.
pub const NIS_CHI2_2DOF_LOW: f64 = 0.050_636_616_366_209; // 2.5th percentile
pub const NIS_CHI2_2DOF_MEDIAN: f64 = 1.386_294_361_12; // 50th percentile (-2 ln 0.5)
pub const NIS_CHI2_2DOF_HIGH: f64 = 7.377_758_908_23; // 97.5th percentile

/// χ²(2) quantiles for the sky-plane NEES (same distribution as
/// [`NIS_CHI2_2DOF_LOW`]/[`NIS_CHI2_2DOF_HIGH`] — both are quadratic forms of
/// a 2-D Gaussian error against its own covariance — kept as separate
/// constants so NEES calibration reporting doesn't read as reusing NIS
/// thresholds by accident).
pub const NEES_CHI2_2DOF_LOW: f64 = NIS_CHI2_2DOF_LOW;
pub const NEES_CHI2_2DOF_MEDIAN: f64 = NIS_CHI2_2DOF_MEDIAN;
pub const NEES_CHI2_2DOF_HIGH: f64 = NIS_CHI2_2DOF_HIGH;

/// χ²(6) quantiles, used to judge the full 6-DOF cartesian NEES
/// (position + velocity error against [`KFState::cartesian_covariance`]).
pub const NEES_CHI2_6DOF_LOW: f64 = 1.237_344; // 2.5th percentile
pub const NEES_CHI2_6DOF_MEDIAN: f64 = 5.348_121; // 50th percentile
pub const NEES_CHI2_6DOF_HIGH: f64 = 14.449_375; // 97.5th percentile

/// Why a trajectory's predict/update loop ([`study_kalman_asteroid`])
/// stopped where it did.
///
/// The first three variants are never produced by `study_kalman_asteroid`
/// itself — they describe a trajectory that never reached (or never
/// finished setting up) its loop, assigned by the caller
/// (`crate::trajectory_processing::process_one_trajectory`). The rest are
/// assigned from inside the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrajStopReason {
    /// Fewer than 3 observations (before deduplication) — too short to be
    /// worth bootstrapping at all. Assigned by the caller, before
    /// `study_kalman_asteroid` is even called.
    NotEnoughPoints,
    /// No same-night (`dt < 0.5 d`) pair found anywhere in the
    /// (deduplicated) trajectory — [`init_bank_from_first_pair`] had
    /// nothing to seed a bank from.
    BootstrapFailed,
    /// The bootstrap pair consumed the whole (deduplicated) trajectory —
    /// nothing left to predict/update against.
    NoProcessableObs,
    /// Every observation after the bootstrap pair was processed — the loop
    /// ran to completion.
    ReachedEnd,
    /// The bank collapsed because at least one live hypothesis was rejected
    /// by the chi-square gate (`BankStep.n_gated > 0` — see
    /// `KFBankConfig::gate_chi2`). Takes priority over
    /// [`Self::CollapsedByPropagation`] when a step produces both gated and
    /// failed hypotheses, since gating is the deliberate/expected mechanism.
    CollapsedByGating,
    /// The bank collapsed with zero gated hypotheses — every live
    /// hypothesis failed to propagate (Kepler solver/Jacobian failure,
    /// e.g. a `dt` too small or too extreme for the two-body solver).
    CollapsedByPropagation,
    /// The bank survived the step but produced a degenerate result
    /// afterward: `KFBank::best()` returned `None`, or the per-step
    /// diagnostics couldn't be computed (singular sky covariance, etc.).
    DegenerateState,
}

impl TrajStopReason {
    /// Stable, human-readable label for console output.
    pub fn label(self) -> &'static str {
        match self {
            TrajStopReason::NotEnoughPoints => "not enough points",
            TrajStopReason::BootstrapFailed => "bootstrap failed (no intra-night pair)",
            TrajStopReason::NoProcessableObs => "no processable obs after bootstrap",
            TrajStopReason::ReachedEnd => "reached end",
            TrajStopReason::CollapsedByGating => "collapsed (gating)",
            TrajStopReason::CollapsedByPropagation => "collapsed (propagation failure)",
            TrajStopReason::DegenerateState => "degenerate state",
        }
    }

    /// All variants, in the fixed order used for reporting.
    pub fn all() -> [TrajStopReason; 7] {
        [
            TrajStopReason::NotEnoughPoints,
            TrajStopReason::BootstrapFailed,
            TrajStopReason::NoProcessableObs,
            TrajStopReason::ReachedEnd,
            TrajStopReason::CollapsedByGating,
            TrajStopReason::CollapsedByPropagation,
            TrajStopReason::DegenerateState,
        ]
    }
}

// ── Observer geometry cache ─────────────────────────────────────────────────

/// Observer heliocentric `(position, velocity)` at some epoch, in AU / AU per day.
type HelioGeometry = (Vector3<f64>, Vector3<f64>);

/// One hypothesis's Bayesian weight paired with its propagated Kalman state —
/// the raw `(weight, state)` mixture a bank predicts at some epoch, before
/// any [`crate::kalman_traj::compute_search_region`] reduction.
pub type WeightedHypothesisState<'a> = (f64, KFState<'a>);

/// One predict/update step's recorded `(mixture, observation)` pair — see
/// [`study_kalman_asteroid`]'s `mixture_recorder` parameter and
/// [`recompute_search_region_metrics`].
pub type MixtureStep<'a> = (Vec<WeightedHypothesisState<'a>>, Observation);

/// Precomputed `ObsId -> (r_obs, v_obs)` observer heliocentric geometry,
/// resolved once for every observation of a given trajectory population and
/// reused across every subsequent [`study_kalman_asteroid`] call.
///
/// This geometry depends only on `(obs_dataset, obs, kalman_ctx)` — never on
/// [`KFBankConfig`]/[`NightAdvanceParams`]/`KalmanConfig`'s `q0`/`dt_ref`, or
/// on any other tunable parameter — so it is valid for the lifetime of a
/// whole calibration run (see `fink_fat_eval::kf_calibration`), not just one
/// candidate. Building it once up front turns what would otherwise be a
/// repeated, non-trivial ephemeris lookup (observer cache construction +
/// JPL interpolation + light-time correction — see
/// [`fink_fat_engine::topocentric_kf::observer_state::EphemState::helio_observer_state`])
/// into a single O(1) map lookup per step.
pub struct ObserverGeometryCache(AHashMap<ObsId, HelioGeometry>);

impl ObserverGeometryCache {
    /// Resolve the geometry of every observation belonging to `traj_ids`,
    /// once, in parallel (read-only work, safe to run concurrently —
    /// same pattern as
    /// [`crate::trajectory_processing::process_all_trajectories`]).
    /// Observations whose observer/ephemeris lookup fails are simply
    /// omitted — [`Self::get`] falls back to a direct (uncached)
    /// resolution for those, so nothing is lost, only slower for that one
    /// observation.
    pub fn build(obs_dataset: &ObsDataset, context: &KalmanContext, traj_ids: &[TrajId]) -> Self {
        let entries: Vec<(ObsId, HelioGeometry)> = traj_ids
            .par_iter()
            .filter_map(|traj_id| {
                crate::trajectory_processing::materialize_contiguous_traj(obs_dataset, traj_id).ok()
            })
            .flat_map_iter(|traj| {
                traj.iter()
                    .filter_map(|obs| {
                        resolve_geometry(obs_dataset, context, obs).map(|g| (*obs.id(), g))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Self(entries.into_iter().collect())
    }

    /// Cached geometry for `obs`, falling back to a direct (uncached)
    /// resolution on a miss.
    pub fn get(
        &self,
        obs_dataset: &ObsDataset,
        context: &KalmanContext,
        obs: &Observation,
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        self.0
            .get(obs.id())
            .copied()
            .or_else(|| resolve_geometry(obs_dataset, context, obs))
    }
}

/// Resolve one observation's observer heliocentric geometry directly
/// (uncached) — the primitive [`ObserverGeometryCache`] wraps.
pub fn resolve_geometry(
    obs_dataset: &ObsDataset,
    context: &KalmanContext,
    obs: &Observation,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let observer = obs_dataset.get_observer(*obs.id())?;
    let helio_state = context
        .get_ephem()
        .helio_observer_state(observer, obs.mjd_tt())
        .ok()?;
    Some((helio_state.helio_cart_pos, helio_state.helio_cart_vel))
}

// ── Epoch deduplication ─────────────────────────────────────────────────────

/// Drop observations whose epoch falls within `tolerance_days` of the
/// previous *kept* observation's epoch (`traj` assumed already sorted by
/// epoch; the first of each duplicate cluster is kept).
///
/// Two detections at (near-)identical epochs are a structural case the
/// Kalman filter can't handle: [`KFState::predict`]/`propagate_to_epoch`
/// light-time-corrects and integrates over `dt`, and `dt ≈ 0` against a
/// spatially-distinct second detection produces a spuriously huge
/// innovation (and NIS) rather than a real filter inconsistency — polluting
/// both completion and NIS calibration statistics for a reason that has
/// nothing to do with how well the filter tracks the object.
///
/// Returns the deduplicated trajectory and the number of observations
/// dropped.
pub fn dedupe_by_epoch(traj: &[Observation], tolerance_days: f64) -> (Vec<Observation>, usize) {
    let mut kept: Vec<Observation> = Vec::with_capacity(traj.len());
    let mut n_removed = 0;
    for obs in traj {
        match kept.last() {
            Some(prev) if obs.mjd_tt() - prev.mjd_tt() < tolerance_days => {
                n_removed += 1;
            }
            _ => kept.push(obs.clone()),
        }
    }
    (kept, n_removed)
}

// ── Grid / bank initialisation ────────────────────────────────────────────────

pub fn init_bank_from_first_pair<'ctx, 'bank_config>(
    traj: &[Observation],
    obs_dataset: &ObsDataset,
    context: &'ctx KalmanContext,
    bank_config: &'bank_config KFBankConfig,
    grid_config: &GridConfig,
) -> Option<(usize, KFBank<'ctx, 'bank_config>)> {
    tracing::debug!(
        n_obs = traj.len(),
        "Scanning observations for first intra-night pair (dt < 1 day)"
    );

    let (idx, pair) = traj
        .windows(2)
        .enumerate()
        .find(|(_, w)| w[1].mjd_tt() - w[0].mjd_tt() < 0.5)?;

    let bank = KFBank::from_grid(
        obs_dataset,
        &pair[0],
        &pair[1],
        context,
        grid_config,
        bank_config,
    )
    .ok()?;
    log_bank_init(&bank);
    Some((idx, bank))
}

fn log_bank_init(bank: &KFBank<'_, '_>) {
    let epoch = bank
        .hypotheses()
        .first()
        .map(|h| h.kf.epoch)
        .unwrap_or(f64::NAN);

    tracing::debug!(
        epoch_mjd = epoch,
        n_hypotheses = bank.len(),
        "KFBank initialised"
    );

    if let Some(best) = bank.best() {
        tracing::debug!(
            ra_rad = best.kf.state[0],
            dec_rad = best.kf.state[1],
            ra_dot = best.kf.state[2],
            dec_dot = best.kf.state[3],
            rho_au = best.kf.state[4],
            rho_dot = best.kf.state[5],
            weight = best.weight(),
            "Best seed (attributable state)"
        );
    }
}

// ── Per-step diagnostics ──────────────────────────────────────────────────────

pub struct StepDiag {
    pub equ_pred: EquCoord,
    pub equ_obs: EquCoord,
    pub residual_ra_arcsec: f64,
    pub residual_dec_arcsec: f64,
    sigma_ra_arcsec: f64,
    sigma_dec_arcsec: f64,
    separation_arcsec_from_best_kf: f64,
    separation_arcsec_from_region: f64,
    mahalanobis_distance: f64,
    separation_in_sigma: f64,
    pos_cov_trace_au2: f64,
    vel_cov_trace_au2_day2: f64,
    predicted_range_au: f64,
    gain_frobenius_norm: f64,
    nis: f64,
    // ── Ground-truth-based metrics (only if a `TruthLookup` was supplied) ──
    pos_error_arcsec: Option<f64>,
    range_error_au: Option<f64>,
    cart_pos_error_au: Option<f64>,
    cart_vel_error_au_day: Option<f64>,
    nees_sky: Option<f64>,
    nees_cart: Option<f64>,
}

pub fn compute_step_diag(
    best_kf: &KFState,
    obs: &Observation,
    region_center: Option<&EquCoord>,
    truth_lookup: Option<&TruthLookup>,
    nis: f64,
) -> Option<StepDiag> {
    let equ_pred = best_kf.to_equ_coord().ok()?;

    let equ_obs = obs.equ_coord();
    let sigma_sky = best_kf.sky_covariance().ok()?;

    let (residual_ra_arcsec, residual_dec_arcsec, _residual_ra_raw_rad) =
        sky_residuals_arcsec(&equ_pred, equ_obs);

    let sigma_ra_arcsec = equ_pred.ra_error * RAD_TO_ARCSEC;
    let sigma_dec_arcsec = equ_pred.dec_error * RAD_TO_ARCSEC;
    let separation_arcsec = equ_pred.angular_separation(equ_obs).to_degrees() * 3600.;
    let sep_arcsec_region_center = {
        let center = region_center?;
        center.angular_separation(equ_obs).to_degrees() * 3600.
    };

    let mahalanobis_distance =
        mahalanobis_distance(residual_ra_arcsec, residual_dec_arcsec, &sigma_sky);
    let separation_in_sigma =
        separation_in_sigma(separation_arcsec, sigma_ra_arcsec, sigma_dec_arcsec);

    let (pos_cov_trace_au2, vel_cov_trace_au2_day2, predicted_range_au) =
        filter_health_metrics(best_kf);

    let truth = truth_lookup.and_then(|lookup| lookup.get(*obs.id()));
    let TruthMetrics {
        pos_error_arcsec,
        range_error_au,
        cart_pos_error_au,
        cart_vel_error_au_day,
        nees_sky,
        nees_cart,
    } = match truth {
        Some(truth) => {
            compute_truth_metrics(best_kf, &equ_pred, &sigma_sky, predicted_range_au, truth)
        }
        None => TruthMetrics::default(),
    };

    Some(StepDiag {
        equ_pred,
        equ_obs: *equ_obs,
        residual_ra_arcsec,
        residual_dec_arcsec,
        sigma_ra_arcsec,
        sigma_dec_arcsec,
        separation_arcsec_from_best_kf: separation_arcsec,
        separation_arcsec_from_region: sep_arcsec_region_center,
        mahalanobis_distance,
        separation_in_sigma,
        pos_cov_trace_au2,
        vel_cov_trace_au2_day2,
        predicted_range_au,
        gain_frobenius_norm: best_kf.last_gain_frobenius_norm(),
        nis,
        pos_error_arcsec,
        range_error_au,
        cart_pos_error_au,
        cart_vel_error_au_day,
        nees_sky,
        nees_cart,
    })
}

// ── Search region ─────────────────────────────────────────────────────────────

pub struct SearchRegionDiag {
    radius_arcsec: f64,
    semi_major_3sigma_arcsec: f64,
    semi_minor_3sigma_arcsec: f64,
    position_angle_deg: f64,
    obs_within_radius: bool,
}

impl SearchRegionDiag {
    pub fn nan_fallback() -> Self {
        Self {
            radius_arcsec: f64::NAN,
            semi_major_3sigma_arcsec: f64::NAN,
            semi_minor_3sigma_arcsec: f64::NAN,
            position_angle_deg: f64::NAN,
            obs_within_radius: false,
        }
    }
}

/// Reduce an already-propagated `(weight, state)` mixture (see
/// [`KFBank::predicted_mixture`]) into a [`SearchRegion`], applying `top_k`
/// first (the mixture is captured pre-`top_k` so it stays reusable for any
/// `top_k`/`search_region_chi2`/`radius_strategy` combination — see
/// [`recompute_search_region_metrics`]).
pub fn compute_search_region(
    predicted: &[(f64, KFState)],
    advance_params: &NightAdvanceParams,
    search_region_chi2: f64,
) -> Option<SearchRegion> {
    let mut predicted = predicted.to_vec();
    advance_params.top_k.apply(&mut predicted);
    let obs_noise = Vector2::from(advance_params.obs_noise);
    search_region_from_mixture(
        &predicted,
        obs_noise,
        advance_params.radius_strategy,
        search_region_chi2,
    )
    .ok()
}

/// Cheaply recompute `(pct_within_search_radius, mean_search_radius_arcsec)`
/// for a candidate `(advance_params, search_region_chi2)` pair against a
/// trajectory's already-recorded per-step mixtures — see
/// [`study_kalman_asteroid`]'s `mixture_recorder` parameter. No ephemeris
/// lookup, no Kepler solve, no gating: purely the geometric/statistical
/// reduction [`compute_search_region`] performs.
///
/// One `(mixture, observation)` pair per predict/update step actually
/// processed — see [`study_kalman_asteroid`]'s `mixture_recorder` parameter.
pub fn recompute_search_region_metrics(
    steps: &[MixtureStep],
    advance_params: &NightAdvanceParams,
    search_region_chi2: f64,
) -> (f64, f64) {
    let n = steps.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }

    let mut n_within = 0usize;
    let mut radii_arcsec: Vec<f64> = Vec::with_capacity(n);

    for (predicted, obs) in steps {
        let Some(region) = compute_search_region(predicted, advance_params, search_region_chi2)
        else {
            continue;
        };
        let sep = separation_from_search_center(&region, obs.equ_coord());
        if sep <= region.radius_rad {
            n_within += 1;
        }
        radii_arcsec.push(region.radius_rad * RAD_TO_ARCSEC);
    }

    let pct_within_search_radius = 100.0 * n_within as f64 / n as f64;
    let mean_search_radius_arcsec = if radii_arcsec.is_empty() {
        f64::NAN
    } else {
        radii_arcsec.iter().sum::<f64>() / radii_arcsec.len() as f64
    };
    (pct_within_search_radius, mean_search_radius_arcsec)
}

pub fn search_region_diag(region: &SearchRegion, equ_obs: &EquCoord) -> SearchRegionDiag {
    let radius_arcsec = region.radius_rad * RAD_TO_ARCSEC;

    let best_s = region
        .components
        .iter()
        .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
        .map(|c| c.s);

    let (semi_major, semi_minor, pa_deg) =
        best_s
            .as_ref()
            .map(sky_ellipse_params)
            .unwrap_or((f64::NAN, f64::NAN, f64::NAN));

    let sep = separation_from_search_center(region, equ_obs);

    tracing::trace!(
        separation_arcsec = sep.to_degrees() * 3600.0,
        "Separation from region center"
    );

    SearchRegionDiag {
        radius_arcsec,
        semi_major_3sigma_arcsec: 3.0 * semi_major,
        semi_minor_3sigma_arcsec: 3.0 * semi_minor,
        position_angle_deg: pa_deg,
        obs_within_radius: sep <= region.radius_rad,
    }
}

// ── Logging ───────────────────────────────────────────────────────────────────

pub fn log_bank_report(report: &BankStep) {
    tracing::trace!(
        n_before = report.n_before,
        n_after = report.n_after,
        n_gated = report.n_gated,
        n_failed = report.n_failed,
        n_effective = report.n_effective,
        best_weight = report.best_weight,
        "Bank step"
    );
}

pub fn log_step_diag(diag: &StepDiag, best_id: u64, step: usize) {
    tracing::trace!(
        best_id,
        pred_ra = diag.equ_pred.ra,
        pred_dec = diag.equ_pred.dec,
        obs_ra = diag.equ_obs.ra,
        obs_dec = diag.equ_obs.dec,
        residual_ra_arcsec = diag.residual_ra_arcsec,
        residual_dec_arcsec = diag.residual_dec_arcsec,
        separation_arcsec_from_best_kf = diag.separation_arcsec_from_best_kf,
        separation_in_sigma = diag.separation_in_sigma,
        mahalanobis = diag.mahalanobis_distance,
        sigma_ra_arcsec = diag.sigma_ra_arcsec,
        sigma_dec_arcsec = diag.sigma_dec_arcsec,
        pos_cov_trace_au2 = diag.pos_cov_trace_au2,
        vel_cov_trace_au2_day2 = diag.vel_cov_trace_au2_day2,
        predicted_range_au = diag.predicted_range_au,
        gain_frobenius_norm = diag.gain_frobenius_norm,
        nis = diag.nis,
        nis_in_3sigma = diag.nis <= 9.0,
        "Step diagnostics"
    );

    if diag.mahalanobis_distance > 5.0 {
        tracing::trace!(
            mahalanobis = diag.mahalanobis_distance,
            step = step + 1,
            "Large Mahalanobis distance — possible filter divergence"
        );
    }
}

pub fn log_search_region(sr: &SearchRegionDiag, region: &SearchRegion, equ_obs: &EquCoord) {
    tracing::trace!(
        target: "search_region",
        center_ra = region.center_ra,
        center_dec = region.center_dec,
        radius_arcsec = sr.radius_arcsec,
        obs_ra = equ_obs.ra,
        obs_dec = equ_obs.dec,
        separation_arcsec = separation_from_search_center(region, equ_obs) * RAD_TO_ARCSEC,
        obs_within_radius = sr.obs_within_radius,
        semi_major_3sigma_arcsec = sr.semi_major_3sigma_arcsec,
        semi_minor_3sigma_arcsec = sr.semi_minor_3sigma_arcsec,
        position_angle_deg = sr.position_angle_deg,
        "Search region"
    );
}

// ── Result assembly ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn assemble_result(
    epoch: f64,
    dt: f64,
    diag: &StepDiag,
    report: &BankStep,
    sr: &SearchRegionDiag,
    inflation_lambda: f64,
    radius_spread_arcsec: f64,
    radius_component_arcsec: f64,
    map_rho_sigma_au: f64,
    map_rhodot_sigma: f64,
) -> KFStudyResult {
    KFStudyResult {
        epoch,
        dt,
        separation_arcsec_from_best_kf: diag.separation_arcsec_from_best_kf,
        separation_arcsec_from_region: diag.separation_arcsec_from_region,
        residual_ra_arcsec: diag.residual_ra_arcsec,
        residual_dec_arcsec: diag.residual_dec_arcsec,
        sigma_ra_arcsec: diag.sigma_ra_arcsec,
        sigma_dec_arcsec: diag.sigma_dec_arcsec,
        mahalanobis_distance: diag.mahalanobis_distance,
        separation_in_sigma: diag.separation_in_sigma,
        pos_cov_trace_au2: diag.pos_cov_trace_au2,
        vel_cov_trace_au2_day2: diag.vel_cov_trace_au2_day2,
        predicted_range_au: diag.predicted_range_au,
        gain_frobenius_norm: diag.gain_frobenius_norm,
        n_hypotheses_before: report.n_before,
        n_hypotheses_after: report.n_after,
        n_gated: report.n_gated,
        n_effective: report.n_effective,
        best_weight: report.best_weight,
        search_region_radius_arcsec: sr.radius_arcsec,
        region_semi_major_3sigma_arcsec: sr.semi_major_3sigma_arcsec,
        region_semi_minor_3sigma_arcsec: sr.semi_minor_3sigma_arcsec,
        region_position_angle_deg: sr.position_angle_deg,
        radius_spread_arcsec,
        radius_component_arcsec,
        map_rho_sigma_au,
        map_rhodot_sigma,
        obs_within_search_radius: sr.obs_within_radius,
        obs_within_3sigma_region: diag.nis <= 9.0,
        nis: diag.nis,
        pos_error_arcsec: diag.pos_error_arcsec,
        range_error_au: diag.range_error_au,
        cart_pos_error_au: diag.cart_pos_error_au,
        cart_vel_error_au_day: diag.cart_vel_error_au_day,
        nees_sky: diag.nees_sky,
        nees_cart: diag.nees_cart,
        inflation_lambda,
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Diagnostic record for a single predict-then-update step of the
/// topocentric Kalman filter bank along a known asteroid trajectory.
///
/// Each field captures a quantity that characterises either the **prediction
/// quality** of the best (MAP) hypothesis, the **filter health**, or the
/// **bank health** (number of surviving hypotheses, effective sample size, …).
///
/// The search region fields are derived from [`SearchRegion`] computed
/// **before** the update step, i.e. they reflect the genuine predictive
/// uncertainty at the time an association decision would be made.
///
/// Units
/// -----
/// All angular quantities are in **arcseconds** for readability.
/// Times are in **days** (MJD TT).
/// Distances are in **AU**.
#[derive(Debug, Clone)]
pub struct KFStudyResult {
    /// MJD (TT) of the observation used in this step.
    pub epoch: f64,
    /// Time elapsed since the previous observation (days).
    pub dt: f64,

    // ── Prediction quality (best/MAP hypothesis) ──────────────────────────
    /// Angular separation between the predicted sky position of the best kf and the true
    /// observation, in arcseconds.
    pub separation_arcsec_from_best_kf: f64,
    /// Angular separation between the predicted sky region center and the true
    /// observation, in arcseconds.
    pub separation_arcsec_from_region: f64,
    /// Residual in right ascension: $\Delta\alpha \cdot \cos\delta$,
    /// in arcseconds (signed, positive eastward).
    pub residual_ra_arcsec: f64,
    /// Residual in declination: $\Delta\delta$, in arcseconds (signed).
    pub residual_dec_arcsec: f64,

    // ── Predicted uncertainty (before update) ─────────────────────────────
    /// Predicted 1-σ uncertainty on RA from $\Sigma_{sky} = HPH^\top$,
    /// in arcseconds.
    pub sigma_ra_arcsec: f64,
    /// Predicted 1-σ uncertainty on Dec from $\Sigma_{sky} = HPH^\top$,
    /// in arcseconds.
    pub sigma_dec_arcsec: f64,
    /// Normalised separation (Mahalanobis distance in the 2-D sky plane)
    /// w.r.t. $\Sigma_{sky}$ alone (no measurement noise).
    pub mahalanobis_distance: f64,
    /// Ratio $\theta / \sigma_{sky}$, dimensionless.
    pub separation_in_sigma: f64,

    // ── Covariance / filter health (best hypothesis) ──────────────────────
    /// Trace of the $3\times3$ position covariance block $P_{pp}$, in AU².
    pub pos_cov_trace_au2: f64,
    /// Trace of the $3\times3$ velocity covariance block $P_{vv}$,
    /// in (AU/day)².
    pub vel_cov_trace_au2_day2: f64,
    /// Estimated topocentric range $\rho$ from the best hypothesis, in AU.
    pub predicted_range_au: f64,
    /// Norm of the Kalman gain $\|K\|_F$ of the best hypothesis.
    pub gain_frobenius_norm: f64,

    // ── Bank health ────────────────────────────────────────────────────────
    /// Number of live hypotheses before this step's gating/pruning/merging.
    pub n_hypotheses_before: usize,
    /// Number of live hypotheses after this step's gating/pruning/merging.
    pub n_hypotheses_after: usize,
    /// Hypotheses rejected by the chi-square gate this step.
    pub n_gated: usize,
    /// Effective sample size $1 / \sum_i w_i^2$ of the bank.
    pub n_effective: f64,
    /// Posterior weight of the best (MAP) hypothesis.
    pub best_weight: f64,

    // ── Predicted search region (before update) ───────────────────────────
    /// Conservative bounding radius of the [`SearchRegion`] (union of all
    /// per-hypothesis 3-σ ellipses plus centroid offsets), in arcseconds.
    pub search_region_radius_arcsec: f64,
    /// Semi-major axis of the best-hypothesis predicted 3-σ ellipse on the
    /// sky ($S = \Sigma_{sky} + R$), in arcseconds.
    pub region_semi_major_3sigma_arcsec: f64,
    /// Semi-minor axis of the best-hypothesis predicted 3-σ ellipse on the
    /// sky ($S = \Sigma_{sky} + R$), in arcseconds.
    pub region_semi_minor_3sigma_arcsec: f64,
    /// Position angle of the uncertainty ellipse major axis, in degrees
    /// east of north.
    pub region_position_angle_deg: f64,
    /// Search-radius contribution from the **between-mode spread**
    /// `Σ wᵢ Δμᵢ Δμᵢᵀ` (multimodal ρ,ρ̇ disagreement), in arcseconds —
    /// see `compute_radius_decomposition`. From the pre-update mixture.
    pub radius_spread_arcsec: f64,
    /// Search-radius contribution from the **within-mode covariance**
    /// `Σ wᵢ Sᵢ` (each hypothesis's own sky covariance), in arcseconds —
    /// see `compute_radius_decomposition`. From the pre-update mixture.
    pub radius_component_arcsec: f64,
    /// MAP (max-weight) pre-update hypothesis's range 1-σ (`√P[4,4]`), in AU —
    /// the range-observability signal. From the pre-update mixture.
    pub map_rho_sigma_au: f64,
    /// MAP (max-weight) pre-update hypothesis's range-rate 1-σ (`√P[5,5]`),
    /// in AU/day. From the pre-update mixture.
    pub map_rhodot_sigma: f64,
    /// Whether the true observation falls within the conservative bounding
    /// radius of the [`SearchRegion`].
    pub obs_within_search_radius: bool,
    /// Whether the true observation falls within the 3-σ predicted ellipse
    /// of the best hypothesis, i.e. NIS $\leq 9$.
    pub obs_within_3sigma_region: bool,

    // ── Normalized Innovation Squared ─────────────────────────────────────
    /// Normalized Innovation Squared (NIS) for this update step.
    ///
    /// $$\text{NIS} = \nu^\top S^{-1} \nu$$
    ///
    /// where $\nu \in \mathbb{R}^2$ is the innovation vector and
    /// $S = HPH^\top + R$ is the innovation covariance. Under a
    /// well-calibrated filter, NIS follows a $\chi^2(2)$ distribution,
    /// so the expected value is 2.0.
    pub nis: f64,

    // ── Ground-truth comparison (only if a ground-truth file was supplied) ─
    /// Angular separation between the predicted sky position and the *true*
    /// sky position, in arcseconds. `None` if no ground truth is available
    /// for this observation.
    pub pos_error_arcsec: Option<f64>,
    /// Predicted topocentric range minus the *true* topocentric range, in AU.
    pub range_error_au: Option<f64>,
    /// Norm of the topocentric position error vector (estimate − truth), in AU.
    pub cart_pos_error_au: Option<f64>,
    /// Norm of the topocentric velocity error vector (estimate − truth), in AU/day.
    pub cart_vel_error_au_day: Option<f64>,
    /// NEES in the 2-D sky plane: $e^\top \Sigma_{sky}^{-1} e$ where
    /// $e$ = predicted sky position − true sky position. Under a
    /// well-calibrated filter this follows $\chi^2(2)$ (see
    /// [`NEES_CHI2_2DOF_MEDIAN`]).
    pub nees_sky: Option<f64>,
    /// Full 6-DOF NEES in topocentric Cartesian space (position + velocity),
    /// via [`fink_fat_engine::topocentric_kf::single_kalman::KFState::cartesian_covariance`].
    /// Under a well-calibrated filter this follows $\chi^2(6)$ (see
    /// [`NEES_CHI2_6DOF_MEDIAN`]). `None` if the covariance was singular.
    pub nees_cart: Option<f64>,

    /// Fading-memory covariance-inflation factor $\lambda$ that was applied
    /// to *this* step's propagation, recomputed from the pre-step MAP
    /// hypothesis's `nis_ema`
    /// (see [`fink_fat_engine::topocentric_kf::single_kalman::propagate`]'s
    /// module doc). `1.0` means no inflation (the consistency dead-zone);
    /// values above `1.0` (up to the configured `max_inflation`) mean the
    /// filter's own recent NIS history was judged inconsistent and its
    /// covariance was re-opened before this step's update.
    pub inflation_lambda: f64,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Everything [`study_kalman_asteroid`] learned about a trajectory: not
/// just the final bank/per-step results, but *why* the loop stopped where
/// it did and how many observations were actually available to it — needed
/// to compute an unbiased `completion_fraction`
/// (`crate::trajectory_processing::TrajSummary`) and to build a
/// dataset-wide stop-reason histogram
/// (`crate::trajectory_processing::RunCounters`).
pub struct StudyOutcome<'a, 'bank_config> {
    pub bank: Option<KFBank<'a, 'bank_config>>,
    pub results: Vec<KFStudyResult>,
    pub stop_reason: TrajStopReason,
    /// Index (into the deduplicated trajectory) of the bootstrap pair's
    /// first observation — `None` only for [`TrajStopReason::BootstrapFailed`].
    pub bootstrap_idx: Option<usize>,
    /// Observations available to the predict/update loop after the
    /// bootstrap pair (`deduplicated_len - (bootstrap_idx + 2)`) — the
    /// unbiased denominator for `completion_fraction`, unlike
    /// `n_obs_total - 2` which ignores any observations the bootstrap scan
    /// had to skip before finding a same-night pair.
    pub n_processable: usize,
    /// Observations dropped by [`dedupe_by_epoch`] (near-identical epoch to
    /// the previous kept observation) before bootstrapping.
    pub n_obs_deduplicated: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn study_kalman_asteroid<'a, 'bank_config>(
    traj: &[Observation],
    obs_dataset: &ObsDataset,
    context: &'a KalmanContext,
    bank_config: &'bank_config KFBankConfig,
    grid_config: &GridConfig,
    advance_params: &NightAdvanceParams,
    geometry_cache: &ObserverGeometryCache,
    mut mixture_recorder: Option<&mut Vec<MixtureStep<'a>>>,
    truth_lookup: Option<&TruthLookup>,
) -> StudyOutcome<'a, 'bank_config> {
    tracing::debug!(n_obs = traj.len(), "Starting Kalman filter bank study");

    let (traj, n_obs_deduplicated) =
        dedupe_by_epoch(traj, advance_params.visit_epoch_tolerance_days);
    let traj = traj.as_slice();
    if n_obs_deduplicated > 0 {
        tracing::debug!(
            n_obs_deduplicated,
            "Dropped near-duplicate-epoch observations"
        );
    }

    let (idx_first_obs, mut bank) =
        match init_bank_from_first_pair(traj, obs_dataset, context, bank_config, grid_config) {
            Some(pair) => {
                tracing::debug!(
                    bootstrap_idx = pair.0,
                    n_hypotheses = pair.1.len(),
                    "Bootstrap successful"
                );
                pair
            }
            None => {
                tracing::debug!("Bootstrap failed, returning empty results");
                return StudyOutcome {
                    bank: None,
                    results: Vec::new(),
                    stop_reason: TrajStopReason::BootstrapFailed,
                    bootstrap_idx: None,
                    n_processable: 0,
                    n_obs_deduplicated,
                };
            }
        };

    let observations_to_process = &traj[idx_first_obs + 2..];
    let n_obs = observations_to_process.len();
    tracing::debug!(
        n_obs,
        n_skipped = idx_first_obs + 2,
        "Starting predict/update loop"
    );

    if n_obs == 0 {
        return StudyOutcome {
            bank: Some(bank),
            results: Vec::new(),
            stop_reason: TrajStopReason::NoProcessableObs,
            bootstrap_idx: Some(idx_first_obs),
            n_processable: 0,
            n_obs_deduplicated,
        };
    }

    let mut t_prev = traj[idx_first_obs + 1].mjd_tt();
    let mut kf_results: Vec<KFStudyResult> = Vec::with_capacity(n_obs);
    let mut stop_reason = TrajStopReason::ReachedEnd;

    for (step, obs) in observations_to_process.iter().enumerate() {
        let epoch = obs.mjd_tt();
        let dt = epoch - t_prev;

        tracing::trace!(
            step = step + 1,
            n_obs,
            epoch_mjd = epoch,
            dt_days = dt,
            "Step header"
        );

        let geometry = geometry_cache.get(obs_dataset, context, obs);
        // One `predicted_mixture` call here (which does the same two-body
        // propagation `predict_search_region` used to do internally) instead
        // of asking the bank for a `SearchRegion` directly — this is the
        // exact raw `(weight, state)` mixture `mixture_recorder` needs, so
        // recording costs nothing beyond the propagation this step already
        // pays for either way.
        let predicted_mixture =
            geometry.map(|(r_obs, v_obs)| bank.predicted_mixture(epoch, r_obs, v_obs));
        let region = predicted_mixture.as_ref().and_then(|predicted| {
            compute_search_region(predicted, advance_params, bank_config.search_region_chi2)
        });

        // Proper predictive NIS: innovation of `obs` against the *pre-update*
        // predicted mixture (computed here, before `bank.step` absorbs `obs`).
        // Must be taken now — after the step the state has already fitted the
        // observation, which is exactly the post-fit-residual bug this replaces.
        let predictive_nis = predicted_mixture
            .as_ref()
            .and_then(|m| compute_predictive_nis(m, obs))
            .unwrap_or(f64::NAN);

        // Split the (pre-update) search radius into its between-mode spread
        // and within-mode covariance parts, plus the MAP hypothesis's
        // range/range-rate σ — the diagnostic for whether the gate is wide
        // because of a multimodal ρ,ρ̇ mixture or a single hypothesis's
        // range-driven covariance.
        let decomp = predicted_mixture
            .as_ref()
            .and_then(|m| compute_radius_decomposition(m, bank_config.search_region_chi2));
        let radius_spread_arcsec = decomp.as_ref().map_or(f64::NAN, |d| d.spread_arcsec);
        let radius_component_arcsec = decomp.as_ref().map_or(f64::NAN, |d| d.component_arcsec);
        let map_rho_sigma_au = decomp.as_ref().map_or(f64::NAN, |d| d.map_rho_sigma_au);
        let map_rhodot_sigma = decomp.as_ref().map_or(f64::NAN, |d| d.map_rhodot_sigma);

        tracing::trace!(
            n_hypotheses = bank.len(),
            epoch_mjd = epoch,
            "Stepping bank"
        );

        // Fading-memory inflation factor λ that `propagate_covariance` will
        // apply *this* step, recomputed here from the pre-step MAP
        // hypothesis's `nis_ema` (a public field) — mirrors the exact
        // formula in `fink_fat_engine::topocentric_kf::single_kalman::propagate`,
        // reading the same (now configurable) thresholds from `context`
        // rather than duplicating hardcoded constants.
        let inflation_lambda = bank.best().and_then(|h| h.kf.nis_ema).map_or(1.0, |ema| {
            (ema / context.get_inflation_chi2_threshold()).clamp(1.0, context.get_max_inflation())
        });

        // Snapshot before step in case it collapses.
        let snapshot = bank.clone();
        // `step_with_geometry` when geometry is already known (the common
        // case, thanks to `geometry_cache`) skips a redundant per-hypothesis
        // ephemeris lookup `step` would otherwise perform internally — see
        // `KFBank::step_with_geometry`'s doc. Falls back to `step` (which
        // resolves geometry itself) on the rare cache-miss-and-direct-lookup-
        // also-failed case, for identical failure semantics either way.
        let report = match geometry {
            Some((r_obs, v_obs)) => bank.step_with_geometry(r_obs, v_obs, obs),
            None => bank.step(obs_dataset, obs),
        };
        log_bank_report(&report);

        if report.collapsed {
            stop_reason = match (report.n_gated, report.n_failed) {
                (0, f) if f > 0 => TrajStopReason::CollapsedByPropagation,
                (g, _) if g > 0 => TrajStopReason::CollapsedByGating,
                _ => TrajStopReason::DegenerateState,
            };
            tracing::trace!(step = step + 1, ?stop_reason, "Bank collapsed, stopping");
            bank = snapshot;
            break;
        }

        let best = match bank.best() {
            Some(b) => b,
            None => {
                tracing::warn!(step = step + 1, "Bank non-empty but best() returned None");
                stop_reason = TrajStopReason::DegenerateState;
                bank = snapshot;
                break;
            }
        };

        let equ_reg_center = region
            .clone()
            .map(|region| EquCoord::new(region.center_ra, 0., region.center_dec, 0.));
        let diag = match compute_step_diag(
            &best.kf,
            obs,
            equ_reg_center.as_ref(),
            truth_lookup,
            predictive_nis,
        ) {
            Some(d) => d,
            None => {
                tracing::warn!(step = step + 1, "compute_step_diag returned None, stopping");
                stop_reason = TrajStopReason::DegenerateState;
                bank = snapshot;
                break;
            }
        };
        log_step_diag(&diag, best.id, step);

        let sr = match &region {
            Some(r) => {
                let d = search_region_diag(r, &diag.equ_obs);
                log_search_region(&d, r, &diag.equ_obs);
                d
            }
            None => {
                tracing::trace!(step = step + 1, "Could not compute SearchRegion");
                SearchRegionDiag::nan_fallback()
            }
        };

        tracing::trace!(orbit = %best.kf.to_orbit(), "Best KF orbit");

        // Recorded only once this step is confirmed to produce a
        // `KFStudyResult` (past every early-`break` above), so
        // `mixture_recorder`'s entries stay 1:1 aligned with `kf_results` —
        // required by `recompute_search_region_metrics`.
        if let (Some(recorder), Some(predicted)) = (&mut mixture_recorder, predicted_mixture) {
            recorder.push((predicted, obs.clone()));
        }

        t_prev = epoch;
        kf_results.push(assemble_result(
            epoch,
            dt,
            &diag,
            &report,
            &sr,
            inflation_lambda,
            radius_spread_arcsec,
            radius_component_arcsec,
            map_rho_sigma_au,
            map_rhodot_sigma,
        ));
    }

    StudyOutcome {
        bank: Some(bank),
        results: kf_results,
        stop_reason,
        bootstrap_idx: Some(idx_first_obs),
        n_processable: n_obs,
        n_obs_deduplicated,
    }
}

// ── Pure helper functions (unchanged) ─────────────────────────────────────────

fn sky_residuals_arcsec(equ_pred: &EquCoord, equ_obs: &EquCoord) -> (f64, f64, f64) {
    let cos_dec = equ_pred.dec.cos();
    let d_ra_raw = wrap_angle(equ_obs.ra - equ_pred.ra);
    let d_ra_gc = d_ra_raw * cos_dec * RAD_TO_ARCSEC;
    let d_dec = (equ_obs.dec - equ_pred.dec) * RAD_TO_ARCSEC;
    (d_ra_gc, d_dec, d_ra_raw)
}

fn mahalanobis_distance(d_ra_arcsec: f64, d_dec_arcsec: f64, sigma_sky: &Matrix2<f64>) -> f64 {
    let residual = Vector2::new(d_ra_arcsec * ARCSEC_TO_RAD, d_dec_arcsec * ARCSEC_TO_RAD);
    sigma_sky
        .try_inverse()
        .map(|s_inv| {
            (residual.transpose() * s_inv * residual)[(0, 0)]
                .max(0.0)
                .sqrt()
        })
        .unwrap_or(f64::NAN)
}

fn separation_in_sigma(sep_arcsec: f64, sigma_ra: f64, sigma_dec: f64) -> f64 {
    let sigma_mean = ((sigma_ra * sigma_ra + sigma_dec * sigma_dec) / 2.0).sqrt();
    if sigma_mean > 0.0 {
        sep_arcsec / sigma_mean
    } else {
        f64::NAN
    }
}

fn filter_health_metrics(kf: &KFState) -> (f64, f64, f64) {
    let pos_trace = kf.covariance.fixed_view::<3, 3>(0, 0).trace();
    let vel_trace = kf.covariance.fixed_view::<3, 3>(3, 3).trace();
    let predicted_range_au = kf.state[4];
    (pos_trace, vel_trace, predicted_range_au)
}

fn sky_ellipse_params(s: &Matrix2<f64>) -> (f64, f64, f64) {
    let (a, b, d) = (s[(0, 0)], s[(0, 1)], s[(1, 1)]);
    let mid = (a + d) / 2.0;
    let half_diff = (a - d) / 2.0;
    let delta = (half_diff * half_diff + b * b).sqrt();
    let semi_major = ((mid + delta).max(0.0)).sqrt() * RAD_TO_ARCSEC;
    let semi_minor = ((mid - delta).max(0.0)).sqrt() * RAD_TO_ARCSEC;
    let pa_deg = (0.5 * b.atan2(half_diff)).to_degrees();
    (semi_major, semi_minor, pa_deg)
}

/// Proper **predictive** NIS: the innovation of the observation against the
/// *pre-update* predicted mixture, normalized by the prior innovation
/// covariance `S = S_mix + R`.
///
/// This is the honest filter-consistency diagnostic. The earlier `compute_nis`
/// used the *post-update* best hypothesis (`equ_pred = best_kf.to_equ_coord()`
/// after `bank.step`), so its "innovation" was actually a post-fit residual —
/// tiny by construction (the update pulls the attributable state α,δ straight
/// toward the measurement) and moving the *wrong* way when `R` shrinks. That
/// made the NIS structurally ≪ 1 regardless of any filter parameter. Here we
/// instead consume the raw `(weight, KFState)` mixture propagated to the
/// observation epoch **before** the update (the same `predicted_mixture` used
/// to build the search region).
///
/// The Gaussian-mixture innovation covariance is
/// `S_mix = Σ wᵢ (Sᵢ + Δμᵢ Δμᵢᵀ)` (within-component sky covariance plus the
/// spread of the component means), computed in RAW α/δ — matching the filter's
/// own convention (`H = [I₂|0]`, `sky_covariance()` is raw-α), so no cos(δ)
/// factor is applied. Returns `None` if the mixture is empty / degenerate.
pub fn compute_predictive_nis(mixture: &[(f64, KFState<'_>)], obs: &Observation) -> Option<f64> {
    let wsum: f64 = mixture.iter().map(|(w, _)| *w).sum();
    if wsum <= 0.0 {
        return None;
    }

    // Per-hypothesis (weight, ra, dec, sky covariance), weights renormalized.
    let mut comps: Vec<(f64, f64, f64, Matrix2<f64>)> = Vec::with_capacity(mixture.len());
    for (w, kf) in mixture {
        let c = kf.to_equ_coord().ok()?;
        let s = kf.sky_covariance().ok()?;
        comps.push((w / wsum, c.ra, c.dec, s));
    }
    let (_, ra0, _, _) = *comps.first()?;

    // Weighted mixture mean; RA accumulated as a wrapped offset from ra0 to
    // stay valid across the 0/2π seam.
    let mut mean_ra_off = 0.0;
    let mut mean_dec = 0.0;
    for (w, ra, dec, _) in &comps {
        mean_ra_off += w * wrap_angle(ra - ra0);
        mean_dec += w * dec;
    }
    let mean_ra = ra0 + mean_ra_off;

    // Mixture covariance S_mix = Σ wᵢ (Sᵢ + Δμᵢ Δμᵢᵀ).
    let mut s_mix = Matrix2::zeros();
    for (w, ra, dec, s_i) in &comps {
        let d = Vector2::new(wrap_angle(ra - mean_ra), dec - mean_dec);
        s_mix += *w * (s_i + d * d.transpose());
    }

    let coord = obs.equ_coord();
    let r_mat = Matrix2::from_diagonal(&Vector2::new(
        coord.ra_error * coord.ra_error,
        coord.dec_error * coord.dec_error,
    ));
    let s = s_mix + r_mat;
    let nu = Vector2::new(wrap_angle(coord.ra - mean_ra), coord.dec - mean_dec);
    s.try_inverse()
        .map(|s_inv| (nu.transpose() * s_inv * nu)[(0, 0)])
}

/// Per-step decomposition of the predicted search region, from the pre-update
/// mixture. See [`compute_radius_decomposition`].
pub struct RadiusDecomp {
    /// Between-mode spread contribution to the radius (arcsec).
    pub spread_arcsec: f64,
    /// Within-mode covariance contribution to the radius (arcsec).
    pub component_arcsec: f64,
    /// MAP (max-weight) hypothesis's range 1-σ, `√P[4,4]`, in AU.
    pub map_rho_sigma_au: f64,
    /// MAP (max-weight) hypothesis's range-rate 1-σ, `√P[5,5]`, in AU/day.
    pub map_rhodot_sigma: f64,
}

/// Decompose the search-region radius into its two additive covariance
/// sources. `radius_strategy::MixtureCovariance` sizes the region from
/// `S_mix = Σ wᵢ (Sᵢ + Δμᵢ Δμᵢᵀ)`, i.e. the sum of a **between-mode spread**
/// `Σ wᵢ Δμᵢ Δμᵢᵀ` (distinct ρ,ρ̇ hypotheses predicting different sky
/// positions) and a **within-mode covariance** `Σ wᵢ Sᵢ` (each hypothesis's
/// own `H Pᵢ Hᵀ`). Each contribution is returned on its own as an arcsec
/// radius `√(search_region_chi2)·√(λmax)`, so their relative size reveals
/// whether the wide gate is driven by a multimodal mixture (early steps,
/// before the cap schedule collapses the bank to one hypothesis) or by a
/// single hypothesis's covariance (later steps).
///
/// Also returns the MAP (max-weight) hypothesis's range/range-rate 1-σ
/// (`√P[4,4]`, `√P[5,5]`) — the **range-driven** signal: a strict
/// sky-projected range fraction would need the propagation STM sub-blocks
/// (not stored on `KFState`), so we instead surface the raw range covariance.
/// Read with `component_arcsec`: a flat/large range σ while the within-mode
/// radius stays wide is the range-observability-limited regime.
///
/// Computed from the same pre-update `predicted_mixture` as
/// [`compute_predictive_nis`].
pub fn compute_radius_decomposition(
    mixture: &[(f64, KFState<'_>)],
    search_region_chi2: f64,
) -> Option<RadiusDecomp> {
    let wsum: f64 = mixture.iter().map(|(w, _)| *w).sum();
    if wsum <= 0.0 {
        return None;
    }
    let mut comps: Vec<(f64, f64, f64, Matrix2<f64>)> = Vec::with_capacity(mixture.len());
    let mut map_weight = f64::NEG_INFINITY;
    let mut map_rho_var = f64::NAN;
    let mut map_rhodot_var = f64::NAN;
    for (w, kf) in mixture {
        let c = kf.to_equ_coord().ok()?;
        let s = kf.sky_covariance().ok()?;
        comps.push((w / wsum, c.ra, c.dec, s));
        if *w > map_weight {
            map_weight = *w;
            map_rho_var = kf.covariance[(4, 4)];
            map_rhodot_var = kf.covariance[(5, 5)];
        }
    }
    let (_, ra0, _, _) = *comps.first()?;
    let mut mean_ra_off = 0.0;
    let mut mean_dec = 0.0;
    for (w, ra, dec, _) in &comps {
        mean_ra_off += w * wrap_angle(ra - ra0);
        mean_dec += w * dec;
    }
    let mean_ra = ra0 + mean_ra_off;

    let mut s_comp = Matrix2::zeros();
    let mut s_spread = Matrix2::zeros();
    for (w, ra, dec, s_i) in &comps {
        s_comp += *w * s_i;
        let d = Vector2::new(wrap_angle(ra - mean_ra), dec - mean_dec);
        s_spread += *w * (d * d.transpose());
    }

    // `sky_ellipse_params(_).0` is the semi-major axis √(λmax) already in
    // arcsec; scaling by √(search_region_chi2) matches the radius formula.
    let k = search_region_chi2.max(0.0).sqrt();
    let spread_arcsec = k * sky_ellipse_params(&s_spread).0;
    let component_arcsec = k * sky_ellipse_params(&s_comp).0;
    Some(RadiusDecomp {
        spread_arcsec,
        component_arcsec,
        map_rho_sigma_au: map_rho_var.max(0.0).sqrt(),
        map_rhodot_sigma: map_rhodot_var.max(0.0).sqrt(),
    })
}

/// NEES in the 2-D sky plane: the same quadratic-form shape as
/// [`compute_predictive_nis`], but the error is (predicted sky position −
/// *true* sky position) instead of the innovation (predicted vs. *observed*),
/// and normalised only by
/// $\Sigma_{sky}$ (the estimator's own covariance), not $\Sigma_{sky} + R$ —
/// this is the classical NEES definition $e^\top P^{-1} e$.
fn compute_nees_sky(residual_ra_rad: f64, residual_dec_rad: f64, sigma_sky: &Matrix2<f64>) -> f64 {
    let e = Vector2::new(residual_ra_rad, residual_dec_rad);
    sigma_sky
        .try_inverse()
        .map(|s_inv| (e.transpose() * s_inv * e)[(0, 0)])
        .unwrap_or(f64::NAN)
}

/// Rotate a ground-truth (position, velocity) pair from equatorial J2000 —
/// the frame [`TruthState`]'s Cartesian fields are provided in, matching the
/// equatorial (RA, Dec) convention of their source ephemeris — into the
/// ecliptic J2000 frame `KFState::to_cartesian`/`r_obs`/`v_obs` are expressed
/// in (see [`fink_fat_engine::topocentric_kf::conversion::attributable_to_cartesian`]'s
/// doc: `unit_los` rotates equatorial (α, δ) into ecliptic via
/// `ROT_EQUMJ2000_TO_ECLMJ2000`). Comparing without this rotation produces
/// spurious multi-AU "errors" purely from the ~23.4° axis mismatch between
/// the two frames.
fn truth_pos_vel_ecliptic(truth: &TruthState) -> (Vector3<f64>, Vector3<f64>) {
    (
        ROT_EQUMJ2000_TO_ECLMJ2000 * truth.pos_au,
        ROT_EQUMJ2000_TO_ECLMJ2000 * truth.vel_au_day,
    )
}

/// Full 6-DOF NEES in topocentric Cartesian space (position + velocity),
/// using [`KFState::cartesian_covariance`]. See that method's doc for why
/// comparing the *topocentric offset* (`to_cartesian() - (r_obs, v_obs)`)
/// against ground truth is valid even though `to_cartesian()` itself returns
/// a heliocentric state: subtracting the deterministic observer state
/// changes neither the error nor the covariance.
fn compute_nees_cart(best_kf: &KFState, truth: &TruthState) -> Option<f64> {
    let cart = best_kf.to_cartesian();
    let pos_topo = cart.pos - best_kf.r_obs;
    let vel_topo = cart.vel - best_kf.v_obs;
    let (truth_pos, truth_vel) = truth_pos_vel_ecliptic(truth);

    let e = Vector6::new(
        pos_topo.x - truth_pos.x,
        pos_topo.y - truth_pos.y,
        pos_topo.z - truth_pos.z,
        vel_topo.x - truth_vel.x,
        vel_topo.y - truth_vel.y,
        vel_topo.z - truth_vel.z,
    );

    let cov = best_kf.cartesian_covariance();
    let nees = cov
        .try_inverse()
        .map(|c_inv| (e.transpose() * c_inv * e)[(0, 0)])?;

    // A true quadratic form e^T C^-1 e is never negative for a valid
    // (positive-definite) covariance. A negative or absurdly large result
    // means `cov` was too ill-conditioned for `try_inverse` (plain Gaussian
    // elimination, no regularization) to invert reliably — e.g. a bank that
    // has converged to a near-singular covariance after many observations.
    // Reporting that as a literal NEES value would silently poison
    // dataset-wide aggregates with nonsensical outliers, so treat it as
    // "not computable" instead.
    (nees.is_finite() && nees >= 0.0).then_some(nees)
}

/// Ground-truth comparison metrics for one step: sky-plane and range
/// position error (RMSE inputs, no covariance needed), plus the 2-D and
/// 6-DOF NEES. All `None` (via [`Default`]) when no ground truth is
/// available for this step.
#[derive(Default)]
struct TruthMetrics {
    pos_error_arcsec: Option<f64>,
    range_error_au: Option<f64>,
    cart_pos_error_au: Option<f64>,
    cart_vel_error_au_day: Option<f64>,
    nees_sky: Option<f64>,
    nees_cart: Option<f64>,
}

/// RMSE-oriented ground-truth comparison metrics for one step: sky-plane and
/// range position error (no covariance needed), plus the 2-D and 6-DOF NEES.
fn compute_truth_metrics(
    best_kf: &KFState,
    equ_pred: &EquCoord,
    sigma_sky: &Matrix2<f64>,
    predicted_range_au: f64,
    truth: &TruthState,
) -> TruthMetrics {
    let equ_truth = EquCoord::new(
        truth.ra_deg.to_radians(),
        0.0,
        truth.dec_deg.to_radians(),
        0.0,
    );

    let pos_error_arcsec = equ_pred.angular_separation(&equ_truth).to_degrees() * 3600.;
    let range_error_au = predicted_range_au - truth.range_au;

    let cart = best_kf.to_cartesian();
    let pos_topo = cart.pos - best_kf.r_obs;
    let vel_topo = cart.vel - best_kf.v_obs;
    let (truth_pos, truth_vel) = truth_pos_vel_ecliptic(truth);
    let cart_pos_error_au = (pos_topo - truth_pos).norm();
    let cart_vel_error_au_day = (vel_topo - truth_vel).norm();

    let (_, _, residual_ra_raw_rad) = sky_residuals_arcsec(equ_pred, &equ_truth);
    let residual_dec_raw_rad = truth.dec_deg.to_radians() - equ_pred.dec;
    let nees_sky = compute_nees_sky(residual_ra_raw_rad, residual_dec_raw_rad, sigma_sky);

    let nees_cart = compute_nees_cart(best_kf, truth);

    TruthMetrics {
        pos_error_arcsec: Some(pos_error_arcsec),
        range_error_au: Some(range_error_au),
        cart_pos_error_au: Some(cart_pos_error_au),
        cart_vel_error_au_day: Some(cart_vel_error_au_day),
        nees_sky: Some(nees_sky),
        nees_cart,
    }
}

fn separation_from_search_center(region: &SearchRegion, equ_obs: &EquCoord) -> f64 {
    let equ_center = EquCoord::new(region.center_ra, 0., region.center_dec, 0.);
    equ_center.angular_separation(equ_obs)
}

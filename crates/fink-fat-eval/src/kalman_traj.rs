use fink_fat_engine::topocentric_kf::{
    KFState, KalmanContext,
    kalman_bank::{
        BankStep, KFBank,
        config::KFBankConfig,
        ellipse_region_finder::{MixOrMax, RadiusStrategy, SearchRegion, TopK},
        hypothesis_cap::HypothesisCapSchedule,
        seed_grid::GridConfig,
    },
    update::wrap_angle,
};
use nalgebra::{Matrix2, Vector2};
use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, observation::Observation},
};

const RAD_TO_ARCSEC: f64 = 3600.0 * 180.0 / std::f64::consts::PI;
const ARCSEC_TO_RAD: f64 = 1.0 / RAD_TO_ARCSEC;

// ── Grid / bank initialisation ────────────────────────────────────────────────

fn default_grid_config() -> GridConfig {
    GridConfig {
        rho_max: 100.,
        n_rho: 50,
        n_rho_dot: 15,
        ..GridConfig::default()
    }
}

fn default_bank_config() -> KFBankConfig {
    KFBankConfig {
        gate_chi2: 23.0,
        search_region_chi2: 20.0,
        weight_floor: 1e-5,
        min_hypotheses: 5,
        likelihood_window: 3,
        cap_schedule: HypothesisCapSchedule::Logarithmic {
            start: 500,
            end: 5,
            n_obs_full: 15,
        },
        merge_position_au: 0.02,
    }
}

pub fn init_bank_from_first_pair<'ctx>(
    traj: &[Observation],
    obs_dataset: &ObsDataset,
    context: &'ctx KalmanContext,
) -> Option<(usize, KFBank<'ctx>)> {
    tracing::debug!(
        n_obs = traj.len(),
        "Scanning observations for first intra-night pair (dt < 1 day)"
    );

    let (idx, pair) = traj
        .windows(2)
        .enumerate()
        .find(|(_, w)| w[1].mjd_tt() - w[0].mjd_tt() < 0.5)?;

    println!(
        "Observation pairs used to init the kalman : \n firs obs: \n{}\n\n second obs: \n{}\n\n",
        pair[0], pair[1]
    );

    let grid_config = default_grid_config();

    let bank = KFBank::from_grid(
        obs_dataset,
        &pair[0],
        &pair[1],
        context,
        &grid_config,
        default_bank_config(),
    )
    .ok()?;
    log_bank_init(&bank);
    Some((idx, bank))
}

fn log_bank_init(bank: &KFBank<'_>) {
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

struct StepDiag {
    equ_pred: EquCoord,
    equ_obs: EquCoord,
    residual_ra_arcsec: f64,
    residual_dec_arcsec: f64,
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
}

fn compute_step_diag(
    best_kf: &KFState,
    obs: &Observation,
    region_center: Option<&EquCoord>,
) -> Option<StepDiag> {
    let equ_pred = best_kf.to_equ_coord().ok()?;

    println!("best kf equ_coord : {}", equ_pred);

    let equ_obs = obs.equ_coord();
    let sigma_sky = best_kf.sky_covariance().ok()?;

    let (residual_ra_arcsec, residual_dec_arcsec, residual_ra_raw_rad) =
        sky_residuals_arcsec(&equ_pred, &equ_obs);

    let sigma_ra_arcsec = equ_pred.ra_error * RAD_TO_ARCSEC;
    let sigma_dec_arcsec = equ_pred.dec_error * RAD_TO_ARCSEC;
    let separation_arcsec = equ_pred.angular_separation(&equ_obs).to_degrees() * 3600.;
    let sep_arcsec_region_center = match region_center {
        Some(center) => center.angular_separation(&equ_obs).to_degrees() * 3600.,
        None => return None,
    };

    println!("separation from best kf: {} arcsecond", separation_arcsec);

    let mahalanobis_distance =
        mahalanobis_distance(residual_ra_arcsec, residual_dec_arcsec, &sigma_sky);
    let separation_in_sigma =
        separation_in_sigma(separation_arcsec, sigma_ra_arcsec, sigma_dec_arcsec);

    let (pos_cov_trace_au2, vel_cov_trace_au2_day2, predicted_range_au) =
        filter_health_metrics(best_kf);

    let nis = compute_nis(
        residual_ra_raw_rad,
        residual_dec_arcsec * ARCSEC_TO_RAD,
        &sigma_sky,
        obs,
    );

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
    })
}

// ── Search region ─────────────────────────────────────────────────────────────

struct SearchRegionDiag {
    radius_arcsec: f64,
    semi_major_3sigma_arcsec: f64,
    semi_minor_3sigma_arcsec: f64,
    position_angle_deg: f64,
    obs_within_radius: bool,
}

impl SearchRegionDiag {
    fn nan_fallback() -> Self {
        Self {
            radius_arcsec: f64::NAN,
            semi_major_3sigma_arcsec: f64::NAN,
            semi_minor_3sigma_arcsec: f64::NAN,
            position_angle_deg: f64::NAN,
            obs_within_radius: false,
        }
    }
}

fn compute_search_region(
    bank: &KFBank<'_>,
    obs_dataset: &ObsDataset,
    obs: &Observation,
    context: &KalmanContext,
) -> Option<SearchRegion> {
    let observer = obs_dataset.get_observer(*obs.id())?;
    let helio_state = context
        .get_ephem()
        .helio_observer_state(observer, obs.mjd_tt())
        .ok()?;
    let coord = obs.equ_coord();
    let obs_noise = Vector2::new(
        coord.ra_error * coord.ra_error,
        coord.dec_error * coord.dec_error,
    );
    bank.predict_search_region(
        obs.mjd_tt(),
        helio_state.helio_cart_pos,
        helio_state.helio_cart_vel,
        obs_noise,
        TopK::WeightThreshold(0.99),
        RadiusStrategy::Clamped {
            inner: MixOrMax::MixtureCovariance,
            max_arcsec: 30. * 60., // 30 arcminutes
        },
    )
    .ok()
}

fn search_region_diag(region: &SearchRegion, equ_obs: &EquCoord) -> SearchRegionDiag {
    let radius_arcsec = region.radius_rad * RAD_TO_ARCSEC;

    let best_s = region
        .components
        .iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, _, _, s)| *s);

    let (semi_major, semi_minor, pa_deg) =
        best_s
            .as_ref()
            .map(sky_ellipse_params)
            .unwrap_or((f64::NAN, f64::NAN, f64::NAN));

    let sep = separation_from_search_center(region, equ_obs);

    println!(
        "separation from region center: {} arcsec",
        sep.to_degrees() * 3600.
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

fn log_bank_report(report: &BankStep) {
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

fn log_step_diag(diag: &StepDiag, best_id: u64, step: usize) {
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

fn log_search_region(sr: &SearchRegionDiag, region: &SearchRegion, equ_obs: &EquCoord) {
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

fn assemble_result(
    epoch: f64,
    dt: f64,
    diag: &StepDiag,
    report: &BankStep,
    sr: &SearchRegionDiag,
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
        obs_within_search_radius: sr.obs_within_radius,
        obs_within_3sigma_region: diag.nis <= 9.0,
        nis: diag.nis,
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
}

// ── Main entry point ──────────────────────────────────────────────────────────

pub fn study_kalman_asteroid<'a>(
    traj: &[Observation],
    obs_dataset: &ObsDataset,
    context: &'a KalmanContext,
) -> (Option<KFBank<'a>>, Vec<KFStudyResult>) {
    tracing::debug!(n_obs = traj.len(), "Starting Kalman filter bank study");

    let (idx_first_obs, mut bank) = match init_bank_from_first_pair(traj, obs_dataset, context) {
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
            return (None, Vec::new());
        }
    };

    let observations_to_process = &traj[idx_first_obs + 2..];
    let n_obs = observations_to_process.len();
    tracing::debug!(
        n_obs,
        n_skipped = idx_first_obs + 2,
        "Starting predict/update loop"
    );

    let mut t_prev = traj[idx_first_obs + 1].mjd_tt();
    let mut kf_results: Vec<KFStudyResult> = Vec::with_capacity(n_obs);

    println!("=== KALMAN iteration ===\n");

    for (step, obs) in observations_to_process.iter().enumerate() {
        println!("Processing observation: \n{obs}\n----");

        let epoch = obs.mjd_tt();
        let dt = epoch - t_prev;

        println!("dt = {dt}");

        tracing::trace!(
            step = step + 1,
            n_obs,
            epoch_mjd = epoch,
            dt_days = dt,
            "Step header"
        );

        println!(
            "\n step header: step: {}, epoch: {}, dt: {}",
            step, epoch, dt
        );
        println!("nb hypot in bank: {}", bank.len());

        let region = compute_search_region(&bank, obs_dataset, obs, context);

        println!("\n Predicted region : {:?}\n\n", region);

        tracing::trace!(
            n_hypotheses = bank.len(),
            epoch_mjd = epoch,
            "Stepping bank"
        );

        // Snapshot before step in case it collapses.
        let snapshot = bank.clone();
        let report = bank.step(obs_dataset, obs);
        log_bank_report(&report);

        if report.collapsed {
            println!("Bank collapsed, no more KF in the bank");
            tracing::trace!(step = step + 1, "Bank collapsed, stopping");
            bank = snapshot;
            break;
        }

        let best = match bank.best() {
            Some(b) => b,
            None => {
                tracing::warn!(step = step + 1, "Bank non-empty but best() returned None");
                bank = snapshot;
                break;
            }
        };

        let equ_reg_center = region
            .clone()
            .map(|region| EquCoord::new(region.center_ra, 0., region.center_dec, 0.));
        let diag = match compute_step_diag(&best.kf, obs, equ_reg_center.as_ref()) {
            Some(d) => d,
            None => {
                tracing::warn!(step = step + 1, "compute_step_diag returned None, stopping");
                bank = snapshot;
                break;
            }
        };
        println!(
            " === New attributable KF state: \n\n{}\n ==== \n\n",
            best.kf
        );
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

        t_prev = epoch;
        kf_results.push(assemble_result(epoch, dt, &diag, &report, &sr));
    }

    (Some(bank), kf_results)
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

fn compute_nis(
    residual_ra_rad: f64,
    residual_dec_rad: f64,
    sigma_sky: &Matrix2<f64>,
    obs: &Observation,
) -> f64 {
    let coord = obs.equ_coord();
    let r_mat = Matrix2::from_diagonal(&Vector2::new(
        coord.ra_error * coord.ra_error,
        coord.dec_error * coord.dec_error,
    ));
    let s = sigma_sky + r_mat;
    let nu = Vector2::new(residual_ra_rad, residual_dec_rad);
    s.try_inverse()
        .map(|s_inv| (nu.transpose() * s_inv * nu)[(0, 0)])
        .unwrap_or(f64::NAN)
}

fn separation_from_search_center(region: &SearchRegion, equ_obs: &EquCoord) -> f64 {
    let equ_center = EquCoord::new(region.center_ra, 0., region.center_dec, 0.);
    println!("region center coord: {}", equ_center);
    equ_center.angular_separation(equ_obs)
}

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// Radians -> arcsec, used throughout this file to report angular
/// quantities in a human-friendly unit.
const RAD_TO_ARCSEC: f64 = 206_264.806_247_1;

/// One real observation replayed through the bank's MAP (highest-weight)
/// hypothesis: the state it predicted just *before* absorbing the
/// observation (the "decision" the production pipeline made), the actual
/// observation, and the resulting innovation/χ² and posterior state — plus
/// bank-level diagnostics (hypothesis count, effective sample size, search
/// region size) that only make sense for a multi-hypothesis replay.
///
/// This is reconstructed on demand by [`replay_kalman_branch`] — the
/// database only stores each branch's *final* state, not this per-step
/// history (see the lineage page plan for why a replay is needed instead of
/// a stored table).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct KfStep {
    /// 1-based index among replayed (post-bootstrap) real observations.
    pub step: i32,
    pub obs_id: i64,
    pub epoch: f64,

    pub predicted_ra_deg: f64,
    pub predicted_dec_deg: f64,
    pub sigma_pred_ra_arcsec: f64,
    pub sigma_pred_dec_arcsec: f64,

    pub observed_ra_deg: f64,
    pub observed_dec_deg: f64,
    pub sigma_obs_ra_arcsec: f64,
    pub sigma_obs_dec_arcsec: f64,

    pub residual_ra_arcsec: f64,
    pub residual_dec_arcsec: f64,
    /// Raw angular separation between predicted and observed sky position.
    pub separation_arcsec: f64,
    /// Normalized Innovation Squared (χ², 2 d.o.f.) of the MAP hypothesis at
    /// this step.
    pub nis: f64,
    /// Gaussian log-likelihood of the innovation under the MAP hypothesis. A
    /// proxy for the production `cumulative_llr` (which additionally scores
    /// against a clutter background and photometry) — useful to spot which
    /// points drove the fit, not a reproduction of the stored LLR value.
    pub log_likelihood: f64,

    pub posterior_rho_au: f64,
    pub sigma_rho_au: f64,
    pub posterior_rho_dot_au_per_day: f64,
    pub sigma_rho_dot_au_per_day: f64,

    /// Number of hypotheses surviving in the bank after this step.
    pub n_hypotheses: i32,
    /// Effective sample size `1 / Σ wᵢ²` — close to 1 means one hypothesis
    /// dominates, close to `n_hypotheses` means weights are near-uniform.
    pub effective_sample_size: f64,
    /// Conservative bounding radius of the pre-update mixture search region
    /// (the "error box" the production candidate search would have queried
    /// before absorbing this observation).
    pub search_region_radius_arcsec: f64,
}

/// One hypothesis's predicted `(ρ, ρ̇)` at a given replay step, *before* that
/// step's observation is absorbed — the raw material for a scatter plot of
/// every surviving range/range-rate hypothesis over time, not just the MAP
/// one already tracked in [`KfStep`].
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisSnapshot {
    pub step: i32,
    pub rho_au: f64,
    pub rho_dot_au_per_day: f64,
    pub weight: f64,
}

/// Outcome of a replay: the steps successfully processed, every surviving
/// hypothesis's `(ρ, ρ̇)` at each step, plus (if the bank collapsed — every
/// hypothesis gated out or failed to propagate) a message explaining where
/// and why it stopped early. `truncated_at` being `Some` is not a request
/// failure: `steps`/`hypotheses` still hold everything processed before the
/// stop.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayResult {
    pub steps: Vec<KfStep>,
    pub hypotheses: Vec<HypothesisSnapshot>,
    pub truncated_at: Option<String>,
}

/// Consecutive real observations closer than this are treated as a single
/// epoch — mirrors `fink-fat-eval`'s `dedupe_by_epoch`, guarding against a
/// near-singular innovation covariance from a near-zero Δt.
const DEDUPE_TOLERANCE_DAYS: f64 = 1e-6;

/// Replay the lineage's best branch through a multi-hypothesis Kalman
/// filter bank, real observation by real observation: seed a grid of
/// `(ρ, ρ̇)` hypotheses from the first two observations (same bootstrap the
/// production pipeline uses), then advance the whole bank — predict, gate,
/// update, prune, merge — on each subsequent real observation, exactly like
/// `fink-fat-engine`'s `topocentric_kf::kalman_bank::KFBank`. This replaces
/// an earlier single-hypothesis version of this replay, which committed to
/// one circular-orbit guess for ρ/ρ̇ with no way to recover once that guess
/// turned out wrong — the actual cause of the divergences that motivated
/// this rewrite (see the lineage page plan).
#[server]
pub async fn replay_kalman_branch(
    lineage_designation: String,
) -> Result<ReplayResult, ServerFnError> {
    use fink_fat_engine::topocentric_kf::kalman_bank::KFBank;
    use nalgebra::Vector2;
    use photom::{
        coordinates::equatorial::EquCoord,
        observation_dataset::{observation::ObservationInput, ObsDataset},
        observer::dataset::ObserverId,
        photometry::{Filter, Photometry},
    };

    use std::collections::{HashMap, HashSet};

    use crate::{
        get_engine_config, get_kalman_context, get_observatories, get_pool,
        lineage_page::observations_table::ObservationRow,
    };

    fn mpc_code(code: &str) -> Result<[u8; 3], ServerFnError> {
        code.as_bytes()
            .try_into()
            .map_err(|_| ServerFnError::new(format!("invalid MPC observatory code {code:?}")))
    }

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct ObservationRowSql {
        id: i64,
        object_id: String,
        position: i32,
        mjd_tt: f64,
        ra: f64,
        ra_err: f64,
        dec: f64,
        dec_err: f64,
        magnitude: f64,
        mag_err: f64,
        filter: i16,
        mpc_code_obs: String,
    }

    // Queried directly here (duplicating the small query in
    // `observations_table::get_lineage_observations`) rather than calling
    // that `#[server]` function from within this one: calling another
    // server function is dispatched like any other server-function call
    // (even from server-side code), which round-trips through the same
    // (de)serialization path a browser client would use — and large `i64`
    // observation ids were coming back corrupted through that path.
    let pool = get_pool().await;
    let rows: Vec<ObservationRowSql> = sqlx::query_as(
        "WITH best_branch AS (
            SELECT branch_id
            FROM branches
            WHERE lineage_designation = $1
            ORDER BY (
                CASE
                    WHEN cumulative_llr = 'NaN'::double precision THEN 0
                    WHEN cumulative_llr = 'Infinity'::double precision THEN 0
                    WHEN cumulative_llr = '-Infinity'::double precision THEN 0
                    ELSE cumulative_llr
                END
            ) DESC
            LIMIT 1
        )
        SELECT o.id, o.object_id, bo.position, o.mjd_tt, o.ra, o.ra_err, o.dec, o.dec_err,
               o.magnitude, o.mag_err, o.filter, o.mpc_code_obs
        FROM best_branch bb
        JOIN branch_observations bo ON bo.branch_id = bb.branch_id
        JOIN observations o ON o.id = bo.obs_id
        ORDER BY bo.position",
    )
    .bind(&lineage_designation)
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let mut observations: Vec<ObservationRow> = rows
        .into_iter()
        .map(|r| ObservationRow {
            id: r.id,
            object_id: r.object_id,
            position: r.position,
            mjd_tt: r.mjd_tt,
            ra: r.ra,
            ra_err: r.ra_err,
            dec: r.dec,
            dec_err: r.dec_err,
            magnitude: r.magnitude,
            mag_err: r.mag_err,
            filter: r.filter,
            mpc_code_obs: r.mpc_code_obs,
        })
        .collect();
    observations.sort_by(|a, b| a.mjd_tt.total_cmp(&b.mjd_tt));
    observations.dedup_by(|a, b| (a.mjd_tt - b.mjd_tt).abs() < DEDUPE_TOLERANCE_DAYS);

    if observations.len() < 2 {
        return Ok(ReplayResult {
            steps: Vec::new(),
            hypotheses: Vec::new(),
            truncated_at: None,
        });
    }

    // Resolve each distinct observatory once against our own pre-fetched MPC
    // lookup table (`get_observatories`), rather than letting `ObsDataset`
    // resolve `ObserverId::MpcCode` lazily on first use — that lazy path
    // silently returns "no observer" (`OutfitError::ObserverIdIsNone`) if
    // its own fetch fails, which is much harder to diagnose than a clear
    // error here.
    let observatories = get_observatories().await;
    let mut dataset = ObsDataset::empty();
    let mut observer_ids: HashMap<[u8; 3], ObserverId> = HashMap::new();
    for code in observations
        .iter()
        .map(|o| mpc_code(&o.mpc_code_obs))
        .collect::<Result<HashSet<_>, _>>()?
    {
        let observer = observatories.get(&code).ok_or_else(|| {
            ServerFnError::new(format!(
                "MPC observatory code {:?} not found in the observatory list",
                std::str::from_utf8(&code).unwrap_or("?")
            ))
        })?;
        let (new_dataset, id) = dataset.push_observer(observer.clone());
        dataset = new_dataset;
        observer_ids.insert(code, id);
    }

    let mut inputs = Vec::with_capacity(observations.len());
    for obs in &observations {
        let code = mpc_code(&obs.mpc_code_obs)?;
        inputs.push(ObservationInput::new(
            obs.id as u64,
            EquCoord::new(obs.ra, obs.ra_err, obs.dec, obs.dec_err),
            Photometry {
                magnitude: obs.magnitude,
                error: obs.mag_err,
                filter: Filter::Int(obs.filter as u32),
            },
            obs.mjd_tt,
            Some(observer_ids[&code]),
        ));
    }

    let (obs_dataset, _) = dataset
        .push_observation(inputs)
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    let get_obs = |id: i64| {
        obs_dataset
            .get_observation(id as u64)
            .expect("observation was just inserted under this id")
    };

    let engine_config = get_engine_config().await;
    let kalman_context = get_kalman_context().await;

    let mut bank = KFBank::from_grid(
        &obs_dataset,
        get_obs(observations[0].id),
        get_obs(observations[1].id),
        kalman_context,
        &engine_config.seeding_grid_config,
        &engine_config.kfbank_config,
    )
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    if bank.is_empty() {
        return Ok(ReplayResult {
            steps: Vec::new(),
            hypotheses: Vec::new(),
            truncated_at: Some(
                "bootstrap produced no admissible (ρ, ρ̇) hypotheses for this tracklet".into(),
            ),
        });
    }

    let total_steps = observations.len() - 2;
    let mut steps = Vec::with_capacity(total_steps);
    let mut hypotheses = Vec::new();
    let mut truncated_at: Option<String> = None;

    for (step_idx, obs_row) in observations.iter().enumerate().skip(2) {
        // Each fallible stage is tried in this closure rather than with `?`
        // directly in the loop: a failure here (the whole bank gated out or
        // failed to propagate) should truncate the replay, not fail the
        // whole request.
        let step_result: Result<(KfStep, Vec<HypothesisSnapshot>), String> = (|| {
            let step_number = (step_idx - 1) as i32;
            let observation = get_obs(obs_row.id);

            let observer = obs_dataset
                .get_observer(*observation.id())
                .expect("observer resolved from mpc_code_obs at dataset build time");
            let helio = kalman_context
                .get_ephem()
                .helio_observer_state(observer, obs_row.mjd_tt)
                .map_err(|e| e.to_string())?;

            // Every surviving hypothesis's pre-update predicted (ρ, ρ̇) — the
            // raw material for the "ρ/ρ̇ of every hypothesis" plot, and the
            // source of the MAP prediction used below (mirrors what the
            // single-hypothesis replay used to compute from its lone `kf`).
            let mixture =
                bank.predicted_mixture(obs_row.mjd_tt, helio.helio_cart_pos, helio.helio_cart_vel);
            if mixture.is_empty() {
                return Err("no hypothesis could be propagated to this epoch".to_string());
            }
            let hyp_snapshots: Vec<HypothesisSnapshot> = mixture
                .iter()
                .map(|(weight, kf)| HypothesisSnapshot {
                    step: step_number,
                    rho_au: kf.state[4],
                    rho_dot_au_per_day: kf.state[5],
                    weight: *weight,
                })
                .collect();

            let (_, best_pred_kf) = mixture
                .iter()
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .expect("mixture is non-empty");

            let equ_pred = best_pred_kf.to_equ_coord().map_err(|e| e.to_string())?;
            let sigma_sky = best_pred_kf.sky_covariance().map_err(|e| e.to_string())?;

            let obs_noise = Vector2::new(
                obs_row.ra_err * obs_row.ra_err,
                obs_row.dec_err * obs_row.dec_err,
            );
            let region = bank
                .predict_search_region(
                    obs_row.mjd_tt,
                    helio.helio_cart_pos,
                    helio.helio_cart_vel,
                    obs_noise,
                    engine_config.advance_params.top_k,
                    engine_config.advance_params.radius_strategy,
                )
                .map_err(|e| e.to_string())?;

            let residual_ra = fink_fat_engine::topocentric_kf::single_kalman::update::wrap_angle(
                obs_row.ra - equ_pred.ra,
            );
            let residual_dec = obs_row.dec - equ_pred.dec;
            let innovation = Vector2::new(residual_ra, residual_dec);

            let r = nalgebra::Matrix2::from_diagonal(&obs_noise);
            let s = sigma_sky + r;
            let s_inv = s.try_inverse().unwrap_or_else(nalgebra::Matrix2::zeros);

            let nis = (innovation.transpose() * s_inv * innovation)[(0, 0)];
            let log_likelihood = -0.5
                * (nis
                    + (2.0 * std::f64::consts::PI * s)
                        .determinant()
                        .max(1e-300)
                        .ln());

            let separation_rad = (residual_ra * residual_dec.cos()).hypot(residual_dec).abs();

            // Actually advance the bank: propagate + gate + score + update +
            // prune + merge, all in one call. `r_obs`/`v_obs` are already
            // resolved above, so `step_with_geometry` skips redoing that
            // lookup internally.
            let bank_step =
                bank.step_with_geometry(helio.helio_cart_pos, helio.helio_cart_vel, observation);
            if bank_step.collapsed {
                return Err(
                    "every hypothesis was gated out or failed to update on this observation"
                        .to_string(),
                );
            }

            let best_after = bank
                .best()
                .expect("bank.best() is Some when bank_step.collapsed is false");

            let step = KfStep {
                step: step_number,
                obs_id: obs_row.id,
                epoch: obs_row.mjd_tt,

                predicted_ra_deg: equ_pred.ra.to_degrees(),
                predicted_dec_deg: equ_pred.dec.to_degrees(),
                sigma_pred_ra_arcsec: sigma_sky[(0, 0)].max(0.0).sqrt() * RAD_TO_ARCSEC,
                sigma_pred_dec_arcsec: sigma_sky[(1, 1)].max(0.0).sqrt() * RAD_TO_ARCSEC,

                observed_ra_deg: obs_row.ra.to_degrees(),
                observed_dec_deg: obs_row.dec.to_degrees(),
                sigma_obs_ra_arcsec: obs_row.ra_err * RAD_TO_ARCSEC,
                sigma_obs_dec_arcsec: obs_row.dec_err * RAD_TO_ARCSEC,

                residual_ra_arcsec: residual_ra * RAD_TO_ARCSEC,
                residual_dec_arcsec: residual_dec * RAD_TO_ARCSEC,
                separation_arcsec: separation_rad * RAD_TO_ARCSEC,
                nis,
                log_likelihood,

                posterior_rho_au: best_after.kf.state[4],
                sigma_rho_au: best_after.kf.covariance[(4, 4)].max(0.0).sqrt(),
                posterior_rho_dot_au_per_day: best_after.kf.state[5],
                sigma_rho_dot_au_per_day: best_after.kf.covariance[(5, 5)].max(0.0).sqrt(),

                n_hypotheses: bank_step.n_after as i32,
                effective_sample_size: bank_step.n_effective,
                search_region_radius_arcsec: region.radius_rad * RAD_TO_ARCSEC,
            };

            Ok((step, hyp_snapshots))
        })();

        match step_result {
            Ok((step, hyp_snapshots)) => {
                steps.push(step);
                hypotheses.extend(hyp_snapshots);
            }
            Err(message) => {
                truncated_at = Some(format!(
                    "stopped after {}/{total_steps} observations: {message}",
                    steps.len(),
                ));
                break;
            }
        }
    }

    Ok(ReplayResult {
        steps,
        hypotheses,
        truncated_at,
    })
}

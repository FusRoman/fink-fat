//! Export [`KFStudyResult`] / [`TrajSummary`] series to Parquet via Polars,
//! for offline analysis (NEES/NIS/RMSE calibration checks, plotting) outside
//! this crate's own text reports.

use std::fs::File;

use anyhow::{Context, Result};
use camino::Utf8Path;
use photom::TrajId;
use polars::prelude::*;

use crate::{kalman_traj::KFStudyResult, trajectory_processing::TrajSummary};

/// Write one row per predict/update step across every `(TrajId, results)` pair
/// to a Parquet file at `out_path`.
pub fn export_steps_parquet(
    rows: &[(TrajId, Vec<KFStudyResult>)],
    out_path: impl AsRef<Utf8Path>,
) -> Result<()> {
    let n: usize = rows.iter().map(|(_, r)| r.len()).sum();

    let mut traj_id = Vec::with_capacity(n);
    let mut epoch = Vec::with_capacity(n);
    let mut dt = Vec::with_capacity(n);
    let mut separation_arcsec_from_best_kf = Vec::with_capacity(n);
    let mut separation_arcsec_from_region = Vec::with_capacity(n);
    let mut residual_ra_arcsec = Vec::with_capacity(n);
    let mut residual_dec_arcsec = Vec::with_capacity(n);
    let mut sigma_ra_arcsec = Vec::with_capacity(n);
    let mut sigma_dec_arcsec = Vec::with_capacity(n);
    let mut mahalanobis_distance = Vec::with_capacity(n);
    let mut separation_in_sigma = Vec::with_capacity(n);
    let mut pos_cov_trace_au2 = Vec::with_capacity(n);
    let mut vel_cov_trace_au2_day2 = Vec::with_capacity(n);
    let mut predicted_range_au = Vec::with_capacity(n);
    let mut gain_frobenius_norm = Vec::with_capacity(n);
    let mut n_hypotheses_before = Vec::with_capacity(n);
    let mut n_hypotheses_after = Vec::with_capacity(n);
    let mut n_gated = Vec::with_capacity(n);
    let mut n_effective = Vec::with_capacity(n);
    let mut best_weight = Vec::with_capacity(n);
    let mut search_region_radius_arcsec = Vec::with_capacity(n);
    let mut region_semi_major_3sigma_arcsec = Vec::with_capacity(n);
    let mut region_semi_minor_3sigma_arcsec = Vec::with_capacity(n);
    let mut region_position_angle_deg = Vec::with_capacity(n);
    let mut obs_within_search_radius = Vec::with_capacity(n);
    let mut obs_within_3sigma_region = Vec::with_capacity(n);
    let mut nis = Vec::with_capacity(n);
    let mut pos_error_arcsec: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut range_error_au: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut cart_pos_error_au: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut cart_vel_error_au_day: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut nees_sky: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut nees_cart: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut inflation_lambda = Vec::with_capacity(n);

    for (id, results) in rows {
        for r in results {
            traj_id.push(id.to_string());
            epoch.push(r.epoch);
            dt.push(r.dt);
            separation_arcsec_from_best_kf.push(r.separation_arcsec_from_best_kf);
            separation_arcsec_from_region.push(r.separation_arcsec_from_region);
            residual_ra_arcsec.push(r.residual_ra_arcsec);
            residual_dec_arcsec.push(r.residual_dec_arcsec);
            sigma_ra_arcsec.push(r.sigma_ra_arcsec);
            sigma_dec_arcsec.push(r.sigma_dec_arcsec);
            mahalanobis_distance.push(r.mahalanobis_distance);
            separation_in_sigma.push(r.separation_in_sigma);
            pos_cov_trace_au2.push(r.pos_cov_trace_au2);
            vel_cov_trace_au2_day2.push(r.vel_cov_trace_au2_day2);
            predicted_range_au.push(r.predicted_range_au);
            gain_frobenius_norm.push(r.gain_frobenius_norm);
            n_hypotheses_before.push(r.n_hypotheses_before as u32);
            n_hypotheses_after.push(r.n_hypotheses_after as u32);
            n_gated.push(r.n_gated as u32);
            n_effective.push(r.n_effective);
            best_weight.push(r.best_weight);
            search_region_radius_arcsec.push(r.search_region_radius_arcsec);
            region_semi_major_3sigma_arcsec.push(r.region_semi_major_3sigma_arcsec);
            region_semi_minor_3sigma_arcsec.push(r.region_semi_minor_3sigma_arcsec);
            region_position_angle_deg.push(r.region_position_angle_deg);
            obs_within_search_radius.push(r.obs_within_search_radius);
            obs_within_3sigma_region.push(r.obs_within_3sigma_region);
            nis.push(r.nis);
            pos_error_arcsec.push(r.pos_error_arcsec);
            range_error_au.push(r.range_error_au);
            cart_pos_error_au.push(r.cart_pos_error_au);
            cart_vel_error_au_day.push(r.cart_vel_error_au_day);
            nees_sky.push(r.nees_sky);
            nees_cart.push(r.nees_cart);
            inflation_lambda.push(r.inflation_lambda);
        }
    }

    let mut df = DataFrame::new_infer_height(vec![
        Column::new("traj_id".into(), traj_id),
        Column::new("epoch".into(), epoch),
        Column::new("dt".into(), dt),
        Column::new(
            "separation_arcsec_from_best_kf".into(),
            separation_arcsec_from_best_kf,
        ),
        Column::new(
            "separation_arcsec_from_region".into(),
            separation_arcsec_from_region,
        ),
        Column::new("residual_ra_arcsec".into(), residual_ra_arcsec),
        Column::new("residual_dec_arcsec".into(), residual_dec_arcsec),
        Column::new("sigma_ra_arcsec".into(), sigma_ra_arcsec),
        Column::new("sigma_dec_arcsec".into(), sigma_dec_arcsec),
        Column::new("mahalanobis_distance".into(), mahalanobis_distance),
        Column::new("separation_in_sigma".into(), separation_in_sigma),
        Column::new("pos_cov_trace_au2".into(), pos_cov_trace_au2),
        Column::new("vel_cov_trace_au2_day2".into(), vel_cov_trace_au2_day2),
        Column::new("predicted_range_au".into(), predicted_range_au),
        Column::new("gain_frobenius_norm".into(), gain_frobenius_norm),
        Column::new("n_hypotheses_before".into(), n_hypotheses_before),
        Column::new("n_hypotheses_after".into(), n_hypotheses_after),
        Column::new("n_gated".into(), n_gated),
        Column::new("n_effective".into(), n_effective),
        Column::new("best_weight".into(), best_weight),
        Column::new(
            "search_region_radius_arcsec".into(),
            search_region_radius_arcsec,
        ),
        Column::new(
            "region_semi_major_3sigma_arcsec".into(),
            region_semi_major_3sigma_arcsec,
        ),
        Column::new(
            "region_semi_minor_3sigma_arcsec".into(),
            region_semi_minor_3sigma_arcsec,
        ),
        Column::new(
            "region_position_angle_deg".into(),
            region_position_angle_deg,
        ),
        Column::new("obs_within_search_radius".into(), obs_within_search_radius),
        Column::new("obs_within_3sigma_region".into(), obs_within_3sigma_region),
        Column::new("nis".into(), nis),
        Column::new("pos_error_arcsec".into(), pos_error_arcsec),
        Column::new("range_error_au".into(), range_error_au),
        Column::new("cart_pos_error_au".into(), cart_pos_error_au),
        Column::new("cart_vel_error_au_day".into(), cart_vel_error_au_day),
        Column::new("nees_sky".into(), nees_sky),
        Column::new("nees_cart".into(), nees_cart),
        Column::new("inflation_lambda".into(), inflation_lambda),
    ])
    .context("failed to build per-step metrics DataFrame")?;

    let out_path = out_path.as_ref();
    let file = File::create(out_path)
        .with_context(|| format!("failed to create parquet file at {out_path}"))?;
    ParquetWriter::new(file)
        .finish(&mut df)
        .with_context(|| format!("failed to write parquet file at {out_path}"))?;

    Ok(())
}

/// Write one row per trajectory summary to a Parquet file at `out_path`.
pub fn export_summary_parquet(
    summaries: &[TrajSummary],
    out_path: impl AsRef<Utf8Path>,
) -> Result<()> {
    let n = summaries.len();

    let mut traj_id = Vec::with_capacity(n);
    let mut n_obs_total = Vec::with_capacity(n);
    let mut n_steps = Vec::with_capacity(n);
    let mut completion_fraction = Vec::with_capacity(n);
    let mut stop_reason = Vec::with_capacity(n);
    let mut n_obs_before_bootstrap = Vec::with_capacity(n);
    let mut n_obs_deduplicated = Vec::with_capacity(n);
    let mut n_processable = Vec::with_capacity(n);
    let mut pct_within_3sigma = Vec::with_capacity(n);
    let mut pct_within_search_radius = Vec::with_capacity(n);
    let mut mean_separation_arcsec = Vec::with_capacity(n);
    let mut median_separation_arcsec = Vec::with_capacity(n);
    let mut mean_nis = Vec::with_capacity(n);
    let mut nis_median = Vec::with_capacity(n);
    let mut pct_nis_in_chi2_band = Vec::with_capacity(n);
    let mut nis_calibration_ratio = Vec::with_capacity(n);
    let mut mean_mahalanobis = Vec::with_capacity(n);
    let mut mean_search_radius_arcsec = Vec::with_capacity(n);
    let mut mean_n_hypotheses_after = Vec::with_capacity(n);
    let mut mean_effective_sample_size = Vec::with_capacity(n);
    let mut n_steps_with_truth = Vec::with_capacity(n);
    let mut rmse_pos_arcsec = Vec::with_capacity(n);
    let mut rmse_range_au = Vec::with_capacity(n);
    let mut rmse_cart_pos_au = Vec::with_capacity(n);
    let mut rmse_cart_vel_au_day = Vec::with_capacity(n);
    let mut mean_nees_sky = Vec::with_capacity(n);
    let mut pct_nees_sky_in_chi2_band = Vec::with_capacity(n);
    let mut mean_nees_cart = Vec::with_capacity(n);
    let mut pct_nees_cart_in_chi2_band = Vec::with_capacity(n);
    let mut mean_inflation_lambda = Vec::with_capacity(n);
    let mut pct_steps_inflation_active = Vec::with_capacity(n);

    for s in summaries {
        traj_id.push(s.traj_id.to_string());
        n_obs_total.push(s.n_obs_total as u32);
        n_steps.push(s.n_steps as u32);
        completion_fraction.push(s.completion_fraction);
        stop_reason.push(s.stop_reason.label().to_string());
        n_obs_before_bootstrap.push(s.n_obs_before_bootstrap as u32);
        n_obs_deduplicated.push(s.n_obs_deduplicated as u32);
        n_processable.push(s.n_processable as u32);
        pct_within_3sigma.push(s.pct_within_3sigma);
        pct_within_search_radius.push(s.pct_within_search_radius);
        mean_separation_arcsec.push(s.mean_separation_arcsec);
        median_separation_arcsec.push(s.median_separation_arcsec);
        mean_nis.push(s.mean_nis);
        nis_median.push(s.nis_median);
        pct_nis_in_chi2_band.push(s.pct_nis_in_chi2_band);
        nis_calibration_ratio.push(s.nis_calibration_ratio);
        mean_mahalanobis.push(s.mean_mahalanobis);
        mean_search_radius_arcsec.push(s.mean_search_radius_arcsec);
        mean_n_hypotheses_after.push(s.mean_n_hypotheses_after);
        mean_effective_sample_size.push(s.mean_effective_sample_size);
        n_steps_with_truth.push(s.n_steps_with_truth as u32);
        rmse_pos_arcsec.push(s.rmse_pos_arcsec);
        rmse_range_au.push(s.rmse_range_au);
        rmse_cart_pos_au.push(s.rmse_cart_pos_au);
        rmse_cart_vel_au_day.push(s.rmse_cart_vel_au_day);
        mean_nees_sky.push(s.mean_nees_sky);
        pct_nees_sky_in_chi2_band.push(s.pct_nees_sky_in_chi2_band);
        mean_nees_cart.push(s.mean_nees_cart);
        pct_nees_cart_in_chi2_band.push(s.pct_nees_cart_in_chi2_band);
        mean_inflation_lambda.push(s.mean_inflation_lambda);
        pct_steps_inflation_active.push(s.pct_steps_inflation_active);
    }

    let mut df = DataFrame::new_infer_height(vec![
        Column::new("traj_id".into(), traj_id),
        Column::new("n_obs_total".into(), n_obs_total),
        Column::new("n_steps".into(), n_steps),
        Column::new("completion_fraction".into(), completion_fraction),
        Column::new("stop_reason".into(), stop_reason),
        Column::new("n_obs_before_bootstrap".into(), n_obs_before_bootstrap),
        Column::new("n_obs_deduplicated".into(), n_obs_deduplicated),
        Column::new("n_processable".into(), n_processable),
        Column::new("pct_within_3sigma".into(), pct_within_3sigma),
        Column::new("pct_within_search_radius".into(), pct_within_search_radius),
        Column::new("mean_separation_arcsec".into(), mean_separation_arcsec),
        Column::new("median_separation_arcsec".into(), median_separation_arcsec),
        Column::new("mean_nis".into(), mean_nis),
        Column::new("nis_median".into(), nis_median),
        Column::new("pct_nis_in_chi2_band".into(), pct_nis_in_chi2_band),
        Column::new("nis_calibration_ratio".into(), nis_calibration_ratio),
        Column::new("mean_mahalanobis".into(), mean_mahalanobis),
        Column::new(
            "mean_search_radius_arcsec".into(),
            mean_search_radius_arcsec,
        ),
        Column::new("mean_n_hypotheses_after".into(), mean_n_hypotheses_after),
        Column::new(
            "mean_effective_sample_size".into(),
            mean_effective_sample_size,
        ),
        Column::new("n_steps_with_truth".into(), n_steps_with_truth),
        Column::new("rmse_pos_arcsec".into(), rmse_pos_arcsec),
        Column::new("rmse_range_au".into(), rmse_range_au),
        Column::new("rmse_cart_pos_au".into(), rmse_cart_pos_au),
        Column::new("rmse_cart_vel_au_day".into(), rmse_cart_vel_au_day),
        Column::new("mean_nees_sky".into(), mean_nees_sky),
        Column::new(
            "pct_nees_sky_in_chi2_band".into(),
            pct_nees_sky_in_chi2_band,
        ),
        Column::new("mean_nees_cart".into(), mean_nees_cart),
        Column::new(
            "pct_nees_cart_in_chi2_band".into(),
            pct_nees_cart_in_chi2_band,
        ),
        Column::new("mean_inflation_lambda".into(), mean_inflation_lambda),
        Column::new(
            "pct_steps_inflation_active".into(),
            pct_steps_inflation_active,
        ),
    ])
    .context("failed to build trajectory-summary DataFrame")?;

    let out_path = out_path.as_ref();
    let file = File::create(out_path)
        .with_context(|| format!("failed to create parquet file at {out_path}"))?;
    ParquetWriter::new(file)
        .finish(&mut df)
        .with_context(|| format!("failed to write parquet file at {out_path}"))?;

    Ok(())
}

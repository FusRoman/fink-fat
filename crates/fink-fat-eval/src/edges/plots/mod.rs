//! Edge evaluation plot suite.
//!
//! | Sub-module        | Responsibility                                              |
//! |-------------------|-------------------------------------------------------------|
//! [`distributions`]  | TP vs FP feature distributions (cost, χ², kinematics, …)   |
//! [`predictor_diag`] | Predictor-config diagnostics (offset, cone radius, ratio)   |
//!
//! # Entry-point
//!
//! ```ignore
//! edges::plots::edge_plots(ctx, truth, out_dir)?;
//! ```

pub mod distributions;
pub mod predictor_diag;

use anyhow::Result;
use camino::Utf8Path;
use fink_fat_engine::pipeline::PipelineContext;

use crate::truth_sso::TruthSSO;

use distributions::{collect_edge_distrib_data, plot_edge_distributions};
use predictor_diag::{collect_predictor_data, plot_predictor_diagnostics};

/// Generate the full edge plot suite and write all PNGs to `out_dir`.
///
/// Sub-directories are created automatically if absent.
///
/// # Plots produced
///
/// **Feature distributions (TP vs FP)**:
/// - `edge_cost.png`              – solver-facing edge cost (log scale)
/// - `edge_dt_days.png`           – time gap between seeds (days)
/// - `edge_chi2_pos.png`          – position Mahalanobis χ² (log scale)
/// - `edge_chi2_vel.png`          – velocity Mahalanobis χ² (log scale)
/// - `edge_cos_dtheta_v.png`      – direction alignment cos Δθᵥ
/// - `edge_rel_speed_diff.png`    – relative speed difference
/// - `edge_innov_speed_ratio.png` – innovation-speed ratio (log scale)
/// - `edge_z_flux.png`            – photometry flux z-score
///
/// **Predictor-config diagnostics (TP vs FP)**:
/// - `predictor_angular_offset.png` – predicted-to-actual angular error (arcmin, log)
/// - `predictor_cone_radius.png`    – base cone radius = k_σ·√λ_max (arcmin, log)
/// - `predictor_offset_ratio.png`   – δ / r_base  (cut guide at 1.0)
pub fn edge_plots(ctx: &PipelineContext<'_>, truth: &TruthSSO, out_dir: &Utf8Path) -> Result<()> {
    tracing::info!("computing edge feature distributions (TP vs FP)…");
    let data = collect_edge_distrib_data(ctx, truth)?;
    plot_edge_distributions(data, out_dir)?;

    tracing::info!("computing predictor-config diagnostics…");
    let pred_data = collect_predictor_data(ctx, truth)?;
    let pred_params = ctx.engine_config.edges.predictor_config.clone();
    plot_predictor_diagnostics(pred_data, &pred_params, out_dir)?;

    tracing::info!("edge plots written to {out_dir}");
    Ok(())
}

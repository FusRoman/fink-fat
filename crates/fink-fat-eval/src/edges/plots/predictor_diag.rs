//! Predictor-config diagnostic plots (TP vs FP).
//!
//! **Goal**: help the user decide how to tune [`PredictorParams`] — in
//! particular `k_sigma` and the [`ModelNoise`] schedule — to improve edge
//! purity without hurting recall.
//!
//! For each edge `from → to` (classified as TP or FP) the module computes:
//!
//! | Quantity                 | Symbol                        | Unit    |
//! |--------------------------|-------------------------------|---------|
//! | Angular prediction error | δ = ang_sep(predicted, actual)| arcmin  |
//! | Base cone radius         | r = k_σ · √λ_max(Σ_p)        | arcmin  |
//! | Normalised offset        | δ / r                         | –       |
//!
//! and writes three three-panel overlay charts (histogram / CDF / percentile)
//! with **TP in green** and **FP in red**:
//!
//! | File                              | Metric                         |
//! |-----------------------------------|--------------------------------|
//! `predictor_angular_offset.png`      | δ — prediction error (arcmin)  |
//! `predictor_cone_radius.png`         | r — base cone radius (arcmin)  |
//! `predictor_offset_ratio.png`        | δ / r  (cut guide at 1.0)      |
//!
//! ## Reading the plots
//!
//! * **`predictor_angular_offset.png`**: shows how far the prediction is from
//!   the actual position.  A large separation between TP and FP curves means
//!   the predictor already separates classes spatially.
//!
//! * **`predictor_cone_radius.png`**: shows the current cone size.  A very
//!   wide cone relative to the offset is a sign that `k_sigma` is larger than
//!   needed.
//!
//! * **`predictor_offset_ratio.png`**: the key diagnostic.
//!   - If TP offsets cluster well **below 1.0** while FP offsets extend beyond
//!     1.0, reducing `k_sigma` will cut FPs without losing TPs.
//!   - If TP offsets regularly reach 1.0, the cone is already tight — reduce
//!     `k_sigma` cautiously or increase `ModelNoise` instead.
//!   - The vertical red line is drawn at ratio = 1.0 (boundary of the cone).

use std::path::Path;

use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::{
    astro_math::ang_sep, engine_config::propagator_config::PredictorParams,
    pipeline::PipelineContext,
};
use plotters::prelude::*;

use crate::{
    seeding::plots::chart_utils::{PERCENTILE_PS, percentile_sorted},
    seeding::plots::draw_helpers::{C_CUTOFF, PANEL_H, PLOT_W, fmt_log10_tick},
    truth_sso::{TruthClass, TruthSSO},
};

// ─────────────────────────────────────────────────────────────────────────────
// Colours
// ─────────────────────────────────────────────────────────────────────────────

const C_TP: RGBColor = RGBColor(34, 139, 34);
const C_FP: RGBColor = RGBColor(220, 20, 60);

// ─────────────────────────────────────────────────────────────────────────────
// Data container
// ─────────────────────────────────────────────────────────────────────────────

/// Raw predictor diagnostic quantities, split by TP / FP class.
#[derive(Default)]
pub struct EdgePredictorData {
    /// Angular distance from predicted to actual position (arcmin), TP edges.
    pub angular_offset_arcmin_tp: Vec<f64>,
    /// Angular distance from predicted to actual position (arcmin), FP edges.
    pub angular_offset_arcmin_fp: Vec<f64>,

    /// Base cone radius = k_σ · √λ_max(Σ_p) (arcmin), TP edges.
    /// Does **not** include `pad_cell_radius` or `v_slack`.
    pub cone_radius_arcmin_tp: Vec<f64>,
    /// Base cone radius (arcmin), FP edges.
    pub cone_radius_arcmin_fp: Vec<f64>,

    /// Normalised offset = angular_offset / cone_radius_base, TP edges.
    pub offset_ratio_tp: Vec<f64>,
    /// Normalised offset, FP edges.
    pub offset_ratio_fp: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Collection
// ─────────────────────────────────────────────────────────────────────────────

/// Build [`EdgePredictorData`] by re-running the cone prediction for each edge.
///
/// Uses only `noise` and `k_sigma` from `predictor_params` (i.e., the base
/// cone without cell-padding or velocity slack) so that the plots show the
/// tightest achievable cone at the given `k_sigma`.
pub fn collect_predictor_data(
    ctx: &PipelineContext,
    truth: &TruthSSO,
) -> Result<EdgePredictorData> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;
    let pred_params: &PredictorParams = &ctx.engine_config.edges.predictor_config;

    let mut data = EdgePredictorData::default();

    for edge in ctx.runtime_state.graph.edges.iter() {
        // ── Classify ──────────────────────────────────────────────────────────
        let from_seed = seed_store
            .try_get_seed(edge.from)
            .context("from seed not found")?;
        let to_seed = seed_store
            .try_get_seed(edge.to)
            .context("to seed not found")?;

        let alerts: Vec<_> = from_seed
            .resolve_members(alert_store)
            .context("from seed")?
            .into_iter()
            .chain(
                to_seed
                    .resolve_members(alert_store)
                    .context("to seed")?
                    .into_iter(),
            )
            .collect();

        let class = truth.classify(&alerts);
        if matches!(class, TruthClass::Unknown) {
            continue;
        }

        // ── Predict position at to's epoch ────────────────────────────────────
        let t_target = to_seed.plane.epoch_mid;
        let (ra_pred, dec_pred, r_base) =
            from_seed
                .plane
                .predict_cone_base(t_target, &pred_params.noise, pred_params.k_sigma);

        // ── Angular offset from predicted to actual to-seed position ──────────
        let offset_rad = ang_sep(
            ra_pred,
            dec_pred,
            to_seed.plane.ra_mid,
            to_seed.plane.dec_mid,
        );
        let offset_arcmin = offset_rad.to_degrees() * 60.0;
        let cone_arcmin = r_base.to_degrees() * 60.0;

        // Guard against a degenerate (zero) cone radius.
        let ratio = if r_base > 1e-12 {
            offset_rad / r_base
        } else {
            f64::NAN
        };

        match class {
            TruthClass::TruePositive => {
                data.angular_offset_arcmin_tp.push(offset_arcmin);
                data.cone_radius_arcmin_tp.push(cone_arcmin);
                if ratio.is_finite() {
                    data.offset_ratio_tp.push(ratio);
                }
            }
            TruthClass::FalsePositive => {
                data.angular_offset_arcmin_fp.push(offset_arcmin);
                data.cone_radius_arcmin_fp.push(cone_arcmin);
                if ratio.is_finite() {
                    data.offset_ratio_fp.push(ratio);
                }
            }
            TruthClass::Unknown => {}
        }
    }

    Ok(data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level plot entrypoint
// ─────────────────────────────────────────────────────────────────────────────

/// Write predictor diagnostic PNGs to `out_dir`.
pub fn plot_predictor_diagnostics(
    data: EdgePredictorData,
    pred_params: &PredictorParams,
    out_dir: &Utf8Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let d = out_dir.as_std_path();

    let EdgePredictorData {
        angular_offset_arcmin_tp,
        angular_offset_arcmin_fp,
        cone_radius_arcmin_tp,
        cone_radius_arcmin_fp,
        offset_ratio_tp,
        offset_ratio_fp,
    } = data;

    // 1. Angular prediction error (log scale)
    overlay_metric(
        angular_offset_arcmin_tp,
        angular_offset_arcmin_fp,
        &format!(
            "Prediction error (k_sigma={:.2}, floor={:.2e}, drift={:.2e}/day, curv={:.2e}/day²)",
            pred_params.k_sigma,
            pred_params.noise.variance_floor,
            pred_params.noise.drift_per_day,
            pred_params.noise.curvature_per_day2,
        ),
        "Angular offset  [arcmin]",
        true, // log_x
        None,
        &d.join("predictor_angular_offset.png"),
    )?;

    // 2. Base cone radius (log scale)
    overlay_metric(
        cone_radius_arcmin_tp,
        cone_radius_arcmin_fp,
        &format!("Base cone radius  (k_sigma={:.2})", pred_params.k_sigma),
        "Cone radius  [arcmin]",
        true, // log_x
        None,
        &d.join("predictor_cone_radius.png"),
    )?;

    // 3. Normalised offset (linear scale, vline at ratio = 1.0)
    overlay_metric(
        offset_ratio_tp,
        offset_ratio_fp,
        &format!(
            "Normalised offset = δ / r_base  (k_sigma={:.2}) — cut guide at 1.0",
            pred_params.k_sigma
        ),
        "δ / cone_radius_base  [–]",
        false, // linear
        Some(1.0),
        &d.join("predictor_offset_ratio.png"),
    )?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlay plot: three panels (histogram | CDF | percentile)
// ─────────────────────────────────────────────────────────────────────────────

fn overlay_metric(
    tp: Vec<f64>,
    fp: Vec<f64>,
    title: &str,
    x_label: &str,
    log_x: bool,
    vline: Option<f64>,
    path: &Path,
) -> Result<()> {
    let (tp_plot, fp_plot, vline_plot) = if log_x {
        let t: Vec<f64> = tp.iter().filter(|v| **v > 0.0).map(|v| v.log10()).collect();
        let f: Vec<f64> = fp.iter().filter(|v| **v > 0.0).map(|v| v.log10()).collect();
        let vl = vline.map(|v| v.log10());
        (t, f, vl)
    } else {
        (tp, fp, vline)
    };

    let path_str = path.to_str().context("non-UTF-8 path")?;
    let root = BitMapBackend::new(path_str, (PLOT_W, PANEL_H * 3)).into_drawing_area();
    root.fill(&WHITE).context("fill background")?;

    let panels = root.split_evenly((3, 1));

    draw_overlay_histogram(
        &panels[0],
        title,
        x_label,
        &sort_finite(tp_plot.clone()),
        &sort_finite(fp_plot.clone()),
        50,
        vline_plot,
        log_x,
    )?;
    draw_overlay_cdf(
        &panels[1],
        x_label,
        &sort_finite(tp_plot.clone()),
        &sort_finite(fp_plot.clone()),
        vline_plot,
        log_x,
    )?;
    draw_overlay_percentiles(
        &panels[2],
        x_label,
        &sort_finite(tp_plot),
        &sort_finite(fp_plot),
        log_x,
    )?;

    root.present().context("present bitmap")?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorting helper
// ─────────────────────────────────────────────────────────────────────────────

fn sort_finite(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

fn combined_range(a: &[f64], b: &[f64]) -> (f64, f64) {
    let all_min = a
        .first()
        .copied()
        .unwrap_or(0.0)
        .min(b.first().copied().unwrap_or(0.0));
    let all_max = a
        .last()
        .copied()
        .unwrap_or(1.0)
        .max(b.last().copied().unwrap_or(1.0));
    if (all_max - all_min).abs() < f64::EPSILON {
        (all_min - 1.0, all_max + 1.0)
    } else {
        (all_min, all_max)
    }
}

fn bin_into(edges: &[f64], values: &[f64]) -> Vec<u32> {
    let mut counts = vec![0u32; edges.len().saturating_sub(1)];
    for &v in values {
        if let Some(i) = edges.windows(2).position(|w| v >= w[0] && v < w[1]) {
            counts[i] += 1;
        }
    }
    counts
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel implementations
// ─────────────────────────────────────────────────────────────────────────────

fn draw_overlay_histogram<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    tp: &[f64],
    fp: &[f64],
    n_bins: usize,
    vline: Option<f64>,
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    if tp.is_empty() && fp.is_empty() {
        return Ok(());
    }

    let (x_min, x_max) = combined_range(tp, fp);
    let range = (x_max - x_min).max(f64::EPSILON);
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| x_min + i as f64 * range / n_bins as f64)
        .collect();

    let counts_tp = bin_into(&edges, tp);
    let counts_fp = bin_into(&edges, fp);
    let raw_y_max = counts_tp
        .iter()
        .chain(counts_fp.iter())
        .copied()
        .max()
        .unwrap_or(1) as f64;
    let y_max = (raw_y_max + 1.0).log10() * 1.1;

    let margin: u32 = 30;
    let mut chart = ChartBuilder::on(area)
        .margin(margin)
        .caption(title, ("sans-serif", 13))
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, 0f64..y_max)
        .context("build histogram chart")?;

    chart
        .configure_mesh()
        .x_labels(8)
        .x_label_formatter(&|v| {
            if log_x {
                fmt_log10_tick(*v)
            } else {
                format!("{v:.2}")
            }
        })
        .x_desc(x_label)
        .y_desc("count")
        .y_label_formatter(&|v| {
            let count = (10f64.powf(*v) - 1.0).round() as i64;
            if count <= 0 {
                "0".to_string()
            } else {
                format!("{count}")
            }
        })
        .draw()
        .context("draw mesh")?;

    // FP bars (underneath)
    chart
        .draw_series(counts_fp.iter().enumerate().filter_map(|(i, &c)| {
            if c == 0 {
                return None;
            }
            let x0 = edges[i];
            let x1 = edges[i + 1];
            let h = (c as f64 + 1.0).log10();
            Some(Rectangle::new(
                [(x0, 0.0), (x1, h)],
                C_FP.mix(0.45).filled(),
            ))
        }))
        .context("draw FP bars")?
        .label("FP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_FP.filled()));

    // TP bars (on top)
    chart
        .draw_series(counts_tp.iter().enumerate().filter_map(|(i, &c)| {
            if c == 0 {
                return None;
            }
            let x0 = edges[i];
            let x1 = edges[i + 1];
            let h = (c as f64 + 1.0).log10();
            Some(Rectangle::new(
                [(x0, 0.0), (x1, h)],
                C_TP.mix(0.45).filled(),
            ))
        }))
        .context("draw TP bars")?
        .label("TP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_TP.filled()));

    if let Some(vl) = vline {
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(vl, 0.0), (vl, y_max)],
                C_CUTOFF.stroke_width(2),
            )))
            .context("draw vline histogram")?;
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .context("draw legend")?;
    Ok(())
}

fn draw_overlay_cdf<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    x_label: &str,
    tp: &[f64],
    fp: &[f64],
    vline: Option<f64>,
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    if tp.is_empty() && fp.is_empty() {
        return Ok(());
    }

    let (x_min, x_max) = combined_range(tp, fp);

    let margin: u32 = 30;
    let mut chart = ChartBuilder::on(area)
        .margin(margin)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, 0f64..1f64)
        .context("build CDF chart")?;

    chart
        .configure_mesh()
        .x_labels(8)
        .x_label_formatter(&|v| {
            if log_x {
                fmt_log10_tick(*v)
            } else {
                format!("{v:.2}")
            }
        })
        .x_desc(x_label)
        .y_desc("Cumulative fraction")
        .draw()
        .context("draw cdf mesh")?;

    let cdf_pts = |sorted: &[f64]| -> Vec<(f64, f64)> {
        let n = sorted.len();
        sorted
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, (i + 1) as f64 / n as f64))
            .collect()
    };

    if !fp.is_empty() {
        chart
            .draw_series(LineSeries::new(cdf_pts(fp), &C_FP))
            .context("draw FP CDF")?
            .label("FP")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], C_FP));
    }

    if !tp.is_empty() {
        chart
            .draw_series(LineSeries::new(cdf_pts(tp), &C_TP))
            .context("draw TP CDF")?
            .label("TP")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], C_TP));
    }

    if let Some(vl) = vline {
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(vl, 0.0), (vl, 1.0)],
                C_CUTOFF.stroke_width(2),
            )))
            .context("draw vline CDF")?;
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .context("draw cdf legend")?;
    Ok(())
}

fn draw_overlay_percentiles<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    x_label: &str,
    tp: &[f64],
    fp: &[f64],
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    if tp.is_empty() && fp.is_empty() {
        return Ok(());
    }

    let pvals_tp: Vec<(f64, f64)> = if tp.is_empty() {
        vec![]
    } else {
        PERCENTILE_PS
            .iter()
            .map(|&p| (p as f64, percentile_sorted(tp, p)))
            .collect()
    };
    let pvals_fp: Vec<(f64, f64)> = if fp.is_empty() {
        vec![]
    } else {
        PERCENTILE_PS
            .iter()
            .map(|&p| (p as f64, percentile_sorted(fp, p)))
            .collect()
    };

    let y_max = pvals_tp
        .iter()
        .chain(pvals_fp.iter())
        .map(|&(_, v)| v)
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;
    let y_max = if y_max.is_finite() && y_max > 0.0 {
        y_max
    } else {
        1.0
    };

    let y_min_data = pvals_tp
        .iter()
        .chain(pvals_fp.iter())
        .map(|&(_, v)| v)
        .fold(f64::INFINITY, f64::min);
    let y_min = if y_min_data.is_finite() {
        if y_min_data >= 0.0 {
            0f64.min(y_min_data * 0.9)
        } else {
            y_min_data * 1.1
        }
    } else {
        0.0
    };

    let margin: u32 = 30;
    let mut chart = ChartBuilder::on(area)
        .margin(margin)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..100f64, y_min..y_max)
        .context("build percentile chart")?;

    {
        chart
            .configure_mesh()
            .x_labels(10)
            .x_desc("Percentile")
            .y_desc(x_label)
            .y_label_formatter(&|v| {
                if log_x {
                    fmt_log10_tick(*v)
                } else {
                    format!("{v:.3}")
                }
            })
            .draw()
            .context("draw pct mesh")?;
    }

    if !fp.is_empty() {
        chart
            .draw_series(LineSeries::new(pvals_fp.clone(), &C_FP))
            .context("draw FP pct")?
            .label("FP")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], C_FP));
    }

    if !tp.is_empty() {
        chart
            .draw_series(LineSeries::new(pvals_tp.clone(), &C_TP))
            .context("draw TP pct")?
            .label("TP")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], C_TP));
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .context("draw pct legend")?;
    Ok(())
}

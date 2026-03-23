//! Collect and plot edge feature distributions split by TP / FP class.
//!
//! For each edge in the graph the module extracts the following quantities
//! and writes one three-panel `MetricPlot` (histogram / CDF / percentile) per
//! metric, with TP edges in green and FP edges in red:
//!
//! | File                        | Metric                                      |
//! |-----------------------------|---------------------------------------------|
//! `edge_cost.png`               | Solver-facing edge cost                     |
//! `edge_dt_days.png`            | Time gap between the two seeds (days)       |
//! `edge_chi2_pos.png`           | Position Mahalanobis χ² (log scale)         |
//! `edge_chi2_vel.png`           | Velocity Mahalanobis χ² (log scale)         |
//! `edge_cos_dtheta_v.png`       | cos Δθᵥ (direction alignment)               |
//! `edge_rel_speed_diff.png`     | Relative speed difference                   |
//! `edge_innov_speed_ratio.png`  | Innovation-speed ratio                      |
//! `edge_z_mag.png`             | Photometry mag z-score                     |

use std::path::Path;

use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::{graph::edge::edge_features::EdgeFeatures, pipeline::PipelineContext};

use crate::{
    seeding::plots::draw_helpers::{C_CUTOFF, PANEL_H, PLOT_W, fmt_log10_tick},
    truth_sso::{TruthClass, TruthSSO},
};

use plotters::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Colour overrides for TP / FP overlay
// ─────────────────────────────────────────────────────────────────────────────

const C_TP: RGBColor = RGBColor(34, 139, 34);
const C_FP: RGBColor = RGBColor(220, 20, 60);

// ─────────────────────────────────────────────────────────────────────────────
// Raw data container
// ─────────────────────────────────────────────────────────────────────────────

/// Per-edge numeric quantities collected during a single pass over the graph.
#[derive(Default)]
pub struct EdgeDistribData {
    // TP series
    pub cost_tp: Vec<f64>,
    pub dt_days_tp: Vec<f64>,
    pub chi2_pos_tp: Vec<f64>,
    pub chi2_vel_tp: Vec<f64>,
    pub cos_dtheta_tp: Vec<f64>,
    pub rel_speed_tp: Vec<f64>,
    pub innov_speed_tp: Vec<f64>,
    pub z_mag_tp: Vec<f64>,

    // FP series
    pub cost_fp: Vec<f64>,
    pub dt_days_fp: Vec<f64>,
    pub chi2_pos_fp: Vec<f64>,
    pub chi2_vel_fp: Vec<f64>,
    pub cos_dtheta_fp: Vec<f64>,
    pub rel_speed_fp: Vec<f64>,
    pub innov_speed_fp: Vec<f64>,
    pub z_mag_fp: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Collection
// ─────────────────────────────────────────────────────────────────────────────

/// Collect edge-level feature distributions for every edge in the graph.
///
/// Features are computed fresh (using `EdgeFeatures::compute_features`) so
/// that we do not need to store them during pipeline runtime.
/// Edges are labelled TP / FP by resolving their member alerts against `truth`.
pub fn collect_edge_distrib_data(
    ctx: &PipelineContext,
    truth: &TruthSSO,
) -> Result<EdgeDistribData> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;

    let mut data = EdgeDistribData::default();

    for edge in ctx.runtime_state.graph.edges.iter() {
        let from_seed = seed_store
            .try_get_seed(edge.from)
            .context("from seed not found")?;
        let to_seed = seed_store
            .try_get_seed(edge.to)
            .context("to seed not found")?;

        // Classify this edge.
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
        // Skip unknown edges (partial truth coverage).
        let is_tp = match class {
            TruthClass::TruePositive => true,
            TruthClass::FalsePositive => false,
            TruthClass::Unknown => continue,
        };

        // Compute features afresh.
        let features = EdgeFeatures::compute_features(from_seed, to_seed);

        macro_rules! push {
            ($field_tp:ident, $field_fp:ident, $val:expr) => {
                if is_tp {
                    data.$field_tp.push($val);
                } else {
                    data.$field_fp.push($val);
                }
            };
        }

        push!(cost_tp, cost_fp, edge.cost);
        push!(dt_days_tp, dt_days_fp, edge.dt_days);
        push!(chi2_pos_tp, chi2_pos_fp, features.position.chi2_pos);
        push!(chi2_vel_tp, chi2_vel_fp, features.velocity.chi2_vel);
        push!(cos_dtheta_tp, cos_dtheta_fp, features.velocity.cos_dtheta_v);
        push!(rel_speed_tp, rel_speed_fp, features.velocity.rel_speed_diff);
        push!(
            innov_speed_tp,
            innov_speed_fp,
            features.velocity.innov_speed_ratio
        );
        push!(z_mag_tp, z_mag_fp, features.photometry.z_mag);
    }

    Ok(data)
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
// Top-level plot entrypoint
// ─────────────────────────────────────────────────────────────────────────────

/// Write all edge distribution charts to `out_dir`.
pub fn plot_edge_distributions(data: EdgeDistribData, out_dir: &Utf8Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let d = out_dir.as_std_path();

    let EdgeDistribData {
        cost_tp,
        cost_fp,
        dt_days_tp,
        dt_days_fp,
        chi2_pos_tp,
        chi2_pos_fp,
        chi2_vel_tp,
        chi2_vel_fp,
        cos_dtheta_tp,
        cos_dtheta_fp,
        rel_speed_tp,
        rel_speed_fp,
        innov_speed_tp,
        innov_speed_fp,
        z_mag_tp,
        z_mag_fp,
    } = data;

    overlay_metric(
        sort_finite(cost_tp),
        sort_finite(cost_fp),
        "Edge cost (TP vs FP)",
        "cost",
        true,
        None,
        &d.join("edge_cost.png"),
    )?;

    overlay_metric(
        sort_finite(dt_days_tp),
        sort_finite(dt_days_fp),
        "Edge Δt (TP vs FP)",
        "Δt (days)",
        false,
        None,
        &d.join("edge_dt_days.png"),
    )?;

    overlay_metric(
        sort_finite(chi2_pos_tp),
        sort_finite(chi2_pos_fp),
        "Position χ² (TP vs FP)",
        "χ²_pos",
        true,
        None,
        &d.join("edge_chi2_pos.png"),
    )?;

    overlay_metric(
        sort_finite(chi2_vel_tp),
        sort_finite(chi2_vel_fp),
        "Velocity χ² (TP vs FP)",
        "χ²_vel",
        true,
        None,
        &d.join("edge_chi2_vel.png"),
    )?;

    overlay_metric(
        sort_finite(cos_dtheta_tp),
        sort_finite(cos_dtheta_fp),
        "cos Δθᵥ direction alignment (TP vs FP)",
        "cos Δθᵥ",
        false,
        None,
        &d.join("edge_cos_dtheta_v.png"),
    )?;

    overlay_metric(
        sort_finite(rel_speed_tp),
        sort_finite(rel_speed_fp),
        "Relative speed diff (TP vs FP)",
        "rel_speed_diff",
        false,
        None,
        &d.join("edge_rel_speed_diff.png"),
    )?;

    overlay_metric(
        sort_finite(innov_speed_tp),
        sort_finite(innov_speed_fp),
        "Innovation-speed ratio (TP vs FP)",
        "innov_speed_ratio",
        true,
        None,
        &d.join("edge_innov_speed_ratio.png"),
    )?;

    overlay_metric(
        sort_finite(z_mag_tp),
        sort_finite(z_mag_fp),
        "Mag z-score (TP vs FP)",
        "z_mag",
        false,
        None,
        &d.join("edge_z_mag.png"),
    )?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlay plot: two distributions (TP green / FP red) in four panels
// histogram | CDF | percentile
// ─────────────────────────────────────────────────────────────────────────────

/// Write a four-panel PNG for two distributions (TP / FP) side by side:
/// - panel 0: histogram overlay (TP + FP)
/// - panel 1: CDF overlay
/// - panel 2: percentile curves
/// - panel 3: CDF overlay zoomed to [0, 1] fraction (useful as cut guide)
fn overlay_metric(
    tp: Vec<f64>,
    fp: Vec<f64>,
    title: &str,
    x_label: &str,
    log_x: bool,
    vline: Option<f64>,
    path: &Path,
) -> Result<()> {
    // Transform to log₁₀ space when requested.
    let (tp_plot, fp_plot, vline_plot) = if log_x {
        let t: Vec<f64> = tp
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.log10())
            .collect();
        let f: Vec<f64> = fp
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.log10())
            .collect();
        let vl = vline.filter(|&v| v > 0.0).map(|v| v.log10());
        (t, f, vl)
    } else {
        (tp.clone(), fp.clone(), vline)
    };

    let path_str = path.to_str().context("non-UTF-8 path")?;
    // 3 rows: histogram, CDF, percentile
    let root = BitMapBackend::new(path_str, (PLOT_W, PANEL_H * 3)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("canvas fill: {e:?}"))?;

    {
        let panels = root.split_evenly((3, 1));

        draw_overlay_histogram(
            &panels[0],
            (title, x_label),
            &tp_plot,
            &fp_plot,
            60,
            vline_plot,
            log_x,
        )
        .context("histogram panel")?;
        draw_overlay_cdf(&panels[1], x_label, &tp_plot, &fp_plot, vline_plot, log_x)
            .context("CDF panel")?;
        draw_overlay_percentiles(&panels[2], x_label, &tp_plot, &fp_plot, log_x)
            .context("percentile panel")?;
    }

    root.present()
        .map_err(|e| anyhow::anyhow!("write PNG: {e:?}"))?;

    tracing::debug!("wrote {}", path.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel implementations
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

fn draw_overlay_histogram<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    labels: (&str, &str),
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
    let (title, x_label) = labels;
    if tp.is_empty() && fp.is_empty() {
        return Ok(());
    }

    let (x_min, x_max) = combined_range(tp, fp);
    let range = (x_max - x_min).max(f64::EPSILON);
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| x_min + (i as f64 / n_bins as f64) * range)
        .collect();

    let counts_tp = bin_into(&edges, tp);
    let counts_fp = bin_into(&edges, fp);
    let raw_y_max = counts_tp
        .iter()
        .chain(counts_fp.iter())
        .copied()
        .max()
        .unwrap_or(1) as f64;
    let y_max = (raw_y_max + 1.0).log10() * 1.15;

    let margin: u32 = 30;
    let mut chart = ChartBuilder::on(area)
        .caption(format!("{title} – Histogram"), ("sans-serif", 18))
        .margin(margin)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    {
        let mut mesh = chart.configure_mesh();
        mesh.x_desc(x_label)
            .y_desc("count")
            .y_label_formatter(&|v| {
                let count = (10f64.powf(*v) - 1.0).round() as i64;
                if count <= 0 {
                    "0".to_string()
                } else {
                    format!("{count}")
                }
            });
        if log_x {
            mesh.x_label_formatter(&|x| fmt_log10_tick(*x))
                .draw()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        } else {
            mesh.draw().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    // FP bars (plotted first so TP overlaps)
    chart
        .draw_series((0..n_bins).filter_map(|i| {
            if i + 1 < edges.len() && counts_fp[i] > 0 {
                let h = (counts_fp[i] as f64 + 1.0).log10();
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], h)],
                    C_FP.mix(0.55).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("FP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_FP.mix(0.55).filled()));

    // TP bars
    chart
        .draw_series((0..n_bins).filter_map(|i| {
            if i + 1 < edges.len() && counts_tp[i] > 0 {
                let h = (counts_tp[i] as f64 + 1.0).log10();
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], h)],
                    C_TP.mix(0.65).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("TP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_TP.mix(0.65).filled()));

    if let Some(vl) = vline {
        let label = if log_x {
            format!("cut {}", fmt_log10_tick(vl))
        } else {
            format!("cut {vl:.3}")
        };
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(vl, 0f64), (vl, y_max)],
                ShapeStyle::from(&C_CUTOFF).stroke_width(2),
            )))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label(label)
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_CUTOFF).stroke_width(2),
                )
            });
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

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
        .caption("CDF", ("sans-serif", 18))
        .margin(margin)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f64..1.05f64)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    {
        let mut mesh = chart.configure_mesh();
        mesh.x_desc(x_label).y_desc("cumulative fraction");
        if log_x {
            mesh.x_label_formatter(&|x| fmt_log10_tick(*x))
                .draw()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        } else {
            mesh.draw().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    let cdf_pts = |sorted: &[f64]| -> Vec<(f64, f64)> {
        let n = sorted.len();
        sorted
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, (i + 1) as f64 / n as f64))
            .collect()
    };

    // FP CDF
    if !fp.is_empty() {
        let pts = cdf_pts(fp);
        chart
            .draw_series(LineSeries::new(
                pts,
                ShapeStyle::from(&C_FP).stroke_width(2),
            ))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label("FP")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_FP).stroke_width(2),
                )
            });
    }

    // TP CDF
    if !tp.is_empty() {
        let pts = cdf_pts(tp);
        chart
            .draw_series(LineSeries::new(
                pts,
                ShapeStyle::from(&C_TP).stroke_width(2),
            ))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label("TP")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_TP).stroke_width(2),
                )
            });
    }

    if let Some(vl) = vline {
        let tp_frac = tp.partition_point(|&x| x <= vl) as f64 / tp.len().max(1) as f64;
        let label = if log_x {
            format!("cut {} TP→{:.1}%", fmt_log10_tick(vl), tp_frac * 100.0)
        } else {
            format!("cut {vl:.3} TP→{:.1}%", tp_frac * 100.0)
        };
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(vl, 0f64), (vl, 1.0)],
                ShapeStyle::from(&C_CUTOFF).stroke_width(2),
            )))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label(label)
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_CUTOFF).stroke_width(2),
                )
            });
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

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
    use crate::seeding::plots::chart_utils::{PERCENTILE_PS, percentile_sorted};

    if tp.is_empty() && fp.is_empty() {
        return Ok(());
    }

    let pvals_tp: Vec<(f64, f64)> = PERCENTILE_PS
        .iter()
        .map(|&p| (p, percentile_sorted(tp, p / 100.0)))
        .collect();
    let pvals_fp: Vec<(f64, f64)> = PERCENTILE_PS
        .iter()
        .map(|&p| (p, percentile_sorted(fp, p / 100.0)))
        .collect();

    let y_max = pvals_tp
        .iter()
        .chain(pvals_fp.iter())
        .filter_map(|&(_, v)| if v.is_finite() { Some(v) } else { None })
        .fold(f64::NEG_INFINITY, f64::max)
        .max(f64::EPSILON)
        * 1.1;

    let y_min_data = pvals_tp
        .iter()
        .chain(pvals_fp.iter())
        .filter_map(|&(_, v)| if v.is_finite() { Some(v) } else { None })
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
        .caption("Percentiles", ("sans-serif", 18))
        .margin(margin)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..100f64, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    {
        let mut mesh = chart.configure_mesh();
        mesh.x_desc("percentile").y_desc(x_label);
        if log_x {
            mesh.y_label_formatter(&|y| fmt_log10_tick(*y))
                .draw()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        } else {
            mesh.draw().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    // FP series
    if !fp.is_empty() {
        chart
            .draw_series(LineSeries::new(
                pvals_fp.clone(),
                ShapeStyle::from(&C_FP).stroke_width(2),
            ))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label("FP")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_FP).stroke_width(2),
                )
            });
        chart
            .draw_series(
                pvals_fp
                    .iter()
                    .map(|&(p, v)| Circle::new((p, v), 4, C_FP.filled())),
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    // TP series
    if !tp.is_empty() {
        chart
            .draw_series(LineSeries::new(
                pvals_tp.clone(),
                ShapeStyle::from(&C_TP).stroke_width(2),
            ))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label("TP")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_TP).stroke_width(2),
                )
            });
        chart
            .draw_series(
                pvals_tp
                    .iter()
                    .map(|&(p, v)| Circle::new((p, v), 4, C_TP.filled())),
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Binning helper (shared x-axis edges for overlay)
// ─────────────────────────────────────────────────────────────────────────────

/// Bin `values` (sorted) into pre-computed `edges`.
fn bin_into(edges: &[f64], values: &[f64]) -> Vec<u32> {
    let n_bins = edges.len().saturating_sub(1);
    let mut counts = vec![0u32; n_bins];
    if n_bins == 0 || values.is_empty() {
        return counts;
    }
    let mut bin = 0usize;
    for &v in values {
        while bin + 1 < n_bins && v >= edges[bin + 1] {
            bin += 1;
        }
        counts[bin] += 1;
    }
    counts
}

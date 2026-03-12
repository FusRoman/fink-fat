//! Generic drawing helpers: [`MetricPlot`] saves a three-panel chart
//! (histogram, CDF, percentile curve) for a single numeric metric.
//!
//! Panel functions are generic over [`DrawingBackend`] so the same logic can be
//! exercised in tests with an in-memory backend.

use std::path::Path;

use anyhow::{Context, Result};
use plotters::prelude::*;

use super::chart_utils::{PERCENTILE_PS, histogram_bins, percentile_sorted};

// ─────────────────────────────────────────────────────────────────────────────
// Log-scale tick formatter
// ─────────────────────────────────────────────────────────────────────────────

/// Format a log₁₀-space coordinate back to human-readable original scale.
///
/// Input `log_val` is `log10(original_value)`.
/// The output adapts precision based on magnitude:
/// - ≥ 100 → no decimal places
/// - ≥  10 → one decimal place
/// - ≥   1 → two decimal places
/// - < 1   → three significant decimal places
pub fn fmt_log10_tick(log_val: f64) -> String {
    let v = 10f64.powf(log_val);
    if v >= 100.0 {
        format!("{:.0}", v)
    } else if v >= 10.0 {
        format!("{:.1}", v)
    } else if v >= 1.0 {
        format!("{:.2}", v)
    } else {
        format!("{:.3}", v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour palette
// ─────────────────────────────────────────────────────────────────────────────

pub const C_HIST: RGBColor = RGBColor(70, 130, 180);
pub const C_CDF: RGBColor = RGBColor(34, 139, 34);
pub const C_CUTOFF: RGBColor = RGBColor(220, 20, 60);
pub const C_PCTILE: RGBColor = RGBColor(255, 165, 0);

// ─────────────────────────────────────────────────────────────────────────────
// Canvas constants
// ─────────────────────────────────────────────────────────────────────────────

pub const PLOT_W: u32 = 900;
pub const PANEL_H: u32 = 320;
const MARGIN: u32 = 30;

// ─────────────────────────────────────────────────────────────────────────────
// MetricPlot – builder + save
// ─────────────────────────────────────────────────────────────────────────────

/// A numeric dataset + display options for a single metric.
///
/// Call [`MetricPlot::save_to`] to write a three-panel PNG:
/// histogram → CDF → percentile curve (top to bottom).
pub struct MetricPlot {
    pub title: String,
    pub x_label: String,
    /// Pre-sorted values (ascending).  The caller is responsible for sorting.
    pub sorted: Vec<f64>,
    pub n_bins: usize,
    /// Optional vertical reference line drawn in all three panels (e.g. config cutoff).
    pub vline: Option<f64>,
    /// When `true`, x-axis values are transformed to log₁₀ internally and tick
    /// labels are shown in the original scale.  Values ≤ 0 are silently dropped.
    /// Useful for heavily right-skewed distributions (angular speed, separations…).
    pub log_x: bool,
}

impl MetricPlot {
    /// Create a new plot from a **sorted** value vector.
    pub fn new(title: impl Into<String>, x_label: impl Into<String>, sorted: Vec<f64>) -> Self {
        Self {
            title: title.into(),
            x_label: x_label.into(),
            sorted,
            n_bins: 50,
            vline: None,
            log_x: false,
        }
    }

    /// Enable log₁₀ scaling on the x-axis.
    ///
    /// When active, values ≤ 0 are filtered out before plotting and all
    /// tick labels are shown in the original (non-log) scale.
    pub fn with_log_x(mut self) -> Self {
        self.log_x = true;
        self
    }

    /// Set an optional threshold/cutoff reference line.
    pub fn with_vline(mut self, vl: f64) -> Self {
        self.vline = Some(vl);
        self
    }

    /// Override the number of histogram bins (default: 50).
    pub fn with_n_bins(mut self, n: usize) -> Self {
        self.n_bins = n;
        self
    }

    /// Write a 3-panel (histogram | CDF | percentiles) PNG to `path`.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let path_str = path.to_str().context("non-UTF-8 path")?;
        let root = BitMapBackend::new(path_str, (PLOT_W, PANEL_H * 3)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("canvas fill: {e:?}"))?;

        // When log_x is requested, transform data to log₁₀ space (drop ≤ 0)
        // and transform the vline coordinate accordingly.
        let (plot_sorted, plot_vline): (Vec<f64>, Option<f64>) = if self.log_x {
            let vs: Vec<f64> = self
                .sorted
                .iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| x.log10())
                .collect();
            let vl = self.vline.filter(|&v| v > 0.0).map(|v| v.log10());
            (vs, vl)
        } else {
            (self.sorted.clone(), self.vline)
        };

        {
            let panels = root.split_evenly((3, 1));
            draw_histogram_on(
                &panels[0],
                &self.title,
                &self.x_label,
                &plot_sorted,
                self.n_bins,
                plot_vline,
                self.log_x,
            )
            .context("histogram panel")?;
            draw_cdf_on(
                &panels[1],
                &self.x_label,
                &plot_sorted,
                plot_vline,
                self.log_x,
            )
            .context("CDF panel")?;
            draw_percentiles_on(&panels[2], &self.x_label, &plot_sorted, self.log_x)
                .context("percentile panel")?;
        } // panels dropped here – releases shared Rc refs before present()

        root.present()
            .map_err(|e| anyhow::anyhow!("write PNG: {e:?}"))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel helpers (generic over DrawingBackend)
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a filled histogram into `area`.
///
/// When `log_x` is `true`, data is already in log₁₀ space; tick labels are
/// formatted back to the original scale via [`fmt_log10_tick`], and the
/// cutoff label shows the original value too.
pub fn draw_histogram_on<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    sorted: &[f64],
    n_bins: usize,
    vline: Option<f64>,
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    if sorted.is_empty() {
        return Ok(());
    }
    let (edges, counts) = histogram_bins(sorted, n_bins);
    let x_min = edges[0];
    let x_max = edges[edges.len() - 1];
    let y_max = *counts.iter().max().unwrap_or(&1) as f64 * 1.15;

    let mut chart = ChartBuilder::on(area)
        .caption(format!("{title} - Histogram"), ("sans-serif", 18))
        .margin(MARGIN)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    {
        let mut mesh = chart.configure_mesh();
        mesh.x_desc(x_label).y_desc("count");
        if log_x {
            mesh.x_label_formatter(&|x| fmt_log10_tick(*x))
                .draw()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        } else {
            mesh.draw().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    chart
        .draw_series((0..n_bins).filter_map(|i| {
            if i + 1 < edges.len() {
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], counts[i] as f64)],
                    C_HIST.mix(0.75).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    if let Some(vl) = vline {
        let label = if log_x {
            format!("cutoff = {}", fmt_log10_tick(vl))
        } else {
            format!("cutoff = {vl:.3}")
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
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .draw()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }
    Ok(())
}

/// Draw a CDF (empirical cumulative distribution) into `area`.
///
/// When `log_x` is `true`, data is already in log₁₀ space and tick labels
/// are shown in original scale.  The cutoff legend entry also shows the
/// original (non-log) value.
pub fn draw_cdf_on<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    x_label: &str,
    sorted: &[f64],
    vline: Option<f64>,
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    if sorted.is_empty() {
        return Ok(());
    }
    let n = sorted.len();
    let x_min = sorted[0];
    let x_max = sorted[n - 1];
    let x_range = if (x_max - x_min).abs() < f64::EPSILON {
        (x_min - 1.0)..(x_max + 1.0)
    } else {
        x_min..x_max
    };

    let mut chart = ChartBuilder::on(area)
        .caption("CDF", ("sans-serif", 18))
        .margin(MARGIN)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range, 0f64..1.05f64)
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

    let pts: Vec<(f64, f64)> = sorted
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, (i + 1) as f64 / n as f64))
        .collect();

    chart
        .draw_series(LineSeries::new(
            pts,
            ShapeStyle::from(&C_CDF).stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    if let Some(vl) = vline {
        let cdf_at_vl = sorted.partition_point(|&x| x <= vl) as f64 / n as f64;
        let vl_label = if log_x {
            format!("cutoff {} -> {:.1}%", fmt_log10_tick(vl), cdf_at_vl * 100.0)
        } else {
            format!("cutoff {vl:.3} -> {:.1}%", cdf_at_vl * 100.0)
        };
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(vl, 0f64), (vl, cdf_at_vl)],
                ShapeStyle::from(&C_CUTOFF).stroke_width(2),
            )))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label(vl_label)
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&C_CUTOFF).stroke_width(2),
                )
            });
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .draw()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }
    Ok(())
}

/// Draw a percentile curve (percentile on x, value on y) into `area`.
///
/// When `log_x` is `true`, data is already in log₁₀ space.  The y-axis
/// (metric values) tick labels are shown in original scale.
pub fn draw_percentiles_on<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    x_label: &str,
    sorted: &[f64],
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    if sorted.is_empty() {
        return Ok(());
    }
    let pvals: Vec<(f64, f64)> = PERCENTILE_PS
        .iter()
        .map(|&p| (p, percentile_sorted(sorted, p / 100.0)))
        .collect();

    let y_max = pvals
        .last()
        .map(|&(_, v)| if v.is_finite() { v * 1.1 } else { 1.0 })
        .unwrap_or(1.0)
        .max(f64::EPSILON);

    // y_min must be derived from data, not hardcoded to 0: when log_x is true
    // the y values are in log₁₀ space and can be negative (original value < 1).
    // Mapping a point below the chart's y range causes an integer overflow in
    // plotters' coordinate translation.
    let y_min_data = pvals
        .iter()
        .filter_map(|&(_, v)| if v.is_finite() { Some(v) } else { None })
        .fold(f64::INFINITY, f64::min);
    let y_min = if y_min_data.is_finite() {
        // Add a small margin below the lowest value.
        if y_min_data >= 0.0 {
            0f64.min(y_min_data * 0.9)
        } else {
            y_min_data * 1.1
        }
    } else {
        0.0
    };

    let mut chart = ChartBuilder::on(area)
        .caption("Percentiles", ("sans-serif", 18))
        .margin(MARGIN)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..100f64, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    {
        let mut mesh = chart.configure_mesh();
        mesh.x_desc("percentile").y_desc(x_label);
        if log_x {
            // y-axis encodes log10 values; show original scale on ticks
            mesh.y_label_formatter(&|y| fmt_log10_tick(*y))
                .draw()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        } else {
            mesh.draw().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    chart
        .draw_series(LineSeries::new(
            pvals.clone(),
            ShapeStyle::from(&C_PCTILE).stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .draw_series(
            pvals
                .iter()
                .map(|&(p, v)| Circle::new((p, v), 4, C_PCTILE.filled())),
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    Ok(())
}

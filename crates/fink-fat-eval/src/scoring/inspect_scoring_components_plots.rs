//! Plotting utilities for the `inspect_scoring_components` binary.
//!
//! This module contains all of the helper functions used to build
//! histograms and overlay plots. By isolating the plotting code
//! here we decouple rendering concerns from the main pipeline logic
//! found in `inspect_scoring_components_pipeline`. The functions in
//! this module operate on simple slices of numeric data and produce
//! PNG files using the `plotters` crate.

use anyhow::Result;
use camino::Utf8PathBuf;
use plotters::prelude::*;

/// Compute two histograms with the same binning.  Returns the per-bin
/// counts for each histogram along with the maximum count seen in either
/// histogram.  Values falling outside `[xmin, xmax]` are ignored.
pub fn hist2(a: &[f64], b: &[f64], xmin: f64, xmax: f64, bins: usize) -> (Vec<u32>, Vec<u32>, u32) {
    let mut ha = vec![0u32; bins];
    let mut hb = vec![0u32; bins];
    let mut y_max = 0u32;
    for &x in a.iter() {
        if let Some(k) = bin_index(x, xmin, xmax, bins) {
            ha[k] += 1;
            y_max = y_max.max(ha[k]).max(hb[k]);
        }
    }
    for &x in b.iter() {
        if let Some(k) = bin_index(x, xmin, xmax, bins) {
            hb[k] += 1;
            y_max = y_max.max(ha[k]).max(hb[k]);
        }
    }
    (ha, hb, y_max)
}

/// Compute a single histogram.  Returns the per-bin counts and the
/// maximum count.  Values falling outside `[xmin, xmax]` are ignored.
pub fn hist1(a: &[f64], xmin: f64, xmax: f64, bins: usize) -> (Vec<u32>, u32) {
    let mut ha = vec![0u32; bins];
    let mut y_max = 0u32;
    for &x in a.iter() {
        if let Some(k) = bin_index(x, xmin, xmax, bins) {
            ha[k] += 1;
            y_max = y_max.max(ha[k]);
        }
    }
    (ha, y_max)
}

/// Compute the bin index for a value given the histogram bounds and
/// number of bins.  Returns `None` if the value is not finite or
/// outside the `[xmin, xmax]` range.
pub fn bin_index(x: f64, xmin: f64, xmax: f64, bins: usize) -> Option<usize> {
    if !x.is_finite() || xmax <= xmin {
        return None;
    }
    if x < xmin || x > xmax {
        return None;
    }
    let t = (x - xmin) / (xmax - xmin);
    let mut k = (t * bins as f64) as isize;
    if k == bins as isize {
        k = bins as isize - 1;
    }
    if k < 0 || k >= bins as isize {
        return None;
    }
    Some(k as usize)
}

/// Determine a reasonable plotting range for a dataset.  The minimum is
/// clamped to zero if all values are positive; the maximum can be
/// optionally clipped.  A small epsilon padding is added to avoid
/// collapsing to a single-point range.
pub fn data_range(data: &[f64], clip_xmax: Option<f64>) -> (f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    for &x in data.iter() {
        if x.is_finite() {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
        }
    }
    if !xmin.is_finite() || !xmax.is_finite() || xmin == xmax {
        xmin = 0.0;
        xmax = 1.0;
    }
    if xmin > 0.0 {
        xmin = 0.0;
    }
    if let Some(c) = clip_xmax {
        if c.is_finite() && c > xmin {
            xmax = xmax.min(c);
        }
    }
    let eps = (xmax - xmin) * 1e-6;
    (xmin - eps, xmax + eps)
}

/// Fraction of `None` values given the count of missing and present values.
/// Returns 0.0 if the denominator is zero.
pub fn frac_none(n_none: usize, n_some: usize) -> f64 {
    let denom = (n_none + n_some) as f64;
    if denom <= 0.0 {
        0.0
    } else {
        n_none as f64 / denom
    }
}

/// Draw an overlaid histogram of two classes ("same" and "different").  The
/// output PNG is written to `out`.  The y-axis can be optionally log-scaled.
pub fn plot_overlay_hist(
    out: &Utf8PathBuf,
    title: impl Into<String>,
    x_label: impl Into<String>,
    same: &[f64],
    diff: &[f64],
    bins: usize,
    clip_xmax: Option<f64>,
    log_y: bool,
) -> Result<()> {
    let title = title.into();
    let x_label = x_label.into();
    // Merge data to determine a common range.
    let mut all = Vec::with_capacity(same.len() + diff.len());
    all.extend_from_slice(same);
    all.extend_from_slice(diff);
    if all.is_empty() {
        eprintln!("skip empty plot {}", out);
        return Ok(());
    }
    let (xmin, xmax) = data_range(&all, clip_xmax);
    let (h_same, h_diff, y_max) = hist2(same, diff, xmin, xmax, bins);
    let root = BitMapBackend::new(out.as_str(), (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_range = if log_y {
        0f64..((y_max as f64 + 1.0).ln().max(1.0))
    } else {
        0f64..(y_max as f64 * 1.05 + 1.0)
    };
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("sans-serif", 35))
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(xmin..xmax, y_range)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(if log_y { "ln(count+1)" } else { "count" })
        .axis_desc_style(("sans-serif", 22))
        .label_style(("sans-serif", 18))
        .draw()?;
    let bin_w = (xmax - xmin) / bins as f64;
    // Draw background (different class) bars first.
    chart
        .draw_series((0..bins).map(|k| {
            let x0 = xmin + k as f64 * bin_w;
            let x1 = x0 + bin_w;
            let y = if log_y {
                (h_diff[k] as f64 + 1.0).ln()
            } else {
                h_diff[k] as f64
            };
            Rectangle::new(
                [(x0, 0.0), (x1, y)],
                RGBColor(255, 120, 120).mix(0.45).filled(),
            )
        }))?
        .label("different asteroid")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 8), (x + 18, y + 8)],
                RGBColor(255, 120, 120).mix(0.45).filled(),
            )
        });
    chart
        .draw_series((0..bins).map(|k| {
            let x0 = xmin + k as f64 * bin_w;
            let x1 = x0 + bin_w;
            let y = if log_y {
                (h_same[k] as f64 + 1.0).ln()
            } else {
                h_same[k] as f64
            };
            Rectangle::new(
                [(x0, 0.0), (x1, y)],
                RGBColor(120, 120, 255).mix(0.45).filled(),
            )
        }))?
        .label("same asteroid")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 8), (x + 18, y + 8)],
                RGBColor(120, 120, 255).mix(0.45).filled(),
            )
        });
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .label_font(("sans-serif", 20))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    root.present()?;
    Ok(())
}

/// Draw a single-class histogram.  The y-axis can be log-scaled, and
/// optional clipping of the x-range is supported.  If `data` is empty the
/// plot is skipped with a message on stderr.
pub fn plot_hist(
    out: &Utf8PathBuf,
    title: impl Into<String>,
    x_label: impl Into<String>,
    data: &[f64],
    bins: usize,
    clip_xmax: Option<f64>,
    log_y: bool,
) -> Result<()> {
    let title = title.into();
    let x_label = x_label.into();
    if data.is_empty() {
        eprintln!("skip empty plot {}", out);
        return Ok(());
    }
    let (xmin, xmax) = data_range(data, clip_xmax);
    let (h, y_max) = hist1(data, xmin, xmax, bins);
    let root = BitMapBackend::new(out.as_str(), (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_range = if log_y {
        0f64..((y_max as f64 + 1.0).ln().max(1.0))
    } else {
        0f64..(y_max as f64 * 1.05 + 1.0)
    };
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("sans-serif", 35))
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(xmin..xmax, y_range)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(if log_y { "ln(count+1)" } else { "count" })
        .axis_desc_style(("sans-serif", 22))
        .label_style(("sans-serif", 18))
        .draw()?;
    let bin_w = (xmax - xmin) / bins as f64;
    chart.draw_series((0..bins).map(|k| {
        let x0 = xmin + k as f64 * bin_w;
        let x1 = x0 + bin_w;
        let y = if log_y {
            (h[k] as f64 + 1.0).ln()
        } else {
            h[k] as f64
        };
        Rectangle::new([(x0, 0.0), (x1, y)], BLACK.mix(0.25).filled())
    }))?;
    root.present()?;
    Ok(())
}

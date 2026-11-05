//! `plotters` charts summarizing a [`SeedingReport`] across a whole dataset.

use anyhow::{Context, Result};
use camino::Utf8Path;
use plotters::prelude::*;

use crate::seed_bank_report::report::SeedingReport;

pub(crate) const CHART_WIDTH: u32 = 1200;
pub(crate) const CHART_HEIGHT: u32 = 700;
pub(crate) const MARGIN: u32 = 20;
pub(crate) const LABEL_AREA: u32 = 60;

/// One named, colored series of `(x, y)` points for [`draw_line_chart`].
///
/// `pub(crate)` so other report modules (e.g. `tracking_report::plots`) can
/// reuse this generic chart primitive instead of duplicating the `plotters`
/// boilerplate.
pub(crate) struct Series {
    pub(crate) label: &'static str,
    pub(crate) color: RGBColor,
    pub(crate) points: Vec<(f64, f64)>,
}

/// Draw one or more line+point series sharing the same axes to `output_path`,
/// optionally overlaid with horizontal reference lines (e.g. χ² confidence
/// bounds for a NIS/NEES chart, reproducing the style at
/// <https://kalman-filter.com/normalized-estimation-error-squared/>).
///
/// The x/y ranges are derived from the data (padded by 5%) rather than
/// hardcoded, so this works unchanged whether it's plotting a branch count
/// in the thousands or a percentage in `[0, 100]`. `hlines` values are folded
/// into the y-range so a bound outside the data's own span is never clipped.
pub(crate) fn draw_line_chart(
    output_path: &Utf8Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[Series],
    hlines: &[(f64, RGBColor, &str)],
) -> Result<()> {
    let root =
        BitMapBackend::new(output_path.as_str(), (CHART_WIDTH, CHART_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let all_points: Vec<(f64, f64)> = series
        .iter()
        .flat_map(|s| s.points.iter().copied())
        .collect();
    let (x_range, mut y_range) = padded_ranges(all_points.iter());
    let hline_min = hlines
        .iter()
        .map(|(y, _, _)| *y)
        .fold(f64::INFINITY, f64::min);
    let hline_max = hlines
        .iter()
        .map(|(y, _, _)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    y_range = y_range.start.min(hline_min)..y_range.end.max(hline_max);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28))
        .margin(MARGIN)
        .x_label_area_size(LABEL_AREA)
        .y_label_area_size(LABEL_AREA)
        .build_cartesian_2d(x_range.clone(), y_range)
        .with_context(|| format!("failed to build chart area for {output_path}"))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    for s in series {
        chart
            .draw_series(LineSeries::new(s.points.iter().copied(), s.color))?
            .label(s.label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], s.color));
        chart.draw_series(
            s.points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 2, s.color.filled())),
        )?;
    }

    for &(y, color, label) in hlines {
        chart
            .draw_series(std::iter::once(PathElement::new(
                [(x_range.start, y), (x_range.end, y)],
                color.stroke_width(2),
            )))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }

    if series.len() > 1 || !hlines.is_empty() {
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()
        .with_context(|| format!("failed to write chart to {output_path}"))?;
    Ok(())
}

/// Data-derived axis ranges, padded by 5% on each side so points don't sit
/// flush against the chart border. Falls back to `[0, 1]` for an empty
/// series (nothing to plot, but the chart must still build).
pub(crate) fn padded_ranges<'a>(
    points: impl Iterator<Item = &'a (f64, f64)>,
) -> (std::ops::Range<f64>, std::ops::Range<f64>) {
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y) in points {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    if !x_min.is_finite() {
        return (0.0..1.0, 0.0..1.0);
    }
    let pad = |lo: f64, hi: f64| {
        let span = (hi - lo).max(f64::EPSILON);
        (lo - 0.05 * span)..(hi + 0.05 * span)
    };
    (pad(x_min, x_max), pad(y_min, y_max))
}

/// Number of Kalman banks (seed branches) produced per night, across the
/// dataset — highlights unusually dense nights.
pub fn plot_branches_per_night(report: &SeedingReport, output_path: &Utf8Path) -> Result<()> {
    let points = report
        .per_night
        .iter()
        .map(|n| (n.night_id.0 as f64, n.n_branches as f64))
        .collect();

    draw_line_chart(
        output_path,
        "Seed branches (Kalman banks) produced per night",
        "Night id",
        "Branches",
        &[Series {
            label: "Branches",
            color: BLUE,
            points,
        }],
        &[],
    )
}

/// Recall (pure-seed coverage) and purity, both as percentages, per night.
pub fn plot_recall_purity_per_night(report: &SeedingReport, output_path: &Utf8Path) -> Result<()> {
    let recall_points = report
        .per_night
        .iter()
        .map(|n| (n.night_id.0 as f64, n.recall_pct()))
        .filter(|(_, y)| y.is_finite())
        .collect();
    let purity_points = report
        .per_night
        .iter()
        .map(|n| (n.night_id.0 as f64, n.purity_pct()))
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "Seeding recall and purity per night",
        "Night id",
        "Percent",
        &[
            Series {
                label: "Recall (pure seed found)",
                color: BLUE,
                points: recall_points,
            },
            Series {
                label: "Purity",
                color: RED,
                points: purity_points,
            },
        ],
        &[],
    )
}

/// Histogram of the number of live hypotheses per Kalman bank, aggregated
/// over every bank produced across the whole dataset.
pub fn plot_hypotheses_histogram(report: &SeedingReport, output_path: &Utf8Path) -> Result<()> {
    const N_BINS: usize = 40;

    let mut counts = report.all_hypothesis_counts();
    counts.sort_by(f64::total_cmp);
    let (edges, bin_counts) = histogram_bins(&counts, N_BINS);

    let root =
        BitMapBackend::new(output_path.as_str(), (CHART_WIDTH, CHART_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = edges.last().copied().unwrap_or(1.0).max(1.0);
    let y_max = bin_counts.iter().copied().max().unwrap_or(1).max(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Hypotheses per Kalman bank (all nights)",
            ("sans-serif", 28),
        )
        .margin(MARGIN)
        .x_label_area_size(LABEL_AREA)
        .y_label_area_size(LABEL_AREA)
        .build_cartesian_2d(0.0..x_max, 0u32..(y_max + y_max / 10 + 1))
        .with_context(|| format!("failed to build chart area for {output_path}"))?;

    chart
        .configure_mesh()
        .x_desc("Hypotheses in bank")
        .y_desc("Number of banks")
        .draw()?;

    chart.draw_series(bin_counts.iter().enumerate().map(|(i, &count)| {
        Rectangle::new([(edges[i], 0), (edges[i + 1], count)], BLUE.filled())
    }))?;

    root.present()
        .with_context(|| format!("failed to write chart to {output_path}"))?;
    Ok(())
}

/// Compute equispaced histogram bins from a **pre-sorted** slice. Returns
/// `(edges, counts)` where `edges` has `n_bins + 1` entries and `counts`
/// has `n_bins` entries; values fall into `[edges[i], edges[i+1])`, with
/// the last bin closed on both ends.
pub(crate) fn histogram_bins(sorted: &[f64], n_bins: usize) -> (Vec<f64>, Vec<u32>) {
    let n_bins = n_bins.max(1);
    if sorted.is_empty() {
        return (vec![0.0, 1.0], vec![0]);
    }
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    let range = (hi - lo).max(f64::EPSILON);

    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| lo + (i as f64 / n_bins as f64) * range)
        .collect();

    let mut counts = vec![0u32; n_bins];
    let mut bin = 0usize;
    for &v in sorted {
        while bin + 1 < n_bins && v >= edges[bin + 1] {
            bin += 1;
        }
        counts[bin] += 1;
    }
    (edges, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_bins_cover_every_value() {
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let (edges, counts) = histogram_bins(&data, 2);
        assert_eq!(edges.len(), 3);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.iter().sum::<u32>(), data.len() as u32);
    }

    #[test]
    fn histogram_bins_handles_empty_input() {
        let (edges, counts) = histogram_bins(&[], 10);
        assert_eq!(edges, vec![0.0, 1.0]);
        assert_eq!(counts, vec![0]);
    }

    #[test]
    fn padded_ranges_falls_back_for_empty_input() {
        let points: Vec<(f64, f64)> = vec![];
        let (x, y) = padded_ranges(points.iter());
        assert_eq!(x, 0.0..1.0);
        assert_eq!(y, 0.0..1.0);
    }
}

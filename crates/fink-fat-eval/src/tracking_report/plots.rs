//! `plotters` charts summarizing a [`TrackingReport`] across a whole run:
//! night-after-night line charts plus dataset-wide aggregate histograms.
//!
//! Reuses the generic chart primitives from [`crate::seed_bank_report::plots`]
//! (`draw_line_chart`, `histogram_bins`) rather than duplicating the
//! `plotters` boilerplate.

use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use plotters::prelude::*;

use crate::seed_bank_report::plots::{
    CHART_HEIGHT, CHART_WIDTH, LABEL_AREA, MARGIN, Series, draw_line_chart, histogram_bins,
};
use crate::tracking_report::report::{AggregatedTrackingStats, TrackingReport};

/// Number of live branches and active lineages per night.
pub fn plot_branches_and_lineages_per_night(
    report: &TrackingReport,
    output_path: &Utf8Path,
) -> Result<()> {
    let branches = report
        .per_night
        .iter()
        .map(|n| (n.step as f64, n.n_branches as f64))
        .collect();
    let lineages = report
        .per_night
        .iter()
        .map(|n| (n.step as f64, n.n_active_lineages as f64))
        .collect();

    draw_line_chart(
        output_path,
        "Branches and active lineages per night",
        "Night (step)",
        "Count",
        &[
            Series {
                label: "Branches",
                color: BLUE,
                points: branches,
            },
            Series {
                label: "Active lineages",
                color: RED,
                points: lineages,
            },
        ],
    )
}

/// Recall (per-night), purity, and cumulative completeness, as percentages.
pub fn plot_recall_purity_completeness_per_night(
    report: &TrackingReport,
    output_path: &Utf8Path,
) -> Result<()> {
    let series_from = |f: fn(&crate::tracking_report::night_stats::NightTrackingStats) -> f64| -> Vec<(f64, f64)> {
        report
            .per_night
            .iter()
            .map(|n| (n.step as f64, f(n)))
            .filter(|(_, y)| y.is_finite())
            .collect()
    };

    draw_line_chart(
        output_path,
        "Recall, purity and cumulative completeness per night",
        "Night (step)",
        "Percent",
        &[
            Series {
                label: "Recall (pure, tonight)",
                color: BLUE,
                points: series_from(|n| n.recall_pct_tonight),
            },
            Series {
                label: "Purity",
                color: RED,
                points: series_from(|n| n.purity_pct),
            },
            Series {
                label: "Completeness (cumulative)",
                color: GREEN,
                points: series_from(|n| n.completeness_pct_so_far),
            },
        ],
    )
}

/// Mean cumulative LLR and mean effective sample size, per night.
pub fn plot_llr_and_ess_per_night(report: &TrackingReport, output_path: &Utf8Path) -> Result<()> {
    let llr = report
        .per_night
        .iter()
        .map(|n| (n.step as f64, n.cumulative_llr.mean))
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "Mean cumulative LLR per night",
        "Night (step)",
        "Mean cumulative LLR",
        &[Series {
            label: "Cumulative LLR",
            color: BLUE,
            points: llr,
        }],
    )
}

/// Mean Kalman error-box radius (bank-level predictive box, and
/// per-hypothesis 1σ box), per night.
pub fn plot_error_box_radius_per_night(
    report: &TrackingReport,
    output_path: &Utf8Path,
) -> Result<()> {
    let hypothesis = report
        .per_night
        .iter()
        .map(|n| (n.step as f64, n.hypothesis_error_box_radius_arcsec.mean))
        .filter(|(_, y)| y.is_finite())
        .collect();
    let bank = report
        .per_night
        .iter()
        .filter_map(|n| {
            n.bank_error_box_radius_arcsec
                .as_ref()
                .map(|s| (n.step as f64, s.mean))
        })
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "Mean Kalman error-box radius per night",
        "Night (step)",
        "Radius (arcsec)",
        &[
            Series {
                label: "Per-hypothesis (1σ)",
                color: BLUE,
                points: hypothesis,
            },
            Series {
                label: "Per-bank (predictive)",
                color: RED,
                points: bank,
            },
        ],
    )
}

/// Mean number of next-night observations landing inside a bank's predicted
/// error box, per night — a crowding/ambiguity proxy.
pub fn plot_observations_in_box_per_night(
    report: &TrackingReport,
    output_path: &Utf8Path,
) -> Result<()> {
    let points = report
        .per_night
        .iter()
        .filter_map(|n| {
            n.n_observations_in_box_next_night
                .as_ref()
                .map(|s| (n.step as f64, s.mean))
        })
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "Mean observations inside next-night error box, per night",
        "Night (step)",
        "Observations in box",
        &[Series {
            label: "Observations in box",
            color: BLUE,
            points,
        }],
    )
}

/// Wall-clock time spent advancing each night.
pub fn plot_timing_per_night(report: &TrackingReport, output_path: &Utf8Path) -> Result<()> {
    let points = report
        .per_night
        .iter()
        .map(|n| (n.step as f64, n.elapsed_ms))
        .collect();

    draw_line_chart(
        output_path,
        "Wall time per night",
        "Night (step)",
        "Milliseconds",
        &[Series {
            label: "advance_one_night",
            color: BLUE,
            points,
        }],
    )
}

/// Generic equispaced histogram over a pre-computed sample vector, sharing
/// the exact bin/draw logic `seed_bank_report::plots` uses for the
/// hypotheses-per-bank histogram.
fn plot_histogram(
    samples: &[f64],
    title: &str,
    x_label: &str,
    output_path: &Utf8Path,
) -> Result<()> {
    const N_BINS: usize = 40;

    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (edges, bin_counts) = histogram_bins(&sorted, N_BINS);

    let root =
        BitMapBackend::new(output_path.as_str(), (CHART_WIDTH, CHART_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = edges.last().copied().unwrap_or(1.0).max(1.0);
    let y_max = bin_counts.iter().copied().max().unwrap_or(1).max(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28))
        .margin(MARGIN)
        .x_label_area_size(LABEL_AREA)
        .y_label_area_size(LABEL_AREA)
        .build_cartesian_2d(0.0..x_max, 0u32..(y_max + y_max / 10 + 1))
        .with_context(|| format!("failed to build chart area for {output_path}"))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("Count")
        .draw()?;

    chart.draw_series(bin_counts.iter().enumerate().map(|(i, &count)| {
        Rectangle::new([(edges[i], 0), (edges[i + 1], count)], BLUE.filled())
    }))?;

    root.present()
        .with_context(|| format!("failed to write chart to {output_path}"))?;
    Ok(())
}

/// Bar chart of a small number of named integer counts (e.g. the per-object
/// outcome category breakdown).
fn plot_bar_chart(
    counts: &[(&str, usize)],
    title: &str,
    y_label: &str,
    output_path: &Utf8Path,
) -> Result<()> {
    let root =
        BitMapBackend::new(output_path.as_str(), (CHART_WIDTH, CHART_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_max = counts
        .iter()
        .map(|&(_, c)| c as u32)
        .max()
        .unwrap_or(1)
        .max(1);
    let n = counts.len() as i32;
    let labels: Vec<&str> = counts.iter().map(|&(label, _)| label).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28))
        .margin(MARGIN)
        .x_label_area_size(LABEL_AREA * 2)
        .y_label_area_size(LABEL_AREA)
        .build_cartesian_2d(0..n, 0u32..(y_max + y_max / 10 + 1))
        .with_context(|| format!("failed to build chart area for {output_path}"))?;

    chart
        .configure_mesh()
        .y_desc(y_label)
        .x_desc("Outcome")
        .x_labels(labels.len())
        .x_label_formatter(&|x| {
            labels
                .get(*x as usize)
                .map(|s| s.to_string())
                .unwrap_or_default()
        })
        .disable_x_mesh()
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &(_, count))| {
        Rectangle::new([(i as i32, 0), (i as i32 + 1, count as u32)], BLUE.filled())
    }))?;

    root.present()
        .with_context(|| format!("failed to write chart to {output_path}"))?;
    Ok(())
}

/// Bar chart of how many multi-detection ground-truth objects fall into
/// each [`crate::tracking_report::object_outcome::ObjectOutcome`] category —
/// see that module's doc for what each category means and how to act on it.
pub fn plot_object_outcome_breakdown(
    outcome_counts: &[(&str, usize)],
    output_path: &Utf8Path,
) -> Result<()> {
    plot_bar_chart(
        outcome_counts,
        "Ground-truth object outcome breakdown",
        "Objects",
        output_path,
    )
}

/// Histogram of the best pure-branch coverage ratio ever reached for every
/// multi-detection object (`best_pure_coverage / n_obs_so_far`, 0 for
/// objects never captured purely). Distinguishes "the tracker never had a
/// good branch for this object" (mass near 0) from "the tracker had a good
/// branch at some point, but the end-of-run snapshot doesn't show it" (mass
/// near 1 despite a low final completeness).
pub fn plot_best_pure_coverage_histogram(samples: &[f64], output_path: &Utf8Path) -> Result<()> {
    plot_histogram(
        samples,
        "Best-ever pure-branch coverage ratio per object",
        "Coverage ratio (best pure branch / total observations seen)",
        output_path,
    )
}

/// Aggregated histograms over the whole run: error-box radii (bank +
/// hypothesis level, one sample per night-mean — see
/// [`AggregatedTrackingStats`]'s doc), branches-per-lineage, and lineage
/// survival duration.
pub fn plot_aggregated_histograms(
    aggregated: &AggregatedTrackingStats,
    output_dir: &Utf8Path,
) -> Result<Vec<Utf8PathBuf>> {
    let plots = [
        (
            "hypothesis_error_box_radius_histogram.png",
            aggregated.hypothesis_error_box_radius_samples.as_slice(),
            "Per-hypothesis error-box radius (nightly mean, arcsec)",
            "Radius (arcsec)",
        ),
        (
            "bank_error_box_radius_histogram.png",
            aggregated.bank_error_box_radius_samples.as_slice(),
            "Per-bank predictive error-box radius (nightly mean, arcsec)",
            "Radius (arcsec)",
        ),
        (
            "branches_per_lineage_histogram.png",
            aggregated.branches_per_lineage_samples.as_slice(),
            "Branches per lineage (nightly mean)",
            "Branches per lineage",
        ),
        (
            "lineage_survival_histogram.png",
            aggregated.lineage_survival_samples.as_slice(),
            "Lineage survival duration (died lineages, nights)",
            "Nights survived",
        ),
    ];

    let mut written = Vec::with_capacity(plots.len());
    for (filename, samples, title, x_label) in plots {
        let path = output_dir.join(filename);
        plot_histogram(samples, title, x_label, &path)?;
        written.push(path);
    }
    Ok(written)
}

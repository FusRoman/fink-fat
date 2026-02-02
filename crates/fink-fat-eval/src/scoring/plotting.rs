use anyhow::{Context, Result};
use camino::Utf8Path;
use plotters::prelude::*;

use crate::seeding::plotting::histogram;

/// Plot two overlaid histograms (same vs different) into a PNG.
pub fn plot_two_histograms(
    out_path: &Utf8Path,
    width: u32,
    height: u32,
    title: &str,
    x_label: &str,
    same: &[f64],
    diff: &[f64],
    bins: usize,
) -> Result<()> {
    let root = BitMapBackend::new(out_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let all_iter = same
        .iter()
        .chain(diff.iter())
        .copied()
        .filter(|x| x.is_finite());

    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    for x in all_iter {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
    }
    anyhow::ensure!(
        xmin.is_finite() && xmax.is_finite() && xmin < xmax,
        "empty or degenerate score range"
    );

    let pad = 0.02 * (xmax - xmin);
    xmin -= pad;
    xmax += pad;

    let h_same = histogram(same, xmin, xmax, bins);
    let h_diff = histogram(diff, xmin, xmax, bins);

    let ymax = h_same
        .iter()
        .chain(h_diff.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as i32;

    let mut chart = ChartBuilder::on(&root)
        .margin(18)
        .caption(title, ("sans-serif", 28))
        .x_label_area_size(44)
        .y_label_area_size(60)
        .build_cartesian_2d(xmin..xmax, 0i32..ymax)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("count")
        .disable_mesh()
        .draw()?;

    let bin_w = (xmax - xmin) / (bins as f64);

    let style_same = BLUE.mix(0.35).filled();
    let style_diff = RED.mix(0.35).filled();

    chart.draw_series((0..bins).map(|k| {
        let x0 = xmin + (k as f64) * bin_w;
        let x1 = x0 + bin_w;
        Rectangle::new([(x0, 0), (x1, h_same[k] as i32)], style_same.clone())
    }))?;

    chart.draw_series((0..bins).map(|k| {
        let x0 = xmin + (k as f64) * bin_w;
        let x1 = x0 + bin_w;
        Rectangle::new([(x0, 0), (x1, h_diff[k] as i32)], style_diff.clone())
    }))?;

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.draw(&Rectangle::new([(30, 40), (60, 60)], style_same))?;
    root.draw(&Text::new(
        "same asteroid",
        (70, 42),
        ("sans-serif", 18).into_font(),
    ))?;
    root.draw(&Rectangle::new([(30, 70), (60, 90)], style_diff))?;
    root.draw(&Text::new(
        "different asteroid",
        (70, 72),
        ("sans-serif", 18).into_font(),
    ))?;

    root.present().context("failed to write PNG")?;
    Ok(())
}

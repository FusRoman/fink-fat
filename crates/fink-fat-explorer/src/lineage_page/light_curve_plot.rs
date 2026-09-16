use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{ErrorData, ErrorType, Marker, Mode, Title},
    layout::{Axis, AxisType, Layout, Margin},
    Plot, Scatter,
};

use super::observations_table::ObservationRow;
use super::x_axis::XAxisUnit;
#[cfg(target_arch = "wasm32")]
use super::x_axis::{x_values_for_observations, XAxisValues};

/// Inverse of prep_alert.py's `mapping_band = {"u": 0, "g": 1, "r": 2, "i":
/// 3, "z": 4, "y": 5}` — LSST-only, no other survey stores photometry here.
fn band_name(filter: i16) -> String {
    match filter {
        0 => "u".to_string(),
        1 => "g".to_string(),
        2 => "r".to_string(),
        3 => "i".to_string(),
        4 => "z".to_string(),
        5 => "y".to_string(),
        n => format!("Filter {n}"),
    }
}

/// Standard LSST ugrizy plotting palette.
fn band_color(filter: i16) -> &'static str {
    match filter {
        0 => "#56b4e9",
        1 => "#008060",
        2 => "#ff4000",
        3 => "#850000",
        4 => "#6600cc",
        5 => "#000000",
        _ => "#888888",
    }
}

/// Observation indices grouped by filter, in `u,g,r,i,z,y` order (then any
/// unrecognised codes), so the legend is stable regardless of which bands a
/// given lineage happens to have observations in.
fn group_by_band(observations: &[ObservationRow]) -> Vec<(i16, Vec<usize>)> {
    let mut groups: Vec<(i16, Vec<usize>)> = Vec::new();
    for band in 0..=5i16 {
        let indices: Vec<usize> = observations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.filter == band)
            .map(|(i, _)| i)
            .collect();
        if !indices.is_empty() {
            groups.push((band, indices));
        }
    }
    for (i, o) in observations.iter().enumerate() {
        if !(0..=5).contains(&o.filter) {
            match groups.iter_mut().find(|(b, _)| *b == o.filter) {
                Some((_, indices)) => indices.push(i),
                None => groups.push((o.filter, vec![i])),
            }
        }
    }
    groups
}

/// The x part of the hover template, matched to whichever axis unit is
/// currently selected so the value shown makes sense next to it.
fn x_hover_format(unit: XAxisUnit) -> &'static str {
    match unit {
        XAxisUnit::ObservationIndex => "Obs #%{x:.0f}",
        XAxisUnit::DaysSinceFirst => "%{x:.2f} d since first obs",
        XAxisUnit::IsoUtcDate => "%{x|%Y-%m-%d %H:%M} UTC",
    }
}

/// `hovertemplate` can't reference `%{error_y}` (Plotly.js doesn't expose
/// error arrays to hover templates), so `mag_err` is threaded through as
/// `customdata` and d3-formatted inline instead.
fn hover_template(unit: XAxisUnit) -> String {
    format!(
        "{}<br>%{{y:.3f}} ± %{{customdata:.3f}} mag",
        x_hover_format(unit)
    )
}

fn magnitude_range(observations: &[ObservationRow]) -> (f64, f64) {
    let min = observations
        .iter()
        .map(|o| o.magnitude)
        .fold(f64::INFINITY, f64::min);
    let max = observations
        .iter()
        .map(|o| o.magnitude)
        .fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// One `Scatter` trace for a single band's observations, magnitude on y
/// with its error bars, colored/named by `band`.
#[cfg(target_arch = "wasm32")]
fn band_trace<X>(
    band: i16,
    x: Vec<X>,
    observations: &[ObservationRow],
    indices: &[usize],
    x_axis_unit: XAxisUnit,
) -> Box<Scatter<X, f64>>
where
    X: serde::Serialize + Clone + Default + 'static,
{
    let y: Vec<f64> = indices.iter().map(|&i| observations[i].magnitude).collect();
    let y_err: Vec<f64> = indices.iter().map(|&i| observations[i].mag_err).collect();
    Scatter::new(x, y)
        .name(band_name(band))
        .mode(Mode::Markers)
        .marker(Marker::new().color(band_color(band)).size(7))
        .error_y(
            ErrorData::new(ErrorType::Data)
                .array(y_err.clone())
                .symmetric(true),
        )
        .custom_data(y_err)
        .hover_template(hover_template(x_axis_unit))
}

/// Light curve: magnitude vs. time, one trace per LSST filter band.
#[component]
pub fn LightCurvePlot(observations: Vec<ObservationRow>, x_axis_unit: XAxisUnit) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(observations, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            if observations.is_empty() {
                return;
            }

            // Computed once over every observation (not per band), so
            // "days since first"/the date axis anchor to the lineage's
            // overall first observation rather than that band's first.
            let x_all = x_values_for_observations(x_axis_unit, &observations);

            let mut plot = Plot::new();
            let mut x_axis = Axis::new().title(Title::from(x_axis_unit.axis_title()));
            if let XAxisValues::Date(_) = x_all {
                x_axis = x_axis.type_(AxisType::Date);
            }

            let groups = group_by_band(&observations);
            match &x_all {
                XAxisValues::Numeric(x) => {
                    for (band, indices) in &groups {
                        let x: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
                        plot.add_trace(band_trace(*band, x, &observations, indices, x_axis_unit));
                    }
                }
                XAxisValues::Date(x) => {
                    for (band, indices) in &groups {
                        let x: Vec<String> = indices.iter().map(|&i| x[i].clone()).collect();
                        plot.add_trace(band_trace(*band, x, &observations, indices, x_axis_unit));
                    }
                }
            }

            // Reversed range: brighter (smaller magnitude) at the top.
            // Cartesian `Axis::auto_range()` only accepts `bool` in
            // plotly-rs 0.14 (no `AutoRange::Reversed`, that's polar-only),
            // so a descending `.range()` is the only way to flip it.
            let (min_mag, max_mag) = magnitude_range(&observations);
            let y_axis = Axis::new()
                .title(Title::from("Magnitude"))
                .range(vec![max_mag + 0.2, min_mag - 0.2]);

            let layout = Layout::new()
                .height(420)
                .margin(Margin::new().top(20).right(20))
                .x_axis(x_axis)
                .y_axis(y_axis);
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-light-curve-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-light-curve-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body",
                h2 { class: "card-title", "Light curve" }
                div {
                    id: "lineage-light-curve-plot",
                    style: "width: 100%;",
                    onmounted: move |_| is_mounted.set(true),
                }
            }
        }
    }
}

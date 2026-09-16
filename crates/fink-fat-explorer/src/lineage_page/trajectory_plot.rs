use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{color::Rgba, DashType, ErrorData, ErrorType, Line, Marker, Mode, Title},
    layout::{Axis, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::KfStep;
use super::observations_table::ObservationRow;
use super::x_axis::XAxisUnit;
#[cfg(target_arch = "wasm32")]
use super::x_axis::{format_time_labels, x_values_for_observations, x_values_for_steps};

const DEG_PER_ARCSEC: f64 = 1.0 / 3600.0;

/// `hovertemplate` can't reference `%{error_x}`/`%{error_y}` (Plotly.js
/// doesn't expose error arrays to hover templates), so the time label and
/// both sky errors are pre-formatted into the single string `customdata`
/// carries per point.
fn format_hover_extra(time_label: &str, ra_err_arcsec: f64, dec_err_arcsec: f64) -> String {
    format!("σ RA ±{ra_err_arcsec:.3}″  σ Dec ±{dec_err_arcsec:.3}″<br>{time_label}")
}

const TRAJECTORY_HOVER_TEMPLATE: &str = "RA: %{x:.6f}°<br>Dec: %{y:.6f}°<br>%{customdata}";

/// Linearly interpolated marker sizes from `from` (first/oldest point) to
/// `to` (last/newest point) — a cheap direction cue, since this `plotly`
/// version has no auto-oriented arrow marker to show which way a track runs.
/// `Marker::size` only accepts `usize`, hence rounding here rather than at
/// each call site.
fn size_gradient(n: usize, from: usize, to: usize) -> Vec<usize> {
    let (from, to) = (from as f64, to as f64);
    match n {
        0 => Vec::new(),
        1 => vec![to as usize],
        _ => (0..n)
            .map(|i| (from + (to - from) * (i as f64) / ((n - 1) as f64)).round() as usize)
            .collect(),
    }
}

/// Sky-plane trajectory: the observed track (with its own astrometric error
/// bars) overlaid with the Kalman filter's pre-update prediction at each
/// real observation — the "decision" the filter made, with its own error
/// box, before absorbing that point.
#[component]
pub fn TrajectoryPlot(
    observations: Vec<ObservationRow>,
    replay: Vec<KfStep>,
    x_axis_unit: XAxisUnit,
) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(observations, replay, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let obs_ra: Vec<f64> = observations.iter().map(|o| o.ra.to_degrees()).collect();
            let obs_dec: Vec<f64> = observations.iter().map(|o| o.dec.to_degrees()).collect();
            let obs_ra_err: Vec<f64> = observations.iter().map(|o| o.ra_err.to_degrees()).collect();
            let obs_dec_err: Vec<f64> = observations
                .iter()
                .map(|o| o.dec_err.to_degrees())
                .collect();

            let obs_time_labels = format_time_labels(
                x_axis_unit,
                &x_values_for_observations(x_axis_unit, &observations),
            );
            let obs_hover: Vec<String> = observations
                .iter()
                .zip(&obs_time_labels)
                .map(|(o, time_label)| {
                    format_hover_extra(
                        time_label,
                        o.ra_err.to_degrees() * 3600.0,
                        o.dec_err.to_degrees() * 3600.0,
                    )
                })
                .collect();

            let pred_ra: Vec<f64> = replay.iter().map(|s| s.predicted_ra_deg).collect();
            let pred_dec: Vec<f64> = replay.iter().map(|s| s.predicted_dec_deg).collect();
            let pred_ra_err: Vec<f64> = replay
                .iter()
                .map(|s| s.sigma_pred_ra_arcsec * DEG_PER_ARCSEC)
                .collect();
            let pred_dec_err: Vec<f64> = replay
                .iter()
                .map(|s| s.sigma_pred_dec_arcsec * DEG_PER_ARCSEC)
                .collect();
            let pred_time_labels =
                format_time_labels(x_axis_unit, &x_values_for_steps(x_axis_unit, &replay));
            let pred_hover: Vec<String> = replay
                .iter()
                .zip(&pred_time_labels)
                .map(|(s, time_label)| {
                    format_hover_extra(time_label, s.sigma_pred_ra_arcsec, s.sigma_pred_dec_arcsec)
                })
                .collect();

            let mut plot = Plot::new();

            plot.add_trace(
                Scatter::new(obs_ra, obs_dec)
                    .name("Observations")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#2d7fd2").size_array(size_gradient(
                        observations.len(),
                        5,
                        11,
                    )))
                    .line(
                        Line::new()
                            .dash(DashType::Dot)
                            .width(1.5)
                            .color(Rgba::new(45, 127, 210, 0.35)),
                    )
                    .error_x(
                        ErrorData::new(ErrorType::Data)
                            .array(obs_ra_err)
                            .symmetric(true),
                    )
                    .error_y(
                        ErrorData::new(ErrorType::Data)
                            .array(obs_dec_err)
                            .symmetric(true),
                    )
                    .custom_data(obs_hover)
                    .hover_template(TRAJECTORY_HOVER_TEMPLATE),
            );

            plot.add_trace(
                Scatter::new(pred_ra, pred_dec)
                    .name("Kalman prediction (pre-update)")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#d2422d").size_array(size_gradient(
                        replay.len(),
                        4,
                        10,
                    )))
                    .line(
                        Line::new()
                            .dash(DashType::Dot)
                            .width(1.5)
                            .color(Rgba::new(210, 66, 45, 0.35)),
                    )
                    .error_x(
                        ErrorData::new(ErrorType::Data)
                            .array(pred_ra_err)
                            .symmetric(true),
                    )
                    .error_y(
                        ErrorData::new(ErrorType::Data)
                            .array(pred_dec_err)
                            .symmetric(true),
                    )
                    .custom_data(pred_hover)
                    .hover_template(TRAJECTORY_HOVER_TEMPLATE),
            );

            let layout = Layout::new()
                .height(420)
                .margin(Margin::new().top(20).right(20))
                .x_axis(Axis::new().title(Title::from("RA (deg)")))
                .y_axis(Axis::new().title(Title::from("Dec (deg)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-trajectory-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-trajectory-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body",
                h2 { class: "card-title", "Trajectory & Kalman predictions" }
                p { class: "text-xs opacity-60",
                    "Error bars: astrometric 1σ for observations, propagated sky covariance (pre-update) for predictions."
                }
                div {
                    id: "lineage-trajectory-plot",
                    style: "width: 100%;",
                    onmounted: move |_| is_mounted.set(true),
                }
            }
        }
    }
}

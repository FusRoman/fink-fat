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
use crate::skybot_search::{SkybotHit, MAX_RADIUS_ARCSEC, MIN_RADIUS_ARCSEC};

const DEG_PER_ARCSEC: f64 = 1.0 / 3600.0;

/// `hovertemplate` can't reference `%{error_x}`/`%{error_y}` (Plotly.js
/// doesn't expose error arrays to hover templates), so the time label and
/// both sky errors are pre-formatted into the single string `customdata`
/// carries per point.
fn format_hover_extra(time_label: &str, ra_err_arcsec: f64, dec_err_arcsec: f64) -> String {
    format!("σ RA ±{ra_err_arcsec:.3}″  σ Dec ±{dec_err_arcsec:.3}″<br>{time_label}")
}

const TRAJECTORY_HOVER_TEMPLATE: &str = "RA: %{x:.6f}°<br>Dec: %{y:.6f}°<br>%{customdata}";

/// One Skybot hit's hover text: name, class, and whichever of magnitude/
/// distance/positional-error Skybot actually returned for it (all optional
/// in the response, per `skybot_search::parsing::RawSkybotRow`).
#[cfg(target_arch = "wasm32")]
fn format_skybot_hover(hit: &SkybotHit) -> String {
    let mut lines = vec![format!("<b>{}</b>", hit.name), hit.class.clone()];
    if let Some(vmag) = hit.vmag {
        lines.push(format!("V mag {vmag:.2}"));
    }
    if let Some(err) = hit.err_arcsec {
        lines.push(format!("pos. error ±{err:.2}″"));
    }
    if let Some(dg) = hit.geocentric_distance_au {
        lines.push(format!("Δ (geocentric) {dg:.3} au"));
    }
    if let Some(dh) = hit.heliocentric_distance_au {
        lines.push(format!("r (heliocentric) {dh:.3} au"));
    }
    lines.join("<br>")
}

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
    /// Skybot matches found so far (grows as the background search job
    /// progresses) — see [`crate::skybot_search`].
    skybot_hits: Vec<SkybotHit>,
    skybot_running: bool,
    skybot_processed: usize,
    skybot_total: usize,
    skybot_radius_arcsec: f64,
    on_skybot_radius_change: EventHandler<f64>,
    on_skybot_search: EventHandler<()>,
    on_toggle_skybot_panel: EventHandler<()>,
) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);
    let has_skybot_hits = !skybot_hits.is_empty();

    use_effect(use_reactive!(|(
        observations,
        replay,
        x_axis_unit,
        skybot_hits,
    )| {
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

            if !skybot_hits.is_empty() {
                let skybot_ra: Vec<f64> = skybot_hits.iter().map(|h| h.ra_deg).collect();
                let skybot_dec: Vec<f64> = skybot_hits.iter().map(|h| h.dec_deg).collect();
                let skybot_hover: Vec<String> =
                    skybot_hits.iter().map(format_skybot_hover).collect();

                plot.add_trace(
                    Scatter::new(skybot_ra, skybot_dec)
                        .name("Skybot matches")
                        .mode(Mode::Markers)
                        .marker(
                            Marker::new()
                                .color("#2ba84a")
                                .symbol(plotly::common::MarkerSymbol::Diamond)
                                .size(9),
                        )
                        .custom_data(skybot_hover)
                        .hover_template("%{customdata}"),
                );
            }

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
                div { class: "flex flex-wrap items-center justify-between gap-3",
                    h2 { class: "card-title", "Trajectory & Kalman predictions" }
                    div { class: "flex flex-wrap items-center gap-3",
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Radius"
                            input {
                                r#type: "range",
                                class: "range range-xs w-24",
                                min: "{MIN_RADIUS_ARCSEC}",
                                max: "{MAX_RADIUS_ARCSEC}",
                                step: "1",
                                disabled: skybot_running,
                                value: "{skybot_radius_arcsec}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        on_skybot_radius_change.call(v);
                                    }
                                },
                            }
                            span { "{skybot_radius_arcsec:.0}\"" }
                        }
                        button {
                            class: "btn btn-sm btn-outline",
                            r#type: "button",
                            disabled: skybot_running,
                            onclick: move |_| on_skybot_search.call(()),
                            if skybot_running {
                                span { class: "loading loading-spinner loading-xs" }
                                "Searching Skybot ({skybot_processed}/{skybot_total})"
                            } else {
                                "Search Skybot"
                            }
                        }
                        button {
                            class: "btn btn-sm btn-ghost btn-circle",
                            r#type: "button",
                            disabled: !has_skybot_hits,
                            title: "Skybot matches found so far",
                            onclick: move |_| on_toggle_skybot_panel.call(()),
                            "☰"
                        }
                    }
                }
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

use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{ErrorData, ErrorType, Marker, Mode, Title},
    layout::{Axis, AxisType, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::{HypothesisSnapshot, KfStep};
use super::x_axis::XAxisUnit;
#[cfg(target_arch = "wasm32")]
use super::x_axis::{x_values_for_hypotheses, x_values_for_steps, XAxisValues};

/// Evolution of the posterior topocentric range ρ and range-rate ρ̇ of the
/// MAP hypothesis as real observations are absorbed, each with its own 1σ
/// error bar from the filter's posterior covariance — plus a scatter of
/// *every* surviving hypothesis's pre-update (ρ, ρ̇), to see the range
/// ambiguity collapse over time rather than only its final winner.
#[component]
pub fn RhoEvolutionPlot(
    replay: Vec<KfStep>,
    hypotheses: Vec<HypothesisSnapshot>,
    x_axis_unit: XAxisUnit,
) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                h2 { class: "card-title", "ρ / ρ̇ evolution" }
                RhoPlot { replay: replay.clone(), x_axis_unit }
                RhoDotPlot { replay: replay.clone(), x_axis_unit }
                HypothesisRhoScatterPlot { hypotheses, replay, x_axis_unit }
            }
        }
    }
}

#[component]
fn RhoPlot(replay: Vec<KfStep>, x_axis_unit: XAxisUnit) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(replay, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let rho: Vec<f64> = replay.iter().map(|s| s.posterior_rho_au).collect();
            let sigma_rho: Vec<f64> = replay.iter().map(|s| s.sigma_rho_au).collect();

            let mut plot = Plot::new();
            let mut x_axis = Axis::new().title(Title::from(x_axis_unit.axis_title()));
            let trace_x = x_values_for_steps(x_axis_unit, &replay);
            if let XAxisValues::Date(_) = trace_x {
                x_axis = x_axis.type_(AxisType::Date);
            }
            match trace_x {
                XAxisValues::Numeric(x) => plot.add_trace(
                    Scatter::new(x, rho)
                        .name("ρ (posterior)")
                        .mode(Mode::LinesMarkers)
                        .marker(Marker::new().color("#8d2dd2"))
                        .error_y(
                            ErrorData::new(ErrorType::Data)
                                .array(sigma_rho)
                                .symmetric(true),
                        ),
                ),
                XAxisValues::Date(x) => plot.add_trace(
                    Scatter::new(x, rho)
                        .name("ρ (posterior)")
                        .mode(Mode::LinesMarkers)
                        .marker(Marker::new().color("#8d2dd2"))
                        .error_y(
                            ErrorData::new(ErrorType::Data)
                                .array(sigma_rho)
                                .symmetric(true),
                        ),
                ),
            }

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(x_axis)
                .y_axis(Axis::new().title(Title::from("ρ (AU)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-rho-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-rho-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-rho-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

#[component]
fn RhoDotPlot(replay: Vec<KfStep>, x_axis_unit: XAxisUnit) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(replay, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let rho_dot: Vec<f64> = replay
                .iter()
                .map(|s| s.posterior_rho_dot_au_per_day)
                .collect();
            let sigma_rho_dot: Vec<f64> =
                replay.iter().map(|s| s.sigma_rho_dot_au_per_day).collect();

            let mut plot = Plot::new();
            let mut x_axis = Axis::new().title(Title::from(x_axis_unit.axis_title()));
            let trace_x = x_values_for_steps(x_axis_unit, &replay);
            if let XAxisValues::Date(_) = trace_x {
                x_axis = x_axis.type_(AxisType::Date);
            }
            match trace_x {
                XAxisValues::Numeric(x) => plot.add_trace(
                    Scatter::new(x, rho_dot)
                        .name("ρ̇ (posterior)")
                        .mode(Mode::LinesMarkers)
                        .marker(Marker::new().color("#d2642d"))
                        .error_y(
                            ErrorData::new(ErrorType::Data)
                                .array(sigma_rho_dot)
                                .symmetric(true),
                        ),
                ),
                XAxisValues::Date(x) => plot.add_trace(
                    Scatter::new(x, rho_dot)
                        .name("ρ̇ (posterior)")
                        .mode(Mode::LinesMarkers)
                        .marker(Marker::new().color("#d2642d"))
                        .error_y(
                            ErrorData::new(ErrorType::Data)
                                .array(sigma_rho_dot)
                                .symmetric(true),
                        ),
                ),
            }

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(x_axis)
                .y_axis(Axis::new().title(Title::from("ρ̇ (AU/day)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-rho-dot-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-rho-dot-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-rho-dot-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

/// Scatter of every surviving hypothesis's pre-update (ρ, ρ̇) at each replay
/// step — shows the range/range-rate ambiguity narrowing over time, rather
/// than only the MAP hypothesis [`RhoPlot`]/[`RhoDotPlot`] already track.
#[component]
fn HypothesisRhoScatterPlot(
    hypotheses: Vec<HypothesisSnapshot>,
    replay: Vec<KfStep>,
    x_axis_unit: XAxisUnit,
) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(hypotheses, replay, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let rho: Vec<f64> = hypotheses.iter().map(|h| h.rho_au).collect();
            let rho_dot: Vec<f64> = hypotheses.iter().map(|h| h.rho_dot_au_per_day).collect();

            let axis_title = Title::from(x_axis_unit.axis_title());
            let trace_x = x_values_for_hypotheses(x_axis_unit, &hypotheses, &replay);
            let is_date = matches!(trace_x, XAxisValues::Date(_));

            let mut rho_plot = Plot::new();
            let mut rho_dot_plot = Plot::new();
            match trace_x {
                XAxisValues::Numeric(x) => {
                    rho_plot.add_trace(
                        Scatter::new(x.clone(), rho)
                            .name("hypotheses")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#8d2dd2").opacity(0.35).size(5)),
                    );
                    rho_dot_plot.add_trace(
                        Scatter::new(x, rho_dot)
                            .name("hypotheses")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#d2642d").opacity(0.35).size(5)),
                    );
                }
                XAxisValues::Date(x) => {
                    rho_plot.add_trace(
                        Scatter::new(x.clone(), rho)
                            .name("hypotheses")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#8d2dd2").opacity(0.35).size(5)),
                    );
                    rho_dot_plot.add_trace(
                        Scatter::new(x, rho_dot)
                            .name("hypotheses")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#d2642d").opacity(0.35).size(5)),
                    );
                }
            }

            let mut rho_x_axis = Axis::new().title(axis_title.clone());
            let mut rho_dot_x_axis = Axis::new().title(axis_title);
            if is_date {
                rho_x_axis = rho_x_axis.type_(AxisType::Date);
                rho_dot_x_axis = rho_dot_x_axis.type_(AxisType::Date);
            }

            rho_plot.set_layout(
                Layout::new()
                    .height(200)
                    .margin(Margin::new().top(10).right(10).bottom(30))
                    .show_legend(false)
                    .x_axis(rho_x_axis)
                    .y_axis(Axis::new().title(Title::from("ρ, all hypotheses (AU)"))),
            );
            rho_dot_plot.set_layout(
                Layout::new()
                    .height(200)
                    .margin(Margin::new().top(10).right(10).bottom(30))
                    .show_legend(false)
                    .x_axis(rho_dot_x_axis)
                    .y_axis(Axis::new().title(Title::from("ρ̇, all hypotheses (AU/day)"))),
            );

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-hypothesis-rho-plot", &rho_plot).await;
                    plotly::bindings::react("lineage-hypothesis-rho-dot-plot", &rho_dot_plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-hypothesis-rho-plot", &rho_plot).await;
                    plotly::bindings::new_plot("lineage-hypothesis-rho-dot-plot", &rho_dot_plot)
                        .await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-hypothesis-rho-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
        div { id: "lineage-hypothesis-rho-dot-plot", style: "width: 100%;" }
    }
}

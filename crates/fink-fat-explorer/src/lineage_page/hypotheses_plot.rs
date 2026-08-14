use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Marker, Mode, Title},
    layout::{Axis, AxisType, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::KfStep;
use super::x_axis::XAxisUnit;
#[cfg(target_arch = "wasm32")]
use super::x_axis::{x_values_for_steps, XAxisValues};

/// Bank-level diagnostics of the multi-hypothesis replay: how many
/// hypotheses survive at each real observation, and how large the
/// pre-update search region is — both should shrink as the range/range-rate
/// ambiguity collapses.
#[component]
pub fn HypothesesPlot(replay: Vec<KfStep>, x_axis_unit: XAxisUnit) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                h2 { class: "card-title", "Hypothesis bank" }
                HypothesisCountPlot { replay: replay.clone(), x_axis_unit }
                SearchRegionPlot { replay, x_axis_unit }
            }
        }
    }
}

#[component]
fn HypothesisCountPlot(replay: Vec<KfStep>, x_axis_unit: XAxisUnit) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(replay, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let n_hypotheses: Vec<f64> = replay.iter().map(|s| s.n_hypotheses as f64).collect();
            let effective: Vec<f64> = replay.iter().map(|s| s.effective_sample_size).collect();

            let mut plot = Plot::new();
            let mut x_axis = Axis::new().title(Title::from(x_axis_unit.axis_title()));
            match x_values_for_steps(x_axis_unit, &replay) {
                XAxisValues::Numeric(x) => {
                    plot.add_trace(
                        Scatter::new(x.clone(), n_hypotheses)
                            .name("Live hypotheses")
                            .mode(Mode::LinesMarkers)
                            .marker(Marker::new().color("#2d7fd2")),
                    );
                    plot.add_trace(
                        Scatter::new(x, effective)
                            .name("Effective sample size")
                            .mode(Mode::LinesMarkers)
                            .marker(Marker::new().color("#2dd25d")),
                    );
                }
                XAxisValues::Date(x) => {
                    x_axis = x_axis.type_(AxisType::Date);
                    plot.add_trace(
                        Scatter::new(x.clone(), n_hypotheses)
                            .name("Live hypotheses")
                            .mode(Mode::LinesMarkers)
                            .marker(Marker::new().color("#2d7fd2")),
                    );
                    plot.add_trace(
                        Scatter::new(x, effective)
                            .name("Effective sample size")
                            .mode(Mode::LinesMarkers)
                            .marker(Marker::new().color("#2dd25d")),
                    );
                }
            }

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(x_axis)
                .y_axis(Axis::new().title(Title::from("Hypothesis count")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-hypothesis-count-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-hypothesis-count-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-hypothesis-count-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

#[component]
fn SearchRegionPlot(replay: Vec<KfStep>, x_axis_unit: XAxisUnit) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(replay, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let radius: Vec<f64> = replay
                .iter()
                .map(|s| s.search_region_radius_arcsec)
                .collect();

            let mut plot = Plot::new();
            let mut x_axis = Axis::new().title(Title::from(x_axis_unit.axis_title()));
            match x_values_for_steps(x_axis_unit, &replay) {
                XAxisValues::Numeric(x) => {
                    plot.add_trace(
                        Scatter::new(x, radius)
                            .name("Search region radius")
                            .mode(Mode::LinesMarkers)
                            .marker(Marker::new().color("#d2422d")),
                    );
                }
                XAxisValues::Date(x) => {
                    x_axis = x_axis.type_(AxisType::Date);
                    plot.add_trace(
                        Scatter::new(x, radius)
                            .name("Search region radius")
                            .mode(Mode::LinesMarkers)
                            .marker(Marker::new().color("#d2422d")),
                    );
                }
            }

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(x_axis)
                .y_axis(
                    Axis::new()
                        .title(Title::from("Search region radius (arcsec)"))
                        .type_(AxisType::Log),
                );
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-search-region-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-search-region-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-search-region-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

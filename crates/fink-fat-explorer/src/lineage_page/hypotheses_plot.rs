use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Marker, Mode, Title},
    layout::{Axis, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::KfStep;

/// Bank-level diagnostics of the multi-hypothesis replay: how many
/// hypotheses survive at each real observation, and how large the
/// pre-update search region is — both should shrink as the range/range-rate
/// ambiguity collapses.
#[component]
pub fn HypothesesPlot(replay: Vec<KfStep>) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                h2 { class: "card-title", "Hypothesis bank" }
                HypothesisCountPlot { replay: replay.clone() }
                SearchRegionPlot { replay }
            }
        }
    }
}

#[component]
fn HypothesisCountPlot(replay: Vec<KfStep>) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(replay,)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let steps: Vec<f64> = replay.iter().map(|s| s.step as f64).collect();
            let n_hypotheses: Vec<f64> = replay.iter().map(|s| s.n_hypotheses as f64).collect();
            let effective: Vec<f64> = replay.iter().map(|s| s.effective_sample_size).collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps.clone(), n_hypotheses)
                    .name("Live hypotheses")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#2d7fd2")),
            );
            plot.add_trace(
                Scatter::new(steps, effective)
                    .name("Effective sample size")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#2dd25d")),
            );

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
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
fn SearchRegionPlot(replay: Vec<KfStep>) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(replay,)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let steps: Vec<f64> = replay.iter().map(|s| s.step as f64).collect();
            let radius: Vec<f64> = replay
                .iter()
                .map(|s| s.search_region_radius_arcsec)
                .collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps, radius)
                    .name("Search region radius")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#d2422d")),
            );

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
                .y_axis(Axis::new().title(Title::from("Search region radius (arcsec)")).type_(
                    plotly::layout::AxisType::Log,
                ));
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

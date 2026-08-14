use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{ErrorData, ErrorType, Marker, Mode, Title},
    layout::{Axis, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::{HypothesisSnapshot, KfStep};

/// Evolution of the posterior topocentric range ρ and range-rate ρ̇ of the
/// MAP hypothesis as real observations are absorbed, each with its own 1σ
/// error bar from the filter's posterior covariance — plus a scatter of
/// *every* surviving hypothesis's pre-update (ρ, ρ̇), to see the range
/// ambiguity collapse over time rather than only its final winner.
#[component]
pub fn RhoEvolutionPlot(replay: Vec<KfStep>, hypotheses: Vec<HypothesisSnapshot>) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                h2 { class: "card-title", "ρ / ρ̇ evolution" }
                RhoPlot { replay: replay.clone() }
                RhoDotPlot { replay }
                HypothesisRhoScatterPlot { hypotheses }
            }
        }
    }
}

#[component]
fn RhoPlot(replay: Vec<KfStep>) -> Element {
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
            let rho: Vec<f64> = replay.iter().map(|s| s.posterior_rho_au).collect();
            let sigma_rho: Vec<f64> = replay.iter().map(|s| s.sigma_rho_au).collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps, rho)
                    .name("ρ (posterior)")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#8d2dd2"))
                    .error_y(ErrorData::new(ErrorType::Data).array(sigma_rho).symmetric(true)),
            );

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
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
fn RhoDotPlot(replay: Vec<KfStep>) -> Element {
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
            let rho_dot: Vec<f64> = replay.iter().map(|s| s.posterior_rho_dot_au_per_day).collect();
            let sigma_rho_dot: Vec<f64> = replay.iter().map(|s| s.sigma_rho_dot_au_per_day).collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps, rho_dot)
                    .name("ρ̇ (posterior)")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#d2642d"))
                    .error_y(ErrorData::new(ErrorType::Data).array(sigma_rho_dot).symmetric(true)),
            );

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
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
fn HypothesisRhoScatterPlot(hypotheses: Vec<HypothesisSnapshot>) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(hypotheses,)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let steps: Vec<f64> = hypotheses.iter().map(|h| h.step as f64).collect();
            let rho: Vec<f64> = hypotheses.iter().map(|h| h.rho_au).collect();
            let rho_dot: Vec<f64> = hypotheses.iter().map(|h| h.rho_dot_au_per_day).collect();

            let mut rho_plot = Plot::new();
            rho_plot.add_trace(
                Scatter::new(steps.clone(), rho)
                    .name("hypotheses")
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("#8d2dd2").opacity(0.35).size(5)),
            );
            rho_plot.set_layout(
                Layout::new()
                    .height(200)
                    .margin(Margin::new().top(10).right(10).bottom(30))
                    .show_legend(false)
                    .x_axis(Axis::new().title(Title::from("Real observation #")))
                    .y_axis(Axis::new().title(Title::from("ρ, all hypotheses (AU)"))),
            );

            let mut rho_dot_plot = Plot::new();
            rho_dot_plot.add_trace(
                Scatter::new(steps, rho_dot)
                    .name("hypotheses")
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("#d2642d").opacity(0.35).size(5)),
            );
            rho_dot_plot.set_layout(
                Layout::new()
                    .height(200)
                    .margin(Margin::new().top(10).right(10).bottom(30))
                    .show_legend(false)
                    .x_axis(Axis::new().title(Title::from("Real observation #")))
                    .y_axis(Axis::new().title(Title::from("ρ̇, all hypotheses (AU/day)"))),
            );

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-hypothesis-rho-plot", &rho_plot).await;
                    plotly::bindings::react("lineage-hypothesis-rho-dot-plot", &rho_dot_plot)
                        .await;
                } else {
                    plotly::bindings::new_plot("lineage-hypothesis-rho-plot", &rho_plot).await;
                    plotly::bindings::new_plot(
                        "lineage-hypothesis-rho-dot-plot",
                        &rho_dot_plot,
                    )
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

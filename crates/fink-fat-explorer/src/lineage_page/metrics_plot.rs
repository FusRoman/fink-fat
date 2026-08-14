use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Line, Marker, Mode, Title},
    layout::{Axis, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::KfStep;

/// χ² (2 d.o.f.) gate threshold at 95% confidence — the same value used by
/// the engine's default `inflation_chi2_threshold`
/// (`engine_config.best.yaml`'s `kalman_shared_context.config`).
const CHI2_GATE_95: f64 = 5.991;

/// Evolution of the per-point filter-consistency metrics as real
/// observations are absorbed: χ² (NIS), angular separation from the
/// pre-update prediction, and the single-hypothesis log-likelihood (a proxy
/// for the production LLR — see [`KfStep::log_likelihood`]).
#[component]
pub fn MetricsPlot(replay: Vec<KfStep>) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                h2 { class: "card-title", "Filter consistency metrics" }
                Chi2Plot { replay: replay.clone() }
                SeparationPlot { replay: replay.clone() }
                LogLikelihoodPlot { replay }
            }
        }
    }
}

#[component]
fn Chi2Plot(replay: Vec<KfStep>) -> Element {
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
            let nis: Vec<f64> = replay.iter().map(|s| s.nis).collect();
            let threshold = vec![CHI2_GATE_95; steps.len()];

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps.clone(), nis)
                    .name("χ² (NIS)")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#d2422d")),
            );
            plot.add_trace(
                Scatter::new(steps, threshold)
                    .name("95% gate (2 d.o.f.)")
                    .mode(Mode::Lines)
                    .line(
                        Line::new()
                            .dash(plotly::common::DashType::Dash)
                            .color("#888888"),
                    ),
            );

            let layout = Layout::new()
                .height(180)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
                .y_axis(Axis::new().title(Title::from("χ²")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-chi2-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-chi2-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-chi2-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

#[component]
fn SeparationPlot(replay: Vec<KfStep>) -> Element {
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
            let separation: Vec<f64> = replay.iter().map(|s| s.separation_arcsec).collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps, separation)
                    .name("Distance to prediction")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#2d7fd2")),
            );

            let layout = Layout::new()
                .height(180)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
                .y_axis(Axis::new().title(Title::from("Separation (arcsec)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-separation-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-separation-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-separation-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

#[component]
fn LogLikelihoodPlot(replay: Vec<KfStep>) -> Element {
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
            let mut cumulative = 0.0;
            let cumulative_log_lik: Vec<f64> = replay
                .iter()
                .map(|s| {
                    cumulative += s.log_likelihood;
                    cumulative
                })
                .collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(steps, cumulative_log_lik)
                    .name("Cumulative log-likelihood")
                    .mode(Mode::LinesMarkers)
                    .marker(Marker::new().color("#2dd25d")),
            );

            let layout = Layout::new()
                .height(180)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("Real observation #")))
                .y_axis(Axis::new().title(Title::from("log-likelihood (single hyp.)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-loglik-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-loglik-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "lineage-loglik-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

mod alert_cutouts;
mod hypotheses_plot;
mod identity_card;
mod kf_replay;
mod light_curve_plot;
mod metrics_plot;
pub mod observations_table;
mod plot_tabs;
mod rho_evolution_plot;
mod trajectory_plot;
mod x_axis;

use dioxus::prelude::*;

use alert_cutouts::AlertCarousel;
use identity_card::{get_lineage_summary, IdentityCard};
use kf_replay::{replay_kalman_branch, HypothesisSnapshot, KfStep};
use light_curve_plot::LightCurvePlot;
use observations_table::{get_lineage_observations, ObservationRow, ObservationsTable};
use plot_tabs::PlotTabs;
use trajectory_plot::TrajectoryPlot;
use x_axis::XAxisUnit;

/// Which plot is shown next to the identity card.
#[derive(Clone, Copy, PartialEq)]
enum LineageView {
    Trajectory,
    LightCurve,
}

impl LineageView {
    const ALL: [LineageView; 2] = [LineageView::Trajectory, LineageView::LightCurve];

    fn label(self) -> &'static str {
        match self {
            LineageView::Trajectory => "Trajectory",
            LineageView::LightCurve => "Light curve",
        }
    }
}

/// Detail page for a single lineage, reached by clicking its designation in
/// the homepage's lineage table. `lineage_id` is actually the lineage's
/// *designation* string (the value the table links with), not the numeric
/// `lineage_id` primary key — kept as-is to match the existing route.
#[component]
pub fn LineagePage(lineage_id: String) -> Element {
    let summary_lineage_id = lineage_id.clone();
    let summary_resource = use_resource(use_reactive!(|(summary_lineage_id,)| {
        get_lineage_summary(summary_lineage_id)
    }));

    let observations_lineage_id = lineage_id.clone();
    let observations_resource = use_resource(use_reactive!(|(observations_lineage_id,)| {
        get_lineage_observations(observations_lineage_id)
    }));

    let replay_lineage_id = lineage_id.clone();
    let replay_resource = use_resource(use_reactive!(|(replay_lineage_id,)| {
        replay_kalman_branch(replay_lineage_id)
    }));

    let mut lineage_view = use_signal(|| LineageView::Trajectory);
    let mut x_axis_unit = use_signal(|| XAxisUnit::ObservationIndex);

    // Plain owned snapshots, cheap to pass around as props without holding
    // onto the resources' `Ref` guards across the whole render.
    let observations: Vec<ObservationRow> = match &*observations_resource.read() {
        Some(Ok(Some(data))) => data.observations.clone(),
        _ => Vec::new(),
    };
    let replay: Vec<KfStep> = match &*replay_resource.read() {
        Some(Ok(result)) => result.steps.clone(),
        _ => Vec::new(),
    };
    let hypotheses: Vec<HypothesisSnapshot> = match &*replay_resource.read() {
        Some(Ok(result)) => result.hypotheses.clone(),
        _ => Vec::new(),
    };
    let replay_truncated_at: Option<String> = match &*replay_resource.read() {
        Some(Ok(result)) => result.truncated_at.clone(),
        _ => None,
    };

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                Link {
                    to: crate::Route::Home {},
                    class: "link link-hover text-sm",
                    "← Back to lineages"
                }
            }

            match &*summary_resource.read() {
                Some(Ok(summary)) => rsx! {
                    div { class: "flex flex-col xl:flex-row gap-4 items-stretch",
                        div { class: "xl:w-96 flex-none",
                            IdentityCard { summary: summary.clone() }
                        }
                        div { class: "flex-1 flex flex-col gap-2",
                            div { class: "flex flex-wrap items-center justify-between gap-2",
                                div { class: "join",
                                    for view in LineageView::ALL {
                                        button {
                                            key: "{view.label()}",
                                            class: if lineage_view() == view { "join-item btn btn-sm btn-active" } else { "join-item btn btn-sm" },
                                            onclick: move |_| lineage_view.set(view),
                                            "{view.label()}"
                                        }
                                    }
                                }
                                div { class: "join",
                                    for unit in XAxisUnit::ALL {
                                        button {
                                            key: "{unit.label()}",
                                            class: if x_axis_unit() == unit { "join-item btn btn-sm btn-active" } else { "join-item btn btn-sm" },
                                            onclick: move |_| x_axis_unit.set(unit),
                                            "{unit.label()}"
                                        }
                                    }
                                }
                            }
                            match lineage_view() {
                                LineageView::Trajectory => rsx! {
                                    TrajectoryPlot { observations: observations.clone(), replay: replay.clone(), x_axis_unit: x_axis_unit() }
                                },
                                LineageView::LightCurve => rsx! {
                                    LightCurvePlot { observations: observations.clone(), x_axis_unit: x_axis_unit() }
                                },
                            }
                        }
                    }
                },
                Some(Err(e)) => rsx! {
                    div { class: "alert alert-error", "Failed to load lineage: {e}" }
                },
                None => rsx! {
                    div { class: "flex justify-center py-12",
                        span { class: "loading loading-spinner loading-lg" }
                    }
                },
            }

            match &*replay_resource.read() {
                Some(Ok(_)) => rsx! {
                    if let Some(message) = &replay_truncated_at {
                        div { class: "alert alert-warning", "Replay stopped early: {message}" }
                    }
                    PlotTabs { replay, hypotheses }
                },
                Some(Err(e)) => rsx! {
                    div { class: "alert alert-error", "Failed to replay the Kalman filter: {e}" }
                },
                None => rsx! {
                    div { class: "flex justify-center py-6",
                        span { class: "loading loading-spinner" }
                    }
                },
            }

            match &*observations_resource.read() {
                Some(Ok(_)) => rsx! {
                    AlertCarousel { observations: observations.clone() }
                    ObservationsTable { observations }
                },
                Some(Err(e)) => rsx! {
                    div { class: "alert alert-error", "Failed to load observations: {e}" }
                },
                None => rsx! {
                    div { class: "flex justify-center py-6",
                        span { class: "loading loading-spinner" }
                    }
                },
            }
        }
    }
}

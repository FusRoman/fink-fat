mod hypotheses_plot;
mod identity_card;
mod kf_replay;
mod metrics_plot;
pub mod observations_table;
mod plot_tabs;
mod rho_evolution_plot;
mod trajectory_plot;
mod x_axis;

use dioxus::prelude::*;

use identity_card::{get_lineage_summary, IdentityCard};
use kf_replay::{replay_kalman_branch, HypothesisSnapshot, KfStep};
use observations_table::{get_lineage_observations, ObservationRow, ObservationsTable};
use plot_tabs::PlotTabs;
use trajectory_plot::TrajectoryPlot;

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

    // Plain owned snapshots, cheap to pass around as props without holding
    // onto the resources' `Ref` guards across the whole render.
    let observations: Vec<ObservationRow> = match &*observations_resource.read() {
        Some(Ok(rows)) => rows.clone(),
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
                        TrajectoryPlot { observations: observations.clone(), replay: replay.clone() }
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

mod ades_export_modal;
mod alert_cutouts;
mod cross_match_controls;
mod cross_match_panel;
mod hypotheses_plot;
mod identity_card;
mod kf_replay;
mod light_curve_plot;
mod metrics_plot;
pub mod observations_table;
mod orbit3d_glossary;
mod orbit3d_tab;
mod plot_tabs;
mod rho_evolution_plot;
mod trajectory_plot;
mod x_axis;

use dioxus::prelude::*;

use ades_export_modal::AdesExportModal;
use alert_cutouts::AlertCarousel;
use cross_match_panel::CrossMatchPanel;
use identity_card::{get_lineage_summary, IdentityCard};
use kf_replay::{replay_kalman_branch, HypothesisSnapshot, KfStep};
use light_curve_plot::LightCurvePlot;
use observations_table::{get_lineage_observations, ObservationRow, ObservationsTable};
use orbit3d_tab::LineageOrbit3DTab;
use plot_tabs::PlotTabs;
use trajectory_plot::TrajectoryPlot;
use x_axis::XAxisUnit;

use crate::cnd_search::history::get_last_cnd_query;
use crate::cnd_search::run::start_cnd_search;
use crate::cnd_search::status::get_cnd_job_status;
use crate::cnd_search::{
    CndHit, CndJobView, CndQueryPoint, DEFAULT_ANGLE_SEPARATION_ARCSEC, DEFAULT_TIME_SEPARATION_S,
};
use crate::skybot_search::history::get_last_skybot_query;
use crate::skybot_search::run::start_skybot_search;
use crate::skybot_search::status::get_skybot_job_status;
use crate::skybot_search::{JobStatus, SkybotHit, SkybotJobView, SkybotQueryPoint};
use crate::sleep_ms;

/// Milliseconds between polls of a running Skybot search / CND check job —
/// short enough that matches visibly trickle onto the plot as they arrive.
const CROSS_MATCH_POLL_INTERVAL_MS: u64 = 500;

/// Which plot is shown next to the identity card.
#[derive(Clone, Copy, PartialEq)]
enum LineageView {
    Trajectory,
    LightCurve,
    ThreeD,
}

impl LineageView {
    const ALL: [LineageView; 3] = [
        LineageView::Trajectory,
        LineageView::LightCurve,
        LineageView::ThreeD,
    ];

    fn label(self) -> &'static str {
        match self {
            LineageView::Trajectory => "Trajectory",
            LineageView::LightCurve => "Light curve",
            LineageView::ThreeD => "3D",
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
    // Needed to pack a placeholder MPC designation into obs80 lines when
    // submitting to CND — see `crate::cnd_search::obs80::pack_branch_id`.
    let branch_id: Option<i64> = match &*observations_resource.read() {
        Some(Ok(Some(data))) => Some(data.branch_id),
        _ => None,
    };
    let mpc_codes: Vec<String> = observations
        .iter()
        .map(|obs| obs.mpc_code_obs.clone())
        .collect();
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

    let mut skybot_job_id = use_signal(|| None::<u64>);
    let mut skybot_view = use_signal(|| None::<SkybotJobView>);
    let mut skybot_radius = use_signal(|| 10.0_f64);
    let mut cnd_job_id = use_signal(|| None::<u64>);
    let mut cnd_view = use_signal(|| None::<CndJobView>);
    let mut cnd_time_separation_s = use_signal(|| DEFAULT_TIME_SEPARATION_S);
    let mut cnd_angle_separation_arcsec = use_signal(|| DEFAULT_ANGLE_SEPARATION_ARCSEC);
    let mut cross_match_panel_open = use_signal(|| false);
    let mut ades_modal_open = use_signal(|| false);

    // Last persisted attempt for this lineage, per service (see
    // `skybot_search::history`/`cnd_search::history`), loaded alongside
    // everything else on mount so a past search's matches show up without
    // the user having to re-run it.
    let skybot_history_lineage_id = lineage_id.clone();
    let mut skybot_history_resource = use_resource(use_reactive!(|(skybot_history_lineage_id,)| {
        get_last_skybot_query(skybot_history_lineage_id)
    }));
    let cnd_history_lineage_id = lineage_id.clone();
    let mut cnd_history_resource = use_resource(use_reactive!(|(cnd_history_lineage_id,)| {
        get_last_cnd_query(cnd_history_lineage_id)
    }));

    // Seed `skybot_view`/`cnd_view` from the persisted record the first time
    // it loads, so the plot/panel render it through the exact same signal a
    // live search would use — but only if the user hasn't already started a
    // live search this session, so a slow-resolving history fetch can never
    // clobber it.
    use_effect(move || {
        if skybot_view.read().is_some() || skybot_job_id.read().is_some() {
            return;
        }
        if let Some(Ok(Some(record))) = &*skybot_history_resource.read() {
            skybot_view.set(Some(SkybotJobView {
                status: JobStatus::Done,
                total: record.hits.len(),
                processed: record.hits.len(),
                hits: record.hits.clone(),
                logs: Vec::new(),
                error: None,
            }));
        }
    });
    use_effect(move || {
        if cnd_view.read().is_some() || cnd_job_id.read().is_some() {
            return;
        }
        if let Some(Ok(Some(record))) = &*cnd_history_resource.read() {
            cnd_view.set(Some(CndJobView {
                status: JobStatus::Done,
                total: record.hits.len(),
                processed: record.hits.len(),
                hits: record.hits.clone(),
                logs: Vec::new(),
                error: None,
            }));
        }
    });

    // Poll a running Skybot search job until it's no longer `Running` — same
    // idiom as `orbit_fit_page`'s fit-status poll, just on a shorter
    // interval so matches visibly trickle onto the plot.
    use_effect(move || {
        let Some(id) = *skybot_job_id.read() else {
            return;
        };
        spawn(async move {
            loop {
                match get_skybot_job_status(id).await {
                    Ok(view) => {
                        let running = matches!(view.status, JobStatus::Running);
                        skybot_view.set(Some(view));
                        if !running {
                            // The job just persisted its own attempt; reload
                            // the history so the last-checked date/delta
                            // shown next to the button reflects it right
                            // away instead of going stale until next visit.
                            skybot_history_resource.restart();
                            break;
                        }
                    }
                    Err(_) => break,
                }
                sleep_ms(CROSS_MATCH_POLL_INTERVAL_MS).await;
            }
        });
    });

    // Same poll idiom for the CND check job.
    use_effect(move || {
        let Some(id) = *cnd_job_id.read() else {
            return;
        };
        spawn(async move {
            loop {
                match get_cnd_job_status(id).await {
                    Ok(view) => {
                        let running = matches!(view.status, JobStatus::Running);
                        cnd_view.set(Some(view));
                        if !running {
                            cnd_history_resource.restart();
                            break;
                        }
                    }
                    Err(_) => break,
                }
                sleep_ms(CROSS_MATCH_POLL_INTERVAL_MS).await;
            }
        });
    });

    let skybot_last_queried_at: Option<String> = match &*skybot_history_resource.read() {
        Some(Ok(Some(record))) => Some(record.queried_at.clone()),
        _ => None,
    };
    let skybot_delta_days: Option<f64> = match &*skybot_history_resource.read() {
        Some(Ok(Some(record))) => Some(record.delta_days),
        _ => None,
    };
    let cnd_last_queried_at: Option<String> = match &*cnd_history_resource.read() {
        Some(Ok(Some(record))) => Some(record.queried_at.clone()),
        _ => None,
    };
    let cnd_delta_days: Option<f64> = match &*cnd_history_resource.read() {
        Some(Ok(Some(record))) => Some(record.delta_days),
        _ => None,
    };

    let skybot_hits: Vec<SkybotHit> = skybot_view
        .read()
        .as_ref()
        .map(|view| view.hits.clone())
        .unwrap_or_default();
    let skybot_running = matches!(
        skybot_view.read().as_ref().map(|view| view.status),
        Some(JobStatus::Running)
    );
    let skybot_processed = skybot_view.read().as_ref().map_or(0, |view| view.processed);
    let skybot_total = skybot_view.read().as_ref().map_or(0, |view| view.total);

    let cnd_hits: Vec<CndHit> = cnd_view
        .read()
        .as_ref()
        .map(|view| view.hits.clone())
        .unwrap_or_default();
    let cnd_running = matches!(
        cnd_view.read().as_ref().map(|view| view.status),
        Some(JobStatus::Running)
    );
    let cnd_processed = cnd_view.read().as_ref().map_or(0, |view| view.processed);
    let cnd_total = cnd_view.read().as_ref().map_or(0, |view| view.total);

    let launch_skybot_observations = observations.clone();
    let launch_skybot_lineage_id = lineage_id.clone();
    let launch_skybot = move |_: ()| {
        let points: Vec<SkybotQueryPoint> = launch_skybot_observations
            .iter()
            .enumerate()
            .map(|(source_index, obs)| SkybotQueryPoint {
                source_index,
                ra_deg: obs.ra.to_degrees(),
                dec_deg: obs.dec.to_degrees(),
                mjd_tt: obs.mjd_tt,
            })
            .collect();
        let radius_arcsec = skybot_radius();
        let lineage_designation = launch_skybot_lineage_id.clone();
        skybot_view.set(None);
        spawn(async move {
            if let Ok(id) = start_skybot_search(points, radius_arcsec, lineage_designation).await {
                skybot_job_id.set(Some(id));
            }
        });
    };

    let launch_cnd_observations = observations.clone();
    let launch_cnd_lineage_id = lineage_id.clone();
    let launch_cnd = move |_: ()| {
        let Some(branch_id) = branch_id else {
            return;
        };
        let points: Vec<CndQueryPoint> = launch_cnd_observations
            .iter()
            .enumerate()
            .map(|(source_index, obs)| CndQueryPoint {
                source_index,
                obs_id: obs.id,
                branch_id,
                ra_deg: obs.ra.to_degrees(),
                dec_deg: obs.dec.to_degrees(),
                mjd_tt: obs.mjd_tt,
                magnitude: obs.magnitude,
                filter: obs.filter,
                mpc_code_obs: obs.mpc_code_obs.clone(),
            })
            .collect();
        let time_separation_s = cnd_time_separation_s();
        let angle_separation_arcsec = cnd_angle_separation_arcsec();
        let lineage_designation = launch_cnd_lineage_id.clone();
        cnd_view.set(None);
        spawn(async move {
            if let Ok(id) = start_cnd_search(
                points,
                lineage_designation,
                time_separation_s,
                angle_separation_arcsec,
            )
            .await
            {
                cnd_job_id.set(Some(id));
            }
        });
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
                                button {
                                    class: "btn btn-sm btn-outline",
                                    r#type: "button",
                                    onclick: move |_| ades_modal_open.set(true),
                                    "Export ADES"
                                }
                            }
                            match lineage_view() {
                                LineageView::Trajectory => rsx! {
                                    TrajectoryPlot {
                                        observations: observations.clone(),
                                        replay: replay.clone(),
                                        x_axis_unit: x_axis_unit(),
                                        skybot_hits: skybot_hits.clone(),
                                        skybot_running,
                                        skybot_processed,
                                        skybot_total,
                                        skybot_radius_arcsec: skybot_radius(),
                                        skybot_last_queried_at: skybot_last_queried_at.clone(),
                                        skybot_delta_days,
                                        on_skybot_radius_change: move |v| skybot_radius.set(v),
                                        on_skybot_search: launch_skybot,
                                        cnd_hits: cnd_hits.clone(),
                                        cnd_running,
                                        cnd_processed,
                                        cnd_total,
                                        cnd_time_separation_s: cnd_time_separation_s(),
                                        cnd_angle_separation_arcsec: cnd_angle_separation_arcsec(),
                                        cnd_last_queried_at: cnd_last_queried_at.clone(),
                                        cnd_delta_days,
                                        on_cnd_time_separation_change: move |v| cnd_time_separation_s.set(v),
                                        on_cnd_angle_separation_change: move |v| cnd_angle_separation_arcsec.set(v),
                                        on_cnd_search: launch_cnd,
                                        on_toggle_results_panel: move |_| cross_match_panel_open.set(!cross_match_panel_open()),
                                    }
                                },
                                LineageView::LightCurve => rsx! {
                                    LightCurvePlot { observations: observations.clone(), x_axis_unit: x_axis_unit() }
                                },
                                LineageView::ThreeD => rsx! {
                                    LineageOrbit3DTab { lineage_id: lineage_id.clone() }
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

            CrossMatchPanel {
                skybot_hits,
                cnd_hits,
                open: cross_match_panel_open(),
                on_close: move |_| cross_match_panel_open.set(false),
            }

            AdesExportModal {
                lineage_designation: lineage_id.clone(),
                mpc_codes: mpc_codes.clone(),
                open: ades_modal_open(),
                on_close: move |_| ades_modal_open.set(false),
            }
        }
    }
}

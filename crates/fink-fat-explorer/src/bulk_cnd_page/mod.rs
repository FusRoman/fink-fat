mod bulk_cnd_progress;

use dioxus::prelude::*;

use crate::bulk_cnd::run::start_bulk_cnd_check;
use crate::bulk_cnd::status::get_bulk_cnd_job_status;
use crate::bulk_cnd::{BulkCndJobView, JobStatus};
use crate::cnd_search::{
    clamp_angle_separation_arcsec, clamp_time_separation_s, DEFAULT_ANGLE_SEPARATION_ARCSEC,
    DEFAULT_TIME_SEPARATION_S, MAX_ANGLE_SEPARATION_ARCSEC, MAX_TIME_SEPARATION_S,
    MIN_ANGLE_SEPARATION_ARCSEC, MIN_TIME_SEPARATION_S,
};
use crate::sleep_ms;

use bulk_cnd_progress::BulkCndProgress;

/// Milliseconds between polls of the running bulk CND job's status — same
/// cadence as `bulk_orbit_fit_page`'s poll loop.
const POLL_INTERVAL_MS: u64 = 1000;

/// Page for checking every branch with a converged n-body fit against the
/// MPC's Check Near-Duplicates API, to flag trajectories that are likely
/// already-published observations rather than new discoveries. Results are
/// written straight into `cnd_queries` (one row per branch); the lineage
/// page reads them back to show a "last checked" date and overlay any
/// matches on the trajectory plot.
#[component]
pub fn BulkCndPage() -> Element {
    let mut time_separation_s = use_signal(|| DEFAULT_TIME_SEPARATION_S);
    let mut angle_separation_arcsec = use_signal(|| DEFAULT_ANGLE_SEPARATION_ARCSEC);

    let mut job_id = use_signal(|| None::<u64>);
    let mut job_view = use_signal(|| None::<BulkCndJobView>);
    let mut launch_error = use_signal(|| None::<String>);

    use_effect(move || {
        let Some(id) = *job_id.read() else {
            return;
        };
        spawn(async move {
            loop {
                match get_bulk_cnd_job_status(id).await {
                    Ok(view) => {
                        let running = matches!(view.status, JobStatus::Running);
                        job_view.set(Some(view));
                        if !running {
                            break;
                        }
                    }
                    Err(e) => {
                        launch_error.set(Some(format!("Failed to poll bulk CND status: {e}")));
                        break;
                    }
                }
                sleep_ms(POLL_INTERVAL_MS).await;
            }
        });
    });

    let is_running = matches!(
        job_view.read().as_ref().map(|v| v.status),
        Some(JobStatus::Running)
    );

    let launch = move |_| {
        let time_separation_s = time_separation_s();
        let angle_separation_arcsec = angle_separation_arcsec();
        launch_error.set(None);
        job_view.set(None);
        spawn(async move {
            match start_bulk_cnd_check(time_separation_s, angle_separation_arcsec).await {
                Ok(id) => job_id.set(Some(id)),
                Err(e) => {
                    launch_error.set(Some(format!("Failed to start the bulk CND check: {e}")))
                }
            }
        });
    };

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                Link {
                    to: crate::Route::Home {},
                    class: "link link-hover text-sm",
                    "← Back to home"
                }
            }

            div { class: "alert alert-success",
                "Checks every branch with a converged n-body fit against the MPC's Check \
                 Near-Duplicates API, in sequential batches, and writes one result row per \
                 branch straight into the cnd_queries table. This can take a while."
            }

            if let Some(message) = &*launch_error.read() {
                div { class: "alert alert-error", "{message}" }
            }

            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body gap-3",
                    div { class: "flex flex-wrap items-end gap-4",
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Time separation"
                            input {
                                r#type: "range",
                                class: "range range-xs w-32",
                                min: "{MIN_TIME_SEPARATION_S}",
                                max: "{MAX_TIME_SEPARATION_S}",
                                step: "0.5",
                                disabled: is_running,
                                value: "{time_separation_s}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        time_separation_s.set(clamp_time_separation_s(v));
                                    }
                                },
                            }
                            span { "{time_separation_s:.1}s" }
                        }
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Angle separation"
                            input {
                                r#type: "range",
                                class: "range range-xs w-32",
                                min: "{MIN_ANGLE_SEPARATION_ARCSEC}",
                                max: "{MAX_ANGLE_SEPARATION_ARCSEC}",
                                step: "0.5",
                                disabled: is_running,
                                value: "{angle_separation_arcsec}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        angle_separation_arcsec.set(clamp_angle_separation_arcsec(v));
                                    }
                                },
                            }
                            span { "{angle_separation_arcsec:.1}\"" }
                        }
                        button {
                            class: "btn btn-primary btn-sm",
                            r#type: "button",
                            disabled: is_running,
                            onclick: launch,
                            "Start bulk CND check"
                        }
                    }
                }
            }

            if let Some(view) = &*job_view.read() {
                BulkCndProgress {
                    total_branches: view.total_branches,
                    total_observations: view.total_observations,
                    processed_observations: view.processed_observations,
                    branches_with_match: view.branches_with_match,
                    elapsed_secs: view.elapsed_secs,
                    logs: view.logs.clone(),
                    failed_job: matches!(view.status, JobStatus::Failed),
                    error: view.error.clone(),
                }
            }
        }
    }
}

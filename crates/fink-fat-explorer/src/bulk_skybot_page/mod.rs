mod bulk_skybot_progress;

use dioxus::prelude::*;

use crate::bulk_skybot::run::start_bulk_skybot_search;
use crate::bulk_skybot::status::{
    get_current_skybot_bulk_job_status, request_kill_skybot_bulk_job,
};
use crate::bulk_skybot::{SkybotBulkJobStatus, SkybotBulkJobView};
use crate::skybot_search::{MAX_RADIUS_ARCSEC, MIN_RADIUS_ARCSEC};
use crate::sleep_ms;

use bulk_skybot_progress::BulkSkybotProgress;

/// Milliseconds between polls of the current bulk Skybot job — same cadence
/// as `bulk_cnd_page`'s poll loop. Unlike every other bulk/single job on
/// this page's sibling pages, this poll has no job id to pass: there's at
/// most one bulk Skybot job ever, and `get_current_skybot_bulk_job_status`
/// always answers "what's the current one" straight from Postgres — which
/// is exactly what makes returning to this page after navigating away (or
/// after the whole server restarted) show the right thing immediately.
const POLL_INTERVAL_MS: u64 = 1000;

/// Page for bulk-checking every branch with a converged n-body fit against
/// Skybot. Unlike `bulk_cnd_page` (seconds to minutes), this can run for
/// hours at real dataset sizes, so its job is durable in Postgres rather
/// than process memory — see `crate::bulk_skybot`'s module doc comment —
/// and this page reflects that: no job id, a Stop button while running, and
/// the current/last job's state is fetched fresh on every mount rather than
/// assumed lost the moment the user navigates away.
#[component]
pub fn BulkSkybotPage() -> Element {
    let mut radius = use_signal(|| 10.0_f64);
    let mut job_view = use_signal(|| None::<SkybotBulkJobView>);
    let mut launch_error = use_signal(|| None::<String>);
    let mut killing = use_signal(|| false);
    let mut loaded_once = use_signal(|| false);

    // Poll for as long as a job is running — including one already running
    // when this page is first opened, not just one this page itself
    // started.
    use_effect(move || {
        spawn(async move {
            loop {
                match get_current_skybot_bulk_job_status().await {
                    Ok(view) => {
                        let running = matches!(
                            view.as_ref().map(|v| v.status),
                            Some(SkybotBulkJobStatus::Running)
                        );
                        // Keep the radius control in sync with whatever job
                        // is actually running (or most recently ran) —
                        // otherwise it silently keeps showing its default
                        // value even while a job started at a different
                        // radius (e.g. from another browser tab, or before a
                        // server restart) is the one actually in progress,
                        // which is confusing since the two numbers can
                        // legitimately disagree.
                        if let Some(view) = &view {
                            radius.set(view.radius_arcsec);
                        }
                        job_view.set(view);
                        loaded_once.set(true);
                        killing.set(false);
                        if !running {
                            break;
                        }
                    }
                    Err(_) => break,
                }
                sleep_ms(POLL_INTERVAL_MS).await;
            }
        });
    });

    let is_running = matches!(
        job_view.read().as_ref().map(|v| v.status),
        Some(SkybotBulkJobStatus::Running)
    );

    let launch = move |_| {
        let radius_arcsec = radius();
        launch_error.set(None);
        spawn(async move {
            match start_bulk_skybot_search(radius_arcsec).await {
                Ok(()) => {
                    // Kick the poll loop again immediately rather than
                    // waiting up to POLL_INTERVAL_MS for the first update.
                    if let Ok(view) = get_current_skybot_bulk_job_status().await {
                        job_view.set(view);
                    }
                }
                Err(e) => launch_error.set(Some(format!("Failed to start: {e}"))),
            }
        });
    };

    let kill = move |_| {
        killing.set(true);
        spawn(async move {
            let _ = request_kill_skybot_bulk_job().await;
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

            div { class: "alert alert-info",
                "Checks every branch with a converged n-body fit against Skybot, one \
                 observation at a time (up to 10 requests in flight), prioritizing \
                 never-checked observations first, then the oldest-checked. At real \
                 dataset sizes this can take hours — it keeps running if you leave this \
                 page, and resumes where it left off if stopped or interrupted."
            }

            if let Some(message) = &*launch_error.read() {
                div { class: "alert alert-error", "{message}" }
            }

            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body gap-3",
                    div { class: "flex flex-wrap items-end gap-4",
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Radius"
                            input {
                                r#type: "range",
                                class: "range range-xs w-32",
                                min: "{MIN_RADIUS_ARCSEC}",
                                max: "{MAX_RADIUS_ARCSEC}",
                                step: "1",
                                disabled: is_running,
                                value: "{radius}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        radius.set(v);
                                    }
                                },
                            }
                            span { "{radius():.0}\"" }
                        }
                        button {
                            class: "btn btn-primary btn-sm",
                            r#type: "button",
                            disabled: is_running,
                            onclick: launch,
                            "Start bulk Skybot check"
                        }
                    }
                }
            }

            if let Some(view) = &*job_view.read() {
                BulkSkybotProgress {
                    status: view.status,
                    radius_arcsec: view.radius_arcsec,
                    total_observations: view.total_observations,
                    processed_observations: view.processed_observations,
                    matched_observations: view.matched_observations,
                    error: view.error.clone(),
                    logs: view.logs.clone(),
                    on_kill: kill,
                    killing: killing(),
                }
            } else if loaded_once() {
                p { class: "text-sm opacity-60", "No bulk Skybot check has ever been run." }
            }
        }
    }
}

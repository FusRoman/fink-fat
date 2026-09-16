mod bulk_fit_progress;

use dioxus::prelude::*;

use crate::bulk_orbit_fit::run::start_bulk_orbit_fit;
use crate::bulk_orbit_fit::status::get_bulk_orbit_fit_job_status;
use crate::bulk_orbit_fit::{BulkOrbitFitJobView, JobStatus};
use crate::orbit_fit::OrbitFitParams;
use crate::orbit_fit_page::FitParamsForm;
use crate::sleep_ms;

use bulk_fit_progress::BulkFitProgress;

/// Milliseconds between polls of the running bulk fit job's status — a bulk
/// fit runs far longer than a single-lineage one, so this can be leisurely
/// (same cadence as the homepage snapshot rebuild's own poll loop).
const POLL_INTERVAL_MS: u64 = 1000;

/// Page for fitting *every* eligible trajectory (>= 3 observations, >= 0.25
/// day baseline) with `outfit`'s Gauss IOD + n-body differential correction,
/// run fully independently per branch (no Kalman seed) and in parallel —
/// unlike `orbit_fit_page::OrbitFitPage`, which fits a single lineage seeded
/// from its current Kalman orbit. Reuses that page's `FitParamsForm`
/// unmodified; results are written straight to `orbit_fits` rather than
/// shown here.
#[component]
pub fn BulkOrbitFitPage() -> Element {
    let params = use_signal(OrbitFitParams::default);

    let mut job_id = use_signal(|| None::<u64>);
    let mut job_view = use_signal(|| None::<BulkOrbitFitJobView>);
    let mut launch_error = use_signal(|| None::<String>);

    use_effect(move || {
        let Some(id) = *job_id.read() else {
            return;
        };
        spawn(async move {
            loop {
                match get_bulk_orbit_fit_job_status(id).await {
                    Ok(view) => {
                        let running = matches!(view.status, JobStatus::Running);
                        job_view.set(Some(view));
                        if !running {
                            break;
                        }
                    }
                    Err(e) => {
                        launch_error.set(Some(format!("Failed to poll bulk fit status: {e}")));
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
        let fit_params = params.read().clone();
        launch_error.set(None);
        job_view.set(None);
        spawn(async move {
            match start_bulk_orbit_fit(fit_params).await {
                Ok(id) => job_id.set(Some(id)),
                Err(e) => launch_error.set(Some(format!("Failed to start the bulk fit: {e}"))),
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

            div { class: "alert alert-info",
                "Fits every eligible trajectory (branch) independently — Gauss IOD + \
                 n-body differential correction, no Kalman seed — and writes each \
                 result straight into the orbit_fits table. This can take a while."
            }

            if let Some(message) = &*launch_error.read() {
                div { class: "alert alert-error", "{message}" }
            }

            FitParamsForm {
                params,
                launch_disabled: is_running,
                on_launch: launch,
            }

            if let Some(view) = &*job_view.read() {
                BulkFitProgress {
                    total: view.total,
                    processed: view.processed,
                    succeeded: view.succeeded,
                    failed: view.failed,
                    elapsed_secs: view.elapsed_secs,
                    logs: view.logs.clone(),
                    failed_job: matches!(view.status, JobStatus::Failed),
                    error: view.error.clone(),
                }
            }
        }
    }
}

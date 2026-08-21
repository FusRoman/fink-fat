mod fit_params_form;
mod fit_progress;
mod fit_result;
mod help_tooltip;
mod residuals_plot;
mod selectable_observations_table;
mod x_axis;

use std::collections::HashSet;

use dioxus::prelude::*;

use crate::lineage_page::observations_table::{get_lineage_observations, ObservationRow};
use crate::orbit_fit::latest::get_latest_orbit_fit_result;
use crate::orbit_fit::run::start_orbit_fit;
use crate::orbit_fit::status::get_orbit_fit_job_status;
use crate::orbit_fit::{
    JobStatus, OrbitFitJobView, OrbitFitParams, OrbitFitResult, MIN_BASELINE_DAYS, MIN_OBSERVATIONS,
};

use fit_params_form::FitParamsForm;
use fit_progress::FitProgress;
use fit_result::FitResult;
use selectable_observations_table::SelectableObservationsTable;

/// Milliseconds between polls of the running fit job's status.
const POLL_INTERVAL_MS: u64 = 700;

async fn sleep_ms(ms: u64) {
    #[cfg(target_arch = "wasm32")]
    {
        gloo_timers::future::TimeoutFuture::new(ms as u32).await;
    }
    // `tokio` is only pulled in behind the `server` feature (it doesn't
    // build for wasm32) — this branch only exists for native builds that
    // also enable `server` (i.e. `dx serve` running the SSR/liveview
    // server). A native build with neither `wasm32` nor `server" has no
    // client to poll from, so there's nothing useful to sleep for.
    #[cfg(all(not(target_arch = "wasm32"), feature = "server"))]
    {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    }
}

/// Page for fitting a lineage's orbit with `outfit`'s n-body least-squares
/// pipeline instead of the production Kalman filter: pick which
/// observations participate, tune the fit parameters, launch it, watch it
/// run, then inspect the resulting orbit.
#[component]
pub fn OrbitFitPage(lineage_id: String) -> Element {
    let observations_lineage_id = lineage_id.clone();
    let observations_resource = use_resource(use_reactive!(|(observations_lineage_id,)| {
        get_lineage_observations(observations_lineage_id)
    }));

    let observations: Vec<ObservationRow> = match &*observations_resource.read() {
        Some(Ok(rows)) => rows.clone(),
        _ => Vec::new(),
    };

    let mut selected = use_signal(HashSet::<i64>::new);
    let mut initialized = use_signal(|| false);
    if !*initialized.read() && !observations.is_empty() {
        selected.set(observations.iter().map(|o| o.id).collect());
        initialized.set(true);
    }

    let params = use_signal(OrbitFitParams::default);

    let mut job_id = use_signal(|| None::<u64>);
    let mut job_view = use_signal(|| None::<OrbitFitJobView>);
    let mut launch_error = use_signal(|| None::<String>);

    // Poll the running job's status until it's no longer `Running`.
    use_effect(move || {
        let Some(id) = *job_id.read() else {
            return;
        };
        spawn(async move {
            loop {
                match get_orbit_fit_job_status(id).await {
                    Ok(view) => {
                        let running = matches!(view.status, JobStatus::Running);
                        job_view.set(Some(view));
                        if !running {
                            break;
                        }
                    }
                    Err(e) => {
                        launch_error.set(Some(format!("Failed to poll fit status: {e}")));
                        break;
                    }
                }
                sleep_ms(POLL_INTERVAL_MS).await;
            }
        });
    });

    // Land directly on the lineage's last fit, if any, instead of the form.
    let latest_lineage_id = lineage_id.clone();
    let latest_resource = use_resource(use_reactive!(|(latest_lineage_id,)| {
        get_latest_orbit_fit_result(latest_lineage_id)
    }));

    let mut show_form = use_signal(|| false);
    let mut show_form_initialized = use_signal(|| false);
    use_effect(move || {
        if *show_form_initialized.read() {
            return;
        }
        if let Some(Ok(latest)) = &*latest_resource.read() {
            if latest.is_none() {
                show_form.set(true);
            }
            show_form_initialized.set(true);
        }
    });

    // Collapse the form back down once a freshly-launched fit completes —
    // the new result takes over at the top of the page.
    use_effect(move || {
        if let Some(view) = &*job_view.read() {
            if matches!(view.status, JobStatus::Done) && view.result.is_some() {
                show_form.set(false);
            }
        }
    });

    let job_result: Option<OrbitFitResult> = job_view
        .read()
        .as_ref()
        .filter(|v| matches!(v.status, JobStatus::Done))
        .and_then(|v| v.result.clone());
    let displayed_result: Option<OrbitFitResult> = job_result.or_else(|| {
        latest_resource
            .read()
            .as_ref()
            .and_then(|r| r.as_ref().ok().cloned().flatten())
    });

    let selected_ids: Vec<i64> = selected.read().iter().copied().collect();
    let n_selected = selected_ids.len();
    let baseline_days = {
        let selected_set = selected.read();
        let mjds: Vec<f64> = observations
            .iter()
            .filter(|o| selected_set.contains(&o.id))
            .map(|o| o.mjd_tt)
            .collect();
        match (
            mjds.iter().cloned().fold(f64::INFINITY, f64::min),
            mjds.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ) {
            (min, max) if min.is_finite() && max.is_finite() => max - min,
            _ => 0.0,
        }
    };

    let guard_rail_message = if n_selected < MIN_OBSERVATIONS {
        Some(format!(
            "Select at least {MIN_OBSERVATIONS} observations (currently {n_selected})."
        ))
    } else if baseline_days < MIN_BASELINE_DAYS {
        Some(format!(
            "The selected observations span only {baseline_days:.3} days; at least \
             {MIN_BASELINE_DAYS} days are needed for a reliable fit."
        ))
    } else {
        None
    };

    let is_running = matches!(
        job_view.read().as_ref().map(|v| v.status),
        Some(JobStatus::Running)
    );

    let launch_lineage_id = lineage_id.clone();
    let launch = move |_| {
        let lineage_id = launch_lineage_id.clone();
        let observation_ids = selected_ids.clone();
        let fit_params = params.read().clone();
        launch_error.set(None);
        job_view.set(None);
        spawn(async move {
            match start_orbit_fit(lineage_id, observation_ids, fit_params).await {
                Ok(id) => job_id.set(Some(id)),
                Err(e) => launch_error.set(Some(format!("Failed to start the fit: {e}"))),
            }
        });
    };

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                Link {
                    to: crate::Route::LineagePage {
                        lineage_id: lineage_id.clone(),
                    },
                    class: "link link-hover text-sm",
                    "← Back to lineage"
                }
            }

            if let Some(message) = &guard_rail_message {
                div { class: "alert alert-warning", "{message}" }
            }
            if let Some(message) = &*launch_error.read() {
                div { class: "alert alert-error", "{message}" }
            }

            if !*show_form_initialized.read() {
                div { class: "flex justify-center py-10",
                    span { class: "loading loading-spinner" }
                }
            }

            if let Some(result) = &displayed_result {
                FitResult { lineage_id: lineage_id.clone(), result: result.clone() }
                if !*show_form.read() {
                    div { class: "flex justify-center",
                        button {
                            class: "btn btn-sm btn-outline",
                            r#type: "button",
                            onclick: move |_| show_form.set(true),
                            "Redo fit"
                        }
                    }
                }
            }

            if *show_form.read() {
                FitParamsForm {
                    params,
                    launch_disabled: guard_rail_message.is_some() || is_running,
                    on_launch: launch,
                }

                div { class: "collapse collapse-arrow bg-base-100 shadow-sm",
                    input { r#type: "checkbox" }
                    div { class: "collapse-title font-medium",
                        "Observations ({n_selected}/{observations.len()} selected) — expand to include/exclude observations"
                    }
                    div { class: "collapse-content",
                        match &*observations_resource.read() {
                            Some(Ok(_)) => rsx! {
                                SelectableObservationsTable { observations: observations.clone(), selected }
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

                match &*job_view.read() {
                    Some(view) => {
                        match view.status {
                            JobStatus::Running => rsx! {
                                FitProgress { logs: view.logs.clone() }
                            },
                            JobStatus::Failed => rsx! {
                                div { class: "alert alert-error", "Fit failed: {view.error.clone().unwrap_or_default()}" }
                                FitProgress { logs: view.logs.clone(), failed: true, error: view.error.clone() }
                            },
                            JobStatus::Done if view.result.is_none() => rsx! {
                                div { class: "alert alert-error", "Fit completed but returned no result." }
                            },
                            JobStatus::Done => rsx! {},
                        }
                    }
                    None => rsx! {},
                }
            }
        }
    }
}

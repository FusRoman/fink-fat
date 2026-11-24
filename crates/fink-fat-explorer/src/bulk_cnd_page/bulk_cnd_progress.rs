use dioxus::prelude::*;

/// Progress feedback for a running bulk CND check: a daisyUI progress bar
/// over processed/total observations (batches are observation-sized, not
/// branch-sized — see `crate::bulk_cnd::BulkCndJobView`), how many branches
/// got at least one match so far, an ETA extrapolated from elapsed time, and
/// the job's coarse log. Same shape as
/// `bulk_orbit_fit_page::BulkFitProgress`, kept as its own small component
/// (not a shared generic one) with CND-appropriate copy — this codebase
/// already has one dedicated progress component per bulk job rather than a
/// shared abstraction (`orbit_fit_page::FitProgress` vs
/// `bulk_orbit_fit_page::BulkFitProgress`).
#[component]
pub fn BulkCndProgress(
    total_branches: usize,
    total_observations: usize,
    processed_observations: usize,
    branches_with_match: usize,
    elapsed_secs: f64,
    logs: Vec<String>,
    #[props(default = false)] failed_job: bool,
    #[props(default = None)] error: Option<String>,
) -> Element {
    let eta_secs = if processed_observations > 0 && processed_observations < total_observations {
        elapsed_secs / processed_observations as f64
            * (total_observations - processed_observations) as f64
    } else {
        0.0
    };

    let text = match (failed_job, &error) {
        (true, Some(message)) => format!("{}\nFailed: {message}", logs.join("\n")),
        _ => logs.join("\n"),
    };
    let n_logs = logs.len();

    use_effect(use_reactive!(|(n_logs,)| {
        let _ = n_logs;
        document::eval(
            "const el = document.getElementById('bulk-cnd-log');\
             if (el) { el.scrollTop = el.scrollHeight; }",
        );
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-3",
                div { class: "flex items-center gap-3",
                    if failed_job {
                        span { class: "flex items-center justify-center w-8 h-8 rounded-full bg-error/20 text-error text-xl font-bold",
                            "✗"
                        }
                        span { class: "font-medium", "Bulk CND check failed" }
                    } else if processed_observations >= total_observations && total_observations > 0 {
                        span { class: "flex items-center justify-center w-8 h-8 rounded-full bg-success/20 text-success text-xl font-bold",
                            "✓"
                        }
                        span { class: "font-medium", "Bulk CND check finished" }
                    } else {
                        span { class: "loading loading-spinner loading-lg" }
                        span { class: "font-medium", "Checking every converged trajectory against the MPC..." }
                    }
                }

                progress {
                    class: "progress progress-primary w-full",
                    value: "{processed_observations}",
                    max: "{total_observations.max(1)}",
                }
                div { class: "text-sm opacity-80",
                    "{processed_observations}/{total_observations} observations across {total_branches} branches — {branches_with_match} branch(es) with a match so far"
                    if !failed_job && processed_observations < total_observations {
                        " — ETA {eta_secs as u64}s"
                    }
                }

                textarea {
                    id: "bulk-cnd-log",
                    class: "textarea textarea-bordered font-mono text-xs h-64 w-full",
                    readonly: true,
                    value: "{text}",
                }
            }
        }
    }
}

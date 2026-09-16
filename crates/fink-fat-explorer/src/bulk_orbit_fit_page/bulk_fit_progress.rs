use dioxus::prelude::*;

/// Progress feedback for a running bulk orbit fit: a daisyUI progress bar
/// over processed/total branches, succeeded/failed counts, an ETA
/// extrapolated from the elapsed time so far, and the pipeline's own coarse
/// log (same rationale as `orbit_fit_page::FitProgress` — `outfit` doesn't
/// expose per-branch progress hooks).
#[component]
pub fn BulkFitProgress(
    total: usize,
    processed: usize,
    succeeded: usize,
    failed: usize,
    elapsed_secs: f64,
    logs: Vec<String>,
    #[props(default = false)] failed_job: bool,
    #[props(default = None)] error: Option<String>,
) -> Element {
    let eta_secs = if processed > 0 && processed < total {
        elapsed_secs / processed as f64 * (total - processed) as f64
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
            "const el = document.getElementById('bulk-orbit-fit-log');\
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
                        span { class: "font-medium", "Bulk fit failed" }
                    } else if processed >= total && total > 0 {
                        span { class: "flex items-center justify-center w-8 h-8 rounded-full bg-success/20 text-success text-xl font-bold",
                            "✓"
                        }
                        span { class: "font-medium", "Bulk fit finished" }
                    } else {
                        span { class: "loading loading-spinner loading-lg" }
                        span { class: "font-medium", "Fitting every eligible trajectory..." }
                    }
                }

                progress {
                    class: "progress progress-primary w-full",
                    value: "{processed}",
                    max: "{total.max(1)}",
                }
                div { class: "text-sm opacity-80",
                    "{processed}/{total} trajectories — {succeeded} fit, {failed} failed"
                    if !failed_job && processed < total {
                        " — ETA {eta_secs as u64}s"
                    }
                }

                textarea {
                    id: "bulk-orbit-fit-log",
                    class: "textarea textarea-bordered font-mono text-xs h-64 w-full",
                    readonly: true,
                    value: "{text}",
                }
            }
        }
    }
}

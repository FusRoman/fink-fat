use dioxus::prelude::*;

use crate::bulk_skybot::SkybotBulkJobStatus;

/// Renders the icon + headline for one [`SkybotBulkJobStatus`] — pulled out
/// as its own pure function (rather than inlined `match` arms in the
/// component body) so the status-to-headline mapping is unit-testable
/// without rendering anything.
///
/// # Return
///
/// `(icon_classes, headline)`.
fn status_headline(status: SkybotBulkJobStatus) -> (&'static str, &'static str) {
    match status {
        SkybotBulkJobStatus::Running => {
            ("", "Checking every converged trajectory against Skybot...")
        }
        SkybotBulkJobStatus::Done => ("bg-success/20 text-success", "Bulk Skybot check finished"),
        SkybotBulkJobStatus::Killed => ("bg-warning/20 text-warning", "Bulk Skybot check stopped"),
        SkybotBulkJobStatus::Interrupted => (
            "bg-warning/20 text-warning",
            "Bulk Skybot check was interrupted by a server restart",
        ),
        SkybotBulkJobStatus::Failed => ("bg-error/20 text-error", "Bulk Skybot check failed"),
    }
}

/// Progress feedback for the bulk Skybot job. Unlike
/// `bulk_cnd_page::BulkCndProgress`, this job's own scale (potentially hours,
/// tens of thousands of observations) means it can be resumed or killed
/// across many separate page visits, so this component also shows a kill
/// button while running and distinguishes `Killed`/`Interrupted` from a
/// plain success/failure — kept as its own small component rather than a
/// shared one with `BulkCndProgress`, matching this codebase's existing
/// one-progress-component-per-bulk-job precedent
/// (`orbit_fit_page::FitProgress` vs `bulk_orbit_fit_page::BulkFitProgress`).
#[component]
pub fn BulkSkybotProgress(
    status: SkybotBulkJobStatus,
    radius_arcsec: f64,
    total_observations: i64,
    processed_observations: i64,
    matched_observations: i64,
    error: Option<String>,
    logs: Vec<String>,
    on_kill: EventHandler<()>,
    #[props(default = false)] killing: bool,
) -> Element {
    let is_running = matches!(status, SkybotBulkJobStatus::Running);
    let (icon_class, headline) = status_headline(status);
    let text = match &error {
        Some(message) => format!("{}\nError: {message}", logs.join("\n")),
        None => logs.join("\n"),
    };
    let n_logs = logs.len();

    use_effect(use_reactive!(|(n_logs,)| {
        let _ = n_logs;
        document::eval(
            "const el = document.getElementById('bulk-skybot-log');\
             if (el) { el.scrollTop = el.scrollHeight; }",
        );
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-3",
                div { class: "flex items-center justify-between gap-3",
                    div { class: "flex items-center gap-3",
                        if is_running {
                            span { class: "loading loading-spinner loading-lg" }
                        } else {
                            span { class: "flex items-center justify-center w-8 h-8 rounded-full text-xl font-bold {icon_class}",
                                match status {
                                    SkybotBulkJobStatus::Done => "✓",
                                    SkybotBulkJobStatus::Failed => "✗",
                                    _ => "■",
                                }
                            }
                        }
                        span { class: "font-medium", "{headline}" }
                    }
                    if is_running {
                        button {
                            class: "btn btn-sm btn-outline btn-error",
                            r#type: "button",
                            disabled: killing,
                            onclick: move |_| on_kill.call(()),
                            if killing {
                                span { class: "loading loading-spinner loading-xs" }
                                "Stopping"
                            } else {
                                "Stop"
                            }
                        }
                    }
                }

                progress {
                    class: "progress progress-primary w-full",
                    value: "{processed_observations}",
                    max: "{total_observations.max(1)}",
                }
                div { class: "text-sm opacity-80",
                    "{processed_observations}/{total_observations} observations checked (radius {radius_arcsec:.0}\") — {matched_observations} match(es) found"
                }

                textarea {
                    id: "bulk-skybot-log",
                    class: "textarea textarea-bordered font-mono text-xs h-48 w-full",
                    readonly: true,
                    value: "{text}",
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_headline_distinguishes_killed_from_interrupted() {
        assert_ne!(
            status_headline(SkybotBulkJobStatus::Killed).1,
            status_headline(SkybotBulkJobStatus::Interrupted).1
        );
    }

    #[test]
    fn status_headline_distinguishes_every_status() {
        let statuses = [
            SkybotBulkJobStatus::Running,
            SkybotBulkJobStatus::Done,
            SkybotBulkJobStatus::Killed,
            SkybotBulkJobStatus::Interrupted,
            SkybotBulkJobStatus::Failed,
        ];
        let headlines: Vec<&str> = statuses.iter().map(|s| status_headline(*s).1).collect();
        let mut unique = headlines.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), headlines.len());
    }
}

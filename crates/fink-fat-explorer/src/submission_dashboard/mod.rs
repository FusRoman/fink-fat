//! The "Submission" page: reached from the homepage's navbar button (next
//! to "🔭 Cross-match"), it lets the user copy the exact `fink-fat submit`
//! command for every currently-eligible, not-yet-submitted lineage (or
//! download the matching CSV), and shows every past submission attempt with
//! a per-row "Refresh status" action.

mod data;

use dioxus::prelude::*;

use crate::submitter_config_form::SubmitterConfigFields;
use data::{
    get_submission_candidates, get_submission_history, refresh_submission_status,
    SubmissionCandidate, SubmissionRow,
};

/// Above this many eligible lineages, the generated command switches from a
/// literal `--lineages a,b,c` list to `--csv eligible_lineages.csv` (paired
/// with the "Download CSV" button) — long enough to type/paste comfortably,
/// short enough that a handful of lineages don't need a separate file.
const MAX_INLINE_LINEAGES: usize = 5;

/// daisyUI badge color class for one `mpc_submissions.verdict` value.
fn verdict_badge_class(verdict: &str) -> &'static str {
    match verdict {
        "accepted" => "badge-success",
        "pending" => "badge-info",
        "rejected" | "error" => "badge-error",
        _ => "badge-ghost",
    }
}

/// Builds the copy-pastable `fink-fat submit` command for a batch of
/// eligible lineage designations. Pure — the CSV-vs-inline-list choice and
/// exact flag set are worth testing without a browser.
///
/// # Arguments
/// * `lineage_designations` — every currently-eligible, unsubmitted lineage.
///
/// # Return
/// The full command text, ready to paste into a shell (`\`-continued across
/// lines) or hand to a clipboard-write call verbatim.
fn submit_command(lineage_designations: &[String]) -> String {
    let source_flag = if lineage_designations.is_empty() {
        "--lineages <FF...>".to_string()
    } else if lineage_designations.len() <= MAX_INLINE_LINEAGES {
        format!("--lineages {}", lineage_designations.join(","))
    } else {
        "--csv eligible_lineages.csv".to_string()
    };

    [
        "fink-fat submit \\".to_string(),
        format!("  {source_flag} \\"),
        "  --submitter-config submission.yaml \\".to_string(),
        "  --database-url $DATABASE_URL \\".to_string(),
        "  --endpoint production".to_string(),
    ]
    .join("\n")
}

/// Builds the `eligible_lineages.csv` text (one `lineage_designation`
/// column) for the "Download CSV" button. Pure.
///
/// # Arguments
/// * `lineage_designations` — every currently-eligible, unsubmitted lineage.
///
/// # Return
/// The CSV text, header included.
fn eligible_lineages_csv(lineage_designations: &[String]) -> String {
    let mut csv = String::from("lineage_designation\n");
    for designation in lineage_designations {
        csv.push_str(designation);
        csv.push('\n');
    }
    csv
}

#[component]
pub fn SubmissionDashboardPage() -> Element {
    let candidates_resource = use_resource(get_submission_candidates);
    let mut history = use_signal(Vec::<SubmissionRow>::new);
    let mut refreshing_id = use_signal(|| None::<i64>);
    let submitter_config = use_signal(crate::submitter_config_form::default_submitter_config);

    use_effect(move || {
        spawn(async move {
            if let Ok(rows) = get_submission_history().await {
                history.set(rows);
            }
        });
    });

    let candidates: Vec<SubmissionCandidate> = match &*candidates_resource.read() {
        Some(Ok(rows)) => rows.clone(),
        _ => Vec::new(),
    };
    let lineage_designations: Vec<String> = candidates
        .iter()
        .map(|c| c.lineage_designation.clone())
        .collect();
    let command = submit_command(&lineage_designations);

    let copy_command = {
        let command = command.clone();
        move |_| {
            let eval = document::eval(
                "const data = await dioxus.recv();
                 await navigator.clipboard.writeText(data.text);",
            );
            let _ = eval.send(serde_json::json!({ "text": command }));
        }
    };

    let download_csv = move |_| {
        let csv = eligible_lineages_csv(&lineage_designations);
        let eval = document::eval(
            "const data = await dioxus.recv();
             const blob = new Blob([data.csv], { type: 'text/csv' });
             const url = URL.createObjectURL(blob);
             const a = document.createElement('a');
             a.href = url;
             a.download = 'eligible_lineages.csv';
             document.body.appendChild(a);
             a.click();
             a.remove();
             URL.revokeObjectURL(url);",
        );
        let _ = eval.send(serde_json::json!({ "csv": csv }));
    };

    let download_submitter_config = move |_| {
        let Ok(yaml) = submitter_config().to_yaml() else {
            return;
        };
        let eval = document::eval(
            "const data = await dioxus.recv();
             const blob = new Blob([data.yaml], { type: 'text/yaml' });
             const url = URL.createObjectURL(blob);
             const a = document.createElement('a');
             a.href = url;
             a.download = 'submission.yaml';
             document.body.appendChild(a);
             a.click();
             a.remove();
             URL.revokeObjectURL(url);",
        );
        let _ = eval.send(serde_json::json!({ "yaml": yaml }));
    };

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                Link { to: crate::Route::Home {}, class: "link link-hover text-sm", "← Back to home" }
            }

            div { class: "stats shadow bg-base-100",
                div { class: "stat",
                    div { class: "stat-title", "Eligible, not yet submitted" }
                    div { class: "stat-value text-primary", "{candidates.len()}" }
                }
                div { class: "stat",
                    div { class: "stat-title", "Submitted (all time)" }
                    div { class: "stat-value", "{history.read().len()}" }
                }
            }

            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body gap-3",
                    h2 { class: "card-title", "Prepare a submission" }
                    match &*candidates_resource.read() {
                        None => rsx! {
                            div { class: "flex justify-center py-6",
                                span { class: "loading loading-spinner loading-md" }
                            }
                        },
                        Some(Err(e)) => rsx! {
                            div { class: "alert alert-error", "Failed to load candidates: {e}" }
                        },
                        Some(Ok(_)) if candidates.is_empty() => rsx! {
                            p { class: "text-sm opacity-60",
                                "No lineage is currently eligible (Well-sampled discovery / \
                                 Discovery) and unsubmitted."
                            }
                        },
                        Some(Ok(_)) => rsx! {
                            p { class: "text-sm opacity-70",
                                "{candidates.len()} lineage(s) ready to submit. Fill in a \
                                 submitter config (see `fink-fat submit --help`), then run:"
                            }
                            div { class: "mockup-code text-xs whitespace-pre",
                                pre { "data-prefix": "$", code { "{command}" } }
                            }
                            div { class: "flex gap-2",
                                button {
                                    class: "btn btn-sm btn-outline",
                                    r#type: "button",
                                    onclick: copy_command,
                                    "📋 Copy command"
                                }
                                button {
                                    class: "btn btn-sm btn-outline",
                                    r#type: "button",
                                    onclick: download_csv,
                                    "⬇ Download eligible_lineages.csv"
                                }
                            }
                        },
                    }
                }
            }

            div { class: "collapse collapse-arrow bg-base-100 shadow-sm",
                input { r#type: "checkbox" }
                div { class: "collapse-title font-semibold", "Generate submitter config (submission.yaml)" }
                div { class: "collapse-content",
                    div { class: "flex flex-col gap-3 pt-2",
                        p { class: "text-xs opacity-70",
                            "Fill in your submitter/telescope identity once, download it as \
                             `submission.yaml`, then pass it to `fink-fat submit \
                             --submitter-config submission.yaml`."
                        }
                        SubmitterConfigFields { config: submitter_config, on_change: move |_| {} }
                        button {
                            class: "btn btn-sm btn-outline self-start",
                            r#type: "button",
                            onclick: download_submitter_config,
                            "⬇ Download submission.yaml"
                        }
                    }
                }
            }

            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body p-0",
                    h2 { class: "card-title p-4 pb-0", "Submitted lineages" }
                    if history.read().is_empty() {
                        p { class: "p-6 text-sm opacity-60", "No submission recorded yet." }
                    } else {
                        div { class: "overflow-x-auto",
                            table { class: "table table-zebra table-sm",
                                thead {
                                    tr {
                                        th { "Lineage" }
                                        th { "Endpoint" }
                                        th { "Submission ID" }
                                        th { "Verdict" }
                                        th { "Submitted at" }
                                        th { "" }
                                    }
                                }
                                tbody {
                                    for row in history.read().iter() {
                                        tr { key: "{row.id}",
                                            td {
                                                Link {
                                                    to: crate::Route::LineagePage {
                                                        lineage_id: row.lineage_designation.clone(),
                                                    },
                                                    class: "link link-primary font-medium",
                                                    "{row.lineage_designation}"
                                                }
                                            }
                                            td {
                                                span {
                                                    class: if row.endpoint == "production" { "badge badge-sm badge-warning" } else { "badge badge-sm badge-ghost" },
                                                    "{row.endpoint}"
                                                }
                                            }
                                            td { class: "text-xs opacity-70 font-mono",
                                                {row.submission_id.clone().unwrap_or_else(|| "—".to_string())}
                                            }
                                            td {
                                                span {
                                                    class: "badge badge-sm {verdict_badge_class(&row.verdict)}",
                                                    "{row.verdict}"
                                                }
                                            }
                                            td { class: "text-xs opacity-70", "{row.submitted_at}" }
                                            td {
                                                button {
                                                    class: "btn btn-xs btn-outline",
                                                    r#type: "button",
                                                    disabled: refreshing_id() == Some(row.id),
                                                    onclick: {
                                                        let row_id = row.id;
                                                        move |_| {
                                                            refreshing_id.set(Some(row_id));
                                                            spawn(async move {
                                                                if let Ok(updated) = refresh_submission_status(row_id).await {
                                                                    history.with_mut(|rows| {
                                                                        if let Some(r) = rows.iter_mut().find(|r| r.id == row_id) {
                                                                            *r = updated;
                                                                        }
                                                                    });
                                                                }
                                                                refreshing_id.set(None);
                                                            });
                                                        }
                                                    },
                                                    if refreshing_id() == Some(row.id) {
                                                        span { class: "loading loading-spinner loading-xs" }
                                                    } else {
                                                        "Refresh"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_command_uses_inline_lineages_under_the_threshold() {
        let designations = vec!["FF2026abc".to_string(), "FF2026def".to_string()];
        let command = submit_command(&designations);
        assert!(command.contains("--lineages FF2026abc,FF2026def"));
        assert!(!command.contains("--csv"));
    }

    #[test]
    fn submit_command_switches_to_csv_above_the_threshold() {
        let designations: Vec<String> = (0..MAX_INLINE_LINEAGES + 1)
            .map(|i| format!("FF2026{i:04}"))
            .collect();
        let command = submit_command(&designations);
        assert!(command.contains("--csv eligible_lineages.csv"));
        assert!(!command.contains("--lineages"));
    }

    #[test]
    fn submit_command_is_a_valid_shell_continuation() {
        let command = submit_command(&["FF2026abc".to_string()]);
        let lines: Vec<&str> = command.lines().collect();
        assert!(lines[..lines.len() - 1].iter().all(|l| l.ends_with('\\')));
        assert!(!lines.last().unwrap().ends_with('\\'));
    }

    #[test]
    fn eligible_lineages_csv_has_the_expected_header_and_rows() {
        let csv = eligible_lineages_csv(&["FF2026abc".to_string(), "FF2026def".to_string()]);
        assert_eq!(csv, "lineage_designation\nFF2026abc\nFF2026def\n");
    }

    #[test]
    fn eligible_lineages_csv_of_an_empty_list_is_just_the_header() {
        assert_eq!(eligible_lineages_csv(&[]), "lineage_designation\n");
    }

    #[test]
    fn verdict_badge_class_covers_every_known_verdict() {
        assert_eq!(verdict_badge_class("accepted"), "badge-success");
        assert_eq!(verdict_badge_class("pending"), "badge-info");
        assert_eq!(verdict_badge_class("rejected"), "badge-error");
        assert_eq!(verdict_badge_class("error"), "badge-error");
        assert_eq!(verdict_badge_class("unknown"), "badge-ghost");
    }
}

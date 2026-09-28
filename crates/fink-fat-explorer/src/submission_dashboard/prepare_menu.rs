//! The "☰ Prepare submission" burger dropdown: everything needed to get a
//! `fink-fat submit` command ready (the copy-pastable command, the eligible-
//! lineages CSV, and the submitter-config generator), kept out of the
//! Submission page's main column so that page reads as one thing — a status
//! dashboard — with submission *preparation* tucked one click away as a
//! secondary tool. Same open/backdrop/panel mechanics as the homepage's
//! `ToolsMenu` (private to that module, not directly linkable from here),
//! reused rather than reinvented.

use dioxus::prelude::*;

use crate::submission_dashboard::data::SubmissionCandidate;
use crate::submitter_config_form::SubmitterConfigFields;

/// Above this many eligible lineages, the generated command switches from a
/// literal `--lineages a,b,c` list to `--csv eligible_lineages.csv` (paired
/// with the "Download CSV" button) — long enough to type/paste comfortably,
/// short enough that a handful of lineages don't need a separate file.
const MAX_INLINE_LINEAGES: usize = 5;

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

/// The burger dropdown itself.
///
/// # Arguments
/// * `candidates` — every currently-eligible, not-yet-submitted lineage
///   (from [`crate::submission_dashboard::data::get_submission_candidates`]),
///   passed down from the page so it isn't fetched twice.
#[component]
pub fn PrepareSubmissionMenu(candidates: Vec<SubmissionCandidate>) -> Element {
    let mut open = use_signal(|| false);
    let mut config_open = use_signal(|| false);
    let submitter_config = use_signal(crate::submitter_config_form::default_submitter_config);

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
        div { class: "relative",
            button {
                class: "btn btn-sm btn-outline btn-accent gap-1",
                r#type: "button",
                onclick: move |_| open.set(!open()),
                "☰ Prepare submission"
            }
            if open() {
                div { class: "fixed inset-0 z-40", onclick: move |_| open.set(false) }
                div {
                    class: "absolute right-0 mt-2 w-[34rem] max-w-[calc(100vw-2rem)] card bg-base-100 shadow-xl z-50 p-4 flex flex-col gap-3",
                    // Inline rather than Tailwind utility classes for the
                    // height cap: this panel is `position: absolute`, taken
                    // out of normal document flow, so nothing upstream
                    // guarantees the page's own scroll extends far enough
                    // to reach its bottom once it's taller than the
                    // viewport — the panel must scroll *itself*. `style`
                    // sidesteps any risk of the Tailwind class scanner not
                    // having regenerated `assets/main.css` yet for a
                    // newly-added utility (bitten by this once already).
                    style: "max-height: 80vh; overflow-y: auto;",
                    div {
                        h3 { class: "font-semibold", "Prepare a submission" }
                        if candidates.is_empty() {
                            p { class: "text-sm opacity-60 mt-1",
                                "No lineage is currently eligible (Well-sampled discovery / \
                                 Discovery) and unsubmitted."
                            }
                        } else {
                            p { class: "text-sm opacity-70 mt-1",
                                "{candidates.len()} lineage(s) ready to submit. Fill in a \
                                 submitter config below, then run:"
                            }
                            div { class: "mockup-code text-xs whitespace-pre mt-2",
                                pre { "data-prefix": "$", code { "{command}" } }
                            }
                            div { class: "flex gap-2 mt-2",
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
                        }
                    }

                    div { class: "bg-base-200 rounded-box",
                        button {
                            class: "btn btn-sm btn-ghost w-full justify-between",
                            r#type: "button",
                            onclick: move |_| config_open.set(!config_open()),
                            span { class: "font-semibold", "Generate submitter config (submission.yaml)" }
                            span { if config_open() { "▾" } else { "▸" } }
                        }
                        // A plain signal-toggled section rather than daisyUI's
                        // `.collapse` (checkbox + `grid-template-rows: 1fr`
                        // animation): that component sets `overflow: hidden`
                        // on itself, and once nested inside this panel's own
                        // `overflow-y: auto` scroll region, the browser's grid
                        // sizing resolved its `1fr` row against the *panel's*
                        // constrained height instead of the content's natural
                        // height — silently clipping the form with no
                        // scrollbar anywhere. A plain block-level toggle has
                        // no such row-sizing step to get confused, so it
                        // scrolls correctly inside the panel.
                        if config_open() {
                            div { class: "flex flex-col gap-3 p-3 pt-0",
                                p { class: "text-xs opacity-70",
                                    "Fill in your submitter/telescope identity once, download it \
                                     as `submission.yaml`, then pass it to `fink-fat submit \
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
}

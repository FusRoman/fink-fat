//! The "Submission" page: reached from the homepage's navbar button (next
//! to "🔭 Cross-match"), it tracks every `fink-fat submit` attempt on
//! record — status, MPC links, and (for production submissions) WAMO's
//! per-observation detail. Preparing a *new* submission (the copy-pastable
//! command, the eligible-lineages CSV, the submitter-config generator) is a
//! separate concern, kept behind the [`prepare_menu::PrepareSubmissionMenu`]
//! burger button rather than sharing this column, so the page reads as one
//! thing: a status dashboard.

mod data;
mod prepare_menu;

use dioxus::prelude::*;
use fink_fat_ades::mpc_submission::MPC_SUBMISSION_STATUS_URL;
use fink_fat_ades::wamo::WamoObservation;

use data::{
    get_submission_candidates, get_submission_history, refresh_submission_status, SubmissionRow,
};
use prepare_menu::PrepareSubmissionMenu;

/// daisyUI badge color class for one `mpc_submissions.verdict` value.
fn verdict_badge_class(verdict: &str) -> &'static str {
    match verdict {
        "accepted" => "badge-success",
        "pending" => "badge-info",
        "rejected" | "error" => "badge-error",
        _ => "badge-ghost",
    }
}

/// Whether a row has anything worth expanding into a detail panel — a bare
/// `pending` row with neither a coarse verdict detail nor a WAMO lookup yet
/// has nothing to show. Pure.
///
/// # Arguments
/// * `row` — the submission row.
///
/// # Return
/// `true` if the row's detail toggle should be shown at all.
fn has_expandable_detail(row: &SubmissionRow) -> bool {
    row.verdict_detail.is_some() || row.wamo_detail.is_some()
}

/// Parses a row's stored `wamo_detail` JSON back into the observations it
/// holds. Defensive rather than fallible: this is fink-fat's own
/// previously-written JSON, so a parse failure would mean a schema drift
/// bug, not bad external input — surfacing it as an empty list (a state the
/// UI already renders sensibly, as "not linked yet") is better than an
/// error banner for what is, at worst, a display glitch.
///
/// # Arguments
/// * `wamo_detail` — the row's `wamo_detail` field.
///
/// # Return
/// The observations, empty if `wamo_detail` is `None` or fails to parse.
fn parse_wamo_observations(wamo_detail: &Option<serde_json::Value>) -> Vec<WamoObservation> {
    wamo_detail
        .as_ref()
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default()
}

/// The one-line summary badge text for a WAMO observation's IAU designation
/// — the actual payoff of a real submission once MPC assigns one. Pure.
///
/// # Arguments
/// * `observation` — one WAMO match.
///
/// # Return
/// `Some("Designated (NNNNN)")` if MPC assigned a designation, `None`
/// otherwise (a match can exist — e.g. an identification against a known
/// object — without a fresh designation).
fn wamo_designation_label(observation: &WamoObservation) -> Option<String> {
    observation
        .iau_desig
        .as_deref()
        .map(|desig| format!("Designated ({desig})"))
}

/// Link to MPC's test-submission status page for a given submission — same
/// pattern as `lineage_page::ades_export_modal`'s `mpc_status_url`. Only
/// meaningful for `endpoint = "test"`: there is no public, browsable
/// equivalent page for production submissions (confirmed by probing while
/// building this feature) — production status is shown inline instead, via
/// the coarse verdict and WAMO detail already fetched by "Refresh".
///
/// # Arguments
/// * `submission_id` — the MPC-assigned submission id.
///
/// # Return
/// The status page URL.
fn mpc_test_status_url(submission_id: &str) -> String {
    format!("{MPC_SUBMISSION_STATUS_URL}?id={submission_id}")
}

#[component]
pub fn SubmissionDashboardPage() -> Element {
    let candidates_resource = use_resource(get_submission_candidates);
    let mut history = use_signal(Vec::<SubmissionRow>::new);
    let mut refreshing_id = use_signal(|| None::<i64>);
    let mut expanded_id = use_signal(|| None::<i64>);

    use_effect(move || {
        spawn(async move {
            if let Ok(rows) = get_submission_history().await {
                history.set(rows);
            }
        });
    });

    let candidates = match &*candidates_resource.read() {
        Some(Ok(rows)) => rows.clone(),
        _ => Vec::new(),
    };

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                div { class: "flex-1",
                    Link { to: crate::Route::Home {}, class: "link link-hover text-sm", "← Back to home" }
                }
                div { class: "flex-none", PrepareSubmissionMenu { candidates: candidates.clone() } }
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
                div { class: "card-body p-0",
                    h2 { class: "card-title p-4 pb-0", "Submitted lineages" }
                    if history.read().is_empty() {
                        p { class: "p-6 text-sm opacity-60", "No submission recorded yet." }
                    } else {
                        div { class: "overflow-x-auto",
                            table { class: "table table-zebra table-sm",
                                thead {
                                    tr {
                                        th { "" }
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
                                                if has_expandable_detail(row) {
                                                    button {
                                                        class: "btn btn-xs btn-ghost",
                                                        r#type: "button",
                                                        onclick: {
                                                            let row_id = row.id;
                                                            move |_| {
                                                                expanded_id.set(
                                                                    if expanded_id() == Some(row_id) { None } else { Some(row_id) },
                                                                );
                                                            }
                                                        },
                                                        if expanded_id() == Some(row.id) { "▾" } else { "▸" }
                                                    }
                                                }
                                            }
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
                                                match (&row.submission_id, row.endpoint.as_str()) {
                                                    (Some(submission_id), "test") => rsx! {
                                                        a {
                                                            class: "link",
                                                            href: "{mpc_test_status_url(submission_id)}",
                                                            target: "_blank",
                                                            rel: "noopener noreferrer",
                                                            "{submission_id}"
                                                        }
                                                    },
                                                    (Some(submission_id), _) => rsx! { "{submission_id}" },
                                                    (None, _) => rsx! { "—" },
                                                }
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
                                        if expanded_id() == Some(row.id) {
                                            tr { key: "{row.id}-detail",
                                                td { colspan: "7", class: "bg-base-200",
                                                    div { class: "flex flex-col gap-3 p-3 text-sm",
                                                        if let Some(detail) = &row.verdict_detail {
                                                            div {
                                                                div { class: "font-semibold text-xs opacity-70 mb-1", "Verdict detail" }
                                                                if let Some(comments) = detail.get("comments").and_then(|v| v.as_array()) {
                                                                    ul { class: "list-disc pl-4 text-xs",
                                                                        for comment in comments.iter().filter_map(|c| c.as_str()) {
                                                                            li { key: "{comment}", "{comment}" }
                                                                        }
                                                                    }
                                                                } else {
                                                                    pre { class: "text-xs whitespace-pre-wrap", "{detail}" }
                                                                }
                                                            }
                                                        }
                                                        if row.endpoint == "production" {
                                                            div {
                                                                div { class: "font-semibold text-xs opacity-70 mb-1 flex items-center gap-2",
                                                                    "WAMO detail"
                                                                    if let Some(checked_at) = &row.wamo_checked_at {
                                                                        span { class: "font-normal opacity-60", "(checked {checked_at})" }
                                                                    }
                                                                }
                                                                {
                                                                    let observations = parse_wamo_observations(&row.wamo_detail);
                                                                    if row.wamo_detail.is_none() {
                                                                        rsx! {
                                                                            p { class: "text-xs opacity-60",
                                                                                "Not checked yet — click Refresh to query WAMO."
                                                                            }
                                                                        }
                                                                    } else if observations.is_empty() {
                                                                        rsx! {
                                                                            p { class: "text-xs opacity-60",
                                                                                "MPC hasn't linked/published this submission yet — \
                                                                                 normal for a submission still in processing, \
                                                                                 check back later."
                                                                            }
                                                                        }
                                                                    } else {
                                                                        rsx! {
                                                                            div { class: "flex flex-col gap-2",
                                                                                for observation in &observations {
                                                                                    div {
                                                                                        key: "{observation.obsid}",
                                                                                        class: "border border-base-300 rounded p-2 flex flex-col gap-1",
                                                                                        if let Some(label) = wamo_designation_label(observation) {
                                                                                            span { class: "badge badge-success badge-sm self-start", "{label}" }
                                                                                        }
                                                                                        span { "{observation.status_decoded}" }
                                                                                        if let Some(obs80) = &observation.obs80 {
                                                                                            code { class: "text-xs block", "{obs80}" }
                                                                                        }
                                                                                        if let Some(reference) = &observation.reference {
                                                                                            span { class: "text-xs opacity-70", "Reference: {reference}" }
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

    fn row(
        verdict_detail: Option<serde_json::Value>,
        wamo_detail: Option<serde_json::Value>,
    ) -> SubmissionRow {
        SubmissionRow {
            id: 1,
            lineage_designation: "FF2026abc".to_string(),
            endpoint: "production".to_string(),
            submission_id: Some("sub-1".to_string()),
            verdict: "pending".to_string(),
            submitted_at: "2026-09-28T00:00:00+00:00".to_string(),
            verdict_checked_at: None,
            verdict_detail,
            wamo_detail,
            wamo_checked_at: None,
        }
    }

    #[test]
    fn verdict_badge_class_covers_every_known_verdict() {
        assert_eq!(verdict_badge_class("accepted"), "badge-success");
        assert_eq!(verdict_badge_class("pending"), "badge-info");
        assert_eq!(verdict_badge_class("rejected"), "badge-error");
        assert_eq!(verdict_badge_class("error"), "badge-error");
        assert_eq!(verdict_badge_class("unknown"), "badge-ghost");
    }

    #[test]
    fn has_expandable_detail_false_when_neither_detail_is_present() {
        assert!(!has_expandable_detail(&row(None, None)));
    }

    #[test]
    fn has_expandable_detail_true_when_verdict_detail_is_present() {
        assert!(has_expandable_detail(&row(
            Some(serde_json::json!({})),
            None
        )));
    }

    #[test]
    fn has_expandable_detail_true_when_wamo_was_checked_even_if_empty() {
        // `Some([])` means "WAMO was queried, nothing found (yet)" — still
        // worth expanding to show that informational state.
        assert!(has_expandable_detail(&row(
            None,
            Some(serde_json::json!([]))
        )));
    }

    #[test]
    fn parse_wamo_observations_empty_for_none() {
        assert!(parse_wamo_observations(&None).is_empty());
    }

    #[test]
    fn parse_wamo_observations_empty_for_malformed_json() {
        assert!(parse_wamo_observations(&Some(serde_json::json!({"not": "a list"}))).is_empty());
    }

    #[test]
    fn parse_wamo_observations_round_trips_real_shape() {
        let value = serde_json::json!([{
            "iau_desig": "380635",
            "input_type": "submission_block_id",
            "obs80": "c0635 ...",
            "obsid": "L4eBVG000000CfiO010000A9a",
            "obssubid": null,
            "ref": "MPS   826083",
            "status": "P",
            "status_decoded": "matched",
            "submission_block_id": "2017-10-10T12:17:02.000_0000CfiO_01",
            "submission_id": "2017-10-10T12:17:02.000_0000CfiO"
        }]);
        let observations = parse_wamo_observations(&Some(value));
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].iau_desig.as_deref(), Some("380635"));
    }

    #[test]
    fn wamo_designation_label_present_when_designated() {
        let mut observations = parse_wamo_observations(&Some(serde_json::json!([{
            "iau_desig": "380635",
            "input_type": "submission_block_id",
            "obs80": null,
            "obsid": "obs-1",
            "obssubid": null,
            "ref": null,
            "status": "P",
            "status_decoded": "matched",
            "submission_block_id": null,
            "submission_id": null
        }])));
        let observation = observations.remove(0);
        assert_eq!(
            wamo_designation_label(&observation),
            Some("Designated (380635)".to_string())
        );
    }

    #[test]
    fn wamo_designation_label_absent_without_a_designation() {
        let mut observations = parse_wamo_observations(&Some(serde_json::json!([{
            "iau_desig": null,
            "input_type": "submission_block_id",
            "obs80": null,
            "obsid": "obs-1",
            "obssubid": null,
            "ref": null,
            "status": "P",
            "status_decoded": "matched",
            "submission_block_id": null,
            "submission_id": null
        }])));
        let observation = observations.remove(0);
        assert_eq!(wamo_designation_label(&observation), None);
    }

    #[test]
    fn mpc_test_status_url_embeds_the_submission_id() {
        assert_eq!(
            mpc_test_status_url("2026-09-28T09:18:41.271_00000lJM"),
            "https://submit-test.minorplanetcenter.net/submission_status/query/?id=2026-09-28T09:18:41.271_00000lJM"
        );
    }
}

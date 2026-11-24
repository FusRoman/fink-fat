//! The "Submission" page: reached from the homepage's navbar button (next
//! to "🔭 Cross-match"), it tracks every `fink-fat submit` attempt on
//! record — status, MPC links, and (for production submissions) WAMO's
//! per-observation detail. Preparing a *new* submission (the copy-pastable
//! command, the eligible-lineages CSV, the submitter-config generator) is a
//! separate concern, kept behind the [`prepare_menu::PrepareSubmissionMenu`]
//! burger button rather than sharing this column, so the page reads as one
//! thing: a status dashboard.
//!
//! The history is filtered (endpoint, verdict, free text, submission date
//! range), sorted by submission time and paginated **server-side**, so the
//! page stays usable with a very large number of submissions. The pure query
//! model is in [`query`], the controls in [`filters_bar`] and
//! [`date_range_picker`].
//!
//! A production submission's state in MPC's pipeline (processing queue,
//! publication, rejection...) is read from its stored WAMO observations by
//! [`wamo_view`], shown as a badge in the table and broken down per state
//! in the expanded row.

mod ades_xml_modal;
mod calendar_bridge;
mod data;
mod date_range_picker;
mod filters_bar;
mod prepare_menu;
mod query;
mod wamo_view;

use dioxus::prelude::*;
use fink_fat_ades::mpc_submission::MPC_SUBMISSION_STATUS_URL;

use ades_xml_modal::{ades_file_name, AdesXmlModal};
use data::{
    get_submission_candidates, list_submissions, refresh_submission_status, SubmissionPage,
};
use filters_bar::FiltersBar;
use prepare_menu::PrepareSubmissionMenu;
use query::{
    total_pages, DateRange, EndpointFilter, SubmissionQuery, VerdictFilter, SUBMISSIONS_PAGE_SIZE,
};
use wamo_view::{WamoDetailSection, WamoSummaryBadge};

use crate::homepage::interaction::{Pagination, SortDirection};

/// Debounce applied to the search box before it triggers a query, in ms.
const SEARCH_DEBOUNCE_MS: u64 = 250;

/// daisyUI badge color class for one `mpc_submissions.verdict` value.
fn verdict_badge_class(verdict: &str) -> &'static str {
    match verdict {
        "accepted" => "badge-success",
        "pending" => "badge-info",
        "rejected" | "error" => "badge-error",
        _ => "badge-ghost",
    }
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

/// Arrow shown next to the sorted column header.
///
/// # Arguments
/// * `sort` — the active sort direction.
///
/// # Return
/// `"▲"` for ascending, `"▼"` for descending.
fn sort_arrow(sort: SortDirection) -> &'static str {
    match sort {
        SortDirection::Asc => "▲",
        SortDirection::Desc => "▼",
    }
}

/// Message shown instead of the table when the current page has no row.
///
/// # Arguments
/// * `filters_active` — whether any filter is set.
///
/// # Return
/// A "nothing recorded" message without filters, a "no match" one with.
fn empty_list_message(filters_active: bool) -> &'static str {
    if filters_active {
        "No submission matches these filters."
    } else {
        "No submission recorded yet."
    }
}

#[component]
pub fn SubmissionDashboardPage() -> Element {
    let candidates_resource = use_resource(get_submission_candidates);
    let mut refreshing_id = use_signal(|| None::<i64>);
    let mut expanded_id = use_signal(|| None::<i64>);
    let mut xml_modal_row = use_signal(|| None::<(i64, String)>);

    let mut endpoint_filter = use_signal(EndpointFilter::default);
    let mut verdict_filter = use_signal(VerdictFilter::default);
    let mut search = use_signal(String::new);
    let mut date_range = use_signal(|| None::<DateRange>);
    let mut sort = use_signal(|| SortDirection::Desc);
    let mut current_page = use_signal(|| 0_i64);
    let mut page_data = use_signal(|| None::<SubmissionPage>);

    // The search text the query actually uses, trailing the search box by
    // `SEARCH_DEBOUNCE_MS`; `debounce_generation` lets a newer keystroke
    // invalidate an in-flight timer (same pattern as the homepage branch tab).
    let mut debounced_search = use_signal(String::new);
    let mut debounce_generation = use_signal(|| 0_u64);
    use_effect(move || {
        let text = search();
        // `peek`: reading the generation here would make this effect
        // retrigger itself.
        let generation = *debounce_generation.peek() + 1;
        debounce_generation.set(generation);
        spawn(async move {
            crate::sleep_ms(SEARCH_DEBOUNCE_MS).await;
            if *debounce_generation.peek() == generation {
                debounced_search.set(text);
            }
        });
    });

    // Every signal read inside the future is a dependency, so the list is
    // refetched on any filter, sort or page change.
    let page_resource = use_resource(move || async move {
        list_submissions(SubmissionQuery {
            endpoint: endpoint_filter(),
            verdict: verdict_filter(),
            search: debounced_search(),
            date_range: date_range(),
            sort: sort(),
            page: current_page(),
        })
        .await
    });

    // Mirror the fetched page into a signal the "Refresh" buttons can patch
    // in place, and follow the server's page clamping.
    use_effect(move || {
        if let Some(Ok(fetched)) = &*page_resource.read() {
            if *current_page.peek() != fetched.page {
                current_page.set(fetched.page);
            }
            page_data.set(Some(fetched.clone()));
        }
    });

    // A narrower result set may not have the current page any more: go back
    // to the first page whenever the filters or the sort change.
    use_effect(move || {
        let _ = (
            endpoint_filter(),
            verdict_filter(),
            debounced_search(),
            date_range(),
            sort(),
        );
        if *current_page.peek() != 0 {
            current_page.set(0);
        }
    });

    let filters_active = SubmissionQuery {
        endpoint: endpoint_filter(),
        verdict: verdict_filter(),
        search: search(),
        date_range: date_range(),
        ..SubmissionQuery::default()
    }
    .has_active_filters();

    let rows = page_data
        .read()
        .as_ref()
        .map(|page| page.rows.clone())
        .unwrap_or_default();
    let (total_filtered, total_all) = page_data
        .read()
        .as_ref()
        .map_or((0, 0), |page| (page.total_filtered, page.total_all));

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
                    div { class: "stat-value", "{total_all}" }
                }
                if filters_active {
                    div { class: "stat",
                        div { class: "stat-title", "Matching the filters" }
                        div { class: "stat-value", "{total_filtered}" }
                    }
                }
            }

            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body p-0",
                    h2 { class: "card-title p-4 pb-0", "Submitted lineages" }
                    FiltersBar {
                        endpoint: endpoint_filter,
                        verdict: verdict_filter,
                        search,
                        date_range,
                        filters_active,
                        on_clear: move |_| {
                            endpoint_filter.set(EndpointFilter::All);
                            verdict_filter.set(VerdictFilter::All);
                            search.set(String::new());
                            debounced_search.set(String::new());
                            date_range.set(None);
                        },
                    }
                    if page_data.read().is_none() {
                        p { class: "p-6 text-sm opacity-60", "Loading…" }
                    } else if rows.is_empty() {
                        p { class: "p-6 text-sm opacity-60", "{empty_list_message(filters_active)}" }
                    } else {
                        div { class: "overflow-x-auto",
                            table { class: "table table-zebra table-sm",
                                thead {
                                    tr {
                                        th { "" }
                                        th { "Lineage" }
                                        th { "Endpoint" }
                                        th { "Submission ID" }
                                        th { title: "MPC's coarse ingestion check", "Ingest verdict" }
                                        th { title: "Per-observation state from MPC's WAMO lookup (production only)", "MPC processing" }
                                        th {
                                            class: "cursor-pointer select-none",
                                            title: "Sort by submission time",
                                            onclick: move |_| sort.set(sort().toggled()),
                                            "Submitted at {sort_arrow(sort())}"
                                        }
                                        th { "" }
                                    }
                                }
                                tbody {
                                    for row in rows.iter() {
                                        tr { key: "{row.id}",
                                            td {
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
                                            td {
                                                if row.endpoint == "production" {
                                                    WamoSummaryBadge { wamo_detail: row.wamo_detail.clone() }
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
                                                                    page_data.with_mut(|page| {
                                                                        let patched = page
                                                                            .as_mut()
                                                                            .and_then(|p| p.rows.iter_mut().find(|r| r.id == row_id));
                                                                        if let Some(r) = patched {
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
                                                td { colspan: "8", class: "bg-base-200",
                                                    div { class: "flex flex-col gap-3 p-3 text-sm",
                                                        button {
                                                            class: "btn btn-xs btn-outline self-start",
                                                            r#type: "button",
                                                            onclick: {
                                                                let row_id = row.id;
                                                                let lineage_designation = row.lineage_designation.clone();
                                                                move |_| xml_modal_row.set(Some((row_id, lineage_designation.clone())))
                                                            },
                                                            "🔍 View ADES XML"
                                                        }
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
                                                            WamoDetailSection {
                                                                submission_id: row.submission_id.clone(),
                                                                wamo_detail: row.wamo_detail.clone(),
                                                                wamo_checked_at: row.wamo_checked_at.clone(),
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
                        div { class: "px-4 pb-4",
                            Pagination {
                                current_page: move |page| current_page.set(page),
                                page: current_page(),
                                total_pages: total_pages(total_filtered, SUBMISSIONS_PAGE_SIZE),
                                total_lineages: total_filtered,
                                item_label: "submissions".to_string(),
                            }
                        }
                    }
                }
            }
        }

        AdesXmlModal {
            id: xml_modal_row().map(|(id, _)| id).unwrap_or_default(),
            file_name: xml_modal_row()
                .map(|(_, designation)| ades_file_name(&designation))
                .unwrap_or_default(),
            open: xml_modal_row().is_some(),
            on_close: move |_| xml_modal_row.set(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_badge_class_covers_every_known_verdict() {
        assert_eq!(verdict_badge_class("accepted"), "badge-success");
        assert_eq!(verdict_badge_class("pending"), "badge-info");
        assert_eq!(verdict_badge_class("rejected"), "badge-error");
        assert_eq!(verdict_badge_class("error"), "badge-error");
        assert_eq!(verdict_badge_class("unknown"), "badge-ghost");
    }

    #[test]
    fn sort_arrow_matches_direction() {
        assert_eq!(sort_arrow(SortDirection::Asc), "▲");
        assert_eq!(sort_arrow(SortDirection::Desc), "▼");
    }

    #[test]
    fn empty_list_message_distinguishes_filtered_from_empty() {
        assert_eq!(empty_list_message(false), "No submission recorded yet.");
        assert_eq!(
            empty_list_message(true),
            "No submission matches these filters."
        );
    }

    #[test]
    fn mpc_test_status_url_embeds_the_submission_id() {
        assert_eq!(
            mpc_test_status_url("2026-09-28T09:18:41.271_00000lJM"),
            "https://submit-test.minorplanetcenter.net/submission_status/query/?id=2026-09-28T09:18:41.271_00000lJM"
        );
    }
}

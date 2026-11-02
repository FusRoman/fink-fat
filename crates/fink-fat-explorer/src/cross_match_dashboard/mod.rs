mod data;

use dioxus::prelude::*;

use data::get_cross_match_status;
pub use data::CrossMatchStatusRow;

/// Which rows [`CrossMatchDashboardPage`]'s filter row currently shows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ServiceFilter {
    All,
    Skybot,
    Mpc,
    Both,
}

impl ServiceFilter {
    const ALL: [ServiceFilter; 4] = [
        ServiceFilter::All,
        ServiceFilter::Skybot,
        ServiceFilter::Mpc,
        ServiceFilter::Both,
    ];

    fn label(self) -> &'static str {
        match self {
            ServiceFilter::All => "All",
            ServiceFilter::Skybot => "Skybot",
            ServiceFilter::Mpc => "MPC",
            ServiceFilter::Both => "Both",
        }
    }

    fn matches(self, row: &CrossMatchStatusRow) -> bool {
        let skybot = !row.skybot_object_names.is_empty();
        let mpc = row.cnd_hit_count > 0;
        match self {
            ServiceFilter::All => true,
            ServiceFilter::Skybot => skybot,
            ServiceFilter::Mpc => mpc,
            ServiceFilter::Both => skybot && mpc,
        }
    }
}

/// Trims an RFC 3339 timestamp down to `"YYYY-MM-DD HH:MM"` — same idea as
/// `lineage_page::trajectory_plot::format_queried_at_minute`, duplicated
/// here rather than shared (this codebase already keeps this kind of small
/// per-module formatting helper colocated, e.g. `skybot_search::history`
/// and `cnd_search::history` each have their own `elapsed_days`).
fn format_checked_at(queried_at: Option<&str>) -> String {
    match queried_at.and_then(|s| s.get(0..16)) {
        Some(prefix) => prefix.replacen('T', " ", 1),
        None => "—".to_string(),
    }
}

/// Dashboard of every lineage with a positive Skybot and/or MPC (CND)
/// cross-match, reached from the homepage's Tools menu. Queries Postgres
/// fresh on load (see `data.rs`'s doc comment for why this isn't served
/// from the homepage's in-RAM snapshot) and links each row straight to its
/// lineage detail page.
#[component]
pub fn CrossMatchDashboardPage() -> Element {
    let status_resource = use_resource(get_cross_match_status);
    let mut filter = use_signal(|| ServiceFilter::All);
    let mut search_query = use_signal(String::new);

    let rows: Vec<CrossMatchStatusRow> = match &*status_resource.read() {
        Some(Ok(rows)) => rows.clone(),
        _ => Vec::new(),
    };

    let skybot_count = rows
        .iter()
        .filter(|r| !r.skybot_object_names.is_empty())
        .count();
    let mpc_count = rows.iter().filter(|r| r.cnd_hit_count > 0).count();
    let both_count = rows
        .iter()
        .filter(|r| !r.skybot_object_names.is_empty() && r.cnd_hit_count > 0)
        .count();

    let search = search_query().to_lowercase();
    let visible_rows: Vec<&CrossMatchStatusRow> = rows
        .iter()
        .filter(|row| filter().matches(row))
        .filter(|row| search.is_empty() || row.lineage_designation.to_lowercase().contains(&search))
        .collect();

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                Link {
                    to: crate::Route::Home {},
                    class: "link link-hover text-sm",
                    "← Back to home"
                }
            }

            match &*status_resource.read() {
                Some(Ok(_)) => rsx! {
                    div { class: "stats shadow bg-base-100",
                        div { class: "stat",
                            div { class: "stat-title", "Lineages with a match" }
                            div { class: "stat-value text-primary", "{rows.len()}" }
                        }
                        div { class: "stat",
                            div { class: "stat-title", "Skybot" }
                            div { class: "stat-value text-success", "{skybot_count}" }
                        }
                        div { class: "stat",
                            div { class: "stat-title", "MPC (CND)" }
                            div { class: "stat-value text-success", "{mpc_count}" }
                        }
                        div { class: "stat",
                            div { class: "stat-title", "Both" }
                            div { class: "stat-value", "{both_count}" }
                        }
                    }

                    div { class: "flex flex-wrap items-center gap-3",
                        div { class: "join",
                            for option in ServiceFilter::ALL {
                                button {
                                    key: "{option.label()}",
                                    class: if filter() == option { "join-item btn btn-sm btn-active" } else { "join-item btn btn-sm" },
                                    onclick: move |_| filter.set(option),
                                    "{option.label()}"
                                }
                            }
                        }
                        input {
                            r#type: "text",
                            placeholder: "Filter by designation...",
                            class: "input input-bordered input-sm w-64",
                            value: "{search_query}",
                            oninput: move |evt| search_query.set(evt.value()),
                        }
                    }

                    div { class: "card bg-base-100 shadow-sm",
                        div { class: "card-body p-0",
                            if visible_rows.is_empty() {
                                p { class: "p-6 text-sm opacity-60",
                                    if rows.is_empty() {
                                        "No lineages have a positive cross-match yet."
                                    } else {
                                        "No lineages match this filter."
                                    }
                                }
                            } else {
                                div { class: "overflow-x-auto",
                                    table { class: "table table-zebra table-sm",
                                        thead {
                                            tr {
                                                th { "Lineage" }
                                                th { "Skybot" }
                                                th { "Skybot last checked" }
                                                th { "MPC (CND)" }
                                                th { "MPC last checked" }
                                            }
                                        }
                                        tbody {
                                            for row in visible_rows {
                                                tr { key: "{row.lineage_designation}",
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
                                                        if row.skybot_object_names.is_empty() {
                                                            span { class: "badge badge-sm badge-ghost", "—" }
                                                        } else {
                                                            span {
                                                                class: "badge badge-sm badge-success",
                                                                title: "{row.skybot_object_names.join(\", \")}",
                                                                "{row.skybot_object_names.len()} object(s)"
                                                            }
                                                        }
                                                    }
                                                    td { class: "text-xs opacity-70",
                                                        "{format_checked_at(row.skybot_queried_at.as_deref())}"
                                                    }
                                                    td {
                                                        if row.cnd_hit_count == 0 {
                                                            span { class: "badge badge-sm badge-ghost", "—" }
                                                        } else {
                                                            span {
                                                                class: "badge badge-sm badge-success",
                                                                "{row.cnd_hit_count} observation(s)"
                                                            }
                                                        }
                                                    }
                                                    td { class: "text-xs opacity-70",
                                                        "{format_checked_at(row.cnd_queried_at.as_deref())}"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                Some(Err(e)) => rsx! {
                    div { class: "alert alert-error", "Failed to load cross-match status: {e}" }
                },
                None => rsx! {
                    div { class: "flex justify-center py-12",
                        span { class: "loading loading-spinner loading-lg" }
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(skybot: &[&str], cnd_hits: usize) -> CrossMatchStatusRow {
        CrossMatchStatusRow {
            lineage_designation: "test".to_string(),
            skybot_queried_at: Some("2026-09-23T10:00:00+00:00".to_string()),
            skybot_object_names: skybot.iter().map(|s| s.to_string()).collect(),
            cnd_queried_at: Some("2026-09-23T10:00:00+00:00".to_string()),
            cnd_hit_count: cnd_hits,
        }
    }

    #[test]
    fn service_filter_all_matches_everything() {
        assert!(ServiceFilter::All.matches(&row(&[], 0)));
        assert!(ServiceFilter::All.matches(&row(&["Ceres"], 3)));
    }

    #[test]
    fn service_filter_skybot_ignores_cnd() {
        assert!(ServiceFilter::Skybot.matches(&row(&["Ceres"], 0)));
        assert!(!ServiceFilter::Skybot.matches(&row(&[], 3)));
    }

    #[test]
    fn service_filter_mpc_ignores_skybot() {
        assert!(ServiceFilter::Mpc.matches(&row(&[], 3)));
        assert!(!ServiceFilter::Mpc.matches(&row(&["Ceres"], 0)));
    }

    #[test]
    fn service_filter_both_requires_both() {
        assert!(ServiceFilter::Both.matches(&row(&["Ceres"], 3)));
        assert!(!ServiceFilter::Both.matches(&row(&["Ceres"], 0)));
        assert!(!ServiceFilter::Both.matches(&row(&[], 3)));
    }

    #[test]
    fn format_checked_at_trims_to_the_minute() {
        assert_eq!(
            format_checked_at(Some("2026-09-23T08:04:32.940721+00:00")),
            "2026-09-23 08:04"
        );
    }

    #[test]
    fn format_checked_at_reports_a_dash_when_never_checked() {
        assert_eq!(format_checked_at(None), "—");
    }
}

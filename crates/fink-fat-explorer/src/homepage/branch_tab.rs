use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::homepage::family::DynamicalFamily;
use crate::homepage::interaction::{Pagination, SortColumn, SortDirection, PAGE_SIZE};
use crate::Route;

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct Branch {
    branch_id: i64,
    lineage_id: i64,
    designation: String,
    lineage_designation: String,
    cumulative_llr: f64,
    n_real_updates: i64,
    family: DynamicalFamily,
    arc_length_days: f64,
    n_nights: i64,
    median_inter_night_dt_days: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct LineageGroup {
    lineage_id: i64,
    best: Branch,
    others: Vec<Branch>,
}

/// One page of the listing, plus the total number of lineages matching the
/// current filters (which the pagination footer needs).
#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct LineagePage {
    groups: Vec<LineageGroup>,
    total_lineages: i64,
}

/// Served entirely from the in-RAM homepage snapshot — see
/// [`crate::homepage::snapshot`]. Filtering by family used to force a
/// `LATERAL` best-hypothesis lookup plus a `kf_state` primary-key probe for
/// every branch in the database, twice per call (count, then page); it is now
/// a scan over a few hundred thousand pre-parsed enum values.
///
/// `None` means the snapshot has not finished building yet; the caller polls.
#[server]
async fn list_lineages(
    page: i64,
    sort_column: Option<SortColumn>,
    sort_direction: SortDirection,
    search_query: String,
    hidden_families: Vec<DynamicalFamily>,
) -> Result<Option<LineagePage>, ServerFnError> {
    use crate::homepage::{
        interaction::PAGE_SIZE,
        snapshot::{self, PageQuery},
    };

    let Some(snap) = snapshot::snapshot().await else {
        return Ok(None);
    };

    // Lowercased once here rather than per candidate: the snapshot stores each
    // lineage designation pre-lowercased, so this reproduces the old
    // `ILIKE '%…%'` as a plain substring test.
    let search = search_query.trim().to_lowercase();

    let (page_indices, total_lineages) = snap.page(&PageQuery {
        search: &search,
        hidden_families: &hidden_families,
        sort_column,
        descending: sort_direction == SortDirection::Desc,
        offset: (page * PAGE_SIZE).max(0) as usize,
        limit: PAGE_SIZE as usize,
    });

    let to_branch = |row: &snapshot::BranchRow| Branch {
        branch_id: row.branch_id,
        lineage_id: row.lineage_id,
        designation: row.designation.to_string(),
        lineage_designation: row.lineage_designation.to_string(),
        cumulative_llr: row.cumulative_llr,
        n_real_updates: row.n_real_updates,
        family: row.family,
        arc_length_days: row.arc_length_days,
        n_nights: row.n_nights,
        median_inter_night_dt_days: row.median_inter_night_dt_days,
    };

    let groups: Vec<LineageGroup> = page_indices
        .into_iter()
        .map(|idx| {
            let entry = &snap.lineages[idx as usize];
            LineageGroup {
                lineage_id: entry.lineage_id,
                best: to_branch(&snap.branches[entry.best as usize]),
                others: snap.others_idx[entry.others.start as usize..entry.others.end as usize]
                    .iter()
                    .map(|&i| to_branch(&snap.branches[i as usize]))
                    .collect(),
            }
        })
        .collect();

    Ok(Some(LineagePage {
        groups,
        total_lineages,
    }))
}

/// Rebuilds the homepage snapshot in the background — what the navbar's
/// refresh button calls after a `fink-fat convert`. Returns immediately; the
/// previous snapshot keeps being served until the new one is ready.
#[server]
pub async fn refresh_snapshot() -> Result<(), ServerFnError> {
    crate::homepage::snapshot::request_refresh();
    Ok(())
}

#[component]
fn TabHeader(
    sort_column: Option<SortColumn>,
    sort_direction: SortDirection,
    on_sort: EventHandler<SortColumn>,
) -> Element {
    let llr_arrow: &'static str = match (sort_column, sort_direction) {
        (Some(SortColumn::CumulativeLlr), SortDirection::Asc) => "▲",
        (Some(SortColumn::CumulativeLlr), SortDirection::Desc) => "▼",
        _ => "⇅",
    };

    let updates_arrow: &'static str = match (sort_column, sort_direction) {
        (Some(SortColumn::Updates), SortDirection::Asc) => "▲",
        (Some(SortColumn::Updates), SortDirection::Desc) => "▼",
        _ => "⇅",
    };

    let llr_active_class = if sort_column == Some(SortColumn::CumulativeLlr) {
        "text-primary"
    } else {
        ""
    };

    let updates_active_class = if sort_column == Some(SortColumn::Updates) {
        "text-primary"
    } else {
        ""
    };

    let family_arrow: &'static str = match (sort_column, sort_direction) {
        (Some(SortColumn::Family), SortDirection::Asc) => "▲",
        (Some(SortColumn::Family), SortDirection::Desc) => "▼",
        _ => "⇅",
    };

    let family_active_class = if sort_column == Some(SortColumn::Family) {
        "text-primary"
    } else {
        ""
    };

    let arc_arrow: &'static str = match (sort_column, sort_direction) {
        (Some(SortColumn::ArcLength), SortDirection::Asc) => "▲",
        (Some(SortColumn::ArcLength), SortDirection::Desc) => "▼",
        _ => "⇅",
    };

    let arc_active_class = if sort_column == Some(SortColumn::ArcLength) {
        "text-primary"
    } else {
        ""
    };

    let nights_arrow: &'static str = match (sort_column, sort_direction) {
        (Some(SortColumn::Nights), SortDirection::Asc) => "▲",
        (Some(SortColumn::Nights), SortDirection::Desc) => "▼",
        _ => "⇅",
    };

    let nights_active_class = if sort_column == Some(SortColumn::Nights) {
        "text-primary"
    } else {
        ""
    };

    let median_dt_arrow: &'static str = match (sort_column, sort_direction) {
        (Some(SortColumn::MedianInterNightDt), SortDirection::Asc) => "▲",
        (Some(SortColumn::MedianInterNightDt), SortDirection::Desc) => "▼",
        _ => "⇅",
    };

    let median_dt_active_class = if sort_column == Some(SortColumn::MedianInterNightDt) {
        "text-primary"
    } else {
        ""
    };

    rsx! {
        div { class: "grid grid-cols-8 gap-4 px-4 py-2 text-sm font-semibold opacity-60",
            span { "Designation" }
            span { "Lineage" }
            span {
                class: "cursor-pointer select-none flex items-center gap-1 hover:text-primary transition-colors {family_active_class}",
                onclick: move |_| on_sort.call(SortColumn::Family),
                "Family"
                span { class: "text-xs", "{family_arrow}" }
            }
            span {
                class: "cursor-pointer select-none flex items-center gap-1 hover:text-primary transition-colors {llr_active_class}",
                onclick: move |_| on_sort.call(SortColumn::CumulativeLlr),
                "Cumulative LLR"
                span { class: "text-xs", "{llr_arrow}" }
            }
            span {
                class: "cursor-pointer select-none flex items-center gap-1 hover:text-primary transition-colors {updates_active_class}",
                onclick: move |_| on_sort.call(SortColumn::Updates),
                "Updates"
                span { class: "text-xs", "{updates_arrow}" }
            }
            span {
                class: "cursor-pointer select-none flex items-center gap-1 hover:text-primary transition-colors {arc_active_class}",
                onclick: move |_| on_sort.call(SortColumn::ArcLength),
                "Arc (days)"
                span { class: "text-xs", "{arc_arrow}" }
            }
            span {
                class: "cursor-pointer select-none flex items-center gap-1 hover:text-primary transition-colors {nights_active_class}",
                onclick: move |_| on_sort.call(SortColumn::Nights),
                "Nights"
                span { class: "text-xs", "{nights_arrow}" }
            }
            span {
                class: "cursor-pointer select-none flex items-center gap-1 hover:text-primary transition-colors {median_dt_active_class}",
                onclick: move |_| on_sort.call(SortColumn::MedianInterNightDt),
                "Median Δt (days)"
                span { class: "text-xs", "{median_dt_arrow}" }
            }
        }
    }
}

/// "—" for branches with <=1 night (no inter-night gap to measure).
fn format_median_dt(days: Option<f64>) -> String {
    match days {
        Some(v) => format!("{v:.1}"),
        None => "—".to_string(),
    }
}

#[component]
fn LineageTable(groups: Vec<LineageGroup>) -> Element {
    rsx! {
        div { class: "flex flex-col gap-2",
            for group in groups {
                div {
                    key: "{group.lineage_id}",
                    tabindex: "0",
                    class: "collapse collapse-arrow bg-base-100 border border-base-300",

                    div { class: "collapse-title",
                        div { class: "grid grid-cols-8 gap-4 items-center",
                            span { class: "font-medium", "{group.best.designation}" }
                            span {
                                Link {
                                    to: Route::LineagePage {
                                        lineage_id: group.best.lineage_designation.clone(),
                                    },
                                    class: "link link-primary font-medium",
                                    "{group.best.lineage_designation}"
                                }
                            }
                            span {
                                span {
                                    class: "badge badge-sm text-white border-0",
                                    style: "background-color: {group.best.family.color()};",
                                    "{group.best.family}"
                                }
                            }
                            span { "{group.best.cumulative_llr:.2}" }
                            span { "{group.best.n_real_updates}" }
                            span { "{group.best.arc_length_days:.1}" }
                            span { "{group.best.n_nights}" }
                            span { "{format_median_dt(group.best.median_inter_night_dt_days)}" }
                        }
                    }

                    div { class: "collapse-content",
                        if group.others.is_empty() {
                            p { class: "text-sm opacity-60 px-2",
                                "No other branches for this lineage."
                            }
                        } else {
                            div { class: "flex flex-col gap-1",
                                for branch in &group.others {
                                    div {
                                        key: "{branch.branch_id}",
                                        class: "grid grid-cols-8 gap-4 items-center px-2 py-1 text-sm hover:bg-base-200 rounded",
                                        span { "{branch.designation}" }
                                        span { "{branch.lineage_designation}" }
                                        span {
                                            span {
                                                class: "badge badge-sm text-white border-0",
                                                style: "background-color: {branch.family.color()};",
                                                "{branch.family}"
                                            }
                                        }
                                        span { "{branch.cumulative_llr:.2}" }
                                        span { "{branch.n_real_updates}" }
                                        span { "{branch.arc_length_days:.1}" }
                                        span { "{branch.n_nights}" }
                                        span { "{format_median_dt(branch.median_inter_night_dt_days)}" }
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

/// Idle time after the last keystroke before the search is actually issued.
/// The query itself is now sub-millisecond, but a round-trip per character
/// still floods the network tab and races its own responses.
const SEARCH_DEBOUNCE_MS: u64 = 200;

/// How often to re-check whether the homepage snapshot has finished building.
const WARMUP_POLL_MS: u64 = 1000;

#[component]
pub fn BranchTab(
    search_query: String,
    hidden_families: Signal<HashSet<DynamicalFamily>>,
    refresh_token: Signal<u64>,
) -> Element {
    let mut current_page = use_signal(|| 0_i64);
    let mut sort_column = use_signal(|| None::<SortColumn>);
    let mut sort_direction = use_signal(|| SortDirection::Desc);

    // The search text the resource actually queries on, trailing the prop by
    // `SEARCH_DEBOUNCE_MS`. `debounce_generation` lets a newer keystroke
    // invalidate an in-flight timer: only the timer whose generation is still
    // current is allowed to commit.
    let mut debounced_search = use_signal(String::new);
    let mut debounce_generation = use_signal(|| 0_u64);

    use_effect(use_reactive!(|(search_query,)| {
        // `peek`, not a read: subscribing to the generation here would make
        // this effect retrigger itself on every keystroke it handles.
        let generation = *debounce_generation.peek() + 1;
        debounce_generation.set(generation);

        spawn(async move {
            crate::sleep_ms(SEARCH_DEBOUNCE_MS).await;
            if *debounce_generation.peek() == generation {
                debounced_search.set(search_query);
            }
        });
    }));

    // Every dependency is a signal read inside the future, so `use_resource`
    // subscribes to all of them and refetches on a legend toggle, a sort, a
    // page change, a debounced search or a snapshot refresh.
    let mut lineages_resource = use_resource(move || async move {
        // Sorted for a stable request shape.
        let mut hidden: Vec<DynamicalFamily> = hidden_families().into_iter().collect();
        hidden.sort();

        let _ = refresh_token();

        list_lineages(
            current_page(),
            sort_column(),
            sort_direction(),
            debounced_search(),
            hidden,
        )
        .await
    });

    // Reset to page 0 whenever the (debounced) search text changes. `peek`
    // reads the page without subscribing to it: subscribing would make this
    // effect re-run on every pagination and immediately bounce the user back
    // to page 0.
    use_effect(move || {
        let _ = debounced_search();
        if *current_page.peek() != 0 {
            current_page.set(0);
        }
    });

    // Same, for the family filter — fewer lineages match, so the current page
    // may no longer exist.
    use_effect(move || {
        let _ = hidden_families();
        if *current_page.peek() != 0 {
            current_page.set(0);
        }
    });

    // `Ok(None)` means the in-RAM snapshot is still being built (first request
    // after a server start, or a rebuild that has not landed yet). Poll rather
    // than leaving the table empty.
    use_effect(move || {
        let warming = matches!(&*lineages_resource.read(), Some(Ok(None)));
        if warming {
            spawn(async move {
                crate::sleep_ms(WARMUP_POLL_MS).await;
                lineages_resource.restart();
            });
        }
    });

    let mut toggle_sort = move |col: SortColumn| {
        if sort_column() == Some(col) {
            sort_direction.set(sort_direction().toggled());
        } else {
            sort_column.set(Some(col));
            sort_direction.set(SortDirection::Desc);
        }
        current_page.set(0);
    };

    let data = &*lineages_resource.read();

    match data {
        Some(Ok(Some(LineagePage {
            groups,
            total_lineages,
        }))) => {
            let total_pages = (*total_lineages + PAGE_SIZE - 1) / PAGE_SIZE;
            let page = current_page();

            rsx! {
                div { class: "card bg-base-100 shadow-sm",
                    div { class: "card-body",
                        div { class: "flex items-center justify-between mb-2",
                            h2 { class: "card-title", "Recent lineages" }
                            select { class: "select select-bordered select-sm w-40",
                                option { "All" }
                                option { "Active" }
                                option { "Archived" }
                            }
                        }

                        div { class: "max-h-[32vh] overflow-y-auto pr-1",
                            TabHeader {
                                sort_column: sort_column(),
                                sort_direction: sort_direction(),
                                on_sort: move |col| toggle_sort(col),
                            }

                            if groups.is_empty() {
                                p { class: "text-sm opacity-60 px-4 py-6 text-center",
                                    "No lineage found."
                                }
                            } else {
                                LineageTable { groups: groups.clone() }
                            }
                        }

                        Pagination {
                            current_page: move |new_page| current_page.set(new_page),
                            page,
                            total_pages,
                            total_lineages: *total_lineages,
                        }
                    }
                }
            }
        }
        // Snapshot still building — the `use_effect` above is polling.
        Some(Ok(None)) => rsx! {
            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body items-center text-center gap-2",
                    span { class: "loading loading-dots loading-lg" }
                    p { class: "text-sm opacity-60", "Building the lineage index..." }
                }
            }
        },
        Some(Err(e)) => rsx! {
            p { "Error: {e}" }
        },
        None => rsx! {
            p { "Loading..." }
        },
    }
}

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::homepage::family::DynamicalFamily;
use crate::homepage::interaction::{Pagination, SortColumn, SortDirection, PAGE_SIZE};

/// Columns needed to fetch a page of branches (best + others) and classify
/// their dynamical family in the same round-trip — `branches` joined
/// directly to `kf_state` (via each branch's best hypothesis), instead of
/// fetching `branches` alone and then re-joining separately just for the
/// family.
#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
struct BranchOrbitalRow {
    branch_id: i64,
    lineage_id: i64,
    designation: String,
    lineage_designation: String,
    cumulative_llr: f64,
    n_real_updates: i64,
    ra: f64,
    dec: f64,
    ra_dot: f64,
    dec_dot: f64,
    rho: f64,
    rho_dot: f64,
    epoch: f64,
    r_obs_x: f64,
    r_obs_y: f64,
    r_obs_z: f64,
    v_obs_x: f64,
    v_obs_y: f64,
    v_obs_z: f64,
}

/// Columns needed to rank every lineage matching the search filter by
/// dynamical family (used only when sorting by `SortColumn::Family`).
#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
struct RankingOrbitalRow {
    lineage_id: i64,
    ra: f64,
    dec: f64,
    ra_dot: f64,
    dec_dot: f64,
    rho: f64,
    rho_dot: f64,
    epoch: f64,
    r_obs_x: f64,
    r_obs_y: f64,
    r_obs_z: f64,
    v_obs_x: f64,
    v_obs_y: f64,
    v_obs_z: f64,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct Branch {
    branch_id: i64,
    lineage_id: i64,
    designation: String,
    lineage_designation: String,
    cumulative_llr: f64,
    n_real_updates: i64,
    family: DynamicalFamily,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct LineageGroup {
    lineage_id: i64,
    best: Branch,
    others: Vec<Branch>,
}

/// SQL fragment that neutralizes non-finite floats (NaN/±Infinity) to 0.0.
/// Applied everywhere cumulative_llr is used for ordering — Postgres treats
/// NaN as *larger* than any other float (including Infinity) when sorting,
/// so leaving it unsanitized silently corrupts both "best per lineage"
/// selection and column sorting.
const SANITIZED_LLR_EXPR: &str = "
    CASE
        WHEN cumulative_llr = 'NaN'::double precision THEN 0
        WHEN cumulative_llr = 'Infinity'::double precision THEN 0
        WHEN cumulative_llr = '-Infinity'::double precision THEN 0
        ELSE cumulative_llr
    END
";

#[server]
async fn list_lineages(
    page: i64,
    sort_column: Option<SortColumn>,
    sort_direction: SortDirection,
    search_query: String,
) -> Result<(Vec<LineageGroup>, i64), ServerFnError> {
    use crate::{
        get_pool,
        homepage::{
            family::classify_from_attributable_state,
            interaction::{SortColumn, PAGE_SIZE},
        },
    };
    use nalgebra::Vector3;
    use rayon::prelude::*;

    let pool = get_pool().await;

    // Empty search -> match everything (NULL pattern), otherwise a
    // case-insensitive substring match on lineage_designation. `%` is
    // appended on both sides here in Rust rather than in SQL, so the
    // user's raw text stays a plain bound value.
    let pattern: Option<String> = {
        let trimmed = search_query.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(format!("%{trimmed}%"))
        }
    };

    let total_lineages: (i64,) = sqlx::query_as(
        "SELECT COUNT(DISTINCT lineage_id) FROM branches
         WHERE $1::text IS NULL OR lineage_designation ILIKE $1",
    )
    .bind(&pattern)
    .fetch_one(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let offset = page * PAGE_SIZE;

    // Sorting by dynamical family can't be expressed in SQL — the family is
    // derived from kf_state (attributable state -> Cartesian -> Keplerian),
    // not stored as a column. So this path scans every lineage matching the
    // search filter, classifies its best branch, sorts in Rust, then
    // paginates. Same conversion cost as the (a, e) plot's full-population
    // query — cheap (closed-form arithmetic), just heavier than the other
    // sort columns which stay 100% SQL.
    let ordered_ids: Vec<i64> = if sort_column == Some(SortColumn::Family) {
        let ranking_query = format!(
            "WITH sanitized AS (
                SELECT
                    branch_id,
                    lineage_id,
                    {SANITIZED_LLR_EXPR} AS cumulative_llr
                FROM branches
                WHERE $1::text IS NULL OR lineage_designation ILIKE $1
            ),
            best_branches AS (
                SELECT DISTINCT ON (lineage_id) lineage_id, branch_id
                FROM sanitized
                ORDER BY lineage_id, cumulative_llr DESC
            )
            SELECT bb.lineage_id,
                    ks.ra, ks.dec, ks.ra_dot, ks.dec_dot, ks.rho, ks.rho_dot, ks.epoch,
                    ks.r_obs_x, ks.r_obs_y, ks.r_obs_z, ks.v_obs_x, ks.v_obs_y, ks.v_obs_z
            FROM best_branches bb
            CROSS JOIN LATERAL (
                SELECT hypothesis_id
                FROM hypotheses h
                WHERE h.branch_id = bb.branch_id
                ORDER BY h.log_weight DESC
                LIMIT 1
            ) bh
            JOIN kf_state ks ON ks.hypothesis_id = bh.hypothesis_id"
        );

        let rows: Vec<RankingOrbitalRow> = sqlx::query_as(sqlx::AssertSqlSafe(ranking_query))
            .bind(&pattern)
            .fetch_all(pool)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;

        let mut ranked: Vec<(i64, DynamicalFamily)> = rows
            .into_par_iter()
            .map(|row| {
                let family = classify_from_attributable_state(
                    row.ra,
                    row.dec,
                    row.ra_dot,
                    row.dec_dot,
                    row.rho,
                    row.rho_dot,
                    row.epoch,
                    Vector3::new(row.r_obs_x, row.r_obs_y, row.r_obs_z),
                    Vector3::new(row.v_obs_x, row.v_obs_y, row.v_obs_z),
                )
                .unwrap_or(DynamicalFamily::Unknown);
                (row.lineage_id, family)
            })
            .collect();

        ranked.sort_by(|a, b| {
            let ord = a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0));
            match sort_direction {
                SortDirection::Asc => ord,
                SortDirection::Desc => ord.reverse(),
            }
        });

        ranked
            .into_iter()
            .skip(offset as usize)
            .take(PAGE_SIZE as usize)
            .map(|(lineage_id, _)| lineage_id)
            .collect()
    } else {
        let (order_col, dir) = match sort_column {
            Some(SortColumn::CumulativeLlr) => ("cumulative_llr", sort_direction.sql()),
            Some(SortColumn::Updates) => ("n_real_updates", sort_direction.sql()),
            Some(SortColumn::Family) => unreachable!("handled above"),
            None => ("lineage_id", "ASC"),
        };

        // SAFE: order_col/dir come from a closed match over SortColumn/SortDirection
        // just above — never raw user input. `pattern` (real user input) stays a
        // bound parameter ($3), never interpolated.
        let lineage_query = format!(
            "WITH sanitized AS (
                SELECT
                    lineage_id,
                    lineage_designation,
                    {SANITIZED_LLR_EXPR} AS cumulative_llr,
                    n_real_updates
                FROM branches
                WHERE $3::text IS NULL OR lineage_designation ILIKE $3
            ),
            best_branches AS (
                SELECT DISTINCT ON (lineage_id) lineage_id, cumulative_llr, n_real_updates
                FROM sanitized
                ORDER BY lineage_id, cumulative_llr DESC
            )
            SELECT lineage_id FROM best_branches
            ORDER BY {order_col} {dir}
            LIMIT $1 OFFSET $2"
        );

        let page_lineage_ids: Vec<(i64,)> = sqlx::query_as(sqlx::AssertSqlSafe(lineage_query))
            .bind(PAGE_SIZE)
            .bind(offset)
            .bind(&pattern)
            .fetch_all(pool)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;

        page_lineage_ids.into_iter().map(|(id,)| id).collect()
    };

    if ordered_ids.is_empty() {
        return Ok((Vec::new(), total_lineages.0));
    }

    let branches_query = format!(
        "SELECT b.branch_id, b.lineage_id, b.designation, b.lineage_designation,
                {SANITIZED_LLR_EXPR} AS cumulative_llr, b.n_real_updates,
                ks.ra, ks.dec, ks.ra_dot, ks.dec_dot, ks.rho, ks.rho_dot, ks.epoch,
                ks.r_obs_x, ks.r_obs_y, ks.r_obs_z, ks.v_obs_x, ks.v_obs_y, ks.v_obs_z
         FROM branches b
         CROSS JOIN LATERAL (
             SELECT hypothesis_id
             FROM hypotheses h
             WHERE h.branch_id = b.branch_id
             ORDER BY h.log_weight DESC
             LIMIT 1
         ) bh
         JOIN kf_state ks ON ks.hypothesis_id = bh.hypothesis_id
         WHERE b.lineage_id = ANY($1)
         ORDER BY b.lineage_id, cumulative_llr DESC"
    );

    let branch_rows: Vec<BranchOrbitalRow> = sqlx::query_as(sqlx::AssertSqlSafe(branches_query))
        .bind(&ordered_ids)
        .fetch_all(pool)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    let branches: Vec<Branch> = branch_rows
        .into_par_iter()
        .map(|row| {
            let family = classify_from_attributable_state(
                row.ra,
                row.dec,
                row.ra_dot,
                row.dec_dot,
                row.rho,
                row.rho_dot,
                row.epoch,
                Vector3::new(row.r_obs_x, row.r_obs_y, row.r_obs_z),
                Vector3::new(row.v_obs_x, row.v_obs_y, row.v_obs_z),
            )
            .unwrap_or(DynamicalFamily::Unknown);

            Branch {
                branch_id: row.branch_id,
                lineage_id: row.lineage_id,
                designation: row.designation,
                lineage_designation: row.lineage_designation,
                cumulative_llr: row.cumulative_llr,
                n_real_updates: row.n_real_updates,
                family,
            }
        })
        .collect();

    let mut groups: Vec<LineageGroup> = Vec::new();
    for branch in branches {
        match groups.last_mut() {
            Some(last) if last.lineage_id == branch.lineage_id => {
                last.others.push(branch);
            }
            _ => groups.push(LineageGroup {
                lineage_id: branch.lineage_id,
                best: branch,
                others: Vec::new(),
            }),
        }
    }

    let position: HashMap<i64, usize> = ordered_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    groups.sort_by_key(|g| position[&g.lineage_id]);

    Ok((groups, total_lineages.0))
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

    rsx! {
        div { class: "grid grid-cols-5 gap-4 px-4 py-2 text-sm font-semibold opacity-60",
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
        }
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
                        div { class: "grid grid-cols-5 gap-4 items-center",
                            span { class: "font-medium", "{group.best.designation}" }
                            span { "{group.best.lineage_designation}" }
                            span {
                                span {
                                    class: "badge badge-sm text-white border-0",
                                    style: "background-color: {group.best.family.color()};",
                                    "{group.best.family}"
                                }
                            }
                            span { "{group.best.cumulative_llr:.2}" }
                            span { "{group.best.n_real_updates}" }
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
                                        class: "grid grid-cols-5 gap-4 items-center px-2 py-1 text-sm hover:bg-base-200 rounded",
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

#[component]
pub fn BranchTab(search_query: String) -> Element {
    let mut current_page = use_signal(|| 0_i64);
    let mut sort_column = use_signal(|| None::<SortColumn>);
    let mut sort_direction = use_signal(|| SortDirection::Desc);

    let search_query_for_resource = search_query.clone();

    let lineages_resource =
        use_resource(use_reactive!(|(search_query_for_resource,)| async move {
            list_lineages(
                current_page(),
                sort_column(),
                sort_direction(),
                search_query_for_resource,
            )
            .await
        }));

    // Reset to page 0 whenever the search text changes.
    use_effect(use_reactive!(|(search_query,)| {
        let _ = search_query;
        current_page.set(0);
    }));

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
        Some(Ok((groups, total_lineages))) => {
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
        Some(Err(e)) => rsx! {
            p { "Error: {e}" }
        },
        None => rsx! {
            p { "Loading..." }
        },
    }
}

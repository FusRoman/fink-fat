use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::format_epoch::format_epoch;
use crate::homepage::family::DynamicalFamily;

/// Summary of a lineage's best branch (highest `cumulative_llr`), for the
/// identity card at the top-left of the lineage page.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct LineageSummary {
    pub lineage_id: i64,
    pub lineage_designation: String,
    pub branch_id: i64,
    pub branch_designation: String,
    pub family: DynamicalFamily,
    pub cumulative_llr: f64,
    pub n_real_updates: i64,
    pub n_observations: i64,
    pub arc_length_days: f64,
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub ra: f64,
    pub dec: f64,
    pub rho: f64,
    pub rho_dot: f64,
    pub epoch: f64,
}

#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
struct LineageSummaryRow {
    lineage_id: i64,
    lineage_designation: String,
    branch_id: i64,
    branch_designation: String,
    cumulative_llr: f64,
    n_real_updates: i64,
    dynamic_family: String,
    semi_major_axis: f64,
    eccentricity: f64,
    ra: f64,
    dec: f64,
    rho: f64,
    rho_dot: f64,
    epoch: f64,
    n_observations: i64,
    arc_length_days: f64,
}

/// Same non-finite sanitization used by `homepage::branch_tab::list_lineages`,
/// duplicated here rather than shared because it is a tiny SQL fragment tied
/// to the literal column name it sanitizes in each query.
const SANITIZED_LLR_EXPR: &str = "
    CASE
        WHEN cumulative_llr = 'NaN'::double precision THEN 0
        WHEN cumulative_llr = 'Infinity'::double precision THEN 0
        WHEN cumulative_llr = '-Infinity'::double precision THEN 0
        ELSE cumulative_llr
    END
";

#[server]
pub async fn get_lineage_summary(
    lineage_designation: String,
) -> Result<Option<LineageSummary>, ServerFnError> {
    use crate::get_pool;

    let pool = get_pool().await;

    let query = format!(
        "WITH best_branch AS (
            SELECT branch_id, lineage_id, lineage_designation, designation,
                   {SANITIZED_LLR_EXPR} AS cumulative_llr, n_real_updates, arc_length_days
            FROM branches
            WHERE lineage_designation = $1
            ORDER BY {SANITIZED_LLR_EXPR} DESC
            LIMIT 1
        )
        SELECT
            bb.lineage_id, bb.lineage_designation,
            bb.branch_id, bb.designation AS branch_designation,
            bb.cumulative_llr, bb.n_real_updates,
            ks.dynamic_family, ks.semi_major_axis, ks.eccentricity,
            ks.ra, ks.dec, ks.rho, ks.rho_dot, ks.epoch,
            COALESCE(agg.n_observations, 0) AS n_observations,
            bb.arc_length_days
        FROM best_branch bb
        CROSS JOIN LATERAL (
            SELECT hypothesis_id
            FROM hypotheses h
            WHERE h.branch_id = bb.branch_id
            ORDER BY h.log_weight DESC
            LIMIT 1
        ) bh
        JOIN kf_state ks ON ks.hypothesis_id = bh.hypothesis_id
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS n_observations
            FROM branch_observations bo
            WHERE bo.branch_id = bb.branch_id
        ) agg ON true"
    );

    let row: Option<LineageSummaryRow> = sqlx::query_as(sqlx::AssertSqlSafe(query))
        .bind(&lineage_designation)
        .fetch_optional(pool)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(row.map(|r| LineageSummary {
        lineage_id: r.lineage_id,
        lineage_designation: r.lineage_designation,
        branch_id: r.branch_id,
        branch_designation: r.branch_designation,
        family: DynamicalFamily::from_label(&r.dynamic_family),
        cumulative_llr: r.cumulative_llr,
        n_real_updates: r.n_real_updates,
        n_observations: r.n_observations,
        arc_length_days: r.arc_length_days,
        semi_major_axis: r.semi_major_axis,
        eccentricity: r.eccentricity,
        ra: r.ra,
        dec: r.dec,
        rho: r.rho,
        rho_dot: r.rho_dot,
        epoch: r.epoch,
    }))
}

/// Identity card: lineage designation, family badge, and the key numbers
/// (arc length, point count, orbit) at a glance. Top-left of the page.
#[component]
pub fn IdentityCard(summary: Option<LineageSummary>) -> Element {
    let Some(summary) = summary else {
        return rsx! {
            div { class: "card bg-base-100 shadow-sm",
                div { class: "card-body",
                    p { class: "text-sm opacity-60", "Lineage not found." }
                }
            }
        };
    };

    let perihelion = summary.semi_major_axis * (1.0 - summary.eccentricity);
    let aphelion = summary.semi_major_axis * (1.0 + summary.eccentricity);

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-3",
                div {
                    h1 { class: "text-3xl font-bold tracking-tight", "{summary.lineage_designation}" }
                    p { class: "text-xs opacity-60",
                        "branch {summary.branch_designation} · lineage_id {summary.lineage_id}"
                    }
                }

                span {
                    class: "badge badge-lg text-white border-0 w-fit",
                    style: "background-color: {summary.family.color()};",
                    "{summary.family}"
                }

                div { class: "divider my-0" }

                div { class: "grid grid-cols-2 gap-x-4 gap-y-2 text-sm",
                    IdentityStat {
                        label: "Arc length",
                        value: format!("{:.2} days", summary.arc_length_days),
                    }
                    IdentityStat {
                        label: "Observations",
                        value: format!("{}", summary.n_observations),
                    }
                    IdentityStat {
                        label: "Real updates",
                        value: format!("{}", summary.n_real_updates),
                    }
                    IdentityStat {
                        label: "Cumulative LLR",
                        value: format!("{:.2}", summary.cumulative_llr),
                    }
                    IdentityStat {
                        label: "Semi-major axis",
                        value: format!("{:.4} AU", summary.semi_major_axis),
                    }
                    IdentityStat {
                        label: "Eccentricity",
                        value: format!("{:.4}", summary.eccentricity),
                    }
                    IdentityStat {
                        label: "Perihelion",
                        value: format!("{:.4} AU", perihelion),
                    }
                    IdentityStat {
                        label: "Aphelion",
                        value: format!("{:.4} AU", aphelion),
                    }
                    IdentityStat {
                        label: "ρ (range)",
                        value: format!("{:.4} AU", summary.rho),
                    }
                    IdentityStat {
                        label: "ρ̇ (range rate)",
                        value: format!("{:.2e} AU/day", summary.rho_dot),
                    }
                }

                p { class: "text-xs opacity-50", "State epoch: {format_epoch(summary.epoch)}" }

                Link {
                    to: crate::Route::OrbitFitPage {
                        lineage_id: summary.lineage_designation.clone(),
                    },
                    class: "btn btn-sm btn-primary",
                    "Fit orbit (n-body)"
                }
            }
        }
    }
}

#[component]
fn IdentityStat(label: &'static str, value: String) -> Element {
    rsx! {
        div {
            p { class: "text-xs opacity-60", "{label}" }
            p { class: "font-medium", "{value}" }
        }
    }
}

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::format_epoch::format_epoch;
use crate::homepage::family::DynamicalFamily;
use crate::homepage::quality_tier::QualityTier;

/// Summary of a lineage's best branch (highest `cumulative_llr`), for the
/// identity card at the top-left of the lineage page.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct LineageSummary {
    pub lineage_id: i64,
    pub lineage_designation: String,
    pub branch_id: i64,
    pub branch_designation: String,
    pub family: DynamicalFamily,
    pub quality_tier: QualityTier,
    pub cumulative_llr: f64,
    pub n_real_updates: i64,
    pub n_observations: i64,
    pub arc_length_days: f64,
    /// Semi-major axis, eccentricity and epoch below come from the best
    /// available orbit estimate — an `orbit_fits` row (n-body differential
    /// correction preferred over a Gauss-IOD-only fit) if one exists for
    /// this lineage, falling back to the Kalman-bank state otherwise. See
    /// [`orbit_source_label`](Self::orbit_source_label) for which one this
    /// particular value came from.
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub epoch: f64,
    /// "N-body fit" / "IOD fit" / "Kalman estimate" — which source
    /// `semi_major_axis`/`eccentricity`/`epoch` above came from.
    pub orbit_source_label: String,
    /// Always the Kalman-bank topocentric attributable state — RA/Dec/range
    /// aren't part of an `orbit_fits` row's heliocentric Keplerian elements,
    /// so these stay Kalman-sourced regardless of `orbit_source_label`.
    pub ra: f64,
    pub dec: f64,
    pub rho: f64,
    pub rho_dot: f64,
}

#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
struct LineageSummaryRow {
    lineage_id: i64,
    lineage_designation: String,
    branch_id: i64,
    branch_designation: String,
    cumulative_llr: f64,
    n_real_updates: i64,
    n_nights: i64,
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

/// This lineage's best branch's latest `orbit_fits` row, preferring an
/// n-body differential-correction fit over a Gauss-IOD-only one whenever
/// both exist (rather than always just the most recent by timestamp, as
/// `orbit_fit::latest::get_latest_orbit_fit_result` does — that page shows
/// "the fit you just ran", this card shows "the best estimate we have").
/// `None` if the lineage has never been fitted, in which case the card falls
/// back to the Kalman-bank state already read by the caller's main query.
#[cfg(feature = "server")]
async fn best_orbit_fit(
    pool: &sqlx::PgPool,
    lineage_designation: &str,
) -> Result<Option<(crate::best_orbit::OrbitCandidate, f64)>, sqlx::Error> {
    use crate::best_orbit::{OrbitCandidate, PREFER_NBODY_ORDER_BY};
    use crate::fit_pipeline::fit::FitMethod;

    #[derive(sqlx::FromRow)]
    struct Row {
        fit_method: String,
        keplerian: sqlx::types::Json<crate::fit_pipeline::fit::KeplerianView>,
        reference_epoch: f64,
    }

    let query = format!(
        "SELECT fit_method, keplerian, reference_epoch
         FROM orbit_fits
         WHERE lineage_designation = $1
         ORDER BY {PREFER_NBODY_ORDER_BY}
         LIMIT 1"
    );

    let row: Option<Row> = sqlx::query_as(sqlx::AssertSqlSafe(query))
        .bind(lineage_designation)
        .fetch_optional(pool)
        .await?;

    Ok(row.map(|r| {
        (
            OrbitCandidate {
                fit_method: FitMethod::from_column(&r.fit_method),
                semi_major_axis_au: r.keplerian.0.semi_major_axis_au,
                eccentricity: r.keplerian.0.eccentricity,
            },
            r.reference_epoch,
        )
    }))
}

/// This branch's quality tier, computed the same way
/// `homepage::quality_tier::assign_quality_tier` is fed for the homepage
/// table — but scoped to a single `branch_id` with small, targeted queries
/// rather than the homepage snapshot's whole-table batch load (which would
/// be wasteful to run just to render one card).
#[cfg(feature = "server")]
async fn compute_quality_tier(
    pool: &sqlx::PgPool,
    branch_id: i64,
    n_nights: i64,
) -> Result<QualityTier, sqlx::Error> {
    use crate::fit_pipeline::fit::FitMethod;
    use crate::fit_pipeline::params::{MIN_BASELINE_DAYS, MIN_OBSERVATIONS};
    use crate::homepage::quality_tier::{assign_quality_tier, LatestFit};

    let eligible: bool = sqlx::query_scalar(
        "SELECT COUNT(*) >= $1 AND (COALESCE(MAX(o.mjd_tt) - MIN(o.mjd_tt), 0)) >= $2
         FROM branch_observations bo
         JOIN observations o ON o.id = bo.obs_id
         WHERE bo.branch_id = $3",
    )
    .bind(MIN_OBSERVATIONS as i64)
    .bind(MIN_BASELINE_DAYS)
    .bind(branch_id)
    .fetch_one(pool)
    .await?;

    let latest_fit_row: Option<(String, i32, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
        "SELECT fit_method, num_measurements, fitted_at
         FROM orbit_fits WHERE branch_id = $1
         ORDER BY fitted_at DESC LIMIT 1",
    )
    .bind(branch_id)
    .fetch_optional(pool)
    .await?;
    let latest_fit = latest_fit_row.map(|(fit_method, num_measurements, fitted_at)| LatestFit {
        fit_method: FitMethod::from_column(&fit_method),
        num_measurements,
        fitted_at,
    });

    let latest_failure_at: Option<chrono::DateTime<chrono::Utc>> = sqlx::query_scalar(
        "SELECT attempted_at FROM orbit_fit_failures
         WHERE branch_id = $1 ORDER BY attempted_at DESC LIMIT 1",
    )
    .bind(branch_id)
    .fetch_optional(pool)
    .await?;

    let well_sampled_nights: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM (
            SELECT o.night_id, COUNT(*) AS n
            FROM branch_observations bo
            JOIN observations o ON o.id = bo.obs_id
            WHERE bo.branch_id = $1
            GROUP BY o.night_id
         ) per_night WHERE n >= 2",
    )
    .bind(branch_id)
    .fetch_one(pool)
    .await?;

    Ok(assign_quality_tier(
        eligible,
        latest_fit.as_ref(),
        latest_failure_at,
        n_nights,
        well_sampled_nights,
    ))
}

#[server]
pub async fn get_lineage_summary(
    lineage_designation: String,
) -> Result<Option<LineageSummary>, ServerFnError> {
    use crate::best_orbit::resolve_best_orbit;
    use crate::get_pool;

    let pool = get_pool().await;

    let query = format!(
        "WITH best_branch AS (
            SELECT branch_id, lineage_id, lineage_designation, designation,
                   {SANITIZED_LLR_EXPR} AS cumulative_llr, n_real_updates, arc_length_days, n_nights
            FROM branches
            WHERE lineage_designation = $1
            ORDER BY {SANITIZED_LLR_EXPR} DESC
            LIMIT 1
        )
        SELECT
            bb.lineage_id, bb.lineage_designation,
            bb.branch_id, bb.designation AS branch_designation,
            bb.cumulative_llr, bb.n_real_updates, bb.n_nights,
            ks.semi_major_axis, ks.eccentricity,
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

    let Some(r) = row else {
        return Ok(None);
    };

    let quality_tier = compute_quality_tier(pool, r.branch_id, r.n_nights)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    let best_fit = best_orbit_fit(pool, &lineage_designation)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    let epoch = best_fit
        .as_ref()
        .map(|(_, reference_epoch)| *reference_epoch)
        .unwrap_or(r.epoch);

    let best_orbit = resolve_best_orbit(
        best_fit.map(|(candidate, _)| candidate),
        (r.semi_major_axis, r.eccentricity),
    );
    let semi_major_axis = best_orbit.semi_major_axis_au;
    let eccentricity = best_orbit.eccentricity;
    let family = best_orbit.family;
    let orbit_source_label = best_orbit.source.label();

    Ok(Some(LineageSummary {
        lineage_id: r.lineage_id,
        lineage_designation: r.lineage_designation,
        branch_id: r.branch_id,
        branch_designation: r.branch_designation,
        family,
        quality_tier,
        cumulative_llr: r.cumulative_llr,
        n_real_updates: r.n_real_updates,
        n_observations: r.n_observations,
        arc_length_days: r.arc_length_days,
        semi_major_axis,
        eccentricity,
        epoch,
        orbit_source_label: orbit_source_label.to_string(),
        ra: r.ra,
        dec: r.dec,
        rho: r.rho,
        rho_dot: r.rho_dot,
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

                div { class: "flex flex-wrap items-center gap-2",
                    span {
                        class: "badge badge-lg text-white border-0 w-fit",
                        style: "background-color: {summary.family.color()};",
                        "{summary.family}"
                    }
                    span {
                        class: "badge badge-lg {summary.quality_tier.badge_class()}",
                        title: "{summary.quality_tier.label()}",
                        "{summary.quality_tier.glyph()} {summary.quality_tier.label()}"
                    }
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

                p { class: "text-xs opacity-50",
                    "Orbit: {summary.orbit_source_label} · State epoch: {format_epoch(summary.epoch)}"
                }

                Link {
                    to: crate::Route::OrbitFitPage {
                        lineage_id: summary.lineage_designation.clone(),
                    },
                    class: "btn btn-sm btn-primary",
                    "Fit orbit"
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

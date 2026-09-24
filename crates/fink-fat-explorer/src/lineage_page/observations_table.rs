use crate::format_epoch::iso_utc;
use crate::survey::{observation_link, ObsLink};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// One real observation belonging to a lineage's best branch, in track
/// order. Now defined in the shared [`fink_fat_ades::model`] crate (the
/// `fink-fat submit` CLI needs the exact same shape to build an ADES
/// document from its own, synchronous-`postgres`-backed query) and
/// re-exported here under this module's original path.
pub use fink_fat_ades::model::ObservationRow;

/// The lineage page's observatory link for one observation — an inherent
/// method can't be added to [`ObservationRow`] directly any more (it's
/// defined in another crate), so this extension trait keeps the `obs.link()`
/// call site unchanged.
trait ObservationRowExt {
    fn link(&self) -> ObsLink;
}

impl ObservationRowExt for ObservationRow {
    fn link(&self) -> ObsLink {
        observation_link(&self.object_id, &self.mpc_code_obs)
    }
}

#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
struct ObservationRowSql {
    id: i64,
    object_id: String,
    position: i32,
    mjd_tt: f64,
    ra: f64,
    ra_err: f64,
    dec: f64,
    dec_err: f64,
    magnitude: f64,
    mag_err: f64,
    filter: i16,
    mpc_code_obs: String,
    night_id: i64,
}

/// Maps a raw SQL row into the shared, DB-client-agnostic [`ObservationRow`].
/// A plain function rather than `impl From<ObservationRowSql> for
/// ObservationRow`: with `ObservationRow` now defined in `fink-fat-ades`,
/// that impl would implement a foreign trait (`From`) for a foreign type
/// from this crate's `ObservationRowSql` parameter, which the orphan rule
/// only allows when the target type itself is local.
fn observation_row_from_sql(r: ObservationRowSql) -> ObservationRow {
    ObservationRow {
        id: r.id,
        object_id: r.object_id,
        position: r.position,
        mjd_tt: r.mjd_tt,
        ra: r.ra,
        ra_err: r.ra_err,
        dec: r.dec,
        dec_err: r.dec_err,
        magnitude: r.magnitude,
        mag_err: r.mag_err,
        filter: r.filter,
        mpc_code_obs: r.mpc_code_obs,
        night_id: r.night_id,
    }
}

/// Fetch a branch's observations in track order — the query shared by
/// [`get_lineage_observations`] and the ADES export server fn
/// (`crate::ades::server_fns::export_and_validate_ades`), so both read from
/// the exact same rows rather than duplicating the SQL.
#[cfg(feature = "server")]
pub(crate) async fn fetch_branch_observations(
    branch_id: i64,
) -> Result<Vec<ObservationRow>, sqlx::Error> {
    use crate::get_pool;

    let pool = get_pool().await;
    let rows: Vec<ObservationRowSql> = sqlx::query_as(
        "SELECT o.id, o.object_id, bo.position, o.mjd_tt, o.ra, o.ra_err, o.dec, o.dec_err,
                o.magnitude, o.mag_err, o.filter, o.mpc_code_obs, o.night_id
         FROM branch_observations bo
         JOIN observations o ON o.id = bo.obs_id
         WHERE bo.branch_id = $1
         ORDER BY bo.position",
    )
    .bind(branch_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(observation_row_from_sql).collect())
}

/// A lineage's best branch (highest `cumulative_llr`) and its observations —
/// the single resolution point for "which branch does this lineage's page
/// operate on". Callers that need to act on that branch specifically (e.g.
/// `orbit_fit_page`, which must fit and later store results against the same
/// `branch_id` it displayed observations for) get it from here rather than
/// re-resolving "the lineage's best branch" independently, which previously
/// risked the fit page silently picking a different branch than the one its
/// observation list was built from.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct LineageObservations {
    pub branch_id: i64,
    pub observations: Vec<ObservationRow>,
}

/// Every real observation of a lineage's best branch, ordered as they were
/// absorbed by the filter. `None` if the lineage has no branches at all.
#[server]
pub async fn get_lineage_observations(
    lineage_designation: String,
) -> Result<Option<LineageObservations>, ServerFnError> {
    use crate::orbit_fit::run::resolve_best_branch_id;

    let Some(branch_id) = resolve_best_branch_id(&lineage_designation)
        .await
        .map_err(ServerFnError::new)?
    else {
        return Ok(None);
    };

    let observations = fetch_branch_observations(branch_id)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(Some(LineageObservations {
        branch_id,
        observations,
    }))
}

/// Scrollable table of every observation used by the lineage's best branch —
/// position, epoch, sky position with errors, and photometry.
#[component]
pub fn ObservationsTable(observations: Vec<ObservationRow>) -> Element {
    let mut show_utc = use_signal(|| true);

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                h2 { class: "card-title", "Observations ({observations.len()})" }

                div { class: "max-h-[50vh] overflow-y-auto",
                    table { class: "table table-zebra table-pin-rows table-sm",
                        thead {
                            tr {
                                th { "#" }
                                th { "ObjectId" }
                                th {
                                    label { class: "flex items-center gap-2 cursor-pointer normal-case font-normal",
                                        span {
                                            class: if !show_utc() { "font-bold" } else { "text-base-content/50" },
                                            "MJD (TT)"
                                        }
                                        input {
                                            r#type: "checkbox",
                                            class: "toggle toggle-sm",
                                            checked: show_utc(),
                                            onchange: move |evt| show_utc.set(evt.checked()),
                                        }
                                        span {
                                            class: if show_utc() { "font-bold" } else { "text-base-content/50" },
                                            "ISO (UTC)"
                                        }
                                    }
                                }
                                th { "RA (deg)" }
                                th { "Dec (deg)" }
                                th { "Magnitude" }
                                th { "Filter" }
                                th { "Observatory" }
                            }
                        }
                        tbody {
                            for obs in &observations {
                                tr { key: "{obs.id}",
                                    td { "{obs.position}" }
                                    td {
                                        match obs.link() {
                                            ObsLink::Valid(href) => rsx! {
                                                a {
                                                    href: "{href}",
                                                    target: "_blank",
                                                    rel: "noopener noreferrer",
                                                    class: "link link-secondary",
                                                    "{obs.object_id}"
                                                }
                                            },
                                            ObsLink::Unknown(code) => rsx! {
                                                span { class: "text-error", title: "Unknown observatory code: {code}",
                                                    "{obs.object_id} (unknown observatory)"
                                                }
                                            },
                                        }
                                    }
                                    td {
                                        if show_utc() {
                                            "{iso_utc(obs.mjd_tt)}"
                                        } else {
                                            "{obs.mjd_tt:.5}"
                                        }
                                    }
                                    td {
                                        "{obs.ra.to_degrees():.6} ± {(obs.ra_err.to_degrees() * 3600.0):.3}\""
                                    }
                                    td {
                                        "{obs.dec.to_degrees():.6} ± {(obs.dec_err.to_degrees() * 3600.0):.3}\""
                                    }
                                    td { "{obs.magnitude:.2} ± {obs.mag_err:.2}" }
                                    td { "{obs.filter}" }
                                    td { "{obs.mpc_code_obs}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

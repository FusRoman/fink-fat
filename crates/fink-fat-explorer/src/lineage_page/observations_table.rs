use crate::format_epoch::iso_utc;
use crate::survey::{observation_link, ObsLink};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

impl ObservationRow {
    fn link(&self) -> ObsLink {
        observation_link(&self.object_id, &self.mpc_code_obs)
    }
}

/// One real observation belonging to a lineage's best branch, in track order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationRow {
    pub id: i64,
    pub object_id: String,
    pub position: i32,
    pub mjd_tt: f64,
    pub ra: f64,
    pub ra_err: f64,
    pub dec: f64,
    pub dec_err: f64,
    pub magnitude: f64,
    pub mag_err: f64,
    pub filter: i16,
    pub mpc_code_obs: String,
    /// Observatory-longitude-aware observation night bucket, pre-computed at
    /// ingestion (`observations.night_id`) — used by the ADES export to
    /// detect nights with a single observation ("singleton" nights, which
    /// the MPC rejects an entire batch for containing).
    pub night_id: i64,
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

impl From<ObservationRowSql> for ObservationRow {
    fn from(r: ObservationRowSql) -> Self {
        Self {
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

    Ok(rows.into_iter().map(ObservationRow::from).collect())
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

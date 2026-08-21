use crate::format_epoch::iso_utc;
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

pub enum Survey {
    ZTF,
    LSST,
}

impl Survey {
    pub fn get_link(&self) -> &'static str {
        match self {
            Survey::ZTF => "https://ztf.fink-portal.org/",
            Survey::LSST => "https://lsst.fink-portal.org/",
        }
    }

    pub fn from_code_obs(code_obs: &str) -> Option<Self> {
        match code_obs {
            "X05" => Some(Survey::LSST),
            "I41" => Some(Survey::ZTF),
            _ => None,
        }
    }
}

enum ObsLink {
    Valid(String),
    Unknown(String),
}

impl ObservationRow {
    fn link(&self) -> ObsLink {
        match Survey::from_code_obs(&self.mpc_code_obs) {
            Some(s) => ObsLink::Valid(format!("{}{}", s.get_link(), self.object_id)),
            None => ObsLink::Unknown(self.mpc_code_obs.clone()),
        }
    }
}

/// One real observation belonging to a lineage's best branch, in track order.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
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
        }
    }
}

/// Every real observation of a lineage's best branch (highest
/// `cumulative_llr`), ordered as they were absorbed by the filter.
#[server]
pub async fn get_lineage_observations(
    lineage_designation: String,
) -> Result<Vec<ObservationRow>, ServerFnError> {
    use crate::get_pool;

    let pool = get_pool().await;

    let rows: Vec<ObservationRowSql> = sqlx::query_as(
        "WITH best_branch AS (
            SELECT branch_id
            FROM branches
            WHERE lineage_designation = $1
            ORDER BY (
                CASE
                    WHEN cumulative_llr = 'NaN'::double precision THEN 0
                    WHEN cumulative_llr = 'Infinity'::double precision THEN 0
                    WHEN cumulative_llr = '-Infinity'::double precision THEN 0
                    ELSE cumulative_llr
                END
            ) DESC
            LIMIT 1
        )
        SELECT o.id, o.object_id, bo.position, o.mjd_tt, o.ra, o.ra_err, o.dec, o.dec_err,
               o.magnitude, o.mag_err, o.filter, o.mpc_code_obs
        FROM best_branch bb
        JOIN branch_observations bo ON bo.branch_id = bb.branch_id
        JOIN observations o ON o.id = bo.obs_id
        ORDER BY bo.position",
    )
    .bind(&lineage_designation)
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(rows.into_iter().map(ObservationRow::from).collect())
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

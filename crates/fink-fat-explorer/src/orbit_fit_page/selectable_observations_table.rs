use std::collections::HashSet;

use dioxus::prelude::*;

use crate::lineage_page::observations_table::ObservationRow;

/// Same layout as `lineage_page::observations_table::ObservationsTable`, plus
/// a leading checkbox column so the user can exclude specific observations
/// from the fit. All observations are selected by default.
#[component]
pub fn SelectableObservationsTable(
    observations: Vec<ObservationRow>,
    selected: Signal<HashSet<i64>>,
) -> Element {
    let all_selected =
        !observations.is_empty() && observations.iter().all(|o| selected.read().contains(&o.id));
    let all_ids: Vec<i64> = observations.iter().map(|o| o.id).collect();

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                div { class: "flex items-center justify-between",
                    h2 { class: "card-title", "Observations ({observations.len()})" }
                    label { class: "label cursor-pointer gap-2",
                        span { class: "label-text", "Select all" }
                        input {
                            r#type: "checkbox",
                            class: "checkbox checkbox-sm",
                            checked: all_selected,
                            onchange: move |evt| {
                                let checked = evt.checked();
                                let mut selected = selected.write();
                                if checked {
                                    selected.extend(all_ids.iter().copied());
                                } else {
                                    selected.clear();
                                }
                            },
                        }
                    }
                }

                div { class: "max-h-[50vh] overflow-y-auto",
                    table { class: "table table-zebra table-pin-rows table-sm",
                        thead {
                            tr {
                                th {}
                                th { "#" }
                                th { "ObjectId" }
                                th { "MJD (TT)" }
                                th { "RA (deg)" }
                                th { "Dec (deg)" }
                                th { "Magnitude" }
                                th { "Filter" }
                                th { "Observatory" }
                            }
                        }
                        tbody {
                            for obs in observations {
                                {
                                    let obs_id = obs.id;
                                    let is_selected = selected.read().contains(&obs_id);
                                    rsx! {
                                        tr { key: "{obs.id}",
                                            td {
                                                input {
                                                    r#type: "checkbox",
                                                    class: "checkbox checkbox-sm",
                                                    checked: is_selected,
                                                    onchange: move |evt| {
                                                        let mut selected = selected.write();
                                                        if evt.checked() {
                                                            selected.insert(obs_id);
                                                        } else {
                                                            selected.remove(&obs_id);
                                                        }
                                                    },
                                                }
                                            }
                                            td { "{obs.position}" }
                                            td {
                                                a {
                                                    href: "https://ztf.fink-portal.org/{obs.object_id}",
                                                    target: "_blank",
                                                    rel: "noopener noreferrer",
                                                    class: "link link-secondary",
                                                    "{obs.object_id}"
                                                }
                                            }
                                            td { "{obs.mjd_tt:.6}" }
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
    }
}

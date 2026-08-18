use dioxus::prelude::*;

use crate::format_epoch::format_epoch;
use crate::orbit_fit::{
    history::get_orbit_fit_history, KeplerianView, ObsSelectionView, OrbitDelta, OrbitFitResult,
};

use super::residuals_plot::ResidualsPlot;

#[component]
pub fn FitResult(lineage_id: String, result: OrbitFitResult) -> Element {
    rsx! {
        div { class: "flex flex-col gap-4",
            div { class: "flex flex-col xl:flex-row gap-4 items-stretch",
                KeplerianCard { keplerian: result.keplerian.clone() }
                if let Some(delta) = result.delta_vs_kalman.clone() {
                    DeltaCard { title: "Δ vs current Kalman orbit", delta }
                }
                if let Some(delta) = result.delta_vs_previous_fit.clone() {
                    DeltaCard { title: "Δ vs previous Outfit fit", delta }
                } else {
                    div { class: "card bg-base-100 shadow-sm flex-1",
                        div { class: "card-body items-center justify-center",
                            p { class: "text-sm opacity-60", "First Outfit fit for this lineage." }
                        }
                    }
                }
            }

            MetricsCard { result: result.clone() }

            ResidualsPlot { residuals: result.residuals.clone() }

            ResidualsTable { residuals: result.residuals.clone() }

            HistorySection { lineage_id }
        }
    }
}

#[component]
fn KeplerianCard(keplerian: KeplerianView) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-2",
                h2 { class: "card-title", "Fitted orbit (Keplerian)" }
                div { class: "grid grid-cols-2 gap-x-4 gap-y-2 text-sm",
                    ElementStat {
                        label: "Semi-major axis",
                        value: format!("{:.6} AU", keplerian.semi_major_axis_au),
                        sigma: keplerian.sigma_semi_major_axis_au.map(|s| format!("± {s:.6} AU")),
                    }
                    ElementStat {
                        label: "Eccentricity",
                        value: format!("{:.6}", keplerian.eccentricity),
                        sigma: keplerian.sigma_eccentricity.map(|s| format!("± {s:.6}")),
                    }
                    ElementStat {
                        label: "Inclination",
                        value: format!("{:.4}°", keplerian.inclination_deg),
                        sigma: keplerian.sigma_inclination_deg.map(|s| format!("± {s:.4}°")),
                    }
                    ElementStat {
                        label: "Ascending node (Ω)",
                        value: format!("{:.4}°", keplerian.ascending_node_longitude_deg),
                        sigma: keplerian
                            .sigma_ascending_node_longitude_deg
                            .map(|s| format!("± {s:.4}°")),
                    }
                    ElementStat {
                        label: "Argument of periapsis (ω)",
                        value: format!("{:.4}°", keplerian.periapsis_argument_deg),
                        sigma: keplerian
                            .sigma_periapsis_argument_deg
                            .map(|s| format!("± {s:.4}°")),
                    }
                    ElementStat {
                        label: "Mean anomaly (M)",
                        value: format!("{:.4}°", keplerian.mean_anomaly_deg),
                        sigma: keplerian.sigma_mean_anomaly_deg.map(|s| format!("± {s:.4}°")),
                    }
                }
            }
        }
    }
}

#[component]
fn ElementStat(label: &'static str, value: String, sigma: Option<String>) -> Element {
    rsx! {
        div {
            p { class: "text-xs opacity-60", "{label}" }
            p { class: "font-medium", "{value}" }
            if let Some(sigma) = sigma {
                p { class: "text-xs opacity-50", "{sigma}" }
            }
        }
    }
}

#[component]
fn DeltaCard(title: &'static str, delta: OrbitDelta) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-2",
                h2 { class: "card-title", "{title}" }
                div { class: "grid grid-cols-2 gap-x-4 gap-y-2 text-sm",
                    div {
                        p { class: "text-xs opacity-60", "Δa" }
                        p { class: "font-medium", "{delta.delta_semi_major_axis_au:.6} AU" }
                    }
                    div {
                        p { class: "text-xs opacity-60", "Δe" }
                        p { class: "font-medium", "{delta.delta_eccentricity:.6}" }
                    }
                    div {
                        p { class: "text-xs opacity-60", "Δi" }
                        p { class: "font-medium", "{delta.delta_inclination_deg:.4}°" }
                    }
                    div {
                        p { class: "text-xs opacity-60", "ΔΩ" }
                        p { class: "font-medium", "{delta.delta_ascending_node_longitude_deg:.4}°" }
                    }
                    div {
                        p { class: "text-xs opacity-60", "Δω" }
                        p { class: "font-medium", "{delta.delta_periapsis_argument_deg:.4}°" }
                    }
                    div {
                        p { class: "text-xs opacity-60", "ΔM" }
                        p { class: "font-medium", "{delta.delta_mean_anomaly_deg:.4}°" }
                    }
                }
            }
        }
    }
}

#[component]
fn MetricsCard(result: OrbitFitResult) -> Element {
    let status_badge = if result.converged {
        ("badge-success", "Converged")
    } else {
        ("badge-error", "Did not converge cleanly")
    };

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-2",
                div { class: "flex items-center gap-3",
                    h2 { class: "card-title", "Fit quality" }
                    span { class: "badge {status_badge.0}", "{status_badge.1}" }
                }
                div { class: "grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-2 text-sm",
                    ElementStat {
                        label: "Normalised RMS",
                        value: format!("{:.4}", result.normalised_rms),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Reduced χ²",
                        value: format!("{:.4}", result.reduced_chi2),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Degrees of freedom",
                        value: format!("{}", result.degrees_of_freedom),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Newton iterations",
                        value: format!("{}", result.total_newton_iterations),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Observations used",
                        value: format!("{}", result.n_observations_used),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Observations rejected",
                        value: format!("{}", result.n_observations_rejected),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Scalar measurements",
                        value: format!("{}", result.num_measurements),
                        sigma: None,
                    }
                    ElementStat {
                        label: "Reference epoch",
                        value: format_epoch(result.reference_epoch),
                        sigma: None,
                    }
                }
            }
        }
    }
}

#[component]
fn ResidualsTable(residuals: Vec<crate::orbit_fit::ObsResidual>) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                h2 { class: "card-title", "Per-observation residuals ({residuals.len()})" }
                div { class: "max-h-[40vh] overflow-y-auto",
                    table { class: "table table-zebra table-pin-rows table-sm",
                        thead {
                            tr {
                                th { "MJD (TT)" }
                                th { "Δα cos δ (arcsec)" }
                                th { "Δδ (arcsec)" }
                                th { "χ" }
                                th { "Status" }
                            }
                        }
                        tbody {
                            for r in residuals {
                                tr { key: "{r.obs_id}",
                                    td { "{r.mjd_tt:.6}" }
                                    td { "{r.residual_ra_arcsec:.4}" }
                                    td { "{r.residual_dec_arcsec:.4}" }
                                    td { "{r.chi:.3}" }
                                    td {
                                        match r.selection {
                                            ObsSelectionView::Kept => rsx! {
                                                span { class: "badge badge-success badge-sm", "kept" }
                                            },
                                            ObsSelectionView::Rejected => rsx! {
                                                span { class: "badge badge-error badge-sm", "rejected" }
                                            },
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

#[component]
fn HistorySection(lineage_id: String) -> Element {
    let history_lineage_id = lineage_id.clone();
    let history_resource = use_resource(use_reactive!(|(history_lineage_id,)| {
        get_orbit_fit_history(history_lineage_id)
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                h2 { class: "card-title", "Fit history" }
                match &*history_resource.read() {
                    Some(Ok(rows)) if !rows.is_empty() => rsx! {
                        div { class: "max-h-[30vh] overflow-y-auto",
                            table { class: "table table-zebra table-pin-rows table-sm",
                                thead {
                                    tr {
                                        th { "Fitted at" }
                                        th { "Observations" }
                                        th { "Normalised RMS" }
                                        th { "a (AU)" }
                                    }
                                }
                                tbody {
                                    for row in rows {
                                        tr { key: "{row.id}",
                                            td { "{row.fitted_at}" }
                                            td { "{row.n_observations_used}" }
                                            td { "{row.normalised_rms:.4}" }
                                            td { "{row.semi_major_axis_au:.6}" }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    Some(Ok(_)) => rsx! {
                        p { class: "text-sm opacity-60", "No previous fits for this lineage." }
                    },
                    Some(Err(e)) => rsx! {
                        div { class: "alert alert-error", "Failed to load fit history: {e}" }
                    },
                    None => rsx! {
                        div { class: "flex justify-center py-4",
                            span { class: "loading loading-spinner" }
                        }
                    },
                }
            }
        }
    }
}

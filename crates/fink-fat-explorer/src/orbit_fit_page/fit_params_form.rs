use dioxus::prelude::*;

use crate::orbit_fit::{ObsErrorModelChoice, OrbitFitParams, PerturberChoice, PropagatorChoice};

use super::help_tooltip::HelpTooltip;

/// Exhaustive form for every tunable parameter of an Outfit orbit fit,
/// grouped into collapsible sections. Bound directly to a
/// `Signal<OrbitFitParams>` owned by the page. The IOD/Gauss section is
/// informational: the fit always seeds from the current Kalman orbit rather
/// than running Gauss IOD, so those fields have no effect on the outcome —
/// kept in the form for completeness and because a future "recompute without
/// a Kalman seed" mode could use them.
#[component]
pub fn FitParamsForm(
    params: Signal<OrbitFitParams>,
    launch_disabled: bool,
    on_launch: EventHandler<MouseEvent>,
) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-3",
                div { class: "flex items-center justify-between",
                    h2 { class: "card-title", "Fit parameters" }
                    button {
                        class: "btn btn-sm btn-ghost",
                        r#type: "button",
                        onclick: move |_| params.set(OrbitFitParams::default()),
                        "Reset to defaults"
                    }
                }

                ErrorModelSection { params }
                DynamicalModelSection { params }
                DifferentialCorrectionSection { params }

                // div { class: "collapse collapse-arrow bg-base-200",
                //     input { r#type: "checkbox" }
                //     div { class: "collapse-title text-sm font-medium",
                //         "IOD / Gauss (not used — the fit seeds from the current Kalman orbit)"
                //     }
                //     div { class: "collapse-content",
                //         IodSection { params }
                //     }
                // }
                div { class: "flex justify-center pt-2",
                    button {
                        class: "btn btn-outline btn-primary",
                        r#type: "button",
                        disabled: launch_disabled,
                        onclick: move |evt| on_launch.call(evt),
                        "Fit orbit"
                    }
                }
            }
        }
    }
}

#[component]
fn ErrorModelSection(params: Signal<OrbitFitParams>) -> Element {
    rsx! {
        div { class: "form-control w-full max-w-xs gap-1",
            label { class: "label",
                span { class: "label-text", "Observation error model" }
                HelpTooltip {
                    text: "Which calibrated astrometric-error model is applied to each observation before fitting. FCCT14 is the standard general-purpose choice; CBM10 and VFCC17 are alternative calibrations from the literature. Leave on FCCT14 unless comparing against a specific external pipeline.",
                }
            }
            select {
                class: "select select-bordered select-sm",
                value: match params.read().error_model {
                    ObsErrorModelChoice::Fcct14 => "fcct14",
                    ObsErrorModelChoice::Cbm10 => "cbm10",
                    ObsErrorModelChoice::Vfcc17 => "vfcc17",
                },
                onchange: move |evt| {
                    params.write().error_model = match evt.value().as_str() {
                        "cbm10" => ObsErrorModelChoice::Cbm10,
                        "vfcc17" => ObsErrorModelChoice::Vfcc17,
                        _ => ObsErrorModelChoice::Fcct14,
                    };
                },
                option { value: "fcct14", "{ObsErrorModelChoice::Fcct14.label()}" }
                option { value: "cbm10", "{ObsErrorModelChoice::Cbm10.label()}" }
                option { value: "vfcc17", "{ObsErrorModelChoice::Vfcc17.label()}" }
            }
        }
    }
}

#[component]
fn DynamicalModelSection(params: Signal<OrbitFitParams>) -> Element {
    let is_nbody = matches!(params.read().propagator, PropagatorChoice::NBody);

    rsx! {
        div { class: "flex flex-col gap-2",
            div { class: "form-control w-full max-w-xs gap-1",
                label { class: "label",
                    span { class: "label-text", "Dynamical model" }
                    HelpTooltip {
                        text: "Two-body uses pure Keplerian motion (matches what the production Kalman filter assumes); N-body integrates gravitational perturbations from the selected planets for a more physically accurate fit. N-body is slower but is the point of this tool — keep it on unless debugging.",
                    }
                }
                select {
                    class: "select select-bordered select-sm",
                    value: if is_nbody { "nbody" } else { "twobody" },
                    onchange: move |evt| {
                        params.write().propagator = if evt.value() == "nbody" {
                            PropagatorChoice::NBody
                        } else {
                            PropagatorChoice::TwoBody
                        };
                    },
                    option { value: "nbody", "N-body (perturbed)" }
                    option { value: "twobody", "Two-body (Keplerian, matches the Kalman filter)" }
                }
            }

            if is_nbody {
                div { class: "flex items-center gap-1",
                    span { class: "label-text text-xs opacity-70", "Perturbers" }
                    HelpTooltip {
                        text: "Which planets are included as N-body perturbers (the Sun is always included). Jupiter and Saturn dominate for most asteroid orbits; add more only for objects that pass close to other planets or need extra precision.",
                    }
                }
                div { class: "flex flex-wrap gap-3",
                    for planet in PerturberChoice::ALL {
                        label {
                            key: "{planet.label()}",
                            class: "label cursor-pointer gap-2",
                            input {
                                r#type: "checkbox",
                                class: "checkbox checkbox-sm",
                                checked: params.read().perturbers.contains(&planet),
                                onchange: move |evt| {
                                    let mut params = params.write();
                                    if evt.checked() {
                                        if !params.perturbers.contains(&planet) {
                                            params.perturbers.push(planet);
                                        }
                                    } else {
                                        params.perturbers.retain(|p| *p != planet);
                                    }
                                },
                            }
                            span { class: "label-text", "{planet.label()}" }
                        }
                    }
                }
                p { class: "text-xs opacity-60", "The Sun is always included as a perturber." }
            }
        }
    }
}

#[component]
fn DifferentialCorrectionSection(params: Signal<OrbitFitParams>) -> Element {
    rsx! {
        div { class: "grid grid-cols-2 md:grid-cols-3 gap-3",
            UsizeField {
                label: "Max Newton iterations",
                value: params.read().max_newton_iterations,
                on_change: move |v| params.write().max_newton_iterations = v,
                help: "Upper bound on Newton correction steps per outlier-rejection pass. Defaults to 30; raise it if the fit reports non-convergence with residuals still decreasing, lower it to fail fast on bad data. Valid range: any positive integer, typically 10-100.",
            }
            UsizeField {
                label: "Max outlier-rejection passes",
                value: params.read().max_outlier_rejection_passes,
                on_change: move |v| params.write().max_outlier_rejection_passes = v,
                help: "How many times the fit re-runs Newton correction after dropping outlier observations. Defaults to 10. Increase for noisy datasets with many bad points; 0 disables outlier rejection entirely (equivalent to turning off the checkbox below).",
            }
            F64Field {
                label: "Convergence threshold",
                value: params.read().convergence_threshold,
                on_change: move |v| params.write().convergence_threshold = v,
                help: "Relative RMS change below which the fit is considered converged. Defaults to 1e-4 (0.0001). Tighter (smaller) values demand more precision but need more iterations; looser values converge faster but less precisely. Typical range: 1e-6 to 1e-2.",
            }
            F64Field {
                label: "Convergence-before-rejection threshold",
                value: params.read().convergence_before_rejection_threshold,
                on_change: move |v| params.write().convergence_before_rejection_threshold = v,
                help: "How many times looser than the final convergence threshold the fit must be before it's allowed to start rejecting outliers. Defaults to 2 (i.e. 2x looser). Higher values delay outlier rejection until the fit is closer to converged, avoiding rejecting good points too early.",
            }
            F64Field {
                label: "RMS stagnation ratio",
                value: params.read().rms_stagnation_ratio,
                on_change: move |v| params.write().rms_stagnation_ratio = v,
                help: "If the RMS doesn't improve by at least this ratio between passes, the fit is considered stagnated. Defaults to 0.98 (98% of previous RMS). Closer to 1.0 = more tolerant of slow progress; must stay in (0, 1].",
            }
            F64Field {
                label: "RMS divergence ratio",
                value: params.read().rms_divergence_ratio,
                on_change: move |v| params.write().rms_divergence_ratio = v,
                help: "If the RMS grows by more than this ratio between passes, the fit is considered diverging and stops. Defaults to 1.5 (50% worse). Lower it to fail fast on unstable fits; must stay above 1.0.",
            }
            UsizeField {
                label: "Max stagnation iterations",
                value: params.read().max_stagnation_iterations,
                on_change: move |v| params.write().max_stagnation_iterations = v,
                help: "How many consecutive stagnated passes (see RMS stagnation ratio) are tolerated before giving up. Defaults to 3. Raise it for fits that plateau briefly before improving again.",
            }
            label { class: "label cursor-pointer gap-2 col-span-2 md:col-span-3 justify-start",
                input {
                    r#type: "checkbox",
                    class: "checkbox checkbox-sm",
                    checked: params.read().enable_outlier_rejection,
                    onchange: move |evt| params.write().enable_outlier_rejection = evt.checked(),
                }
                span { class: "label-text", "Enable outlier rejection" }
                HelpTooltip {
                    text: "When on, observations with high residuals are progressively excluded from the fit across passes. Turn off to force-fit every selected observation, useful for inspecting how a suspected bad point affects the solution.",
                }
            }
        }
    }
}

#[component]
fn IodSection(params: Signal<OrbitFitParams>) -> Element {
    rsx! {
        div { class: "grid grid-cols-2 md:grid-cols-3 gap-3",
            UsizeField {
                label: "Monte-Carlo noise realizations",
                value: params.read().n_noise_realizations,
                on_change: move |v| params.write().n_noise_realizations = v,
            }
            F64Field {
                label: "Noise scale",
                value: params.read().noise_scale,
                on_change: move |v| params.write().noise_scale = v,
            }
            F64Field {
                label: "Max Δt between observations (days)",
                value: params.read().dtmax,
                on_change: move |v| params.write().dtmax = v,
            }
            F64Field {
                label: "Min Δt within a triplet (days)",
                value: params.read().dt_min,
                on_change: move |v| params.write().dt_min = v,
            }
            F64Field {
                label: "Max Δt within a triplet (days)",
                value: params.read().dt_max_triplet,
                on_change: move |v| params.write().dt_max_triplet = v,
            }
            F64Field {
                label: "Optimal interval time (days)",
                value: params.read().optimal_interval_time,
                on_change: move |v| params.write().optimal_interval_time = v,
            }
            UsizeField {
                label: "Max observations for triplets",
                value: params.read().max_obs_for_triplets,
                on_change: move |v| params.write().max_obs_for_triplets = v,
            }
            U32Field {
                label: "Max triplets",
                value: params.read().max_triplets,
                on_change: move |v| params.write().max_triplets = v,
            }
            F64Field {
                label: "Gap max (days)",
                value: params.read().gap_max,
                on_change: move |v| params.write().gap_max = v,
            }
            F64Field {
                label: "Max eccentricity",
                value: params.read().max_ecc,
                on_change: move |v| params.write().max_ecc = v,
            }
            F64Field {
                label: "Max perihelion (AU)",
                value: params.read().max_perihelion_au,
                on_change: move |v| params.write().max_perihelion_au = v,
            }
            F64Field {
                label: "Min ρ₂ (AU)",
                value: params.read().min_rho2_au,
                on_change: move |v| params.write().min_rho2_au = v,
            }
            U32Field {
                label: "Aberth max iterations",
                value: params.read().aberth_max_iter,
                on_change: move |v| params.write().aberth_max_iter = v,
            }
            F64Field {
                label: "Aberth epsilon",
                value: params.read().aberth_eps,
                on_change: move |v| params.write().aberth_eps = v,
            }
            F64Field {
                label: "Kepler epsilon",
                value: params.read().kepler_eps,
                on_change: move |v| params.write().kepler_eps = v,
            }
            UsizeField {
                label: "Max tested solutions",
                value: params.read().max_tested_solutions,
                on_change: move |v| params.write().max_tested_solutions = v,
            }
            F64Field {
                label: "r₂ min (AU)",
                value: params.read().r2_min_au,
                on_change: move |v| params.write().r2_min_au = v,
            }
            F64Field {
                label: "r₂ max (AU)",
                value: params.read().r2_max_au,
                on_change: move |v| params.write().r2_max_au = v,
            }
            F64Field {
                label: "Newton epsilon",
                value: params.read().newton_eps,
                on_change: move |v| params.write().newton_eps = v,
            }
            UsizeField {
                label: "Newton max iterations",
                value: params.read().newton_max_it,
                on_change: move |v| params.write().newton_max_it = v,
            }
            F64Field {
                label: "Root imaginary epsilon",
                value: params.read().root_imag_eps,
                on_change: move |v| params.write().root_imag_eps = v,
            }
        }
    }
}

#[component]
fn F64Field(
    label: &'static str,
    value: f64,
    on_change: EventHandler<f64>,
    #[props(default = "")] help: &'static str,
) -> Element {
    rsx! {
        div { class: "form-control gap-1",
            label { class: "label py-0",
                span { class: "label-text text-xs", "{label}" }
                if !help.is_empty() {
                    HelpTooltip { text: help }
                }
            }
            input {
                r#type: "number",
                step: "any",
                class: "input input-bordered input-sm",
                value: "{value}",
                oninput: move |evt| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        on_change.call(v);
                    }
                },
            }
        }
    }
}

#[component]
fn UsizeField(
    label: &'static str,
    value: usize,
    on_change: EventHandler<usize>,
    #[props(default = "")] help: &'static str,
) -> Element {
    rsx! {
        div { class: "form-control gap-1",
            label { class: "label py-0",
                span { class: "label-text text-xs", "{label}" }
                if !help.is_empty() {
                    HelpTooltip { text: help }
                }
            }
            input {
                r#type: "number",
                step: "1",
                min: "0",
                class: "input input-bordered input-sm",
                value: "{value}",
                oninput: move |evt| {
                    if let Ok(v) = evt.value().parse::<usize>() {
                        on_change.call(v);
                    }
                },
            }
        }
    }
}

#[component]
fn U32Field(
    label: &'static str,
    value: u32,
    on_change: EventHandler<u32>,
    #[props(default = "")] help: &'static str,
) -> Element {
    rsx! {
        div { class: "form-control gap-1",
            label { class: "label py-0",
                span { class: "label-text text-xs", "{label}" }
                if !help.is_empty() {
                    HelpTooltip { text: help }
                }
            }
            input {
                r#type: "number",
                step: "1",
                min: "0",
                class: "input input-bordered input-sm",
                value: "{value}",
                oninput: move |evt| {
                    if let Ok(v) = evt.value().parse::<u32>() {
                        on_change.call(v);
                    }
                },
            }
        }
    }
}

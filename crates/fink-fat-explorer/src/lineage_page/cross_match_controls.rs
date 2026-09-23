use dioxus::prelude::*;

use crate::cnd_search::{
    clamp_angle_separation_arcsec, clamp_time_separation_s, MAX_ANGLE_SEPARATION_ARCSEC,
    MAX_TIME_SEPARATION_S, MIN_ANGLE_SEPARATION_ARCSEC, MIN_TIME_SEPARATION_S,
};
use crate::skybot_search::{MAX_RADIUS_ARCSEC, MIN_RADIUS_ARCSEC};

use super::trajectory_plot::format_last_checked;

/// Single "Cross-match ▾" toggle button plus a small anchored panel grouping
/// both cross-match services (Skybot, MPC/CND) — introduced because two full
/// button+controls rows side by side would clutter the trajectory plot's
/// header. No daisyUI `dropdown` precedent exists elsewhere in this codebase
/// (checked before writing this), so the open/close/backdrop mechanics here
/// copy `ades_export_modal.rs`'s already-established pattern (a full-viewport
/// backdrop `div` closes the panel on outside click), just anchored under
/// the toggle button instead of centered, and without the dark tint since
/// this isn't a blocking dialog.
#[component]
pub fn CrossMatchControls(
    skybot_radius_arcsec: f64,
    skybot_running: bool,
    skybot_processed: usize,
    skybot_total: usize,
    skybot_last_queried_at: Option<String>,
    skybot_delta_days: Option<f64>,
    on_skybot_radius_change: EventHandler<f64>,
    on_skybot_search: EventHandler<()>,
    cnd_time_separation_s: f64,
    cnd_angle_separation_arcsec: f64,
    cnd_running: bool,
    cnd_processed: usize,
    cnd_total: usize,
    cnd_last_queried_at: Option<String>,
    cnd_delta_days: Option<f64>,
    on_cnd_time_separation_change: EventHandler<f64>,
    on_cnd_angle_separation_change: EventHandler<f64>,
    on_cnd_search: EventHandler<()>,
) -> Element {
    let mut open = use_signal(|| false);

    rsx! {
        div { class: "relative",
            button {
                class: "btn btn-sm btn-outline",
                r#type: "button",
                onclick: move |_| open.set(!open()),
                if skybot_running || cnd_running {
                    span { class: "loading loading-spinner loading-xs" }
                }
                "Cross-match ▾"
            }
            if open() {
                div {
                    class: "fixed inset-0 z-40",
                    onclick: move |_| open.set(false),
                }
                div { class: "absolute right-0 mt-2 w-80 card bg-base-100 shadow-xl z-50 p-4 flex flex-col gap-3",
                    div { class: "flex flex-col gap-2",
                        h4 { class: "font-semibold text-sm", "Skybot" }
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Radius"
                            input {
                                r#type: "range",
                                class: "range range-xs flex-1",
                                min: "{MIN_RADIUS_ARCSEC}",
                                max: "{MAX_RADIUS_ARCSEC}",
                                step: "1",
                                disabled: skybot_running,
                                value: "{skybot_radius_arcsec}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        on_skybot_radius_change.call(v);
                                    }
                                },
                            }
                            span { "{skybot_radius_arcsec:.0}\"" }
                        }
                        button {
                            class: "btn btn-xs btn-outline",
                            r#type: "button",
                            disabled: skybot_running,
                            onclick: move |_| on_skybot_search.call(()),
                            if skybot_running {
                                span { class: "loading loading-spinner loading-xs" }
                                "Searching ({skybot_processed}/{skybot_total})"
                            } else {
                                "Search Skybot"
                            }
                        }
                        p { class: "text-xs opacity-60",
                            "{format_last_checked(\"Skybot\", skybot_last_queried_at.as_deref(), skybot_delta_days)}"
                        }
                    }

                    div { class: "divider my-0" }

                    div { class: "flex flex-col gap-2",
                        h4 { class: "font-semibold text-sm", "MPC (CND)" }
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Δt"
                            input {
                                r#type: "range",
                                class: "range range-xs flex-1",
                                min: "{MIN_TIME_SEPARATION_S}",
                                max: "{MAX_TIME_SEPARATION_S}",
                                step: "0.5",
                                disabled: cnd_running,
                                value: "{cnd_time_separation_s}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        on_cnd_time_separation_change.call(clamp_time_separation_s(v));
                                    }
                                },
                            }
                            span { "{cnd_time_separation_s:.1}s" }
                        }
                        label { class: "flex items-center gap-2 text-xs opacity-70",
                            "Δθ"
                            input {
                                r#type: "range",
                                class: "range range-xs flex-1",
                                min: "{MIN_ANGLE_SEPARATION_ARCSEC}",
                                max: "{MAX_ANGLE_SEPARATION_ARCSEC}",
                                step: "0.5",
                                disabled: cnd_running,
                                value: "{cnd_angle_separation_arcsec}",
                                oninput: move |evt| {
                                    if let Ok(v) = evt.value().parse::<f64>() {
                                        on_cnd_angle_separation_change.call(clamp_angle_separation_arcsec(v));
                                    }
                                },
                            }
                            span { "{cnd_angle_separation_arcsec:.1}\"" }
                        }
                        button {
                            class: "btn btn-xs btn-outline",
                            r#type: "button",
                            disabled: cnd_running,
                            onclick: move |_| on_cnd_search.call(()),
                            if cnd_running {
                                span { class: "loading loading-spinner loading-xs" }
                                "Checking ({cnd_processed}/{cnd_total})"
                            } else {
                                "Check MPC"
                            }
                        }
                        p { class: "text-xs opacity-60",
                            "{format_last_checked(\"MPC\", cnd_last_queried_at.as_deref(), cnd_delta_days)}"
                        }
                    }
                }
            }
        }
    }
}

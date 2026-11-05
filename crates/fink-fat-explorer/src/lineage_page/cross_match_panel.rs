use dioxus::prelude::*;

use crate::cnd_search::CndHit;
use crate::skybot_search::{dedup_hits_by_name, SkybotHit};

/// Right-side overlay listing every distinct cross-match hit found so far,
/// from both services (Skybot, MPC/CND) — merged into one panel rather than
/// one per service, so the "☰" toggle in
/// [`super::trajectory_plot::TrajectoryPlot`] stays a single button. A
/// section is skipped entirely when that service's hit list is empty.
#[component]
pub fn CrossMatchPanel(
    skybot_hits: Vec<SkybotHit>,
    cnd_hits: Vec<CndHit>,
    open: bool,
    on_close: EventHandler<()>,
) -> Element {
    if !open {
        return rsx! {};
    }

    let distinct_skybot_hits = dedup_hits_by_name(&skybot_hits);

    rsx! {
        div {
            class: "fixed inset-0 z-50 bg-black/40",
            onclick: move |_| on_close.call(()),
        }
        div { class: "fixed inset-y-0 right-0 z-50 w-80 max-w-full bg-base-100 shadow-xl flex flex-col",
            div { class: "flex items-center justify-between p-4 border-b border-base-300",
                h3 { class: "font-semibold", "Cross-match results" }
                button {
                    class: "btn btn-sm btn-circle btn-ghost",
                    r#type: "button",
                    onclick: move |_| on_close.call(()),
                    "✕"
                }
            }
            div { class: "flex-1 overflow-y-auto p-4 flex flex-col gap-4",
                if distinct_skybot_hits.is_empty() && cnd_hits.is_empty() {
                    p { class: "text-sm opacity-60", "No cross-match results yet." }
                } else {
                    if !distinct_skybot_hits.is_empty() {
                        div { class: "flex flex-col gap-2",
                            h4 { class: "text-xs font-semibold uppercase opacity-60",
                                "Skybot ({distinct_skybot_hits.len()})"
                            }
                            for hit in distinct_skybot_hits {
                                SkybotHitRow { key: "{hit.name}", hit }
                            }
                        }
                    }
                    if !cnd_hits.is_empty() {
                        div { class: "flex flex-col gap-2",
                            h4 { class: "text-xs font-semibold uppercase opacity-60",
                                "MPC near-duplicates ({cnd_hits.len()})"
                            }
                            for hit in cnd_hits {
                                CndHitRow { key: "{hit.obs_id}", hit }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn SkybotHitRow(hit: SkybotHit) -> Element {
    rsx! {
        div { class: "rounded-box bg-base-200 p-3 flex flex-col gap-1",
            div { class: "font-medium text-sm", "{hit.name}" }
            div { class: "text-xs opacity-70", "{hit.class}" }
            if let Some(vmag) = hit.vmag {
                div { class: "text-xs opacity-70", "V mag {vmag:.2}" }
            }
            if let Some(url) = &hit.ssodnet_url {
                a {
                    class: "link link-primary text-xs",
                    href: "{url}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "View on SSODNet"
                }
            }
        }
    }
}

#[component]
fn CndHitRow(hit: CndHit) -> Element {
    rsx! {
        div { class: "rounded-box bg-base-200 p-3 flex flex-col gap-1",
            div { class: "font-medium text-sm", "Observation {hit.obs_id}" }
            div { class: "text-xs opacity-70",
                "{hit.n_matches} published observation(s) nearby"
            }
            div { class: "text-xs opacity-70",
                "closest: Δt {hit.min_time_separation_s:.1}s, Δθ {hit.min_angle_separation_arcsec:.2}″"
            }
            if let Some(obs80) = &hit.closest_match_obs80 {
                div { class: "text-xs font-mono opacity-60 break-all", "{obs80}" }
            }
        }
    }
}

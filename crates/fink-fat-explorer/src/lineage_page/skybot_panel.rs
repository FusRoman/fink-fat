use dioxus::prelude::*;

use crate::skybot_search::{dedup_hits_by_name, SkybotHit};

/// Right-side overlay listing every distinct Skybot match found so far, each
/// with a clickable SSODNet link — kept off the main page (rather than an
/// inline table under the plot) so it doesn't add permanent weight to an
/// already dense layout. Opened via the "burger" button next to the Skybot
/// search controls in [`super::trajectory_plot::TrajectoryPlot`].
#[component]
pub fn SkybotPanel(hits: Vec<SkybotHit>, open: bool, on_close: EventHandler<()>) -> Element {
    if !open {
        return rsx! {};
    }

    let distinct_hits = dedup_hits_by_name(&hits);

    rsx! {
        div {
            class: "fixed inset-0 z-50 bg-black/40",
            onclick: move |_| on_close.call(()),
        }
        div { class: "fixed inset-y-0 right-0 z-50 w-80 max-w-full bg-base-100 shadow-xl flex flex-col",
            div { class: "flex items-center justify-between p-4 border-b border-base-300",
                h3 { class: "font-semibold", "Skybot matches ({distinct_hits.len()})" }
                button {
                    class: "btn btn-sm btn-circle btn-ghost",
                    r#type: "button",
                    onclick: move |_| on_close.call(()),
                    "✕"
                }
            }
            div { class: "flex-1 overflow-y-auto p-4 flex flex-col gap-3",
                if distinct_hits.is_empty() {
                    p { class: "text-sm opacity-60", "No objects found yet." }
                } else {
                    for hit in distinct_hits {
                        SkybotPanelRow { key: "{hit.name}", hit }
                    }
                }
            }
        }
    }
}

#[component]
fn SkybotPanelRow(hit: SkybotHit) -> Element {
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

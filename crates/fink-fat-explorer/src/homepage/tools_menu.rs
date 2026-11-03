use dioxus::prelude::*;

use crate::homepage::refresh_status::RefreshUi;

/// Inline style drawing the refresh progress gauge as the button's
/// background: the filled part in a translucent primary color, the rest
/// transparent so the button's own (possibly disabled) background shows.
///
/// # Arguments
/// * `refresh` — the current refresh state.
///
/// # Return
/// The `style` value, empty when no build is running.
fn gauge_style(refresh: &RefreshUi) -> String {
    refresh
        .gauge_percent()
        .map(|percent| {
            format!(
                "background-image: linear-gradient(to right, \
                 color-mix(in oklab, var(--color-primary) 40%, transparent) {percent}%, \
                 transparent {percent}%);"
            )
        })
        .unwrap_or_default()
}

/// Single "Tools ▾" toggle button plus a small anchored panel grouping the
/// homepage's maintenance *actions* (bulk orbit fit, bulk MPC/CND check,
/// bulk Skybot check, refresh / force refresh) — introduced because three
/// separate always-visible buttons crowded the navbar. The cross-match dashboard is deliberately **not**
/// in here despite being a similarly secondary destination: it's a page to
/// browse, not a tool to run, so it gets its own direct navbar link instead
/// (see `homepage::mod::Home`) rather than being buried one click deeper
/// alongside actions that mutate state. Same open/close/backdrop mechanics
/// as
/// `lineage_page::cross_match_controls::CrossMatchControls`, which grouped
/// the per-lineage Skybot/CND controls for the same reason; kept as a
/// distinct component rather than a shared one since the two menus group
/// unrelated actions for unrelated pages, not two instances of the same
/// thing.
#[component]
pub fn ToolsMenu(refresh: RefreshUi, on_refresh: EventHandler<bool>) -> Element {
    let mut open = use_signal(|| false);
    let busy = refresh.is_busy();
    let gauge = gauge_style(&refresh);
    let label = refresh.button_label();
    let detail = refresh.detail();
    let failed = matches!(refresh, RefreshUi::Failed(_));

    rsx! {
        div { class: "relative",
            // Mirrors the gauge so a running refresh stays visible once the
            // panel is closed.
            button {
                class: "btn btn-sm btn-outline",
                r#type: "button",
                style: "{gauge}",
                onclick: move |_| open.set(!open()),
                "Tools ▾"
            }
            if open() {
                div {
                    class: "fixed inset-0 z-40",
                    onclick: move |_| open.set(false),
                }
                div { class: "absolute right-0 mt-2 w-64 card bg-base-100 shadow-xl z-50 p-2 flex flex-col gap-1",
                    Link {
                        to: crate::Route::BulkOrbitFitPage {},
                        class: "btn btn-sm btn-primary justify-start",
                        onclick: move |_| open.set(false),
                        "Fit all trajectories"
                    }
                    Link {
                        to: crate::Route::BulkCndPage {},
                        class: "btn btn-sm btn-success justify-start",
                        onclick: move |_| open.set(false),
                        "Check MPC (CND)"
                    }
                    Link {
                        to: crate::Route::BulkSkybotPage {},
                        class: "btn btn-sm btn-info justify-start",
                        onclick: move |_| open.set(false),
                        "Search Skybot (bulk)"
                    }
                    // Unlike the `Link`s above (which navigate away and close
                    // the menu on click), the refresh controls deliberately do
                    // NOT close the panel: the refresh runs in place on this
                    // page, and the gauge/detail line below are its feedback.
                    div { class: "join w-full",
                        button {
                            class: if failed { "join-item btn btn-sm flex-1 justify-start text-error" } else { "join-item btn btn-sm flex-1 justify-start" },
                            r#type: "button",
                            disabled: busy,
                            style: "{gauge}",
                            title: "Reload the in-memory index if the database changed (new convert, orbit fits, cross-matches)",
                            onclick: move |_| on_refresh.call(false),
                            if matches!(refresh, RefreshUi::Checking) {
                                span { class: "loading loading-spinner loading-xs" }
                            }
                            "{label}"
                        }
                        button {
                            class: "join-item btn btn-sm",
                            r#type: "button",
                            disabled: busy,
                            title: "Force a full rebuild, even if no change was detected",
                            onclick: move |_| on_refresh.call(true),
                            "⟳"
                        }
                    }
                    if let Some(detail) = detail {
                        p { class: if failed { "text-xs px-1 text-error break-words" } else { "text-xs px-1 opacity-70" },
                            "{detail}"
                        }
                    }
                }
            }
        }
    }
}

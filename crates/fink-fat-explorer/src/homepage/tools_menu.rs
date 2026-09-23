use dioxus::prelude::*;

/// Single "Tools ▾" toggle button plus a small anchored panel grouping the
/// homepage's maintenance actions (bulk orbit fit, bulk MPC/CND check,
/// refresh) — introduced because three separate always-visible buttons
/// crowded the navbar. Same open/close/backdrop mechanics as
/// `lineage_page::cross_match_controls::CrossMatchControls`, which grouped
/// the per-lineage Skybot/CND controls for the same reason; kept as a
/// distinct component rather than a shared one since the two menus group
/// unrelated actions for unrelated pages, not two instances of the same
/// thing.
#[component]
pub fn ToolsMenu(refreshing: bool, on_refresh: EventHandler<()>) -> Element {
    let mut open = use_signal(|| false);

    rsx! {
        div { class: "relative",
            button {
                class: "btn btn-sm btn-outline",
                r#type: "button",
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
                    button {
                        class: "btn btn-sm justify-start",
                        r#type: "button",
                        disabled: refreshing,
                        title: "Reload the in-memory index from the database (after a fink-fat convert)",
                        // Unlike the two `Link`s above (which navigate away
                        // and close the menu on click), this one deliberately
                        // does NOT close the panel: refreshing runs in place
                        // on this page, and the spinner/"Refreshing" state
                        // below is the only feedback the user gets that it's
                        // working — closing the menu would hide it.
                        onclick: move |_| on_refresh.call(()),
                        if refreshing {
                            span { class: "loading loading-spinner loading-xs" }
                            "Refreshing"
                        } else {
                            "Refresh"
                        }
                    }
                }
            }
        }
    }
}

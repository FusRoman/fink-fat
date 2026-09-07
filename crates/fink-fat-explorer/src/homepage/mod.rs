pub mod branch_tab;
pub mod dynamic_pop_plot;
pub mod family;
pub mod interaction;
#[cfg(feature = "server")]
pub mod snapshot;
pub mod stats_count;

use dioxus::prelude::*;
use std::collections::HashSet;

use crate::homepage::{
    branch_tab::{refresh_snapshot, BranchTab},
    dynamic_pop_plot::DynamicPopPlot,
    family::DynamicalFamily,
    stats_count::{get_snapshot_version, StatsBanner},
};

/// How often the refresh button checks whether the rebuild it asked for has
/// landed. The rebuild is a single large query, so this can be leisurely.
const REFRESH_POLL_MS: u64 = 1000;

/// Give up watching for a requested rebuild after this long, so a failed
/// rebuild does not leave a poll loop running for the life of the session.
const REFRESH_POLL_TIMEOUT_MS: u64 = 10 * 60 * 1000;

#[component]
pub fn Home() -> Element {
    let mut search_input = use_signal(String::new);

    // Families the user toggled off in the plot legend. Shared with both the
    // plot (which hides their traces) and the table (which filters them out
    // of its query) — storing the *hidden* set rather than the visible one
    // means the empty default naturally reads as "no filter", so neither
    // component has to know the full list of families up front.
    let hidden_families = use_signal(HashSet::<DynamicalFamily>::new);

    // Bumped once a requested snapshot rebuild has actually landed. All three
    // data components read it inside their resource futures, so bumping it
    // refetches the whole page against the new snapshot.
    let mut refresh_token = use_signal(|| 0_u64);
    let mut refreshing = use_signal(|| false);

    let request_refresh = move |_| {
        if refreshing() {
            return;
        }
        refreshing.set(true);

        spawn(async move {
            // The version at click time: the rebuild has landed when the
            // server reports a different one. Asking for the version first
            // means a rebuild that finishes unusually fast cannot slip past.
            let before = get_snapshot_version().await.ok().flatten();

            if refresh_snapshot().await.is_err() {
                refreshing.set(false);
                return;
            }

            let mut waited = 0;
            loop {
                crate::sleep_ms(REFRESH_POLL_MS).await;
                waited += REFRESH_POLL_MS;

                match get_snapshot_version().await {
                    Ok(current) if current.is_some() && current != before => {
                        refresh_token += 1;
                        break;
                    }
                    // A transport error mid-rebuild is not fatal; keep waiting
                    // until the timeout.
                    _ => {}
                }

                if waited >= REFRESH_POLL_TIMEOUT_MS {
                    break;
                }
            }

            refreshing.set(false);
        });
    };

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col",
            div { class: "navbar bg-base-100 shadow-sm px-6 gap-4 sticky top-0 z-20",
                div { class: "flex-none",
                    span { class: "text-lg font-semibold", "fink-fat-explorer" }
                }
                div { class: "flex-1 min-w-0", StatsBanner { refresh_token } }
                div { class: "flex-none flex items-center gap-2",
                    input {
                        r#type: "text",
                        placeholder: "Search a designation...",
                        class: "input input-bordered input-sm w-64",
                        oninput: move |evt| search_input.set(evt.value()),
                    }
                    button {
                        class: "btn btn-sm btn-ghost",
                        disabled: refreshing(),
                        title: "Reload the in-memory index from the database (after a fink-fat convert)",
                        onclick: request_refresh,
                        if refreshing() {
                            span { class: "loading loading-spinner loading-xs" }
                            "Refreshing"
                        } else {
                            "Refresh"
                        }
                    }
                }
            }

            div { class: "p-6 flex flex-col gap-6 flex-1 min-h-0",
                DynamicPopPlot { hidden_families, refresh_token }

                div { class: "grid grid-cols-1",
                    BranchTab {
                        search_query: search_input(),
                        hidden_families,
                        refresh_token,
                    }
                }
            }
        }
    }
}

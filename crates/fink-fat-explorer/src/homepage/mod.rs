pub mod branch_tab;
pub mod dynamic_pop_plot;
pub mod family;
pub mod interaction;
pub mod stats_count;

use dioxus::prelude::*;

use crate::homepage::{
    branch_tab::BranchTab, dynamic_pop_plot::DynamicPopPlot, stats_count::StatsBanner,
};

#[component]
pub fn Home() -> Element {
    let mut search_input = use_signal(String::new);

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col",
            div { class: "navbar bg-base-100 shadow-sm px-6 gap-4 sticky top-0 z-20",
                div { class: "flex-none",
                    span { class: "text-lg font-semibold", "fink-fat-explorer" }
                }
                div { class: "flex-1 min-w-0", StatsBanner {} }
                div { class: "flex-none",
                    input {
                        r#type: "text",
                        placeholder: "Search a designation...",
                        class: "input input-bordered input-sm w-64",
                        oninput: move |evt| search_input.set(evt.value()),
                    }
                }
            }

            div { class: "p-6 flex flex-col gap-6 flex-1 min-h-0",
                DynamicPopPlot {}

                div { class: "grid grid-cols-1",
                    BranchTab { search_query: search_input() }
                }
            }
        }
    }
}

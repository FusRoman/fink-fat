use dioxus::prelude::*;

#[component]
pub fn LineagePage(lineage_id: String) -> Element {
    rsx! {
        div {
            h1 { "Détail de l'item : {lineage_id}" }
        }
    }
}

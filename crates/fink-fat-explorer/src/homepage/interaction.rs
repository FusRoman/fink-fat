use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SortColumn {
    CumulativeLlr,
    Updates,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SortDirection {
    Asc,
    Desc,
}

impl SortDirection {
    pub fn toggled(self) -> Self {
        match self {
            SortDirection::Asc => SortDirection::Desc,
            SortDirection::Desc => SortDirection::Asc,
        }
    }

    pub fn sql(self) -> &'static str {
        match self {
            SortDirection::Asc => "ASC",
            SortDirection::Desc => "DESC",
        }
    }
}

pub const PAGE_SIZE: i64 = 25;

#[component]
pub fn Pagination(
    current_page: EventHandler<i64>,
    page: i64,
    total_pages: i64,
    total_lineages: i64,
) -> Element {
    rsx! {
        div { class: "flex items-center justify-between mt-4",
            span { class: "text-sm opacity-70",
                "Page {page + 1} of {total_pages} ({total_lineages} lineages)"
            }
            div { class: "flex items-center gap-2",
                input {
                    r#type: "number",
                    class: "input input-bordered input-sm w-20",
                    min: "1",
                    max: "{total_pages}",
                    value: "{page + 1}",
                    onchange: move |evt| {
                        if let Ok(selected) = evt.value().parse::<i64>() {
                            let clamped = (selected - 1).clamp(0, total_pages - 1);
                            current_page.call(clamped);
                        }
                    },
                }
                div { class: "join",
                    button {
                        class: "join-item btn btn-sm",
                        disabled: page <= 0,
                        onclick: move |_| {
                            if page > 0 {
                                current_page.call(page - 1);
                            }
                        },
                        "«"
                    }
                    button {
                        class: "join-item btn btn-sm",
                        disabled: page + 1 >= total_pages,
                        onclick: move |_| {
                            if page + 1 < total_pages {
                                current_page.call(page + 1);
                            }
                        },
                        "»"
                    }
                }
            }
        }
    }
}

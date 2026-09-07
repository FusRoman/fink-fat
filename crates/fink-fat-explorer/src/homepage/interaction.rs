use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortColumn {
    CumulativeLlr,
    Updates,
    Family,
    ArcLength,
    Nights,
    MedianInterNightDt,
}

impl SortColumn {
    /// Every variant, in declaration order. The homepage snapshot builds one
    /// pre-sorted permutation per entry of this array, indexed by
    /// [`Self::index`], so the two must stay in sync — hence the explicit
    /// listing here rather than a hand-written count somewhere else.
    pub const ALL: [SortColumn; 6] = [
        SortColumn::CumulativeLlr,
        SortColumn::Updates,
        SortColumn::Family,
        SortColumn::ArcLength,
        SortColumn::Nights,
        SortColumn::MedianInterNightDt,
    ];

    /// Position of this column in [`Self::ALL`].
    pub fn index(self) -> usize {
        match self {
            SortColumn::CumulativeLlr => 0,
            SortColumn::Updates => 1,
            SortColumn::Family => 2,
            SortColumn::ArcLength => 3,
            SortColumn::Nights => 4,
            SortColumn::MedianInterNightDt => 5,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

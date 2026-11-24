//! Filter controls of the "Submission" page: endpoint and verdict segmented
//! controls, the free-text search box and the date-range picker.
//!
//! Purely presentational: every control writes into a signal owned by the
//! page, which turns them into a [`super::query::SubmissionQuery`].

use dioxus::prelude::*;

use super::date_range_picker::DateRangePicker;
use super::query::{DateRange, EndpointFilter, VerdictFilter};

/// Index of `selected` within `options`.
///
/// # Arguments
/// * `options` — the choices, in display order.
/// * `selected` — the current choice.
///
/// # Return
/// Its position, or `None` if it is not among `options`.
fn selected_index<T: PartialEq>(options: &[T], selected: &T) -> Option<usize> {
    options.iter().position(|option| option == selected)
}

/// A daisyUI segmented control (a `join` of buttons) over string labels.
///
/// # Arguments
/// * `labels` — button labels, in display order.
/// * `selected` — index of the active button (`None` highlights nothing).
/// * `on_select` — called with the index of the clicked button.
#[component]
fn SegmentedControl(
    labels: Vec<&'static str>,
    selected: Option<usize>,
    on_select: EventHandler<usize>,
) -> Element {
    rsx! {
        div { class: "join",
            for (index, label) in labels.into_iter().enumerate() {
                button {
                    key: "{label}",
                    class: if selected == Some(index) { "join-item btn btn-sm btn-active btn-primary" } else { "join-item btn btn-sm" },
                    r#type: "button",
                    onclick: move |_| on_select.call(index),
                    "{label}"
                }
            }
        }
    }
}

/// A labelled filter group: small caption above its control.
///
/// # Arguments
/// * `caption` — the caption text.
/// * `children` — the control.
#[component]
fn FilterGroup(caption: &'static str, children: Element) -> Element {
    rsx! {
        div { class: "flex flex-col gap-1",
            span { class: "text-xs opacity-60", "{caption}" }
            {children}
        }
    }
}

/// The filter bar above the submissions table.
///
/// # Arguments
/// * `endpoint` — endpoint filter, written on click.
/// * `verdict` — verdict filter, written on click.
/// * `search` — raw search text, written on every keystroke (the page
///   debounces it).
/// * `date_range` — selected submission-date range.
/// * `filters_active` — whether any filter is set; enables "Clear filters".
/// * `on_clear` — called when the user clicks "Clear filters".
#[component]
pub fn FiltersBar(
    endpoint: Signal<EndpointFilter>,
    verdict: Signal<VerdictFilter>,
    search: Signal<String>,
    date_range: Signal<Option<DateRange>>,
    filters_active: bool,
    on_clear: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "flex flex-wrap items-end gap-4 p-4",
            FilterGroup { caption: "Endpoint",
                SegmentedControl {
                    labels: EndpointFilter::ALL.iter().map(|f| f.label()).collect::<Vec<_>>(),
                    selected: selected_index(&EndpointFilter::ALL, &endpoint()),
                    on_select: move |index: usize| endpoint.set(EndpointFilter::ALL[index]),
                }
            }
            FilterGroup { caption: "Verdict",
                SegmentedControl {
                    labels: VerdictFilter::ALL.iter().map(|f| f.label()).collect::<Vec<_>>(),
                    selected: selected_index(&VerdictFilter::ALL, &verdict()),
                    on_select: move |index: usize| verdict.set(VerdictFilter::ALL[index]),
                }
            }
            FilterGroup { caption: "Submitted between",
                DateRangePicker { range: date_range }
            }
            FilterGroup { caption: "Lineage or submission ID",
                input {
                    class: "input input-bordered input-sm w-64",
                    r#type: "search",
                    placeholder: "e.g. FF2026…",
                    value: "{search}",
                    oninput: move |evt| search.set(evt.value()),
                }
            }
            button {
                class: "btn btn-sm btn-ghost",
                r#type: "button",
                disabled: !filters_active,
                onclick: move |_| on_clear.call(()),
                "Clear filters"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_index_finds_the_active_option() {
        assert_eq!(
            selected_index(&EndpointFilter::ALL, &EndpointFilter::Production),
            Some(2)
        );
        assert_eq!(
            selected_index(&VerdictFilter::ALL, &VerdictFilter::All),
            Some(0)
        );
    }

    #[test]
    fn selected_index_is_none_when_absent() {
        assert_eq!(
            selected_index(&[EndpointFilter::Test], &EndpointFilter::Production),
            None
        );
    }
}

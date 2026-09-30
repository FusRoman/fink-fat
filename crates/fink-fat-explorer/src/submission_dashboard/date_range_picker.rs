//! Date-range picker of the "Submission" page, backed by
//! [vanilla-calendar-pro](https://github.com/uvarov-frontend/vanilla-calendar-pro)
//! (MIT), loaded from the jsDelivr CDN through `Dioxus.toml`.
//!
//! The widget itself is driven through the typed
//! [`CalendarHandle`](super::calendar_bridge::CalendarHandle); this module
//! only bridges its selection into a `Signal<Option<DateRange>>`.

use std::cell::RefCell;
use std::rc::Rc;

use dioxus::prelude::*;

use super::calendar_bridge::CalendarHandle;
use super::query::DateRange;

/// DOM id of the element the calendar mounts into.
const CALENDAR_ELEMENT_ID: &str = "submission-date-range-calendar";

/// Turns the calendar's `,`-joined selection into a range.
///
/// # Arguments
/// * `joined` — `YYYY-MM-DD` days joined by `,`, as reported by the widget
///   (possibly empty).
///
/// # Return
/// The range, or `None` if no valid day is present.
fn range_from_joined(joined: &str) -> Option<DateRange> {
    let days: Vec<&str> = joined.split(',').collect();
    DateRange::from_selected(&days)
}

/// Button + popover calendar selecting an inclusive range of days.
///
/// The signal is the single source of truth: setting it to `None` from
/// elsewhere (e.g. a "Clear filters" button) also empties the widget.
///
/// # Arguments
/// * `range` — the selected range, written on every calendar click and on
///   "Clear".
#[component]
pub fn DateRangePicker(range: Signal<Option<DateRange>>) -> Element {
    let mut open = use_signal(|| false);
    // Owns the mounted calendar; dropping it (component unmount) destroys it.
    let calendar = use_hook(|| Rc::new(RefCell::new(None::<CalendarHandle>)));

    // Mount once, after the first render has put the element in the DOM.
    use_effect({
        let calendar = calendar.clone();
        move || {
            *calendar.borrow_mut() =
                Some(CalendarHandle::mount(CALENDAR_ELEMENT_ID, move |joined| {
                    range.set(range_from_joined(&joined))
                }));
        }
    });

    // Keep the widget in sync when the range is cleared from outside.
    use_effect({
        let calendar = calendar.clone();
        move || {
            if range().is_none() {
                if let Some(handle) = calendar.borrow().as_ref() {
                    handle.clear();
                }
            }
        }
    });

    let label = range()
        .map(|r| r.label())
        .unwrap_or_else(|| "Any date".to_string());

    rsx! {
        div { class: "relative",
            div { class: "join",
                button {
                    class: if range().is_some() { "join-item btn btn-sm btn-primary" } else { "join-item btn btn-sm" },
                    r#type: "button",
                    onclick: move |_| open.set(!open()),
                    "📅 {label}"
                }
                if range().is_some() {
                    button {
                        class: "join-item btn btn-sm btn-ghost",
                        r#type: "button",
                        title: "Clear the date range",
                        onclick: move |_| range.set(None),
                        "✕"
                    }
                }
            }
            if open() {
                // Click-catcher closing the popover when clicking outside it.
                div {
                    class: "fixed inset-0 z-10",
                    onclick: move |_| open.set(false),
                }
            }
            // Always rendered (hidden when closed) so the widget is mounted
            // once and keeps its selection across open/close.
            div {
                class: if open() { "absolute z-20 mt-2 shadow-lg rounded-box bg-base-100" } else { "hidden" },
                div { id: CALENDAR_ELEMENT_ID }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_from_joined_spans_min_to_max() {
        let range = range_from_joined("2026-09-03,2026-09-01,2026-09-02").unwrap();
        assert_eq!(range.label(), "2026-09-01 → 2026-09-03");
    }

    #[test]
    fn range_from_joined_single_day() {
        assert_eq!(
            range_from_joined("2026-09-05").unwrap().label(),
            "2026-09-05"
        );
    }

    #[test]
    fn range_from_joined_empty_selection_is_none() {
        assert!(range_from_joined("").is_none());
    }

    #[test]
    fn range_from_joined_ignores_garbage() {
        assert!(range_from_joined("nope,").is_none());
    }
}

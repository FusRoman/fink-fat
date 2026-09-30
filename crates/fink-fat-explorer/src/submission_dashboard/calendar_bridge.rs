//! Typed Rust handle over the vanilla-calendar-pro widget
//! (`window.VanillaCalendarPro`, loaded from the CDN through `Dioxus.toml`).
//!
//! On wasm32 the handle calls a small inline ES module through
//! `wasm_bindgen` (same approach as `homepage::dynamic_pop_plot`'s plotly
//! click binding): no `document::eval`, no JavaScript in Rust strings. On
//! other targets (native `cargo check`, server-side rendering) the handle is
//! an inert stand-in, since there is no DOM to mount into.

#[cfg(target_arch = "wasm32")]
mod js {
    use wasm_bindgen::closure::Closure;

    #[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
const calendars = new Map();

export function mount_calendar(element_id, on_change) {
    const lib = window.VanillaCalendarPro;
    if (!lib) {
        console.warn("vanilla-calendar-pro is not loaded (CDN unreachable?)");
        return;
    }
    const element = document.getElementById(element_id);
    if (!element || calendars.has(element_id)) { return; }
    const calendar = new lib.Calendar(element, {
        selectionDatesMode: "multiple-ranged",
        selectedTheme: "system",
        onClickDate(self) {
            on_change(self.context.selectedDates.join(","));
        },
    });
    calendar.init();
    calendars.set(element_id, calendar);
}

export function clear_calendar(element_id) {
    const calendar = calendars.get(element_id);
    if (!calendar) { return; }
    calendar.selectedDates = [];
    calendar.update({ dates: true });
}

export function destroy_calendar(element_id) {
    const calendar = calendars.get(element_id);
    if (!calendar) { return; }
    calendar.destroy();
    calendars.delete(element_id);
}
"#)]
    extern "C" {
        /// Mounts a ranged calendar on `element_id` (a no-op if the element
        /// is missing, already mounted, or the library did not load) and calls
        /// `on_change` with the selected dates joined by `,` on every click.
        pub fn mount_calendar(element_id: &str, on_change: &Closure<dyn FnMut(String)>);

        /// Empties the selection of the calendar mounted on `element_id`.
        pub fn clear_calendar(element_id: &str);

        /// Unmounts the calendar on `element_id`.
        pub fn destroy_calendar(element_id: &str);
    }
}

/// A calendar mounted on a DOM element; unmounts it when dropped.
///
/// Owns the Rust callback handed to JavaScript, so the callback stays valid
/// exactly as long as the calendar is mounted.
pub struct CalendarHandle {
    #[cfg(target_arch = "wasm32")]
    element_id: String,
    #[cfg(target_arch = "wasm32")]
    _on_change: wasm_bindgen::closure::Closure<dyn FnMut(String)>,
}

impl CalendarHandle {
    /// Mounts a ranged calendar on the element with the given id.
    ///
    /// # Arguments
    /// * `element_id` — id of an element already present in the DOM.
    /// * `on_change` — called on every click with the selected days as
    ///   `YYYY-MM-DD` strings joined by `,` (empty when nothing is selected).
    ///
    /// # Return
    /// The handle. Mounting silently does nothing if the element or the
    /// library is missing (a warning is logged in the browser console).
    #[cfg(target_arch = "wasm32")]
    pub fn mount(element_id: &str, on_change: impl FnMut(String) + 'static) -> Self {
        let on_change = wasm_bindgen::closure::Closure::new(on_change);
        js::mount_calendar(element_id, &on_change);
        Self {
            element_id: element_id.to_string(),
            _on_change: on_change,
        }
    }

    /// Non-wasm stand-in: there is no DOM, so nothing is mounted.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mount(_element_id: &str, _on_change: impl FnMut(String) + 'static) -> Self {
        Self {}
    }

    /// Empties the calendar's selection (does not call `on_change`).
    pub fn clear(&self) {
        #[cfg(target_arch = "wasm32")]
        js::clear_calendar(&self.element_id);
    }
}

impl Drop for CalendarHandle {
    fn drop(&mut self) {
        #[cfg(target_arch = "wasm32")]
        js::destroy_calendar(&self.element_id);
    }
}

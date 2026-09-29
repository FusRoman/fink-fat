//! Shared shell for this app's "?" help widgets (the homepage's
//! `population_plot_glossary`/`orbit3d_population_glossary`/
//! `lineage_table_glossary`, and the lineage page's per-plot glossaries): a
//! "?" icon whose popover opens on hover, can be *pinned* open (and
//! scrolled) by clicking the icon, and closes on an outside click or
//! Escape.
//!
//! Driven by Dioxus state rather than a CSS-only tooltip (daisyUI's
//! `tooltip`), which closes as soon as the pointer crosses the gap between
//! the icon and the bubble — so its content could neither be reached nor
//! scrolled — and whose size limits are inline styles here so they do not
//! depend on Tailwind having generated a utility class.
//!
//! Same mechanic as the lineage page's 3D-view glossary
//! (`lineage_page::orbit3d_glossary::PlotGlossary`), which predates this
//! shared shell and still has its own copy — it was the only consumer at
//! the time, so unifying them would only have added an import for its own
//! sake. Every widget added since reuses this one instead.
//!
//! A crate-root module (not nested under `homepage`, where it originated)
//! because both `homepage` and `lineage_page` now mount widgets built on
//! it.

use dioxus::prelude::*;

/// Width of the glossary popover: two columns of definitions, but never
/// wider than the window.
const POPOVER_WIDTH: &str = "min(40rem, calc(100vw - 2rem))";
/// Height limit of the glossary popover: most of the window, so the content
/// scrolls inside the popover instead of running off the screen.
const POPOVER_MAX_HEIGHT: &str = "min(75vh, 36rem)";

/// A "?" icon that reveals `children` on hover, and keeps it open (and
/// scrollable) once clicked.
///
/// # Arguments
/// * `aria_label` — accessible name for the icon button, specific to what
///   it explains (e.g. `"Help for the (a, e) plot"`).
/// * `children` — the popover's content, typically an "About this ..."
///   paragraph followed by one or more `dl` reference lists.
#[component]
pub fn HelpPopover(aria_label: &'static str, children: Element) -> Element {
    let mut hovered = use_signal(|| false);
    let mut pinned = use_signal(|| false);
    let open = hovered() || pinned();
    let icon_opacity = if open { "1" } else { "0.6" };

    rsx! {
        div {
            style: "position: relative; display: inline-block;",
            onmouseenter: move |_| hovered.set(true),
            onmouseleave: move |_| hovered.set(false),
            onkeydown: move |evt| {
                if evt.key() == Key::Escape {
                    pinned.set(false);
                    hovered.set(false);
                }
            },
            // While pinned, a click anywhere outside the popover closes it.
            if pinned() {
                div {
                    style: "position: fixed; inset: 0; z-index: 40;",
                    onclick: move |_| {
                        pinned.set(false);
                        hovered.set(false);
                    },
                }
            }
            button {
                r#type: "button",
                class: "inline-flex items-center justify-center w-5 h-5 rounded-full border border-current text-xs leading-none cursor-help",
                style: "position: relative; z-index: 50; opacity: {icon_opacity};",
                aria_label: "{aria_label}",
                aria_expanded: "{open}",
                onclick: move |_| pinned.toggle(),
                "?"
            }
            if open {
                div { style: "position: absolute; top: 100%; left: 0; z-index: 50; padding-top: 0.375rem;",
                    div {
                        class: "bg-base-100 rounded-box p-3 text-xs text-left",
                        style: "width: {POPOVER_WIDTH}; max-height: {POPOVER_MAX_HEIGHT}; overflow-y: auto; \
                                border: 1px solid rgba(128, 128, 128, 0.35); \
                                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);",
                        div { class: "flex items-baseline justify-between mb-2",
                            span { class: "font-semibold", "Help" }
                            span { class: "opacity-60",
                                if pinned() {
                                    "click outside or press Esc to close"
                                } else {
                                    "click the ? to keep it open and scroll"
                                }
                            }
                        }
                        {children}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The popover must fit in the window: its size limits are relative to
    /// the viewport, never fixed pixel sizes that a small window could not
    /// hold.
    #[test]
    fn the_popover_is_sized_relative_to_the_viewport() {
        assert!(POPOVER_WIDTH.contains("100vw"), "{POPOVER_WIDTH}");
        assert!(POPOVER_MAX_HEIGHT.contains("vh"), "{POPOVER_MAX_HEIGHT}");
    }
}

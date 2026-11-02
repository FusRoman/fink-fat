//! Shared shell for this app's hover popovers — mostly "?" help widgets
//! (the homepage's `population_plot_glossary`/`orbit3d_population_glossary`/
//! `lineage_table_glossary`, and the lineage page's per-plot glossaries),
//! but also the footer's text-labeled "Acknowledgments"/"Useful links"
//! triggers: a small trigger whose popover opens on hover, can be *pinned*
//! open (and scrolled) by clicking it, and closes on an outside click or
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

/// Default trigger: the small circular "?" icon every glossary widget uses.
const DEFAULT_TRIGGER_LABEL: &str = "?";
/// Default trigger button styling: a small circular outline, sized for a
/// single glyph like [`DEFAULT_TRIGGER_LABEL`].
const DEFAULT_TRIGGER_CLASS: &str = "inline-flex items-center justify-center w-5 h-5 rounded-full \
    border border-current text-xs leading-none cursor-help";
/// Default popover heading — generic, since most triggers are the "?" icon
/// and the heading is the only place naming what it explains.
const DEFAULT_HEADING: &str = "Help";

/// A small trigger — by default the "?" icon every glossary widget uses,
/// but any short label works (e.g. the footer's "Acknowledgments"/"Useful
/// links") — that reveals `children` on hover, and keeps it open (and
/// scrollable) once clicked.
///
/// # Arguments
/// * `aria_label` — accessible name for the trigger, specific to what it
///   reveals (e.g. `"Help for the (a, e) plot"`).
/// * `trigger_label` — the trigger's visible text; defaults to
///   [`DEFAULT_TRIGGER_LABEL`] (`"?"`).
/// * `trigger_class` — the trigger `button`'s class; defaults to
///   [`DEFAULT_TRIGGER_CLASS`], the small circular outline sized for a
///   single glyph — pass something else for a text label (e.g. a plain
///   underlined-on-hover style) so it doesn't render as a tiny circle
///   around several words.
/// * `heading` — the popover's own title, top-left; defaults to
///   [`DEFAULT_HEADING`] (`"Help"`) — pass something more specific (e.g.
///   the same text as `trigger_label`) when the trigger isn't the generic
///   "?" icon.
/// * `open_upward` — opens the popover above the trigger instead of below
///   it; `false` by default. For a trigger near the bottom of the page
///   (e.g. the footer), opening downward would grow the page itself.
/// * `children` — the popover's content, typically an "About this ..."
///   paragraph followed by one or more `dl` reference lists.
#[component]
pub fn HelpPopover(
    aria_label: &'static str,
    #[props(default = DEFAULT_TRIGGER_LABEL)] trigger_label: &'static str,
    #[props(default = DEFAULT_TRIGGER_CLASS)] trigger_class: &'static str,
    #[props(default = DEFAULT_HEADING)] heading: &'static str,
    #[props(default = false)] open_upward: bool,
    children: Element,
) -> Element {
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
                class: "{trigger_class}",
                style: "position: relative; z-index: 50; opacity: {icon_opacity};",
                aria_label: "{aria_label}",
                aria_expanded: "{open}",
                onclick: move |_| pinned.toggle(),
                "{trigger_label}"
            }
            if open {
                div {
                    style: if open_upward { "position: absolute; bottom: 100%; left: 0; z-index: 50; padding-bottom: 0.375rem;" } else { "position: absolute; top: 100%; left: 0; z-index: 50; padding-top: 0.375rem;" },
                    div {
                        class: "bg-base-100 rounded-box p-3 text-xs text-left",
                        style: "width: {POPOVER_WIDTH}; max-height: {POPOVER_MAX_HEIGHT}; overflow-y: auto; \
                                border: 1px solid rgba(128, 128, 128, 0.35); \
                                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);",
                        div { class: "flex items-baseline justify-between mb-2",
                            span { class: "font-semibold", "{heading}" }
                            span { class: "opacity-60",
                                if pinned() {
                                    "click outside or press Esc to close"
                                } else {
                                    "click to keep it open and scroll"
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

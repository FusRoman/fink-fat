use dioxus::prelude::*;

/// Small "?" bubble that shows `text` in a hover tooltip — plain-language
/// help for a fit parameter (what it does, how to tune it, sensible
/// bounds). Pure CSS (DaisyUI's `tooltip`/`data-tip`), no JS required.
#[component]
pub fn HelpTooltip(text: &'static str) -> Element {
    rsx! {
        span {
            class: "tooltip tooltip-right z-10",
            "data-tip": "{text}",
            span {
                class: "inline-flex items-center justify-center w-4 h-4 rounded-full border border-current text-[10px] leading-none opacity-60 cursor-help ml-1 shrink-0",
                "?"
            }
        }
    }
}

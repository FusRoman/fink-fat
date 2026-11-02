//! The site-wide footer: acknowledgments, useful links, and the resolved
//! version of this app plus the 3 libraries its data pipeline is built on.
//!
//! Mounted once in [`crate::App`], around the router, rather than by each
//! page individually — see that component's doc comment for why a single
//! mount point covers every route.
//!
//! A single compact bar (`bg-base-300`): the description line, then a
//! centered row with the Fink logo, the software-version badges, the
//! GitHub link, and the "Acknowledgments"/"Useful links" content —
//! revealed on hover/click through [`crate::help_popover::HelpPopover`]
//! (the same shell used everywhere else in the app for a "?" icon), here
//! with a text label instead of "?", rather than in their own always-shown
//! columns — the bar stays this one compact size regardless of how much
//! text either section holds.

mod content;
mod versions;

use dioxus::prelude::*;

use crate::help_popover::HelpPopover;
use content::Logo;

/// Bundles and resolves one of [`content::Logo`]'s known logos to its
/// asset. `asset!` needs a string literal at its own call site, so this is
/// the one place that literal has to live — an exhaustive match, so a new
/// [`Logo`] variant added in `content.rs` fails to compile here until it
/// has one too, rather than silently rendering nothing.
fn resolve_logo(logo: Logo) -> Asset {
    match logo {
        Logo::Rust => asset!("/assets/rust_logo.svg"),
        Logo::Ferris => asset!("/assets/ferris.svg"),
        Logo::Dioxus => asset!("/assets/dioxus_logo.png"),
    }
}

/// GitHub's "mark-github" Octicon, inlined so the link doesn't depend on an
/// external icon font/sprite — MIT-licensed, from GitHub's own Primer
/// Octicons set (<https://github.com/primer/octicons>).
const GITHUB_MARK_PATH: &str = "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 \
    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63\
    -.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 \
    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 \
    2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 \
    3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42\
    -3.58-8-8-8z";

/// A small "opens in a new tab" hint, appended after external links —
/// cheap, dependency-free polish next to the plain text link list it
/// replaces.
const EXTERNAL_LINK_HINT: &str = "↗";

/// This app's own one-line description, shown centered above the bar's
/// logo/version/GitHub row.
const APP_DESCRIPTION: &str =
    "fink-fat tracks minor-planet candidates in the alert stream distributed by the Fink broker.";

/// Trigger styling for the "Acknowledgments"/"Useful links" text triggers —
/// a dotted underline rather than [`crate::help_popover`]'s default
/// circular "?" button, which would look wrong around a whole word.
const TEXT_TRIGGER_CLASS: &str = "opacity-70 hover:opacity-100 cursor-pointer underline \
    decoration-dotted underline-offset-4";

/// The site-wide footer. Pure presentation: no props, no state — every
/// piece of content comes from [`content`] (acknowledgments and links) and
/// [`versions::software_versions`] (build-time-resolved version numbers).
#[component]
pub fn Footer() -> Element {
    rsx! {
        div { class: "bg-base-300 text-base-content border-t border-base-300 px-10 py-5",
            p {
                class: "text-center text-sm opacity-80 mb-3 tracking-wide",
                style: "font-family: 'Orbitron', sans-serif;",
                "{APP_DESCRIPTION}"
            }
            div { class: "flex flex-wrap items-center justify-center gap-3 text-xs",
                img {
                    src: asset!("/assets/fink_broker_logo.svg"),
                    alt: "Fink broker",
                    class: "h-20 w-auto",
                }
                span { class: "opacity-30", "•" }
                span { class: "opacity-70", "fink-fat" }
                span { class: "opacity-30", "•" }
                HelpPopover {
                    aria_label: "Acknowledgments",
                    trigger_label: "Acknowledgments",
                    trigger_class: TEXT_TRIGGER_CLASS,
                    heading: "Acknowledgments",
                    open_upward: true,
                    for ack in content::ACKNOWLEDGMENTS {
                        p { class: "text-sm opacity-80 max-w-xs leading-relaxed",
                            a {
                                class: "link link-hover font-medium",
                                href: "{ack.url}",
                                target: "_blank",
                                rel: "noopener noreferrer",
                                "{ack.name}"
                            }
                            if !ack.logos.is_empty() {
                                span { class: "inline-flex items-center gap-1 ml-1.5 align-middle",
                                    for logo in ack.logos {
                                        img {
                                            src: resolve_logo(*logo),
                                            alt: "",
                                            class: "h-4 w-auto inline-block",
                                        }
                                    }
                                }
                            }
                            " — {ack.blurb}"
                        }
                    }
                }
                span { class: "opacity-30", "•" }
                HelpPopover {
                    aria_label: "Useful links",
                    trigger_label: "Useful links",
                    trigger_class: TEXT_TRIGGER_CLASS,
                    heading: "Useful links",
                    open_upward: true,
                    div { class: "flex flex-col gap-2",
                        for link in content::USEFUL_LINKS {
                            a {
                                class: "link link-hover",
                                href: "{link.url}",
                                target: "_blank",
                                rel: "noopener noreferrer",
                                "{link.label} "
                                span { class: "opacity-50 text-xs", "{EXTERNAL_LINK_HINT}" }
                            }
                        }
                    }
                }
                span { class: "opacity-30", "•" }
                for sw in versions::software_versions() {
                    a {
                        class: "badge badge-ghost badge-sm hover:badge-outline",
                        href: "{sw.url}",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        "{sw.name} v{sw.version}"
                    }
                }
                span { class: "opacity-30", "•" }
                a {
                    class: "link link-hover inline-flex items-center gap-1.5 opacity-70 hover:opacity-100",
                    href: "{content::GITHUB_URL}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    svg {
                        class: "h-3.5 w-3.5 fill-current shrink-0",
                        view_box: "0 0 16 16",
                        "aria-hidden": "true",
                        path { d: "{GITHUB_MARK_PATH}" }
                    }
                    "GitHub"
                }
            }
        }
    }
}

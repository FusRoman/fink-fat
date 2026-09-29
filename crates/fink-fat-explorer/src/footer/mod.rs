//! The site-wide footer: acknowledgments, useful links, and the resolved
//! version of this app plus the 3 libraries its data pipeline is built on.
//!
//! Mounted once in [`crate::App`], around the router, rather than by each
//! page individually — see that component's doc comment for why a single
//! mount point covers every route.

mod content;
mod versions;

use dioxus::prelude::*;

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

/// The site-wide footer. Pure presentation: no props, no state — every
/// piece of content comes from [`content`] (acknowledgments and links) and
/// [`versions::software_versions`] (build-time-resolved version numbers).
///
/// `sm:footer-horizontal` is daisyUI's own opt-in for laying its columns
/// out side by side (`footer` alone stacks them, mobile-first) — spread
/// across the full page width from the small breakpoint up, single column
/// only on a narrow phone.
#[component]
pub fn Footer() -> Element {
    rsx! {
        footer {
            class: "footer sm:footer-horizontal bg-base-200 text-base-content border-t border-base-300 \
                     px-10 py-8 gap-8",
            aside { class: "max-w-xs",
                img {
                    src: asset!("/assets/fink_broker_logo.svg"),
                    alt: "Fink broker",
                    class: "w-full h-auto mb-3",
                }
                p { class: "text-sm opacity-70",
                    "fink-fat tracks minor-planet candidates in the alert stream distributed by the \
                     Fink broker."
                }
            }
            nav {
                h6 { class: "footer-title", "Acknowledgments" }
                for ack in content::ACKNOWLEDGMENTS {
                    div { class: "flex items-start gap-2 max-w-xs",
                        if !ack.logos.is_empty() {
                            div { class: "flex items-center gap-1.5 shrink-0 pt-0.5",
                                for logo in ack.logos {
                                    img {
                                        src: resolve_logo(*logo),
                                        alt: "",
                                        class: "h-5 w-auto",
                                    }
                                }
                            }
                        }
                        p { class: "text-sm opacity-80",
                            a {
                                class: "link link-hover font-medium",
                                href: "{ack.url}",
                                target: "_blank",
                                rel: "noopener noreferrer",
                                "{ack.name}"
                            }
                            " — {ack.blurb}"
                        }
                    }
                }
            }
            nav {
                h6 { class: "footer-title", "Useful links" }
                for link in content::USEFUL_LINKS {
                    a {
                        class: "link link-hover",
                        href: "{link.url}",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        "{link.label}"
                    }
                }
                a {
                    class: "link link-hover inline-flex items-center gap-1.5",
                    href: "{content::GITHUB_URL}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    svg {
                        class: "h-4 w-4 fill-current shrink-0",
                        view_box: "0 0 16 16",
                        "aria-hidden": "true",
                        path { d: "{GITHUB_MARK_PATH}" }
                    }
                    "fink-fat on GitHub"
                }
            }
            nav {
                h6 { class: "footer-title", "Software" }
                for sw in versions::software_versions() {
                    a {
                        class: "link link-hover text-sm",
                        href: "{sw.url}",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        "{sw.name} v{sw.version}"
                    }
                }
            }
        }
    }
}

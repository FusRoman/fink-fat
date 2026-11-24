//! User-facing documentation page of the REST API (`/api-docs`).
//!
//! The page is organised as one tab per endpoint category
//! ([`ApiCategory`]); inside a tab each endpoint is a collapsible
//! [`endpoint::EndpointAccordion`]. To document a new endpoint, add it to an
//! existing category component (or add a category: a variant of
//! [`ApiCategory`] plus its component).
//!
//! Everything that could drift from the implementation is derived from it:
//! the route path comes from [`crate::api::REVERSE_SEARCH_PATH`], the sample
//! JSON response is serialized from [`crate::api::ReverseSearchResponse`],
//! and the Python/Rust samples are the real, compiled files under
//! `examples/` (the Rust one is built with the rest of the workspace).

mod alerts;
mod code_tabs;
mod endpoint;

use dioxus::prelude::*;

use alerts::AlertsEndpoints;

/// A group of related endpoints, shown as one tab of the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ApiCategory {
    /// Endpoints starting from an alert (`object_id`).
    Alerts,
}

impl ApiCategory {
    /// Every category, in tab order.
    const ALL: [ApiCategory; 1] = [ApiCategory::Alerts];

    /// Tab title.
    ///
    /// # Return
    ///
    /// The category's display name.
    fn label(self) -> &'static str {
        match self {
            ApiCategory::Alerts => "Alerts",
        }
    }
}

/// Documentation page of the REST API.
#[component]
pub fn ApiDocsPage() -> Element {
    let mut category = use_signal(|| ApiCategory::Alerts);

    rsx! {
        div { class: "min-h-screen bg-base-200 flex flex-col gap-4 p-4",
            div { class: "navbar bg-base-100 shadow-sm px-6 rounded-box",
                Link {
                    to: crate::Route::Home {},
                    class: "link link-hover text-sm",
                    "← Back to home"
                }
            }

            div { class: "max-w-4xl w-full mx-auto flex flex-col gap-6",
                section { class: "flex flex-col gap-2",
                    h1 { class: "text-3xl font-bold", "REST API" }
                    p {
                        "fink-fat-explorer exposes a small, versioned JSON API so that scripts and \
                         external tools can query the trajectory database without going through \
                         the web interface. All routes live under "
                        code { "/api/v1" }
                        " on the same host and port as this site, return JSON and use standard \
                         HTTP status codes. No authentication is required for now."
                    }
                    p {
                        "There is no rate limit yet, but the API shares its server with the web \
                         interface: avoid tight loops over very large id lists."
                    }
                }

                section { class: "flex flex-col gap-2",
                    h2 { class: "text-2xl font-semibold", "Concepts" }
                    ul { class: "list-disc pl-6 flex flex-col gap-1",
                        li {
                            b { "Alert" }
                            " — a single detection. Its identifier is the "
                            code { "object_id" }
                            " of the observation — for LSST, the DIA source id (e.g. 313699504971841537)."
                        }
                        li {
                            b { "Lineage" }
                            " — a candidate asteroid trajectory reconstructed from several \
                             nights of alerts."
                        }
                        li {
                            b { "Branch" }
                            " — one hypothesis of a lineage. A lineage can have several \
                             branches; the best one is the branch with the highest cumulative \
                             log-likelihood ratio."
                        }
                    }
                }

                section { class: "flex flex-col gap-3",
                    h2 { class: "text-2xl font-semibold", "Endpoints" }
                    div { role: "tablist", class: "tabs tabs-box self-start",
                        for cat in ApiCategory::ALL {
                            button {
                                key: "{cat.label()}",
                                role: "tab",
                                class: if category() == cat { "tab tab-active" } else { "tab" },
                                onclick: move |_| category.set(cat),
                                "{cat.label()}"
                            }
                        }
                    }
                    match category() {
                        ApiCategory::Alerts => rsx! {
                            AlertsEndpoints {}
                        },
                    }
                }
            }
        }
    }
}

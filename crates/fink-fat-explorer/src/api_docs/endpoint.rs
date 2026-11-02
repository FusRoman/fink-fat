//! Collapsible documentation entry of one REST endpoint.

use dioxus::prelude::*;

/// One endpoint's documentation, as a daisyUI accordion item: the title row
/// (method badge, path, summary) is always visible, the body unfolds on click.
///
/// # Arguments
///
/// * `method` - HTTP method, e.g. `"GET"`.
/// * `path` - route path, e.g. `"/api/v1/alerts/{object_id}/lineages"`.
/// * `summary` - one-line description of the endpoint.
/// * `open` - whether the entry starts unfolded.
/// * `children` - the documentation body.
#[component]
pub fn EndpointAccordion(
    method: &'static str,
    path: &'static str,
    summary: &'static str,
    #[props(default = false)] open: bool,
    children: Element,
) -> Element {
    let badge = match method {
        "GET" => "badge badge-success",
        "POST" => "badge badge-info",
        "DELETE" => "badge badge-error",
        _ => "badge badge-warning",
    };

    rsx! {
        div { class: "collapse collapse-arrow bg-base-100 border border-base-300 shadow-sm",
            input { r#type: "checkbox", checked: open, "aria-label": "{summary}" }
            div { class: "collapse-title flex flex-wrap items-center gap-3",
                span { class: "{badge}", "{method}" }
                code { class: "text-sm", "{path}" }
                span { class: "text-sm opacity-70", "{summary}" }
            }
            div { class: "collapse-content flex flex-col gap-3", {children} }
        }
    }
}

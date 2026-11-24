//! Documentation of the "Alerts" category: reverse search from an alert to
//! its lineages.

use dioxus::prelude::*;

use super::code_tabs::{CodeTab, CodeTabs};
use super::endpoint::EndpointAccordion;
use crate::api::{
    BatchReverseSearchRequest, BatchReverseSearchResponse, LineageMatch, ReverseSearchResponse,
    BATCH_REVERSE_SEARCH_PATH, MAX_BATCH_SIZE, REVERSE_SEARCH_PATH,
};

const PYTHON_EXAMPLE: &str = include_str!("../../examples/reverse_search.py");
const RUST_EXAMPLE: &str = include_str!("../../examples/reverse_search.rs");

const BATCH_PYTHON_EXAMPLE: &str = include_str!("../../examples/reverse_search_batch.py");
const BATCH_RUST_EXAMPLE: &str = include_str!("../../examples/reverse_search_batch.rs");

/// Real alert of the running database that belongs to a single lineage.
const SAMPLE_OBJECT_ID_SINGLE: &str = "313791827801014320";
/// Identifier that matches no observation.
const SAMPLE_UNKNOWN_OBJECT_ID: &str = "000000000000000000";

/// `[dependencies]` needed by the Rust sample, shown above it.
const RUST_DEPENDENCIES: &str = r#"// Cargo.toml
// [dependencies]
// reqwest = { version = "0.12", features = ["json"] }
// serde = { version = "1", features = ["derive"] }
// tokio = { version = "1", features = ["macros", "rt-multi-thread"] }

"#;

/// Sample object id used in the documentation's `curl` command and JSON: a
/// real LSST alert that belongs to two lineages (checked against the running
/// database).
const SAMPLE_OBJECT_ID: &str = "313699504971841537";

/// Sample response, serialized from the real response type.
///
/// # Return
///
/// Pretty-printed JSON of the real two-lineage match of [`SAMPLE_OBJECT_ID`].
fn sample_response_json() -> String {
    let response = ReverseSearchResponse {
        object_id: SAMPLE_OBJECT_ID.to_string(),
        lineages: [
            (171, "FF2025ouzemdyvufld", 219280),
            (428, "FF2025ixcfnuzdkkak", 7260),
        ]
        .into_iter()
        .map(|(lineage_id, designation, branch_id)| LineageMatch {
            lineage_id,
            lineage_designation: designation.to_string(),
            best_branch_id: branch_id,
            matching_branch_ids: vec![branch_id],
            url: format!("/lineage/{designation}"),
        })
        .collect(),
    };
    serde_json::to_string_pretty(&response).unwrap_or_default()
}

/// The `curl` sample command.
///
/// # Return
///
/// A one-line `curl` invocation of the reverse-search endpoint.
fn curl_example() -> String {
    let path = REVERSE_SEARCH_PATH.replace("{object_id}", SAMPLE_OBJECT_ID);
    format!("curl -s http://localhost:8080{path}")
}

/// Builds the [`LineageMatch`] of a single-branch lineage (real sample data).
///
/// # Arguments
///
/// * `lineage_id` - numeric lineage id.
/// * `designation` - lineage designation.
/// * `branch_id` - the only branch, both matching and best.
///
/// # Return
///
/// The match, with the lineage page URL derived from `designation`.
fn single_branch_match(lineage_id: i64, designation: &str, branch_id: i64) -> LineageMatch {
    LineageMatch {
        lineage_id,
        lineage_designation: designation.to_string(),
        best_branch_id: branch_id,
        matching_branch_ids: vec![branch_id],
        url: format!("/lineage/{designation}"),
    }
}

/// Sample batch request body, serialized from the real request type.
///
/// # Return
///
/// Pretty-printed JSON asking for two known alerts and an unknown one.
fn batch_request_json() -> String {
    let request = BatchReverseSearchRequest {
        object_ids: vec![
            SAMPLE_OBJECT_ID.to_string(),
            SAMPLE_OBJECT_ID_SINGLE.to_string(),
            SAMPLE_UNKNOWN_OBJECT_ID.to_string(),
        ],
    };
    serde_json::to_string_pretty(&request).unwrap_or_default()
}

/// Sample batch response (real data), serialized from the real response type.
///
/// # Return
///
/// Pretty-printed JSON answering [`batch_request_json`].
fn batch_response_json() -> String {
    let response = BatchReverseSearchResponse {
        results: vec![
            ReverseSearchResponse {
                object_id: SAMPLE_OBJECT_ID.to_string(),
                lineages: vec![
                    single_branch_match(171, "FF2025ouzemdyvufld", 219280),
                    single_branch_match(428, "FF2025ixcfnuzdkkak", 7260),
                ],
            },
            ReverseSearchResponse {
                object_id: SAMPLE_OBJECT_ID_SINGLE.to_string(),
                lineages: vec![single_branch_match(4337, "FF2025xzqzwqmvdqqa", 180356)],
            },
        ],
        unknown_object_ids: vec![SAMPLE_UNKNOWN_OBJECT_ID.to_string()],
    };
    serde_json::to_string_pretty(&response).unwrap_or_default()
}

/// The batch `curl` sample command.
///
/// # Return
///
/// A multi-line `curl` invocation posting [`batch_request_json`].
fn batch_curl_example() -> String {
    let body = serde_json::to_string(&BatchReverseSearchRequest {
        object_ids: vec![
            SAMPLE_OBJECT_ID.to_string(),
            SAMPLE_OBJECT_ID_SINGLE.to_string(),
        ],
    })
    .unwrap_or_default();
    format!(
        "curl -s -X POST http://localhost:8080{BATCH_REVERSE_SEARCH_PATH} \\\n  -H 'Content-Type: application/json' \\\n  -d '{body}'"
    )
}

/// Field reference of a [`LineageMatch`].
const RESPONSE_FIELDS: &[(&str, &str, &str)] = &[
    ("object_id", "string", "The alert identifier that was searched."),
    (
        "lineages",
        "array",
        "Lineages containing the alert, ordered by lineage_id. Empty if the alert is known but not attached to any lineage.",
    ),
    ("lineages[].lineage_id", "integer", "Numeric identifier of the lineage."),
    ("lineages[].lineage_designation", "string", "Human-readable designation of the lineage."),
    (
        "lineages[].best_branch_id",
        "integer",
        "Best branch of the whole lineage (highest cumulative log-likelihood ratio). It is not necessarily one of the branches containing the alert.",
    ),
    (
        "lineages[].matching_branch_ids",
        "array of integers",
        "Branches of the lineage that contain the alert, ascending.",
    ),
    (
        "lineages[].url",
        "string",
        "Relative URL of the lineage page in the explorer.",
    ),
];

/// HTTP status codes of the reverse-search endpoint.
const STATUS_CODES: &[(&str, &str)] = &[
    ("200", "The alert exists. `lineages` may be empty."),
    ("404", "No alert has this object_id."),
    (
        "500",
        "Internal error. The body is a generic message; details are only in the server logs.",
    ),
];

/// Endpoints of the "Alerts" category, as collapsible entries.
#[component]
pub fn AlertsEndpoints() -> Element {
    let tabs = vec![
        CodeTab::new("Python", "python", PYTHON_EXAMPLE),
        CodeTab::new("Rust", "rust", format!("{RUST_DEPENDENCIES}{RUST_EXAMPLE}")),
        CodeTab::new("curl", "bash", curl_example()),
    ];

    rsx! {
        EndpointAccordion {
            method: "GET",
            path: REVERSE_SEARCH_PATH,
            summary: "Find the lineages of an alert",
            p { "Reverse search: given an alert, list the lineages that contain it." }
            p {
                code { "object_id" }
                " (path parameter) — the alert identifier."
            }

            h4 { class: "font-semibold mt-2", "Response" }
            div { class: "overflow-x-auto bg-base-100 rounded-box border border-base-300",
                table { class: "table table-sm",
                    thead {
                        tr {
                            th { "Field" }
                            th { "Type" }
                            th { "Description" }
                        }
                    }
                    tbody {
                        for (name , kind , description) in RESPONSE_FIELDS {
                            tr { key: "{name}",
                                td {
                                    code { "{name}" }
                                }
                                td { "{kind}" }
                                td { "{description}" }
                            }
                        }
                    }
                }
            }
            CodeTabs { tabs: vec![CodeTab::new("JSON", "json", sample_response_json())] }

            h4 { class: "font-semibold mt-2", "Status codes" }
            div { class: "overflow-x-auto bg-base-100 rounded-box border border-base-300",
                table { class: "table table-sm",
                    tbody {
                        for (status , description) in STATUS_CODES {
                            tr { key: "{status}",
                                td {
                                    code { "{status}" }
                                }
                                td { "{description}" }
                            }
                        }
                    }
                }
            }

            h4 { class: "font-semibold mt-2", "Examples" }
            p {
                "Each sample takes an object id, queries this endpoint and prints the \
                 matching lineages. Set "
                code { "FINK_FAT_URL" }
                " to target another server than "
                code { "http://localhost:8080" }
                "."
            }
            CodeTabs { tabs }
        }

        EndpointAccordion {
            method: "POST",
            path: BATCH_REVERSE_SEARCH_PATH,
            summary: "Find the lineages of several alerts at once",
            p {
                "Reverse search for a whole list of alerts: for each alert, list the lineages \
                 that contain it. The lookup is done with two database queries whatever the \
                 list length, so it is much cheaper than one request per alert."
            }
            p {
                "Send a JSON body with an "
                code { "object_ids" }
                " array of at most {MAX_BATCH_SIZE} distinct alert identifiers (duplicates are \
                 ignored). An identifier that matches no alert does not fail the request: it \
                 is listed in "
                code { "unknown_object_ids" }
                " of the response."
            }

            h4 { class: "font-semibold mt-2", "Response" }
            div { class: "overflow-x-auto bg-base-100 rounded-box border border-base-300",
                table { class: "table table-sm",
                    thead {
                        tr {
                            th { "Field" }
                            th { "Type" }
                            th { "Description" }
                        }
                    }
                    tbody {
                        for (name , kind , description) in BATCH_RESPONSE_FIELDS {
                            tr { key: "{name}",
                                td {
                                    code { "{name}" }
                                }
                                td { "{kind}" }
                                td { "{description}" }
                            }
                        }
                    }
                }
            }
            CodeTabs {
                tabs: vec![
                    CodeTab::new("Request", "json", batch_request_json()),
                    CodeTab::new("Response", "json", batch_response_json()),
                ],
            }

            h4 { class: "font-semibold mt-2", "Status codes" }
            div { class: "overflow-x-auto bg-base-100 rounded-box border border-base-300",
                table { class: "table table-sm",
                    tbody {
                        for (status , description) in BATCH_STATUS_CODES {
                            tr { key: "{status}",
                                td {
                                    code { "{status}" }
                                }
                                td { "{description}" }
                            }
                        }
                    }
                }
            }

            h4 { class: "font-semibold mt-2", "Examples" }
            CodeTabs {
                tabs: vec![
                    CodeTab::new("Python", "python", BATCH_PYTHON_EXAMPLE),
                    CodeTab::new(
                        "Rust",
                        "rust",
                        format!("{RUST_DEPENDENCIES}{BATCH_RUST_EXAMPLE}"),
                    ),
                    CodeTab::new("curl", "bash", batch_curl_example()),
                ],
            }
        }
    }
}

/// Field reference of the batch response.
const BATCH_RESPONSE_FIELDS: &[(&str, &str, &str)] = &[
    (
        "results",
        "array",
        "One entry per known alert, in request order, holding its object_id and its lineages (each with lineage_id, lineage_designation, best_branch_id, matching_branch_ids and url). lineages is empty if the alert is in no lineage.",
    ),
    (
        "unknown_object_ids",
        "array of strings",
        "Requested identifiers that match no observation, in request order.",
    ),
];

/// HTTP status codes of the batch endpoint.
const BATCH_STATUS_CODES: &[(&str, &str)] = &[
    (
        "200",
        "The request was processed. Unknown alerts are reported in unknown_object_ids.",
    ),
    (
        "400",
        "object_ids is empty or has more distinct identifiers than the limit.",
    ),
    (
        "500",
        "Internal error. The body is a generic message; details are only in the server logs.",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_json_round_trips_through_the_response_type() {
        let parsed: ReverseSearchResponse =
            serde_json::from_str(&sample_response_json()).expect("valid response JSON");
        assert_eq!(parsed.object_id, SAMPLE_OBJECT_ID);
        assert_eq!(parsed.lineages.len(), 2);
        assert_eq!(parsed.lineages[1].matching_branch_ids, vec![7260]);
    }

    #[test]
    fn batch_samples_round_trip_through_the_real_types() {
        let request: BatchReverseSearchRequest =
            serde_json::from_str(&batch_request_json()).expect("valid request JSON");
        assert_eq!(request.object_ids.len(), 3);
        let response: BatchReverseSearchResponse =
            serde_json::from_str(&batch_response_json()).expect("valid response JSON");
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.unknown_object_ids, vec![SAMPLE_UNKNOWN_OBJECT_ID]);
        assert!(request.object_ids.len() <= MAX_BATCH_SIZE);
    }

    #[test]
    fn batch_examples_target_the_documented_route() {
        assert!(BATCH_PYTHON_EXAMPLE.contains(BATCH_REVERSE_SEARCH_PATH));
        assert!(BATCH_RUST_EXAMPLE.contains(BATCH_REVERSE_SEARCH_PATH));
        assert!(batch_curl_example().contains(BATCH_REVERSE_SEARCH_PATH));
    }

    #[test]
    fn examples_target_the_documented_route() {
        let prefix = "/api/v1/alerts/";
        assert!(REVERSE_SEARCH_PATH.starts_with(prefix));
        assert!(PYTHON_EXAMPLE.contains(prefix));
        assert!(RUST_EXAMPLE.contains(prefix));
        assert!(curl_example().contains(prefix));
        assert!(!curl_example().contains('{'));
    }
}

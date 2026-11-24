//! Batch reverse search: find the lineages of several alerts in one request.
//!
//! Usage: `reverse_search_batch <object_id> [<object_id>...]`. The server
//! address is read from the `FINK_FAT_URL` environment variable (default
//! `http://localhost:8080`).
//!
//! Needs `reqwest` (feature `json`), `tokio` (features `macros`,
//! `rt-multi-thread`) and `serde` (feature `derive`).

use serde::{Deserialize, Serialize};

/// Request body of `POST /api/v1/alerts/lineages`.
#[derive(Debug, Serialize)]
struct BatchRequest {
    object_ids: Vec<String>,
}

/// A lineage that contains an alert.
#[derive(Debug, Deserialize)]
struct LineageMatch {
    lineage_id: i64,
    lineage_designation: String,
    best_branch_id: i64,
    matching_branch_ids: Vec<i64>,
    url: String,
}

/// Lineages of one known alert.
#[derive(Debug, Deserialize)]
struct AlertResult {
    object_id: String,
    lineages: Vec<LineageMatch>,
}

/// Response of `POST /api/v1/alerts/lineages`.
#[derive(Debug, Deserialize)]
struct BatchResponse {
    results: Vec<AlertResult>,
    unknown_object_ids: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let object_ids: Vec<String> = std::env::args().skip(1).collect();
    if object_ids.is_empty() {
        return Err("usage: reverse_search_batch <object_id> [<object_id>...]".into());
    }
    let base_url =
        std::env::var("FINK_FAT_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());

    let response = reqwest::Client::new()
        .post(format!("{base_url}/api/v1/alerts/lineages"))
        .json(&BatchRequest { object_ids })
        .send()
        .await?
        .error_for_status()?;
    let body: BatchResponse = response.json().await?;

    for result in body.results {
        if result.lineages.is_empty() {
            println!(
                "{}: known alert, but not part of any lineage",
                result.object_id
            );
        }
        for lineage in result.lineages {
            println!(
                "{} -> {} (id {}): best branch {}, matching branches {:?} -> {}{}",
                result.object_id,
                lineage.lineage_designation,
                lineage.lineage_id,
                lineage.best_branch_id,
                lineage.matching_branch_ids,
                base_url,
                lineage.url,
            );
        }
    }
    for object_id in body.unknown_object_ids {
        println!("{object_id}: unknown alert");
    }
    Ok(())
}

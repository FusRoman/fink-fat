//! Reverse search: find the lineages that contain a given alert.
//!
//! Usage: `reverse_search <object_id>`. The server address is read from the
//! `FINK_FAT_URL` environment variable (default `http://localhost:8080`).
//!
//! Needs `reqwest` (feature `json`), `tokio` (features `macros`,
//! `rt-multi-thread`) and `serde` (feature `derive`).

use reqwest::StatusCode;
use serde::Deserialize;

/// A lineage that contains the searched alert.
#[derive(Debug, Deserialize)]
struct LineageMatch {
    lineage_id: i64,
    lineage_designation: String,
    best_branch_id: i64,
    matching_branch_ids: Vec<i64>,
    url: String,
}

/// Response of `GET /api/v1/alerts/{object_id}/lineages`.
#[derive(Debug, Deserialize)]
struct ReverseSearchResponse {
    object_id: String,
    lineages: Vec<LineageMatch>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let object_id = std::env::args()
        .nth(1)
        .ok_or("usage: reverse_search <object_id>")?;
    let base_url =
        std::env::var("FINK_FAT_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());

    let response = reqwest::Client::new()
        .get(format!("{base_url}/api/v1/alerts/{object_id}/lineages"))
        .send()
        .await?;

    match response.status() {
        StatusCode::OK => {
            let body: ReverseSearchResponse = response.json().await?;
            if body.lineages.is_empty() {
                println!(
                    "{}: known alert, but not part of any lineage",
                    body.object_id
                );
            }
            for lineage in body.lineages {
                println!(
                    "{} (id {}): best branch {}, matching branches {:?} -> {}{}",
                    lineage.lineage_designation,
                    lineage.lineage_id,
                    lineage.best_branch_id,
                    lineage.matching_branch_ids,
                    base_url,
                    lineage.url,
                );
            }
        }
        StatusCode::NOT_FOUND => println!("{object_id}: unknown alert"),
        status => return Err(format!("unexpected status {status}").into()),
    }
    Ok(())
}

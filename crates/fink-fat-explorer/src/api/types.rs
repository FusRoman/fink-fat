//! JSON payloads of the REST API.
//!
//! These types do not depend on the `server` feature, so the front end can
//! deserialize them too.

use serde::{Deserialize, Serialize};

/// A lineage that contains the searched alert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageMatch {
    /// Numeric identifier of the lineage.
    pub lineage_id: i64,
    /// Human-readable designation of the lineage.
    pub lineage_designation: String,
    /// Best branch of the whole lineage (highest sanitized cumulative LLR),
    /// which is not necessarily one of the branches containing the alert.
    pub best_branch_id: i64,
    /// Branches of the lineage that contain the searched alert, ascending.
    pub matching_branch_ids: Vec<i64>,
    /// Relative URL of the lineage page in the explorer.
    pub url: String,
}

/// Response of the alert → lineages reverse search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReverseSearchResponse {
    /// The alert identifier that was searched (`observations.object_id`).
    pub object_id: String,
    /// Lineages containing the alert, ordered by `lineage_id`. Empty when the
    /// alert is known but not attached to any lineage.
    pub lineages: Vec<LineageMatch>,
}

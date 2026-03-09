use serde::{Deserialize, Serialize};

use crate::graph::edge::{Edge, EdgeKey};

/// One journal operation to transform the edge set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EdgeOp {
    /// Insert or replace an edge for a given `(from, to)` key.
    ///
    /// This is used both for new edges and for updating existing edges
    /// (cost/active, etc.).
    ///
    /// The key is implicit: `edge.key()` returns `EdgeKey { from: edge.from, to: edge.to }`.
    Upsert {
        /// Edge payload (core + endpoints). The identity key is `edge.key()`.
        edge: Edge,
    },

    /// Remove an edge for the given `(from, to)` key.
    ///
    /// Notes
    /// -----
    /// Removal is optional in some pipelines (you can instead keep edges but set
    /// `active=false`). Using `Remove` keeps the compacted snapshot smaller.
    Remove {
        /// Edge identity.
        key: EdgeKey,
    },
}

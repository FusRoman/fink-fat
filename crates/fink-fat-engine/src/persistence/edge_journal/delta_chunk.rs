use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::{
    night_id::NightId,
    persistence::{
        EDGE_JOURNAL_SCHEMA_VERSION, edge_journal::edge_op::EdgeOp, envelope::DiskEnvelope,
        error::PersistenceIoError,
    },
};

/// A per-night delta file payload.
///
/// This is typically persisted as `edges/delta-nid=<NightId>.bin`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeDeltaChunk {
    /// Night that produced these operations.
    pub night_id: NightId,
    /// Unix timestamp (seconds) for traceability.
    pub created_unix_s: i64,
    /// Operations produced by the pipeline for this night.
    pub ops: Vec<EdgeOp>,
}

impl EdgeDeltaChunk {
    /// Construct a new [`EdgeDeltaChunk`] for the given night.
    #[inline]
    fn new(night_id: NightId, created_unix_s: i64, ops: Vec<EdgeOp>) -> Self {
        Self {
            night_id,
            created_unix_s,
            ops,
        }
    }

    pub fn write(
        path: &Utf8PathBuf,
        night_id: NightId,
        created_unix_s: i64,
        ops: Vec<EdgeOp>,
    ) -> Result<(), PersistenceIoError> {
        let chunk = EdgeDeltaChunk::new(night_id, created_unix_s, ops);
        let env = DiskEnvelope::new(chunk, EDGE_JOURNAL_SCHEMA_VERSION, created_unix_s);
        env.save_enveloped(path)?;
        Ok(())
    }
}

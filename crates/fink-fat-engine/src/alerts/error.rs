use thiserror::Error;

use crate::{alerts::{AlertKey, DiaSourceId}, night_id::NightId};

#[derive(Debug, Error)]
pub enum InsertError {
    #[error(
        "Night mismatch: expected {expected:?}, found {found:?} at index {index} (dia_source_id: {dia_source_id})"
    )]
    NightMismatch {
        expected: NightId,
        found: NightId,
        dia_source_id: u64,
        index: usize,
    },

    #[error("Duplicate alert key: {0:?}")]
    DuplicateKey(AlertKey),

    #[error("Duplicate dia_source_id: {0}")]
    DuplicateId(DiaSourceId),
}

//! Error types for alert insertion into the [`AlertStore`](crate::alerts::store::AlertStore).
//!
//! These errors are raised when an alert cannot be inserted because it
//! violates a uniqueness or consistency constraint enforced by the store.

use thiserror::Error;

use crate::{alerts::{AlertKey, DiaSourceId}, night_id::NightId};

/// Error returned when inserting an alert into the [`AlertStore`](crate::alerts::store::AlertStore) fails.
///
/// The store enforces two invariants on insertion:
///
/// - **Night consistency**: all alerts appended to a given night must carry a
///   matching [`NightId`].
/// - **Identifier uniqueness**: no two alerts may share the same
///   [`AlertKey`] or [`DiaSourceId`].
///
/// Variants
/// --------
/// - [`NightMismatch`](Self::NightMismatch) – the alert's night does not match
///   the target night collection.
/// - [`DuplicateKey`](Self::DuplicateKey) – an alert with the same composite
///   key `(NightId, DiaSourceId)` already exists.
/// - [`DuplicateId`](Self::DuplicateId) – an alert with the same
///   `dia_source_id` already exists (globally unique constraint).
#[derive(Debug, Error)]
pub enum InsertError {
    /// The alert's [`NightId`] does not match the night it is being inserted into.
    ///
    /// This typically indicates a bug in the ingestion or rekeying logic
    /// rather than bad input data, since alerts are grouped by night before
    /// insertion.
    #[error(
        "Night mismatch: expected {expected:?}, found {found:?} at index {index} (dia_source_id: {dia_source_id})"
    )]
    NightMismatch {
        /// Night the store expected for this insertion batch.
        expected: NightId,
        /// Night actually found in the alert's key.
        found: NightId,
        /// The `dia_source_id` of the offending alert (for diagnostics).
        dia_source_id: u64,
        /// Position of the alert in the insertion batch.
        index: usize,
    },

    /// An alert with the same composite key `(NightId, DiaSourceId)` is already
    /// present in the store.
    #[error("Duplicate alert key: {0:?}")]
    DuplicateKey(AlertKey),

    /// An alert with the same `dia_source_id` is already present in the store.
    ///
    /// `dia_source_id` is expected to be globally unique across all nights.
    #[error("Duplicate dia_source_id: {0}")]
    DuplicateId(DiaSourceId),
}

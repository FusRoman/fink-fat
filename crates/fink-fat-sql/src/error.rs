//! Error type of the SQL loaders.

use thiserror::Error;

/// Errors raised while writing fink-fat tables to Postgres.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SqlError {
    /// Connecting to Postgres, or any statement / `COPY` in the transaction,
    /// failed. The transaction is dropped without commit, so Postgres
    /// discards every change made so far.
    #[error("Postgres error: {0}")]
    Postgres(#[from] postgres::Error),

    /// A schema migration ([`crate::schema::migrate`]) failed — the batch is
    /// rolled back, the database stays at its previous version — or the
    /// migration history does not match this binary: an applied migration
    /// was edited since (divergent) or is unknown to this (older) binary
    /// (missing). See the crate README, "Schema migrations".
    #[error("schema migration error: {0}")]
    Migration(#[from] refinery::Error),
}

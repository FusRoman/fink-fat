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
}

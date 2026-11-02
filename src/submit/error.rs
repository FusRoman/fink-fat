//! Typed errors for `fink-fat submit`.

use camino::Utf8PathBuf;
use thiserror::Error;

/// Everything that can go wrong running `fink-fat submit`, either for the
/// whole run (bad arguments, can't reach Postgres) or for one lineage within
/// it (folded into a per-lineage outcome by
/// [`crate::submit::process_lineage`] rather than aborting the whole batch —
/// see that function's docs).
#[derive(Debug, Error)]
pub enum SubmitError {
    /// Neither `--lineages` nor `--csv` was given (`clap`'s own
    /// `ArgGroup`/`conflicts_with` catches "both given"; this covers
    /// "neither given", which those attributes alone don't enforce for two
    /// plain `Option` fields).
    #[error("either --lineages or --csv must be given")]
    NoLineagesSpecified,

    /// `--csv` was given but its `lineage_designation` header column is
    /// missing.
    #[error("CSV file '{path}' has no 'lineage_designation' column")]
    CsvMissingColumn { path: Utf8PathBuf },

    /// Reading the `--csv` file itself failed.
    #[error("failed to read CSV file '{path}': {source}")]
    CsvRead {
        path: Utf8PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Reading the `--submitter-config` file itself failed.
    #[error("failed to read submitter config '{path}': {source}")]
    SubmitterConfigRead {
        path: Utf8PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The `--submitter-config` file's YAML didn't parse.
    #[error("failed to parse submitter config '{path}': {source}")]
    SubmitterConfigParse {
        path: Utf8PathBuf,
        #[source]
        source: serde_yaml::Error,
    },

    /// The `--submitter-config` file parsed, but is missing required fields
    /// (a header field `check_local_schema_violations` would reject, caught
    /// early rather than once per lineage in the batch).
    #[error("submitter config '{path}' is invalid: {}", .reasons.join("; "))]
    SubmitterConfigInvalid {
        path: Utf8PathBuf,
        reasons: Vec<String>,
    },

    /// A Postgres query (connect, eligibility check, fetch, or persist)
    /// failed.
    #[error("database error: {0}")]
    Database(#[from] postgres::Error),

    /// An MPC request (the submission POST itself) failed at the transport
    /// level (network/timeout/non-2xx) — a request that succeeded but
    /// returned an unparseable body is a [`Self::Ades`] instead.
    #[error("MPC request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Any pure ADES/MPC logic failure (invalid trkSub, unknown band, schema
    /// violation building the document, unparseable MPC response, ...).
    #[error(transparent)]
    Ades(#[from] fink_fat_ades::error::AdesError),
}

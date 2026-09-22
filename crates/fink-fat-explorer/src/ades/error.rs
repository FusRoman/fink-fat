//! Typed errors for ADES document construction, local schema validation,
//! submission to MPC's `submit_xml_test` endpoint, and polling its
//! test-submission status page for the real ingest verdict.

use thiserror::Error;

/// Everything that can go wrong building, serializing, or submitting an ADES
/// pre-submission XML file for a lineage.
#[derive(Debug, Error)]
pub enum AdesError {
    /// `trkSub` normalization failed (e.g. empty designation, or every
    /// character was stripped by the allowed-charset filter).
    #[error("cannot derive a valid trkSub from designation '{designation}': {reason}")]
    InvalidTrkSub { designation: String, reason: String },

    /// An observation's `filter` index has no known ADES band mapping.
    #[error("unknown filter index {filter} has no ADES band mapping")]
    UnknownBand { filter: i16 },

    /// Converting an MJD(TT) epoch to an ADES `obsTime` string failed.
    #[error("failed to convert MJD(TT) {mjd_tt} to an ADES obsTime: {reason}")]
    ObsTimeConversion { mjd_tt: f64, reason: String },

    /// The lineage has no observations left to export (either it had none to
    /// begin with, or every observation was removed as belonging to a
    /// singleton night).
    #[error("lineage '{lineage_designation}' has no observations to export")]
    NoObservations { lineage_designation: String },

    /// `quick_xml`'s serializer failed on an otherwise well-formed
    /// `AdesDocument`.
    #[cfg(feature = "server")]
    #[error("failed to serialize ADES document to XML: {0}")]
    XmlSerialize(#[from] quick_xml::SeError),

    /// An outbound HTTP request to MPC itself failed (network/timeout/
    /// non-2xx) — either the initial `submit_xml_test` POST or a later
    /// status-page GET, as opposed to a request that succeeded but returned
    /// an unparseable body.
    #[cfg(feature = "server")]
    #[error("MPC request failed: {0}")]
    McpRequest(#[from] reqwest::Error),

    /// MPC's `submit_xml_test` response body didn't contain the expected
    /// `"Submission ID is ..."` acknowledgement pattern.
    #[error("could not parse MPC submission acknowledgement: {0}")]
    McpSubmissionResponseParse(String),

    /// MPC's test-submission status page body matched none of the three
    /// known shapes (pending/"no such submission ID", valid, invalid) — a
    /// guard in case MPC changes that (undocumented) page's template.
    #[error("could not parse MPC submission status page: {0}")]
    McpStatusPageUnrecognized(String),

    /// The submission's status never resolved to `valid`/`invalid` within
    /// the polling budget — MPC may still be processing it.
    #[error("MPC status check for submission '{submission_id}' timed out")]
    McpStatusPollTimedOut { submission_id: String },
}

//! Typed errors for ADES document construction, local schema validation, and
//! the informational submission to MPC's `submit_xml_test` endpoint.

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

    /// The outbound POST to MPC's `submit_xml_test` endpoint itself failed
    /// (network/timeout/non-2xx), as opposed to a parseable acknowledgement.
    #[cfg(feature = "server")]
    #[error("MPC submission request failed: {0}")]
    McpSubmissionRequest(#[from] reqwest::Error),

    /// MPC's `submit_xml_test` response body didn't contain the expected
    /// `"Submission ID is ..."` acknowledgement pattern.
    #[error("could not parse MPC submission acknowledgement: {0}")]
    McpSubmissionResponseParse(String),
}

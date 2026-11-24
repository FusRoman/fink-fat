//! Typed errors for ADES document construction, local schema validation,
//! submission to MPC (`submit_xml`/`submit_xml_test`), and parsing the
//! responses of every MPC status-checking service this crate knows about
//! (the test-submission status page, the production Submission Status API,
//! WAMO).
//!
//! This crate performs no I/O itself (see the crate-level docs), so network
//! failures are not represented via a typed HTTP-client error — each caller
//! (`fink-fat-explorer`'s async `reqwest::Client`, `fink-fat`'s CLI
//! `reqwest::blocking::Client`) maps its own request failure to
//! [`AdesError::McpRequest`] with `.to_string()`, keeping this crate free of
//! a `reqwest` dependency.

use thiserror::Error;

/// Everything that can go wrong building, serializing, submitting, or
/// checking the status of an ADES pre-submission XML file for a lineage.
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
    #[error("failed to serialize ADES document to XML: {0}")]
    XmlSerialize(#[from] quick_xml::SeError),

    /// An outbound HTTP request to MPC itself failed (network/timeout/
    /// non-2xx) — the initial submission POST, or a later status-page/API
    /// GET, as opposed to a request that succeeded but returned an
    /// unparseable body. Carries the caller's HTTP client error rendered to
    /// a string (see the module docs for why this isn't a typed `reqwest`
    /// error).
    #[error("MPC request failed: {0}")]
    McpRequest(String),

    /// MPC's `submit_xml`/`submit_xml_test` response body didn't contain the
    /// expected `"Submission ID is ..."` acknowledgement pattern.
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

    /// MPC's production Submission Status API returned a body that couldn't
    /// be decoded as the documented `{accepted, pipeline_entry_time,
    /// fault_events}` JSON shape.
    #[error("could not parse MPC submission-status API response: {0}")]
    SubmissionStatusApiParse(String),

    /// A [`crate::wamo::build_wamo_request_body`] call would have carried
    /// more identifiers than WAMO's documented ~50,000-identifier limit per
    /// call.
    #[error("WAMO request has {count} identifiers, over the documented limit of {limit}")]
    WamoTooManyIdentifiers { count: usize, limit: usize },

    /// MPC's WAMO API returned a body that couldn't be decoded as its
    /// documented JSON response shape.
    #[error("could not parse WAMO API response: {0}")]
    WamoResponseParse(String),
}

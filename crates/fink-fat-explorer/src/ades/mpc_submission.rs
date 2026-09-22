//! Pure request-shaping and response-parsing for MPC's `submit_xml_test`
//! endpoint. The actual `reqwest` call lives in `server_fns.rs`; this module
//! only builds inputs and interprets outputs, so both are unit testable
//! without a live network call.
//!
//! A live probe of this endpoint (valid ADES, an ADES with `dec` out of
//! range, plain garbage text, and a non-ADES XML document) showed it is
//! **asynchronous**: every one of those inputs produced the exact same kind
//! of response — an acknowledgement carrying a `Submission ID`, never a
//! pass/fail verdict. The real validation report is emailed to the `ac2`
//! address afterward. This module therefore only extracts that
//! acknowledgement; it must never be treated as a validity check (see
//! `crate::ades::schema_validation` for the actual, local, blocking check).

use crate::ades::error::AdesError;

/// MPC's `submit_xml_test` endpoint URL.
pub const MPC_SUBMIT_XML_TEST_URL: &str = "https://www.minorplanetcenter.net/submit_xml_test";

/// Everything needed to build the `multipart/form-data` body for
/// `submit_xml_test`, decoupled from `reqwest::multipart::Form` so it can be
/// constructed and inspected in a unit test without pulling in `reqwest`.
#[derive(Debug, Clone, PartialEq)]
pub struct McpSubmissionRequest {
    /// The ADES XML content, not yet prefixed with `<`.
    pub xml: String,
    pub ack_message: String,
    pub ac2_email: String,
}

impl McpSubmissionRequest {
    /// The exact string that must become the multipart `source` field's
    /// value: the XML content prefixed with the literal `<` MPC's endpoint
    /// expects, mirroring curl's `-F "source=<file"` convention (confirmed
    /// against the live endpoint).
    pub fn source_field_value(&self) -> String {
        format!("<{}", self.xml)
    }
}

/// Marker text MPC's acknowledgement response is built around, observed on
/// the live endpoint: `"[ack]. Submission ID is <id>"`.
const SUBMISSION_ID_MARKER: &str = "Submission ID is ";

/// Parse MPC's raw `submit_xml_test` response body into the submission ID it
/// carries. This is **not** a validity verdict — see the module docs.
///
/// # Errors
/// Returns [`AdesError::McpSubmissionResponseParse`] if the body doesn't
/// contain the expected acknowledgement marker (e.g. the endpoint's own
/// input-size guard error, or an unrelated error page).
pub fn parse_mpc_submission_response(body: &str) -> Result<String, AdesError> {
    let id = body
        .split(SUBMISSION_ID_MARKER)
        .nth(1)
        .map(|rest| rest.trim().trim_end_matches('.').to_string())
        .filter(|id| !id.is_empty());

    id.ok_or_else(|| AdesError::McpSubmissionResponseParse(body.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_field_value_has_literal_prefix() {
        let request = McpSubmissionRequest {
            xml: "<ades/>".to_string(),
            ack_message: "ack".to_string(),
            ac2_email: "user@example.com".to_string(),
        };
        assert_eq!(request.source_field_value(), "<<ades/>");
    }

    #[test]
    fn parse_mpc_submission_response_extracts_id_from_live_fixture() {
        let body =
            "[fink-fat ADES export test].  Submission ID is 2026-09-22T12:46:00.699_00000kpA";
        assert_eq!(
            parse_mpc_submission_response(body).unwrap(),
            "2026-09-22T12:46:00.699_00000kpA"
        );
    }

    #[test]
    fn parse_mpc_submission_response_rejects_body_without_marker() {
        let body = "error: data size (170 bytes) not in allowed range (200 to 1000000000) bytes";
        assert!(matches!(
            parse_mpc_submission_response(body),
            Err(AdesError::McpSubmissionResponseParse(_))
        ));
    }
}

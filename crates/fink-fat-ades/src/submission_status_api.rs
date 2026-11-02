//! Request-shaping and response-parsing for MPC's documented, production
//! **Submission Status API** — coarse "did the pipeline accept this
//! submission for processing" status, keyed by submission ID. Distinct from
//! [`crate::wamo`] (fine, per-observation detail) and from
//! [`crate::mpc_submission`]'s test-tier HTML status page (which reports a
//! real valid/invalid verdict, but only for `submit_xml_test` submissions).
//!
//! Live-probed (read-only `GET`, no submission involved) while implementing
//! this module: **the endpoint takes its `submission_id` as a JSON request
//! body, not a query parameter** — a `GET` with a `submission_id=...` query
//! string alone 400s with `"/submission-status requires a JSON request
//! payload"`. A nonexistent submission ID 404s with a **plain-text** body
//! (`` submission `<id>` not found ``, not JSON) — callers must check the
//! HTTP status before calling [`parse_submission_status_response`], which
//! only handles the 200 JSON shape.

use serde::{Deserialize, Serialize};

use crate::error::AdesError;

/// MPC's production Submission Status API endpoint.
pub const MPC_SUBMISSION_STATUS_API_URL: &str =
    "https://data.minorplanetcenter.net/api/submission-status";

/// The JSON body every request to [`MPC_SUBMISSION_STATUS_API_URL`] must
/// send (as the body of a `GET`, per the live-probed behavior documented
/// above — unusual, but confirmed: a query parameter alone is rejected).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SubmissionStatusRequest<'a> {
    pub submission_id: &'a str,
}

/// One warning/error MPC's pipeline logged while processing a submission —
/// present even for an `accepted` submission (a warning that didn't prevent
/// acceptance). `phase`/`failure_code` are documented as MPC-internal and
/// not guaranteed stable across MPC releases, so they're kept as opaque
/// integers rather than a typed enum.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct FaultEvent {
    pub message: String,
    pub phase: i64,
    pub failure_code: i64,
}

/// The documented JSON shape of a successful (`HTTP 200`) response.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SubmissionStatusResponse {
    /// Whether MPC's ingest pipeline accepted the submission at all —
    /// *not* a final valid/invalid astrometric verdict, only "was it well
    /// formed enough to enter processing".
    pub accepted: bool,
    /// When the submission entered the processing pipeline, or `None` if
    /// still pending.
    pub pipeline_entry_time: Option<String>,
    #[serde(default)]
    pub fault_events: Vec<FaultEvent>,
}

/// Parse a `200 OK` response body from [`MPC_SUBMISSION_STATUS_API_URL`].
///
/// # Arguments
/// * `body` — the raw JSON response body text.
///
/// # Return
/// The parsed [`SubmissionStatusResponse`].
///
/// # Errors
/// Returns [`AdesError::SubmissionStatusApiParse`] if `body` isn't valid
/// JSON matching the documented shape. Does **not** handle a `404` (unknown
/// submission ID) — that response is plain text, not JSON; callers must
/// branch on the HTTP status code before calling this function.
pub fn parse_submission_status_response(body: &str) -> Result<SubmissionStatusResponse, AdesError> {
    serde_json::from_str(body).map_err(|e| AdesError::SubmissionStatusApiParse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submission_status_request_serializes_the_documented_body_shape() {
        let request = SubmissionStatusRequest {
            submission_id: "2026-09-22T12:46:00.699_00000kpA",
        };
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"submission_id":"2026-09-22T12:46:00.699_00000kpA"}"#
        );
    }

    #[test]
    fn parse_submission_status_response_accepts_the_documented_shape() {
        let body = r#"{
            "accepted": true,
            "pipeline_entry_time": "2026-09-22T12:46:05.123Z",
            "fault_events": [
                {"message": "minor precision warning", "phase": 2, "failure_code": 17}
            ]
        }"#;
        let parsed = parse_submission_status_response(body).unwrap();
        assert!(parsed.accepted);
        assert_eq!(
            parsed.pipeline_entry_time.as_deref(),
            Some("2026-09-22T12:46:05.123Z")
        );
        assert_eq!(parsed.fault_events.len(), 1);
        assert_eq!(parsed.fault_events[0].failure_code, 17);
    }

    #[test]
    fn parse_submission_status_response_defaults_fault_events_when_absent() {
        let body = r#"{"accepted": false, "pipeline_entry_time": null}"#;
        let parsed = parse_submission_status_response(body).unwrap();
        assert!(!parsed.accepted);
        assert!(parsed.pipeline_entry_time.is_none());
        assert!(parsed.fault_events.is_empty());
    }

    #[test]
    fn parse_submission_status_response_rejects_the_real_404_plain_text_body() {
        // Live-captured: a nonexistent submission ID 404s with this exact
        // plain-text body, not JSON — callers must branch on HTTP status
        // before reaching this parser (see the module docs).
        let body = "submission `fink-fat-nonexistent-probe-0001` not found";
        assert!(matches!(
            parse_submission_status_response(body),
            Err(AdesError::SubmissionStatusApiParse(_))
        ));
    }

    #[test]
    fn parse_submission_status_response_rejects_malformed_json() {
        assert!(matches!(
            parse_submission_status_response("not json at all"),
            Err(AdesError::SubmissionStatusApiParse(_))
        ));
    }
}

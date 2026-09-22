//! Pure request-shaping and response-parsing for MPC's `submit_xml_test`
//! endpoint and its (undocumented) test-submission status page. The actual
//! `reqwest` calls live in `server_fns.rs`; this module only builds inputs
//! and interprets outputs, so both are unit testable without a live network
//! call.
//!
//! A live probe of `submit_xml_test` (valid ADES, an ADES with `dec` out of
//! range, plain garbage text, and a non-ADES XML document) showed it is
//! **asynchronous**: every one of those inputs produced the exact same kind
//! of response — an acknowledgement carrying a `Submission ID`, never a
//! pass/fail verdict, and (contrary to what its form implies) it does not
//! reliably email a report either. The real verdict comes from a second,
//! separate, undocumented endpoint discovered by further live probing:
//! `submit-test.minorplanetcenter.net/submission_status/query/?id=<id>`
//! (distinct from MPC's officially documented, JSON, *production*
//! Submission Status API at `data.minorplanetcenter.net/api/submission-status`,
//! which doesn't recognize `submit_xml_test` ids — confirmed by testing it
//! directly). This status page takes a few seconds to become queryable after
//! submission (bare `"no such submission ID '<id>'"` text body until then),
//! so [`crate::ades::server_fns`] polls it rather than fetching it once.

use crate::ades::error::AdesError;

/// MPC's `submit_xml_test` endpoint URL.
pub const MPC_SUBMIT_XML_TEST_URL: &str = "https://www.minorplanetcenter.net/submit_xml_test";

/// Everything needed to build the `multipart/form-data` body for
/// `submit_xml_test`, decoupled from `reqwest::multipart::Form` so it can be
/// constructed and inspected in a unit test without pulling in `reqwest`.
#[derive(Debug, Clone, PartialEq)]
pub struct McpSubmissionRequest {
    pub xml: String,
    pub ack_message: String,
    pub ac2_email: String,
}

impl McpSubmissionRequest {
    /// The exact string that must become the multipart `source` field's
    /// value: the raw XML content, unmodified.
    ///
    /// MPC's own docs show `curl -F "source=<myobs.xml"` and warn "be sure
    /// to include the `<`" — that `<` is **curl's own command-line syntax**
    /// for "read this field's value from a file" (as opposed to `@file`,
    /// which uploads it as a distinct multipart file part with its own
    /// filename/content-type). It is never part of the bytes curl actually
    /// sends. An earlier version of this method prepended a literal `<` to
    /// the XML text here, which corrupted every submission with a spurious
    /// leading `<<?xml ...` — MPC's parser rejected all of them with `StartTag:
    /// invalid element name, line 1, column 2`, silently masquerading as a
    /// real ADES validation failure.
    pub fn source_field_value(&self) -> String {
        self.xml.clone()
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

/// MPC's test-submission status page base URL — query it with `?id=<submission_id>`
/// (`reqwest`'s `.query(&[("id", id)])` handles percent-encoding).
pub const MPC_SUBMISSION_STATUS_URL: &str =
    "https://submit-test.minorplanetcenter.net/submission_status/query/";

/// Text the status page's body starts with when the submission hasn't been
/// ingested yet (observed for up to ~5-10s after submitting) — a bare
/// plain-text response, not the templated HTML page.
const PENDING_MARKER: &str = "no such submission ID";

const FINAL_STATUS_MARKER: &str = "<b>Final Status:</b>";
const COMMENT_OPEN: &str = "<pre style=\"margin: 0\"><code>";
const COMMENT_CLOSE: &str = "</code></pre>";
const NO_WARNINGS_PLACEHOLDER: &str = "(No warnings or errors reported)";

/// The real verdict from one poll of MPC's test-submission status page —
/// unlike `submit_xml_test`'s own response, this carries an actual
/// valid/invalid outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum SubmissionStatusOutcome {
    /// Not ingested yet; the caller should wait and poll again.
    Pending,
    Valid,
    /// MPC's own per-phase error/warning messages, HTML entities already
    /// unescaped, one entry per `<pre><code>` block in the page's table
    /// (a single entry may itself contain multiple newline-separated lines).
    Invalid {
        comments: Vec<String>,
    },
}

/// Parse one response body from MPC's test-submission status page.
///
/// # Errors
/// Returns [`AdesError::McpStatusPageUnrecognized`] if the body matches
/// neither the "pending" plain-text shape nor the templated HTML page with a
/// `Final Status` of `valid`/`invalid` — a guard in case MPC changes this
/// (undocumented) page's template.
pub fn parse_submission_status_page(body: &str) -> Result<SubmissionStatusOutcome, AdesError> {
    let trimmed = body.trim();
    if trimmed.starts_with(PENDING_MARKER) {
        return Ok(SubmissionStatusOutcome::Pending);
    }

    let status = body
        .split(FINAL_STATUS_MARKER)
        .nth(1)
        .and_then(|rest| rest.split("</p>").next())
        .map(str::trim);

    match status {
        Some("valid") => Ok(SubmissionStatusOutcome::Valid),
        Some("invalid") => Ok(SubmissionStatusOutcome::Invalid {
            comments: extract_comments(body),
        }),
        _ => Err(AdesError::McpStatusPageUnrecognized(trimmed.to_string())),
    }
}

/// Extract every `<pre style="margin: 0"><code>...</code></pre>` comment
/// block from the status page's error/warning table, HTML-unescaped, with
/// the "no problems" placeholder row filtered out.
fn extract_comments(body: &str) -> Vec<String> {
    let mut comments = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find(COMMENT_OPEN) {
        rest = &rest[start + COMMENT_OPEN.len()..];
        let Some(end) = rest.find(COMMENT_CLOSE) else {
            break;
        };
        let unescaped = unescape_html_entities(&rest[..end]);
        rest = &rest[end + COMMENT_CLOSE.len()..];

        if unescaped != NO_WARNINGS_PLACEHOLDER {
            comments.push(unescaped);
        }
    }
    comments
}

/// Unescape the small, fixed set of HTML entities MPC's auto-generated
/// status page uses (a Django-style `escape()` filter): `&amp;`, `&lt;`,
/// `&gt;`, `&quot;`, `&#x27;`. Not a general HTML-entity decoder — every
/// entity this page can produce comes from echoing back fink-fat's own ADES
/// XML text inside plain-English error messages, so this fixed table covers
/// every case actually observed.
fn unescape_html_entities(s: &str) -> String {
    // `&amp;` must be unescaped last: unescaping it first could turn an
    // already-escaped sequence like a literal "&amp;lt;" into "<" instead of
    // the intended "&lt;".
    s.replace("&#x27;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_field_value_is_the_raw_xml_unmodified() {
        // Regression test: an earlier version prepended a literal `<`,
        // corrupting every real submission (see the method's doc comment).
        let request = McpSubmissionRequest {
            xml: "<?xml version=\"1.0\"?><ades/>".to_string(),
            ack_message: "ack".to_string(),
            ac2_email: "user@example.com".to_string(),
        };
        assert_eq!(request.source_field_value(), request.xml);
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

    // Real fixtures captured from live probes of
    // `submit-test.minorplanetcenter.net/submission_status/query/` during
    // implementation — see the module docs for why this endpoint exists.

    const PENDING_FIXTURE: &str = "no such submission ID '2026-09-22T14:54:56.149_00000kpq'";

    const VALID_FIXTURE: &str = r#"<!DOCTYPE html>
<html lang="en">
<body>
<p><b>Submission ID:</b> 2026-09-22T14:54:56.149_00000kpq</p>
<p><b>Final Status:</b> valid</p>
<h2>Warnings and Errors During Processing:</h2>
<table>
  <tr><th>phase</th><th>comment</th><th>timestamp</th></tr>
  <tr>
    <td></td>
    <td>
      <pre style="margin: 0"><code>(No warnings or errors reported)</code></pre>
    </td>
    <td></td>
  </tr>
</table>
</body>
</html>"#;

    const INVALID_DUPLICATE_FIXTURE: &str = r#"<!DOCTYPE html>
<html lang="en">
<body>
<p><b>Submission ID:</b> 2026-09-22T14:53:51.492_00000kpp</p>
<p><b>Final Status:</b> invalid</p>
<h2>Warnings and Errors During Processing:</h2>
<table>
  <tr><th>phase</th><th>comment</th><th>timestamp</th></tr>
  <tr>
    <td>2</td>
    <td>
      <pre style="margin: 0"><code>exact duplicate of submission 2026-09-22T12:46:00.699_00000kpA</code></pre>
    </td>
    <td>Sept. 22, 2026, 2:53 p.m.</td>
  </tr>
  <tr>
    <td>2</td>
    <td>
      <pre style="margin: 0"><code>duplicate of submission 2026-09-22T12:46:00.699_00000kpA</code></pre>
    </td>
    <td>Sept. 22, 2026, 2:53 p.m.</td>
  </tr>
  <tr>
    <td>2</td>
    <td>
      <pre style="margin: 0"><code>ingest failed</code></pre>
    </td>
    <td>Sept. 22, 2026, 2:53 p.m.</td>
  </tr>
</table>
</body>
</html>"#;

    #[test]
    fn parse_submission_status_page_recognizes_pending() {
        assert_eq!(
            parse_submission_status_page(PENDING_FIXTURE).unwrap(),
            SubmissionStatusOutcome::Pending
        );
    }

    #[test]
    fn parse_submission_status_page_recognizes_valid_with_placeholder_row() {
        assert_eq!(
            parse_submission_status_page(VALID_FIXTURE).unwrap(),
            SubmissionStatusOutcome::Valid
        );
    }

    #[test]
    fn parse_submission_status_page_extracts_invalid_comments() {
        let outcome = parse_submission_status_page(INVALID_DUPLICATE_FIXTURE).unwrap();
        assert_eq!(
            outcome,
            SubmissionStatusOutcome::Invalid {
                comments: vec![
                    "exact duplicate of submission 2026-09-22T12:46:00.699_00000kpA".to_string(),
                    "duplicate of submission 2026-09-22T12:46:00.699_00000kpA".to_string(),
                    "ingest failed".to_string(),
                ],
            }
        );
    }

    #[test]
    fn parse_submission_status_page_rejects_unrecognized_body() {
        assert!(matches!(
            parse_submission_status_page("<html>something MPC never sent before</html>"),
            Err(AdesError::McpStatusPageUnrecognized(_))
        ));
    }

    #[test]
    fn unescape_html_entities_handles_the_entities_seen_in_real_error_messages() {
        // Taken from a real captured obsTime schema-validation error.
        let escaped = "Element &#x27;obsTime&#x27;: &#x27;2026-04-04T23:59:23.254790&#x27; is not a valid value of the union type &#x27;TimeType&#x27;.";
        assert_eq!(
            unescape_html_entities(escaped),
            "Element 'obsTime': '2026-04-04T23:59:23.254790' is not a valid value of the union type 'TimeType'."
        );
    }
}

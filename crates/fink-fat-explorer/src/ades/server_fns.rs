//! `#[server]` orchestration for ADES export: fetch a lineage's observations
//! → remove singleton-night observations → build the ADES document →
//! validate it locally against `submit.xsd` → if locally valid, submit it to
//! MPC's `submit_xml_test` endpoint → poll MPC's test-submission status page
//! for the real ingest verdict (see `crate::ades::mpc_submission` for why a
//! second, separate request is needed for that verdict).
//!
//! This is the only module in `crate::ades` that performs I/O; every step it
//! calls into (`model`, `schema_validation`, `xml`, `mpc_submission`) is a
//! pure function.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ades::model::{AdesHeaderInput, SingletonNightSummary, SubmissionAdvisory};

/// How long to wait between polls of MPC's test-submission status page.
#[cfg(feature = "server")]
const MPC_STATUS_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// How many times to poll before giving up — ~40s of budget, comfortably
/// above the ~5-10s ingest delay observed live.
#[cfg(feature = "server")]
const MPC_STATUS_POLL_MAX_ATTEMPTS: usize = 20;

/// The real outcome of submitting an ADES document to MPC, once its
/// (undocumented) test-submission status page has been polled to a
/// conclusion. This — not just [`AdesExportResult::schema_valid`] — is what
/// gates the download button: `schema_valid` is only a fast local pre-check
/// that avoids wasting an MPC round trip on a document already known not to
/// conform.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum McpVerdict {
    /// `schema_valid` was false — never submitted to MPC at all.
    NotSubmitted,
    /// The initial `submit_xml_test` POST itself failed (network/timeout/
    /// non-2xx), or its response couldn't be parsed for a submission ID.
    SubmissionFailed(String),
    /// A submission ID was obtained, but its status never resolved to
    /// `valid`/`invalid` within the polling budget — MPC may still be
    /// processing it. Re-check later with [`check_mpc_submission_status`]
    /// rather than resubmitting (which risks a spurious MPC "duplicate"
    /// rejection if the content hasn't changed).
    PollTimedOut {
        submission_id: String,
    },
    Valid {
        submission_id: String,
    },
    /// MPC's own per-phase error/warning messages — more authoritative than
    /// fink-fat's local `submit.xsd` approximation.
    Invalid {
        submission_id: String,
        comments: Vec<String>,
    },
}

/// Everything the export modal needs after one export round trip: the
/// generated XML text (so the client can trigger the download without a
/// second server round trip), the local `submit.xsd` conformance pre-check,
/// the non-blocking submission advisories, and MPC's real verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdesExportResult {
    pub xml: String,
    pub file_name: String,
    pub schema_valid: bool,
    pub schema_violations: Vec<String>,
    pub singleton_summary: SingletonNightSummary,
    pub advisory: SubmissionAdvisory,
    pub mpc_verdict: McpVerdict,
}

impl AdesExportResult {
    /// Whether the download button should be enabled: local pre-check
    /// passed *and* MPC itself confirmed the submission as valid.
    pub fn download_allowed(&self) -> bool {
        self.schema_valid && matches!(self.mpc_verdict, McpVerdict::Valid { .. })
    }
}

/// Poll MPC's test-submission status page for `submission_id` until it
/// resolves to `valid`/`invalid`, or the polling budget is exhausted.
///
/// # Errors
/// Propagates [`crate::ades::error::AdesError::McpRequest`] if a poll's GET
/// request itself fails,
/// [`crate::ades::error::AdesError::McpStatusPageUnrecognized`] if a
/// response matches none of the known page shapes, or returns
/// [`crate::ades::error::AdesError::McpStatusPollTimedOut`] if every attempt
/// came back `Pending`.
#[cfg(feature = "server")]
async fn poll_mpc_submission_status(
    submission_id: &str,
) -> Result<crate::ades::mpc_submission::SubmissionStatusOutcome, crate::ades::error::AdesError> {
    use crate::ades::error::AdesError;
    use crate::ades::mpc_submission::{
        parse_submission_status_page, SubmissionStatusOutcome, MPC_SUBMISSION_STATUS_URL,
    };
    use crate::get_http_client;

    let client = get_http_client().await;
    for _ in 0..MPC_STATUS_POLL_MAX_ATTEMPTS {
        let body = client
            .get(MPC_SUBMISSION_STATUS_URL)
            .query(&[("id", submission_id)])
            .send()
            .await
            .map_err(|e| AdesError::McpRequest(e.to_string()))?
            .text()
            .await
            .map_err(|e| AdesError::McpRequest(e.to_string()))?;

        match parse_submission_status_page(&body)? {
            SubmissionStatusOutcome::Pending => {
                tokio::time::sleep(MPC_STATUS_POLL_INTERVAL).await;
            }
            outcome
            @ (SubmissionStatusOutcome::Valid | SubmissionStatusOutcome::Invalid { .. }) => {
                return Ok(outcome)
            }
        }
    }

    Err(AdesError::McpStatusPollTimedOut {
        submission_id: submission_id.to_string(),
    })
}

/// Turn the outcome of [`poll_mpc_submission_status`] into the `McpVerdict`
/// the client sees — shared by [`submit_and_await_mpc_verdict`] and
/// [`check_mpc_submission_status`] so both fold the same set of outcomes the
/// same way.
#[cfg(feature = "server")]
fn verdict_from_poll_result(
    submission_id: String,
    result: Result<
        crate::ades::mpc_submission::SubmissionStatusOutcome,
        crate::ades::error::AdesError,
    >,
) -> McpVerdict {
    use crate::ades::error::AdesError;
    use crate::ades::mpc_submission::SubmissionStatusOutcome;

    match result {
        Ok(SubmissionStatusOutcome::Valid) => McpVerdict::Valid { submission_id },
        Ok(SubmissionStatusOutcome::Invalid { comments }) => McpVerdict::Invalid {
            submission_id,
            comments,
        },
        Ok(SubmissionStatusOutcome::Pending) => unreachable!(
            "poll_mpc_submission_status only returns Ok on Valid/Invalid, Pending loops internally"
        ),
        Err(AdesError::McpStatusPollTimedOut { submission_id }) => {
            McpVerdict::PollTimedOut { submission_id }
        }
        Err(err) => McpVerdict::SubmissionFailed(err.to_string()),
    }
}

/// Submit `xml` to MPC's `submit_xml_test` endpoint and poll its status page
/// to a conclusion, producing the real [`McpVerdict`]. Never returns an
/// `Err` itself — every failure mode (request, parsing, timeout) is folded
/// into a `McpVerdict` variant, since none of them should abort the whole
/// export (the XML is still valid and downloadable once `schema_valid` and
/// a `Valid` verdict agree).
#[cfg(feature = "server")]
async fn submit_and_await_mpc_verdict(xml: &str, header: &AdesHeaderInput) -> McpVerdict {
    use crate::ades::mpc_submission::{
        parse_mpc_submission_response, McpSubmissionRequest, MPC_SUBMIT_XML_TEST_URL,
    };
    use crate::get_http_client;

    let request = McpSubmissionRequest {
        xml: xml.to_string(),
        ack_message: header.ack_message.clone(),
        ac2_email: header.ac2_email.clone(),
    };
    let client = get_http_client().await;
    let submission_response = client
        .post(MPC_SUBMIT_XML_TEST_URL)
        .multipart(
            reqwest::multipart::Form::new()
                .text("source", request.source_field_value())
                .text("ack", request.ack_message.clone())
                .text("ac2", request.ac2_email.clone()),
        )
        .send()
        .await
        .and_then(|r| r.error_for_status());

    let submission_id = match submission_response {
        Ok(response) => match response.text().await {
            Ok(body) => match parse_mpc_submission_response(&body) {
                Ok(id) => id,
                Err(err) => return McpVerdict::SubmissionFailed(err.to_string()),
            },
            Err(err) => {
                return McpVerdict::SubmissionFailed(format!("failed to read MPC response: {err}"))
            }
        },
        Err(err) => {
            return McpVerdict::SubmissionFailed(format!("MPC submission request failed: {err}"))
        }
    };

    let result = poll_mpc_submission_status(&submission_id).await;
    verdict_from_poll_result(submission_id, result)
}

/// Build this lineage's ADES pre-submission XML from its best branch's
/// observations and the user-confirmed header form, validate it locally,
/// and — if valid — submit it to MPC and wait for its real verdict (see the
/// module docs for why that takes a submission plus a separate status poll).
#[server]
pub async fn export_and_validate_ades(
    lineage_designation: String,
    header: AdesHeaderInput,
) -> Result<AdesExportResult, ServerFnError> {
    use crate::ades::model::{remove_singleton_nights, NightObservation};
    use crate::ades::schema_validation::check_local_schema_violations;
    use crate::ades::xml::{ades_document_to_xml, build_ades_document};
    use crate::lineage_page::observations_table::fetch_branch_observations;
    use crate::orbit_fit::run::resolve_best_branch_id;

    let Some(branch_id) = resolve_best_branch_id(&lineage_designation)
        .await
        .map_err(ServerFnError::new)?
    else {
        return Err(ServerFnError::new(format!(
            "lineage '{lineage_designation}' has no branches"
        )));
    };

    let observations = fetch_branch_observations(branch_id)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    let night_observations: Vec<NightObservation> = observations
        .into_iter()
        .map(|observation| NightObservation {
            night_id: observation.night_id,
            observation,
        })
        .collect();

    let (kept, singleton_summary) = remove_singleton_nights(&night_observations);
    let advisory = crate::ades::model::check_submission_recommendations(&kept);

    let schema_violations = check_local_schema_violations(&lineage_designation, &kept, &header);
    let schema_valid = schema_violations.is_empty();

    // Building the document can still fail even when the local schema check
    // passes (e.g. genuinely no observations survived), in which case we
    // surface that as a schema violation rather than a hard server error —
    // the user should see it as "why the badge is red", not as a broken page.
    let doc = match build_ades_document(&lineage_designation, &kept, &header) {
        Ok(doc) => doc,
        Err(err) => {
            return Ok(AdesExportResult {
                xml: String::new(),
                file_name: String::new(),
                schema_valid: false,
                schema_violations: {
                    let mut violations = schema_violations;
                    violations.push(err.to_string());
                    violations
                },
                singleton_summary,
                advisory,
                mpc_verdict: McpVerdict::NotSubmitted,
            });
        }
    };

    let xml = ades_document_to_xml(&doc).map_err(|e| ServerFnError::new(e.to_string()))?;
    let file_name = format!("{}.ades.xml", doc.obs_block.obs_data.optical[0].trk_sub);

    if !schema_valid {
        return Ok(AdesExportResult {
            xml,
            file_name,
            schema_valid: false,
            schema_violations,
            singleton_summary,
            advisory,
            mpc_verdict: McpVerdict::NotSubmitted,
        });
    }

    let mpc_verdict = submit_and_await_mpc_verdict(&xml, &header).await;

    Ok(AdesExportResult {
        xml,
        file_name,
        schema_valid: true,
        schema_violations,
        singleton_summary,
        advisory,
        mpc_verdict,
    })
}

/// Re-check an already-submitted lineage's MPC status without submitting
/// again — used after [`McpVerdict::PollTimedOut`], since resubmitting
/// identical content risks MPC flagging it as a duplicate.
#[server]
pub async fn check_mpc_submission_status(
    submission_id: String,
) -> Result<McpVerdict, ServerFnError> {
    let result = poll_mpc_submission_status(&submission_id).await;
    Ok(verdict_from_poll_result(submission_id, result))
}

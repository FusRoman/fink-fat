//! `#[server]` orchestration for ADES export: fetch a lineage's observations
//! → remove singleton-night observations → build the ADES document →
//! validate it locally against `submit.xsd` → if locally valid, submit it to
//! MPC's `submit_xml_test` endpoint for an informational acknowledgement.
//!
//! This is the only module in `crate::ades` that performs I/O; every step it
//! calls into (`model`, `schema_validation`, `xml`, `mpc_submission`) is a
//! pure function.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ades::model::{AdesHeaderInput, SingletonNightSummary, SubmissionAdvisory};

/// Everything the export modal needs after one export round trip: the
/// generated XML text (so the client can trigger the download without a
/// second server round trip), the local `submit.xsd` conformance verdict
/// (the sole gate on the download button), the non-blocking submission
/// advisories, and — only when locally valid — the outcome of submitting the
/// file to MPC's `submit_xml_test` endpoint for its own (asynchronous, later
/// emailed) validation report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdesExportResult {
    pub xml: String,
    pub file_name: String,
    pub schema_valid: bool,
    pub schema_violations: Vec<String>,
    pub singleton_summary: SingletonNightSummary,
    pub advisory: SubmissionAdvisory,
    /// `None` if `schema_valid` is false (never submitted). Otherwise
    /// `Some(Ok(submission_id))` or `Some(Err(message))` if the MPC request
    /// itself failed — either way this never changes `schema_valid`.
    pub mpc_submission: Option<Result<String, String>>,
}

/// Build this lineage's ADES pre-submission XML from its best branch's
/// observations and the user-confirmed header form, validate it locally,
/// and — if valid — submit it to MPC's `submit_xml_test` endpoint for an
/// acknowledgement. See the module docs for why that submission is not a
/// validity verdict.
#[server]
pub async fn export_and_validate_ades(
    lineage_designation: String,
    header: AdesHeaderInput,
) -> Result<AdesExportResult, ServerFnError> {
    use crate::ades::model::{remove_singleton_nights, NightObservation};
    use crate::ades::mpc_submission::{
        parse_mpc_submission_response, McpSubmissionRequest, MPC_SUBMIT_XML_TEST_URL,
    };
    use crate::ades::schema_validation::check_local_schema_violations;
    use crate::ades::xml::{ades_document_to_xml, build_ades_document};
    use crate::get_http_client;
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
                mpc_submission: None,
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
            mpc_submission: None,
        });
    }

    let request = McpSubmissionRequest {
        xml: xml.clone(),
        ack_message: header.ack_message.clone(),
        ac2_email: header.ac2_email.clone(),
    };
    let client = get_http_client().await;
    let mpc_submission = match client
        .post(MPC_SUBMIT_XML_TEST_URL)
        .multipart(
            reqwest::multipart::Form::new()
                .text("source", request.source_field_value())
                .text("ack", request.ack_message.clone())
                .text("ac2", request.ac2_email.clone()),
        )
        .send()
        .await
        .and_then(|r| r.error_for_status())
    {
        Ok(response) => match response.text().await {
            Ok(body) => Some(parse_mpc_submission_response(&body).map_err(|e| e.to_string())),
            Err(e) => Some(Err(format!("failed to read MPC response: {e}"))),
        },
        Err(e) => Some(Err(format!("MPC submission request failed: {e}"))),
    };

    Ok(AdesExportResult {
        xml,
        file_name,
        schema_valid: true,
        schema_violations,
        singleton_summary,
        advisory,
        mpc_submission,
    })
}

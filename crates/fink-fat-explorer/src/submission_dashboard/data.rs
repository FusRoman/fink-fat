//! Server functions for the "Submission" page: a fresh-query "which
//! lineages are ready to submit" candidate list, and a fresh-query
//! submission history/status table.
//!
//! Both deliberately bypass the homepage snapshot's own caching for the
//! `mpc_submissions` half of the picture — same rationale as
//! `cross_match_dashboard::data`'s doc comment: that table is written live,
//! by `fink-fat submit` runs and (once "Refresh status" is implemented)
//! this page's own status checks, with no `fink-fat convert`-tied
//! invalidation hook. The eligibility half (`QualityTier`) *is* read from
//! the snapshot — it's exactly the same immutable-until-`convert` data the
//! homepage badge itself shows, so there's no reason to re-derive it with a
//! fresh query here.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// One lineage ready to hand to `fink-fat submit`: eligible
/// ([`fink_fat_ades::quality_tier::QualityTier::is_submission_eligible`])
/// and not already present in `mpc_submissions`.
///
/// This is a **preview**, not the authoritative gate: `fink-fat submit`'s
/// own step-0/step-1 checks (which additionally catch an observation-id
/// overlap under a renamed/merged lineage) are what actually decide whether
/// a submission proceeds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubmissionCandidate {
    pub lineage_designation: String,
    pub quality_tier_label: String,
}

/// Every currently-eligible, not-yet-submitted lineage, sorted by
/// designation — feeds the "Prepare a submission" panel's count, CSV
/// download, and copy-pastable `fink-fat submit` command.
///
/// # Return
/// The candidates. Empty (not an error) while the homepage snapshot is
/// still warming up.
///
/// # Errors
/// The `mpc_submissions` query failing, as a `ServerFnError`.
#[server]
pub async fn get_submission_candidates() -> Result<Vec<SubmissionCandidate>, ServerFnError> {
    use crate::get_pool;
    use crate::homepage::snapshot;
    use std::collections::HashSet;

    let Some(snap) = snapshot::snapshot().await else {
        return Ok(Vec::new());
    };

    let pool = get_pool().await;
    let already_submitted: HashSet<String> =
        sqlx::query_scalar("SELECT DISTINCT lineage_designation FROM mpc_submissions")
            .fetch_all(pool)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?
            .into_iter()
            .collect();

    let mut candidates: Vec<SubmissionCandidate> = snap
        .lineages
        .iter()
        .map(|entry| &snap.branches[entry.best as usize])
        .filter(|branch| branch.quality_tier.is_submission_eligible())
        .filter(|branch| !already_submitted.contains(branch.lineage_designation.as_ref()))
        .map(|branch| SubmissionCandidate {
            lineage_designation: branch.lineage_designation.to_string(),
            quality_tier_label: branch.quality_tier.label().to_string(),
        })
        .collect();
    candidates.sort_by(|a, b| a.lineage_designation.cmp(&b.lineage_designation));
    Ok(candidates)
}

/// One `fink-fat submit` attempt on record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubmissionRow {
    pub id: i64,
    pub lineage_designation: String,
    /// `"test"` or `"production"` — see
    /// [`fink_fat_ades::mpc_submission::SubmitEndpoint`].
    pub endpoint: String,
    /// `None` if the submission POST itself failed before MPC ever
    /// acknowledged it.
    pub submission_id: Option<String>,
    /// `"pending"` | `"accepted"` | `"rejected"` | `"error"`.
    pub verdict: String,
    pub submitted_at: String,
    pub verdict_checked_at: Option<String>,
    /// The coarse verdict's own detail — `{"comments": [...]}` (test tier)
    /// or `{"pipeline_entry_time": ..., "fault_events": [...]}`
    /// (production), whichever `refresh_submission_status` last stored.
    pub verdict_detail: Option<serde_json::Value>,
    /// The last WAMO lookup's matching observations
    /// (`Vec<fink_fat_ades::wamo::WamoObservation>`, serialized), `Some([])`
    /// if WAMO was queried but found nothing (yet) — a normal state for a
    /// production submission MPC hasn't finished processing — and `None` if
    /// WAMO has never been queried for this row at all (always the case for
    /// test-tier rows, which WAMO never indexes).
    pub wamo_detail: Option<serde_json::Value>,
    pub wamo_checked_at: Option<String>,
}

/// Fetches one `mpc_submissions` row by id.
///
/// # Arguments
/// * `pool` — the Postgres pool.
/// * `id` — the row's `id`.
///
/// # Return
/// The row, or `None` if no row with that id exists.
///
/// # Errors
/// The query failing, as `sqlx::Error`.
#[cfg(feature = "server")]
async fn fetch_submission_row(
    pool: &sqlx::PgPool,
    id: i64,
) -> Result<Option<SubmissionRow>, sqlx::Error> {
    #[derive(sqlx::FromRow)]
    struct SubmissionRowSql {
        id: i64,
        lineage_designation: String,
        endpoint: String,
        submission_id: Option<String>,
        verdict: String,
        submitted_at: chrono::DateTime<chrono::Utc>,
        verdict_checked_at: Option<chrono::DateTime<chrono::Utc>>,
        verdict_detail: Option<serde_json::Value>,
        wamo_detail: Option<serde_json::Value>,
        wamo_checked_at: Option<chrono::DateTime<chrono::Utc>>,
    }

    let row: Option<SubmissionRowSql> = sqlx::query_as(
        "SELECT id, lineage_designation, endpoint, submission_id, verdict, \
                submitted_at, verdict_checked_at, verdict_detail, wamo_detail, wamo_checked_at
         FROM mpc_submissions
         WHERE id = $1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| SubmissionRow {
        id: r.id,
        lineage_designation: r.lineage_designation,
        endpoint: r.endpoint,
        submission_id: r.submission_id,
        verdict: r.verdict,
        submitted_at: r.submitted_at.to_rfc3339(),
        verdict_checked_at: r.verdict_checked_at.map(|t| t.to_rfc3339()),
        verdict_detail: r.verdict_detail,
        wamo_detail: r.wamo_detail,
        wamo_checked_at: r.wamo_checked_at.map(|t| t.to_rfc3339()),
    }))
}

/// Every submission attempt on record, most recent first.
///
/// # Return
/// The submission history rows.
///
/// # Errors
/// The `mpc_submissions` query failing, as a `ServerFnError`.
#[server]
pub async fn get_submission_history() -> Result<Vec<SubmissionRow>, ServerFnError> {
    use crate::get_pool;

    #[derive(sqlx::FromRow)]
    struct SubmissionRowSql {
        id: i64,
        lineage_designation: String,
        endpoint: String,
        submission_id: Option<String>,
        verdict: String,
        submitted_at: chrono::DateTime<chrono::Utc>,
        verdict_checked_at: Option<chrono::DateTime<chrono::Utc>>,
        verdict_detail: Option<serde_json::Value>,
        wamo_detail: Option<serde_json::Value>,
        wamo_checked_at: Option<chrono::DateTime<chrono::Utc>>,
    }

    let pool = get_pool().await;
    let rows: Vec<SubmissionRowSql> = sqlx::query_as(
        "SELECT id, lineage_designation, endpoint, submission_id, verdict, \
                submitted_at, verdict_checked_at, verdict_detail, wamo_detail, wamo_checked_at
         FROM mpc_submissions
         ORDER BY submitted_at DESC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(rows
        .into_iter()
        .map(|r| SubmissionRow {
            id: r.id,
            lineage_designation: r.lineage_designation,
            endpoint: r.endpoint,
            submission_id: r.submission_id,
            verdict: r.verdict,
            submitted_at: r.submitted_at.to_rfc3339(),
            verdict_checked_at: r.verdict_checked_at.map(|t| t.to_rfc3339()),
            verdict_detail: r.verdict_detail,
            wamo_detail: r.wamo_detail,
            wamo_checked_at: r.wamo_checked_at.map(|t| t.to_rfc3339()),
        })
        .collect())
}

/// Picks which of a WAMO response's two identifier lookups to keep for
/// storage: the bare submission ID's own matches if it found any, else the
/// guessed first block ID's — whichever actually matched. Neither matching
/// is a normal, expected state (not an error): MPC hasn't linked/published
/// the submission yet, or (for a submission split into more than one
/// block) the guessed `_01` block wasn't the right one.
///
/// # Arguments
/// * `response` — the parsed WAMO response, queried for both identifiers.
/// * `submission_id` — the bare submission ID.
/// * `first_block_id` — [`fink_fat_ades::wamo::first_block_id`] applied to
///   `submission_id`.
///
/// # Return
/// The matching observations, empty if neither identifier matched.
#[cfg(feature = "server")]
fn select_wamo_observations<'a>(
    response: &'a fink_fat_ades::wamo::WamoResponse,
    submission_id: &str,
    first_block_id: &str,
) -> Vec<&'a fink_fat_ades::wamo::WamoObservation> {
    let by_submission_id = response.observations_for(submission_id);
    if !by_submission_id.is_empty() {
        return by_submission_id;
    }
    response.observations_for(first_block_id)
}

/// Queries WAMO for a production submission's per-observation detail (both
/// the bare submission ID and its guessed first block ID — see
/// [`select_wamo_observations`]), returning the matches ready to store.
/// Never queried for test-tier submissions — WAMO never indexes
/// `submit_xml_test` submissions (confirmed by live probing while building
/// this feature), so it would only ever come back empty for those.
///
/// # Arguments
/// * `submission_id` — the bare submission ID to look up.
///
/// # Return
/// The matching observations as a JSON array (possibly empty — a normal
/// "not processed yet" state).
///
/// # Errors
/// The WAMO request or response parsing failing, as a `ServerFnError`.
#[cfg(feature = "server")]
async fn fetch_wamo_detail(submission_id: &str) -> Result<serde_json::Value, ServerFnError> {
    use crate::get_http_client;
    use fink_fat_ades::wamo::{
        build_wamo_request_body, first_block_id, parse_wamo_response, WamoIdentifier, MPC_WAMO_URL,
    };

    let block_id = first_block_id(submission_id);
    let identifiers = [
        WamoIdentifier::SubmissionBlockId(submission_id.to_string()),
        WamoIdentifier::SubmissionBlockId(block_id.clone()),
    ];
    let body =
        build_wamo_request_body(&identifiers).map_err(|e| ServerFnError::new(e.to_string()))?;

    let client = get_http_client().await;
    let text = client
        .get(MPC_WAMO_URL)
        .json(&body)
        .send()
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?
        .text()
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;
    let response = parse_wamo_response(&text).map_err(|e| ServerFnError::new(e.to_string()))?;

    let observations = select_wamo_observations(&response, submission_id, &block_id);
    serde_json::to_value(observations).map_err(|e| ServerFnError::new(e.to_string()))
}

/// Re-checks one submission's status against MPC (never re-submits) and
/// persists the result. `endpoint = "test"` polls
/// [`fink_fat_ades::mpc_submission`]'s test-tier status page once (the same
/// page `ades::server_fns` already polls in a loop right after submitting —
/// here it's a single, user-triggered check, not a wait-to-conclusion poll).
/// `endpoint = "production"` calls the documented
/// [`fink_fat_ades::submission_status_api`], then additionally queries
/// [`fink_fat_ades::wamo`] for per-observation detail (test-tier rows skip
/// this — WAMO never indexes them). All of these are read-only GETs against
/// MPC — this never submits anything.
///
/// # Arguments
/// * `id` — the `mpc_submissions.id` row to refresh.
///
/// # Return
/// The row's updated state.
///
/// # Errors
/// The row not existing, having no `submission_id` yet (the original POST
/// never got an ack), or the MPC request/parse itself failing — as a
/// `ServerFnError`.
#[server]
pub async fn refresh_submission_status(id: i64) -> Result<SubmissionRow, ServerFnError> {
    use crate::{get_http_client, get_pool};
    use fink_fat_ades::mpc_submission::{parse_submission_status_page, MPC_SUBMISSION_STATUS_URL};
    use fink_fat_ades::submission_status_api::{
        parse_submission_status_response, SubmissionStatusRequest, MPC_SUBMISSION_STATUS_API_URL,
    };

    #[derive(sqlx::FromRow)]
    struct SubmissionIdentity {
        lineage_designation: String,
        endpoint: String,
        submission_id: Option<String>,
    }

    let pool = get_pool().await;
    let identity: SubmissionIdentity = sqlx::query_as(
        "SELECT lineage_designation, endpoint, submission_id FROM mpc_submissions WHERE id = $1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?
    .ok_or_else(|| ServerFnError::new(format!("no mpc_submissions row with id {id}")))?;

    let Some(submission_id) = identity.submission_id else {
        return Err(ServerFnError::new(format!(
            "lineage '{}' has no MPC submission id to check (the original submission POST \
             never got an acknowledgement)",
            identity.lineage_designation
        )));
    };
    let is_production = identity.endpoint == "production";

    let client = get_http_client().await;
    let (verdict, verdict_detail): (&str, Option<serde_json::Value>) = if is_production {
        let response = client
            .get(MPC_SUBMISSION_STATUS_API_URL)
            .json(&SubmissionStatusRequest {
                submission_id: &submission_id,
            })
            .send()
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            (
                "error",
                Some(serde_json::json!({"error": "submission id not found"})),
            )
        } else {
            let body = response
                .text()
                .await
                .map_err(|e| ServerFnError::new(e.to_string()))?;
            let parsed = parse_submission_status_response(&body)
                .map_err(|e| ServerFnError::new(e.to_string()))?;
            let verdict = if parsed.accepted {
                "accepted"
            } else {
                "rejected"
            };
            (
                verdict,
                Some(serde_json::json!({
                    "pipeline_entry_time": parsed.pipeline_entry_time,
                    "fault_events": parsed.fault_events,
                })),
            )
        }
    } else {
        let body = client
            .get(MPC_SUBMISSION_STATUS_URL)
            .query(&[("id", submission_id.as_str())])
            .send()
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?
            .text()
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;
        match parse_submission_status_page(&body).map_err(|e| ServerFnError::new(e.to_string()))? {
            fink_fat_ades::mpc_submission::SubmissionStatusOutcome::Pending => ("pending", None),
            fink_fat_ades::mpc_submission::SubmissionStatusOutcome::Valid => ("accepted", None),
            fink_fat_ades::mpc_submission::SubmissionStatusOutcome::Invalid { comments } => (
                "rejected",
                Some(serde_json::json!({ "comments": comments })),
            ),
        }
    };

    let wamo_detail = if is_production {
        Some(fetch_wamo_detail(&submission_id).await?)
    } else {
        None
    };

    sqlx::query(
        "UPDATE mpc_submissions
         SET verdict = $1, verdict_checked_at = now(), verdict_detail = $2,
             wamo_detail = COALESCE($3, wamo_detail),
             wamo_checked_at = CASE WHEN $3 IS NOT NULL THEN now() ELSE wamo_checked_at END
         WHERE id = $4",
    )
    .bind(verdict)
    .bind(&verdict_detail)
    .bind(&wamo_detail)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    fetch_submission_row(pool, id)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?
        .ok_or_else(|| ServerFnError::new(format!("row {id} vanished after update")))
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use fink_fat_ades::wamo::{parse_wamo_response, WamoObservation};

    fn observation(obsid: &str, iau_desig: Option<&str>) -> WamoObservation {
        WamoObservation {
            iau_desig: iau_desig.map(str::to_string),
            input_type: "submission_block_id".to_string(),
            obs80: Some("c0635         C2017 10 10.35217 ...".to_string()),
            obsid: obsid.to_string(),
            obssubid: None,
            reference: Some("MPS   826083".to_string()),
            status: "P".to_string(),
            status_decoded: "matched".to_string(),
            submission_block_id: Some("2017-10-10T12:17:02.000_0000CfiO_01".to_string()),
            submission_id: Some("2017-10-10T12:17:02.000_0000CfiO".to_string()),
        }
    }

    #[test]
    fn select_wamo_observations_prefers_the_bare_submission_id_match() {
        let response = fink_fat_ades::wamo::WamoResponse {
            found: vec![
                [("sub-1".to_string(), vec![observation("obs-a", None)])]
                    .into_iter()
                    .collect(),
                [(
                    "sub-1_01".to_string(),
                    vec![observation("obs-b", Some("380635"))],
                )]
                .into_iter()
                .collect(),
            ],
            malformed: vec![],
            not_found: vec![],
        };

        let selected = select_wamo_observations(&response, "sub-1", "sub-1_01");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].obsid, "obs-a");
    }

    #[test]
    fn select_wamo_observations_falls_back_to_the_block_id() {
        let response = fink_fat_ades::wamo::WamoResponse {
            found: vec![[(
                "sub-1_01".to_string(),
                vec![observation("obs-b", Some("380635"))],
            )]
            .into_iter()
            .collect()],
            malformed: vec![],
            not_found: vec!["sub-1".to_string()],
        };

        let selected = select_wamo_observations(&response, "sub-1", "sub-1_01");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].obsid, "obs-b");
    }

    #[test]
    fn select_wamo_observations_empty_when_neither_identifier_matched() {
        let response = parse_wamo_response(
            r#"{"found": [], "malformed": [], "not_found": ["sub-1", "sub-1_01"]}"#,
        )
        .unwrap();
        assert!(select_wamo_observations(&response, "sub-1", "sub-1_01").is_empty());
    }
}

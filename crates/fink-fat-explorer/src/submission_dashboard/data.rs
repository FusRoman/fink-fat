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
    }

    let pool = get_pool().await;
    let rows: Vec<SubmissionRowSql> = sqlx::query_as(
        "SELECT id, lineage_designation, endpoint, submission_id, verdict, \
                submitted_at, verdict_checked_at
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
        })
        .collect())
}

/// Re-checks one submission's status against MPC (never re-submits) and
/// persists the result. `endpoint = "test"` polls
/// [`fink_fat_ades::mpc_submission`]'s test-tier status page once (the same
/// page `ades::server_fns` already polls in a loop right after submitting —
/// here it's a single, user-triggered check, not a wait-to-conclusion poll).
/// `endpoint = "production"` calls the documented
/// [`fink_fat_ades::submission_status_api`]. Both are read-only GETs against
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

    let client = get_http_client().await;
    let (verdict, verdict_detail): (&str, Option<serde_json::Value>) = if identity.endpoint
        == "production"
    {
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

    sqlx::query(
        "UPDATE mpc_submissions SET verdict = $1, verdict_checked_at = now(), verdict_detail = $2 \
         WHERE id = $3",
    )
    .bind(verdict)
    .bind(&verdict_detail)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    get_submission_history()
        .await?
        .into_iter()
        .find(|row| row.id == id)
        .ok_or_else(|| ServerFnError::new(format!("row {id} vanished after update")))
}

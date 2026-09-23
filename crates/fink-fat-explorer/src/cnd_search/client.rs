//! HTTP client for the Minor Planet Center's Check Near-Duplicates (CND) API
//! (<https://docs.minorplanetcenter.net/mpc-ops-docs/apis/cnd/>).
//!
//! Shared by both the per-lineage job ([`super::run`]) and the bulk job
//! (`crate::bulk_cnd::run`) so the batching/request/resilience logic lives in
//! exactly one place.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Deserialize;

const CND_URL: &str = "https://data.minorplanetcenter.net/api/cnd";

/// Maximum obs80 lines submitted per CND request.
///
/// The API itself accepts up to 10,000 observations per request, but the
/// companion Python analysis project (`report_mpc_cnd_check.py`) found that
/// MPC's own gateway times out (`504 Gateway Time-out`) around 60s of
/// processing: batches of 5000 reliably hit that timeout, while batches of
/// 1000 consistently complete in a couple of seconds. Both the per-lineage
/// and bulk jobs submit batches sequentially (not concurrently, unlike
/// Skybot's fan-out) — there's no evidence concurrent batches help, and it
/// avoids piling load onto a server that's already documented as fragile
/// around gateway timeouts.
pub const CND_BATCH_SIZE: usize = 1000;

const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Matches found (keyed by obs80 line) plus a `"<line>: <error>"` entry for
/// every line [`query_cnd_batch_resilient`] gave up isolating — factored out
/// purely to keep that function's boxed-future return type readable.
type CndBatchResult = (HashMap<String, Vec<CndMatch>>, Vec<String>);

/// One match the CND API found for a submitted observation: another,
/// already-published MPC observation close to it in time and sky position.
#[derive(Clone, Debug, Deserialize)]
pub struct CndMatch {
    /// The matched, already-published observation's own obs80 record.
    pub obs80: String,
    pub time_separation_s: f64,
    pub angle_separation_arcsec: f64,
}

/// The API returns `null` (not `[]`) for a submitted line with no matches —
/// confirmed against the live service, not just documented — so this has to
/// be `Option<Vec<_>>`, not `Vec<_>`: deserializing straight into
/// `HashMap<String, Vec<CndMatch>>` fails on essentially every real response,
/// since most submitted observations have no match.
#[derive(Deserialize)]
struct CndResponse {
    results: HashMap<String, Option<Vec<CndMatch>>>,
}

/// Submits one batch of obs80 lines (at most [`CND_BATCH_SIZE`], though this
/// doesn't enforce that — batching is the caller's responsibility) to the
/// CND API and returns the matches found for each line.
///
/// Prefer [`query_cnd_batch_resilient`] for actual job use: a single
/// malformed or server-crashing line anywhere in the batch fails this whole
/// call, discarding every other (perfectly fine) observation in it — a real
/// risk confirmed against the live service, where one specific packed
/// placeholder designation reliably 500s the server for any batch containing
/// it, unrelated to our own obs80 formatting being valid.
///
/// # Arguments
///
/// * `client` — shared HTTP client (`crate::get_http_client()`).
/// * `obs80_lines` — the batch to submit.
/// * `time_separation_s`, `angle_separation_arcsec` — CND's match
///   thresholds.
///
/// # Return
///
/// A map from each submitted obs80 line to its matches, empty (never
/// missing) if none were found.
///
/// # Errors
///
/// The request failing (network error, non-2xx status, unparseable JSON), as
/// a display string.
pub async fn query_cnd_batch(
    client: &reqwest::Client,
    obs80_lines: &[String],
    time_separation_s: f64,
    angle_separation_arcsec: f64,
) -> Result<HashMap<String, Vec<CndMatch>>, String> {
    let payload = serde_json::json!({
        "obs": obs80_lines,
        "time_separation_s": time_separation_s,
        "angle_separation_arcsec": angle_separation_arcsec,
    });

    let response = client
        .get(CND_URL)
        .json(&payload)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;

    let parsed: CndResponse = response.json().await.map_err(|e| e.to_string())?;
    Ok(parsed
        .results
        .into_iter()
        .map(|(line, matches)| (line, matches.unwrap_or_default()))
        .collect())
}

/// [`query_cnd_batch`], but a failing batch is bisected and retried on each
/// half instead of discarding the whole thing — recursing down to individual
/// lines if needed, so one bad observation only ever costs itself rather
/// than everything it happened to share a batch with.
///
/// This is deliberately not the default behavior baked into
/// [`query_cnd_batch`] itself: bisection only makes sense once a batch is
/// known to be failing, and callers that just want the simple whole-batch
/// call (e.g. a future health check) shouldn't pay for the recursion setup.
///
/// Bisection is sequential (one request at a time, same rationale as
/// [`CND_BATCH_SIZE`]'s doc comment), so a batch containing several
/// server-crashing lines — not just the one we originally found — can take
/// a while to fully resolve: confirmed in practice on real data, where a
/// batch needing deep bisection made the caller's progress counter (which
/// only advanced once per whole top-level batch) sit still for minutes even
/// though the job was actively working. `progress` exists specifically to
/// fix that: it advances every time a slice of lines is *finally* resolved
/// (whether by a successful request or by being skipped as an isolated bad
/// line), at every recursion depth, not just at the top, so a caller wiring
/// it to the same job-progress counter shown in the UI gets live feedback
/// throughout a slow bisection instead of an apparently frozen job.
///
/// # Arguments
///
/// * `progress` — incremented by the number of lines resolved, as they
///   resolve (not just once at the end). Typically the job's own
///   `processed` counter, so UI progress advances continuously.
/// * the rest are the same as [`query_cnd_batch`].
///
/// # Return
///
/// `(matches, skipped)` — every line's matches that could be determined, and
/// a human-readable `"<line>: <error>"` entry for each line that still
/// failed once isolated to a batch of one (logged by the caller rather than
/// silently dropped, so a systematically bad line is visible in the job's
/// log instead of just vanishing from the results).
pub fn query_cnd_batch_resilient<'a>(
    client: &'a reqwest::Client,
    obs80_lines: &'a [String],
    time_separation_s: f64,
    angle_separation_arcsec: f64,
    progress: &'a AtomicUsize,
) -> Pin<Box<dyn Future<Output = CndBatchResult> + Send + 'a>> {
    Box::pin(async move {
        match query_cnd_batch(
            client,
            obs80_lines,
            time_separation_s,
            angle_separation_arcsec,
        )
        .await
        {
            Ok(results) => {
                progress.fetch_add(obs80_lines.len(), Ordering::Relaxed);
                (results, Vec::new())
            }
            Err(message) => {
                let Some((first, rest)) = obs80_lines.split_first() else {
                    return (HashMap::new(), Vec::new());
                };
                if rest.is_empty() {
                    progress.fetch_add(1, Ordering::Relaxed);
                    return (HashMap::new(), vec![format!("{first}: {message}")]);
                }
                let mid = obs80_lines.len() / 2;
                let (mut results, mut skipped) = query_cnd_batch_resilient(
                    client,
                    &obs80_lines[..mid],
                    time_separation_s,
                    angle_separation_arcsec,
                    progress,
                )
                .await;
                let (more_results, more_skipped) = query_cnd_batch_resilient(
                    client,
                    &obs80_lines[mid..],
                    time_separation_s,
                    angle_separation_arcsec,
                    progress,
                )
                .await;
                results.extend(more_results);
                skipped.extend(more_skipped);
                (results, skipped)
            }
        }
    })
}

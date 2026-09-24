//! `fink-fat submit`: submit MPC-eligible lineages' observations to the
//! Minor Planet Center (MPC) as ADES XML files, recording the outcome
//! durably in `mpc_submissions`
//! (`src/converter/sql/create_mpc_submission_tables.sql`) so a later run —
//! or `fink-fat-explorer`'s submission dashboard — can tell what has already
//! been sent.
//!
//! Per lineage, in order:
//! 0. **Already-submitted check** — by `lineage_designation`, and by
//!    observation-id overlap against every prior submission (catches the
//!    same observations resurfacing under a renamed/merged lineage).
//! 1. **Eligibility check** — the lineage's best branch must be at
//!    [`fink_fat_ades::quality_tier::QualityTier::WellSampledDiscovery`] or
//!    [`fink_fat_ades::quality_tier::QualityTier::Discovery`] (see
//!    [`QualityTier::is_submission_eligible`](fink_fat_ades::quality_tier::QualityTier::is_submission_eligible)).
//! 2. **Build + validate the ADES document** — `--dry-run` stops here,
//!    before anything leaves the process.
//! 3. **Submit to MPC** — `--endpoint test` (the default) hits
//!    `submit_xml_test`; `--endpoint production` hits the real `submit_xml`.
//! 4. **Persist the outcome** to `mpc_submissions`, whether the submission
//!    itself succeeded or failed (a failed attempt is still recorded, so a
//!    rerun doesn't retry it silently without `--force`).
//!
//! `submit` has no [`fink_fat_engine::engine_config::EngineConfig`] in
//! scope: it never touches the engine's snapshot/archive log, only
//! Postgres (via the synchronous `postgres` crate, matching
//! `crate::converter::sql`) and MPC (via `reqwest::blocking`) — see the
//! architecture note in the project's implementation plan for why the ADES
//! data (RA/Dec/mag/band per observation) is only reachable through
//! Postgres, not the raw journal.
//!
//! Every step above is a thin wrapper around a pure function (parsing,
//! decision, formatting) covered by this module's `tests`; the only real
//! I/O is [`process_lineage`]'s Postgres client and MPC HTTP request.

pub mod error;
pub mod logging;

use std::collections::{HashMap, HashSet};

use camino::{Utf8Path, Utf8PathBuf};
use fink_fat_engine::engine_config::log_level::LogLevel;
use postgres::{Client, NoTls};
use tracing::{error, info, warn};

use fink_fat_ades::{
    model::{
        AdesHeaderInput, FETCH_BRANCH_OBSERVATIONS_QUERY, NightObservation, ObservationRow,
        RESOLVE_BEST_BRANCH_QUERY, check_submission_recommendations, normalize_trk_sub,
        remove_singleton_nights,
    },
    mpc_submission::{McpSubmissionRequest, SubmitEndpoint, parse_mpc_submission_response},
    quality_tier::{
        CROSS_MATCH_LINEAGES_QUERY, ELIGIBLE_BRANCH_QUERY, FitMethod,
        LATEST_ORBIT_FIT_FAILURE_QUERY, LATEST_ORBIT_FIT_QUERY, LatestFit, MIN_BASELINE_DAYS,
        MIN_OBSERVATIONS, QualityTier, WELL_SAMPLED_NIGHTS_QUERY, assign_quality_tier,
    },
    schema_validation::check_local_schema_violations,
    xml::{ades_document_to_xml, build_ades_document},
};

use crate::submit::error::SubmitError;

/// Everything `fink-fat submit` needs, already parsed from the CLI (see
/// `src/init_cli.rs`'s `FinkFatCommands::Submit` variant) — a dedicated
/// struct rather than a long parameter list, since this subcommand has ten
/// flags.
pub struct SubmitArgs {
    pub lineages: Option<String>,
    pub csv: Option<Utf8PathBuf>,
    pub database_url: String,
    pub submitter_config: Utf8PathBuf,
    pub endpoint: SubmitEndpoint,
    pub dry_run: bool,
    pub force: bool,
    pub logs: bool,
    pub log_file: Option<Utf8PathBuf>,
    pub log_level: LogLevel,
}

/// Submitter/telescope identity fields, loaded from a YAML file
/// (`--submitter-config`). Now defined in the shared
/// [`fink_fat_ades::submitter_config`] crate — the exact same type
/// `fink-fat-explorer`'s Submission page generates as a YAML download, so a
/// config built through that form is guaranteed to load back here without
/// surprises — and re-exported here under this module's original path.
pub use fink_fat_ades::submitter_config::{SubmitterConfig, default_ack_message};

/// Loads and validates a `--submitter-config` file. A free function rather
/// than a method on [`SubmitterConfig`]: that type is defined in
/// `fink-fat-ades` now, and file I/O is specific to this CLI (the shared
/// crate stays filesystem-free — see its crate docs), so this wrapper lives
/// here instead.
///
/// # Arguments
/// * `path` — path to the YAML file.
///
/// # Return
/// The validated config.
///
/// # Errors
/// [`SubmitError::SubmitterConfigRead`] if the file can't be read,
/// [`SubmitError::SubmitterConfigParse`] if it isn't valid YAML matching
/// this shape, [`SubmitError::SubmitterConfigInvalid`] if required fields
/// are missing — checked once up front rather than once per lineage in the
/// batch.
fn load_submitter_config(path: &Utf8Path) -> Result<SubmitterConfig, SubmitError> {
    let text =
        std::fs::read_to_string(path).map_err(|source| SubmitError::SubmitterConfigRead {
            path: path.to_path_buf(),
            source,
        })?;
    let config: SubmitterConfig =
        serde_yaml::from_str(&text).map_err(|source| SubmitError::SubmitterConfigParse {
            path: path.to_path_buf(),
            source,
        })?;

    let reasons = config.validation_errors();
    if !reasons.is_empty() {
        return Err(SubmitError::SubmitterConfigInvalid {
            path: path.to_path_buf(),
            reasons,
        });
    }
    Ok(config)
}

/// Parses a comma-separated `--lineages` argument. Trims whitespace and
/// drops empty entries; does not deduplicate (a duplicate is caught by the
/// already-submitted check on its second pass, and preserving input order
/// keeps the printed per-lineage log predictable).
///
/// # Arguments
/// * `raw` — the raw `--lineages` value.
///
/// # Return
/// The lineage designations, in order.
pub fn parse_lineage_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Parses lineage designations out of already-read CSV text: every value of
/// the `lineage_designation` header column. A minimal comma-splitting
/// parser, not a full CSV grammar — fink-fat's own CSV export (the
/// explorer's "Download CSV" button) never quotes or embeds a comma in a
/// designation, so this doesn't handle quoted fields.
///
/// # Arguments
/// * `csv_text` — the full CSV file text.
///
/// # Return
/// The lineage designations, in row order.
///
/// # Errors
/// Returns an error string if the header has no `lineage_designation`
/// column.
pub fn parse_csv_lineages(csv_text: &str) -> Result<Vec<String>, String> {
    let mut lines = csv_text.lines();
    let header = lines.next().unwrap_or_default();
    let column_index = header
        .split(',')
        .position(|c| c.trim() == "lineage_designation")
        .ok_or_else(|| "no 'lineage_designation' column in the CSV header".to_string())?;

    Ok(lines
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| line.split(',').nth(column_index))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect())
}

fn load_csv_lineages(path: &Utf8Path) -> Result<Vec<String>, SubmitError> {
    let text = std::fs::read_to_string(path).map_err(|source| SubmitError::CsvRead {
        path: path.to_path_buf(),
        source,
    })?;
    parse_csv_lineages(&text).map_err(|_| SubmitError::CsvMissingColumn {
        path: path.to_path_buf(),
    })
}

/// Resolves `--lineages`/`--csv` into the final list of lineage designations
/// to process.
///
/// # Errors
/// [`SubmitError::NoLineagesSpecified`] if neither was given (or, in
/// principle, both — `clap`'s `conflicts_with` already prevents that at
/// parse time). [`SubmitError::CsvRead`]/[`SubmitError::CsvMissingColumn`]
/// for a bad `--csv` file.
fn collect_lineage_designations(
    lineages: Option<&str>,
    csv: Option<&Utf8Path>,
) -> Result<Vec<String>, SubmitError> {
    match (lineages, csv) {
        (Some(raw), None) => Ok(parse_lineage_list(raw)),
        (None, Some(path)) => load_csv_lineages(path),
        _ => Err(SubmitError::NoLineagesSpecified),
    }
}

/// Extracts the observation ids from a set of kept observations, in track
/// order — the exact set sent to MPC and recorded in
/// `mpc_submissions.observation_ids`.
///
/// # Arguments
/// * `observations` — the observations surviving singleton-night removal.
///
/// # Return
/// The observation ids, in the same order.
pub fn observation_ids(observations: &[NightObservation]) -> Vec<i64> {
    observations.iter().map(|o| o.observation.id).collect()
}

/// The outcome of processing one lineage — every branch of `fink-fat
/// submit`'s pipeline, folded into one type so [`summarize`] and
/// [`format_outcome_line`] can be pure functions over it.
#[derive(Debug, Clone, PartialEq)]
pub enum LineageOutcome {
    /// Step 0: already submitted (by exact designation, or by observation-id
    /// overlap with a prior submission under a possibly different
    /// designation).
    AlreadySubmitted {
        previous_lineage_designation: String,
        previous_submission_id: Option<String>,
    },
    /// The lineage has no branches at all.
    LineageNotFound,
    /// Step 1: the lineage's best branch isn't at an MPC-submission-eligible
    /// quality tier.
    Ineligible { tier: QualityTier },
    /// Step 2: the ADES document failed local `submit.xsd` validation.
    ValidationFailed { violations: Vec<String> },
    /// `--dry-run` stopped the pipeline after step 2, before anything was
    /// sent to MPC.
    DryRunStopped { n_observations: usize },
    /// Step 3+4: submitted to MPC and recorded.
    Submitted {
        submission_id: String,
        mpc_submissions_row_id: i64,
    },
    /// Step 3 failed (network/timeout/non-2xx/unparseable ack), but the
    /// attempt was still recorded (step 4) with `verdict = 'error'`.
    SubmissionRequestFailed {
        reason: String,
        mpc_submissions_row_id: i64,
    },
}

/// Aggregate counts across a whole `fink-fat submit` run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubmitSummary {
    pub submitted: usize,
    pub dry_run: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Tallies a batch's per-lineage results into a [`SubmitSummary`]. Pure.
///
/// # Arguments
/// * `results` — one entry per lineage processed, in order.
///
/// # Return
/// The aggregate counts.
pub fn summarize(results: &[Result<LineageOutcome, SubmitError>]) -> SubmitSummary {
    let mut summary = SubmitSummary::default();
    for result in results {
        match result {
            Ok(LineageOutcome::Submitted { .. }) => summary.submitted += 1,
            Ok(LineageOutcome::DryRunStopped { .. }) => summary.dry_run += 1,
            Ok(LineageOutcome::AlreadySubmitted { .. }) | Ok(LineageOutcome::Ineligible { .. }) => {
                summary.skipped += 1
            }
            Ok(LineageOutcome::LineageNotFound)
            | Ok(LineageOutcome::ValidationFailed { .. })
            | Ok(LineageOutcome::SubmissionRequestFailed { .. })
            | Err(_) => summary.failed += 1,
        }
    }
    summary
}

/// Renders one lineage's outcome as a single-line, human-readable summary —
/// what `submit` prints per lineage (independent of `--logs`'s structured
/// tracing output, which the caller also emits for the same event). Pure.
///
/// # Arguments
/// * `lineage_designation` — the lineage the outcome is for.
/// * `result` — its processing result.
///
/// # Return
/// The formatted line (no trailing newline).
pub fn format_outcome_line(
    lineage_designation: &str,
    result: &Result<LineageOutcome, SubmitError>,
) -> String {
    match result {
        Ok(LineageOutcome::AlreadySubmitted {
            previous_lineage_designation,
            previous_submission_id,
        }) => {
            let id_suffix = previous_submission_id
                .as_deref()
                .map(|id| format!(", submission_id={id}"))
                .unwrap_or_default();
            format!(
                "{lineage_designation}: SKIPPED (already submitted as \
                 '{previous_lineage_designation}'{id_suffix})"
            )
        }
        Ok(LineageOutcome::LineageNotFound) => {
            format!("{lineage_designation}: FAILED (lineage has no branches)")
        }
        Ok(LineageOutcome::Ineligible { tier }) => {
            format!("{lineage_designation}: SKIPPED (not eligible for submission, tier={tier})")
        }
        Ok(LineageOutcome::ValidationFailed { violations }) => format!(
            "{lineage_designation}: FAILED (ADES schema violations: {})",
            violations.join("; ")
        ),
        Ok(LineageOutcome::DryRunStopped { n_observations }) => format!(
            "{lineage_designation}: DRY-RUN (would submit {n_observations} observation(s), \
             not sent to MPC)"
        ),
        Ok(LineageOutcome::Submitted { submission_id, .. }) => {
            format!("{lineage_designation}: SUBMITTED (submission_id={submission_id})")
        }
        Ok(LineageOutcome::SubmissionRequestFailed { reason, .. }) => {
            format!("{lineage_designation}: FAILED (MPC submission request failed: {reason})")
        }
        Err(err) => format!("{lineage_designation}: FAILED ({err})"),
    }
}

/// Everything [`assign_quality_tier`] needs about every branch, fetched once
/// per `fink-fat submit` run (not once per lineage) — mirrors
/// `fink-fat-explorer::homepage::snapshot::QualityIndex`, built from the
/// same shared queries.
struct QualityIndex {
    eligible: HashSet<i64>,
    latest_fit: HashMap<i64, LatestFit>,
    latest_failure_at: HashMap<i64, chrono::DateTime<chrono::Utc>>,
    well_sampled_nights: HashMap<i64, i64>,
    cross_match: HashSet<String>,
}

impl QualityIndex {
    fn tier_for(&self, branch_id: i64, lineage_designation: &str, n_nights: i64) -> QualityTier {
        assign_quality_tier(
            self.eligible.contains(&branch_id),
            self.latest_fit.get(&branch_id),
            self.latest_failure_at.get(&branch_id).copied(),
            n_nights,
            self.well_sampled_nights
                .get(&branch_id)
                .copied()
                .unwrap_or(0),
            self.cross_match.contains(lineage_designation),
        )
    }
}

fn build_quality_index(client: &mut Client) -> Result<QualityIndex, SubmitError> {
    let eligible: HashSet<i64> = client
        .query(
            ELIGIBLE_BRANCH_QUERY,
            &[&MIN_OBSERVATIONS, &MIN_BASELINE_DAYS],
        )?
        .iter()
        .map(|row| row.get::<_, i64>(0))
        .collect();

    let latest_fit: HashMap<i64, LatestFit> = client
        .query(LATEST_ORBIT_FIT_QUERY, &[])?
        .iter()
        .map(|row| {
            let fit_method: String = row.get("fit_method");
            (
                row.get::<_, i64>("branch_id"),
                LatestFit {
                    fit_method: FitMethod::from_column(&fit_method),
                    num_measurements: row.get("num_measurements"),
                    fitted_at: row.get("fitted_at"),
                },
            )
        })
        .collect();

    let latest_failure_at: HashMap<i64, chrono::DateTime<chrono::Utc>> = client
        .query(LATEST_ORBIT_FIT_FAILURE_QUERY, &[])?
        .iter()
        .map(|row| (row.get::<_, i64>("branch_id"), row.get("attempted_at")))
        .collect();

    let well_sampled_nights: HashMap<i64, i64> = client
        .query(WELL_SAMPLED_NIGHTS_QUERY, &[])?
        .iter()
        .map(|row| {
            (
                row.get::<_, i64>("branch_id"),
                row.get::<_, i64>("well_sampled_nights"),
            )
        })
        .collect();

    let cross_match: HashSet<String> = client
        .query(CROSS_MATCH_LINEAGES_QUERY, &[])?
        .iter()
        .map(|row| row.get::<_, String>(0))
        .collect();

    Ok(QualityIndex {
        eligible,
        latest_fit,
        latest_failure_at,
        well_sampled_nights,
        cross_match,
    })
}

fn resolve_best_branch(
    client: &mut Client,
    lineage_designation: &str,
) -> Result<Option<i64>, SubmitError> {
    Ok(client
        .query_opt(RESOLVE_BEST_BRANCH_QUERY, &[&lineage_designation])?
        .map(|row| row.get::<_, i64>(0)))
}

fn fetch_n_nights(client: &mut Client, branch_id: i64) -> Result<i64, SubmitError> {
    let row = client.query_one(
        "SELECT n_nights FROM branches WHERE branch_id = $1",
        &[&branch_id],
    )?;
    Ok(row.get(0))
}

fn fetch_branch_observations(
    client: &mut Client,
    branch_id: i64,
) -> Result<Vec<ObservationRow>, SubmitError> {
    Ok(client
        .query(FETCH_BRANCH_OBSERVATIONS_QUERY, &[&branch_id])?
        .iter()
        .map(|row| ObservationRow {
            id: row.get("id"),
            object_id: row.get("object_id"),
            position: row.get("position"),
            mjd_tt: row.get("mjd_tt"),
            ra: row.get("ra"),
            ra_err: row.get("ra_err"),
            dec: row.get("dec"),
            dec_err: row.get("dec_err"),
            magnitude: row.get("magnitude"),
            mag_err: row.get("mag_err"),
            filter: row.get("filter"),
            mpc_code_obs: row.get("mpc_code_obs"),
            night_id: row.get("night_id"),
        })
        .collect())
}

/// One prior submission that makes the current lineage/observation set
/// ineligible for step 0 (already-submitted).
struct PreviousSubmission {
    lineage_designation: String,
    submission_id: Option<String>,
}

/// Step 0: checks whether `lineage_designation` (exact match) or any of
/// `observation_ids` (array-overlap with a prior submission's own
/// `observation_ids`, regardless of that submission's lineage designation)
/// has already been submitted.
fn already_submitted(
    client: &mut Client,
    lineage_designation: &str,
    observation_ids: &[i64],
) -> Result<Option<PreviousSubmission>, SubmitError> {
    let row = client.query_opt(
        "SELECT lineage_designation, submission_id
         FROM mpc_submissions
         WHERE lineage_designation = $1 OR observation_ids && $2
         ORDER BY submitted_at DESC
         LIMIT 1",
        &[&lineage_designation, &observation_ids],
    )?;
    Ok(row.map(|row| PreviousSubmission {
        lineage_designation: row.get(0),
        submission_id: row.get(1),
    }))
}

#[allow(clippy::too_many_arguments)]
fn insert_submission_row(
    client: &mut Client,
    lineage_designation: &str,
    branch_id: i64,
    trk_sub: &str,
    endpoint: SubmitEndpoint,
    submission_id: Option<&str>,
    observation_ids: &[i64],
    xml: &str,
    header: &AdesHeaderInput,
    verdict: &str,
    verdict_detail: Option<&str>,
) -> Result<i64, SubmitError> {
    let endpoint_column = if endpoint.is_production() {
        "production"
    } else {
        "test"
    };
    // `postgres-types`' plain `ToSql for String`/`for &str` only `accepts()`
    // text-ish column types (TEXT/VARCHAR/...), not JSON/JSONB — binding a
    // pre-serialized JSON *string* against `verdict_detail JSONB` (even with
    // a `$N::jsonb` cast in the SQL) fails at serialization time with
    // "error serializing parameter N", since the driver picks the column's
    // real type for the wire format, not the cast's apparent target.
    // `postgres::types::Json<T>` (the `with-serde_json-1` feature) is the
    // correct wrapper: it `accepts()` JSON/JSONB directly and handles
    // JSONB's binary version-byte prefix itself.
    let verdict_detail_value =
        verdict_detail.map(|detail| postgres::types::Json(serde_json::json!({ "error": detail })));

    let row = client.query_one(
        "INSERT INTO mpc_submissions
            (lineage_designation, branch_id, trk_sub, endpoint, submission_id,
             observation_ids, xml, ack_message, ac2_email, verdict, verdict_detail)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
         RETURNING id",
        &[
            &lineage_designation,
            &branch_id,
            &trk_sub,
            &endpoint_column,
            &submission_id,
            &observation_ids,
            &xml,
            &header.ack_message,
            &header.ac2_email,
            &verdict,
            &verdict_detail_value,
        ],
    )?;
    Ok(row.get(0))
}

/// Step 3: POSTs the ADES XML to MPC and parses the acknowledgement.
fn submit_to_mpc(
    http_client: &reqwest::blocking::Client,
    xml: &str,
    header: &AdesHeaderInput,
    endpoint: SubmitEndpoint,
) -> Result<String, SubmitError> {
    let request = McpSubmissionRequest {
        xml: xml.to_string(),
        ack_message: header.ack_message.clone(),
        ac2_email: header.ac2_email.clone(),
    };
    let body = http_client
        .post(endpoint.submit_url())
        .multipart(
            reqwest::blocking::multipart::Form::new()
                .text("source", request.source_field_value())
                .text("ack", request.ack_message.clone())
                .text("ac2", request.ac2_email.clone()),
        )
        .send()?
        .error_for_status()?
        .text()?;
    Ok(parse_mpc_submission_response(&body)?)
}

/// Runs the full per-lineage pipeline (steps 0-4, see the module docs). Never
/// aborts the whole batch on a per-lineage problem — every expected failure
/// mode (already submitted, ineligible, schema violation, MPC request
/// failure) is folded into an `Ok(LineageOutcome)`; only an unexpected I/O
/// failure (Postgres unreachable, etc.) surfaces as `Err`.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    skip(client, http_client, quality_index, submitter_config),
    fields(lineage = %lineage_designation)
)]
fn process_lineage(
    client: &mut Client,
    http_client: &reqwest::blocking::Client,
    quality_index: &QualityIndex,
    lineage_designation: &str,
    submitter_config: &SubmitterConfig,
    endpoint: SubmitEndpoint,
    dry_run: bool,
    force: bool,
) -> Result<LineageOutcome, SubmitError> {
    let Some(branch_id) = resolve_best_branch(client, lineage_designation)? else {
        warn!("lineage has no branches, skipping");
        return Ok(LineageOutcome::LineageNotFound);
    };

    let observations = fetch_branch_observations(client, branch_id)?;
    let night_observations: Vec<NightObservation> = observations
        .into_iter()
        .map(|observation| NightObservation {
            night_id: observation.night_id,
            observation,
        })
        .collect();
    let (kept, singleton_summary) = remove_singleton_nights(&night_observations);
    let obs_ids = observation_ids(&kept);

    if !force && let Some(previous) = already_submitted(client, lineage_designation, &obs_ids)? {
        info!(
            previous_lineage = %previous.lineage_designation,
            previous_submission_id = ?previous.submission_id,
            "already submitted, skipping"
        );
        return Ok(LineageOutcome::AlreadySubmitted {
            previous_lineage_designation: previous.lineage_designation,
            previous_submission_id: previous.submission_id,
        });
    }

    let n_nights = fetch_n_nights(client, branch_id)?;
    let tier = quality_index.tier_for(branch_id, lineage_designation, n_nights);
    if !force && !tier.is_submission_eligible() {
        info!(?tier, "not eligible for submission, skipping");
        return Ok(LineageOutcome::Ineligible { tier });
    }

    let today = fink_fat_ades::format_epoch::today_utc_date().unwrap_or_default();
    let header = submitter_config
        .clone()
        .into_header(lineage_designation, &today);

    let violations = check_local_schema_violations(lineage_designation, &kept, &header);
    if !violations.is_empty() {
        warn!(?violations, "ADES schema validation failed");
        return Ok(LineageOutcome::ValidationFailed { violations });
    }

    let doc = build_ades_document(lineage_designation, &kept, &header)?;
    let xml = ades_document_to_xml(&doc)?;
    let advisory = check_submission_recommendations(&kept);
    for warning in &advisory.warnings {
        warn!(%warning, "submission-quality advisory");
    }
    info!(
        n_observations = kept.len(),
        singleton_nights_removed = singleton_summary.singleton_night_count,
        "ADES document built and locally validated"
    );

    if dry_run {
        info!("dry-run: stopping before MPC submission");
        return Ok(LineageOutcome::DryRunStopped {
            n_observations: kept.len(),
        });
    }

    let trk_sub = normalize_trk_sub(lineage_designation)?.as_str().to_string();

    match submit_to_mpc(http_client, &xml, &header, endpoint) {
        Ok(submission_id) => {
            let row_id = insert_submission_row(
                client,
                lineage_designation,
                branch_id,
                &trk_sub,
                endpoint,
                Some(&submission_id),
                &obs_ids,
                &xml,
                &header,
                "pending",
                None,
            )?;
            info!(submission_id = %submission_id, ?endpoint, "submitted to MPC");
            Ok(LineageOutcome::Submitted {
                submission_id,
                mpc_submissions_row_id: row_id,
            })
        }
        Err(err) => {
            let reason = err.to_string();
            let row_id = insert_submission_row(
                client,
                lineage_designation,
                branch_id,
                &trk_sub,
                endpoint,
                None,
                &obs_ids,
                &xml,
                &header,
                "error",
                Some(&reason),
            )?;
            error!(error = %reason, "MPC submission request failed");
            Ok(LineageOutcome::SubmissionRequestFailed {
                reason,
                mpc_submissions_row_id: row_id,
            })
        }
    }
}

/// Runs `fink-fat submit`: see the module docs for the full pipeline.
///
/// # Arguments
/// * `args` — the parsed CLI arguments.
///
/// # Errors
/// Any failure that aborts the whole run rather than just one lineage:
/// loading `--submitter-config`, resolving `--lineages`/`--csv`, or
/// connecting to Postgres.
pub fn submit(args: SubmitArgs) -> Result<(), Box<dyn std::error::Error>> {
    let _log_guard =
        logging::init_submit_logging(args.logs, args.log_file.as_deref(), args.log_level)?;

    let lineage_designations =
        collect_lineage_designations(args.lineages.as_deref(), args.csv.as_deref())?;
    if lineage_designations.is_empty() {
        println!("No lineage designations given.");
        return Ok(());
    }

    let submitter_config = load_submitter_config(&args.submitter_config)?;

    println!("Connecting to Postgres...");
    let mut client = Client::connect(&args.database_url, NoTls).map_err(SubmitError::Database)?;
    client.batch_execute(include_str!(
        "../converter/sql/create_mpc_submission_tables.sql"
    ))?;

    println!("Building eligibility index...");
    let quality_index = build_quality_index(&mut client)?;
    let http_client = reqwest::blocking::Client::new();

    println!(
        "Processing {} lineage(s) ({}{})...",
        lineage_designations.len(),
        if args.endpoint.is_production() {
            "PRODUCTION"
        } else {
            "test"
        },
        if args.dry_run { ", dry-run" } else { "" }
    );

    let mut results = Vec::with_capacity(lineage_designations.len());
    for lineage_designation in &lineage_designations {
        let result = process_lineage(
            &mut client,
            &http_client,
            &quality_index,
            lineage_designation,
            &submitter_config,
            args.endpoint,
            args.dry_run,
            args.force,
        );
        println!("{}", format_outcome_line(lineage_designation, &result));
        results.push(result);
    }

    let summary = summarize(&results);
    println!(
        "submit run complete: {} submitted, {} dry-run, {} skipped, {} failed (of {} total)",
        summary.submitted,
        summary.dry_run,
        summary.skipped,
        summary.failed,
        lineage_designations.len()
    );
    info!(
        submitted = summary.submitted,
        dry_run = summary.dry_run,
        skipped = summary.skipped,
        failed = summary.failed,
        total = lineage_designations.len(),
        "submit run complete"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lineage_list_trims_and_drops_empty_entries() {
        assert_eq!(
            parse_lineage_list(" FF2026abc , FF2026def,, FF2026ghi "),
            vec!["FF2026abc", "FF2026def", "FF2026ghi"]
        );
    }

    #[test]
    fn parse_lineage_list_of_a_single_designation() {
        assert_eq!(parse_lineage_list("FF2026abc"), vec!["FF2026abc"]);
    }

    #[test]
    fn parse_csv_lineages_extracts_the_named_column_regardless_of_position() {
        let csv = "n_nights,lineage_designation,quality\n5,FF2026abc,Discovery\n8,FF2026def,WellSampledDiscovery\n";
        assert_eq!(
            parse_csv_lineages(csv).unwrap(),
            vec!["FF2026abc".to_string(), "FF2026def".to_string()]
        );
    }

    #[test]
    fn parse_csv_lineages_skips_blank_lines() {
        let csv = "lineage_designation\nFF2026abc\n\nFF2026def\n";
        assert_eq!(
            parse_csv_lineages(csv).unwrap(),
            vec!["FF2026abc".to_string(), "FF2026def".to_string()]
        );
    }

    #[test]
    fn parse_csv_lineages_rejects_a_missing_column() {
        let csv = "designation\nFF2026abc\n";
        assert!(parse_csv_lineages(csv).is_err());
    }

    // `default_ack_message`/`SubmitterConfig` themselves are now defined and
    // tested in `fink_fat_ades::submitter_config` — see that crate's tests
    // for `validation_errors`/`into_header`/`to_yaml` coverage. This module
    // only adds `load_submitter_config`'s own file-I/O wrapping, tested
    // below via `collect_lineage_designations`-style pure-function coverage
    // where possible; the read/parse/validate glue itself needs a real file
    // and isn't unit tested here (matches other file-loading code in this
    // codebase, e.g. `EngineConfig::load_engine_config_validated`).

    fn night_obs(id: i64, night_id: i64) -> NightObservation {
        NightObservation {
            night_id,
            observation: ObservationRow {
                id,
                object_id: format!("obj{id}"),
                position: 0,
                mjd_tt: 60000.0,
                ra: 1.0,
                ra_err: 0.0,
                dec: 0.5,
                dec_err: 0.0,
                magnitude: 19.0,
                mag_err: 0.1,
                filter: 2,
                mpc_code_obs: "I41".to_string(),
                night_id,
            },
        }
    }

    #[test]
    fn observation_ids_preserves_order() {
        let observations = vec![night_obs(3, 1), night_obs(1, 1), night_obs(2, 2)];
        assert_eq!(observation_ids(&observations), vec![3, 1, 2]);
    }

    #[test]
    fn summarize_counts_every_outcome_kind() {
        let results: Vec<Result<LineageOutcome, SubmitError>> = vec![
            Ok(LineageOutcome::Submitted {
                submission_id: "id1".to_string(),
                mpc_submissions_row_id: 1,
            }),
            Ok(LineageOutcome::DryRunStopped { n_observations: 5 }),
            Ok(LineageOutcome::AlreadySubmitted {
                previous_lineage_designation: "FF2026abc".to_string(),
                previous_submission_id: Some("id0".to_string()),
            }),
            Ok(LineageOutcome::Ineligible {
                tier: QualityTier::Unconstrained,
            }),
            Ok(LineageOutcome::LineageNotFound),
            Ok(LineageOutcome::ValidationFailed {
                violations: vec!["bad".to_string()],
            }),
            Ok(LineageOutcome::SubmissionRequestFailed {
                reason: "timeout".to_string(),
                mpc_submissions_row_id: 2,
            }),
        ];
        let summary = summarize(&results);
        assert_eq!(
            summary,
            SubmitSummary {
                submitted: 1,
                dry_run: 1,
                skipped: 2,
                failed: 3,
            }
        );
    }

    #[test]
    fn format_outcome_line_reports_submission_id_when_already_submitted() {
        let outcome = Ok(LineageOutcome::AlreadySubmitted {
            previous_lineage_designation: "FF2026abc".to_string(),
            previous_submission_id: Some("2026-01-01T00:00:00_0000abcd".to_string()),
        });
        let line = format_outcome_line("FF2026abc", &outcome);
        assert!(line.contains("SKIPPED"));
        assert!(line.contains("2026-01-01T00:00:00_0000abcd"));
    }

    #[test]
    fn format_outcome_line_reports_dry_run() {
        let outcome = Ok(LineageOutcome::DryRunStopped { n_observations: 7 });
        let line = format_outcome_line("FF2026abc", &outcome);
        assert!(line.contains("DRY-RUN"));
        assert!(line.contains('7'));
    }

    #[test]
    fn collect_lineage_designations_requires_one_source() {
        assert!(matches!(
            collect_lineage_designations(None, None),
            Err(SubmitError::NoLineagesSpecified)
        ));
    }

    #[test]
    fn collect_lineage_designations_uses_the_lineages_flag() {
        assert_eq!(
            collect_lineage_designations(Some("FF2026abc,FF2026def"), None).unwrap(),
            vec!["FF2026abc".to_string(), "FF2026def".to_string()]
        );
    }
}

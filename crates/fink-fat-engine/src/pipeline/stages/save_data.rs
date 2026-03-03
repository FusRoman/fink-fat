use std::time::{SystemTime, UNIX_EPOCH};

use camino::Utf8PathBuf;

use crate::{
    Alert,
    alerts::AlertSlice,
    engine_config::{EngineConfig, pipeline_policy::PersistPolicy},
    error::EngineError,
    graph::edge::Edge,
    night_id::NightId,
    persistence::{
        PersistenceManager,
        compression::Compression,
        edge_journal::edge_op::EdgeOp,
        error::PersistenceIoError,
        layout::PersistenceLayout,
        manifest::{Manifest, NightManifestEntry},
    },
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
    seeding::{SeedNode, SeedNodeSlice},
};

/// Return the current Unix timestamp (seconds).
fn now_unix_s() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Data for one night that needs to be written to disk.
struct NightSaveTask {
    nid: NightId,
    alerts: Vec<Alert>,
    seeds: Vec<SeedNode>,
}

/// Result of saving one night's files.
type NightSaveOutcome = Result<(NightId, Utf8PathBuf, Utf8PathBuf, u64, u64), PersistenceIoError>;

/// Collected outcomes from all per-night parallel writes.
type NightSaveOutcomes = Vec<NightSaveOutcome>;

/// Per-night file metadata after successful saves (night id, alert path, seed path, counts).
type NightEntries = Vec<(NightId, Utf8PathBuf, Utf8PathBuf, u64, u64)>;

/// Save alerts + seeds for a single night, returning the absolute paths.
///
/// This free function is designed to be called from a scoped thread: it
/// accepts owned data (no borrows into shared state) and returns owned paths.
fn save_night_files(
    task: NightSaveTask,
    layout: &PersistenceLayout,
    manifest_snap: &Manifest,
    compression: Compression,
) -> NightSaveOutcome {
    let alert_path =
        task.alerts
            .as_slice()
            .save_alerts_night(layout, manifest_snap, task.nid, compression)?;
    let seed_path =
        task.seeds
            .as_slice()
            .save_seeds_night(layout, manifest_snap, task.nid, compression)?;
    Ok((
        task.nid,
        alert_path,
        seed_path,
        task.alerts.len() as u64,
        task.seeds.len() as u64,
    ))
}

/// Convert a thread-panic payload into a well-formed [`EngineError::StageFailed`].
///
/// When a scoped thread panics, [`std::thread::ScopedJoinHandle::join`] returns
/// `Err(Box<dyn Any + Send>)`.  This helper extracts the human-readable panic
/// message (if the payload is a `&str` or `String`) and wraps it so the caller
/// can propagate it naturally with `?` instead of re-panicking with `.expect()`.
fn panic_to_engine_error(payload: Box<dyn std::any::Any + Send>, context: &str) -> EngineError {
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(panic with opaque payload)".to_string());
    EngineError::StageFailed {
        stage: PipelineStage::SavePersistedData,
        message: format!("{context}: {msg}"),
    }
}

/// Write alert and seed files for all nights in parallel using scoped threads.
///
/// Each night gets its own OS thread; all threads are joined before returning.
/// Per-night thread panics are captured and converted to [`EngineError`] so
/// the caller can propagate them with `?`.
fn write_nights_parallel(
    tasks: Vec<NightSaveTask>,
    layout: &PersistenceLayout,
    manifest_snap: &Manifest,
    compression: Compression,
) -> Result<NightSaveOutcomes, EngineError> {
    std::thread::scope(|s| {
        // Capture the night id before moving the task so it can appear in
        // any error message produced by panic_to_engine_error.
        let handles: Vec<_> = tasks
            .into_iter()
            .map(|task| {
                let nid = task.nid;
                (
                    nid,
                    s.spawn(|| save_night_files(task, layout, manifest_snap, compression)),
                )
            })
            .collect();

        // Join ALL handles first so the scope never exits with live threads,
        // then convert panics and collect results.
        let joined: Vec<_> = handles
            .into_iter()
            .map(|(nid, h)| (nid, h.join()))
            .collect();

        joined
            .into_iter()
            .map(|(nid, r)| {
                r.map_err(|p| {
                    panic_to_engine_error(
                        p,
                        &format!("alert/seed write thread for night {nid:?} panicked"),
                    )
                })
            })
            .collect::<Result<NightSaveOutcomes, EngineError>>()
    })
}

/// Inputs for [`write_edge_journal`], bundled to keep argument count low.
struct EdgeJournalInput {
    /// Manifest clone owned by the edge-write thread; only `edge_journal` is
    /// mutated, leaving `nights` untouched for the post-scope merge.
    manifest: Manifest,
    current_night: NightId,
    created_unix_s: i64,
    /// When `true`, a full snapshot compaction is performed instead of an
    /// incremental delta append.
    should_compact: bool,
    edge_ops: Vec<EdgeOp>,
    /// Pre-cloned edge set, present only when `should_compact` is `true`.
    edges_for_compact: Option<Vec<Edge>>,
}

/// Write the edge journal for the current night — either a full snapshot
/// compaction or an incremental delta — and return the updated manifest.
///
/// The manifest inside `input` is an owned clone; this function mutates only
/// its `edge_journal` field so the result can be merged with the night-write
/// manifest without conflict.
fn write_edge_journal(
    persistence: &PersistenceManager,
    engine_config: &EngineConfig,
    input: EdgeJournalInput,
) -> Result<(Manifest, bool), EngineError> {
    let EdgeJournalInput {
        mut manifest,
        current_night,
        created_unix_s,
        should_compact,
        edge_ops,
        edges_for_compact,
    } = input;

    if should_compact {
        persistence.compact_edges(
            &mut manifest,
            engine_config,
            current_night,
            created_unix_s,
            edges_for_compact.unwrap(),
        )?;
        Ok((manifest, true))
    } else {
        persistence.write_edge_delta(
            &mut manifest,
            engine_config,
            current_night,
            created_unix_s,
            edge_ops,
        )?;
        Ok((manifest, false))
    }
}

/// Upsert per-night file paths into `manifest.nights` and return the total
/// number of alert records that were saved.
///
/// Paths stored in the manifest are relative to the persistence root so that
/// the store remains portable; absolute paths are converted via `layout`.
fn apply_night_entries(
    manifest: &mut Manifest,
    entries: &NightEntries,
    layout: &PersistenceLayout,
) -> u64 {
    let mut alerts_saved: u64 = 0;
    for (nid, alert_abs, seed_abs, n_alerts, n_seeds) in entries {
        let rel_alert = layout
            .to_relative(alert_abs)
            .unwrap_or_else(|| alert_abs.clone());
        let rel_seed = layout
            .to_relative(seed_abs)
            .unwrap_or_else(|| seed_abs.clone());
        manifest.upsert_night(NightManifestEntry::new(
            *nid,
            rel_alert,
            rel_seed,
            Some(*n_alerts),
            Some(*n_seeds),
        ));
        alerts_saved += n_alerts;
    }
    alerts_saved
}

pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::SavePersistedData,
        hooks,
        StageMeta {
            label: PipelineStage::SavePersistedData.label().to_string(),
            total: Some(3), // 1 for alerts + 1 for edges + 1 for orbits (if full) + 1 for manifest
        },
        |stage_sink| {
            let created_unix_s = now_unix_s();
            let persist_policy = ctx.plan.persist;

            let current_night = match ctx.runtime_state.alert_store.last_night() {
                Some(n) => n,
                None => {
                    stage_sink.inc(3); // Increment all remaining steps since there's no data to save.
                    return Ok(vec![("skipped", 1)]);
                }
            };

            // ----------------------------------------------------------------
            // Prepare inputs for concurrent I/O.
            //
            // All owned data is extracted from `ctx` here so that the borrows
            // on `ctx.runtime_state` are released before entering the scoped
            // threads, letting the post-scope code use `ctx` freely.
            // ----------------------------------------------------------------
            let manifest = ctx.runtime_state.manifest.clone();

            let edge_ops = ctx.runtime_state.graph.drain_pending_ops();
            let n_edge_ops = edge_ops.len() as u64;

            let should_compact = manifest.edge_journal.deltas.len().saturating_add(1)
                >= ctx.engine_config.compact_graph_every_delta;
            let edges_for_compact = should_compact.then(|| ctx.runtime_state.graph.edges.clone());

            let nights_sorted = ctx.runtime_state.alert_store.nights_sorted();
            let night_tasks: Vec<NightSaveTask> = nights_sorted
                .iter()
                .filter_map(|&nid| {
                    let alerts = ctx.runtime_state.alert_store.get(&nid)?.to_vec();
                    let seeds = ctx.runtime_state.seed_store.get(&nid)?.to_vec();
                    Some(NightSaveTask { nid, alerts, seeds })
                })
                .collect();

            let persistence: &PersistenceManager = ctx.persistence;
            let engine_config = ctx.engine_config;
            let layout = persistence.layout();
            let compression = engine_config.binary_compression;

            // Snapshot of the manifest used by night-write threads for envelope
            // metadata; `created_unix_s` is set so all envelopes share the same
            // timestamp for this run.
            let mut manifest_snap = manifest.clone();
            manifest_snap.created_unix_s = created_unix_s;

            // Independent clone for the edge journal write — it only mutates
            // `edge_journal`; the merge step below reconciles `nights` and
            // `edge_journal` into a single manifest before writing to disk.
            let manifest_for_edges = manifest.clone();

            // ----------------------------------------------------------------
            // Concurrent I/O: night file writes and edge journal.
            //
            // `write_nights_parallel` (Phase B) and `write_edge_journal`
            // (Phase C) target entirely separate directories and mutate
            // disjoint manifest fields, so they can run simultaneously.
            // Both handles are joined unconditionally before inspecting errors,
            // ensuring the scope never exits with live threads.
            // ----------------------------------------------------------------
            let (night_save_outcomes, (manifest_after_edges, compacted)) = std::thread::scope(
                |s| -> Result<(NightSaveOutcomes, (Manifest, bool)), EngineError> {
                    let nights_handle = s.spawn(|| {
                        write_nights_parallel(night_tasks, layout, &manifest_snap, compression)
                    });
                    let edges_handle = s.spawn(move || {
                        write_edge_journal(
                            persistence,
                            engine_config,
                            EdgeJournalInput {
                                manifest: manifest_for_edges,
                                current_night,
                                created_unix_s,
                                should_compact,
                                edge_ops,
                                edges_for_compact,
                            },
                        )
                    });

                    let nights_joined = nights_handle.join();
                    let edges_joined = edges_handle.join();

                    let night_outcomes = nights_joined
                        .map_err(|p| panic_to_engine_error(p, "night file writes thread panicked"))
                        .and_then(|r| r)?;
                    let edge_res = edges_joined
                        .map_err(|p| panic_to_engine_error(p, "edge journal write thread panicked"))
                        .and_then(|r| r)?;

                    Ok((night_outcomes, edge_res))
                },
            )?;

            stage_sink.inc(1); // Alert/Seed files + edge journal are done at this point.

            // ----------------------------------------------------------------
            // Collect results and merge the two manifest updates.
            //
            //   manifest.nights       ← per-night file paths from writes above
            //   manifest.edge_journal ← delta/snapshot ref from edge journal
            // ----------------------------------------------------------------
            let night_entries: NightEntries = night_save_outcomes
                .into_iter()
                .collect::<Result<_, PersistenceIoError>>()?;

            let mut manifest = manifest;
            let alerts_saved = apply_night_entries(&mut manifest, &night_entries, layout);
            manifest.edge_journal = manifest_after_edges.edge_journal;
            manifest.created_unix_s = created_unix_s;

            // ----------------------------------------------------------------
            // Export orbit Parquet files (Full persistence policy only).
            // ----------------------------------------------------------------
            let mut orbits_exported: u64 = 0;
            if matches!(persist_policy, PersistPolicy::Full) {
                ctx.runtime_state
                    .export_orbit_parquets(ctx.persistence.layout(), current_night)?;
                orbits_exported = ctx.runtime_state.orbit_results.len() as u64;
            }
            stage_sink.inc(1);

            // ----------------------------------------------------------------
            // Write the single authoritative manifest to disk.
            //
            // This is the only manifest write in the entire stage; neither
            // `write_edge_journal` nor `write_nights_parallel` writes it.
            // ----------------------------------------------------------------
            ctx.persistence.save_manifest(&manifest)?;
            ctx.runtime_state.manifest = manifest;
            stage_sink.inc(1);

            Ok(vec![
                ("current_night", current_night.0 as u64),
                ("nights_saved", nights_sorted.len() as u64),
                ("alerts_saved", alerts_saved),
                ("edge_ops_written", n_edge_ops),
                ("edge_compacted", compacted as u64),
                ("orbits_exported", orbits_exported),
            ])
        },
    )
}

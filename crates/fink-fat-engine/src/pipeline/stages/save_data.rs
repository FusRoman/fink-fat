use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    engine_config::pipeline_policy::PersistPolicy,
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
};

/// Return the current Unix timestamp (seconds).
fn now_unix_s() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
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
            total: Some(4), // 1 for alerts + 1 for edges + 1 for orbits (if full) + 1 for manifest
        },
        |stage_sink| {
            let created_unix_s = now_unix_s();
            let persist_policy = ctx.plan.persist;

            // Determine the current night from the alert store.
            let current_night = match ctx.runtime_state.alert_store.last_night() {
                Some(n) => n,
                None => {
                    // Nothing ingested — nothing to persist.
                    stage_sink.inc(5);
                    return Ok(vec![("skipped", 1)]);
                }
            };

            let mut manifest = ctx.runtime_state.manifest.clone();

            // -----------------------------------------------------------------
            // 1. Save alerts for each night that is present in the store.
            // -----------------------------------------------------------------
            let mut alerts_saved: u64 = 0;
            let nights_sorted = ctx.runtime_state.alert_store.nights_sorted();
            for &nid in &nights_sorted {
                if let Some(alerts) = ctx.runtime_state.alert_store.get(&nid)
                    && let Some(seeds) = ctx.runtime_state.seed_store.get(&nid)
                {
                    ctx.persistence.save_night_manifest(
                        &mut manifest,
                        nid,
                        created_unix_s,
                        alerts,
                        seeds,
                        ctx.engine_config.binary_compression,
                    )?;
                    alerts_saved += alerts.len() as u64;
                }
            }
            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 2 + 3. Drain pending edge operations and decide: write a delta
            //         or compact the entire history into a new snapshot.
            //
            //    The DAG accumulates EdgeOps in an internal buffer whenever
            //    edges are added, removed, or mutated.  We drain that buffer
            //    here so the decision can be made before touching disk.
            //
            //    Compaction path (preferred when the delta count is high):
            //      - Pass the current in-memory edge set directly to
            //        `compact_edges_and_cleanup`.  No delta file is written;
            //        the snapshot replaces the entire history in one write.
            //      - Avoids the costly reload of every previous delta that the
            //        old design performed inside `compact_to_snapshot`.
            //
            //    Delta path:
            //      - Write an incremental delta for this run's operations only
            //        via `commit_night` (which also handles file cleanup).
            // -----------------------------------------------------------------
            let edge_ops = ctx.runtime_state.graph.drain_pending_ops();
            let n_edge_ops = edge_ops.len() as u64;

            // Check *before* writing: would this run's delta push us over the
            // compaction threshold?
            let n_deltas = manifest.edge_journal.deltas.len();
            let should_compact =
                n_deltas.saturating_add(1) >= ctx.engine_config.compact_graph_every_delta;

            let compacted = if should_compact {
                // Pass the in-memory edge set directly — no disk reload needed.
                let edges = ctx.runtime_state.graph.edges.clone();
                ctx.persistence.compact_edges_and_cleanup(
                    &mut manifest,
                    ctx.engine_config,
                    current_night,
                    created_unix_s,
                    edges,
                )?;
                true
            } else {
                // Write an incremental delta for this run's edge operations.
                manifest = ctx.persistence.commit_night(
                    manifest.clone(),
                    ctx.engine_config,
                    current_night,
                    created_unix_s,
                    edge_ops,
                )?;
                false
            };
            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 4. Export orbit Parquet files (track members + orbital params).
            //
            //    Only performed under PersistPolicy::Full.
            // -----------------------------------------------------------------
            let mut orbits_exported: u64 = 0;
            if matches!(persist_policy, PersistPolicy::Full) {
                ctx.runtime_state
                    .export_orbit_parquets(ctx.persistence.layout(), current_night)?;
                orbits_exported = ctx.runtime_state.orbit_results.len() as u64;
            }
            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 5. Persist updated manifest to disk.
            // -----------------------------------------------------------------
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

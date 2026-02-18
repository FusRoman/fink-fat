use crate::{
    error::EngineError,
    night_id::PairingMode,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        progress_sink::ProgressSink,
        stages::{PipelineStage, run_stage},
    },
    spacetime_bucket::healpix_binner::HealpixBinner,
};

/// BuildEdges stage: construct inter-night edges from seeds, anchored to the latest night.
///
/// Semantics
/// ---------
/// - Determine `right_night` as the **latest night present** in the borrowed `SeedStore`
///   within the runtime `NightWindow`.
/// - Enumerate all `(left_night, right_night)` pairs such that:
///   - `left_night < right_night`
///   - `right_night - left_night <= max_gap_nights`
/// - For each pair, build candidate edges between `left` seeds and `right` seeds
///   using [`RuntimeGraph::add_inter_night_edges`].
///
/// Notes
/// -----
/// - This stage assumes that per-night seed vectors are already sorted by `SeedNode` order
///   (primary key `plane.epoch_mid`), so `add_inter_night_edges` must **not** re-sort `right_nodes`.
/// - The stage writes the resulting owned graph to `ctx.runtime_state.graph`.
pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
    stage_sink: &dyn ProgressSink,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::BuildEdges,
        hooks,
        StageMeta {
            label: PipelineStage::BuildEdges.label().to_string(),
            total: None,
        },
        stage_sink,
        |stage_sink| {
            // -----------------------------------------------------------------
            // 0) Preconditions
            // -----------------------------------------------------------------
            let window: PairingMode =
                ctx.runtime_state
                    .window
                    .ok_or_else(|| EngineError::StageFailed {
                        stage: PipelineStage::BuildEdges,
                        message: "missing RuntimeState.window (PairingMode)".to_string(),
                    })?;

            let edge_config = &ctx.engine_config.edges;

            // -----------------------------------------------------------------
            // 3) Early exit: gap=0 => no possible (left<right) pair
            // -----------------------------------------------------------------
            let max_gap = ctx.engine_config.max_gap_nights();
            if max_gap == 0 {
                return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
            }

            // -----------------------------------------------------------------
            // 4) Determine (left -> latest-right) pairs
            // -----------------------------------------------------------------
            let mut pairs = ctx
                .runtime_state
                .seed_store
                .night_pairs_iter(window)
                .peekable();

            // If there is no eligible pair, nothing to do.
            let Some((_, right_night)) = pairs.peek().copied() else {
                return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
            };

            // `right_nodes` must exist and be non-empty, otherwise nothing to connect.
            let right_nodes = match ctx.runtime_state.seed_store.get(&right_night) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
                }
            };

            // -----------------------------------------------------------------
            // 5) Build spatial/time binners used by the edge builder
            // -----------------------------------------------------------------
            let spatial_binner = HealpixBinner::new(ctx.engine_config.healpix_depth);
            let time_binner_width = ctx.engine_config.time_binner_width;

            // ML pool is required only in ML mode (emit_all_edges=false)
            let model_pool = if edge_config.emit_all_edges {
                None
            } else {
                Some(ctx.edge_models)
            };

            // -----------------------------------------------------------------
            // 6) Build edges for each left night -> right night
            // -----------------------------------------------------------------
            let edges_before = ctx.runtime_state.graph.edges.len() as u64;
            let mut pairs_processed: u64 = 0;

            for (left_night, _) in pairs {
                let Some(left_vec) = ctx.runtime_state.seed_store.get(&left_night) else {
                    // Defensive: iterator is derived from keys, so this should not happen
                    continue;
                };
                if left_vec.is_empty() {
                    continue;
                }

                ctx.runtime_state
                    .graph
                    .add_inter_night_edges(
                        left_vec,
                        right_nodes,
                        edge_config,
                        &spatial_binner,
                        time_binner_width,
                        model_pool,
                        stage_sink,
                    )
                    .map_err(|e| EngineError::StageFailed {
                        stage: PipelineStage::BuildEdges,
                        message: format!(
                            "add_inter_night_edges failed for ({left_night},{right_night}): {e:?}"
                        ),
                    })?;

                pairs_processed += 1;
            }

            let edges_added =
                (ctx.runtime_state.graph.edges.len() as u64).saturating_sub(edges_before);

            Ok(vec![
                ("pairs_processed", pairs_processed),
                ("edges_added", edges_added),
            ])
        },
    )
}

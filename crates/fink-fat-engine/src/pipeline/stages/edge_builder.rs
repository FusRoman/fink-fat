use crate::{
    error::EngineError,
    graph::RuntimeGraph,
    night_id::NightWindow,
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
            let window: NightWindow =
                ctx.runtime_state
                    .window
                    .ok_or_else(|| EngineError::StageFailed {
                        stage: PipelineStage::BuildEdges,
                        message: "missing RuntimeState.window (NightWindow)".to_string(),
                    })?;

            let edge_config = &ctx.engine_config.edges;

            // -----------------------------------------------------------------
            // 1) Build borrowed seeds for this window
            // -----------------------------------------------------------------
            // IMPORTANT: `seed_store` must stay alive until `graph_rt.to_owned()`,
            // because `RuntimeGraph` edges borrow `SeedNode` references.
            let seed_store = ctx
                .runtime_state
                .seed_store
                .to_borrowed_window(&ctx.runtime_state.alert_store, window)
                .map_err(|e| EngineError::StageFailed {
                    stage: PipelineStage::BuildEdges,
                    message: format!("seed_store.to_borrowed_window failed: {e:?}"),
                })?;

            // -----------------------------------------------------------------
            // 2) Prepare runtime graph (borrowed)
            // -----------------------------------------------------------------
            let mut graph_rt: RuntimeGraph<'_, '_> = RuntimeGraph::new();

            // -----------------------------------------------------------------
            // 3) Early exit: gap=0 => no possible (left<right) pair
            // -----------------------------------------------------------------
            let max_gap = ctx.engine_config.max_gap_nights();
            if max_gap == 0 {
                ctx.runtime_state.graph = graph_rt.to_owned();
                return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
            }

            // -----------------------------------------------------------------
            // 4) Determine (left -> latest-right) pairs
            // -----------------------------------------------------------------
            let mut pairs = seed_store
                .night_pairs_to_latest_in_window_iter(window, max_gap)
                .peekable();

            // If there is no eligible pair, nothing to do.
            let Some((_, right_night)) = pairs.peek().copied() else {
                ctx.runtime_state.graph = graph_rt.to_owned();
                return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
            };

            // `right_nodes` must exist and be non-empty, otherwise nothing to connect.
            let right_nodes = match seed_store.get(&right_night) {
                Some(v) if !v.is_empty() => v.as_slice(),
                _ => {
                    ctx.runtime_state.graph = graph_rt.to_owned();
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
            let edges_before = graph_rt.edges.len() as u64;
            let mut pairs_processed: u64 = 0;

            for (left_night, _) in pairs {
                let Some(left_vec) = seed_store.get(&left_night) else {
                    // Defensive: iterator is derived from keys, so this should not happen
                    continue;
                };
                if left_vec.is_empty() {
                    continue;
                }

                graph_rt
                    .add_inter_night_edges(
                        left_vec.as_slice(),
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

            // -----------------------------------------------------------------
            // 7) Convert to owned and store in runtime state
            // -----------------------------------------------------------------
            ctx.runtime_state.graph = graph_rt.to_owned();

            let edges_added =
                (ctx.runtime_state.graph.edges.len() as u64).saturating_sub(edges_before);

            Ok(vec![
                ("pairs_processed", pairs_processed),
                ("edges_added", edges_added),
            ])
        },
    )
}

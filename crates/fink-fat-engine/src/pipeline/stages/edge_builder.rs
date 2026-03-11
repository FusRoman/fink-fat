//! BuildEdges stage: construct inter-night edges between seed nodes.
//!
//! # Role in the pipeline
//!
//! `BuildEdges` runs after [`crate::pipeline::stages::seed_builder`] and produces the
//! directed inter-night linkage graph stored in
//! [`RuntimeState::graph`](crate::persistence::runtime_state::RuntimeState::graph).
//! The graph is later consumed by the solver stage.
//!
//! # Algorithm
//!
//! 1. Read the **new night IDs** advertised by `IngestNights` via
//!    [`RuntimeState::get_new_night_ids`](crate::persistence::runtime_state::RuntimeState::get_new_night_ids).
//!    These nights act as **right nights** (the later endpoint of each directed edge).
//! 2. Collect all nights already present in the
//!    [`SeedStore`](crate::seeding::store::SeedStore).
//! 3. For each right night $r$, collect every left night $l$ in the seed store
//!    satisfying $l < r$ and $r - l \leq \text{max\_gap}$.
//! 4. For each valid $(l, r)$ pair, call
//!    [`AlertLinkageDAG::add_inter_night_edges`](crate::graph::AlertLinkageDAG::add_inter_night_edges)
//!    to emit candidate edges.
//!
//! A new night can appear as a **left** night when two or more nights are ingested
//! together and one precedes the other within the gap window (e.g. N5 → N6 in a
//! batch run).
//!
//! # Main entry point
//!
//! - [`run`] — executes the full stage, invoked by the pipeline runner.

use crate::{
    error::EngineError,
    night_id::NightId,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
    spacetime_bucket::healpix_binner::HealpixBinner,
};

/// BuildEdges stage: construct inter-night edges from seeds.
///
/// Semantics
/// ---------
/// - Retrieve the newly ingested night IDs from `ctx.runtime_state.get_new_night_ids()`.
///   These nights act as **right nights** (the anchor side of each edge).
/// - Collect all nights present in the `SeedStore` as candidate **left nights**.
/// - For each `(left_night, right_night)` pair such that:
///   - `right_night` is a new night,
///   - `left_night < right_night`,
///   - `right_night - left_night <= max_gap_nights`,
///     build candidate edges using `RuntimeGraph::add_inter_night_edges`.
///
/// Notes
/// -----
/// - New nights can also appear as left nights if another, later new night exists within
///   the gap window (e.g. two nights ingested together, N5 → N6).
/// - This stage assumes that per-night seed vectors are already sorted by `SeedNode` order
///   (primary key `plane.epoch_mid`), so `add_inter_night_edges` must **not** re-sort `right_nodes`.
/// - The stage writes the resulting owned graph to `ctx.runtime_state.graph`.
pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::BuildEdges,
        hooks,
        StageMeta {
            label: PipelineStage::BuildEdges.label().to_string(),
            total: None,
        },
        |stage_sink| {
            // -----------------------------------------------------------------
            // 0) Preconditions
            // -----------------------------------------------------------------
            let new_nights: Vec<NightId> = ctx
                .runtime_state
                .get_new_night_ids()
                .ok_or_else(|| EngineError::StageFailed {
                    stage: PipelineStage::BuildEdges,
                    message: "runtime state does not contain new night IDs\nThe stage IngestNight must be run before BuildEdges".to_string(),
                })?
                .clone();

            let edge_config = &ctx.engine_config.edges;

            // -----------------------------------------------------------------
            // 1) Early exit: gap=0 => no possible (left < right) pair
            // -----------------------------------------------------------------
            let max_gap = ctx.engine_config.max_gap_nights();
            if max_gap == 0 {
                return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
            }

            // -----------------------------------------------------------------
            // 2) Collect and sort all nights present in the seed store.
            //    New nights act as "right" nights; any night that precedes a
            //    new night (within max_gap) is a valid "left" night.
            // -----------------------------------------------------------------
            let mut all_nights: Vec<NightId> =
                ctx.runtime_state.seed_store.nights().copied().collect();
            all_nights.sort();

            // Sort new nights as well to process them in a deterministic order.
            let mut new_nights_sorted = new_nights.clone();
            new_nights_sorted.sort();

            // Build the list of valid (left, right) pairs grouped by right night.
            // For each right_night in new_nights, collect all left_nights such that:
            //   left_night < right_night  AND  right_night - left_night <= max_gap
            let night_groups: Vec<(NightId, Vec<NightId>)> = new_nights_sorted
                .iter()
                .filter_map(|&right| {
                    let left_nights: Vec<NightId> = all_nights
                        .iter()
                        .filter(|&&left| {
                            left < right
                                && right.value().saturating_sub(left.value()) <= max_gap as u32
                        })
                        .filter(|&&left| {
                            // left night must have non-empty seeds
                            ctx.runtime_state
                                .seed_store
                                .get(&left)
                                .map(|v| !v.is_empty())
                                .unwrap_or(false)
                        })
                        .copied()
                        .collect();

                    if left_nights.is_empty() {
                        None
                    } else {
                        Some((right, left_nights))
                    }
                })
                .collect();

            // Early exit: nothing to connect.
            if night_groups.is_empty() {
                return Ok(vec![("pairs_processed", 0), ("edges_added", 0)]);
            }

            // -----------------------------------------------------------------
            // 3) Compute total left seeds across all valid pairs for progress.
            // -----------------------------------------------------------------
            let total_left_seeds: u64 = night_groups
                .iter()
                .flat_map(|(_, left_nights)| left_nights.iter())
                .filter_map(|n| ctx.runtime_state.seed_store.get(n))
                .map(|v| v.len() as u64)
                .sum();

            tracing::debug!(
                n_right_nights = night_groups.len(),
                total_left_seeds,
                max_gap,
                use_ml_ranking = ctx.engine_config.edges.use_ml_ranking,
                parallel = ctx.engine_config.edges.parallel_left_batches,
                "BuildEdges starting",
            );

            stage_sink.set_total(total_left_seeds);

            // -----------------------------------------------------------------
            // 4) Build spatial/time binners used by the edge builder.
            // -----------------------------------------------------------------
            let spatial_binner = HealpixBinner::new(ctx.engine_config.healpix_depth);
            let time_binner_width = ctx.engine_config.time_binner_width;

            // -----------------------------------------------------------------
            // 5) For each right night, build edges from all valid left nights.
            // -----------------------------------------------------------------
            let edges_before = ctx.runtime_state.graph.edges.len() as u64;
            let mut pairs_processed: u64 = 0;

            for (right_night, left_nights) in &night_groups {
                // Retrieve right seeds; skip if the right night has no seeds.
                let right_nodes = match ctx.runtime_state.seed_store.get(right_night) {
                    Some(v) if !v.is_empty() => v,
                    _ => {
                        tracing::warn!(
                            %right_night,
                            "right night has no seeds in SeedStore, skipping"
                        );
                        continue;
                    }
                };

                for left_night in left_nights {
                    tracing::trace!(%left_night, %right_night, "processing night pair");

                    let Some(left_vec) = ctx.runtime_state.seed_store.get(left_night) else {
                        tracing::warn!(
                            %left_night,
                            "left night has no seeds in SeedStore (unexpected since we pre-filtered valid nights)"
                        );
                        continue;
                    };

                    ctx.runtime_state
                        .graph
                        .add_inter_night_edges(
                            left_vec,
                            right_nodes,
                            edge_config,
                            &spatial_binner,
                            time_binner_width,
                            ctx.edge_models.as_ref(),
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
            }

            let edges_added =
                (ctx.runtime_state.graph.edges.len() as u64).saturating_sub(edges_before);

            tracing::debug!(pairs_processed, edges_added, "BuildEdges complete");

            Ok(vec![
                ("pairs_processed", pairs_processed),
                ("edges_added", edges_added),
            ])
        },
    )
}

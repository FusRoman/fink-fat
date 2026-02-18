// -----------------------------------------------------------------------------
// Edge module: inter-night edge construction, feature computation,
// and optional ML Top-K ranking
// -----------------------------------------------------------------------------
//
// Overview
// --------
// This module implements the core inter-night edge construction stage of the
// fink-fat engine.
//
// It defines:
//
// - `EdgeCore`: solver-facing scalar edge data (cost, dt, active flag).
// - `Edge<'seed_lf, 'alert_lf>`: a directed edge storing references to
//   `SeedNode`s (no ID resolution step needed).
// - Structured, cadence-robust feature computation (`EdgeFeatures`).
// - Optional ONNX-based ML Top-K ranking.
// - Sequential and Rayon-parallel edge building strategies.
//
// The main entrypoint is:
//
//     Edge::build_edges(...)
//
//
// High-level semantics
// --------------------
// Edges represent directed temporal links between seeds from two distinct
// nights (or more generally, two time-separated seed slices).
//
// Each edge:
//
// - points forward in time (`from` older → `to` newer),
// - carries a strictly positive scalar `cost`,
// - carries a strictly positive time gap `dt_days`,
// - is active by default.
//
// Costs are dimensionless and must be strictly positive to avoid:
//
// - zero-cost cycles,
// - negative-weight path degeneracies,
// - NaN propagation in graph solvers.
//
//
//
// Two operational modes
// ---------------------
// Controlled by `EdgeConfig.emit_all_edges`.
//
// 1) emit_all_edges = true
//    ---------------------------------
//    - All candidates returned by `SeedNode::seed_edge_candidates` are emitted.
//    - No ML model is used.
//    - Cost is derived purely from structured physics-inspired features via
//      `EdgeFeatures::kinematic_log_likelihood_cost()`.
//    - This mode is deterministic and useful for debugging or full graph builds.
//
// 2) emit_all_edges = false
//    ---------------------------------
//    - ML Top-K ranking is enabled.
//    - For each left seed:
//        • candidates are generated,
//        • features are computed,
//        • ONNX inference produces p(class=1),
//        • only the Top-K highest-probability candidates are retained.
//    - The solver-facing edge cost is still derived from features.
//    - Requires `model_pool` to be provided.
//
// In ML mode:
//
// - If `model_pool` is `None`, an error is returned.
// - If ONNX inference fails, the error is propagated as `EdgeBuilderError::ModelError(...)`.
//
//
//
// Parallelism model
// -----------------
// Controlled by:
//
// - `edge_config.parallel_left_batches`
// - `edge_config.parallel_left_batch_size`
//
// If enabled:
//
// - Left seeds are split into chunks.
// - Each chunk is processed independently using Rayon.
// - Each worker thread retrieves its own model instance from
//   `EdgeRankingModelPool` (no shared mutable session).
//
// If disabled:
//
// - The same chunking logic is used sequentially.
// - This bounds temporary memory usage and keeps behavior consistent.
//
//
//
// Spatial and temporal indexing
// -----------------------------
// Right-hand seeds are indexed once using `SeedSpatialIndex::build`.
//
// A `UniformTimeBinner` is constructed from:
//
// - the minimum epoch in the right slice,
// - `time_binner_width`.
//
// This index is reused across all chunks (sequential or parallel).
//
//
//
// Error model
// -----------
// All public APIs return:
//
//     Result<_, EdgeBuilderError>
//
// `EdgeBuilderError` includes:
//
// - invalid input seeds,
// - construction errors,
// - ML-related errors (via `EdgeModelError`).
//
// ML errors are wrapped in:
//
//     EdgeBuilderError::ModelError(EdgeModelError)
//
// No function in this module returns `EdgeModelError` directly.
//
//
//
// Lifetimes
// ---------
// `Edge<'seed_lf, 'alert_lf>` stores references to `SeedNode<'alert_lf>`.
//
// Therefore:
//
// - `left` and `right` slices must outlive the returned edges.
// - No cloning of seeds or alerts occurs during edge construction.
// - The graph remains zero-copy with respect to seeds.
//
// Owned persistence is handled via `Edge::to_owned()`.
//
// -----------------------------------------------------------------------------

pub mod edge_features;
pub mod error;
pub mod feature_core;
pub mod photometry_features;
pub mod position_features;
pub mod uncertainty_features;
pub mod velocity_features;

pub mod edge_prediction;
pub mod ranking_topk;

use std::{fmt, ops::Deref};

use serde::{Deserialize, Serialize};

use crate::{
    MJDTT,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        edge_features::EdgeFeatures,
        edge_prediction::EdgeRankingModelPool,
        error::{EdgeBuilderError, EdgeModelError},
        ranking_topk::rank_topk_edges_for_left,
    },
    persistence::edge::EdgeOwned,
    pipeline::progress_sink::ProgressSink,
    seeding::{SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::{spatial_binner::SpatialBinner, uniform_time_binner::UniformTimeBinner},
};

/// Core edge data that can be cheaply cloned and passed around.
///
/// Cost
/// ----
/// `cost` is a **dimensionless** scalar where lower is better. It is assumed to
/// already aggregate all the information needed by downstream solvers.
///
/// We enforce strictly **positive** costs (`> 0`) to avoid degeneracies:
/// - zero-cost cycles,
/// - ambiguous shortest paths,
/// - instability in optimization routines expecting positive weights.
///
/// Attributes
/// ----------
/// * `cost` – Finite strictly positive edge weight (dimensionless).
/// * `dt_days` – Time gap in days (TT), strictly positive.
/// * `active` – Runtime flag for pruning / solver logic / recomputation.
///
/// Notes
/// -----
/// - `active` is not part of feature computation; it is a graph-level control flag.
/// - If you need to store ML probability as well, keep it separate from `cost`
///   (or add a dedicated field).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeCore {
    /// Solver-facing cost (dimensionless, strictly positive).
    pub cost: f64,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f64,
    /// Whether the edge is currently active (used by solvers / CC exact recompute).
    pub active: bool,
}

/// Directed link from an older node to a newer node (forward in time).
///
/// This edge is the solver-facing representation of a potential inter-night link.
/// It stores references to the original [`SeedNode`] objects to avoid any later
/// ID → node resolution step.
///
/// Attributes
/// ----------
/// * `core` – Core edge data (cost, dt_days, active) that can be cheaply cloned and passed around).
/// * `from` – Source [`SeedNode`] (older epoch).
/// * `to` – Target [`SeedNode`] (newer epoch).
#[derive(Clone, Debug)]
pub struct Edge<'seed_lf> {
    pub core: EdgeCore,
    /// Source seed (older epoch).
    pub from: &'seed_lf SeedNode,
    /// Target seed (newer epoch).
    pub to: &'seed_lf SeedNode,
}

impl<'seed_lf> Deref for Edge<'seed_lf> {
    type Target = EdgeCore;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl<'seed_lf> fmt::Display for Edge<'seed_lf> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Edge {{ from: {}, to: {}, dt_days: {:.3}, cost: {:.4}, active: {} }}",
            self.from, self.to, self.dt_days, self.cost, self.active,
        )
    }
}

impl<'seed_lf> Edge<'seed_lf> {
    /// Create a new active edge with validated `cost` and `dt_days`.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node.
    /// * `to` – Target seed node.
    /// * `cost` – Edge cost (must be finite and strictly positive).
    /// * `dt_days` – Time separation in days (must be finite and strictly positive).
    ///
    /// Return
    /// ------
    /// A new [`Edge`] with `active = true`.
    ///
    /// Panics
    /// ------
    /// Panics if `cost` or `dt_days` are not finite or not strictly positive.
    ///
    /// Notes
    /// -----
    /// These assertions are deliberate because invalid weights can silently
    /// break downstream solvers (e.g. negative cycles, NaN propagation).
    pub fn new(from: &'seed_lf SeedNode, to: &'seed_lf SeedNode, cost: f64, dt_days: f64) -> Self {
        assert!(
            cost.is_finite() && cost > 0.0,
            "Edge cost must be finite and > 0."
        );
        assert!(
            dt_days.is_finite() && dt_days > 0.0,
            "dt_days must be finite and > 0."
        );
        Self {
            from,
            to,
            core: EdgeCore {
                cost,
                dt_days,
                active: true,
            },
        }
    }

    /// Convert this edge into an owned version that can be serialized.
    ///
    /// This extracts the necessary information from the `from` and `to` seeds
    /// to reconstruct the edge later without needing to serialize the entire
    /// `SeedNode`s. It relies on the fact that each `SeedNode` has a unique
    /// key (night ID + index in night) that can be used to look it up in a
    /// `SeedStore`.
    ///
    /// Return
    /// ------
    ///   An `EdgeOwned` containing the core edge data and the keys of the `from`
    ///   and `to` seeds.
    pub fn to_owned(&self) -> EdgeOwned {
        EdgeOwned {
            core: self.core.clone(),
            from: self.from.key(),
            to: self.to.key(),
        }
    }

    /* -------------------------- Edge construction API ------------------------- */

    /// Build directed edges between two seed slices.
    ///
    /// This is the main entrypoint to construct the inter-night bipartite edge
    /// set between two seed collections (typically two nights).
    ///
    /// Behavior (two modes)
    /// --------------------
    /// Controlled by `edge_config.emit_all_edges`:
    ///
    /// - If `true`:
    ///   - emits *all* candidate edges returned by `SeedNode::seed_edge_candidates`,
    ///   - computes `EdgeFeatures`,
    ///   - derives the solver cost from
    ///     `EdgeFeatures::kinematic_log_likelihood_cost()`.
    ///
    /// - If `false`:
    ///   - requires `model_pool` to be `Some(...)`,
    ///   - ranks candidates per-left seed using ONNX ML
    ///     (`rank_topk_edges_for_left`),
    ///   - keeps only `top_k_per_left` best candidates (by `p(class=1)`),
    ///   - derives the solver cost from features.
    ///
    /// Parallelism
    /// -----------
    /// Controlled by:
    ///
    /// - `edge_config.parallel_left_batches`
    /// - `edge_config.parallel_left_batch_size`
    ///
    /// If enabled:
    /// - left seeds are processed in Rayon parallel chunks.
    ///
    /// If disabled:
    /// - the same chunking logic is applied sequentially.
    ///
    /// Arguments
    /// ---------
    /// * `left` – Slice of source seeds (earlier epoch).
    /// * `right` – Slice of target seeds (later epoch).
    /// * `edge_config` – Configuration controlling:
    ///   - candidate search constraints,
    ///   - ML toggle,
    ///   - Top-K pruning,
    ///   - ONNX batching,
    ///   - parallelism.
    /// * `spatial_binner` – Spatial partitioner used to index `right`.
    /// * `time_binner_width` – Time bin width (days) for the uniform time index.
    /// * `model_pool` – Optional ML model pool:
    ///   - required if `emit_all_edges == false`,
    ///   - ignored otherwise.
    /// * `progress_sink` – Progress reporter updated per processed chunk.
    ///
    /// Return
    /// ------
    /// * `Ok(Vec<Edge>)` – Constructed edges referencing `left` and `right`.
    /// * `Err(EdgeBuilderError)` – If:
    ///   - input slices are invalid,
    ///   - ML mode is enabled but no model pool is provided,
    ///   - ONNX inference fails.
    ///
    /// Notes
    /// -----
    /// - The returned edge list is **not globally sorted**.
    ///   If deterministic ordering is required, sort at the call site.
    /// - `SeedSpatialIndex::build` is invoked exactly once.
    pub fn build_edges<B: SpatialBinner>(
        left: &'seed_lf [SeedNode],
        right: &'seed_lf [SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner_width: MJDTT,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn ProgressSink,
    ) -> Result<Vec<Self>, EdgeBuilderError> {
        // Init: total work = number of left seeds (units = seeds processed)
        progress_sink.set_total(left.len() as u64);

        // right seed are sorted by epoch_mid, so the first one has the minimum epoch.
        let right_seed_t0 = right.first().map(|s| s.plane.epoch_mid).ok_or_else(|| {
            EdgeBuilderError::InvalidSeeds(
                "no seed in the right seeds slice to get t0 in the edge builder".to_string(),
            )
        })?;

        let time_binner = UniformTimeBinner::new(right_seed_t0, time_binner_width);

        // Build an index over the right-hand seeds for fast candidate lookup.
        let right_index = SeedSpatialIndex::build(right, spatial_binner, &time_binner);

        // Chunking and per-left Top-K.
        let chunk_size = edge_config.parallel_left_batch_size.max(1);
        let top_k = edge_config.top_k_per_left;

        // Select sequential or parallel execution strategy.
        let res = match edge_config.parallel_left_batches {
            true => build_edges_parallel(
                left,
                chunk_size,
                &right_index,
                edge_config,
                top_k,
                model_pool,
                progress_sink,
            ),
            false => build_edges_sequential(
                left,
                chunk_size,
                &right_index,
                edge_config,
                top_k,
                model_pool,
                progress_sink,
            ),
        };

        // Clean: always finish, whether Ok or Err
        progress_sink.finish();

        res
    }
}

/// Process a chunk of left seeds by emitting *all* candidate edges (no ML / no Top-K).
///
/// This is the fast-path for the "emit all edges" debugging mode.
///
/// Arguments
/// ---------
/// * `chunk` – Slice of left-hand seeds processed together.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Candidate-generation configuration (search constraints).
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'alert_lf>>)` – All candidate edges for this chunk.
/// * `Err(EdgeBuilderError)` – Currently never returned here, but kept to share the
///   same error type as the ML path.
///
/// Notes
/// -----
/// - This can generate a very large number of edges; use with care.
/// - Cost is computed from cadence-robust features (parameter-free heuristic).
fn process_chunk_emit_all<'seed_lf>(
    chunk: &'seed_lf [SeedNode],
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
) -> Result<Vec<Edge<'seed_lf>>, EdgeBuilderError> {
    let mut local_edges: Vec<Edge<'seed_lf>> = Vec::new();

    for src in chunk.iter() {
        for to in src.seed_edge_candidates(right_index, edge_config) {
            // Compute cost from structured features (cadence-robust).
            let cost = EdgeFeatures::compute_features(src, to).kinematic_log_likelihood_cost();
            let dt_days = src.delta_days(to);

            local_edges.push(Edge::new(src, to, cost, dt_days));
        }
    }

    Ok(local_edges)
}

/// Process a chunk of left seeds using ML Top-K pruning per-left seed.
///
/// For each source seed:
/// - generate candidates,
/// - compute features in batches,
/// - run ONNX to obtain `p(class=1)`,
/// - keep only the Top-K candidates,
/// - emit edges with solver cost derived from features.
///
/// Arguments
/// ---------
/// * `chunk` – Slice of left-hand seeds processed together.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Candidate-generation and ONNX batching configuration.
/// * `top_k` – Number of candidates kept per left seed.
/// * `model_pool` – Per-thread model pool used to run ONNX inference.
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'alert_lf>>)` – ML-pruned edges for this chunk.
/// * `Err(EdgeBuilderError::ModelError)` – If ONNX inference fails.
///
/// Notes
/// -----
/// - `tmp` is reused to avoid allocations. It stores `(to, edge_cost)` for one `src`.
/// - We run `model_pool.with_mut(...)` per `src` so the model used is the current
///   thread’s instance (no locks).
fn process_chunk_ml_topk<'seed_lf>(
    chunk: &'seed_lf [SeedNode],
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: &EdgeRankingModelPool,
) -> Result<Vec<Edge<'seed_lf>>, EdgeBuilderError> {
    let mut local_edges: Vec<Edge<'seed_lf>> = Vec::new();

    // Temporary per-left output: avoids heap allocation for small top_k.
    let mut tmp: smallvec::SmallVec<[(&'seed_lf SeedNode, f64); 32]> = smallvec::SmallVec::new();

    for src in chunk.iter() {
        // Run ranking using the current thread’s model instance.
        model_pool.with_mut(|model| {
            rank_topk_edges_for_left(
                src,
                right_index,
                edge_config,
                model,
                top_k,
                edge_config.onnx_batch_size,
                &mut tmp,
            )
        })?;

        // Materialize edges for the winners.
        for (right_candidate, edge_cost) in tmp.iter() {
            let dt_days = src.delta_days(right_candidate);
            local_edges.push(Edge::new(src, *right_candidate, *edge_cost, dt_days));
        }
    }

    Ok(local_edges)
}

/// Process one chunk of left seeds according to `edge_config.emit_all_edges`.
///
/// This small dispatcher keeps the sequential/parallel loops clean.
///
/// Arguments
/// ---------
/// * `chunk` – Slice of left-hand seeds processed together.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Edge configuration controlling the mode.
/// * `top_k` – Top-K per-left used in ML mode.
/// * `model_pool` – Required in ML mode, ignored in emit-all mode.
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'alert_lf>>)` – Edges produced for this chunk.
/// * `Err(EdgeBuilderError::ModelError(EdgeModelError::MissingModel))` if ML mode is enabled without a pool.
/// * `Err(EdgeBuilderError::ModelError)` if ONNX inference fails.
fn process_chunk<'seed_lf>(
    chunk: &'seed_lf [SeedNode],
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: Option<&EdgeRankingModelPool>,
) -> Result<Vec<Edge<'seed_lf>>, EdgeBuilderError> {
    match edge_config.emit_all_edges {
        true => process_chunk_emit_all(chunk, right_index, edge_config),
        false => {
            let pool =
                model_pool.ok_or(EdgeBuilderError::ModelError(EdgeModelError::MissingModel))?;
            process_chunk_ml_topk(chunk, right_index, edge_config, top_k, pool)
        }
    }
}

/// Build edges by processing `left` in parallel chunks with Rayon.
///
/// Arguments
/// ---------
/// * `left` – Slice of left-hand seeds.
/// * `chunk_size` – Number of left seeds per Rayon task.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Edge configuration controlling mode and batching.
/// * `top_k` – Top-K per-left used in ML mode.
/// * `model_pool` – Optional model pool, required in ML mode.
/// * `progress_sink` – Progress reporter to update after processing each chunk.
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'seed_lf>>)` – Concatenated edges from all chunks.
/// * `Err(EdgeBuilderError)` – If processing any chunk fails.
///
/// Notes
/// -----
/// - Uses `try_reduce` to concatenate vectors efficiently without global locks.
/// - Each chunk returns its own `Vec<Edge>` which is appended into the accumulator.
fn build_edges_parallel<'seed_lf>(
    left: &'seed_lf [SeedNode],
    chunk_size: usize,
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: Option<&EdgeRankingModelPool>,
    progress_sink: &dyn ProgressSink,
) -> Result<Vec<Edge<'seed_lf>>, EdgeBuilderError> {
    use rayon::prelude::*;

    left.par_chunks(chunk_size)
        .map(|chunk| {
            let out = process_chunk(chunk, right_index, edge_config, top_k, model_pool)?;

            // Update: once per chunk to avoid too many calls
            progress_sink.inc(chunk.len() as u64);

            Ok(out)
        })
        .try_reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            Ok(a)
        })
}

/// Build edges by processing `left` sequentially in chunks.
///
/// Arguments
/// ---------
/// * `left` – Slice of left-hand seeds.
/// * `chunk_size` – Number of left seeds per chunk.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Edge configuration controlling mode and batching.
/// * `top_k` – Top-K per-left used in ML mode.
/// * `model_pool` – Optional model pool, required in ML mode.
/// * `progress_sink` – Progress reporter to update after processing each chunk.
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'seed_lf>>)` – Concatenated edges from all chunks.
/// * `Err(EdgeBuilderError)` – If processing any chunk fails.
///
/// Notes
/// -----
/// Chunking is still useful in sequential mode because it:
/// - bounds temporary memory growth,
/// - aligns behavior with the parallel implementation,
/// - keeps code structure consistent.
fn build_edges_sequential<'seed_lf>(
    left: &'seed_lf [SeedNode],
    chunk_size: usize,
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: Option<&EdgeRankingModelPool>,
    progress_sink: &dyn ProgressSink,
) -> Result<Vec<Edge<'seed_lf>>, EdgeBuilderError> {
    let mut edges: Vec<Edge<'seed_lf>> = Vec::new();

    for chunk in left.chunks(chunk_size) {
        edges.extend(process_chunk(
            chunk,
            right_index,
            edge_config,
            top_k,
            model_pool,
        )?);

        // Update: mark this chunk’s seeds as processed
        progress_sink.inc(chunk.len() as u64);
    }

    Ok(edges)
}

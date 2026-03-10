//! -----------------------------------------------------------------------------
//! Edge module: inter-night edge construction, feature computation,
//! and optional ML Top-K ranking
//! -----------------------------------------------------------------------------
//!
//! Overview
//! --------
//! This module implements the core inter-night edge construction stage of the
//! fink-fat engine.
//!
//! It defines:
//!
//! - `EdgeCore`: solver-facing scalar edge data (cost, dt, active flag).
//! - `Edge<'seed_lf, 'alert_lf>`: a directed edge storing references to
//!   `SeedNode`s (no ID resolution step needed).
//! - Structured, cadence-robust feature computation (`EdgeFeatures`).
//! - Optional ONNX-based ML Top-K ranking.
//! - Sequential and Rayon-parallel edge building strategies.
//!
//! The main entrypoint is:
//!   ```text
//!     Edge::build_edges(...)
//!   ```
//!
//!
//! High-level semantics
//! --------------------
//! Edges represent directed temporal links between seeds from two distinct
//! nights (or more generally, two time-separated seed slices).
//!
//! Each edge:
//!
//! - points forward in time (`from` older → `to` newer),
//! - carries a strictly positive scalar `cost`,
//! - carries a strictly positive time gap `dt_days`,
//! - is active by default.
//!
//! Costs are dimensionless and must be strictly positive to avoid:
//!
//! - zero-cost cycles,
//! - negative-weight path degeneracies,
//! - NaN propagation in graph solvers.
//!
//!
//!
//! Two operational modes (when Top-K is active)
//! ----------------------------------------------
//! Controlled by `EdgeConfig.use_ml_ranking` (only relevant when
//! `top_k_per_left = Some(k)`):
//!
//! 1) use_ml_ranking = false (default)
//!    ----------------------------------
//!    - For each left seed, generate candidates and compute cost via
//!      `EdgeFeatures::compute_cost`.
//!    - Retain only the K lowest-cost candidates (cost-based Top-K).
//!    - No ONNX model is required.
//!    - Cost is derived from `EdgeFeatures::compute_cost` using the variant
//!      configured in `edge_config.cost` (default: `gaussian_chi2`).
//!
//! 2) use_ml_ranking = true
//!    ----------------------------------
//!    - ML Top-K ranking is enabled.
//!    - For each left seed:
//!      • candidates are generated,
//!      • features are computed,
//!      • ONNX inference produces p(class=1),
//!      • only the Top-K highest-probability candidates are retained.
//!    - The solver-facing edge cost is still derived from features.
//!    - Requires `model_pool` to be provided.
//!
//! When `top_k_per_left = None`, all candidates are emitted regardless of
//! `use_ml_ranking`.
//!
//!
//!
//! Parallelism model
//! -----------------
//! Controlled by:
//!
//! - `edge_config.parallel_left_batches`
//! - `edge_config.parallel_left_batch_size`
//!
//! If enabled:
//!
//! - Left seeds are split into chunks.
//! - Each chunk is processed independently using Rayon.
//! - Each worker thread retrieves its own model instance from
//!   `EdgeRankingModelPool` (no shared mutable session).
//!
//! If disabled:
//!
//! - The same chunking logic is used sequentially.
//! - This bounds temporary memory usage and keeps behavior consistent.
//!
//!
//!
//! Spatial and temporal indexing
//! -----------------------------
//! Right-hand seeds are indexed once using `SeedSpatialIndex::build`.
//!
//! A `UniformTimeBinner` is constructed from:
//!
//! - the minimum epoch in the right slice,
//! - `time_binner_width`.
//!
//! This index is reused across all chunks (sequential or parallel).
//!
//!
//!
//! Error model
//! -----------
//! All public APIs return:
//! ```text
//!     Result<_, EdgeBuilderError>
//! ```
//!
//! `EdgeBuilderError` includes:
//!
//! - invalid input seeds,
//! - construction errors,
//! - ML-related errors (via `EdgeModelError`).
//!
//! ML errors are wrapped in:
//! ```text
//!     EdgeBuilderError::ModelError(EdgeModelError)
//! ```
//!
//! No function in this module returns `EdgeModelError` directly.
//!
//! -----------------------------------------------------------------------------

pub mod edge_features;
pub mod error;
pub mod feature_core;
pub mod photometry_features;
pub mod position_features;
pub mod uncertainty_features;
pub mod velocity_features;

pub mod edge_prediction;
pub mod ranking_topk;

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    MJDTT,
    alerts::DiaSourceId,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        edge_features::EdgeFeatures,
        edge_prediction::EdgeRankingModelPool,
        error::{EdgeBuilderError, EdgeModelError},
        ranking_topk::{rank_topk_edges_for_left, rank_topk_edges_for_left_by_cost},
    },
    pipeline::hooks::StageProgress,
    seeding::{SeedKey, SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::{spatial_binner::SpatialBinner, uniform_time_binner::UniformTimeBinner},
};

/// Stable identity for an edge in the persisted graph.
///
/// Notes
/// -----
/// This assumes there is at most one edge per `(from, to)` pair.
/// That matches the usual Fink-FAT semantics: a directed link between two seeds.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EdgeKey {
    /// Source seed (older epoch).
    pub from: SeedKey,
    /// Target seed (newer epoch).
    pub to: SeedKey,
}

/// Directed link from an older node to a newer node (forward in time).
///
/// This edge is the solver-facing representation of a potential inter-night link.
/// It stores references to the original [`SeedNode`] objects to avoid any later
/// ID → node resolution step.
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
/// * `from` – Source [`SeedNode`] (older epoch).
/// * `to` – Target [`SeedNode`] (newer epoch).
///
/// Notes
/// -----
/// - `active` is not part of feature computation; it is a graph-level control flag.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    /// Solver-facing cost (dimensionless, strictly positive).
    pub cost: f64,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f64,
    /// Whether the edge is currently active (used by solvers / CC exact recompute).
    pub active: bool,
    /// Source seed (older epoch).
    pub from: SeedKey,
    /// Target seed (newer epoch).
    pub to: SeedKey,
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Edge {{ from: {}, to: {}, dt_days: {:.3}, cost: {:.4}, active: {} }}",
            self.from, self.to, self.dt_days, self.cost, self.active,
        )
    }
}

impl Edge {
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
    /// `Ok(Edge)` with `active = true`, or an [`EdgeBuilderError::ConstructionError`]
    /// if any invariant is violated.
    ///
    /// Errors
    /// ------
    /// Returns [`EdgeBuilderError::ConstructionError`] if:
    /// - `cost` is not finite or not strictly positive, or
    /// - `dt_days` is not finite or not strictly positive.
    ///
    /// Notes
    /// -----
    /// Returning an error rather than panicking allows callers to handle
    /// degenerate weights gracefully without aborting the process.
    pub fn new(
        from: &SeedNode,
        to: &SeedNode,
        cost: f64,
        dt_days: f64,
    ) -> Result<Self, EdgeBuilderError> {
        if !cost.is_finite() || cost <= 0.0 {
            let from_ids: Vec<DiaSourceId> = from.members.iter().map(|k| k.dia_source_id).collect();
            let to_ids: Vec<DiaSourceId> = to.members.iter().map(|k| k.dia_source_id).collect();
            return Err(EdgeBuilderError::ConstructionError(format!(
                "Edge cost must be finite and > 0, got {cost} \
                 (from dia_source_ids={from_ids:?}, to dia_source_ids={to_ids:?})",
            )));
        }
        if !dt_days.is_finite() || dt_days <= 0.0 {
            let from_ids: Vec<DiaSourceId> = from.members.iter().map(|k| k.dia_source_id).collect();
            let to_ids: Vec<DiaSourceId> = to.members.iter().map(|k| k.dia_source_id).collect();
            return Err(EdgeBuilderError::ConstructionError(format!(
                "dt_days must be finite and > 0, got {dt_days} \
                 (from dia_source_ids={from_ids:?}, to dia_source_ids={to_ids:?})",
            )));
        }
        Ok(Self {
            from: from.key(),
            to: to.key(),
            cost,
            dt_days,
            active: true,
        })
    }

    /// Get the stable identity key for this edge.
    /// This is used for indexing and lookup in the graph and persistence layers.
    ///
    /// Return
    /// ------
    /// An [`EdgeKey`] uniquely identifying this edge by its source and target seeds.
    #[inline]
    pub fn key(&self) -> EdgeKey {
        EdgeKey {
            from: self.from,
            to: self.to,
        }
    }

    /* -------------------------- Edge construction API ------------------------- */

    /// Build directed edges between two seed slices.
    ///
    /// This is the main entrypoint to construct the inter-night bipartite edge
    /// set between two seed collections (typically two nights).
    ///
    /// Behavior
    /// --------
    /// Controlled by `edge_config.top_k_per_left` and `edge_config.use_ml_ranking`:
    ///
    /// - `top_k_per_left = None`:
    ///   - emits all candidate edges returned by `SeedNode::seed_edge_candidates`,
    ///   - computes `EdgeFeatures` and derives solver cost.
    ///
    /// - `top_k_per_left = Some(k)`, `use_ml_ranking = false` (default):
    ///   - ranks candidates per-left seed by physics-based cost,
    ///   - keeps only the `k` lowest-cost candidates.
    ///
    /// - `top_k_per_left = Some(k)`, `use_ml_ranking = true`:
    ///   - ranks candidates per-left seed using ONNX ML (`rank_topk_edges_for_left`),
    ///   - keeps only the `k` highest-probability candidates.
    ///   - requires `model_pool` to be `Some(...)`.
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
    ///   - ranking strategy (`use_ml_ranking`),
    ///   - Top-K limit (`top_k_per_left`),
    ///   - ONNX batching,
    ///   - parallelism.
    /// * `spatial_binner` – Spatial partitioner used to index `right`.
    /// * `time_binner_width` – Time bin width (days) for the uniform time index.
    /// * `model_pool` – Optional ML model pool:
    ///   - required if `use_ml_ranking == true`,
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
        left: &[SeedNode],
        right: &[SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner_width: MJDTT,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn StageProgress,
    ) -> Result<Vec<Self>, EdgeBuilderError> {
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

        tracing::debug!(
            n_left = left.len(),
            n_right = right.len(),
            chunk_size,
            top_k = ?top_k,
            parallel = edge_config.parallel_left_batches,
            use_ml_ranking = edge_config.use_ml_ranking,
            "build_edges starting",
        );
        tracing::trace!(
            right_seed_t0,
            time_binner_width,
            "right-side time binner initialised",
        );

        // Select sequential or parallel execution strategy.
        let edges = match edge_config.parallel_left_batches {
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
        }?;

        let (cost_min, cost_max, cost_mean) = edge_cost_stats(&edges);
        tracing::debug!(
            n_edges = edges.len(),
            cost_min,
            cost_max,
            cost_mean,
            "build_edges complete"
        );
        Ok(edges)
    }
}

/// Compute min, max, and mean cost from a slice of edges.
/// Returns `(0.0, 0.0, 0.0)` if the slice is empty.
#[inline]
fn edge_cost_stats(edges: &[Edge]) -> (f64, f64, f64) {
    if edges.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let (mn, mx, s) = edges.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64),
        |(mn, mx, s), e| (mn.min(e.cost), mx.max(e.cost), s + e.cost),
    );
    (mn, mx, s / edges.len() as f64)
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
///   When `edge_config.max_cost_cut` is set, candidates above the threshold are
///   discarded before materialising the edge, even though no Top-K filtering is
///   otherwise active.
///
/// Return
/// ------
/// * `Ok(Vec<Edge>)` – All candidate edges for this chunk (after optional cost cut).
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
) -> Result<Vec<Edge>, EdgeBuilderError> {
    let mut local_edges: Vec<Edge> = Vec::new();

    for src in chunk.iter() {
        for to in src.seed_edge_candidates(right_index, edge_config) {
            // Compute cost using the configured cost function (covariance model + loss).
            let cost = EdgeFeatures::compute_cost(src, to, &edge_config.cost_config);

            // Hard cost cut: skip candidates whose cost exceeds the threshold.
            if edge_config.max_cost_cut.is_some_and(|max| cost > max) {
                continue;
            }

            let dt_days = src.delta_days(to);
            local_edges.push(Edge::new(src, to, cost, dt_days)?);
        }
    }

    if tracing::enabled!(tracing::Level::TRACE) {
        let (cost_min, cost_max, cost_mean) = edge_cost_stats(&local_edges);
        tracing::trace!(
            chunk_size = chunk.len(),
            edges_in_chunk = local_edges.len(),
            cost_min,
            cost_max,
            cost_mean,
            "process_chunk_emit_all",
        );
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
/// * `Ok(Vec<Edge>)` – ML-pruned edges for this chunk.
/// * `Err(EdgeBuilderError::ModelError)` – If ONNX inference fails.
///
/// Notes
/// -----
/// - `tmp` is reused to avoid allocations. It stores `(to, edge_cost)` for one `src`.
/// - We run `model_pool.with_mut(...)` per `src` so the model used is the current
///   thread’s instance (no locks).
fn process_chunk_ml_topk(
    chunk: &[SeedNode],
    right_index: &SeedSpatialIndex<'_, '_>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: &EdgeRankingModelPool,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    let mut local_edges: Vec<Edge> = Vec::new();

    // Temporary per-left output: avoids heap allocation for small top_k.
    let mut tmp: smallvec::SmallVec<[(&SeedNode, f64); 32]> = smallvec::SmallVec::new();

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
            local_edges.push(Edge::new(src, right_candidate, *edge_cost, dt_days)?);
        }
    }

    if tracing::enabled!(tracing::Level::TRACE) {
        let (cost_min, cost_max, cost_mean) = edge_cost_stats(&local_edges);
        tracing::trace!(
            chunk_size = chunk.len(),
            top_k,
            edges_in_chunk = local_edges.len(),
            cost_min,
            cost_max,
            cost_mean,
            "process_chunk_ml_topk",
        );
    }

    Ok(local_edges)
}

/// Process one chunk of left seeds using cost-based Top-K pruning per-left seed.
///
/// For each source seed:
/// - generate candidates,
/// - compute edge cost via [`EdgeFeatures::compute_cost`],
/// - retain only the `top_k` candidates with the **lowest cost**.
///
/// Arguments
/// ---------
/// * `chunk` – Slice of left-hand seeds processed together.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Candidate-generation and cost function configuration.
/// * `top_k` – Number of candidates kept per left seed.
///
/// Return
/// ------
/// * `Ok(Vec<Edge>)` – Cost-pruned edges for this chunk.
/// * `Err(EdgeBuilderError)` – If edge construction fails (e.g. invalid cost or dt).
fn process_chunk_cost_topk(
    chunk: &[SeedNode],
    right_index: &SeedSpatialIndex<'_, '_>,
    edge_config: &EdgeConfig,
    top_k: usize,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    let mut local_edges: Vec<Edge> = Vec::new();
    let mut tmp: smallvec::SmallVec<[(&SeedNode, f64); 32]> = smallvec::SmallVec::new();

    for src in chunk.iter() {
        rank_topk_edges_for_left_by_cost(src, right_index, edge_config, top_k, &mut tmp);

        for (right_candidate, edge_cost) in tmp.iter() {
            let dt_days = src.delta_days(right_candidate);
            local_edges.push(Edge::new(src, right_candidate, *edge_cost, dt_days)?);
        }
    }

    if tracing::enabled!(tracing::Level::TRACE) {
        let (cost_min, cost_max, cost_mean) = edge_cost_stats(&local_edges);
        tracing::trace!(
            chunk_size = chunk.len(),
            top_k,
            edges_in_chunk = local_edges.len(),
            cost_min,
            cost_max,
            cost_mean,
            "process_chunk_cost_topk",
        );
    }

    Ok(local_edges)
}

/// Process one chunk of left seeds according to the configured ranking strategy.
///
/// Dispatches to one of three implementations based on `top_k` and
/// `edge_config.use_ml_ranking`:
///
/// - `top_k = None` → emit all candidate edges (no filtering).
/// - `top_k = Some(k)` and `use_ml_ranking = true` → ML Top-K via ONNX.
/// - `top_k = Some(k)` and `use_ml_ranking = false` → cost-based Top-K.
///
/// Arguments
/// ---------
/// * `chunk` – Slice of left-hand seeds processed together.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Edge configuration controlling the mode.
/// * `top_k` – Top-K per-left: `None` means emit all.
/// * `model_pool` – Required when `use_ml_ranking = true`, ignored otherwise.
///
/// Return
/// ------
/// * `Ok(Vec<Edge>)` – Edges produced for this chunk.
/// * `Err(EdgeBuilderError::ModelError(EdgeModelError::MissingModel))` if ML mode
///   is requested without a pool.
/// * `Err(EdgeBuilderError::ModelError)` if ONNX inference fails.
fn process_chunk(
    chunk: &[SeedNode],
    right_index: &SeedSpatialIndex<'_, '_>,
    edge_config: &EdgeConfig,
    top_k: Option<usize>,
    model_pool: Option<&EdgeRankingModelPool>,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    match top_k {
        None => process_chunk_emit_all(chunk, right_index, edge_config),
        Some(k) if edge_config.use_ml_ranking => {
            let pool =
                model_pool.ok_or(EdgeBuilderError::ModelError(EdgeModelError::MissingModel))?;
            process_chunk_ml_topk(chunk, right_index, edge_config, k, pool)
        }
        Some(k) => process_chunk_cost_topk(chunk, right_index, edge_config, k),
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
/// * `Ok(Vec<Edge>)` – Concatenated edges from all chunks.
/// * `Err(EdgeBuilderError)` – If processing any chunk fails.
///
/// Notes
/// -----
/// - Uses `try_reduce` to concatenate vectors efficiently without global locks.
/// - Each chunk returns its own `Vec<Edge>` which is appended into the accumulator.
fn build_edges_parallel(
    left: &[SeedNode],
    chunk_size: usize,
    right_index: &SeedSpatialIndex<'_, '_>,
    edge_config: &EdgeConfig,
    top_k: Option<usize>,
    model_pool: Option<&EdgeRankingModelPool>,
    progress_sink: &dyn StageProgress,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    use rayon::prelude::*;

    let n_chunks = left.chunks(chunk_size).count();
    tracing::debug!(
        n_left = left.len(),
        chunk_size,
        n_chunks,
        "build_edges_parallel starting"
    );

    let edges = left
        .par_chunks(chunk_size)
        .map(|chunk| -> Result<Vec<Edge>, EdgeBuilderError> {
            let out = process_chunk(chunk, right_index, edge_config, top_k, model_pool)?;

            // Update: once per chunk to avoid too many calls
            progress_sink.inc(chunk.len() as u64);

            Ok(out)
        })
        .try_reduce(Vec::<Edge>::new, |mut a, mut b| {
            a.append(&mut b);
            Ok(a)
        })?;

    tracing::debug!(n_edges = edges.len(), "build_edges_parallel complete");
    Ok(edges)
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
/// * `Ok(Vec<Edge>)` – Concatenated edges from all chunks.
/// * `Err(EdgeBuilderError)` – If processing any chunk fails.
///
/// Notes
/// -----
/// Chunking is still useful in sequential mode because it:
/// - bounds temporary memory growth,
/// - aligns behavior with the parallel implementation,
/// - keeps code structure consistent.
fn build_edges_sequential(
    left: &[SeedNode],
    chunk_size: usize,
    right_index: &SeedSpatialIndex<'_, '_>,
    edge_config: &EdgeConfig,
    top_k: Option<usize>,
    model_pool: Option<&EdgeRankingModelPool>,
    progress_sink: &dyn StageProgress,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    let n_chunks = left.chunks(chunk_size).count();
    tracing::debug!(
        n_left = left.len(),
        chunk_size,
        n_chunks,
        "build_edges_sequential starting"
    );

    let mut edges: Vec<Edge> = Vec::new();

    for chunk in left.chunks(chunk_size) {
        edges.extend(process_chunk(
            chunk,
            right_index,
            edge_config,
            top_k,
            model_pool,
        )?);

        // Update: mark this chunk's seeds as processed
        progress_sink.inc(chunk.len() as u64);
    }

    tracing::debug!(n_edges = edges.len(), "build_edges_sequential complete");
    Ok(edges)
}

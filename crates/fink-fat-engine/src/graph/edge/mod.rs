//! -----------------------------------------------------------------------------
//! Edge module: inter-night edge construction, feature computation,
//! and optional ML post-filtering
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
//! - Optional ONNX-based ML post-filtering applied to the retained edge set.
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
//! Controlled by `top_k_per_left`:
//!
//! 1) `top_k_per_left = None`
//!    ----------------------------------
//!    - Emit all candidate edges regardless of cost (debug / dataset mode).
//!    - Useful for generating exhaustive training datasets or controlled experiments.
//!    - No pruning; edge set can be very large.
//!
//! 2) `top_k_per_left = Some(k)`
//!    ----------------------------------
//!    - For each left seed, generate candidates and compute cost via
//!      `EdgeFeatures::compute_cost`.
//!    - Retain only the K lowest-cost candidates (cost-based Top-K).
//!    - No ONNX model is required at this stage.
//!    - Cost is derived from `EdgeFeatures::compute_cost` using the variant
//!      configured in `edge_config.cost` (default: `gaussian_chi2`).
//!
//! **ML post-filter** (when `ml_post_filter = true`):
//!
//! Applied one time, after all left seeds have been processed, regardless
//! of which Top-K mode was used above:
//!
//! - Score the **entire retained edge set** with the ONNX classifier (batched).
//! - Discard edges whose `p(class=1) < ml_post_filter_threshold`.
//! - Requires `model_pool` and `edge_ranking_model_path`.
//! - Much cheaper than per-seed ranking: inference runs on the already-pruned set.
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

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    MJDTT,
    alerts::DiaSourceId,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        edge_features::EdgeFeatures,
        edge_prediction::EdgeRankingModelPool,
        error::{EdgeBuilderError, EdgeModelError},
        ranking_topk::rank_topk_edges_for_left_by_cost,
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
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
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
    /// Controlled by `edge_config.top_k_per_left`:
    ///
    /// - `top_k_per_left = None`:
    ///   - emits all candidate edges returned by `SeedNode::seed_edge_candidates`,
    ///   - computes `EdgeFeatures` and derives solver cost.
    ///
    /// - `top_k_per_left = Some(k)`:
    ///   - ranks candidates per-left seed by physics-based cost,
    ///   - keeps only the `k` lowest-cost candidates.
    ///
    /// If `edge_config.ml_post_filter` is `true`, an ONNX classifier is applied
    /// once on the retained set: edges with `p(class=1) < ml_post_filter_threshold`
    /// are discarded.
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
    ///   - Top-K limit (`top_k_per_left`),
    ///   - ML post-filter toggle and threshold,
    ///   - ONNX batching,
    ///   - parallelism.
    /// * `spatial_binner` – Spatial partitioner used to index `right`.
    /// * `time_binner_width` – Time bin width (days) for the uniform time index.
    /// * `model_pool` – Optional ML model pool:
    ///   - required if `edge_config.ml_post_filter == true`,
    ///   - ignored otherwise.
    /// * `progress_sink` – Progress reporter updated per processed chunk.
    ///
    /// Return
    /// ------
    /// * `Ok(Vec<Edge>)` – Constructed edges referencing `left` and `right`.
    /// * `Err(EdgeBuilderError)` – If:
    ///   - input slices are invalid,
    ///   - ML post-filter is enabled but no model pool is provided,
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
            ml_post_filter = edge_config.ml_post_filter,
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
                progress_sink,
            ),
            false => build_edges_sequential(
                left,
                chunk_size,
                &right_index,
                edge_config,
                top_k,
                progress_sink,
            ),
        }?;

        // ML post-filter: score all retained edges and discard those below threshold.
        let edges = if edge_config.ml_post_filter {
            let pool =
                model_pool.ok_or(EdgeBuilderError::ModelError(EdgeModelError::MissingModel))?;
            let n_before = edges.len();
            let edges = apply_ml_post_filter(
                edges,
                left.iter(),
                right.iter(),
                edge_config.ml_post_filter_threshold,
                edge_config.onnx_batch_size,
                pool,
                edge_config.onnx_intra_threads,
            )?;
            tracing::debug!(
                n_before,
                n_after = edges.len(),
                threshold = edge_config.ml_post_filter_threshold,
                "ML post-filter applied",
            );
            edges
        } else {
            edges
        };

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

    /// Build directed edges between two seed slices using a pre-built right-hand index.
    ///
    /// Equivalent to [`Edge::build_edges`] but accepts a [`SeedSpatialIndex`] that
    /// was already constructed by the caller.  This avoids rebuilding the index for
    /// every left night when multiple left nights share the same right night, which
    /// is the common case when `max_gap > 1`.
    ///
    /// Arguments
    /// ---------
    /// * `left`        – Slice of left-hand seeds (earlier epoch).
    /// * `right_index` – Pre-built spatio-temporal index over the right-hand seeds.
    /// * `edge_config` – Edge configuration (same semantics as [`Edge::build_edges`]).
    /// * `model_pool`  – Optional ML model pool (required when `ml_post_filter == true`).
    /// * `progress_sink` – Progress reporter updated per processed chunk.
    ///
    /// Return
    /// ------
    /// * `Ok(Vec<Edge>)` – Constructed edges.
    /// * `Err(EdgeBuilderError)` – Same error conditions as [`Edge::build_edges`].
    pub fn build_edges_with_index<'seed_lf, 'binner_lf>(
        left: &[SeedNode],
        right_index: &SeedSpatialIndex<'seed_lf, 'binner_lf>,
        edge_config: &EdgeConfig,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn StageProgress,
    ) -> Result<Vec<Self>, EdgeBuilderError> {
        let chunk_size = edge_config.parallel_left_batch_size.max(1);
        let top_k = edge_config.top_k_per_left;

        tracing::debug!(
            n_left = left.len(),
            chunk_size,
            top_k = ?top_k,
            parallel = edge_config.parallel_left_batches,
            ml_post_filter = edge_config.ml_post_filter,
            "build_edges_with_index starting",
        );

        let edges = match edge_config.parallel_left_batches {
            true => build_edges_parallel(
                left,
                chunk_size,
                right_index,
                edge_config,
                top_k,
                progress_sink,
            ),
            false => build_edges_sequential(
                left,
                chunk_size,
                right_index,
                edge_config,
                top_k,
                progress_sink,
            ),
        }?;

        // ML post-filter: score all retained edges and discard those below threshold.
        let edges = if edge_config.ml_post_filter {
            let pool =
                model_pool.ok_or(EdgeBuilderError::ModelError(EdgeModelError::MissingModel))?;
            let n_before = edges.len();
            let edges = apply_ml_post_filter(
                edges,
                left.iter(),
                right_index.iter_seeds(),
                edge_config.ml_post_filter_threshold,
                edge_config.onnx_batch_size,
                pool,
                edge_config.onnx_intra_threads,
            )?;
            tracing::debug!(
                n_before,
                n_after = edges.len(),
                threshold = edge_config.ml_post_filter_threshold,
                "ML post-filter applied",
            );
            edges
        } else {
            edges
        };

        let (cost_min, cost_max, cost_mean) = edge_cost_stats(&edges);
        tracing::debug!(
            n_edges = edges.len(),
            cost_min,
            cost_max,
            cost_mean,
            "build_edges_with_index complete"
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
    // Conservative lower-bound capacity: at least one edge per left seed.
    let mut local_edges: Vec<Edge> = Vec::with_capacity(chunk.len());

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
    // top_k edges at most per left seed — exact upper bound.
    let mut local_edges: Vec<Edge> = Vec::with_capacity(chunk.len() * top_k);
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
/// Dispatches to one of two implementations based on `top_k`:
///
/// - `top_k = None` → emit all candidate edges (no filtering).
/// - `top_k = Some(k)` → cost-based Top-K.
///
/// Arguments
/// ---------
/// * `chunk` – Slice of left-hand seeds processed together.
/// * `right_index` – Spatial index over right-hand seeds.
/// * `edge_config` – Edge configuration controlling the mode.
/// * `top_k` – Top-K per-left: `None` means emit all.
///
/// Return
/// ------
/// * `Ok(Vec<Edge>)` – Edges produced for this chunk.
/// * `Err(EdgeBuilderError)` – If edge construction fails.
fn process_chunk(
    chunk: &[SeedNode],
    right_index: &SeedSpatialIndex<'_, '_>,
    edge_config: &EdgeConfig,
    top_k: Option<usize>,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    match top_k {
        None => process_chunk_emit_all(chunk, right_index, edge_config),
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
/// * `top_k` – Top-K per-left (`None` = emit all, `Some(k)` = cost-based Top-K).
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
            let out = process_chunk(chunk, right_index, edge_config, top_k)?;
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
/// * `top_k` – Top-K per-left (`None` = emit all, `Some(k)` = cost-based Top-K).
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
        edges.extend(process_chunk(chunk, right_index, edge_config, top_k)?);

        // Update: mark this chunk's seeds as processed
        progress_sink.inc(chunk.len() as u64);
    }

    tracing::debug!(n_edges = edges.len(), "build_edges_sequential complete");
    Ok(edges)
}

// =============================================================================
// ML post-filter
// =============================================================================

/// Flush one batch of edge features through the ONNX model and retain only
/// edges whose `p(class=1)` meets or exceeds `threshold`.
///
/// Drains and clears both `batch_features` and `batch_edges` on every call.
///
/// Arguments
/// ---------
/// * `batch_features` – Feature rows for the current batch; cleared on return.
/// * `batch_edges` – Edges aligned one-to-one with `batch_features`; drained on return.
/// * `kept` – Accumulator for edges that pass the threshold.
/// * `threshold` – Minimum `p(class=1)` required to keep an edge.
/// * `model_pool` – ONNX model pool used for inference.
/// * `onnx_intra_threads` – Optional intra-op thread count for the ORT session.
///   Applied at lazy session creation; ignored if the session is already live.
///
/// Return
/// ------
/// * `Ok(())` – Batch processed; `kept` updated, both input buffers cleared.
/// * `Err(EdgeBuilderError::ModelError)` – If ONNX inference fails.
#[inline]
fn flush_post_filter_batch(
    batch_features: &mut Vec<EdgeFeatures>,
    batch_edges: &mut Vec<Edge>,
    kept: &mut Vec<Edge>,
    threshold: f32,
    model_pool: &EdgeRankingModelPool,
    onnx_intra_threads: Option<usize>,
) -> Result<(), EdgeBuilderError> {
    let probas = model_pool
        .with_mut(onnx_intra_threads, |model| {
            model.predict_positive_proba(batch_features)
        })
        .map_err(EdgeBuilderError::ModelError)?;
    for (edge, proba) in batch_edges.drain(..).zip(probas) {
        if proba >= threshold {
            kept.push(edge);
        }
    }
    batch_features.clear();
    Ok(())
}

/// Apply ML post-filtering to a set of edges.
///
/// Looks up left and right [`SeedNode`]s by [`SeedKey`], computes
/// [`EdgeFeatures`], and scores each edge through the ONNX model in batches.
/// Only edges with `p(class=1) >= threshold` are kept.
///
/// Arguments
/// ---------
/// * `edges` – Full retained edge set to filter.
/// * `left_seeds` – Iterator over the left-hand [`SeedNode`]s; used to build
///   the `SeedKey → &SeedNode` lookup.
/// * `right_seeds` – Iterator over the right-hand [`SeedNode`]s; used to build
///   the `SeedKey → &SeedNode` lookup.
/// * `threshold` – Minimum `p(class=1)` to retain an edge.
/// * `batch_size` – Number of edges per ONNX inference call.
/// * `model_pool` – ONNX model pool; a session is borrowed per batch.
/// * `onnx_intra_threads` – Optional intra-op thread count for the ORT session.
///   Applied at lazy session creation; ignored if the session is already live.
///
/// Return
/// ------
/// * `Ok(Vec<Edge>)` – Edges that passed the threshold.
/// * `Err(EdgeBuilderError::ModelError)` – If ONNX inference fails.
///
/// Notes
/// -----
/// - If a seed lookup fails for an edge (e.g. seed not present in either
///   iterator), the edge is kept conservatively and a `WARN` is emitted.
/// - Returns `Ok(edges)` immediately if `edges` is empty.
fn apply_ml_post_filter<'a>(
    edges: Vec<Edge>,
    left_seeds: impl Iterator<Item = &'a SeedNode>,
    right_seeds: impl Iterator<Item = &'a SeedNode>,
    threshold: f32,
    batch_size: usize,
    model_pool: &EdgeRankingModelPool,
    onnx_intra_threads: Option<usize>,
) -> Result<Vec<Edge>, EdgeBuilderError> {
    if edges.is_empty() {
        return Ok(edges);
    }

    let batch_size = batch_size.max(1);

    let left_by_key: AHashMap<SeedKey, &SeedNode> = left_seeds.map(|s| (s.key(), s)).collect();
    let right_by_key: AHashMap<SeedKey, &SeedNode> = right_seeds.map(|s| (s.key(), s)).collect();

    let mut kept: Vec<Edge> = Vec::with_capacity(edges.len());
    let mut batch_features: Vec<EdgeFeatures> = Vec::with_capacity(batch_size);
    let mut batch_edges: Vec<Edge> = Vec::with_capacity(batch_size);

    for edge in edges {
        let from_node = left_by_key
            .get(&edge.from)
            .or_else(|| right_by_key.get(&edge.from));
        let to_node = right_by_key
            .get(&edge.to)
            .or_else(|| left_by_key.get(&edge.to));

        match (from_node, to_node) {
            (Some(&from), Some(&to)) => {
                batch_features.push(EdgeFeatures::compute_features(from, to));
                batch_edges.push(edge);
                if batch_features.len() >= batch_size {
                    flush_post_filter_batch(
                        &mut batch_features,
                        &mut batch_edges,
                        &mut kept,
                        threshold,
                        model_pool,
                        onnx_intra_threads,
                    )?;
                }
            }
            _ => {
                // Seed lookup failed — keep the edge conservatively.
                tracing::warn!(
                    from = ?edge.from,
                    to = ?edge.to,
                    "ML post-filter: seed lookup failed, keeping edge"
                );
                kept.push(edge);
            }
        }
    }

    // Flush remaining partial batch.
    if !batch_features.is_empty() {
        flush_post_filter_batch(
            &mut batch_features,
            &mut batch_edges,
            &mut kept,
            threshold,
            model_pool,
            onnx_intra_threads,
        )?;
    }

    Ok(kept)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod edge_mod_tests {
    use std::sync::Arc;

    use camino::Utf8PathBuf;

    use crate::{
        Alert, AlertKey,
        engine_config::edge_config::EdgeConfig,
        graph::edge::edge_prediction::{EdgeRankingModel, EdgeRankingModelPool},
        night_id::NightId,
        pipeline::hooks::NoopProgress,
        seeding::{SeedNode, seed_spatial_index::SeedSpatialIndex, store::SeedStore},
        spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
    };

    use super::Edge;

    // -------------------------------------------------------------------------
    // Fixtures
    // -------------------------------------------------------------------------

    fn model_path() -> Utf8PathBuf {
        let manifest_dir = Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("tests/ml_model/edge_classifier.onnx")
    }

    /// Minimal alert at `(ra, dec)` on `night` at epoch `mjd`, with 1-arcsec
    /// positional errors and band `1` (g).
    fn make_alert(id: u64, night: u32, mjd: f64, ra: f64, dec: f64) -> Alert {
        const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);
        Alert {
            key: AlertKey {
                night_id: NightId::new(night),
                dia_source_id: id,
            },
            ra,
            ra_err: ARCSEC,
            dec,
            dec_err: ARCSEC,
            mjd_tt: mjd,
            flux: 1000.0,
            flux_err: 50.0,
            band: 1,
            observer_mpc_code: Arc::new("500".into()),
        }
    }

    /// Build a `SeedNode` from two alerts 30 min apart on `night`, starting at
    /// `(ra, dec)` with angular velocity `vx` rad/day in RA.
    fn make_seed(
        store: &mut SeedStore,
        night: u32,
        id_a: u64,
        id_b: u64,
        mjd: f64,
        ra: f64,
        dec: f64,
        vx: f64,
    ) -> SeedNode {
        let dt = 0.5 / 24.0; // 30 min in days
        let a = make_alert(id_a, night, mjd, ra, dec);
        let b = make_alert(id_b, night, mjd + dt, ra + vx * dt, dec);
        SeedNode::from_pair(store, NightId::new(night), &a, &b, None)
            .expect("test seeds should form a valid pair")
    }

    /// Build `n` left seeds (night 1, MJD 60000) and `n` right seeds (night 2,
    /// MJD 60001) with kinematically consistent inter-night motion of
    /// `vx = 3e-3 rad/day`.  Seeds are slightly separated in RA so each
    /// left seed has a unique nearest right seed.
    fn build_left_right(n: u32) -> (Vec<SeedNode>, Vec<SeedNode>) {
        let mut store = SeedStore::new();
        let ra0 = 1.0_f64;
        let dec = 0.1_f64;
        let vx = 3e-3_f64; // ~0.17 °/day

        let left: Vec<SeedNode> = (0..n)
            .map(|i| {
                let ra = ra0 + i as f64 * 5e-4;
                make_seed(
                    &mut store,
                    1,
                    i as u64 * 2,
                    i as u64 * 2 + 1,
                    60000.0,
                    ra,
                    dec,
                    vx,
                )
            })
            .collect();

        let right: Vec<SeedNode> = (0..n)
            .map(|i| {
                let ra = ra0 + i as f64 * 5e-4 + vx * 1.0;
                make_seed(
                    &mut store,
                    2,
                    1000 + i as u64 * 2,
                    1001 + i as u64 * 2,
                    60001.0,
                    ra,
                    dec,
                    vx,
                )
            })
            .collect();

        (left, right)
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    /// ML model is loaded via the pool and `build_edges_with_index` returns
    /// edges with strictly positive costs and dt_days.
    #[test]
    fn ml_build_edges_returns_valid_edges() {
        let path = model_path();
        assert!(
            path.exists(),
            "ONNX model not found at {path}; ensure tests/ml_model/edge_classifier.onnx exists"
        );

        let (left, right) = build_left_right(5);
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60001.0, 1.0);
        let right_index = SeedSpatialIndex::build(&right, &spatial_binner, &time_binner);

        let pool = EdgeRankingModelPool::new(&path);
        let cfg = EdgeConfig {
            ml_post_filter: true,
            top_k_per_left: Some(5),
            ..EdgeConfig::default()
        };

        let edges =
            Edge::build_edges_with_index(&left, &right_index, &cfg, Some(&pool), &NoopProgress)
                .expect("build_edges_with_index should succeed");

        assert!(
            !edges.is_empty(),
            "Expected at least one edge between left and right seeds"
        );
        for e in &edges {
            assert!(
                e.cost > 0.0 && e.cost.is_finite(),
                "cost must be finite and strictly positive, got {}",
                e.cost
            );
            assert!(
                e.dt_days > 0.0 && e.dt_days.is_finite(),
                "dt_days must be finite and strictly positive, got {}",
                e.dt_days
            );
        }
    }

    /// ML top-k filter limits the number of edges to at most `top_k` per left seed.
    #[test]
    fn ml_topk_fan_out_bounded_by_top_k() {
        let path = model_path();
        assert!(path.exists(), "ONNX model not found at {path}");

        let top_k = 2usize;
        let (left, right) = build_left_right(5);
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60001.0, 1.0);
        let right_index = SeedSpatialIndex::build(&right, &spatial_binner, &time_binner);

        let pool = EdgeRankingModelPool::new(&path);
        let cfg = EdgeConfig {
            ml_post_filter: true,
            top_k_per_left: Some(top_k),
            ..EdgeConfig::default()
        };

        let edges =
            Edge::build_edges_with_index(&left, &right_index, &cfg, Some(&pool), &NoopProgress)
                .expect("build_edges_with_index should succeed");

        assert!(
            edges.len() <= left.len() * top_k,
            "Expected at most {} edges (top_k={top_k} × n_left={}), got {}",
            left.len() * top_k,
            left.len(),
            edges.len()
        );
    }

    /// Both ML-based and cost-based top-k produce edges for the same input.
    /// This confirms the model is invoked without error and that the two
    /// strategies are interchangeable at the API level.
    #[test]
    fn ml_and_cost_topk_both_produce_edges() {
        let path = model_path();
        assert!(path.exists(), "ONNX model not found at {path}");

        let (left, right) = build_left_right(5);
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60001.0, 1.0);
        let right_index = SeedSpatialIndex::build(&right, &spatial_binner, &time_binner);

        // Run cost-based top-k (no ONNX model required).
        let cfg_cost = EdgeConfig {
            ml_post_filter: false,
            top_k_per_left: Some(5),
            ..EdgeConfig::default()
        };
        let cost_edges =
            Edge::build_edges_with_index(&left, &right_index, &cfg_cost, None, &NoopProgress)
                .expect("cost-based build should succeed");

        // Run ML-based top-k.
        let pool = EdgeRankingModelPool::new(&path);
        let cfg_ml = EdgeConfig {
            ml_post_filter: true,
            top_k_per_left: Some(5),
            ..EdgeConfig::default()
        };
        let ml_edges =
            Edge::build_edges_with_index(&left, &right_index, &cfg_ml, Some(&pool), &NoopProgress)
                .expect("ML-based build should succeed");

        assert!(
            !cost_edges.is_empty(),
            "Cost-based ranking should find edges"
        );
        assert!(!ml_edges.is_empty(), "ML ranking should find edges");
    }

    /// `EdgeRankingModel::predict_positive_proba` returns values in [0, 1] for
    /// a batch of real edge features derived from synthetic seeds.
    #[test]
    fn ml_model_probabilities_are_in_unit_interval() {
        use crate::graph::edge::edge_features::EdgeFeatures;

        let path = model_path();
        assert!(path.exists(), "ONNX model not found at {path}");

        let mut model = EdgeRankingModel::load_edge_ranking_model(&path, None)
            .expect("EdgeRankingModel should load");

        let (left, right) = build_left_right(3);
        let features: Vec<EdgeFeatures> = left
            .iter()
            .flat_map(|l| {
                right
                    .iter()
                    .map(move |r| EdgeFeatures::compute_features(l, r))
            })
            .collect();

        assert!(!features.is_empty());

        let proba = model
            .predict_positive_proba(&features)
            .expect("predict_positive_proba should succeed");

        assert_eq!(
            proba.len(),
            features.len(),
            "One probability per feature row"
        );
        for (i, p) in proba.iter().enumerate() {
            assert!((0.0..=1.0).contains(p), "p[{i}] = {p} is outside [0, 1]");
        }
    }

    /// A pool pointing to a non-existent model path returns an error when
    /// `build_edges_with_index` is called with `use_ml_ranking = true`.
    #[test]
    fn ml_pool_with_missing_model_returns_error() {
        let missing = Utf8PathBuf::from("/nonexistent/path/model.onnx");
        let (left, right) = build_left_right(2);
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60001.0, 1.0);
        let right_index = SeedSpatialIndex::build(&right, &spatial_binner, &time_binner);

        let pool = EdgeRankingModelPool::new(&missing);
        let cfg = EdgeConfig {
            ml_post_filter: true,
            top_k_per_left: Some(1),
            ..EdgeConfig::default()
        };

        let result =
            Edge::build_edges_with_index(&left, &right_index, &cfg, Some(&pool), &NoopProgress);
        assert!(
            result.is_err(),
            "Expected error for missing model path, got Ok"
        );
    }
}

// -----------------------------------------------------------------------------
// Edge module: edges + feature computation + optional ML Top-K ranking
// -----------------------------------------------------------------------------
//
// This module contains the main edge construction pipeline of the fink-fat
// engine. It exposes:
//
// - The `Edge<'alert_lf>` type: a directed link `from -> to` storing references to
//   `SeedNode`s plus a solver-friendly scalar cost.
// - Structured, cadence-robust feature computation (`EdgeFeatures` and friends).
// - Optional ML ranking via ONNX (see `edge_prediction` + `ranking_topk`).
// - Parallel and sequential implementations for building inter-night edges.
//
// Two operational modes
// ---------------------
// The behavior is controlled by `EdgeConfig`:
//
// 1) emit_all_edges = true
//    - No ML ranking and no Top-K pruning.
//    - Emit every candidate returned by `SeedNode::seed_edge_candidates`.
//    - Cost is derived from `EdgeFeatures::kinematic_log_likelihood_cost()`.
//
// 2) emit_all_edges = false
//    - ML Top-K ranking is enabled.
//    - For each left seed, candidates are batched, scored by ONNX, and only the
//      Top-K highest `p(class=1)` candidates are kept.
//    - Cost is still derived from the feature cost, but the candidate set is
//      reduced by the ML model.
//
// Parallelism
// -----------
// Edge building can be run sequentially or with Rayon (`parallel_left_batches`).
// Because ONNX Runtime requires `&mut Session`, multi-thread inference is handled
// via `EdgeRankingModelPool` (one model instance per worker thread).
//
// Lifetimes
// ---------
// `Edge<'alert_lf>` stores references to `SeedNode`s, so the input slices must outlive
// the returned edges.
//
// -----------------------------------------------------------------------------

pub mod edge_features;
pub mod feature_core;
pub mod photometry_features;
pub mod position_features;
pub mod uncertainty_features;
pub mod velocity_features;

pub mod edge_prediction;
pub mod ranking_topk;

use std::fmt;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        edge_features::EdgeFeatures,
        edge_prediction::{EdgeModelError, EdgeRankingModelPool},
        ranking_topk::rank_topk_edges_for_left,
    },
    seeding::{seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::{spatial_binner::SpatialBinner, time_binner::TimeBinner},
};

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
/// * `from` – Source [`SeedNode`] (older epoch).
/// * `to` – Target [`SeedNode`] (newer epoch).
/// * `cost` – Finite strictly positive edge weight (dimensionless).
/// * `dt_days` – Time gap in days (TT), strictly positive.
/// * `active` – Runtime flag for pruning / solver logic / recomputation.
///
/// Notes
/// -----
/// - `active` is not part of feature computation; it is a graph-level control flag.
/// - If you need to store ML probability as well, keep it separate from `cost`
///   (or add a dedicated field).
#[derive(Clone, Debug)]
pub struct Edge<'seed_lf, 'alert_lf> {
    /// Source seed (older epoch).
    pub from: &'seed_lf SeedNode<'alert_lf>,
    /// Target seed (newer epoch).
    pub to: &'seed_lf SeedNode<'alert_lf>,
    /// Solver-facing cost (dimensionless, strictly positive).
    pub cost: f64,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f64,
    /// Whether the edge is currently active (used by solvers / CC exact recompute).
    pub active: bool,
}

impl<'seed_lf, 'alert_lf> fmt::Display for Edge<'seed_lf, 'alert_lf> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Edge {{ from: {}, to: {}, dt_days: {:.3}, cost: {:.4}, active: {} }}",
            self.from, self.to, self.dt_days, self.cost, self.active,
        )
    }
}

impl<'seed_lf, 'alert_lf> Edge<'seed_lf, 'alert_lf> {
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
    pub fn new(
        from: &'seed_lf SeedNode<'alert_lf>,
        to: &'seed_lf SeedNode<'alert_lf>,
        cost: f64,
        dt_days: f64,
    ) -> Self {
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
            cost,
            dt_days,
            active: true,
        }
    }

    /* -------------------------- Edge construction API ------------------------- */

    /// Build directed edges from a set of left-hand seeds to a set of right-hand seeds.
    ///
    /// This is the main entrypoint to construct the inter-night bipartite edge
    /// set between two seed slices (typically two nights).
    ///
    /// Behavior (two modes)
    /// --------------------
    /// Controlled by `edge_config.emit_all_edges`:
    ///
    /// - If `true`:
    ///   - emits *all* candidate edges returned by `SeedNode::seed_edge_candidates`,
    ///   - computes `EdgeFeatures` for each candidate,
    ///   - uses `EdgeFeatures::kinematic_log_likelihood_cost()` as the edge cost.
    ///
    /// - If `false`:
    ///   - requires `model_pool` to be `Some(...)`,
    ///   - ranks candidates per-left seed using ONNX ML (`rank_topk_edges_for_left`),
    ///   - keeps only `top_k_per_left` best candidates (by `p(class=1)`),
    ///   - uses the derived feature cost as the edge cost.
    ///
    /// Parallelism
    /// -----------
    /// Controlled by `edge_config.parallel_left_batches`:
    /// - If `true`: uses Rayon to process chunks of left seeds in parallel.
    /// - If `false`: processes chunks sequentially.
    ///
    /// Chunking is controlled by `edge_config.parallel_left_batch_size`.
    ///
    /// Arguments
    /// ---------
    /// * `left` – Slice of source seeds (earlier night).
    /// * `right` – Slice of target seeds (later night).
    /// * `edge_config` – Configuration controlling:
    ///   - candidate search constraints,
    ///   - `emit_all_edges` toggle,
    ///   - `top_k_per_left` for ML mode,
    ///   - parallel chunking and ONNX batch sizes.
    /// * `spatial_binner` – Spatial partitioner used for indexing `right` seeds.
    /// * `time_binner` – Time binning strategy used for indexing `right` seeds.
    /// * `model_pool` – Optional per-thread ML model pool:
    ///   - required if `emit_all_edges == false`,
    ///   - ignored otherwise.
    ///
    /// Return
    /// ------
    /// * `Ok(Vec<Edge<'alert_lf>>)` – List of constructed edges containing references to
    ///   `SeedNode`s from `left` and `right`.
    /// * `Err(EdgeModelError)` – If ML mode is enabled and:
    ///   - the model pool is missing,
    ///   - or inference fails.
    ///
    /// Notes
    /// -----
    /// - The returned edge list is **not globally sorted** by default.
    ///   If you need global ordering (e.g. for deterministic truncation),
    ///   sort the result at the call site.
    /// - `SeedSpatialIndex::build` is called once and shared across chunks.
    pub fn build_edges<B: SpatialBinner, T: TimeBinner>(
        left: &'seed_lf [SeedNode<'alert_lf>],
        right: &'seed_lf [SeedNode<'alert_lf>],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner: &T,
        model_pool: Option<&EdgeRankingModelPool>,
    ) -> Result<Vec<Self>, EdgeModelError> {
        // Build an index over the right-hand seeds for fast candidate lookup.
        let right_index = SeedSpatialIndex::build(right, spatial_binner, time_binner);

        // Chunking and per-left Top-K.
        let chunk_size = edge_config.parallel_left_batch_size.max(1);
        let top_k = edge_config.top_k_per_left;

        // Select sequential or parallel execution strategy.
        match edge_config.parallel_left_batches {
            true => build_edges_parallel(
                left,
                chunk_size,
                &right_index,
                edge_config,
                top_k,
                model_pool,
            ),
            false => build_edges_sequential(
                left,
                chunk_size,
                &right_index,
                edge_config,
                top_k,
                model_pool,
            ),
        }
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
/// * `Err(EdgeModelError)` – Currently never returned here, but kept to share the
///   same error type as the ML path.
///
/// Notes
/// -----
/// - This can generate a very large number of edges; use with care.
/// - Cost is computed from cadence-robust features (parameter-free heuristic).
fn process_chunk_emit_all<'seed_lf, 'alert_lf>(
    chunk: &'seed_lf [SeedNode<'alert_lf>],
    right_index: &SeedSpatialIndex<'seed_lf, '_, 'alert_lf>,
    edge_config: &EdgeConfig,
) -> Result<Vec<Edge<'seed_lf, 'alert_lf>>, EdgeModelError>
{
    let mut local_edges: Vec<Edge<'seed_lf, 'alert_lf>> = Vec::new();

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
/// * `Err(EdgeModelError)` – If ONNX inference fails.
///
/// Notes
/// -----
/// - `tmp` is reused to avoid allocations. It stores `(to, edge_cost)` for one `src`.
/// - We run `model_pool.with_mut(...)` per `src` so the model used is the current
///   thread’s instance (no locks).
fn process_chunk_ml_topk<'seed_lf, 'alert_lf>(
    chunk: &'seed_lf [SeedNode<'alert_lf>],
    right_index: &SeedSpatialIndex<'seed_lf, '_, 'alert_lf>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: &EdgeRankingModelPool,
) -> Result<Vec<Edge<'seed_lf, 'alert_lf>>, EdgeModelError> {
    let mut local_edges: Vec<Edge<'seed_lf, 'alert_lf>> = Vec::new();

    // Temporary per-left output: avoids heap allocation for small top_k.
    let mut tmp: smallvec::SmallVec<[(&'seed_lf SeedNode<'alert_lf>, f64); 32]> = smallvec::SmallVec::new();

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
/// * `Err(EdgeModelError::MissingModel)` if ML mode is enabled without a pool.
/// * `Err(EdgeModelError)` if ONNX inference fails.
fn process_chunk<'seed_lf, 'alert_lf>(
    chunk: &'seed_lf [SeedNode<'alert_lf>],
    right_index: &SeedSpatialIndex<'seed_lf, '_, 'alert_lf>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: Option<&EdgeRankingModelPool>,
) -> Result<Vec<Edge<'seed_lf, 'alert_lf>>, EdgeModelError> {
    match edge_config.emit_all_edges {
        true => process_chunk_emit_all(chunk, right_index, edge_config),
        false => {
            let pool = model_pool.ok_or(EdgeModelError::MissingModel)?;
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
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'alert_lf>>)` – Concatenated edges from all chunks.
/// * `Err(EdgeModelError)` – If processing any chunk fails.
///
/// Notes
/// -----
/// - Uses `try_reduce` to concatenate vectors efficiently without global locks.
/// - Each chunk returns its own `Vec<Edge>` which is appended into the accumulator.
fn build_edges_parallel<'seed_lf, 'alert_lf>(
    left: &'seed_lf [SeedNode<'alert_lf>],
    chunk_size: usize,
    right_index: &SeedSpatialIndex<'seed_lf, '_, 'alert_lf>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: Option<&EdgeRankingModelPool>,
) -> Result<Vec<Edge<'seed_lf, 'alert_lf>>, EdgeModelError> {
    use rayon::prelude::*;

    left.par_chunks(chunk_size)
        .map(|chunk| process_chunk(chunk, right_index, edge_config, top_k, model_pool))
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
///
/// Return
/// ------
/// * `Ok(Vec<Edge<'alert_lf>>)` – Concatenated edges from all chunks.
/// * `Err(EdgeModelError)` – If processing any chunk fails.
///
/// Notes
/// -----
/// Chunking is still useful in sequential mode because it:
/// - bounds temporary memory growth,
/// - aligns behavior with the parallel implementation,
/// - keeps code structure consistent.
fn build_edges_sequential<'seed_lf, 'alert_lf>(
    left: &'seed_lf [SeedNode<'alert_lf>],
    chunk_size: usize,
    right_index: &SeedSpatialIndex<'seed_lf, '_, 'alert_lf>,
    edge_config: &EdgeConfig,
    top_k: usize,
    model_pool: Option<&EdgeRankingModelPool>,
) -> Result<Vec<Edge<'seed_lf, 'alert_lf>>, EdgeModelError> {
    let mut edges: Vec<Edge<'seed_lf, 'alert_lf>> = Vec::new();

    for chunk in left.chunks(chunk_size) {
        edges.extend(process_chunk(
            chunk,
            right_index,
            edge_config,
            top_k,
            model_pool,
        )?);
    }

    Ok(edges)
}

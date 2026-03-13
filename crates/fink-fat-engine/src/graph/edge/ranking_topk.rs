//! # Batched Top-K ranking for candidate edges (per-left seed)
//!
//! This module implements tight, allocation-aware routines to rank candidate
//! edges for a single left seed against a set of right seeds indexed by
//! [`crate::seeding::seed_spatial_index::SeedSpatialIndex`].
//!
//! ## Problem
//!
//! Given one left [`crate::seeding::SeedNode`], the search window may yield
//! thousands of candidate right seeds. The goal is to retain only the Top-K
//! most promising candidates and return a compact list with pre-computed
//! `edge_cost` values suitable for downstream graph solvers.
//!
//! Two ranking strategies are provided:
//!
//! - **ML ranking** ([`rank_topk_edges_for_left`]): candidates are scored by an
//!   ONNX classifier and ranked by $p(\text{true\_edge})$.
//! - **Cost-based ranking** ([`rank_topk_edges_for_left_by_cost`]): candidates
//!   are ranked directly by the physics-based
//!   [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`] score
//!   (lowest cost = best candidate). No ONNX model is required.
//!
//! ## Performance constraints
//!
//! - ONNX inference is relatively expensive per call; batching is critical
//!   (ML path only).
//! - Top-K maintenance must be $O(\log K)$ per accepted candidate, with $O(1)$
//!   fast rejection once the heap is full.
//! - The full candidate list is never materialized: features are accumulated
//!   in fixed-size batches and discarded after each flush (ML path).
//! - When [`crate::engine_config::edge_config::EdgeConfig::max_cost_cut`] is set,
//!   candidates above the cost threshold are discarded immediately without entering
//!   the heap, reducing unnecessary work across all modes.
//!
//! ## Implementation strategy (ML path)
//!
//! 1. Iterate candidates from
//!    [`crate::seeding::SeedNode::seed_edge_candidates`].
//! 2. Accumulate [`crate::graph::edge::edge_features::EdgeFeatures`] rows into
//!    batches of size `batch_size`.
//! 3. Run one ONNX inference call per batch via
//!    [`crate::graph::edge::edge_prediction::EdgeRankingModel::predict_positive_proba`].
//! 4. Maintain a fixed-capacity `TopK` container (min-heap) so only the best
//!    candidates are retained.
//! 5. Inside the batch-flush step, after computing the cost for each survivor of the
//!    probability threshold check, apply `max_cost_cut` if set: discard the
//!    candidate if its cost exceeds the threshold.
//! 6. At the end, emit winners sorted by descending probability.
//!
//! ## Implementation strategy (cost-based path)
//!
//! 1. Iterate candidates; compute cost inline for each.
//! 2. Apply `max_cost_cut` if set: skip the candidate if its cost exceeds the
//!    threshold before score mapping or heap insertion.
//! 3. Map cost to a score $s = \frac{1}{1+c}$ for use with the same `TopK`
//!    min-heap (higher score = lower cost = better candidate).
//! 4. Emit winners sorted by ascending cost.
//!
//! ## Main types
//!
//! - `TopK` — Fixed-capacity min-heap keeping the K highest-scoring items.
//! - `TopKItem` — Single heap entry carrying a score, a right-seed
//!   reference, and a pre-computed edge cost.
//!
//! ## Public entry points
//!
//! - [`rank_topk_edges_for_left`] — ML-based Top-K: rank by ONNX probability.
//! - [`rank_topk_edges_for_left_by_cost`] — Cost-based Top-K: rank by lowest cost.
//!
//! ## Notes
//!
//! - [`crate::graph::edge::edge_prediction::EdgeRankingModel::predict_positive_proba`]
//!   is assumed to return one score per input row in the same order.
//! - [`crate::graph::edge::edge_features::EdgeFeatures::compute_features`] is
//!   expected to be deterministic and free of NaNs/Infs thanks to
//!   `FeatureCore::finite_or_zero` guards upstream.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use smallvec::SmallVec;

use crate::engine_config::edge_config::{CostConfig, EdgeConfig};
use crate::graph::edge::edge_features::EdgeFeatures;
use crate::graph::edge::edge_prediction::EdgeRankingModel;
use crate::graph::edge::error::EdgeModelError;
use crate::graph::edge::feature_core::FeatureCore;
use crate::seeding::SeedNode;
use crate::seeding::seed_spatial_index::SeedSpatialIndex;

/// Single entry in the [`TopK`] heap: one candidate edge with its ML score.
///
/// Bundles together everything needed to represent a candidate right seed after
/// batched ONNX scoring: the raw probability, a reference to the seed node, and
/// the pre-computed edge cost used by downstream solvers.
///
/// `Ord` is implemented so that [`std::collections::BinaryHeap`] behaves as a
/// **min-heap** on `proba` (lowest score at the top), which enables $O(1)$
/// threshold queries for Top-K pruning.
///
/// Attributes
/// ----------
/// * `proba` – Model score $p(\text{class}=1)$ for this candidate edge,
///   as returned by
///   [`crate::graph::edge::edge_prediction::EdgeRankingModel::predict_positive_proba`].
/// * `to` – Borrowed reference to the right-hand [`crate::seeding::SeedNode`]
///   (edge head). Stored by reference to avoid duplicating seed data.
/// * `edge_cost` – Additive solver cost derived from
///   [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`].
///   Cached here because recomputing it after selection would require
///   re-evaluating the full feature pipeline.
///
/// Notes
/// -----
/// - The lifetime `'seed_lf` ties `to` to the slice of right seeds passed to
///   [`rank_topk_edges_for_left`]; items must not outlive that slice.
#[derive(Debug)]
struct TopKItem<'seed_lf> {
    proba: f32,
    to: &'seed_lf SeedNode,
    edge_cost: f64,
}

impl<'seed_lf> PartialEq for TopKItem<'seed_lf> {
    /// Test equality between two heap items by raw probability bits.
    ///
    /// Bit-level comparison avoids IEEE NaN semantics (`NaN != NaN`) and ensures
    /// consistency with the total ordering defined by [`TopKItem::cmp`].
    ///
    /// Arguments
    /// ---------
    /// * `other` – The item to compare against.
    ///
    /// Return
    /// ------
    /// `true` if `self.proba` and `other.proba` have identical bit patterns.
    fn eq(&self, other: &Self) -> bool {
        self.proba.to_bits() == other.proba.to_bits()
    }
}

impl<'seed_lf> Eq for TopKItem<'seed_lf> {}

impl<'seed_lf> PartialOrd for TopKItem<'seed_lf> {
    /// Partial ordering delegating to the total order defined by [`TopKItem::cmp`].
    ///
    /// Always returns `Some`, making this a total partial order consistent with
    /// `Ord`. The comparison is **reversed** relative to `proba` so that
    /// [`std::collections::BinaryHeap`] (a max-heap) behaves as a min-heap:
    /// the item with the *lowest* probability sits at the top and acts as the
    /// $O(1)$ pruning threshold for Top-K selection.
    ///
    /// Arguments
    /// ---------
    /// * `other` – The item to compare against.
    ///
    /// Return
    /// ------
    /// `Some(Ordering)` — always `Some`; the wrapped ordering is the reverse of
    /// the natural `proba` order.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'seed_lf> Ord for TopKItem<'seed_lf> {
    /// Total ordering for heap operations (reversed min-heap on `proba`).
    ///
    /// Uses `f32::total_cmp` for a fully deterministic float order that handles
    /// NaN, ±0, and subnormals consistently. The result is **reversed** so that
    /// [`std::collections::BinaryHeap`] (which pops the greatest element)
    /// effectively pops the item with the *smallest* probability — the correct
    /// behaviour for a Top-K min-heap.
    ///
    /// Arguments
    /// ---------
    /// * `other` – The item to compare against.
    ///
    /// Return
    /// ------
    /// `Ordering` — the reverse of `self.proba.total_cmp(&other.proba)`.
    fn cmp(&self, other: &Self) -> Ordering {
        other.proba.total_cmp(&self.proba)
    }
}

/// Fixed-capacity container that retains the K [`TopKItem`]s with the highest
/// ML probability.
///
/// Internally wraps a [`std::collections::BinaryHeap`] configured as a
/// **min-heap** (via the reversed `Ord` on [`TopKItem`]): the item with the
/// *lowest* probability among those kept is always at the top, providing an
/// $O(1)$ pruning threshold. Once the heap reaches capacity, any incoming
/// candidate whose probability does not exceed the current minimum is rejected
/// without touching the heap.
///
/// Attributes
/// ----------
/// * `k` – Maximum number of items to keep.
///   When `k == 0` the container accepts nothing.
/// * `heap` – Min-heap (capacity pre-allocated to `k + 1`) storing the
///   currently retained Top-K [`TopKItem`]s.
///
/// Complexity
/// ----------
/// * `push` (accepted entry): $O(\log K)$
/// * `threshold` query: $O(1)$
/// * Fast rejection (heap full): $O(1)$
/// * `into_sorted_desc`: $O(K \log K)$
///
/// Notes
/// -----
/// - Heap capacity is pre-allocated to `k.saturating_add(1)` to avoid
///   reallocation during the brief moment when an old minimum is replaced.
#[derive(Debug)]
struct TopK<'seed_lf> {
    k: usize,
    heap: BinaryHeap<TopKItem<'seed_lf>>,
}

impl<'seed_lf> TopK<'seed_lf> {
    /// Create an empty Top-K container pre-allocated for `k` items.
    ///
    /// Arguments
    /// ---------
    /// * `k` – Number of best items to retain. Pass `0` to create a
    ///   no-op container that discards every candidate.
    ///
    /// Return
    /// ------
    /// A [`TopK`] with an empty heap whose internal capacity is
    /// `k.saturating_add(1)`, ready to accept candidates via [`TopK::push`].
    #[inline]
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Return the current pruning threshold for Top-K insertion.
    ///
    /// Arguments
    /// ---------
    /// Takes no arguments beyond `&self`.
    ///
    /// Return
    /// ------
    /// * $-\infty$ (`f32::NEG_INFINITY`) — when the heap holds fewer than `k`
    ///   items (not yet full; accept all candidates unconditionally).
    /// * The probability of the least-probable kept item — once the heap is
    ///   full; a new candidate must **strictly exceed** this value to displace it.
    ///
    /// Notes
    /// -----
    /// - Called before every candidate to enable $O(1)$ early rejection in
    ///   [`TopK::push`] and [`flush_batch`].
    /// - Returns $-\infty$ even when `k == 0` (the `peek` yields `None`).
    #[inline]
    fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap
                .peek()
                .map(|x| x.proba)
                .unwrap_or(f32::NEG_INFINITY)
        }
    }

    /// Try to insert a candidate into the Top-K set.
    ///
    /// Behavior
    /// --------
    /// - `k == 0`: no-op — the container was created for zero items.
    /// - Heap not full (`len < k`): always insert.
    /// - Heap full (`len == k`):
    ///   - if `item.proba <= min_kept`: reject without touching the heap ($O(1)$);
    ///   - if `item.proba > min_kept`: pop the current minimum and push `item`
    ///     ($O(\log K)$).
    ///
    /// Arguments
    /// ---------
    /// * `item` – Candidate [`TopKItem`] to consider for retention.
    ///
    /// Return
    /// ------
    /// `()` — the decision (keep or discard) is reflected in the heap state.
    ///
    /// Notes
    /// -----
    /// - This is the innermost hot-path operation; it is `#[inline]`.
    /// - Equality (`item.proba == min_kept`) is treated as rejection to avoid
    ///   replacing an equally-scored item with no net gain.
    #[inline]
    fn push(&mut self, item: TopKItem<'seed_lf>) {
        if self.k == 0 {
            return;
        }

        if self.heap.len() < self.k {
            self.heap.push(item);
            return;
        }

        // Heap full: compare to current minimum (top of the min-heap).
        let min_kept = self
            .heap
            .peek()
            .map(|x| x.proba)
            .unwrap_or(f32::NEG_INFINITY);

        if item.proba <= min_kept {
            // Reject: candidate does not belong to the top-k set.
            return;
        }

        // Replace the smallest kept item.
        let _ = self.heap.pop();
        self.heap.push(item);
    }

    /// Consume the container and drain all kept items, sorted best-first.
    ///
    /// Arguments
    /// ---------
    /// Consumes `self`; no additional arguments.
    ///
    /// Return
    /// ------
    /// `Vec<TopKItem<'seed_lf>>` sorted by **descending** probability
    /// (highest `proba` first). The length is at most `k`.
    ///
    /// Notes
    /// -----
    /// - Popping all elements from a min-heap yields them in ascending order,
    ///   not descending; an explicit `sort_by` with `total_cmp` is therefore
    ///   applied after draining.
    /// - Complexity: $O(K \log K)$ for the sort.
    fn into_sorted_desc(self) -> Vec<TopKItem<'seed_lf>> {
        // `BinaryHeap::into_vec` is O(1): it returns the internal storage
        // without any allocation or element-by-element drain.
        let mut out = self.heap.into_vec();
        out.sort_by(|a, b| b.proba.total_cmp(&a.proba));
        out
    }
}

/// Score one full batch of candidate edges and update the [`TopK`] container.
///
/// This is the key batching primitive called from [`rank_topk_edges_for_left`]:
/// - runs a single ONNX inference call for the accumulated `batch_features`,
/// - pairs returned probabilities with their corresponding right-seed candidates,
/// - applies [`TopK::threshold`]-based early rejection before computing cost,
/// - applies the `max_cost_cut` threshold if set (discards high-cost candidates),
/// - inserts survivors into `top`,
/// - drains `batch_to` and clears `batch_features` for reuse.
///
/// Arguments
/// ---------
/// * `src` – Left-hand [`crate::seeding::SeedNode`] (edge tail), required by
///   [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`].
/// * `cost_config` – Cost function parameters forwarded verbatim to
///   `EdgeFeatures::compute_cost`.
/// * `model` – ONNX edge-ranking model; mutable because the ORT session
///   mutates internal state during inference.
/// * `max_cost_cut` – Optional hard upper bound on edge cost. Candidates whose
///   computed cost exceeds this value are discarded before heap insertion.
///   `None` disables the cut.
/// * `top` – [`TopK`] container to update with surviving candidates.
/// * `batch_features` – Accumulated feature rows (one per candidate).
///   Cleared by this function after inference.
/// * `batch_to` – Right-seed references aligned one-to-one with `batch_features`.
///   Drained by this function after inference.
///
/// Return
/// ------
/// * `Ok(())` — batch processed; `top`, `batch_features`, and `batch_to`
///   updated in place.
/// * `Err(`[`crate::graph::edge::error::EdgeModelError`]`)` — ONNX inference
///   failed; `top` may be partially updated.
///
/// Invariants
/// ----------
/// - `batch_features.len()` must equal `batch_to.len()` on entry.
/// - Both vectors are left empty on return (regardless of success or failure
///   path), allowing the caller to push new candidates without reallocating.
///
/// Notes
/// -----
/// - `edge_cost` is computed **only** for candidates that pass the probability
///   threshold, avoiding unnecessary work when the Top-K set is already tight.
/// - The `max_cost_cut` check is applied **after** cost computation and **before**
///   heap insertion, so it runs only for candidates that survived probability
///   screening.
/// - Calling with empty `batch_features` is a no-op and returns `Ok(())` immediately.
#[inline]
#[allow(clippy::too_many_arguments)]
fn flush_batch<'seed_lf>(
    src: &SeedNode,
    cost_config: &CostConfig,
    max_cost_cut: Option<f64>,
    model: &mut EdgeRankingModel,
    top: &mut TopK<'seed_lf>,
    batch_features: &mut Vec<EdgeFeatures>,
    batch_to: &mut Vec<&'seed_lf SeedNode>,
    batch_cores: &mut Vec<FeatureCore>,
) -> Result<(), EdgeModelError> {
    if batch_features.is_empty() {
        // Nothing to do (and ensures batch_to and batch_cores are also empty).
        return Ok(());
    }

    // One ONNX call for the whole batch: returns p(class=1) for each row.
    let probas = model.predict_positive_proba(batch_features.as_slice())?;

    // Drain aligned streams: (seed, core) + probability.
    // The FeatureCore was already built during feature extraction — reusing it
    // avoids a second propagation + projection + covariance computation for
    // every candidate that passes the probability threshold.
    for ((right_candidate, core), proba) in
        batch_to.drain(..).zip(batch_cores.drain(..)).zip(probas)
    {
        // Cheap early reject if this candidate can't enter the current Top-K set.
        if proba <= top.threshold() {
            continue;
        }

        // Reuse the already-built FeatureCore: avoids a second from_nodes call.
        let cost = EdgeFeatures::compute_cost_from_core(&core, src, right_candidate, cost_config);

        // Hard cost cut: discard candidates whose cost exceeds the threshold.
        if max_cost_cut.is_some_and(|max| cost > max) {
            continue;
        }

        top.push(TopKItem {
            proba,
            to: right_candidate,
            edge_cost: cost,
        });
    }

    // Important: clear features so we don't re-score them on the next flush.
    batch_features.clear();
    Ok(())
}

/// Rank all candidate right seeds for one left seed and return the Top-K by ML
/// probability.
///
/// This is the main public entry point for per-left ML-based edge ranking.
/// Given a single source [`crate::seeding::SeedNode`], it enumerates all
/// spatially compatible right seeds, scores them with an ONNX classifier in
/// batches, and writes the best `topk` candidates (with their solver costs)
/// into a caller-provided output buffer.
///
/// Arguments
/// ---------
/// * `src` – Left-hand [`crate::seeding::SeedNode`] (edge tail).
/// * `right_index` – Spatial/time index over the right-hand seed collection;
///   used by [`crate::seeding::SeedNode::seed_edge_candidates`] to enumerate
///   spatially compatible candidates.
/// * `edge_config` – Edge configuration controlling candidate search constraints,
///   cost function parameters, and batch sizing.
/// * `model` – ONNX edge-ranking model (mutable because the ORT session
///   mutates internal state during inference).
/// * `topk` – Maximum number of right-seed candidates to return. Pass `0`
///   to skip ML ranking entirely (output will be empty).
/// * `batch_size` – Number of candidates submitted to the model per ONNX call.
///   Larger values increase throughput but consume more memory. Clamped to a
///   minimum of 1.
/// * `out` – Caller-provided output buffer (cleared on entry). Filled with
///   `(right_seed, edge_cost)` pairs sorted by descending ML probability.
///
/// Return
/// ------
/// * `Ok(())` — `out` contains up to `topk` entries sorted best-first by
///   $p(\text{class}=1)$.
/// * `Err(`[`crate::graph::edge::error::EdgeModelError`]`)` — ONNX inference
///   failed; `out` may be partially filled.
///
/// Notes
/// -----
/// - Output is written into the caller-provided [`smallvec::SmallVec`] to
///   avoid heap allocation in the common case where `topk ≤ 32`.
/// - [`crate::graph::edge::edge_features::EdgeFeatures`] are computed for
///   **every** candidate, but `edge_cost` (via
///   [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`]) is
///   evaluated only for candidates that pass the current Top-K threshold,
///   avoiding unnecessary work.
/// - [`crate::seeding::SeedNode::seed_edge_candidates`] yields candidates in
///   arbitrary order; result correctness does not depend on input order.
/// - [`crate::graph::edge::edge_prediction::EdgeRankingModel::predict_positive_proba`]
///   is assumed to preserve the input row order.
pub fn rank_topk_edges_for_left<'seed_lf>(
    src: &'seed_lf SeedNode,
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
    model: &mut EdgeRankingModel,
    topk: usize,
    batch_size: usize,
    out: &mut SmallVec<[(&'seed_lf SeedNode, f64); 32]>,
) -> Result<(), EdgeModelError> {
    // Output is provided by caller to avoid allocations in hot paths.
    out.clear();

    // Avoid degenerate batch sizes.
    let batch_size = batch_size.max(1);

    // Fixed-capacity Top-K structure (stores only the best candidates).
    let mut top: TopK<'seed_lf> = TopK::new(topk);

    // Batch buffers reused across flushes (all three are kept aligned by index).
    let mut batch_features: Vec<EdgeFeatures> = Vec::with_capacity(batch_size);
    let mut batch_to: Vec<&'seed_lf SeedNode> = Vec::with_capacity(batch_size);
    // FeatureCore is built once per candidate and stored alongside features so
    // that flush_batch can compute the edge cost without rebuilding the core.
    let mut batch_cores: Vec<FeatureCore> = Vec::with_capacity(batch_size);

    // Generate candidate right nodes and batch them for ONNX inference.
    for to in src.seed_edge_candidates(right_index, edge_config) {
        // Build the shared intermediates once; derive both features and the
        // cached core in one pass (avoids a second from_nodes in flush_batch).
        let core = FeatureCore::from_nodes(src, to);
        batch_features.push(EdgeFeatures::from_core(src, to, &core));
        batch_cores.push(core);
        batch_to.push(to);

        // Flush when batch is full.
        if batch_features.len() >= batch_size {
            flush_batch(
                src,
                &edge_config.cost_config,
                edge_config.max_cost_cut,
                model,
                &mut top,
                &mut batch_features,
                &mut batch_to,
                &mut batch_cores,
            )?;
        }
    }

    // Final flush for any remaining candidates.
    flush_batch(
        src,
        &edge_config.cost_config,
        edge_config.max_cost_cut,
        model,
        &mut top,
        &mut batch_features,
        &mut batch_to,
        &mut batch_cores,
    )?;

    // Emit winners into output buffer (best probability first).
    out.extend(
        top.into_sorted_desc()
            .into_iter()
            .map(|it| (it.to, it.edge_cost)),
    );

    Ok(())
}

/// Rank all candidate right seeds for one left seed and return the Top-K by
/// physics-based edge cost (lowest cost first).
///
/// This is the cost-based counterpart to [`rank_topk_edges_for_left`]. It does
/// not require an ONNX model: candidates are ranked directly by the scalar cost
/// derived from [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`].
/// The `topk` candidates with the **lowest** cost are retained.
///
/// Implementation note
/// -------------------
/// Internally, each candidate's cost is mapped to a "score" via
/// $s = \frac{1}{1 + c}$, which is monotonically decreasing in cost.
/// The existing `TopK` min-heap (designed for "highest score wins") can then be
/// reused without modification: keeping the top-$K$ scores is equivalent to
/// keeping the bottom-$K$ costs.
///
/// Arguments
/// ---------
/// * `src` – Left-hand [`crate::seeding::SeedNode`] (edge tail).
/// * `right_index` – Spatial/time index over the right-hand seed collection;
///   used by [`crate::seeding::SeedNode::seed_edge_candidates`] to enumerate
///   spatially compatible candidates.
/// * `edge_config` – Edge configuration controlling candidate search constraints,
///   cost function parameters, and the optional `max_cost_cut` threshold.
/// * `topk` – Maximum number of right-seed candidates to return. Pass `0`
///   to discard all candidates (output will be empty).
/// * `out` – Caller-provided output buffer (cleared on entry). Filled with
///   `(right_seed, edge_cost)` pairs sorted by ascending edge cost (best first).
///
/// Return
/// ------
/// `out` contains up to `topk` entries sorted best-first (lowest cost first).
/// This function is infallible.
///
/// Notes
/// -----
/// - No batching is performed; cost is computed inline for each candidate.
/// - When [`crate::engine_config::edge_config::EdgeConfig::max_cost_cut`] is set,
///   any candidate whose cost exceeds the threshold is discarded before score
///   mapping and heap insertion.
/// - Unlike [`rank_topk_edges_for_left`], this function never returns an error
///   because no ONNX inference is involved.
/// - [`crate::seeding::SeedNode::seed_edge_candidates`] yields candidates in
///   arbitrary order; result correctness does not depend on input order.
pub fn rank_topk_edges_for_left_by_cost<'seed_lf>(
    src: &'seed_lf SeedNode,
    right_index: &SeedSpatialIndex<'seed_lf, '_>,
    edge_config: &EdgeConfig,
    topk: usize,
    out: &mut SmallVec<[(&'seed_lf SeedNode, f64); 32]>,
) {
    out.clear();

    let mut top: TopK<'seed_lf> = TopK::new(topk);

    for to in src.seed_edge_candidates(right_index, edge_config) {
        let cost = EdgeFeatures::compute_cost(src, to, &edge_config.cost_config);

        // Hard cost cut: discard candidates whose cost exceeds the threshold.
        if edge_config.max_cost_cut.is_some_and(|max| cost > max) {
            continue;
        }

        // Map cost to a score in (0, 1]: lower cost → higher score.
        // This lets us reuse the max-score TopK directly.
        let score = (1.0 / (1.0 + cost)) as f32;

        if score <= top.threshold() {
            continue;
        }

        top.push(TopKItem {
            proba: score,
            to,
            edge_cost: cost,
        });
    }

    // Emit winners sorted by descending score = ascending cost (lowest cost first).
    out.extend(
        top.into_sorted_desc()
            .into_iter()
            .map(|it| (it.to, it.edge_cost)),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ranking_topk_tests {
    use super::*;

    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a `TopKItem` with a dummy `SeedNode` reference. Because tests
    /// only exercise the heap logic (not the seed data), we use a static
    /// sentinel to satisfy the lifetime.
    fn dummy_item(proba: f32) -> TopKItem<'static> {
        static DUMMY: std::sync::OnceLock<SeedNode> = std::sync::OnceLock::new();
        let node = DUMMY.get_or_init(SeedNode::default);
        TopKItem {
            proba,
            to: node,
            edge_cost: 0.0,
        }
    }

    /// Score formula extracted for isolated testing.
    fn cost_to_score(cost: f64) -> f32 {
        (1.0 / (1.0 + cost)) as f32
    }

    // -----------------------------------------------------------------------
    // Score-mapping unit tests
    // -----------------------------------------------------------------------

    /// Zero cost maps to the maximum score of 1.0.
    #[test]
    fn score_at_zero_cost_is_one() {
        assert_eq!(cost_to_score(0.0), 1.0_f32);
    }

    /// Score is strictly positive for any finite non-negative cost within the
    /// operationally relevant range. For extremely large costs the `f64→f32`
    /// cast can flush to zero; such values are outside normal pipeline usage.
    #[test]
    fn score_is_strictly_positive() {
        for cost in [1e-9, 0.5, 1.0, 10.0, 1e6, 1e9] {
            assert!(
                cost_to_score(cost) > 0.0,
                "score should be > 0 for cost = {cost}"
            );
        }
    }

    /// Score is at most 1.0 for any non-negative cost.
    #[test]
    fn score_is_at_most_one() {
        for cost in [0.0, 1e-9, 0.5, 1.0, 10.0, 1e6] {
            assert!(
                cost_to_score(cost) <= 1.0,
                "score should be ≤ 1 for cost = {cost}"
            );
        }
    }

    /// A higher cost always yields a strictly lower score (monotone decreasing).
    #[test]
    fn score_is_monotone_decreasing() {
        let pairs = [(0.0, 1.0), (1.0, 2.0), (2.0, 100.0), (0.5, 0.500_001)];
        for (low, high) in pairs {
            assert!(
                cost_to_score(low) > cost_to_score(high),
                "cost {low} < {high} should give higher score"
            );
        }
    }

    // -----------------------------------------------------------------------
    // TopK unit tests
    // -----------------------------------------------------------------------

    /// A `TopK` with `k = 0` discards every push.
    #[test]
    fn topk_zero_capacity_discards_all() {
        let mut top: TopK<'static> = TopK::new(0);
        top.push(dummy_item(0.9));
        top.push(dummy_item(0.5));
        assert!(top.into_sorted_desc().is_empty());
    }

    /// `threshold()` returns `NEG_INFINITY` while the heap is not yet full.
    #[test]
    fn topk_threshold_neg_infinity_before_full() {
        let mut top: TopK<'static> = TopK::new(3);
        assert_eq!(top.threshold(), f32::NEG_INFINITY);
        top.push(dummy_item(0.7));
        assert_eq!(top.threshold(), f32::NEG_INFINITY);
        top.push(dummy_item(0.5));
        assert_eq!(top.threshold(), f32::NEG_INFINITY);
        // Not full yet — still NEG_INFINITY.
    }

    /// Once the heap is full, `threshold()` equals the minimum kept probability.
    #[test]
    fn topk_threshold_is_min_once_full() {
        let mut top: TopK<'static> = TopK::new(2);
        top.push(dummy_item(0.9));
        top.push(dummy_item(0.4));
        // Heap is now full; min is 0.4.
        assert_eq!(top.threshold(), 0.4_f32);
        // Pushing a better candidate replaces the minimum.
        top.push(dummy_item(0.8));
        assert_eq!(top.threshold(), 0.8_f32);
    }

    /// A candidate equal to the threshold is rejected (strict improvement required).
    #[test]
    fn topk_equal_threshold_is_rejected() {
        let mut top: TopK<'static> = TopK::new(2);
        top.push(dummy_item(0.9));
        top.push(dummy_item(0.5));
        // Threshold is now 0.5; pushing 0.5 again must not change the heap size.
        top.push(dummy_item(0.5));
        let out = top.into_sorted_desc();
        assert_eq!(out.len(), 2);
    }

    /// `into_sorted_desc` returns items ordered from highest to lowest probability.
    #[test]
    fn topk_into_sorted_desc_is_ordered() {
        let mut top: TopK<'static> = TopK::new(4);
        for p in [0.3_f32, 0.9, 0.1, 0.7] {
            top.push(dummy_item(p));
        }
        let out = top.into_sorted_desc();
        let probas: Vec<f32> = out.iter().map(|it| it.proba).collect();
        assert_eq!(probas, vec![0.9, 0.7, 0.3, 0.1]);
    }

    /// The heap retains the k highest-scoring items when more are pushed.
    #[test]
    fn topk_retains_best_k_items() {
        let mut top: TopK<'static> = TopK::new(3);
        for p in [0.2_f32, 0.8, 0.5, 0.95, 0.1, 0.6] {
            top.push(dummy_item(p));
        }
        let out = top.into_sorted_desc();
        let probas: Vec<f32> = out.iter().map(|it| it.proba).collect();
        // Best 3: 0.95, 0.8, 0.6
        assert_eq!(probas, vec![0.95, 0.8, 0.6]);
    }

    // -----------------------------------------------------------------------
    // Property-based tests (proptest)
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// For any non-negative cost the score must lie in `(0.0, 1.0]`.
        #[test]
        fn prop_score_range(cost in 0.0_f64..1e12_f64) {
            let s = cost_to_score(cost);
            prop_assert!(s > 0.0, "score must be > 0, got {s} for cost {cost}");
            prop_assert!(s <= 1.0, "score must be ≤ 1, got {s} for cost {cost}");
        }

        /// For any two non-negative costs the score ordering is the inverse of
        /// the cost ordering: a lower cost always produces a higher (or equal) score.
        #[test]
        fn prop_score_monotone(a in 0.0_f64..1e9_f64, b in 0.0_f64..1e9_f64) {
            let sa = cost_to_score(a);
            let sb = cost_to_score(b);
            if a < b {
                prop_assert!(sa > sb, "cost {a} < {b} but score {sa} ≤ {sb}");
            } else if a > b {
                prop_assert!(sa < sb, "cost {a} > {b} but score {sa} ≥ {sb}");
            }
            // a == b: scores are equal (no assertion needed).
        }

        /// A `TopK<k>` fed with `n` items retains exactly `min(n, k)` items.
        #[test]
        fn prop_topk_size(
            k in 1_usize..=16_usize,
            probas in proptest::collection::vec(0.0_f32..=1.0_f32, 0..=32),
        ) {
            let n = probas.len();
            let mut top: TopK<'static> = TopK::new(k);
            for &p in &probas {
                top.push(dummy_item(p));
            }
            let out = top.into_sorted_desc();
            prop_assert_eq!(out.len(), n.min(k));
        }

        /// A `TopK<k>` retains exactly the `k` highest probabilities (best-first).
        #[test]
        fn prop_topk_retains_best(
            k in 1_usize..=8_usize,
            mut probas in proptest::collection::vec(0.0_f32..=1.0_f32, 1_usize..=32),
        ) {
            let mut top: TopK<'static> = TopK::new(k);
            for &p in &probas {
                top.push(dummy_item(p));
            }
            let out = top.into_sorted_desc();

            // Reference: sort descending and take k.
            probas.sort_by(|a, b| b.total_cmp(a));
            let expected: Vec<f32> = probas.iter().copied().take(k).collect();
            let got: Vec<f32> = out.iter().map(|it| it.proba).collect();

            prop_assert_eq!(&got, &expected);
        }

        /// Given a set of costs, cost-based Top-K must retain the `k` lowest
        /// costs. The test compares the **sorted sets** (ascending) to tolerate
        /// tie-breaking ambiguity when two distinct `f64` costs collapse to the
        /// same `f32` score after the `cost_to_score` mapping.
        #[test]
        fn prop_cost_topk_returns_lowest_costs(
            k in 1_usize..=8_usize,
            mut costs in proptest::collection::vec(0.0_f64..1e6_f64, 1_usize..=32),
        ) {
            let n = costs.len();
            // Reference: k smallest costs in ascending order.
            costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let expected: Vec<f64> = costs.iter().copied().take(k.min(n)).collect();

            // Feed through TopK using the same score mapping as the real function.
            // We iterate in ascending cost order so that, in case of f32 ties, the
            // lower-cost item is pushed first (and the higher-cost one is rejected).
            let mut top: TopK<'static> = TopK::new(k);
            for &c in &costs {
                let score = cost_to_score(c);
                if score <= top.threshold() { continue; }
                top.push(TopKItem { proba: score, to: dummy_item(0.0).to, edge_cost: c });
            }

            // Sort both by ascending cost before comparing (avoids tie-break ordering
            // sensitivity when multiple costs share the same f32 score).
            let mut got: Vec<f64> = top
                .into_sorted_desc()
                .into_iter()
                .map(|it| it.edge_cost)
                .collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());

            prop_assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(expected.iter()) {
                prop_assert_eq!(g.to_bits(), e.to_bits());
            }
        }
    }
}

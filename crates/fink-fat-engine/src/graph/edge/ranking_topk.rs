// -----------------------------------------------------------------------------
// Batched Top-K ML ranking for candidate edges (per-left seed)
// -----------------------------------------------------------------------------
//
// This module implements a tight, allocation-aware routine to rank candidate
// edges for a single left seed (`src`) against a set of right seeds indexed by
// a spatial/time index (`SeedSpatialIndex`).
//
// Problem being solved
// --------------------
// Given one left seed, we can generate many candidate right seeds (potentially
// thousands). We want to:
// - compute edge features for each candidate,
// - run an ONNX classifier to estimate `p(true_edge)`,
// - keep only the Top-K candidates by probability,
// - return a compact list of winners with an associated "edge_cost" suitable
//   for downstream graph solvers.
//
// Performance constraints
// -----------------------
// - ONNX inference is relatively expensive per call; batching is critical.
// - Keeping Top-K must be O(log K) per accepted candidate, with fast rejection.
// - We avoid storing everything (candidates, scores) before selecting Top-K.
//
// Implementation strategy
// -----------------------
// 1) Iterate candidates from `seed_edge_candidates(...)`.
// 2) Build features incrementally and accumulate into batches of size `batch_size`.
// 3) Run ONNX once per batch to get probabilities.
// 4) Maintain a fixed-capacity Top-K container (min-heap) so we only keep the best.
// 5) At the end, emit winners in sorted order.
//
// Notes
// -----
// - This code assumes `EdgeRankingModel::predict_positive_proba` returns one
//   score per input row in the same order.
// - `EdgeFeatures::compute_features` is expected to be deterministic and stable
//   (no NaNs/Infs) thanks to `FeatureCore::finite_or_zero` guards upstream.
//

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use smallvec::SmallVec;

use crate::engine_config::edge_config::EdgeConfig;
use crate::graph::edge::edge_features::EdgeFeatures;
use crate::graph::edge::edge_prediction::{EdgeModelError, EdgeRankingModel};
use crate::seeding::seed_node::SeedNode;
use crate::seeding::seed_spatial_index::SeedSpatialIndex;

/// Heap item storing one candidate edge and its ML probability.
///
/// We implement `Ord` so that a `BinaryHeap` behaves like a **min-heap**
/// on `proba` (lowest proba at the top), which is perfect for Top-K pruning.
///
/// Attributes
/// ----------
/// * `proba` – Model score `p(class=1)` for this candidate edge.
/// * `to` – Right-hand seed node (edge head).
/// * `edge_cost` – Additive cost derived from features (used downstream by solvers).
///
/// Notes
/// -----
/// - We store `to` by reference to avoid duplicating seed nodes.
/// - We store `edge_cost` because it is computed from features; computing it
///   later would require recomputing features (expensive).
#[derive(Debug)]
struct TopKItem<'a> {
    proba: f32,
    to: &'a SeedNode,
    edge_cost: f64,
}

impl<'a> PartialEq for TopKItem<'a> {
    /// Equality is based on raw float bits to avoid NaN corner cases.
    ///
    /// Using `to_bits()` makes equality:
    /// - deterministic,
    /// - consistent with total ordering used by `Ord`,
    /// - independent of IEEE NaN comparison semantics.
    fn eq(&self, other: &Self) -> bool {
        self.proba.to_bits() == other.proba.to_bits()
    }
}

impl<'a> Eq for TopKItem<'a> {}

impl<'a> PartialOrd for TopKItem<'a> {
    /// Partial ordering used by `BinaryHeap`.
    ///
    /// We reverse the comparison so that Rust's max-heap `BinaryHeap` behaves
    /// like a min-heap by `proba`:
    /// - the *smallest* probability is at the top,
    /// - which gives an O(1) "current threshold" for Top-K.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.proba.total_cmp(&self.proba))
    }
}

impl<'a> Ord for TopKItem<'a> {
    /// Total ordering for heap operations.
    ///
    /// `total_cmp` provides a total float order (handles NaNs deterministically).
    /// Reversing it turns the heap into a min-heap on `proba`.
    fn cmp(&self, other: &Self) -> Ordering {
        other.proba.total_cmp(&self.proba)
    }
}

/// Fixed-capacity Top-K container (keeps the K highest probabilities).
///
/// Internally uses a min-heap: the smallest of the kept items is always on top,
/// so we can reject new candidates in O(1) if `p <= current_min`.
///
/// Attributes
/// ----------
/// * `k` – Maximum number of items to keep.
/// * `heap` – Min-heap storing the currently kept Top-K items.
///
/// Complexity
/// ----------
/// * Push (when accepted): `O(log K)`
/// * Threshold query: `O(1)`
/// * Fast rejection (when heap full): `O(1)`
///
/// Notes
/// -----
/// - When `k == 0`, the structure keeps nothing.
/// - Capacity is set to `k + 1` (saturated) to reduce reallocations.
#[derive(Debug)]
struct TopK<'a> {
    k: usize,
    heap: BinaryHeap<TopKItem<'a>>,
}

impl<'a> TopK<'a> {
    /// Create an empty Top-K container.
    ///
    /// Arguments
    /// ---------
    /// * `k` – Number of best items to retain.
    ///
    /// Return
    /// ------
    /// A [`TopK`] ready to accept candidates.
    #[inline]
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Current pruning threshold.
    ///
    /// Return
    /// ------
    /// * If the heap is not full yet: `-∞` (keep everything so far).
    /// * Else: the smallest probability among kept items (min of current Top-K).
    ///
    /// Notes
    /// -----
    /// The threshold is the probability a new candidate must exceed to enter
    /// the Top-K set once it is full.
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

    /// Try to insert an item into the Top-K set.
    ///
    /// Behavior
    /// --------
    /// - If `k == 0`: do nothing.
    /// - If heap not full: always insert.
    /// - If heap full:
    ///   - reject immediately if `item.proba <= min_kept`,
    ///   - otherwise replace the current minimum.
    ///
    /// Arguments
    /// ---------
    /// * `item` – Candidate to consider for Top-K retention.
    ///
    /// Notes
    /// -----
    /// This is the core "pruning" operation and is designed to be very cheap.
    #[inline]
    fn push(&mut self, item: TopKItem<'a>) {
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

    /// Consume the Top-K container and return items sorted best-first.
    ///
    /// Return
    /// ------
    /// Vector of items sorted by descending probability (highest first).
    ///
    /// Notes
    /// -----
    /// Popping from the heap does not guarantee sorted order, so we explicitly
    /// sort the resulting vector.
    fn into_sorted_desc(mut self) -> Vec<TopKItem<'a>> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(it) = self.heap.pop() {
            out.push(it);
        }

        // Because we pop from a min-heap, `out` is not necessarily ordered.
        out.sort_by(|a, b| b.proba.total_cmp(&a.proba));
        out
    }
}

/// Flush the current batch through the ONNX model and update the Top-K container.
///
/// This helper is the key batching primitive:
/// - it runs one ONNX inference call for the current `batch_features`,
/// - pairs returned probabilities with the corresponding `to` candidates,
/// - applies fast thresholding and Top-K insertion,
/// - clears the batch vectors for reuse.
///
/// Arguments
/// ---------
/// * `model` – ONNX edge-ranking model (mutable because ORT session run mutates).
/// * `top` – Top-K container to update.
/// * `batch_features` – Feature batch (one row per candidate).
/// * `batch_to` – Candidate right nodes aligned with `batch_features`.
///
/// Return
/// ------
/// * `Ok(())` on success.
/// * `Err(EdgeModelError)` if ONNX inference fails.
///
/// Invariants
/// ----------
/// - `batch_features.len()` must equal `batch_to.len()` in normal usage.
/// - This function drains `batch_to` and clears `batch_features`, so the caller
///   can keep pushing into them without reallocations.
///
/// Notes
/// -----
/// We compute `edge_cost` only for candidates that pass the current threshold.
/// That avoids wasting work when Top-K is already "tight".
#[inline]
fn flush_batch<'a>(
    model: &mut EdgeRankingModel,
    top: &mut TopK<'a>,
    batch_features: &mut Vec<EdgeFeatures>,
    batch_to: &mut Vec<&'a SeedNode>,
) -> Result<(), EdgeModelError> {
    if batch_features.is_empty() {
        // Nothing to do (and ensures batch_to is also empty in normal usage).
        return Ok(());
    }

    // One ONNX call for the whole batch: returns p(class=1) for each row.
    let probas = model.predict_positive_proba(batch_features.as_slice())?;

    // Iterate aligned streams:
    // - candidate seed node (drained),
    // - probability for that candidate,
    // - feature reference (to compute edge_cost if needed).
    for ((right_candidate, proba), edge_features) in batch_to
        .drain(..)
        .zip(probas.into_iter())
        .zip(batch_features.iter())
    {
        // Cheap early reject if this candidate can't enter the current Top-K set.
        if proba <= top.threshold() {
            continue;
        }

        // Keep candidate: store its score and derived cost.
        top.push(TopKItem {
            proba,
            to: right_candidate,
            edge_cost: edge_features.kinematic_log_likelihood_cost(),
        });
    }

    // Important: clear features so we don't re-score them on the next flush.
    batch_features.clear();
    Ok(())
}

/// Rank candidate edges for one left seed and keep only Top-K by ML probability.
///
/// This function is the main public entrypoint for per-left candidate ranking.
/// It returns the best right candidates (with an additive cost) for downstream
/// graph construction/solving.
///
/// Arguments
/// ---------
/// * `src` – Left-hand seed (edge tail).
/// * `right_index` – Spatial/time index for right-hand seeds.
/// * `edge_config` – Edge configuration (candidate search parameters).
/// * `model` – ONNX edge-ranking model (mutable because ORT session run mutates).
/// * `topk` – Number of best candidates to keep.
/// * `batch_size` – ONNX batch size (tradeoff throughput vs latency/memory).
/// * `out` – Output buffer, cleared and filled with `(to, edge_cost)` pairs.
///
/// Return
/// ------
/// * `Ok(())` on success; `out` contains best candidates sorted by descending
///   ML probability (best first), represented as `(to_seed, edge_cost)`.
/// * `Err(EdgeModelError)` if ONNX inference fails.
///
/// Notes
/// -----
/// - This function does not allocate for output in normal usage because it writes
///   into the caller-provided [`SmallVec`].
/// - Feature computation happens for *all* candidates, but `edge_cost` is only
///   computed for candidates that survive the current Top-K threshold.
/// - `batch_size` is clamped to at least 1 to avoid empty batches.
///
/// Correctness assumptions
/// -----------------------
/// - `src.seed_edge_candidates(...)` yields candidates in arbitrary order; ordering
///   does not matter because we keep Top-K.
/// - `EdgeRankingModel::predict_positive_proba` preserves the input row order.
/// - The model score `proba` is comparable across candidates for the same `src`.
pub fn rank_topk_edges_for_left<'a, 'b>(
    src: &'a SeedNode,
    right_index: &'b SeedSpatialIndex<'a, '_>,
    edge_config: &'b EdgeConfig,
    model: &mut EdgeRankingModel,
    topk: usize,
    batch_size: usize,
    out: &mut SmallVec<[(&'a SeedNode, f64); 32]>,
) -> Result<(), EdgeModelError> {
    // Output is provided by caller to avoid allocations in hot paths.
    out.clear();

    // Avoid degenerate batch sizes.
    let batch_size = batch_size.max(1);

    // Fixed-capacity Top-K structure (stores only the best candidates).
    let mut top: TopK<'a> = TopK::new(topk);

    // Batch buffers reused across flushes.
    let mut batch_features: Vec<EdgeFeatures> = Vec::with_capacity(batch_size);
    let mut batch_to: Vec<&'a SeedNode> = Vec::with_capacity(batch_size);

    // Generate candidate right nodes and batch them for ONNX inference.
    for to in src.seed_edge_candidates(right_index, edge_config) {
        // Compute structured features (expensive-ish but pure).
        batch_features.push(EdgeFeatures::compute_features(src, to));

        // Keep pointer to the matching right node (must stay aligned with features).
        batch_to.push(to);

        // Flush when batch is full.
        if batch_features.len() >= batch_size {
            flush_batch(model, &mut top, &mut batch_features, &mut batch_to)?;
        }
    }

    // Final flush for any remaining candidates.
    flush_batch(model, &mut top, &mut batch_features, &mut batch_to)?;

    // Emit winners into output buffer (best probability first).
    //
    // We output (to, edge_cost). If later you also need probabilities, you can
    // store them in the output tuple, or expose a second function returning them.
    out.extend(
        top.into_sorted_desc()
            .into_iter()
            .map(|it| (it.to, it.edge_cost)),
    );

    Ok(())
}

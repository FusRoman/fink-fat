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
#[derive(Debug)]
struct TopKItem<'a> {
    proba: f32,
    to: &'a SeedNode,
    edge_cost: f64,
}

impl<'a> PartialEq for TopKItem<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.proba.to_bits() == other.proba.to_bits()
    }
}

impl<'a> Eq for TopKItem<'a> {}

impl<'a> PartialOrd for TopKItem<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering to turn BinaryHeap into a min-heap by proba.
        // We keep the *smallest* proba at the top.
        Some(other.proba.total_cmp(&self.proba))
    }
}

impl<'a> Ord for TopKItem<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.proba.total_cmp(&self.proba)
    }
}

/// Fixed-capacity Top-K container (keeps the K highest probabilities).
///
/// Internally uses a min-heap: the smallest of the kept items is always on top,
/// so we can reject new candidates in O(1) if `p <= current_min`.
#[derive(Debug)]
struct TopK<'a> {
    k: usize,
    heap: BinaryHeap<TopKItem<'a>>,
}

impl<'a> TopK<'a> {
    #[inline]
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Current pruning threshold:
    /// - if heap not full yet: -inf (keep everything)
    /// - else: smallest proba among kept items
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

    /// Try to insert an item. If the heap is full and the candidate is not better
    /// than the current minimum, it is dropped immediately.
    #[inline]
    fn push(&mut self, item: TopKItem<'a>) {
        if self.k == 0 {
            return;
        }

        if self.heap.len() < self.k {
            self.heap.push(item);
            return;
        }

        // Heap full: compare to current minimum (top of min-heap).
        let min_kept = self
            .heap
            .peek()
            .map(|x| x.proba)
            .unwrap_or(f32::NEG_INFINITY);
        if item.proba <= min_kept {
            // Reject: not in top-k.
            return;
        }

        // Replace the smallest kept item.
        let _ = self.heap.pop();
        self.heap.push(item);
    }

    /// Consume and return items sorted by descending probability (best first).
    fn into_sorted_desc(mut self) -> Vec<TopKItem<'a>> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(it) = self.heap.pop() {
            out.push(it);
        }
        // Because we popped from a min-heap, `out` is not guaranteed sorted.
        out.sort_by(|a, b| b.proba.total_cmp(&a.proba));
        out
    }
}

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

    let probas = model.predict_positive_proba(batch_features.as_slice())?;

    for ((right_candidate, proba), edge_features) in batch_to
        .drain(..)
        .zip(probas.into_iter())
        .zip(batch_features.iter())
    {
        if proba <= top.threshold() {
            continue;
        }

        top.push(TopKItem {
            proba,
            to: right_candidate,
            edge_cost: edge_features.kinematic_log_likelihood_cost(),
        });
    }

    batch_features.clear(); // important: otherwise you'd rescore same features next flush
    Ok(())
}

/// Rank candidate edges for one left seed and keep only Top-K by ML probability.
///
/// Arguments
/// ---------
/// * `src` – Left-hand seed (edge tail).
/// * `right_index` – Spatial index for right-hand seeds.
/// * `edge_config` – Edge configuration (candidate search + predictor params).
/// * `model` – ONNX edge-ranking model (mutable because ORT session run mutates).
/// * `topk` – Number of best candidates to keep.
/// * `batch_size` – ONNX batch size (tradeoff throughput vs latency/memory).
///
/// Return
/// ------
/// * `Ok(Vec<(SeedId, SeedId, f32, EdgeFeatures)>)` – Sorted best edges:
///   `(from_id, to_id, p(class=1), features)` best-first.
/// * `Err(EdgeModelError)` – If ONNX inference fails.
pub fn rank_topk_edges_for_left<'a, 'b>(
    src: &'a SeedNode,
    right_index: &'b SeedSpatialIndex<'a, '_>,
    edge_config: &'b EdgeConfig,
    model: &mut EdgeRankingModel,
    topk: usize,
    batch_size: usize,
    out: &mut SmallVec<[(&'a SeedNode, f64); 32]>,
) -> Result<(), EdgeModelError> {
    out.clear();

    let batch_size = batch_size.max(1);
    let mut top: TopK<'a> = TopK::new(topk);

    let mut batch_features: Vec<EdgeFeatures> = Vec::with_capacity(batch_size);
    let mut batch_to: Vec<&'a SeedNode> = Vec::with_capacity(batch_size);

    for to in src.seed_edge_candidates(right_index, edge_config) {
        batch_features.push(EdgeFeatures::compute_features(src, to));
        batch_to.push(to);

        if batch_features.len() >= batch_size {
            flush_batch(model, &mut top, &mut batch_features, &mut batch_to)?;
        }
    }

    // Final flush.
    flush_batch(model, &mut top, &mut batch_features, &mut batch_to)?;

    out.extend(
        top.into_sorted_desc()
            .into_iter()
            .map(|it| (it.to, it.edge_cost)),
    );
    Ok(())
}

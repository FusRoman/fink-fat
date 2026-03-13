pub mod edge;

use ahash::AHashMap;

use crate::{
    MJDTT,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{Edge, EdgeKey, edge_prediction::EdgeRankingModelPool, error::EdgeBuilderError},
    persistence::edge_journal::edge_op::EdgeOp,
    pipeline::hooks::StageProgress,
    seeding::{SeedKey, SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

#[derive(Debug)]
pub struct AlertLinkageDAG {
    pub out_deg: AHashMap<SeedKey, usize>,
    /// All edges, sorted by [`EdgeKey`] up to index `sorted_len`.
    ///
    /// Edges in `edges[..sorted_len]` are guaranteed to be sorted.
    /// Edges in `edges[sorted_len..]` have been appended but not yet merged
    /// into the sorted prefix.  Call [`AlertLinkageDAG::commit_edges_sort`]
    /// to flush the unsorted tail and restore the full invariant before
    /// using any binary-search operation.
    pub edges: Vec<Edge>,
    /// Length of the sorted prefix of `edges`.
    ///
    /// `binary_search_by_key` operations ([`Self::edge_by_key`],
    /// [`Self::deactivate_edges`]) require `sorted_len == edges.len()`.
    sorted_len: usize,
    /// Pending edge operations accumulated since the last persistence flush.
    ///
    /// Every mutation method (`add_inter_night_edges`, `deactivate_edges`, …)
    /// appends the corresponding [`EdgeOp`] here.
    /// The save stage drains this buffer and writes it as a journal delta.
    pending_ops: Vec<EdgeOp>,
}

impl Default for AlertLinkageDAG {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertLinkageDAG {
    pub fn new() -> Self {
        Self {
            out_deg: AHashMap::new(),
            edges: Vec::new(),
            sorted_len: 0,
            pending_ops: Vec::new(),
        }
    }

    /// Build a DAG from a pre-existing edge set (e.g. loaded from disk).
    ///
    /// The `pending_ops` buffer starts empty because these edges are already
    /// persisted — they do not need to be written again.
    ///
    /// The edge vector is sorted by [`EdgeKey`] so that subsequent lookups
    /// via [`Self::edge_by_key`] can use binary search.
    pub fn from_edges(mut edges: Vec<Edge>) -> Self {
        let mut out_deg = AHashMap::new();

        for edge in edges.iter() {
            *out_deg.entry(edge.from).or_insert(0) += 1;
        }

        edges.sort_unstable_by_key(|e| e.key());
        let n = edges.len();

        Self {
            out_deg,
            edges,
            sorted_len: n,
            pending_ops: Vec::new(),
        }
    }

    // -------------------------------------------------------------------------
    // Pending operations (journal integration)
    // -------------------------------------------------------------------------

    /// Drain all pending edge operations accumulated since the last flush.
    ///
    /// After calling this, the internal buffer is empty. The caller is
    /// responsible for persisting the returned ops via the edge journal.
    pub fn drain_pending_ops(&mut self) -> Vec<EdgeOp> {
        std::mem::take(&mut self.pending_ops)
    }

    /// Number of pending (unflushed) edge operations.
    #[inline]
    pub fn n_pending_ops(&self) -> usize {
        self.pending_ops.len()
    }

    /// Build directed edges between one left night and one right night and
    /// append them to the DAG.
    ///
    /// This is the standard entry point used when no pre-built spatial index is
    /// available.  Internally it delegates to [`Edge::build_edges`], which
    /// constructs a [`crate::seeding::seed_spatial_index::SeedSpatialIndex`]
    /// over `right_nodes` before searching for candidates.
    ///
    /// If the same right night is paired with multiple left nights, prefer
    /// [`Self::add_inter_night_edges_with_index`] to avoid rebuilding the index
    /// on every call.
    ///
    /// Arguments
    /// ---------
    /// * `left_nodes` – Seeds from the earlier (left) night.
    ///   All nodes must belong to the same night.
    /// * `right_nodes` – Seeds from the later (right) night.
    ///   All nodes must belong to the same night and must be **sorted** by
    ///   [`SeedNode`] order (primary key `plane.epoch_mid`).
    /// * `edge_config` – Edge construction parameters:
    ///   candidate search radius, ML toggle, Top-K pruning, parallelism.
    /// * `spatial_binner` – Spatial partitioner used to index `right_nodes`.
    /// * `time_binner_width` – Bin width (days) for the uniform time index
    ///   built over `right_nodes`.
    /// * `model_pool` – Optional ML model pool.
    ///   Required when `edge_config.use_ml_ranking` is `true`.
    /// * `progress_sink` – Progress reporter updated per processed chunk.
    ///
    /// Return
    /// ------
    /// * `Ok(())` – Edges appended; `out_deg` and `pending_ops` updated.
    /// * `Err(EdgeBuilderError)` – Propagated from [`Edge::build_edges`]
    ///   (invalid seed slices, missing model pool, ONNX inference failure).
    ///
    /// Notes
    /// -----
    /// - Newly appended edges are **not** immediately sorted.  Call
    ///   [`Self::commit_edges_sort`] once after all pairs for a given stage
    ///   run to restore the sorted invariant required by binary-search methods.
    /// - Each produced edge is recorded as an [`EdgeOp::Upsert`] in
    ///   `pending_ops` for subsequent journal persistence.
    #[allow(clippy::too_many_arguments)]
    pub fn add_inter_night_edges<B: SpatialBinner>(
        &mut self,
        left_nodes: &[SeedNode],
        right_nodes: &[SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner_width: MJDTT,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn StageProgress,
    ) -> Result<(), EdgeBuilderError> {
        assert!(!left_nodes.is_empty(), "left_nodes must not be empty");
        assert!(!right_nodes.is_empty(), "right_nodes must not be empty");

        debug_assert!(
            left_nodes
                .iter()
                .all(|s| s.night_id() == left_nodes[0].night_id()),
            "left_nodes must all belong to the same night"
        );
        debug_assert!(
            right_nodes
                .iter()
                .all(|s| s.night_id() == right_nodes[0].night_id()),
            "right_nodes must all belong to the same night"
        );

        // Invariant: right_nodes are sorted by epoch_mid (and tie-breakers) already.
        //
        // This is required by generate_topk_edges() / candidate search logic that relies
        // on monotonic epoch ordering.
        debug_assert!(
            right_nodes.windows(2).all(|w| w[0] <= w[1]),
            "right_nodes must be sorted (SeedNode Ord: epoch_mid primary key)"
        );

        let left_night = left_nodes[0].night_id();
        let right_night = right_nodes[0].night_id();
        tracing::debug!(
            %left_night,
            %right_night,
            n_left = left_nodes.len(),
            n_right = right_nodes.len(),
            "add_inter_night_edges",
        );

        let new_edges = Edge::build_edges(
            left_nodes,
            right_nodes,
            edge_config,
            spatial_binner,
            time_binner_width,
            model_pool,
            progress_sink,
        )?;

        let n_new_edges = new_edges.len();
        tracing::debug!(%left_night, %right_night, n_new_edges, "edges built");

        for edge in new_edges {
            *self.out_deg.entry(edge.from).or_insert(0) += 1;

            // Edge is Copy, so both pushes below are cheap bitwise copies
            // — no heap allocation, no .clone() call.
            self.pending_ops.push(EdgeOp::Upsert { edge });
            self.edges.push(edge);
        }
        // Sorting is deferred: call commit_edges_sort() once after all pairs
        // have been processed to maintain the sorted-by-EdgeKey invariant.

        Ok(())
    }

    /// Add inter-night edges using a pre-built right-hand [`SeedSpatialIndex`].
    ///
    /// Like [`Self::add_inter_night_edges`] but skips rebuilding the spatial
    /// index for `right_nodes`. The caller is responsible for building the
    /// index once per right night (before iterating over left nights).
    ///
    /// Arguments
    /// ---------
    /// * `left_nodes`  – Left-hand seeds (older epoch).
    /// * `right_index` – Pre-built spatio-temporal index over right-hand seeds.
    /// * `edge_config` – Edge configuration.
    /// * `model_pool`  – Optional ML model pool.
    /// * `progress_sink` – Progress reporter.
    ///
    /// Return
    /// ------
    /// Same as [`Self::add_inter_night_edges`].
    ///
    /// Notes
    /// -----
    /// This method does **not** sort `edges` after insertion.
    /// Call [`Self::commit_edges_sort`] once after all pairs are processed.
    #[allow(clippy::too_many_arguments)]
    pub fn add_inter_night_edges_with_index<'seed_lf, 'binner_lf>(
        &mut self,
        left_nodes: &[SeedNode],
        right_index: &SeedSpatialIndex<'seed_lf, 'binner_lf>,
        edge_config: &EdgeConfig,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn StageProgress,
    ) -> Result<(), EdgeBuilderError> {
        assert!(!left_nodes.is_empty(), "left_nodes must not be empty");

        debug_assert!(
            left_nodes
                .iter()
                .all(|s| s.night_id() == left_nodes[0].night_id()),
            "left_nodes must all belong to the same night"
        );

        let left_night = left_nodes[0].night_id();
        tracing::debug!(
            %left_night,
            n_left = left_nodes.len(),
            "add_inter_night_edges_with_index",
        );

        let new_edges = Edge::build_edges_with_index(
            left_nodes,
            right_index,
            edge_config,
            model_pool,
            progress_sink,
        )?;

        let n_new_edges = new_edges.len();
        tracing::debug!(%left_night, n_new_edges, "edges built (with_index)");

        for edge in new_edges {
            *self.out_deg.entry(edge.from).or_insert(0) += 1;
            self.pending_ops.push(EdgeOp::Upsert { edge });
            self.edges.push(edge);
        }
        // Sorting is deferred: call commit_edges_sort() once after all pairs.

        Ok(())
    }

    /// Flush all unsorted edges into the sorted prefix.
    ///
    /// After a batch of [`Self::add_inter_night_edges`] or
    /// [`Self::add_inter_night_edges_with_index`] calls, the newly appended
    /// edges are stored unsorted in `edges[sorted_len..]`.  This method:
    ///
    /// 1. Sorts only the new batch — $O(m \log m)$.
    /// 2. Runs a stable sort on the full vector.  Timsort detects the two
    ///    sorted runs (old prefix + new suffix) and merges them in $O(n + m)$.
    ///
    /// Total cost per call: $O(n + m \log m)$ where $n$ is the pre-existing
    /// edge count and $m$ is the number of newly appended edges.
    ///
    /// After the call, `sorted_len == edges.len()` and binary-search
    /// operations are safe again.
    pub fn commit_edges_sort(&mut self) {
        let old_len = self.sorted_len;
        if old_len == self.edges.len() {
            return; // already fully sorted
        }
        self.edges[old_len..].sort_unstable_by_key(|e| e.key());
        self.edges.sort_by_key(|e| e.key());
        self.sorted_len = self.edges.len();
    }

    /// Return a reference to the edge identified by `key`, or `None` if absent.
    ///
    /// Complexity: O(log n) — binary search on the sorted edge vector.
    ///
    /// Arguments
    /// ---------
    /// * `key` – Identity of the edge (source + target seed keys).
    ///
    /// Return
    /// ------
    /// `Some(&Edge)` if found, `None` otherwise.
    #[inline]
    pub fn edge_by_key(&self, key: &EdgeKey) -> Option<&Edge> {
        debug_assert_eq!(
            self.sorted_len,
            self.edges.len(),
            "call commit_edges_sort() before binary-search operations"
        );
        match self.edges.binary_search_by_key(key, |e| e.key()) {
            Ok(idx) => Some(&self.edges[idx]),
            Err(_) => None,
        }
    }

    /// Return a mutable reference to the edge identified by `key`, or `None`.
    ///
    /// Complexity: O(log n) — binary search on the sorted edge vector.
    ///
    /// Arguments
    /// ---------
    /// * `key` – Identity of the edge.
    ///
    /// Return
    /// ------
    /// `Some(&mut Edge)` if found, `None` otherwise.
    #[inline]
    pub fn edge_by_key_mut(&mut self, key: &EdgeKey) -> Option<&mut Edge> {
        debug_assert_eq!(
            self.sorted_len,
            self.edges.len(),
            "call commit_edges_sort() before binary-search operations"
        );
        match self.edges.binary_search_by_key(key, |e| e.key()) {
            Ok(idx) => Some(&mut self.edges[idx]),
            Err(_) => None,
        }
    }

    /// Deactivate a set of edges identified by their keys.
    ///
    /// Each edge in `keys` is looked up via the internal `edge_index`. If
    /// found and currently active, its `active` flag is set to `false` and an
    /// [`EdgeOp::Upsert`] with `active = false` is appended to `pending_ops`
    /// so that the deactivation is persisted by the next [`Self::drain_pending_ops`]
    /// call in the `SaveData` stage.
    ///
    /// Edges that are already inactive or absent from the index are silently
    /// skipped; the deactivation is therefore idempotent.
    ///
    /// Arguments
    /// ---------
    /// * `keys` – Slice of [`EdgeKey`]s to deactivate. Duplicates are handled
    ///   safely (the second occurrence will hit the already-inactive branch
    ///   and be skipped).
    ///
    /// Return
    /// ------
    /// Number of edges that were actually deactivated (transitions from active
    /// to inactive). Already-inactive or missing edges are not counted.
    pub fn deactivate_edges(&mut self, keys: &[EdgeKey]) -> u64 {
        debug_assert_eq!(
            self.sorted_len,
            self.edges.len(),
            "call commit_edges_sort() before binary-search operations"
        );
        let mut n_deactivated: u64 = 0;

        for key in keys {
            let Ok(idx) = self.edges.binary_search_by_key(key, |e| e.key()) else {
                continue;
            };
            if !self.edges[idx].active {
                continue;
            }

            self.edges[idx].active = false;
            self.pending_ops.push(EdgeOp::Upsert {
                edge: self.edges[idx],
            });
            n_deactivated += 1;
        }

        n_deactivated
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::{
        graph::edge::{Edge, EdgeKey},
        night_id::NightId,
        persistence::edge_journal::edge_op::EdgeOp,
        seeding::SeedKey,
    };

    /// Build a minimal `Edge` for testing (no `SeedNode` required).
    fn make_edge(from: (u32, u64), to: (u32, u64)) -> Edge {
        let from_key = SeedKey {
            night_id: NightId(from.0),
            unique_id: from.1,
        };
        let to_key = SeedKey {
            night_id: NightId(to.0),
            unique_id: to.1,
        };
        Edge {
            cost: 1.0,
            dt_days: 1.0,
            active: true,
            from: from_key,
            to: to_key,
        }
    }

    fn edge_key(from: (u32, u64), to: (u32, u64)) -> EdgeKey {
        EdgeKey {
            from: SeedKey {
                night_id: NightId(from.0),
                unique_id: from.1,
            },
            to: SeedKey {
                night_id: NightId(to.0),
                unique_id: to.1,
            },
        }
    }

    /// Active edges in `keys` are deactivated; count matches; untouched
    /// edges remain active.
    #[test]
    fn deactivate_edges_count_and_active_flag() {
        let e0 = make_edge((0, 0), (1, 0));
        let e1 = make_edge((0, 1), (1, 1));
        let e2 = make_edge((0, 2), (1, 2));

        let mut dag = AlertLinkageDAG::from_edges(vec![e0, e1, e2]);
        // `from_edges` starts with an empty pending buffer.
        assert_eq!(dag.n_pending_ops(), 0);

        let keys = vec![edge_key((0, 0), (1, 0)), edge_key((0, 2), (1, 2))];
        let n = dag.deactivate_edges(&keys);

        assert_eq!(n, 2, "two active edges should be deactivated");
        assert!(!dag.edge_by_key(&edge_key((0, 0), (1, 0))).unwrap().active);
        assert!(
            dag.edge_by_key(&edge_key((0, 1), (1, 1))).unwrap().active,
            "untouched edge must remain active"
        );
        assert!(!dag.edge_by_key(&edge_key((0, 2), (1, 2))).unwrap().active);
        assert_eq!(dag.n_pending_ops(), 2, "two Upsert ops appended");
    }

    /// Calling `deactivate_edges` a second time on an already-inactive edge
    /// is idempotent: returns 0 and appends no extra op.
    #[test]
    fn deactivate_edges_idempotent() {
        let e = make_edge((0, 0), (1, 0));
        let mut dag = AlertLinkageDAG::from_edges(vec![e]);
        let key = edge_key((0, 0), (1, 0));

        assert_eq!(dag.deactivate_edges(&[key]), 1);
        assert_eq!(dag.n_pending_ops(), 1);

        // Second call: edge already inactive → no change.
        assert_eq!(dag.deactivate_edges(&[key]), 0);
        assert_eq!(
            dag.n_pending_ops(),
            1,
            "no additional op on already-inactive edge"
        );
    }

    /// Keys absent from the graph are silently ignored.
    #[test]
    fn deactivate_edges_missing_keys_are_ignored() {
        let mut dag = AlertLinkageDAG::new();
        let absent = edge_key((99, 99), (100, 100));
        let n = dag.deactivate_edges(&[absent]);
        assert_eq!(n, 0);
        assert_eq!(dag.n_pending_ops(), 0);
    }

    /// The `EdgeOp` appended to `pending_ops` is an `Upsert` with the correct
    /// key and `active = false`.
    #[test]
    fn deactivate_edges_generates_correct_upsert_ops() {
        let e = make_edge((5, 3), (6, 7));
        let key = edge_key((5, 3), (6, 7));
        let mut dag = AlertLinkageDAG::from_edges(vec![e]);

        dag.deactivate_edges(&[key]);

        let ops = dag.drain_pending_ops();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            EdgeOp::Upsert { edge } => {
                assert_eq!(edge.key(), key);
                assert!(!edge.active, "upserted edge must have active=false");
            }
            other => panic!("expected Upsert, got {other:?}"),
        }
    }
}

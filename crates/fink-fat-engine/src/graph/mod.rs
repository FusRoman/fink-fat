pub mod edge;

use ahash::AHashMap;

use crate::{
    MJDTT,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{Edge, EdgeKey, edge_prediction::EdgeRankingModelPool, error::EdgeBuilderError},
    persistence::edge_journal::edge_op::EdgeOp,
    pipeline::hooks::StageProgress,
    seeding::{SeedKey, SeedNode},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

#[derive(Debug)]
pub struct AlertLinkageDAG {
    pub in_deg: AHashMap<SeedKey, usize>,
    pub out_deg: AHashMap<SeedKey, usize>,
    pub edges: Vec<Edge>,
    /// Reverse index: `EdgeKey` → position in `edges`.
    ///
    /// Maintained automatically by constructors and mutation methods so that
    /// `edge_index[key] == i` iff `edges[i].key() == key`.
    edge_index: AHashMap<EdgeKey, usize>,
    /// Pending edge operations accumulated since the last persistence flush.
    ///
    /// Every mutation method (`add_inter_night_edges`, future `remove_edge`,
    /// `deactivate_edge`, …) appends the corresponding [`EdgeOp`] here.
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
            in_deg: AHashMap::new(),
            out_deg: AHashMap::new(),
            edges: Vec::new(),
            edge_index: AHashMap::new(),
            pending_ops: Vec::new(),
        }
    }

    /// Build a DAG from a pre-existing edge set (e.g. loaded from disk).
    ///
    /// The `pending_ops` buffer starts empty because these edges are already
    /// persisted — they do not need to be written again.
    pub fn from_edges(edges: Vec<Edge>) -> Self {
        let mut in_deg = AHashMap::new();
        let mut out_deg = AHashMap::new();
        let mut edge_index = AHashMap::with_capacity(edges.len());

        for (i, edge) in edges.iter().enumerate() {
            let from = edge.from;
            let to = edge.to;

            *out_deg.entry(from).or_insert(0) += 1;
            *in_deg.entry(to).or_insert(0) += 1;
            edge_index.insert(edge.key(), i);
        }

        Self {
            in_deg,
            out_deg,
            edges,
            edge_index,
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
            let from = edge.from;
            let to = edge.to;

            *self.out_deg.entry(from).or_insert(0) += 1;
            *self.in_deg.entry(to).or_insert(0) += 1;

            let key = edge.key();
            let idx = self.edges.len();
            self.edge_index.insert(key, idx);

            // Track the operation for the journal delta.
            self.pending_ops.push(EdgeOp::Upsert {
                key,
                edge: edge.clone(),
            });

            self.edges.push(edge);
        }

        Ok(())
    }

    /// Return a reference to the edge identified by `key`, or `None` if absent.
    ///
    /// Complexity: O(1) amortized (hash-map lookup).
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
        self.edge_index.get(key).map(|&i| &self.edges[i])
    }

    /// Return a mutable reference to the edge identified by `key`, or `None`.
    ///
    /// Complexity: O(1) amortized.
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
        self.edge_index.get(key).map(|&i| &mut self.edges[i])
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
        let mut n_deactivated: u64 = 0;

        for key in keys {
            let Some(&idx) = self.edge_index.get(key) else {
                continue;
            };
            if !self.edges[idx].active {
                continue;
            }

            self.edges[idx].active = false;
            let deactivated_edge = self.edges[idx].clone();
            self.pending_ops.push(EdgeOp::Upsert {
                key: *key,
                edge: deactivated_edge,
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
mod tests {
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
            EdgeOp::Upsert { key: op_key, edge } => {
                assert_eq!(*op_key, key);
                assert!(!edge.active, "upserted edge must have active=false");
            }
            other => panic!("expected Upsert, got {other:?}"),
        }
    }
}

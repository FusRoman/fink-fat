pub mod edge;

use ahash::AHashMap;

use crate::{
    MJDTT,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{Edge, EdgeKey, edge_prediction::EdgeRankingModelPool, error::EdgeBuilderError},
    persistence::edge_journal::edge_op::EdgeOp,
    pipeline::progress_sink::ProgressSink,
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

    pub fn add_inter_night_edges<B: SpatialBinner>(
        &mut self,
        left_nodes: &[SeedNode],
        right_nodes: &[SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner_width: MJDTT,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn ProgressSink,
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

        let new_edges = Edge::build_edges(
            left_nodes,
            right_nodes,
            edge_config,
            spatial_binner,
            time_binner_width,
            model_pool,
            progress_sink,
        )?;

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
}

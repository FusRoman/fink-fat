//! Directed edge model (hypothesis linking two nodes across nights).

use ahash::AHashMap;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::{node_id::NodeId, score::ScoredEdge},
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

/// Identifier for an edge inside a graph edge array.
///
/// - 0 ≤ id < M where M is the number of edges in the graph.
/// - Permits O(1) access into `Vec<Edge>` via `.idx()`.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Encode,
    Decode,
    PartialOrd,
    Ord,
    Default,
)]
pub struct EdgeId(pub u64);

impl EdgeId {
    /// Return the edge index into a `Vec<Edge>`.
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }

    /// Convenience constructor.
    #[inline]
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeId({})", self.0)
    }
}

impl From<u64> for EdgeId {
    #[inline]
    fn from(value: u64) -> Self {
        EdgeId(value)
    }
}

impl From<usize> for EdgeId {
    #[inline]
    fn from(value: usize) -> Self {
        EdgeId(value as u64)
    }
}

/// Directed link from an older node to a newer node (forward in time).
///
/// Cost
/// ----
/// `cost` is a **dimensionless** score already aggregated upstream:
/// lower is better. Enforce strictly **positive** costs to avoid
/// degeneracies in later solvers.
#[derive(Clone, Debug)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub cost: f64,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f64,
}

impl Edge {
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, cost: f64, dt_days: f64) -> Self {
        assert!(
            cost.is_finite() && cost > 0.0,
            "Edge cost must be finite and > 0."
        );
        assert!(
            dt_days.is_finite() && dt_days > 0.0,
            "dt_days must be finite and > 0."
        );
        Self {
            id,
            from,
            to,
            cost,
            dt_days,
        }
    }

    /* -------------------------- Top-K Edge Generation ------------------------- */

    /// Generate **Top-K scored** edges from `left` to `right` using a prebuilt index.
    ///
    /// Notes
    /// -----
    /// This version returns edges already expressed in graph space (`NodeId`),
    /// avoiding any `(night, seed) -> NodeId` resolution during insertion.
    pub fn generate_topk_edges<B: SpatialBinner>(
        id_start: EdgeId,
        left: &[SeedNode],
        right: &[SeedNode],
        edge_config: &EdgeConfig,
        binner: &B,
        index_right: &SeedSpatialIndex,
        t_right_med: f64,
        left_seed_to_node: &AHashMap<SeedId, NodeId>,
        right_seed_to_node: &AHashMap<SeedId, NodeId>,
        right_id_to_index: &AHashMap<SeedId, usize>,
    ) -> Vec<Self> {
        // Δ revisits (≥ 1)
        let night_left: NightId = left.first().map(|s| s.night_id).unwrap_or_default();
        let night_right: NightId = right.first().map(|s| s.night_id).unwrap_or_default();
        let delta_revisit: u32 = night_right.0.saturating_sub(night_left.0).max(1);

        // 1) generation/scoring Top-K
        let mut edges: Vec<Edge> = Vec::with_capacity(left.len() * edge_config.top_k_per_left);

        let mut next_id = id_start.0;

        for i in left {
            let from_nid = match left_seed_to_node.get(&i.seed_id) {
                Some(&nid) => nid,
                None => continue, // left seed not present in graph layer (should not happen)
            };

            // (a) coarse cone at median time → candidate ids
            let (ra_c, dec_c, r_c) =
                i.predict_cone(t_right_med, binner, &edge_config.predictor_config);
            let cand_iter = SeedSpatialIndex::cone_query(index_right, binner, ra_c, dec_c, r_c);

            // (b) fine score per candidate at its true epoch; early gate & cmax
            let mut scored: Vec<ScoredEdge> = Vec::with_capacity(32);

            for j_id in cand_iter {
                if let Some(&j_idx) = right_id_to_index.get(&j_id) {
                    let j = &right[j_idx];
                    if let Some(se) =
                        ScoredEdge::score(i, j, &edge_config.score_config, delta_revisit)
                    {
                        if let Some(cmax) = edge_config.max_total_edges {
                            if se.cost > cmax as f64 {
                                continue;
                            }
                        }
                        scored.push(se);
                    }
                }
            }

            // (c) Top-K via partial selection instead of full sort
            let k = edge_config.top_k_per_left.min(scored.len());
            if k == 0 {
                continue;
            }

            let (_, _, _) = scored.select_nth_unstable_by(k - 1, |a, b| a.cost.total_cmp(&b.cost));
            scored[..k].sort_by(|a, b| a.cost.total_cmp(&b.cost));
            scored.truncate(k);

            // (d) convert to Edge with ids (NodeId-based)
            for se in scored {
                let to_nid = match right_seed_to_node.get(&se.to) {
                    Some(&nid) => nid,
                    None => continue, // right seed not present in graph layer (should not happen)
                };

                let id = EdgeId(next_id);
                next_id += 1;

                edges.push(Edge::new(id, from_nid, to_nid, se.cost, se.dt_days));
            }
        }

        // 2) Optional global cap using partial selection instead of full sort
        if let Some(max_e) = edge_config.max_total_edges {
            if edges.len() > max_e {
                let (_, _, _) =
                    edges.select_nth_unstable_by(max_e - 1, |a, b| a.cost.total_cmp(&b.cost));
                edges[..max_e].sort_by(|a, b| a.cost.total_cmp(&b.cost));
                edges.truncate(max_e);
            }
        }

        edges
    }
}

//! Directed edge model (hypothesis linking two nodes across nights).

use ahash::AHashMap;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::{edge_id::EdgeId, node_id::NodeId},
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode},
    spacetime_bucket::{spatial_binner::SpatialBinner, time_binner::TimeBinner},
};

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
    /// Whether the edge is currently active (used by solvers / CC exact recompute).
    pub active: bool,
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
            active: true,
        }
    }

    /* -------------------------- Top-K Edge Generation ------------------------- */

    /// Generate **Top-K scored** edges from `left` to `right` using a prebuilt index.
    ///
    /// Notes
    /// -----
    /// This version returns edges already expressed in graph space (`NodeId`),
    /// avoiding any `(night, seed) -> NodeId` resolution during insertion.
    pub fn generate_topk_edges<B: SpatialBinner, T: TimeBinner>(
        id_start: EdgeId,
        left: &[SeedNode],
        right: &[SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner: &T,
        left_seed_to_node: &AHashMap<SeedId, NodeId>,
        right_seed_to_node: &AHashMap<SeedId, NodeId>,
    ) -> Vec<Self> {
        // Δ revisits (≥ 1)
        let night_left: NightId = left.first().map(|s| s.night_id).unwrap_or_default();
        let night_right: NightId = right.first().map(|s| s.night_id).unwrap_or_default();
        let delta_revisit: u32 = night_right.0.saturating_sub(night_left.0).max(1);

        /* ---------- build right SeedId -> index mapping ---------- */
        let right_id_to_index: AHashMap<_, _> = right
            .iter()
            .enumerate()
            .map(|(i, sn)| (sn.seed_id, i))
            .collect();

        // 1) generation/scoring Top-K
        let mut edges: Vec<Edge> = Vec::with_capacity(left.len() * edge_config.top_k_per_left);

        let mut next_id = id_start.0;

        for i in left {
            let from_nid = match left_seed_to_node.get(&i.seed_id) {
                Some(&nid) => nid,
                None => continue, // left seed not present in graph layer (should not happen)
            };

            let mut scored = i.score_edge_candidates(
                right,
                spatial_binner,
                time_binner,
                edge_config,
                delta_revisit,
                &right_id_to_index,
            );

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

pub mod edge_features;
pub mod photometry_features;
pub mod position_features;
pub mod uncertainty_features;
pub mod velocity_features;

pub mod edge_id;
pub mod edge_prediction;
pub mod ranking_topk;
pub mod score;

use std::fmt::{self, Display, Formatter};

use smallvec::SmallVec;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        edge_id::EdgeId,
        edge_prediction::{EdgeModelError, EdgeRankingModel},
        ranking_topk::rank_topk_edges_for_left,
    },
    seeding::{seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
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
pub struct Edge<'a> {
    pub id: EdgeId,
    pub from: &'a SeedNode,
    pub to: &'a SeedNode,
    pub cost: f64,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f64,
    /// Whether the edge is currently active (used by solvers / CC exact recompute).
    pub active: bool,
}

impl<'a> Display for Edge<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Edge {{ id: {}, from: {}, to: {}, Δt: {:.3} d, cost: {:.6}, active: {} }}",
            self.id, self.from.seed_id, self.to.seed_id, self.dt_days, self.cost, self.active,
        )
    }
}

impl<'a, 'b, 'c> Edge<'a> {
    pub fn new(id: EdgeId, from: &'a SeedNode, to: &'a SeedNode, cost: f64, dt_days: f64) -> Self {
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

    /// Generate **Top-K scored directed edges** from a set of left-hand seeds to a
    /// set of right-hand seeds.
    ///
    /// Overview
    /// --------
    /// This routine builds the inter-night graph edges between two consecutive
    /// (or non-consecutive) nights by:
    ///
    /// 1. Scoring candidate edges **independently for each left-hand seed** using
    ///    [`SeedNode::score_edge_candidates`], which applies spatio-temporal
    ///    prefiltering followed by exact kinematic scoring.
    /// 2. Retaining only the **Top-K lowest-cost edges per left seed** using
    ///    partial selection (no full sort).
    /// 3. Converting scored edges into final [`Edge`] objects that directly store
    ///    references to the source and target [`SeedNode`]s.
    /// 4. Optionally enforcing a **global cap** on the total number of edges
    ///    produced across all left-hand seeds.
    ///
    /// The returned edges are immediately usable by downstream graph solvers
    /// (e.g. min-cost flow) without any additional `(night, seed) → node`
    /// resolution step.
    ///
    /// Assumptions
    /// -----------
    /// - All `left` seeds belong to the same night.
    /// - All `right` seeds belong to the same (later) night.
    /// - `right` contains **all seeds referenced by scored edges** (by `SeedId`).
    /// - Costs returned by [`ScoredEdge::score`] are **dimensionless** and
    ///   strictly comparable (lower is better).
    ///
    /// Parameters
    /// ----------
    /// * `id_start` – First [`EdgeId`] to assign; subsequent edges are assigned
    ///   monotonically increasing IDs.
    /// * `left` – Slice of source seeds (earlier night).
    /// * `right` – Slice of target seeds (later night).
    /// * `edge_config` – Configuration controlling:
    ///   - per-left Top-K (`top_k_per_left`),
    ///   - optional global edge cap (`max_total_edges`),
    ///   - scoring and predictor parameters.
    /// * `spatial_binner` – Spatial partitioner used during candidate generation
    ///   (e.g. HEALPix).
    /// * `time_binner` – Time binning strategy used during candidate generation
    ///   (uniform or custom).
    ///
    /// Returns
    /// -------
    /// * `Vec<Edge<'a>>` – Directed edges from `left` to `right`, each carrying:
    ///   - references to the source and target `SeedNode`,
    ///   - the aggregated cost,
    ///   - the inter-night time difference `dt_days`.
    ///
    /// The vector is sorted **only if** a global cap is applied; otherwise the
    /// order reflects per-left Top-K extraction.
    ///
    /// Complexity
    /// ----------
    /// Let:
    /// - `L = left.len()`
    /// - `K = edge_config.top_k_per_left`
    /// - `C_i` be the number of scored candidates generated for left seed `i`
    ///
    /// Per left seed:
    /// - Candidate generation: cost of `score_edge_candidates`
    /// - Top-K selection: `O(C_i)` via `select_nth_unstable`
    ///
    /// Total:
    /// - `O(Σ_i C_i + L · K)` for per-left processing
    /// - Optional global cap: `O(E)` where `E ≤ L · K`
    ///
    /// Memory
    /// ------
    /// - Stores at most `L · K` edges before optional global truncation.
    /// - Uses a temporary `HashMap<SeedId, &SeedNode>` for safe ID → node resolution.
    ///
    /// Notes
    /// -----
    /// - Partial selection is used instead of full sorting to keep runtime linear
    ///   in the number of scored candidates.
    /// - The `SeedId → &SeedNode` map avoids relying on any implicit relationship
    ///   between `SeedId` and slice indices in `right`.
    /// - This function is intentionally sequential; parallelism is expected to be
    ///   handled at a higher level (e.g. across night pairs).
    pub fn generate_topk_edges<B: SpatialBinner, T: TimeBinner>(
        id_start: EdgeId,
        left: &'a [SeedNode],
        right: &'a [SeedNode],
        edge_config: &'c EdgeConfig,
        spatial_binner: &'b B,
        time_binner: &'b T,
        model: &mut EdgeRankingModel,
    ) -> Result<Vec<Self>, EdgeModelError> {
        let top_k = edge_config.top_k_per_left;

        let mut edges = Vec::with_capacity(left.len() * top_k);

        let right_index = SeedSpatialIndex::build(right, spatial_binner, time_binner);

        let mut tmp: SmallVec<[(&SeedNode, f32); 32]> = SmallVec::new();

        for src in left.iter() {
            rank_topk_edges_for_left(
                src,
                &right_index,
                edge_config,
                model,
                top_k,
                edge_config.onnx_batch_size,
                &mut tmp,
            )?;

            for (right_candidate, proba) in tmp.iter() {
                edges.push(Edge::new(
                    id_start,
                    src,
                    *right_candidate,
                    *proba as f64,
                    src.delta_days(right_candidate),
                ));
            }
        }

        Ok(edges)
    }
}

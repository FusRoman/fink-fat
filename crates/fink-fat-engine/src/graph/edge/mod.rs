pub mod edge_id;
pub mod edge_prediction;
pub mod features;
pub mod score;

use std::fmt::{self, Display, Formatter};

use ahash::AHashMap;

use crate::{
    astro_math::trace_2x2,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        edge_id::EdgeId,
        features::{
            EdgeFeatures, EdgePhotometryFeatures, EdgePositionFeatures, EdgeUncertaintyFeatures,
            EdgeVelocityFeatures, FeatureCore,
        },
    },
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
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

impl<'a> Edge<'a> {
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
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner: &T,
    ) -> Vec<Self> {
        // Compute the revisit separation (Δ nights), enforced to be ≥ 1.
        // This is passed downstream to scoring for time-dependent penalties.
        let night_left: NightId = left.first().map(|s| s.night_id).unwrap_or_default();
        let night_right: NightId = right.first().map(|s| s.night_id).unwrap_or_default();
        let delta_revisit: u32 = night_right.0.saturating_sub(night_left.0).max(1);

        let top_k = edge_config.top_k_per_left;
        let max_total = edge_config.max_total_edges;

        // Build once: mapping from SeedId to &SeedNode for the right-hand night.
        //
        // This avoids relying on any implicit `SeedId -> index` correspondence
        // and guarantees safe resolution when converting ScoredEdge → Edge.
        let right_by_id: AHashMap<SeedId, &SeedNode> =
            right.iter().map(|s| (s.seed_id, s)).collect();

        // Upper bound: at most `left.len() * top_k` edges before global truncation.
        let mut edges: Vec<Self> = Vec::with_capacity(left.len() * top_k);
        let mut next_id = id_start.0;

        let right_index = SeedSpatialIndex::build(right, spatial_binner, time_binner);

        // Process each left-hand seed independently.
        for src in left {
            // Generate and score all candidate edges from this source seed.
            let mut scored =
                src.score_edge_candidates(right, &right_index, edge_config, delta_revisit);

            // Retain only the Top-K lowest-cost edges for this source.
            let k = top_k.min(scored.len());
            if k == 0 {
                continue;
            }

            // Partial selection: ensures the K smallest-cost elements are in [..k).
            let _ = scored.select_nth_unstable_by(k - 1, |a, b| a.cost.total_cmp(&b.cost));
            scored[..k].sort_by(|a, b| a.cost.total_cmp(&b.cost));
            scored.truncate(k);

            // Convert scored edges into final graph edges.
            edges.extend(scored.into_iter().map(|se| {
                let id = EdgeId(next_id);
                next_id += 1;

                // Resolve the target seed safely via SeedId.
                let to = *right_by_id
                    .get(&se.to)
                    .expect("ScoredEdge::to must refer to a SeedNode in `right`");

                Edge::new(id, src, to, se.cost, se.dt_days)
            }));
        }

        // Optional global cap on total number of edges.
        // Applied after per-left Top-K extraction.
        if let Some(max_e) = max_total {
            if edges.len() > max_e {
                let _ = edges.select_nth_unstable_by(max_e - 1, |a, b| a.cost.total_cmp(&b.cost));
                edges[..max_e].sort_by(|a, b| a.cost.total_cmp(&b.cost));
                edges.truncate(max_e);
            }
        }

        edges
    }

    // -------------------------------------------------------------------------
    // Feature API (high-level)
    // -------------------------------------------------------------------------

    /// Compute the full cadence-robust feature set.
    #[inline]
    pub fn compute_features(&self) -> EdgeFeatures {
        // Build shared intermediate quantities once.
        let core = FeatureCore::from_edge(self);

        EdgeFeatures {
            position: self.position_features(&core),
            velocity: self.velocity_features(&core),
            uncertainty: self.uncertainty_features(),
            photometry: self.photometry_features(),
        }
    }

    /// Compute only position/innovation features.
    #[inline]
    fn position_features(&self, core: &FeatureCore) -> EdgePositionFeatures {
        EdgePositionFeatures {
            chi2_pos: core.chi2_pos,
            log_chi2_pos: core.log_chi2_pos,
            z_dx: core.z_dx,
            z_dy: core.z_dy,
            z_resid_norm: core.z_resid_norm,
            z_along: core.z_along,
            z_cross: core.z_cross,
            chol_z1: core.chol_z1,
            chol_z2: core.chol_z2,
            chol_z_norm: core.chol_z_norm,
        }
    }

    /// Compute only velocity/kinematic features.
    #[inline]
    fn velocity_features(&self, core: &FeatureCore) -> EdgeVelocityFeatures {
        EdgeVelocityFeatures {
            cos_dtheta_v: core.cos_dtheta_v,
            rel_speed_diff: core.rel_speed_diff,
            innov_speed_ratio: core.innov_speed_ratio,
        }
    }

    /// Compute only uncertainty/quality ratio features.
    #[inline]
    fn uncertainty_features(&self) -> EdgeUncertaintyFeatures {
        let eps = FeatureCore::EPS;

        let cvel_from = self.from.plane.cov_vel;
        let cvel_to = self.to.plane.cov_vel;

        let tr_vel_from = trace_2x2(cvel_from).max(0.0);
        let tr_vel_to = trace_2x2(cvel_to).max(0.0);
        let cov_vel_ratio = FeatureCore::safe_div(tr_vel_to, tr_vel_from + eps);

        EdgeUncertaintyFeatures { cov_vel_ratio }
    }

    /// Compute only photometry features.
    #[inline]
    fn photometry_features(&self) -> EdgePhotometryFeatures {
        let flux_i = self.from.photom.flux_mean as f64;
        let flux_j = self.to.photom.flux_mean as f64;
        let flux_abs_diff = (flux_j - flux_i).abs();

        let sigma_i = self.from.photom.flux_std as f64;
        let sigma_j = self.to.photom.flux_std as f64;

        // Variance floor is intentionally large-ish (in flux units) to avoid
        // exploding z-scores for tiny reported uncertainties.
        let sigma_floor = 1.0_f64;
        let pooled_var = sigma_i * sigma_i + sigma_j * sigma_j + sigma_floor * sigma_floor;

        let z_flux = if pooled_var.is_finite() && pooled_var > 0.0 {
            flux_abs_diff / pooled_var.sqrt()
        } else {
            0.0
        };

        let flux_std_ratio = if sigma_i.is_finite() && sigma_i > 0.0 {
            sigma_j / sigma_i
        } else {
            0.0
        };

        let band_shared = if self.from.photom.shares_any_band(&self.to.photom) {
            1.0
        } else {
            0.0
        };

        EdgePhotometryFeatures {
            z_flux: FeatureCore::finite_or_zero(z_flux),
            flux_std_ratio: FeatureCore::finite_or_zero(flux_std_ratio),
            band_shared,
        }
    }
}

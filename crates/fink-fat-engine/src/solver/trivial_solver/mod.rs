pub mod trivial_config;

use ahash::AHashMap;

use crate::{
    graph::edge::Edge, seeding::seed_node::SeedNode,
    solver::trivial_solver::trivial_config::TrivialSolverConfig,
};

/// A tiny track hypothesis returned by the trivial solver.
///
/// This is intentionally lightweight: it only stores the ordered seed list and a scalar cost.
/// If you have a richer `TrackHypothesis` type in your crate, map into it at the call site.
#[derive(Clone, Debug)]
pub struct TrivialTrack<'seed_lf, 'alert_lf> {
    /// Ordered seeds (in time / night order in practice).
    pub nodes: Vec<&'seed_lf SeedNode<'alert_lf>>,
    /// Sum of selected edge costs along the chain.
    pub cost: f64,
}

/// Output of the trivial solver on one component.
#[derive(Clone, Debug, Default)]
pub struct TrivialSolverOutput<'seed_lf, 'alert_lf> {
    pub tracks: Vec<TrivialTrack<'seed_lf, 'alert_lf>>,
    /// Optional: indices of edges (within the provided edge slice) suggested for deactivation.
    /// Kept empty by default (safe).
    pub proposed_deactivations: Vec<usize>,
}

/// Trivial solver: fast path for very small / near-chain components.
///
/// This solver is meant to be *safe* and *cheap*:
/// - It never explores exponential combinations.
/// - It prefers high-confidence, low-cost chains.
/// - It is conservative (does not aggressively deactivate edges).
#[derive(Clone, Debug, Default)]
pub struct TrivialSolver {
    pub cfg: TrivialSolverConfig,
}

impl TrivialSolver {
    pub fn new(cfg: TrivialSolverConfig) -> Self {
        Self { cfg }
    }

    /// Solve one connected component.
    ///
    /// Parameters
    /// ----------
    /// * `edges` – Global edge slice (or the component-local edge slice).
    /// * `component_nodes` – The list of nodes (seeds) in the component.
    /// * `active_only` – If true, ignore inactive edges.
    ///
    /// Notes
    /// -----
    /// This function only uses edges that have BOTH endpoints in `component_nodes`.
    pub fn solve_component<'seed_lf, 'alert_lf>(
        &self,
        edges: &[Edge<'seed_lf, 'alert_lf>],
        component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
        active_only: bool,
    ) -> TrivialSolverOutput<'seed_lf, 'alert_lf> {
        let cfg = &self.cfg;

        if component_nodes.len() < cfg.min_nodes {
            return TrivialSolverOutput::default();
        }

        // ---------------------------------------------------------------------
        // Build membership map
        // ---------------------------------------------------------------------
        let mut in_comp: AHashMap<_, u32> = AHashMap::default();
        for (i, &n) in component_nodes.iter().enumerate() {
            in_comp.insert(n.core.key, i as u32);
        }

        // ---------------------------------------------------------------------
        // Collect internal edges
        // ---------------------------------------------------------------------
        #[derive(Copy, Clone, Debug)]
        struct E {
            src: u32,
            dst: u32,
            cost: f64,
            edge_idx: usize,
        }

        let mut internal: Vec<E> = Vec::new();
        internal.reserve(64);

        for (ei, e) in edges.iter().enumerate() {
            if active_only && !e.core.active {
                continue;
            }
            let Some(&src) = in_comp.get(&e.from.core.key) else {
                continue;
            };
            let Some(&dst) = in_comp.get(&e.to.core.key) else {
                continue;
            };
            if src == dst {
                continue;
            }

            internal.push(E {
                src,
                dst,
                cost: e.core.cost,
                edge_idx: ei,
            });
        }

        if internal.is_empty() {
            return TrivialSolverOutput::default();
        }

        // ---------------------------------------------------------------------
        // Best incoming / outgoing
        // ---------------------------------------------------------------------
        let mut best_out: Vec<Option<E>> = vec![None; component_nodes.len()];
        let mut best_in: Vec<Option<E>> = vec![None; component_nodes.len()];

        for &ed in &internal {
            let s = ed.src as usize;
            let t = ed.dst as usize;

            if best_out[s].map(|x| ed.cost < x.cost).unwrap_or(true) {
                best_out[s] = Some(ed);
            }
            if best_in[t].map(|x| ed.cost < x.cost).unwrap_or(true) {
                best_in[t] = Some(ed);
            }
        }

        // ---------------------------------------------------------------------
        // Keep only mutual best edges
        // ---------------------------------------------------------------------
        let mut next: Vec<Option<u32>> = vec![None; component_nodes.len()];
        let mut prev: Vec<Option<u32>> = vec![None; component_nodes.len()];
        let mut chosen_edge_cost: Vec<f64> = vec![0.0; component_nodes.len()];
        let mut chosen_edge_idx: Vec<Option<usize>> = vec![None; component_nodes.len()];

        for s in 0..component_nodes.len() {
            let Some(eo) = best_out[s] else { continue };
            let t = eo.dst as usize;

            let Some(ei) = best_in[t] else { continue };
            if ei.src == eo.src && ei.dst == eo.dst {
                next[s] = Some(eo.dst);
                prev[t] = Some(eo.src);
                chosen_edge_cost[s] = eo.cost;
                chosen_edge_idx[s] = Some(eo.edge_idx);
            }
        }

        // ---------------------------------------------------------------------
        // Build tracks
        // ---------------------------------------------------------------------
        let mut visited = vec![false; component_nodes.len()];
        let mut tracks: Vec<TrivialTrack<'seed_lf, 'alert_lf>> = Vec::new();

        // Prefer starts (no incoming)
        for i in 0..component_nodes.len() {
            if prev[i].is_none() && next[i].is_some() {
                let (nodes, cost) =
                    build_from(i, component_nodes, &next, &chosen_edge_cost, &mut visited);
                if nodes.len() >= cfg.min_nodes {
                    tracks.push(TrivialTrack { nodes, cost });
                }
            }
        }

        // Catch remaining cycles / chains
        for i in 0..component_nodes.len() {
            if !visited[i] && next[i].is_some() {
                let (nodes, cost) =
                    build_from(i, component_nodes, &next, &chosen_edge_cost, &mut visited);
                if nodes.len() >= cfg.min_nodes {
                    tracks.push(TrivialTrack { nodes, cost });
                }
            }
        }

        // ---------------------------------------------------------------------
        // Final selection
        // ---------------------------------------------------------------------
        tracks.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if tracks.len() > cfg.max_tracks {
            tracks.truncate(cfg.max_tracks);
        }

        TrivialSolverOutput {
            tracks,
            proposed_deactivations: Vec::new(),
        }
    }
}

        // ---------------------------------------------------------------------
        // Helper function (IMPORTANT: not a closure)
        // ---------------------------------------------------------------------
        fn build_from<'seed_lf, 'alert_lf>(
            start: usize,
            component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
            next: &[Option<u32>],
            chosen_edge_cost: &[f64],
            visited: &mut [bool],
        ) -> (Vec<&'seed_lf SeedNode<'alert_lf>>, f64) {
            let mut nodes = Vec::new();
            let mut cost = 0.0;

            let mut cur = start as u32;
            while !visited[cur as usize] {
                visited[cur as usize] = true;
                nodes.push(component_nodes[cur as usize]);

                let s = cur as usize;
                if let Some(nxt) = next[s] {
                    cost += chosen_edge_cost[s];
                    cur = nxt;
                } else {
                    break;
                }
            }

            (nodes, cost)
        }
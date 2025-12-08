//! Connected components (undirected view) and simple diagnostics.

use std::collections::VecDeque;

use ahash::AHashSet;

use crate::{
    graph::{graph::InterNightGraph, NodeId},
    solver::{
        blob_breaker::solve_blob_breaker, dp_small::solve_trivial_dp, min_cost_flow::solve_mcf_ssp,
    },
    NightId,
};

/// Result of a connected component discovery.
#[derive(Clone, Debug)]
pub struct Component {
    /// Node ids belonging to the component.
    pub nodes: Vec<NodeId>,
    /// Edge count inside the component (directed edges whose endpoints are both inside).
    pub edges: usize,
    /// Nights involved (distinct `night` values).
    pub nights: Vec<NightId>,
}

/// Result of solving a component into vertex-disjoint paths.
#[derive(Clone, Debug)]
pub struct ComponentSolution {
    /// Paths as sequences of node ids (time-ordered).
    pub paths: Vec<Vec<NodeId>>,
    /// Sum of edge costs across all paths.
    pub total_cost: f32,
    /// Number of input nodes covered by the solution.
    pub covered_nodes: usize,
}

impl ComponentSolution {
    pub fn empty() -> Self {
        Self {
            paths: Vec::new(),
            total_cost: 0.0,
            covered_nodes: 0,
        }
    }
}

impl Component {
    /// Heuristic: a **trivial chain** has all nodes with `in_deg ≤ 1` and `out_deg ≤ 1`,
    /// and the component is a disjoint union of **simple paths** (no forks/merges).
    pub fn is_trivial_chain(&self, g: &InterNightGraph) -> bool {
        for &v in &self.nodes {
            if g.in_adj[v as usize].len() > 1 {
                return false;
            }
            if g.out_adj[v as usize].len() > 1 {
                return false;
            }
        }
        true
    }

    /// Count distinct nights.
    pub fn night_count(&self) -> usize {
        self.nights.len()
    }

    /// Categorize for downstream solvers:
    /// - `Trivial`: direct path extraction
    /// - `TwoNights`: bipartite-ready
    /// - `MultiNight`: min-cost-flow-ready
    pub fn category(&self, g: &InterNightGraph) -> ComponentCategory {
        if self.is_trivial_chain(g) {
            ComponentCategory::Trivial
        } else if self.night_count() == 2 {
            ComponentCategory::TwoNights
        } else {
            ComponentCategory::MultiNight
        }
    }

    /// Solve tiny / path-like component using **DP peeling**.
    ///
    /// Parameters
    /// ----------
    /// * `g` – Inter-night graph.
    /// * `min_obs` – Minimal nodes per path (≥ 3 recommended).
    pub fn solve_trivial_dp(&self, g: &InterNightGraph, min_obs: usize) -> ComponentSolution {
        let raw = solve_trivial_dp(g, &self.nodes, min_obs);
        let covered = raw.iter().map(|(p, _)| p.len()).sum();
        let total_cost = raw.iter().map(|(_, c)| *c).sum();
        let paths = raw.into_iter().map(|(p, _)| p).collect();
        ComponentSolution {
            paths,
            total_cost,
            covered_nodes: covered,
        }
    }

    /// Solve multi-night component using **Simplified SSP** (MCF-like).
    ///
    /// Parameters
    /// ----------
    /// * `g` – Inter-night graph.
    /// * `min_obs` – Minimal nodes per path (≥ 3 recommended).
    pub fn solve_multinight_mcf(&self, g: &InterNightGraph, min_obs: usize) -> ComponentSolution {
        let raw = solve_mcf_ssp(g, &self.nodes, min_obs);
        let covered = raw.iter().map(|(p, _)| p.len()).sum();
        let total_cost = raw.iter().map(|(_, c)| *c).sum();
        let paths = raw.into_iter().map(|(p, _)| p).collect();
        ComponentSolution {
            paths,
            total_cost,
            covered_nodes: covered,
        }
    }

    /// Solve huge (blob) component using **partition + peeling**.
    ///
    /// Parameters
    /// ----------
    /// * `g` – Inter-night graph.
    /// * `min_obs` – Minimal nodes per path.
    /// * `max_in_per_right` – Cap for incoming edges per right node (anti-hub).
    /// * `window_nights` – Restrict to the N most-recent nights.
    pub fn solve_blob_breaker(
        &self,
        g: &InterNightGraph,
        min_obs: usize,
        max_in_per_right: usize,
        window_nights: usize,
    ) -> ComponentSolution {
        let raw = solve_blob_breaker(g, &self.nodes, min_obs, max_in_per_right, window_nights);
        let covered = raw.iter().map(|(p, _)| p.len()).sum();
        let total_cost = raw.iter().map(|(_, c)| *c).sum();
        let paths = raw.into_iter().map(|(p, _)| p).collect();
        ComponentSolution {
            paths,
            total_cost,
            covered_nodes: covered,
        }
    }

    /// High-level dispatcher selecting a solver based on **component size**.
    ///
    /// Policy
    /// ------
    /// * `nodes <= thr_tiny` and `self.category() == Trivial` → DP peeling.
    /// * `night_count == 2` and `nodes <= thr_bipartite` → SSP (fast).
    /// * `nodes >= thr_blob` or `edges/nodes >= deg_blob` → blob breaker.
    /// * otherwise → SSP.
    pub fn solve_auto(
        &self,
        g: &InterNightGraph,
        min_obs: usize,
        thr_tiny: usize,
        thr_bipartite: usize,
        thr_blob: usize,
        deg_blob: f32,
        max_in_per_right: usize,
        window_nights: usize,
    ) -> ComponentSolution {
        let n = self.nodes.len();
        let dense = if n > 0 {
            self.edges as f32 / n as f32
        } else {
            0.0
        };
        match self.category(&g) {
            ComponentCategory::Trivial if n <= thr_tiny => self.solve_trivial_dp(g, min_obs),
            ComponentCategory::TwoNights if n <= thr_bipartite => {
                self.solve_multinight_mcf(g, min_obs)
            }
            _ if n >= thr_blob || dense >= deg_blob => {
                self.solve_blob_breaker(g, min_obs, max_in_per_right, window_nights)
            }
            _ => self.solve_multinight_mcf(g, min_obs),
        }
    }
}

/// Simple categorization for solver selection (not enforced here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentCategory {
    Trivial,
    TwoNights,
    MultiNight,
}

/// Compute connected components on the **undirected** projection of the graph.
///
/// Complexity
/// ----------
/// Linear in `|V| + |E|`.
pub fn connected_components(g: &InterNightGraph) -> Vec<Component> {
    let n = g.nodes.len();
    let mut seen = vec![false; n];
    let mut comps = Vec::new();

    for start in 0..n {
        if seen[start] {
            continue;
        }

        let mut q = VecDeque::new();
        let mut nodes = Vec::new();
        let mut node_set = AHashSet::default();

        seen[start] = true;
        q.push_back(start as u32);

        while let Some(v) = q.pop_front() {
            nodes.push(v);
            node_set.insert(v);

            // Undirected neighbors: out ∪ in
            for &eid in &g.out_adj[v as usize] {
                let u = g.edges[eid as usize].to;
                if !seen[u as usize] {
                    seen[u as usize] = true;
                    q.push_back(u);
                }
            }
            for &eid in &g.in_adj[v as usize] {
                let u = g.edges[eid as usize].from;
                if !seen[u as usize] {
                    seen[u as usize] = true;
                    q.push_back(u);
                }
            }
        }

        // Count internal edges
        let mut edges = 0usize;
        for &v in &nodes {
            for &eid in &g.out_adj[v as usize] {
                let u = g.edges[eid as usize].to;
                if node_set.contains(&u) {
                    edges += 1;
                }
            }
        }

        // Distinct nights
        let mut nights: Vec<NightId> = nodes.iter().map(|&v| g.nodes[v as usize].night).collect();
        nights.sort_unstable();
        nights.dedup();

        comps.push(Component {
            nodes,
            edges,
            nights,
        });
    }

    comps
}

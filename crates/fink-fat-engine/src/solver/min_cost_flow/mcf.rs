//! Generic min-cost max-flow (MCMF) implementation for small, integral-capacity networks.
//!
//! # Overview
//! This module implements a classic **successive shortest augmenting path (SSAP)**
//! algorithm with **Johnson potentials** to compute a *min-cost max-flow* in a
//! directed graph with residual capacities.
//!
//! In the context of the `min_cost_flow` solver, this MCMF routine is used as a
//! backend to solve a **bipartite assignment** problem (matching with optional
//! breaks). The formulation uses:
//! - integral capacities (mostly 0/1),
//! - floating-point costs (`f64`) coming from edge scoring.
//!
//! # Algorithm
//! At each iteration, SSAP sends flow along the current cheapest augmenting path
//! from `source` to `sink` in the **residual graph**:
//! 1. Run Dijkstra on **reduced costs** `c'(u,v) = c(u,v) + pot[u] - pot[v]`
//!     (non-negative when potentials are valid).
//! 2. Update potentials: `pot[v] += dist[v]` for all reachable vertices.
//! 3. Augment along the found path by the bottleneck residual capacity.
//!
//! This repeats until either:
//! - `max_flow` units have been sent, or
//! - no augmenting path remains (sink unreachable).
//!
//! # Numerical considerations
//! Costs are `f64` because upstream scoring is floating-point. This is usually
//! fine for deterministic pipelines *as long as*:
//! - costs are reasonably scaled (avoid extreme magnitudes),
//! - tie situations are rare or handled upstream,
//! - and platform differences in floating-point comparisons are acceptable.
//!
//! If strict determinism across platforms becomes critical, consider
//! representing costs as fixed-point integers (e.g., scaled `i64`) and switching
//! to integer costs.
//!
//! # Visibility
//! Types are `pub(super)` because this implementation is intended to be an
//! internal utility for the parent `min_cost_flow` module.
//!
//! See also
//! --------
//! - `assignment::build_network` for the bipartite network built on top of this.
//! - `assignment::extract_matching` for reading saturated edges after solving.

use std::{cmp::Ordering, collections::BinaryHeap};

/// One directed residual edge in the adjacency list representation.
///
/// This is the standard residual-network representation used in MCMF:
/// for every forward edge `u -> v` we also store a reverse edge `v -> u`.
///
/// Fields
/// ------
/// to : usize
///     Destination vertex index.
/// rev : usize
///     Index of the reverse edge in `g[to]` (so we can update residual capacities
///     in O(1) during augmentation).
/// cap : i32
///     Residual capacity (integral). In the assignment use-case, this is mostly
///     0 or 1.
/// cost : f64
///     Per-unit flow cost on this residual edge. Reverse edges carry `-cost`.
#[derive(Clone, Debug)]
pub(super) struct ResidualEdge {
    /// Destination vertex.
    pub(super) to: usize,
    /// Index of the reverse edge in `g[to]`.
    pub(super) rev: usize,
    /// Residual capacity (integer).
    pub(super) cap: i32,
    /// Per-unit cost.
    pub(super) cost: f64,
}

/// Residual network for min-cost max-flow computations.
///
/// The graph is represented as an adjacency list `g`, where `g[u]` is a list of
/// residual edges leaving vertex `u`.
///
/// Notes
/// -----
/// - This implementation assumes integral capacities and uses Dijkstra with
///   potentials for efficiency on graphs with non-negative reduced costs.
/// - For the assignment formulation, graphs are typically small-to-medium and
///   sparse, making this representation adequate.
#[derive(Clone, Debug, Default)]
pub(super) struct MinCostFlow {
    /// Adjacency list of residual edges.
    pub(super) g: Vec<Vec<ResidualEdge>>,
}

impl MinCostFlow {
    /// Create an empty residual network with `n_vertices` vertices.
    ///
    /// Parameters
    /// ----------
    /// n_vertices : usize
    ///     Number of vertices in the network.
    ///
    /// Returns
    /// -------
    /// MinCostFlow
    ///     A graph with `n_vertices` empty adjacency lists.
    pub(super) fn new(n_vertices: usize) -> Self {
        Self {
            g: vec![Vec::new(); n_vertices],
        }
    }

    /// Add a directed edge `from -> to` with residual capacity `cap` and per-unit cost `cost`.
    ///
    /// This also inserts the corresponding reverse residual edge `to -> from`:
    /// - capacity = 0
    /// - cost = `-cost`
    ///
    /// Parameters
    /// ----------
    /// from : usize
    ///     Source vertex index.
    /// to : usize
    ///     Destination vertex index.
    /// cap : i32
    ///     Initial capacity on the forward edge (integral).
    /// cost : f64
    ///     Per-unit cost on the forward edge.
    ///
    /// Notes
    /// -----
    /// The `rev` indices are set so that during augmentation we can update:
    /// - forward edge capacity `cap -= df`,
    /// - reverse edge capacity `cap += df`,
    /// in O(1) time.
    pub(super) fn add_edge(&mut self, from: usize, to: usize, cap: i32, cost: f64) {
        let rev_from = self.g[to].len();
        let rev_to = self.g[from].len();

        self.g[from].push(ResidualEdge {
            to,
            rev: rev_from,
            cap,
            cost,
        });

        self.g[to].push(ResidualEdge {
            to: from,
            rev: rev_to,
            cap: 0,
            cost: -cost,
        });
    }

    /// Compute the min-cost max-flow from `source` to `sink`.
    ///
    /// The routine attempts to send up to `max_flow` units. It returns the amount
    /// of flow actually sent and the corresponding minimum total cost.
    ///
    /// Parameters
    /// ----------
    /// source : usize
    ///     Source vertex index.
    /// sink : usize
    ///     Sink vertex index.
    /// max_flow : i32
    ///     Maximum number of flow units to send.
    ///
    /// Returns
    /// -------
    /// (i32, f64)
    ///     `(flow_sent, total_cost)` where:
    ///     - `flow_sent` is the total flow successfully pushed to `sink`,
    ///     - `total_cost` is the minimum achievable cost for that amount of flow.
    ///
    /// Notes
    /// -----
    /// - Uses SSAP with potentials:
    ///   - Dijkstra runs on reduced costs to ensure non-negative edges.
    ///   - Potentials are updated after each shortest-path computation.
    /// - If `sink` becomes unreachable in the residual graph, augmentation stops.
    /// - This function mutates the residual network (capacities are updated).
    ///
    /// Complexity
    /// ----------
    /// Let `F` be the flow sent, `E` the number of residual edges, and `V` the
    /// number of vertices. Each augmentation performs one Dijkstra:
    /// `O(E log V)` (binary heap), repeated `F` times in the worst case.
    pub(super) fn min_cost_max_flow(
        &mut self,
        source: usize,
        sink: usize,
        max_flow: i32,
    ) -> (i32, f64) {
        let n_vertices = self.g.len();

        let mut total_flow_sent = 0i32;
        let mut total_cost = 0.0;

        // Johnson potentials: ensure reduced costs are non-negative for Dijkstra.
        let mut potential = vec![0.0f64; n_vertices];

        // Working arrays reused across augmentations.
        let mut dist = vec![0.0f64; n_vertices];
        let mut prev_vertex = vec![0usize; n_vertices];
        let mut prev_edge_index = vec![0usize; n_vertices];

        while total_flow_sent < max_flow {
            // 1) Shortest path in reduced costs.
            dist.fill(f64::INFINITY);
            dist[source] = 0.0;

            let mut pq = BinaryHeap::new();
            pq.push(State { v: source, d: 0.0 });

            while let Some(State { v, d }) = pq.pop() {
                if d > dist[v] {
                    continue;
                }

                for (edge_index, edge) in self.g[v].iter().enumerate() {
                    if edge.cap <= 0 {
                        continue;
                    }

                    // Reduced cost: c'(u,v) = c(u,v) + pot[u] - pot[v]
                    let next_dist = dist[v] + edge.cost + potential[v] - potential[edge.to];

                    if next_dist < dist[edge.to] {
                        dist[edge.to] = next_dist;
                        prev_vertex[edge.to] = v;
                        prev_edge_index[edge.to] = edge_index;
                        pq.push(State {
                            v: edge.to,
                            d: next_dist,
                        });
                    }
                }
            }

            // No augmenting path => stop.
            if !dist[sink].is_finite() {
                break;
            }

            // 2) Update potentials (only for reachable nodes).
            for v in 0..n_vertices {
                if dist[v].is_finite() {
                    potential[v] += dist[v];
                }
            }

            // 3) Find bottleneck capacity along the augmenting path.
            let mut add_flow = max_flow - total_flow_sent;
            let mut v = sink;
            while v != source {
                let pv = prev_vertex[v];
                let pe = prev_edge_index[v];
                add_flow = add_flow.min(self.g[pv][pe].cap);
                v = pv;
            }

            // 4) Apply augmentation along the path and accumulate real (non-reduced) costs.
            v = sink;
            while v != source {
                let pv = prev_vertex[v];
                let pe = prev_edge_index[v];
                let rev = self.g[pv][pe].rev;

                self.g[pv][pe].cap -= add_flow;
                self.g[v][rev].cap += add_flow;

                total_cost += self.g[pv][pe].cost * (add_flow as f64);

                v = pv;
            }

            total_flow_sent += add_flow;
        }

        (total_flow_sent, total_cost)
    }
}

/// Priority-queue state for Dijkstra.
///
/// We store `(vertex, distance)` and implement ordering so that Rust's
/// `BinaryHeap` (a max-heap) behaves as a min-heap on `d`.
#[derive(Copy, Clone, Debug)]
struct State {
    v: usize,
    d: f64,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v && self.d == other.d
    }
}
impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for min-heap behavior in `BinaryHeap`.
        other.d.partial_cmp(&self.d)
    }
}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)
    }
}

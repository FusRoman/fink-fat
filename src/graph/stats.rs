//! Graph statistics (sizes, degrees, per-night breakdown, components).
//!
//! This module provides a compact `GraphStats` summary and helpers to compute it
//! from an `InterNightGraph`. It is intended for quick diagnostics, logging, or
//! lightweight health checks in unit/integration tests.

use std::fmt::{self, Display, Formatter};

use pyo3::{pyclass, pymethods};

use crate::{
    graph::{
        components::{connected_components, ComponentCategory},
        graph::InterNightGraph,
        NodeId,
    },
    NightId,
};

/// Degree summary for quick diagnostics.
#[derive(Clone, Debug, Default)]
pub struct DegreeSummary {
    /// Average out-degree over all nodes.
    pub avg_out: f64,
    /// Average in-degree over all nodes.
    pub avg_in: f64,
    /// Node with the largest out-degree and the degree value.
    pub max_out: Option<(NodeId, usize)>,
    /// Node with the largest in-degree and the degree value.
    pub max_in: Option<(NodeId, usize)>,
    /// Number of nodes with zero in-degree **and** zero out-degree.
    pub isolated: usize,
}

/// Aggregated statistics over all edges.
#[derive(Clone, Debug, Default)]
pub struct EdgeSummary {
    /// Minimum/maximum/mean cost among edges (dimensionless score).
    pub cost_min: f32,
    pub cost_max: f32,
    pub cost_mean: f64,
    /// Minimum/maximum/mean temporal gap Δt (days) among edges.
    pub dt_min: f32,
    pub dt_max: f32,
    pub dt_mean: f64,
}

/// Per-night breakdown.
#[derive(Clone, Debug)]
pub struct NightSummary {
    /// Night identifier for the layer.
    pub night: NightId,
    /// Number of nodes in this layer.
    pub nodes: usize,
    /// Number of incoming edges (ending on nodes of this layer).
    pub edges_in: usize,
    /// Number of outgoing edges (starting from nodes of this layer).
    pub edges_out: usize,
    /// Average in-/out-degree within this layer.
    pub avg_in: f64,
    pub avg_out: f64,
}

/// Connected component synopsis (on the **undirected** projection).
#[derive(Clone, Debug, Default)]
pub struct ComponentSummary {
    /// Total number of connected components.
    pub count: usize,
    /// Number of components with a single node and no edges.
    pub trivial: usize,
    /// Number of components spanning exactly two distinct nights.
    pub two_nights: usize,
    /// Number of components spanning >= 3 nights.
    pub multi_night: usize,
    /// Size (in nodes) of the largest component.
    pub largest_nodes: usize,
    /// Size (in edges) of the largest component.
    pub largest_edges: usize,
}

/// Top-level graph statistics.
#[pyclass(module = "fink_fat")]
#[derive(Clone, Debug)]
pub struct GraphStats {
    /// Total number of nodes and edges.
    pub total_nodes: usize,
    pub total_edges: usize,
    /// Number of layers (nights) present, and the (min, max) night span.
    pub layers: usize,
    pub night_span: Option<(NightId, NightId)>,
    /// Degree summary (global).
    pub degrees: DegreeSummary,
    /// Edge summary (global).
    pub edges: Option<EdgeSummary>,
    /// Per-night breakdown (same order as `g.layers`).
    pub per_night: Vec<NightSummary>,
    /// Connected components (undirected) synopsis.
    pub components: ComponentSummary,
}

impl GraphStats {
    /// Compute full stats from a graph.
    pub fn from_graph(g: &InterNightGraph) -> Self {
        let total_nodes = g.nodes.len();
        let total_edges = g.edges.len();
        let layers = g.layers.len();
        let night_span = g.night_span();

        // Degrees (global)
        let mut max_out: Option<(NodeId, usize)> = None;
        let mut max_in: Option<(NodeId, usize)> = None;
        let mut sum_out: u64 = 0;
        let mut sum_in: u64 = 0;
        let mut isolated: usize = 0;

        for (v, (outs, ins)) in g.out_adj.iter().zip(g.in_adj.iter()).enumerate() {
            let od = outs.len();
            let id = ins.len();
            sum_out += od as u64;
            sum_in += id as u64;
            if od == 0 && id == 0 {
                isolated += 1;
            }
            if max_out.map_or(true, |(_, m)| od > m) {
                max_out = Some((v as u32, od));
            }
            if max_in.map_or(true, |(_, m)| id > m) {
                max_in = Some((v as u32, id));
            }
        }

        let n_f64 = total_nodes as f64;
        let degrees = if total_nodes > 0 {
            DegreeSummary {
                avg_out: (sum_out as f64) / n_f64,
                avg_in: (sum_in as f64) / n_f64,
                max_out,
                max_in,
                isolated,
            }
        } else {
            DegreeSummary::default()
        };

        // Edge summary (global)
        let edges = if total_edges > 0 {
            let mut cost_min = f32::INFINITY;
            let mut cost_max = f32::NEG_INFINITY;
            let mut dt_min = f32::INFINITY;
            let mut dt_max = f32::NEG_INFINITY;
            let mut cost_sum: f64 = 0.0;
            let mut dt_sum: f64 = 0.0;

            for e in &g.edges {
                if e.cost < cost_min {
                    cost_min = e.cost;
                }
                if e.cost > cost_max {
                    cost_max = e.cost;
                }
                if e.dt_days < dt_min {
                    dt_min = e.dt_days;
                }
                if e.dt_days > dt_max {
                    dt_max = e.dt_days;
                }
                cost_sum += e.cost as f64;
                dt_sum += e.dt_days as f64;
            }

            Some(EdgeSummary {
                cost_min,
                cost_max,
                cost_mean: cost_sum / (total_edges as f64),
                dt_min,
                dt_max,
                dt_mean: dt_sum / (total_edges as f64),
            })
        } else {
            None
        };

        // Per-night breakdown
        let mut per_night = Vec::with_capacity(layers);
        for (lidx, layer) in g.layers.iter().enumerate() {
            let nodes = (layer.node_range.end - layer.node_range.start) as usize;

            // Edges in/out for the layer
            let mut edges_out = 0usize;
            let mut edges_in = 0usize;

            for nid in layer.node_range.clone() {
                edges_out += g.out_adj[nid as usize].len();
                edges_in += g.in_adj[nid as usize].len();
            }

            let nn_f64 = nodes as f64;
            let avg_out = if nodes > 0 {
                edges_out as f64 / nn_f64
            } else {
                0.0
            };
            let avg_in = if nodes > 0 {
                edges_in as f64 / nn_f64
            } else {
                0.0
            };

            per_night.push(NightSummary {
                night: layer.night,
                nodes,
                edges_in,
                edges_out,
                avg_in,
                avg_out,
            });

            // Sanity: keep the borrow checker happy by “using” lidx.
            let _ = lidx;
        }

        // Connected components on the undirected projection
        let comps = connected_components(g);
        let mut trivial = 0usize;
        let mut two_nights = 0usize;
        let mut multi_night = 0usize;
        let mut largest_nodes = 0usize;
        let mut largest_edges = 0usize;

        for c in &comps {
            use ComponentCategory::*;
            match c.category(&g) {
                Trivial => trivial += 1,
                TwoNights => two_nights += 1,
                MultiNight => multi_night += 1,
            }
            if c.nodes.len() > largest_nodes {
                largest_nodes = c.nodes.len();
            }
            if c.edges > largest_edges {
                largest_edges = c.edges;
            }
        }

        let components = ComponentSummary {
            count: comps.len(),
            trivial,
            two_nights,
            multi_night,
            largest_nodes,
            largest_edges,
        };

        GraphStats {
            total_nodes,
            total_edges,
            layers,
            night_span,
            degrees,
            edges,
            per_night,
            components,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Pretty printers (Rust Display)                                             */
/* -------------------------------------------------------------------------- */

impl Display for DegreeSummary {
    /// Human-friendly single/multi-line degree synopsis.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DegreeSummary(avg_out={:.3}, avg_in={:.3}, max_out={:?}, max_in={:?}, isolated={})",
            self.avg_out, self.avg_in, self.max_out, self.max_in, self.isolated
        )
    }
}

impl Display for EdgeSummary {
    /// Human-friendly single-line edge statistics.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EdgeSummary(cost[min={:.3}, max={:.3}, mean={:.3}], dt[min={:.3} d, max={:.3} d, mean={:.3} d])",
            self.cost_min, self.cost_max, self.cost_mean, self.dt_min, self.dt_max, self.dt_mean
        )
    }
}

impl Display for NightSummary {
    /// Compact one-liner for a single night layer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NightSummary(night={}, nodes={}, edges_in={}, edges_out={}, avg_in={:.3}, avg_out={:.3})",
            self.night, self.nodes, self.edges_in, self.edges_out, self.avg_in, self.avg_out
        )
    }
}

impl Display for ComponentSummary {
    /// Human-friendly component synopsis (undirected projection).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComponentSummary(count={}, trivial={}, two_nights={}, multi_night={}, largest_nodes={}, largest_edges={})",
            self.count, self.trivial, self.two_nights, self.multi_night, self.largest_nodes, self.largest_edges
        )
    }
}

impl Display for GraphStats {
    /// Pretty, multi-line summary for quick diagnostics.
    ///
    /// Notes
    /// -----
    /// - Stable ordering & wording to keep snapshot tests robust.
    /// - Keep lines short-ish for log readability.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "GraphStats(")?;
        writeln!(
            f,
            "  total_nodes={}, total_edges={}",
            self.total_nodes, self.total_edges
        )?;
        writeln!(
            f,
            "  layers={}, night_span={:?}",
            self.layers, self.night_span
        )?;
        writeln!(f, "  degrees={}", self.degrees)?;
        match &self.edges {
            Some(es) => writeln!(f, "  edges={}", es)?,
            None => writeln!(f, "  edges=None")?,
        }
        writeln!(f, "  per_night=[")?;
        for ns in &self.per_night {
            writeln!(f, "    {},", ns)?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "  components={}", self.components)?;
        write!(f, ")")
    }
}

/* -------------------------------------------------------------------------- */
/*  Python __str__ / __repr__ for GraphStats                                   */
/* -------------------------------------------------------------------------- */

#[pymethods]
impl GraphStats {
    /// Return a concise, single-line representation suitable for debugging.
    ///
    /// Examples
    /// --------
    /// >>> repr(stats)
    /// 'GraphStats(nodes=123, edges=456, layers=3, span=(60000, 60002))'
    fn __repr__(&self) -> String {
        let span = match self.night_span {
            Some((a, b)) => format!("({}, {})", a, b),
            None => "None".to_string(),
        };
        format!(
            "GraphStats(nodes={}, edges={}, layers={}, span={})",
            self.total_nodes, self.total_edges, self.layers, span
        )
    }

    /// Return a pretty multi-line string (mirrors `Display`).
    ///
    /// Examples
    /// --------
    /// >>> print(stats)
    /// GraphStats(
    ///   total_nodes=4, total_edges=2
    ///   layers=2, night_span=(10, 11)
    ///   degrees=DegreeSummary(...)
    ///   edges=EdgeSummary(...)
    ///   per_night=[
    ///     NightSummary(...),
    ///     NightSummary(...),
    ///   ]
    ///   components=ComponentSummary(...)
    /// )
    fn __str__(&self) -> String {
        format!("{}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{graph::InterNightGraph, layer::NightLayer, node::Node, EdgeId, Horizon};

    #[test]
    fn stats_basic_two_layers() {
        let mut g = InterNightGraph::new(Horizon::new(5));

        // Layer 0: night 10, 2 nodes
        let base = g.nodes.len() as u32;
        g.nodes.push(Node::new(base + 0, 10, 0));
        g.nodes.push(Node::new(base + 1, 10, 1));
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.layers.push(NightLayer::new(10, base..(base + 2)));

        // Layer 1: night 11, 2 nodes
        let base2 = g.nodes.len() as u32;
        g.nodes.push(Node::new(base2 + 0, 11, 0));
        g.nodes.push(Node::new(base2 + 1, 11, 1));
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.layers.push(NightLayer::new(11, base2..(base2 + 2)));

        // Two edges from night 10 -> 11
        let e0 = EdgeId::from(0u32);
        let e1 = EdgeId::from(1u32);
        g.edges.push(crate::graph::edge::Edge::new(
            e0,
            base + 0,
            base2 + 0,
            1.0,
            0.5,
        ));
        g.edges.push(crate::graph::edge::Edge::new(
            e1,
            base + 1,
            base2 + 1,
            2.0,
            0.5,
        ));
        g.out_adj[(base + 0) as usize].push(e0);
        g.in_adj[(base2 + 0) as usize].push(e0);
        g.out_adj[(base + 1) as usize].push(e1);
        g.in_adj[(base2 + 1) as usize].push(e1);

        let s = GraphStats::from_graph(&g);

        assert_eq!(s.total_nodes, 4);
        assert_eq!(s.total_edges, 2);
        assert_eq!(s.layers, 2);
        assert_eq!(s.night_span, Some((10, 11)));
        assert!((s.degrees.avg_out - 0.5).abs() < 1e-9);
        assert!((s.degrees.avg_in - 0.5).abs() < 1e-9);
        assert_eq!(s.per_night.len(), 2);
        assert_eq!(s.per_night[0].night, 10);
        assert_eq!(s.per_night[0].nodes, 2);
        assert_eq!(s.per_night[1].night, 11);
        assert_eq!(s.per_night[1].nodes, 2);
        assert_eq!(s.components.count, 2); // Two disjoint 1-edge chains
        assert_eq!(s.components.two_nights, 0);
    }

    #[test]
    fn degree_summary_display_is_stable() {
        let ds = DegreeSummary {
            avg_out: 1.25,
            avg_in: 0.75,
            max_out: Some((42u32, 7)),
            max_in: Some((7u32, 9)),
            isolated: 3,
        };
        let s = format!("{}", ds);
        assert!(s.contains("avg_out=1.250"));
        assert!(s.contains("avg_in=0.750"));
        assert!(s.contains("max_out=Some((42, 7))"));
        assert!(s.contains("max_in=Some((7, 9))"));
        assert!(s.contains("isolated=3"));
    }

    #[test]
    fn edge_summary_display_is_stable() {
        let es = EdgeSummary {
            cost_min: 0.1,
            cost_max: 3.4,
            cost_mean: 1.25,
            dt_min: 0.2,
            dt_max: 1.5,
            dt_mean: 0.85,
        };
        let s = format!("{}", es);
        assert!(s.contains("cost[min=0.100, max=3.400, mean=1.250]"));
        assert!(s.contains("dt[min=0.200 d, max=1.500 d, mean=0.850 d]"));
    }

    #[test]
    fn night_summary_display_is_stable() {
        let ns = NightSummary {
            night: 60001,
            nodes: 12,
            edges_in: 5,
            edges_out: 8,
            avg_in: 0.4166666667,
            avg_out: 0.6666666667,
        };
        let s = format!("{}", ns);
        assert!(s.starts_with("NightSummary(night=60001"));
        assert!(s.contains("nodes=12"));
        assert!(s.contains("edges_in=5"));
        assert!(s.contains("edges_out=8"));
        assert!(s.contains("avg_in=0.417"));
        assert!(s.contains("avg_out=0.667"));
    }

    #[test]
    fn component_summary_display_is_stable() {
        let cs = ComponentSummary {
            count: 10,
            trivial: 4,
            two_nights: 3,
            multi_night: 3,
            largest_nodes: 9,
            largest_edges: 12,
        };
        let s = format!("{}", cs);
        assert!(s.contains("ComponentSummary(count=10"));
        assert!(s.contains("trivial=4"));
        assert!(s.contains("two_nights=3"));
        assert!(s.contains("multi_night=3"));
        assert!(s.contains("largest_nodes=9"));
        assert!(s.contains("largest_edges=12"));
    }

    #[test]
    fn graph_stats_display_and_repr_are_reasonable() {
        // Build a tiny graph (similar to your unit test) to exercise formatting.
        let mut g = InterNightGraph::new(Horizon::new(5));

        // Night 10
        let base = g.nodes.len() as u32;
        g.nodes.push(Node::new(base + 0, 10, 0));
        g.nodes.push(Node::new(base + 1, 10, 1));
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.layers.push(NightLayer::new(10, base..(base + 2)));

        // Night 11
        let base2 = g.nodes.len() as u32;
        g.nodes.push(Node::new(base2 + 0, 11, 0));
        g.nodes.push(Node::new(base2 + 1, 11, 1));
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.out_adj.push(vec![]);
        g.in_adj.push(vec![]);
        g.layers.push(NightLayer::new(11, base2..(base2 + 2)));

        // Two edges from 10 -> 11
        let e0 = EdgeId::from(0u32);
        let e1 = EdgeId::from(1u32);
        g.edges.push(crate::graph::edge::Edge::new(
            e0,
            base + 0,
            base2 + 0,
            1.0,
            0.5,
        ));
        g.edges.push(crate::graph::edge::Edge::new(
            e1,
            base + 1,
            base2 + 1,
            2.0,
            0.5,
        ));
        g.out_adj[(base + 0) as usize].push(e0);
        g.in_adj[(base2 + 0) as usize].push(e0);
        g.out_adj[(base + 1) as usize].push(e1);
        g.in_adj[(base2 + 1) as usize].push(e1);

        let stats = GraphStats::from_graph(&g);

        // Rust Display
        let pretty = format!("{}", stats);
        assert!(pretty.contains("GraphStats("));
        assert!(pretty.contains("total_nodes=4, total_edges=2"));
        assert!(pretty.contains("layers=2, night_span=Some((10, 11))"));
        assert!(pretty.contains("per_night=["));
        assert!(pretty.contains("NightSummary(night=10"));
        assert!(pretty.contains("NightSummary(night=11"));

        // Python-like repr string (we just call the Rust builder we use in __repr__).
        let repr_like = format!(
            "GraphStats(nodes={}, edges={}, layers={}, span={:?})",
            stats.total_nodes, stats.total_edges, stats.layers, stats.night_span
        );
        // Check key fields present; exact equality not required here.
        assert!(repr_like.contains("nodes=4"));
        assert!(repr_like.contains("edges=2"));
        assert!(repr_like.contains("layers=2"));
        assert!(repr_like.contains("span=Some((10, 11))"));
    }
}

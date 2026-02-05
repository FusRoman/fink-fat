// use crate::graph::{node_id::NodeId};

// /// One trajectory hypothesis produced by a solver.
// ///
// /// A hypothesis is an ordered chain in time.
// /// Storing both nodes and edges makes post-processing easier:
// /// - edges are useful to deactivate selected links,
// /// - nodes are useful for inspection and downstream building blocks.
// #[derive(Clone, Debug)]
// pub struct TrackHypothesis {
//     /// Nodes in strictly increasing time (night / epoch order).
//     pub nodes: Vec<NodeId>,
//     /// Edges used to connect the nodes (typically len = nodes.len() - 1).
//     pub edges: Vec<EdgeId>,

//     /// Additive cost / score returned by the solver (lower is better if cost).
//     pub cost: f64,

//     /// Optional quick metadata (useful for routing / debug).
//     pub night_span: u32,
//     pub n_nodes: u32,
// }

/// Configuration knobs for the trivial solver (beam search).
#[derive(Clone, Debug)]
pub struct TrivialSolverConfig {
    /// Maximum number of tracks returned per component.
    pub max_tracks: usize,

    /// Minimum number of nodes for a returned track.
    pub min_nodes: usize,

    /// Beam width: maximum number of partial hypotheses kept at each expansion step.
    ///
    /// This is the main global "divergence" control knob.
    pub beam_width: usize,

    /// Optional local pruning: keep only the top-K outgoing edges per node (lowest cost).
    ///
    /// Helps if a node has a very large out-degree.
    pub max_out_per_node: usize,

    /// Maximum number of tracks emitted per source node.
    pub max_tracks_per_source: usize,

    /// Global guardrail: maximum number of edge-expansions in one component.
    pub max_expansions: usize,
}

impl Default for TrivialSolverConfig {
    fn default() -> Self {
        Self {
            max_tracks: 16,
            min_nodes: 3,
            beam_width: 64,
            max_out_per_node: 8,
            max_tracks_per_source: 8,
            max_expansions: 50_000,
        }
    }
}

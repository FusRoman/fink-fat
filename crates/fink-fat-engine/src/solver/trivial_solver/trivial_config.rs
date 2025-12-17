/// Configuration knobs for the trivial solver.
#[derive(Clone, Debug)]
pub struct TrivialSolverConfig {
    /// Maximum number of tracks returned per component.
    pub max_tracks: usize,
    /// Minimum number of nodes for a returned track.
    pub min_nodes: usize,
    /// Maximum number of nodes for a returned track.
    pub max_nodes: usize,
    /// If true, suggest immediate deactivation of edges used by returned tracks.
    pub propose_deactivations: bool,
}

impl Default for TrivialSolverConfig {
    fn default() -> Self {
        Self {
            max_tracks: 8,
            min_nodes: 2,
            max_nodes: 8,
            propose_deactivations: false,
        }
    }
}
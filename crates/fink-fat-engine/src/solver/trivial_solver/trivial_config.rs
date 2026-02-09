/// Configuration knobs for the trivial solver.
#[derive(Clone, Debug)]
pub struct TrivialSolverConfig {
    /// Maximum number of tracks returned per component.
    pub max_tracks: usize,
    /// Minimum number of nodes for a returned track.
    pub min_nodes: usize,
}

impl Default for TrivialSolverConfig {
    fn default() -> Self {
        Self {
            max_tracks: 8,
            min_nodes: 2,
        }
    }
}
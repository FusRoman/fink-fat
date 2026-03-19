/// Configuration parameters for the [`MinCostFlowSolver`].
///
/// # Overview
/// This structure groups all **tunable knobs** controlling the behavior of the
/// min-cost-flow path-cover solver. It governs:
/// - how aggressively the solver links nodes into tracks versus splitting them,
/// - how many alternative global solutions are explored,
/// - how short tracks are filtered,
/// - and whether the solver proposes edge deactivations downstream.
///
/// The configuration is intentionally compact and stable, so it can be:
/// - logged verbosely for reproducibility,
/// - serialized if needed,
/// - and compared across runs or experiments.
///
/// # Key trade-offs
/// The most influential parameter is [`break_penalty`]:
/// - a **large** value strongly favors long, continuous tracks,
/// - a **small** value allows frequent breaks, yielding many short hypotheses.
///
/// The second main control is [`n_solutions`], which determines how many
/// *global* alternatives (K-best approximations) are produced for ambiguous
/// components.
///
/// # Invariants
/// - `min_nodes >= 1`
/// - `n_solutions >= 1` (values < 1 are internally clamped)
///
/// These invariants are assumed by the solver logic and are not re-checked
/// at every call site.
///
/// See also
/// --------
/// - [`MinCostFlowSolver`] for how these parameters affect solving.
/// - `k_best::compute_k_solutions` for the semantics of `n_solutions` and
///   `max_alt_attempts`.
#[derive(Clone, Debug)]
pub struct MinCostFlowConfig {
    /// Minimum number of nodes required for a returned track hypothesis.
    ///
    /// Tracks shorter than this threshold are discarded during reconstruction.
    /// This acts as a *post-solve filter* and does **not** affect the optimization
    /// itself.
    ///
    /// Typical values
    /// --------------  
    /// - `2`: keep pairs and longer tracks (default),
    /// - `3+`: enforce stricter trajectory hypotheses for downstream IOD.
    pub min_nodes: usize,

    /// Penalty paid for leaving a node without a predecessor and/or successor.
    ///
    /// This parameter directly controls the balance between:
    /// - linking nodes into longer chains,
    /// - and splitting them into separate track hypotheses.
    ///
    /// Interpretation
    /// --------------
    /// - **Large values** strongly discourage breaks, producing fewer but longer tracks.
    /// - **Small values** allow frequent breaks, producing many short tracks.
    ///
    /// Notes
    /// -----
    /// - The penalty is applied independently on the predecessor and successor
    ///   sides in the assignment formulation.
    /// - The absolute scale of this value should be comparable to typical edge
    ///   costs for meaningful behavior.
    pub break_penalty: f64,

    /// Number of global solutions to compute (K-best approximation).
    ///
    /// - `1` means only the best (minimum-cost) solution is returned.
    /// - Values `> 1` enable exploration of alternative global path covers
    ///   via the "one-edge deviation" strategy.
    ///
    /// Notes
    /// -----
    /// - Solutions are deduplicated using a stable signature.
    /// - Fewer than `n_solutions` may be returned if alternatives are infeasible
    ///   or collapse to identical solutions.
    pub n_solutions: usize,

    /// Maximum number of alternative re-solves when generating K-best solutions.
    ///
    /// Each alternative forbids exactly one arc selected in the best solution.
    /// In the worst case, the number of such arcs can be large; this parameter
    /// caps the number of re-solves to keep runtime bounded.
    ///
    /// Notes
    /// -----
    /// - A good rule of thumb is to keep this value on the order of
    ///   `O(n_nodes)` or smaller.
    /// - If `max_alt_attempts < n_solutions`, fewer than `n_solutions` may be
    ///   produced even if distinct alternatives exist.
    pub max_alt_attempts: usize,

    /// Whether to propose immediate edge deactivations for returned tracks.
    ///
    /// If enabled, the solver outputs a list of edge ids corresponding to the
    /// selected tracks, suggesting that they should be deactivated in the
    /// global graph.
    ///
    /// Notes
    /// -----
    /// - In many pipelines, it is preferable to **delay deactivation** until
    ///   candidate tracks have been validated by orbit determination (IOD).
    /// - When set to `false`, the solver still returns track hypotheses but does
    ///   not suggest any graph mutation.
    pub propose_deactivations: bool,
}

impl Default for MinCostFlowConfig {
    /// Reasonable default configuration for medium-sized components.
    ///
    /// Defaults
    /// --------
    /// - `min_nodes = 2`
    /// - `break_penalty = 1.0`
    /// - `n_solutions = 3`
    /// - `max_alt_attempts = 64`
    /// - `propose_deactivations = false`
    ///
    /// These defaults are conservative and intended to:
    /// - produce clean, short-to-medium tracks,
    /// - expose a small number of global alternatives,
    /// - avoid premature graph mutation.
    fn default() -> Self {
        Self {
            min_nodes: 2,
            break_penalty: 1.0,
            n_solutions: 3,
            max_alt_attempts: 64,
            propose_deactivations: false,
        }
    }
}

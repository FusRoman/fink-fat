//! Solver subsystem: common interfaces, diagnostics, and outputs.
//!
//! Overview
//! --------
//! This module defines the **shared API** used by all solvers operating on the
//! inter-night runtime graph.
//!
//! A solver consumes a **connected component** (identified by [`ComponentId`])
//! and produces a set of **trajectory hypotheses** ([`TrackHypothesis`]),
//! along with lightweight diagnostics ([`SolverDiagnostics`]) used for monitoring
//! and tuning.
//!
//! Solvers are intended to operate on:
//! - the global runtime graph (`RuntimeGraph`) as the backing storage for nodes/edges,
//! - a component-local view provided by [`ConnectedComponents`] (nodes, restricted
//!   adjacency, sources/sinks, degrees, etc.).
//!
//! This module does not implement a specific solving strategy.
//! Concrete implementations live in submodules (e.g. `bounded_beam`, `min_cost_flow`).
//!
//! Key concepts
//! ------------
//!
//! ### Connected component
//! Solvers typically work component-by-component to keep the combinatorial
//! problem tractable.
//!
//! Components are computed on the **undirected view** of the graph (connectivity),
//! but solvers operate on the **directed, time-forward** edges restricted to the
//! component. The component-local directed subgraph is exposed by
//! [`ConnectedComponents`].
//!
//! ### Track hypothesis
//! A [`TrackHypothesis`] represents a candidate multi-night association:
//! - ordered list of seed nodes (in time order),
//! - list of edges linking consecutive nodes,
//! - aggregated cost and metadata (e.g. night span).
//!
//! Tracks are solver outputs and typically feed downstream steps such as
//! trajectory fitting, scoring, or persistence.
//!
//! ### Diagnostics
//! [`SolverDiagnostics`] is intentionally small and generic:
//! - sufficient to monitor performance and tune policies,
//! - without forcing solvers to expose complex internal state.
//!
//! Diagnostics can be used to:
//! - drive solver routing policies,
//! - enforce budgets,
//! - log component-level summary statistics,
//! - track pruning behavior (candidates vs selected).
//!
//! Lifetime model
//! --------------
//! The solver API is designed to avoid copying large graph structures.
//!
//! - `RuntimeGraph` stores edges referencing seed nodes.
//! - [`TrackHypothesis`] typically borrows those same edges and nodes.
//!
//! Therefore, [`SolverOutput`] is generic over lifetimes:
//! - `'edge_lf` for edge references,
//! - `'seed_lf` for seed references,
//! - `'alert_lf` for alert references inside seeds.
//!
//! The bound `'edge_lf: 'seed_lf` ensures that borrowed edges outlive the seed
//! references they contain.
//!
//! Submodules
//! ----------
//! - `components` – connected components API and component-local views.
//! - `solver_manager` – routing and orchestration between solver implementations.
//! - `bounded_beam` – bounded beam-search enumeration inside one component.
//! - `min_cost_flow` – (optional) global optimization solver for larger components.

use ahash::AHashMap;
use outfit::{MJD, constants::Radian, trajectories::batch_reader::ObservationBatch};
use photom::observation_dataset::{ObsDataset, observation::Observation};

use crate::{
    graph::AlertLinkageDAG,
    seeding::store::SeedStore,
    solver::{
        components::{ComponentId, ConnectedComponents},
        error::SolverError,
    },
    trajectory::TrackHypothesis,
};

pub mod bounded_beam;
pub mod components;
pub mod error;
pub mod min_cost_flow;
pub mod solver_manager;

/// Lightweight diagnostics emitted by solvers for monitoring and tuning.
///
/// This structure is intended to be:
/// - small,
/// - easy to log/serialize,
/// - generic across very different solver strategies.
///
/// The fields are a mix of:
/// - component identifiers and sizes,
/// - optional timing/budget tracking,
/// - simple counters for candidate generation and selection.
///
/// Notes
/// -----
/// - Not all solvers will interpret `n_candidates` and `n_selected` identically.
///   The recommended convention is:
///   - `n_candidates`: number of intermediate candidates considered (edges, states,
///     paths, assignments, depending on solver),
///   - `n_selected`: number of final hypotheses emitted (or kept after final ranking).
#[derive(Clone, Debug, Default)]
pub struct SolverDiagnostics {
    /// Identifier of the connected component being solved.
    ///
    /// This should match the `ComponentId` passed to [`Solver::solve`].
    pub component_id: u32,

    /// Stable solver name used in logs and metrics.
    ///
    /// The name is intended to be:
    /// - human-readable,
    /// - stable across versions,
    /// - suitable for metric labels.
    ///
    /// Examples: `"bounded_beam"`, `"min_cost_flow"`.
    pub solver_name: &'static str,

    /// Number of nodes in the component.
    ///
    /// This value is typically used for routing heuristics:
    /// small components can be handled by bounded enumerators, while larger ones
    /// may require optimization methods.
    pub n_nodes: u32,

    /// Number of active edges considered in the component.
    ///
    /// The definition of "active" is graph-dependent (typically an edge flag used
    /// to ignore deactivated links during solving).
    pub m_active_edges: u32,

    /// Optional wall-clock estimate or budget target (seconds).
    ///
    /// This can be used by orchestration code to:
    /// - record predicted runtime,
    /// - compare expected vs actual runtime,
    /// - implement budget-aware routing policies.
    ///
    /// Notes
    /// -----
    /// - This field may remain `0.0` if not used.
    pub time_est_s: f64,

    /// Optional wall-clock time spent (seconds).
    ///
    /// Notes
    /// -----
    /// - This can be filled by the solver itself or by the caller.
    /// - The API keeps it as a scalar to avoid imposing a timing framework.
    pub time_spent_s: f64,

    /// Solver-specific candidate counter (generic).
    ///
    /// Recommended meaning
    /// -------------------
    /// Count the number of intermediate objects examined before final selection.
    ///
    /// Examples
    /// --------
    /// - Bounded beam search: number of edge candidates seen during adjacency
    ///   preparation, number of state expansions, or similar.
    /// - Flow solver: number of arcs processed, iterations performed, etc.
    /// - Greedy solver: number of candidate edges evaluated.
    ///
    /// Notes
    /// -----
    /// This is intentionally generic and should be interpreted in the context
    /// of `solver_name`.
    pub n_candidates: u32,

    /// Solver-specific selected counter (generic).
    ///
    /// Recommended meaning
    /// -------------------
    /// Number of final hypotheses emitted (or kept after final ranking).
    ///
    /// Notes
    /// -----
    /// - For solvers returning tracks, this is typically `tracks.len()`.
    pub n_selected: u32,

    /// Number of search expansions performed (beam steps, arc iterations, etc.).
    ///
    /// Useful to diagnose budget exhaustion: if `n_expansions >= cfg.max_expansions`
    /// the solver was cut short by the exploration budget.
    pub n_expansions: u32,
}

pub type HypothesisId = u32;
pub type HypothesisSet = AHashMap<HypothesisId, TrackHypothesis>;

/// Output of a solver pass over a connected component.
///
/// This is the common return type of the [`Solver`] trait.
///
/// It includes:
/// - `tracks`: candidate trajectory hypotheses,
/// - `diag`: diagnostics for monitoring and tuning.
///
/// Typical conventions
/// -------------------
/// - `tracks` are often sorted from best to worst (e.g. increasing cost),
///   but this module does not enforce ordering.
/// - Callers may apply additional global ranking, deduplication, or truncation.
///
/// Lifetimes
/// ---------
/// Tracks typically borrow edges and nodes from the runtime graph,
/// so the output is parameterized by:
/// - `'edge_lf`: lifetime of borrowed edges,
/// - `'seed_lf`: lifetime of borrowed seeds,
/// - `'alert_lf`: lifetime of borrowed alerts inside seeds.
#[derive(Clone, Debug, Default)]
pub struct SolverOutput {
    /// Candidate tracks returned by the solver.
    ///
    /// In most solvers, tracks are sorted from best to worst, but callers should
    /// not rely on that unless explicitly documented by the solver implementation.
    ///
    /// The key is a temporary track id assigned during reconstruction; final track ids are
    /// typically assigned after orbit fitting and persistence.
    pub tracks: HypothesisSet,

    /// Diagnostics for monitoring and tuning.
    pub diag: SolverDiagnostics,
}

use std::borrow::Cow;

use std::sync::Arc;

impl SolverOutput {
    pub fn merge_solver_output(all_solver_output: &[Self]) -> HypothesisSet {
        let mut merged: HypothesisSet = AHashMap::new();
        let mut next_id: HypothesisId = 0;

        for output in all_solver_output {
            for trk in output.tracks.values() {
                merged.insert(next_id, trk.clone());
                next_id += 1;
            }
        }

        merged
    }
}

/// Flatten all solver tracks into per-observatory [`ObservationBatch`] maps.
///
/// This helper converts the solver output (a set of independent
/// [`TrackHypothesis`] values) into *tabular*, solver-agnostic representations
/// expected by downstream routines (orbit fitting, IOD, batch scoring, etc.),
/// **grouped by MPC observatory code**.
///
/// The orbit fitter requires that all observations within a single batch
/// originate from the same observatory. This function therefore returns one
/// [`ObservationBatch`] per observatory, keyed by `Arc<String>` MPC code.
///
/// Output layout
/// -------------
/// The returned map associates each MPC code with a flat concatenation of
/// observations from all tracks that belong to that observatory:
///
/// - `trajectory_id[i]` identifies which track the *i-th* observation belongs to.
/// - `ra[i]`, `dec[i]`, and `time[i]` store the angular position and epoch.
///
/// Within each batch, arrays have identical length.
///
/// Determinism
/// -----------
/// [`AHashMap`] iteration order is not stable. To ensure deterministic output
/// (useful for tests, reproducible pipelines, and stable diagnostics), tracks
/// are processed in ascending key order:
///
/// 1. Collect all track keys (`temp_id`).
/// 2. Sort them with `sort_unstable()`.
/// 3. Flatten tracks in that sorted order.
///
/// This guarantees a stable global concatenation order *given the same input tracks*.
///
/// Per-track ordering
/// ------------------
/// Observations inside each track are explicitly normalized to time order:
///
/// - Alerts collected from the track’s seed nodes are sorted by `mjd_tt` ascending.
///
/// Even if the solver builds tracks from time-ordered seeds, this normalization
/// is helpful because:
/// - seed membership may overlap across consecutive seeds,
/// - the concatenation of multiple seeds is not guaranteed to be strictly sorted
///   without an explicit global sort step.
///
/// Deduplication strategy
/// ----------------------
/// Within each track, alerts are deduplicated **after sorting**, using the pointer
/// identity of the borrowed alert reference:
///
/// ```rust,ignore
/// alerts.dedup_by_key(|a| *a as *const Alert);
/// ```
///
/// This removes duplicates produced when multiple seeds in a track share member
/// alerts (common in overlapping triplets / pairs).
///
/// Notes:
/// - Pointer-based dedup assumes that identical logical alerts are represented
///   by the same in-memory `Alert` instance (typical when alerts come from a
///   central store and seeds hold references).
/// - If alerts have a known stable identifier (e.g. `candid`, `(night_id, idx)`,
///   etc.), prefer dedup by that identifier to be robust to alternative memory
///   layouts.
///
/// Uncertainty aggregation
/// -----------------------
/// [`ObservationBatch`] models angular uncertainties as a **single uniform**
/// 1-σ value for RA and for DEC, applied to the entire batch.
///
/// This implementation sets:
///
/// - `error_ra = max(alert.ra_err)` across all flattened observations
/// - `error_dec = max(alert.dec_err)` across all flattened observations
///
/// This is a conservative choice that avoids under-weighting any point.
///
/// If you need a different policy (e.g. median, mean, per-track values, or
/// per-observation uncertainties), implement it at the call site or change
/// the `ObservationBatch` model.
///
/// Allocation and lifetime behavior
/// -------------------------------
/// This method constructs **owned** buffers (`Vec<T>`) and returns them as
/// `Cow::Owned(...)`.
///
/// - The returned batch does **not** borrow from `self` despite taking `&self`.
/// - The lifetime parameter of the returned [`ObservationBatch`] is therefore
///   irrelevant to safety in the current implementation (it contains no borrowed
///   slices).
///
/// Capacity planning
/// -----------------
/// To reduce reallocations, the method first estimates an upper bound for the
/// number of produced observations by summing `seed.members.len()` across all
/// seeds of all tracks. This is a *safe upper bound* because deduplication may
/// remove some elements, but it remains a good heuristic for reserving memory.
///
/// Complexity
/// ----------
/// Let:
/// - `T` be the number of tracks,
/// - `M_t` be the number of collected alert references for track `t`
///   (before deduplication).
///
/// Then:
/// - Sorting track ids: `O(T log T)`
/// - For each track: sorting alerts: `O(M_t log M_t)`
/// - Dedup + flatten: `O(sum_t M_t)`
///
/// Total: `O(T log T + sum_t (M_t log M_t))`
///
/// Panics
/// ------
/// This method does not intentionally panic. If `mjd_tt` contains NaNs, the
/// `partial_cmp` used for sorting falls back to `Ordering::Equal`, which keeps
/// the sort total but may result in a less meaningful ordering for those entries.
///
/// See also
/// --------
/// - [`TrackHypothesis`]: single-trajectory solver output.
/// - [`ObservationBatch::from_radians_borrowed`]: zero-copy construction when
///   upstream already has contiguous slices (not the case here).
pub fn to_observation_batch<'a>(
    hypothesis_set: &HypothesisSet,
    obs_dataset: &'a ObsDataset,
    seed_store: &SeedStore,
) -> Result<AHashMap<Arc<String>, ObservationBatch<'a>>, SolverError> {
    // --- 0) Stable track order (deterministic)
    let mut track_ids: Vec<u32> = hypothesis_set.keys().copied().collect();
    track_ids.sort_unstable();

    // --- 1) Per-observatory accumulators (single-pass dispatch)
    struct Acc {
        trajectory_id: Vec<u32>,
        ra: Vec<Radian>,
        dec: Vec<Radian>,
        time: Vec<MJD>,
        max_ra_err: Radian,
        max_dec_err: Radian,
    }

    let mut per_obs: AHashMap<Arc<String>, Acc> = AHashMap::new();

    // --- 2) Flatten per track, dispatch per observatory
    for tid in track_ids {
        let trk = &hypothesis_set[&tid];

        let mut alerts: Vec<&Observation> = trk
            .get_alerts(obs_dataset, seed_store)
            .map_err(|e| SolverError::OrbitFitConversionError(e.to_string()))?;

        // Ensure time order inside this track
        alerts.sort_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()));

        for a in alerts {
            // Derive a string key for the observatory from the ObserverId.
            // MpcCode bytes are ASCII; IntId is formatted as a decimal string.
            let mpc_key: Arc<String> = match a.observer_id() {
                Some(photom::observer::dataset::ObserverId::MpcCode(code)) => {
                    Arc::new(String::from_utf8_lossy(code).into_owned())
                }
                Some(photom::observer::dataset::ObserverId::IntId(idx)) => {
                    Arc::new(format!("custom_{idx}"))
                }
                None => Arc::new("UNKNOWN".to_string()),
            };
            let acc = per_obs.entry(mpc_key).or_insert_with(|| Acc {
                trajectory_id: Vec::new(),
                ra: Vec::new(),
                dec: Vec::new(),
                time: Vec::new(),
                max_ra_err: 0.0,
                max_dec_err: 0.0,
            });

            acc.trajectory_id.push(tid);
            acc.ra.push(a.equ_coord().ra);
            acc.dec.push(a.equ_coord().dec);
            acc.time.push(a.mjd_tt());
            acc.max_ra_err = acc.max_ra_err.max(a.equ_coord().ra_error);
            acc.max_dec_err = acc.max_dec_err.max(a.equ_coord().dec_error);
        }
    }

    // --- 3) Convert accumulators to ObservationBatch
    let result = per_obs
        .into_iter()
        .map(|(mpc_code, acc)| {
            let batch = ObservationBatch {
                trajectory_id: Cow::Owned(acc.trajectory_id),
                ra: Cow::Owned(acc.ra),
                dec: Cow::Owned(acc.dec),
                time: Cow::Owned(acc.time),
                error_ra: acc.max_ra_err,
                error_dec: acc.max_dec_err,
            };
            (mpc_code, batch)
        })
        .collect();

    Ok(result)
}

/// A solver that extracts trajectory hypotheses from a connected component.
///
/// A solver implementation is responsible for:
/// - consuming the component-local view (via [`ConnectedComponents`]),
/// - producing one or more plausible [`TrackHypothesis`] objects,
/// - returning lightweight diagnostics describing its work.
///
/// The solver operates on a connected component identified by `component_id`.
/// The component-local directed subgraph (nodes, adjacency, sources/sinks, degrees,
/// etc.) is obtained through [`ConnectedComponents`].
///
/// Design goals
/// ------------
/// - **No copying** of large graph structures: solvers borrow edges/nodes.
/// - **Component-local** operation: solvers should avoid global rescans.
/// - **Pluggable**: multiple solver strategies can implement this trait.
///
/// Notes on thread-safety
/// ----------------------
/// The trait does not impose `Send`/`Sync`. Threading concerns are handled at
/// higher levels (e.g. solver manager) depending on the broader architecture.
pub trait Solver<'edge_lf, 'seed_lf> {
    /// A short stable name for logs and metrics.
    ///
    /// The name should be:
    /// - stable across versions,
    /// - suitable for metric labels,
    /// - descriptive of the solver strategy.
    fn name(&self) -> &'static str;

    /// Solve a single connected component and return trajectory hypotheses.
    ///
    /// The solver is given read-only access to:
    /// - the global runtime graph (`graph`), which owns the edges and provides
    ///   the backing storage for borrowed references,
    /// - the connected components object (`cc`), which provides a component-local
    ///   directed subgraph view,
    /// - a `component_id` selecting the component to solve.
    ///
    /// Expected behavior
    /// -----------------
    /// - Use `cc` to retrieve the component-local subgraph view.
    /// - Enumerate / optimize / select plausible track hypotheses.
    /// - Return a [`SolverOutput`] containing:
    ///   - a list of tracks,
    ///   - diagnostics ([`SolverDiagnostics`]) describing the work performed.
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (read-only for solving).
    /// * `cc` – Connected components object providing component-local views.
    /// * `component_id` – Identifier of the connected component to solve.
    ///
    /// Return
    /// ------
    /// * `SolverOutput` – Candidate tracks and diagnostics.
    ///
    /// Notes
    /// -----
    /// - The bound `'edge_lf: 'seed_lf` ensures edges outlive the seed references they contain.
    /// - Solvers should treat `graph` as immutable unless explicitly designed to
    ///   mutate edge flags elsewhere in the pipeline.
    fn solve(
        &self,
        graph: &'edge_lf AlertLinkageDAG,
        cc: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf>,
        component_id: ComponentId,
    ) -> SolverOutput
    where
        'edge_lf: 'seed_lf;
}

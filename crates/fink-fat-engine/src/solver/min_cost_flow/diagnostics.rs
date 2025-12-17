//! Diagnostics helpers for solver passes.
//!
//! # Overview
//! Solvers in this crate typically return a [`SolverOutput`](crate::solver::SolverOutput)
//! augmented with a [`SolverDiagnostics`] structure. The diagnostics object is
//! used for:
//! - logging (per-component tracing),
//! - metrics/telemetry (timings, sizes, counters),
//! - offline analysis of routing decisions (which solver handled which component).
//!
//! This module provides small helpers to construct a consistent baseline
//! diagnostics record for a single solver run, populated with information that
//! is known upfront (component stats, candidate counts, solver name).
//!
//! Notes
//! -----
//! - Diagnostics are intentionally lightweight and cheap to populate.
//! - Timing fields are typically filled by the caller (wall time measurement).
//!
//! See also
//! --------
//! - [`ComponentStats`] for the precomputed statistics collected per component.
//! - [`SolverDiagnostics`] for the full list of fields recorded by solvers.

use crate::solver::{ComponentStats, SolverDiagnostics};

/// Build a baseline [`SolverDiagnostics`] record for one solver pass.
///
/// This helper initializes a default diagnostics structure and fills the
/// fields that can be determined before running the solver logic:
/// - solver identity (`solver_name`),
/// - component size counters (`n_nodes`, `m_active_edges`),
/// - number of candidate nodes passed to the solver (`n_candidates`).
///
/// Parameters
/// ----------
/// solver_name : &'static str
///     Stable identifier for the solver implementation. This should match
///     [`Solver::name`](crate::solver::Solver::name) and remain stable across releases
///     so logs and metrics remain comparable.
/// stats : ComponentStats
///     Precomputed statistics for the connected component (e.g. number of nodes
///     and number of active edges). These are usually computed once during
///     component extraction/routing.
/// n_candidates : usize
///     Number of nodes provided to the solver for this pass. In most cases this
///     equals the component size, but some routing strategies may pass a subset
///     of nodes or apply pre-filtering.
///
/// Returns
/// -------
/// SolverDiagnostics
///     A partially populated diagnostics structure. The caller typically fills
///     additional fields such as:
///     - `time_spent_s`,
///     - selection counts (e.g. `n_selected`),
///     - solver-specific counters.
///
/// Notes
/// -----
/// - This function does not record timing. The caller should measure wall time
///   around the full solve pass and set `diagnostics.time_spent_s` accordingly.
/// - `n_candidates` is stored as `u32` in diagnostics; we perform a lossy cast.
///   This is safe as long as components remain below `u32::MAX` candidates.
pub(super) fn build_diagnostics(
    solver_name: &'static str,
    stats: ComponentStats,
    n_candidates: usize,
) -> SolverDiagnostics {
    let mut diagnostics = SolverDiagnostics::default();
    diagnostics.solver_name = solver_name;
    diagnostics.n_nodes = stats.n_nodes;
    diagnostics.m_active_edges = stats.m_active_edges;
    diagnostics.n_candidates = n_candidates as u32;
    diagnostics
}

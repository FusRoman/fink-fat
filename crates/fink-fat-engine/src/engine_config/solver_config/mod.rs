//! Global solver configuration container.
//!
//! Overview
//! --------
//! This module defines the top-level configuration structure used to control
//! the solver subsystem.
//!
//! The solver layer of the engine is composed of:
//!
//! - A **routing policy** (`SolverPolicy`) that decides which solver family
//!   should be applied to each connected component.
//! - One or more **solver-specific configurations**, such as
//!   [`BoundedBeamConfig`] for the bounded beam-search solver.
//!
//! `SolverConfig` groups these independent configuration blocks into a single
//! structure that can be:
//!
//! - deserialized from configuration files (e.g. YAML / JSON),
//! - passed through the engine as a single object,
//! - versioned and tuned consistently.
//!
//! Architectural role
//! -------------------
//! The solver subsystem has two conceptual layers:
//!
//! 1) **Routing layer**
//!    - Controlled by [`SolverPolicy`].
//!    - Decides *which solver family* to use for a component.
//!
//! 2) **Solver implementation layer**
//!    - Each solver family has its own configuration.
//!    - Example: [`BoundedBeamConfig`] controls beam width, pruning,
//!      output caps, and exploration limits for the bounded beam solver.
//!
//! `SolverConfig` ensures these layers remain cleanly separated while
//! remaining easy to configure from a single entry point.
//!
//! Serialization
//! -------------
//! This structure derives [`Serialize`] and [`Deserialize`] so it can be:
//!
//! - loaded from external configuration files,
//! - stored alongside experiment metadata,
//! - versioned in reproducible pipelines.
//!
//! Default behavior
//! ----------------
//! The default configuration:
//!
//! - uses the default routing heuristics from [`SolverPolicy::default()`],
//! - uses conservative exploration limits from [`BoundedBeamConfig::default()`].
//!
//! These defaults are intended to be safe and predictable, but should be
//! tuned for production workloads and benchmarked datasets.

use serde::{Deserialize, Serialize};

use crate::engine_config::solver_config::{
    bounded_beam_config::BoundedBeamConfig, solver_policy::SolverPolicy,
};

pub mod bounded_beam_config;
pub mod solver_policy;

/// Top-level solver configuration.
///
/// This structure groups:
///
/// - `solver_policy`: routing rules that decide which solver family to use,
/// - `bounded_beam`: parameters specific to the bounded beam solver.
///
/// Design rationale
/// ----------------
/// Keeping routing and solver-specific configuration separated allows:
///
/// - adjusting routing thresholds without changing solver internals,
/// - tuning solver exploration limits independently of routing decisions,
/// - serializing a complete solver configuration as a single object.
///
/// Typical usage
/// -------------
/// 1. Load `SolverConfig` from engine configuration.
/// 2. Use `solver_policy` when building a solve plan.
/// 3. Pass `bounded_beam` to the bounded beam solver when instantiated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Routing policy deciding which solver family is used per component.
    ///
    /// See [`SolverPolicy`] for details on:
    /// - heuristics vs forced routing,
    /// - time estimation model,
    /// - component size and night span thresholds.
    pub solver_policy: SolverPolicy,

    /// Configuration parameters for the bounded beam solver.
    ///
    /// These parameters control:
    /// - beam width,
    /// - local pruning per node,
    /// - maximum number of tracks emitted,
    /// - global exploration limits.
    ///
    /// See [`BoundedBeamConfig`] for detailed semantics.
    pub bounded_beam: BoundedBeamConfig,
}

impl Default for SolverConfig {
    /// Default solver configuration.
    ///
    /// Combines:
    /// - default routing policy,
    /// - default bounded beam configuration.
    ///
    /// Notes
    /// -----
    /// Defaults are intended to be:
    /// - conservative,
    /// - safe in terms of exploration limits,
    /// - suitable for initial experimentation.
    ///
    /// They should be calibrated for large-scale production workloads.
    fn default() -> Self {
        Self {
            solver_policy: SolverPolicy::default(),
            bounded_beam: BoundedBeamConfig::default(),
        }
    }
}

//! # Inter-night edge construction configuration (`EdgeConfig`)
//!
//! This module defines the runtime configuration used to build **directed inter-night
//! edges** between seeds, i.e. potential links `from -> to` where `night(to) > night(from)`.
//!
//! In the engine pipeline, edge construction typically sits after seeding (pairs/triplets)
//! and before connected-components / solving. Its core responsibilities are:
//!
//! 1) **Candidate retrieval**: for each left seed, retrieve a set of plausible right seeds
//!    from a spatial/time index (cone query around the propagated prediction).
//! 2) **Feature computation**: compute `EdgeFeatures` for each candidate edge.
//! 3) **Cost assignment**: derive a solver-facing `cost` from features.
//! 4) **Optional ML ranking + Top-K pruning**: keep only the most promising candidates
//!    per left seed using an ONNX classifier.
//!
//! This configuration controls the above behavior, including the “two modes” design:
//! - **Full emission mode** (`emit_all_edges = true`): return *all* candidate edges
//!   with physics-based costs, without ML ranking and without Top-K pruning.
//! - **ML Top-K mode** (`emit_all_edges = false`): score candidates with ONNX,
//!   keep only `top_k_per_left` per left seed, and use the derived feature cost.
//!
//! The ML mode relies on the ONNX inference utilities (`edge_prediction`) and the
//! batched Top-K selection routine (`ranking_topk`).
//!
//! -----------------------------------------------------------------------------
//! Operational modes
//! -----------------------------------------------------------------------------
//!
//! ## 1) Full emission mode (`emit_all_edges = true`)
//!
//! Intended use-cases:
//! - debugging candidate retrieval,
//! - generating exhaustive datasets for offline training,
//! - validating physics-only gating and feature distributions,
//! - small-scale runs where runtime is not a concern.
//!
//! Behavior:
//! - for each left seed, iterate all candidates returned by the candidate generator
//!   (usually via a spatial/time index),
//! - compute `EdgeFeatures` for each candidate,
//! - build an edge cost from a deterministic physics-derived score
//!   (e.g. `EdgeFeatures::kinematic_log_likelihood_cost()` in the edge module).
//!
//! Consequences:
//! - the edge set may become very large (fan-out grows quickly with cone size),
//! - connected components become denser and solvers cost more,
//! - deterministic and simple (no ML artifacts required).
//!
//! ## 2) ML Top-K mode (`emit_all_edges = false`)
//!
//! Intended use-cases:
//! - production-scale runs,
//! - controlling fan-out per left seed,
//! - improving purity before connected components / solvers.
//!
//! Behavior (per-left seed):
//! - retrieve candidates (same as full mode),
//! - compute `EdgeFeatures`,
//! - batch candidates into ONNX inference calls (`onnx_batch_size`),
//! - extract `p(class=1)` from model output,
//! - keep only the **Top-K** best candidates by probability (`top_k_per_left`),
//! - emit edges for those winners only.
//!
//! Notes:
//! - ML inference is performed by `EdgeRankingModel` / `EdgeRankingModelPool`
//!   (see `edge_prediction`).
//! - Top-K selection is implemented with a fixed-capacity min-heap so memory stays
//!   bounded by `O(K)` per left seed (see `ranking_topk`).
//!
//! -----------------------------------------------------------------------------
//! Parallelism and chunking
//! -----------------------------------------------------------------------------
//!
//! Edge construction is naturally parallelizable over the **left** seeds.
//!
//! If `parallel_left_batches = true`, the engine may process left seeds in
//! Rayon using chunked parallel iteration (conceptually `left.par_chunks(...)`).
//! The chunk size is controlled by `parallel_left_batch_size`.
//!
//! Rationale for chunking:
//! - amortize per-chunk setup costs (e.g., index references),
//! - limit per-thread working set,
//! - reduce overhead from scheduling extremely small tasks.
//!
//! -----------------------------------------------------------------------------
//! Predictor configuration (`predictor_config`)
//! -----------------------------------------------------------------------------
//!
//! Candidate retrieval typically uses a propagated **sky cone** prediction:
//! - propagate the left seed to the target epoch (or target time bins),
//! - derive an uncertainty radius from a plane covariance,
//! - retrieve right-side seeds whose sky position falls in that cone.
//!
//! This behavior is configured by [`PredictorParams`] (`predictor_config`), which
//! controls:
//! - kσ inflation (`k_sigma`),
//! - additive model noise schedule (`noise`),
//! - optional cell-radius padding (`pad_cell_radius`),
//! - time bin handling (`time_bin_dt`),
//! - optional velocity slack (`v_slack`).
//!
//! The predictor is a major driver of candidate fan-out: increasing its radius
//! increases recall but also runtime. In ML Top-K mode, fan-out is later capped
//! by `top_k_per_left`, but inference cost still scales with the number of raw
//! candidates.
//!
//! -----------------------------------------------------------------------------
//! Configuration examples (YAML)
//! -----------------------------------------------------------------------------
//!
//! Full emission mode (debug / dataset generation):
//!
//! ```yaml
//! edges:
//!   emit_all_edges: true
//!   parallel_left_batches: true
//!   parallel_left_batch_size: 512
//!
//!   # In full emission mode, ML-related parameters are typically unused by the
//!   # edge builder, but they may still be present in the config.
//!   edge_ranking_model_path: "model.onnx"
//!   top_k_per_left: 32
//!   onnx_batch_size: 128
//!
//!   predictor_config:
//!     k_sigma: 3.0
//!     noise: { variance_floor: 0.0, drift_per_day: 0.0, curvature_per_day2: 0.0 }
//!     pad_cell_radius: true
//!     time_bin_dt: 0.021
//!     v_slack: 0.0
//! ```
//!
//! ML Top-K mode (production):
//!
//! ```yaml
//! edges:
//!   emit_all_edges: false
//!   edge_ranking_model_path: "edge_ranker.onnx"
//!   top_k_per_left: 32
//!   onnx_batch_size: 128
//!
//!   parallel_left_batches: true
//!   parallel_left_batch_size: 512
//!
//!   predictor_config:
//!     k_sigma: 3.5
//!     noise: { variance_floor: 1.0e-12, drift_per_day: 0.0, curvature_per_day2: 5.0e-14 }
//!     pad_cell_radius: true
//!     time_bin_dt: 0.021
//!     v_slack: 0.0
//! ```
//!
//! Unknown keys are rejected (`deny_unknown_fields`) to catch typos early.
//!
//! -----------------------------------------------------------------------------
//! Validation
//! -----------------------------------------------------------------------------
//!
//! [`EdgeConfig::validate`] currently enforces a minimal invariant:
//! - `top_k_per_left != 0`.
//!
//! Rationale: in ML Top-K mode, a zero value would silently disable all edges.
//! In full emission mode, `top_k_per_left` may not be used, but keeping the
//! constraint avoids configuration ambiguity.
//!
//! Additional checks (often useful in production) can be added if desired:
//! - `onnx_batch_size > 0`,
//! - `parallel_left_batch_size > 0` (or clamp in a single place),
//! - `predictor_config.validate()` (if not already validated upstream).
//!
//! -----------------------------------------------------------------------------
//! See also
//! -----------------------------------------------------------------------------
//!
//! - `crate::graph::edge` (edge construction entrypoint and the two operational modes).
//! - `crate::graph::edge::edge_prediction` (ONNX model loading + inference).
//! - `crate::graph::edge::ranking_topk` (batched per-left Top-K ranking).
//! - [`PredictorParams`] (cone prediction controlling candidate retrieval).

use serde::{Deserialize, Serialize};

use crate::engine_config::{error::EdgeConfigError, propagator_config::PredictorParams};

fn default_false() -> bool {
    false
}

/// Runtime configuration used by the engine to build inter-night edges.
///
/// This structure is intended to be deserialized from configuration files
/// (YAML/TOML/JSON) and then validated before execution.
///
/// Notes
/// -----
/// - `deny_unknown_fields` rejects unknown YAML keys to catch typos early.
/// - `serde(default)` fills missing fields from [`Default`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EdgeConfig {
    /// Path to the ONNX model used for ML edge ranking.
    ///
    /// Used in ML Top-K mode (`emit_all_edges = false`):
    /// - passed to `EdgeRankingModel::load_edge_ranking_model(...)` (or a model pool),
    /// - the model is expected to output a `probabilities` tensor of shape `[N, 2]`
    ///   where column 1 is `p(class=1)` (true edge probability).
    ///
    /// Notes
    /// -----
    /// - In full emission mode (`emit_all_edges = true`), this path is typically unused.
    /// - The file is not validated here; failures usually surface during model loading.
    pub edge_ranking_model_path: String,

    /// Maximum number of edges retained **per left seed** in ML Top-K mode.
    ///
    /// Behavior
    /// --------
    /// In ML Top-K mode (`emit_all_edges = false`), candidates are scored by ONNX
    /// and only the `top_k_per_left` best candidates by `p(class=1)` are retained.
    ///
    /// Trade-off
    /// ---------
    /// - Smaller values reduce runtime and graph density but can reduce recall.
    /// - Larger values increase recall but can create larger connected components
    ///   and increase solver cost.
    ///
    /// Validation
    /// ----------
    /// Must be non-zero (enforced by [`EdgeConfig::validate`]).
    pub top_k_per_left: usize,

    /// Batch size used for ONNX inference when scoring candidates.
    ///
    /// Context
    /// -------
    /// In `ranking_topk`, candidates are accumulated into batches and scored
    /// with one ONNX call per batch. Larger batches improve throughput (amortize
    /// overhead) but increase temporary memory usage.
    ///
    /// Notes
    /// -----
    /// - In full emission mode (`emit_all_edges = true`), this value is typically unused.
    /// - If set too small, throughput can degrade significantly due to per-call overhead.
    pub onnx_batch_size: usize,

    /// Select the edge construction mode.
    ///
    /// Behavior
    /// --------
    /// - If `true`:
    ///   - emit *all* candidate edges returned by candidate generation,
    ///   - do not run ONNX inference,
    ///   - do not apply Top-K pruning.
    /// - If `false`:
    ///   - enable ML ranking and keep only `top_k_per_left` edges per left seed,
    ///   - ONNX inference uses `edge_ranking_model_path` and `onnx_batch_size`.
    ///
    /// Notes
    /// -----
    /// - This field defaults to `false`.
    /// - This is the primary switch controlling “debug/exhaustive” vs “production/pruned”.
    #[serde(default = "default_false")]
    pub emit_all_edges: bool,

    /// Enable parallel processing of left seeds with Rayon.
    ///
    /// Behavior
    /// --------
    /// When enabled, the engine processes chunks of left seeds in parallel.
    /// This can substantially improve throughput for large datasets.
    ///
    /// Notes
    /// -----
    /// - Parallel inference typically requires one ONNX session per thread
    ///   (see `EdgeRankingModelPool`).
    #[serde(default = "default_false")]
    pub parallel_left_batches: bool,

    /// Chunk size for left-side batch processing.
    ///
    /// Context
    /// -------
    /// When `parallel_left_batches = true`, the engine may conceptually run:
    ///
    /// ```text
    /// left.par_chunks(parallel_left_batch_size).for_each(...)
    /// ```
    ///
    /// Notes
    /// -----
    /// - Values of `0` are not meaningful for chunking. Some call sites clamp
    ///   `<= 0` to 1; this struct stores the raw value.
    /// - Tune this to balance Rayon overhead vs per-chunk working set.
    pub parallel_left_batch_size: usize,

    /// Parameters controlling propagation-based cone prediction for candidate retrieval.
    ///
    /// This configuration is used to:
    /// - propagate a left seed to a target epoch,
    /// - compute an uncertainty radius (kσ + model noise),
    /// - optionally pad by spatial cell radius,
    /// - retrieve right-side candidates using that cone.
    pub predictor_config: PredictorParams,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_ranking_model_path: "model.onnx".to_string(),
            top_k_per_left: 32,
            onnx_batch_size: 128,
            emit_all_edges: false,
            parallel_left_batches: false,
            parallel_left_batch_size: 512,
            predictor_config: PredictorParams::default(),
        }
    }
}

impl EdgeConfig {
    /// Validate the configuration for basic invariants.
    ///
    /// Checks performed
    /// ----------------
    /// - `top_k_per_left != 0`.
    ///
    /// Return
    /// ------
    /// - `Ok(())` if valid.
    /// - `Err(EdgeConfigError)` otherwise.
    ///
    /// Notes
    /// -----
    /// This function is intentionally minimal. Depending on where configuration
    /// is loaded, additional validation can be useful (batch sizes, predictor
    /// parameters, model path existence, etc.).
    pub fn validate(&self) -> Result<(), EdgeConfigError> {
        self.predictor_config.validate()?;

        if self.top_k_per_left == 0 {
            return Err(EdgeConfigError::TopKPerLeftZero);
        }
        Ok(())
    }
}

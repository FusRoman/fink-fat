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
//! 4) **Optional Top-K pruning**: keep only the most promising candidates per left seed,
//!    ranked either by an ONNX classifier or by the physics-based cost function.
//!
//! -----------------------------------------------------------------------------
//! Operational modes
//! -----------------------------------------------------------------------------
//!
//! The two orthogonal configuration axes are:
//!
//! - **`top_k_per_left`** — controls *whether* Top-K filtering is applied.
//! - **`use_ml_ranking`** — controls *how* candidates are ranked when filtering is active.
//!
//! ## 1) No filtering (`top_k_per_left = None`)
//!
//! Intended use-cases:
//! - debugging candidate retrieval,
//! - generating exhaustive datasets for offline training,
//! - validating physics-only gating and feature distributions.
//!
//! Behavior:
//! - for each left seed, iterate all candidates returned by the candidate generator,
//! - compute `EdgeFeatures` and edge cost for each candidate,
//! - emit all edges without any Top-K pruning.
//!
//! Consequences:
//! - the edge set can become very large (fan-out grows quickly with cone size),
//! - connected components become denser and solvers cost more,
//! - deterministic and simple — no ranking required.
//!
//! ## 2) Cost-based Top-K (`top_k_per_left = Some(k)`, `use_ml_ranking = false`)
//!
//! Intended use-cases:
//! - production-scale runs without an ONNX model,
//! - controlling fan-out per left seed with physics-derived ranking.
//!
//! Behavior (per-left seed):
//! - retrieve candidates,
//! - compute `EdgeFeatures` and edge cost,
//! - retain only the `k` candidates with the **lowest cost**,
//! - emit edges for those winners only.
//!
//! Notes:
//! - No ONNX model is required.
//! - Cost ranking uses the function configured in `cost` (`CostConfig`).
//! - Top-K selection is implemented with a fixed-capacity min-heap so memory stays
//!   bounded by $O(K)$ per left seed (see `ranking_topk`).
//!
//! ## 3) ML Top-K (`top_k_per_left = Some(k)`, `use_ml_ranking = true`)
//!
//! Intended use-cases:
//! - production-scale runs with a trained ONNX edge classifier,
//! - highest-purity candidate selection before connected components / solvers.
//!
//! Behavior (per-left seed):
//! - retrieve candidates,
//! - compute `EdgeFeatures`,
//! - batch candidates into ONNX inference calls (`onnx_batch_size`),
//! - extract `p(class=1)` from model output,
//! - retain only the `k` candidates with the **highest probability**,
//! - emit edges for those winners only.
//!
//! Notes:
//! - ML inference is performed by `EdgeRankingModel` / `EdgeRankingModelPool`
//!   (see `edge_prediction`).
//! - Requires `edge_ranking_model_path` and a live `EdgeRankingModelPool`.
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
//! increases recall but also runtime. When `top_k_per_left` is `Some(k)`, fan-out
//! is capped per left seed, but feature computation cost still scales with the
//! number of raw candidates.
//!
//! -----------------------------------------------------------------------------
//! Configuration examples (YAML)
//! -----------------------------------------------------------------------------
//!
//! No filtering — emit all candidates (debug / dataset generation):
//!
//! ```yaml
//! edges:
//!   top_k_per_left: ~    # null = no filtering
//!   parallel_left_batches: true
//!   parallel_left_batch_size: 512
//!   predictor_config:
//!     k_sigma: 3.0
//!     noise: { variance_floor: 0.0, drift_per_day: 0.0, curvature_per_day2: 0.0 }
//!     pad_cell_radius: true
//!     time_bin_dt: 0.021
//!     v_slack: 0.0
//!   cost:
//!     variant: gaussian_chi2
//! ```
//!
//! Cost-based Top-K (recommended default for production without ONNX):
//!
//! ```yaml
//! edges:
//!   use_ml_ranking: false   # default; cost-based ranking
//!   top_k_per_left: 32
//!   parallel_left_batches: true
//!   parallel_left_batch_size: 512
//!   predictor_config:
//!     k_sigma: 3.5
//!     noise: { variance_floor: 1.0e-12, drift_per_day: 0.0, curvature_per_day2: 5.0e-14 }
//!     pad_cell_radius: true
//!     time_bin_dt: 0.021
//!     v_slack: 0.0
//!   cost:
//!     variant: singer_cwna
//!     sigma_q: 1.0e-3
//! ```
//!
//! ML Top-K (production with ONNX model):
//!
//! ```yaml
//! edges:
//!   use_ml_ranking: true
//!   edge_ranking_model_path: "edge_ranker.onnx"
//!   top_k_per_left: 32
//!   onnx_batch_size: 128
//!   parallel_left_batches: true
//!   parallel_left_batch_size: 512
//!   predictor_config:
//!     k_sigma: 3.5
//!     noise: { variance_floor: 1.0e-12, drift_per_day: 0.0, curvature_per_day2: 5.0e-14 }
//!     pad_cell_radius: true
//!     time_bin_dt: 0.021
//!     v_slack: 0.0
//!   cost:
//!     variant: singer_cwna
//!     sigma_q: 1.0e-3
//! ```
//!
//! Unknown keys are rejected (`deny_unknown_fields`) to catch typos early.
//!
//! Add `max_cost_cut` to any of the three modes to apply a hard upper bound:
//!
//! ```yaml
//! edges:
//!   max_cost_cut: 5.0      # discard candidates whose cost exceeds 5.0
//!   top_k_per_left: 32
//!   cost:
//!     variant: singer_cwna
//!     sigma_q: 1.0e-3
//! ```
//!
//! -----------------------------------------------------------------------------
//! Cost cut (`max_cost_cut`)
//! -----------------------------------------------------------------------------
//!
//! An optional hard upper bound on the edge cost returned by
//! [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`].
//!
//! When set, any candidate whose computed cost exceeds this threshold is
//! discarded immediately — before Top-K scoring and before being added to the
//! heap. The cut is applied in **all** edge-building modes:
//!
//! - **Emit-all**: candidates above the cut are skipped before materialising the
//!   edge, even though no Top-K filtering is otherwise active.
//! - **Cost-based Top-K**: the cut is applied after computing cost and before the
//!   score mapping; it reduces the number of candidates competing for the K slots.
//! - **ML Top-K**: in the batch-flush routine of
//!   [`crate::graph::edge::ranking_topk`], cost
//!   is computed once a candidate has passed the probability threshold; if the
//!   cost then exceeds `max_cost_cut`, the candidate is still discarded.
//!
//! Notes:
//! - `None` disables the cut; all candidates are processed regardless of cost.
//! - `Some(v)` with `v <= 0.0` is rejected by [`EdgeConfig::validate`] because
//!   costs are strictly positive and such a value would silently discard every
//!   candidate.
//!
//! -----------------------------------------------------------------------------
//! Validation
//! -----------------------------------------------------------------------------
//!
//! [`EdgeConfig::validate`] currently enforces the following invariants:
//! - `top_k_per_left != Some(0)` — a zero limit would silently discard all edges.
//! - `max_cost_cut` must satisfy `v > 0.0` when `Some(v)` — a non-positive cut
//!   would silently discard every candidate.
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
//! - `crate::graph::edge` (edge construction entrypoint and the operational modes).
//! - `crate::graph::edge::edge_prediction` (ONNX model loading + inference).
//! - `crate::graph::edge::ranking_topk` (batched per-left Top-K ranking).
//! - [`PredictorParams`] (cone prediction controlling candidate retrieval).
use serde::{Deserialize, Serialize};

use crate::engine_config::{error::EdgeConfigError, propagator_config::PredictorParams};

fn default_false() -> bool {
    false
}

// =============================================================================
// Cost-function configuration
// =============================================================================

/// Selector for the kinematic cost function used when building graph edges.
///
/// All variants apply the same photometry terms unchanged.  They differ in
/// how the positional/velocity chi² is computed (covariance model) and in
/// which loss function maps chi² to a scalar cost.
///
/// YAML spelling is `snake_case` (e.g., `robust_cauchy`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostVariant {
    /// Backward-compatibility alias for `GaussianChi2` with `sigma_q = 0`.
    ///
    /// Produces identical numerical results to `GaussianChi2` (same ½χ² formula,
    /// same baseline covariances).  Use this to guarantee the same edge costs as
    /// runs predating the configurable cost system.
    KinematicLogLikelihood,

    /// Gaussian ½χ²: standard negative-log-likelihood for a constant-velocity
    /// motion model with no process noise.
    ///
    /// `c = ½(χ²_pos + χ²_vel)` where chi² uses the fixed measurement-noise
    /// covariance.  Cost grows as Δt² for large inter-night gaps.
    #[default]
    GaussianChi2,

    /// Singer CWNA: Gaussian ½χ² with continuous white-noise acceleration
    /// (CWNA) covariance inflation.
    ///
    /// Adds `σ_q² · Δt³/3 · I` to the positional innovation covariance and
    /// `σ_q² · Δt · I` to the velocity covariance.  χ² stays ~O(1) for all
    /// night-gaps when `sigma_q` is well-calibrated (~1e-3 rad/day^(3/2)).
    /// Requires `sigma_q > 0`; falls back to `GaussianChi2` when `sigma_q = 0`.
    SingerCwna, // Continuous White Noise Acceleration (Singer process noise model).

    /// Robust Cauchy loss: `c = ln(1 + χ²_pos/scale) + ln(1 + χ²_vel/scale)`.
    ///
    /// The logarithmic saturation bounds the cost for large residuals, limiting
    /// the influence of high-curvature trajectories or large night-gaps.
    /// Can optionally be combined with CWNA process noise (`sigma_q > 0`).
    RobustCauchy,

    /// Robust Student-t loss: `c = (ν+1)/2 · [ln(1+χ²_pos/ν) + ln(1+χ²_vel/ν)]`.
    ///
    /// Generalises Cauchy (ν=1) toward Gaussian (ν→∞).  Provides a smoother
    /// transition between the two regimes.  Can optionally be combined with
    /// CWNA process noise (`sigma_q > 0`).
    RobustStudentT, // Degrees of freedom ν is configured separately in `student_nu`.
}

/// Parameters for the edge kinematic cost function.
///
/// These settings are exposed in the YAML config under `edges.cost`:
///
/// ```yaml
/// edges:
///   cost:
///     variant: singer_cwna      # gaussian_chi2 | kinematic_log_likelihood | singer_cwna | robust_cauchy | robust_student_t
///     sigma_q: 1.0e-3           # CWNA accel. spectral density [rad·day^(-3/2)]
///     cauchy_scale: 2.0         # Cauchy transition scale
///     student_nu: 3.0           # Student-t degrees of freedom
/// ```
///
/// Notes
/// -----
/// - `sigma_q` is used by `SingerCwna`, `RobustCauchy`, and `RobustStudentT`.
///   Set to `0.0` to disable process noise (pure measurement-noise covariance).
/// - `cauchy_scale` is only used by `RobustCauchy`.
/// - `student_nu` is only used by `RobustStudentT`.
/// - `KinematicLogLikelihood` is the original cost function, included for
///   backward compatibility; it is numerically equivalent to `GaussianChi2`
///   with `sigma_q = 0.0`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CostConfig {
    /// Which kinematic loss function to apply.
    pub variant: CostVariant,

    /// CWNA acceleration spectral density σ_q (rad · day^(−3/2)).
    ///
    /// When non-zero, the positional innovation covariance is inflated by
    /// `σ_q² · Δt³/3 · I` and the velocity covariance by `σ_q² · Δt · I`.
    /// Recommended calibrated value: `1e-3`.
    pub sigma_q: f64,

    /// Transition scale for the Cauchy loss: `ρ(χ²) = ln(1 + χ²/scale)`.
    ///
    /// Defaults to `2.0` (Gaussian and Cauchy regimes cross at χ² = 2·scale).
    pub cauchy_scale: f64,

    /// Degrees of freedom ν for the Student-t loss.
    ///
    /// `ν=1` reproduces Cauchy; `ν→∞` converges to Gaussian.  Default: `3.0`.
    pub student_nu: f64,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            variant: CostVariant::GaussianChi2,
            sigma_q: 0.0,
            cauchy_scale: 2.0,
            student_nu: 3.0,
        }
    }
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
    /// Used when `use_ml_ranking = true`:
    /// - passed to `EdgeRankingModel::load_edge_ranking_model(...)` (or a model pool),
    /// - the model is expected to output a `probabilities` tensor of shape `[N, 2]`
    ///   where column 1 is `p(class=1)` (true edge probability).
    ///
    /// Notes
    /// -----
    /// - When `use_ml_ranking = false`, this path is unused.
    /// - The file is not validated here; failures usually surface during model loading.
    pub edge_ranking_model_path: Option<String>,

    /// Maximum number of edges retained **per left seed** after ranking.
    ///
    /// Behavior
    /// --------
    /// - When `Some(k)`: apply Top-K filtering after ranking (either ML or cost-based),
    ///   retaining only the `k` best candidates per left seed.
    /// - When `None`: no Top-K filtering is applied; all candidate edges are emitted.
    ///
    /// Trade-off
    /// ---------
    /// - Smaller values reduce runtime and graph density but can reduce recall.
    /// - Larger values increase recall but can create larger connected components
    ///   and increase solver cost.
    /// - `None` can produce very large edge sets; use with care on dense nights.
    ///
    /// Validation
    /// ----------
    /// `Some(0)` is rejected by [`EdgeConfig::validate`] as it would silently
    /// discard all edges.
    pub top_k_per_left: Option<usize>,

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
    /// - This value is only relevant when `use_ml_ranking = true`.
    /// - If set too small, throughput can degrade significantly due to per-call overhead.
    pub onnx_batch_size: usize,

    /// Select the ranking strategy for Top-K candidate pruning.
    ///
    /// Behavior
    /// --------
    /// - If `true`:
    ///   - rank candidates using the ONNX ML classifier score `p(class=1)`,
    ///   - requires `edge_ranking_model_path` to point to a valid ONNX model,
    ///   - requires `model_pool` to be provided at the call site.
    /// - If `false` (default):
    ///   - rank candidates using the physics-based cost function
    ///     (`EdgeFeatures::compute_cost` configured via `cost`),
    ///   - lower cost = better candidate,
    ///   - no ONNX model required.
    ///
    /// Notes
    /// -----
    /// - This switch is only meaningful when `top_k_per_left` is `Some(k)`.
    ///   When `top_k_per_left = None`, all candidates are emitted regardless of
    ///   this setting.
    #[serde(default = "default_false")]
    pub use_ml_ranking: bool,
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

    /// Cost function used to assign a scalar weight to each edge.
    ///
    /// This controls both the covariance model (optional CWNA process noise)
    /// and the loss function (Gaussian, Cauchy, Student-t).
    /// The photometry penalty terms are unaffected by this choice.
    ///
    /// In cost-based Top-K mode (`use_ml_ranking = false`, `top_k_per_left = Some(k)`),
    /// this is also the ranking criterion: the K candidates with the lowest cost are
    /// retained per left seed.
    #[serde(rename = "cost")]
    pub cost_config: CostConfig,

    /// Hard upper bound on the edge cost used as a pre-filter.
    ///
    /// Behavior
    /// --------
    /// - `None` (default): no cut is applied; all candidates are evaluated
    ///   regardless of their cost.
    /// - `Some(max)`: any candidate whose
    ///   [`crate::graph::edge::edge_features::EdgeFeatures::compute_cost`]
    ///   result exceeds `max` is discarded immediately — before Top-K scoring
    ///   and before entering the emitted edge list.
    ///
    /// This cut takes effect across all three edge-building modes:
    ///
    /// - **Emit-all** (`top_k_per_left = None`): applied after cost computation,
    ///   before materialising the edge.
    /// - **Cost-based Top-K**: applied after cost computation, before the score
    ///   mapping and heap insertion.
    /// - **ML Top-K**: applied inside the batch-flush routine of
    ///   [`crate::graph::edge::ranking_topk`], after ONNX probability
    ///   scoring and after cost computation, before heap insertion.
    ///
    /// Notes
    /// -----
    /// - `Some(v)` with `v <= 0.0` is rejected by [`EdgeConfig::validate`].
    pub max_cost_cut: Option<f64>,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_ranking_model_path: None,
            top_k_per_left: Some(32),
            onnx_batch_size: 128,
            use_ml_ranking: false,
            parallel_left_batches: false,
            parallel_left_batch_size: 512,
            predictor_config: PredictorParams::default(),
            cost_config: CostConfig::default(),
            max_cost_cut: None,
        }
    }
}

impl EdgeConfig {
    /// Validate the configuration for basic invariants.
    ///
    /// Checks performed
    /// ----------------
    /// - `top_k_per_left != Some(0)`: a zero limit would silently discard all edges.
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

        if self.top_k_per_left == Some(0) {
            return Err(EdgeConfigError::TopKPerLeftZero);
        }

        if let Some(max) = self.max_cost_cut
            && max <= 0.0
        {
            return Err(EdgeConfigError::MaxCostCutNotPositive(max));
        }

        Ok(())
    }
}

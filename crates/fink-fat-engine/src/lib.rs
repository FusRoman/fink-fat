//! # fink-fat-engine
//!
//! **fink-fat-engine** is the core detection-linking library of the Fink-FAT
//! pipeline. It ingests nightly photometric alerts from wide-field sky surveys
//! (ZTF, Vera Rubin Observatory/LSST) and links them across nights to build
//! candidate solar-system object trajectories, which are then forwarded to an
//! orbit estimator for confirmation.
//!
//! The engine is designed to run incrementally: a new night of observations is
//! appended to a persisted runtime graph and the solver revisits only the
//! affected connected components.
//!
//! # Overview
//!
//! The input observation record is an [`Alert`] — a single photometric detection
//! carrying sky position (RA/Dec in radians), epoch (MJD TT), flux, and
//! band. Alerts are grouped by night in an [`AlertStore`].
//!
//! From alerts, the engine builds:
//!
//! 1. **Seeds** ([`crate::seeding::SeedNode`]) — intra-night motion vectors
//!    derived from pairs or triplets of same-night detections. A seed encodes
//!    a sky position, an apparent velocity vector, and photometric aggregates
//!    projected onto a local tangent plane.
//!
//! 2. **Edges** ([`crate::graph::edge::Edge`]) — inter-night kinematic
//!    hypotheses linking a seed from one night to a seed from another. Each
//!    edge carries a set of
//!    [`EdgeFeatures`](crate::graph::edge::edge_features::EdgeFeatures)
//!    (position residuals, velocity residuals, flux ratio, …) and a scalar
//!    cost derived from a configurable cost function.
//!
//! 3. **Trajectories** ([`crate::trajectory::TrackHypothesis`]) — ordered
//!    sequences of seeds connected by edges, produced by a solver operating
//!    on the inter-night directed acyclic graph.
//!
//! # Pipeline
//!
//! The engine exposes a stage-based execution model. A
//! [`crate::pipeline::PipelineRunner`] executes a
//! [`crate::pipeline::PipelinePlan`] — an ordered subset of the seven
//! canonical stages defined by [`crate::pipeline::stages::PipelineStage`]:
//!
//! | # | Stage | Description |
//! |---|-------|-------------|
//! | 0 | `LoadPersistedData` | Deserialises the runtime graph and alert/seed stores from disk. |
//! | 1 | `IngestNights` | Reads an alert batch (Parquet), auto-detects all `night_id` values present, deduplicates, and stores each night independently. |
//! | 2 | `BuildSeeds` | Generates intra-night pairs and triplets; indexes them spatially. |
//! | 3 | `BuildEdges` | Computes inter-night kinematic candidates; optionally applies ML post-filtering. |
//! | 4 | `Solve` | Runs the solver (bounded-beam search) on each connected component. |
//! | 5 | `FitOrbit` | Submits confirmed trajectories to the [`outfit`](https://crates.io/crates/outfit) orbit estimator. |
//! | 6 | `SavePersistedData` | Serialises the updated state to disk via the journal-based persistence layer. |
//!
//! The input Parquet file may contain alerts from **multiple nights** in a
//! single batch. The engine automatically detects all `night_id` values
//! present, seeds each night independently, and constructs inter-night edges
//! while respecting the configured maximum gap (`max_gap_nights`). Ingesting
//! more than one night at a time is supported for late ingestion, testing,
//! and performance evaluation, but is discouraged in normal production
//! operation where nightly incremental runs are preferred.
//!
//! Not all stages need to be present in a given plan — for example, an
//! evaluation run can build seeds and edges without solving or saving.
//! Stages must appear in **strictly increasing canonical order**.
//!
//! ```rust, ignore
//! use fink_fat_engine::pipeline::{
//!     PipelineRunner, PipelinePlan, PipelineInputs, PipelineContext,
//!     hooks::NoopPipelineHooks,
//!     stages::{PipelineStage, alert_inputs::input_uri::InputUri},
//! };
//! use fink_fat_engine::engine_config::{EngineConfig, pipeline_policy::PersistPolicy};
//!
//! // Build a minimal plan: ingest + seed + edge, no persist.
//! let plan = PipelinePlan {
//!     stages: vec![
//!         PipelineStage::IngestNights,
//!         PipelineStage::BuildSeeds,
//!         PipelineStage::BuildEdges,
//!     ],
//!     persist: PersistPolicy::None,
//!     inputs: PipelineInputs {
//!         alerts_uri: InputUri::Parquet("/data/night_1.parquet".into()),
//!     },
//! };
//!
//! let runner = PipelineRunner { plan };
//! // runner.run(&mut ctx, &NoopPipelineHooks) — requires a PipelineContext.
//! ```
//!
//! # Core Abstractions
//!
//! | Type | Module | Role |
//! |------|--------|------|
//! | [`Alert`] | [`alerts`] | Single photometric detection (position, epoch, flux, band). |
//! | [`AlertKey`] | [`alerts`] | Composite identifier `(NightId, DiaSourceId)`. |
//! | [`AlertStore`] | [`alerts::store`] | Per-night indexed alert collection. |
//! | [`SeedNode`](crate::seeding::SeedNode) | [`seeding`] | Intra-night kinematic vector (pair or triplet of alerts). |
//! | [`SeedStore`](crate::seeding::store::SeedStore) | [`seeding::store`] | Indexed collection of seeds. |
//! | [`Edge`](crate::graph::edge::Edge) | [`graph::edge`] | Inter-night kinematic hypothesis with associated cost. |
//! | [`AlertLinkageDAG`](crate::graph::AlertLinkageDAG) | [`graph`] | Runtime directed acyclic graph of seeds and edges. |
//! | [`TrackHypothesis`](crate::trajectory::TrackHypothesis) | [`trajectory`] | Ordered multi-night candidate trajectory. |
//!
//! # Edge Cost Functions
//!
//! Each edge carries a scalar cost computed from its
//! [`EdgeFeatures`](crate::graph::edge::edge_features::EdgeFeatures).
//! The cost variant is selected via
//! [`CostVariant`](crate::engine_config::edge_config::CostVariant) in
//! [`EdgeConfig`](crate::engine_config::edge_config::EdgeConfig).
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `GaussianChi2` | $\frac{1}{2}(\chi^2\_{\mathrm{pos}} + \chi^2\_{\mathrm{vel}})$ — plain Gaussian loss. |
//! | `SingerCwna` | Same Gaussian loss on CWNA-inflated covariances (Singer 1970). |
//! | `RobustCauchy` | $\ln(1 + \chi^2\_{\mathrm{pos}}/\sigma) + \ln(1 + \chi^2\_{\mathrm{vel}}/\sigma)$ — Cauchy M-estimator. |
//! | `RobustStudentT` | $\frac{\nu+1}{2}\bigl[\ln(1+\chi^2\_{\mathrm{pos}}/\nu)+\ln(1+\chi^2\_{\mathrm{vel}}/\nu)\bigr]$ — Student-t M-estimator. |
//! | `KinematicLogLikelihood` | Alias for `GaussianChi2` with `sigma_q = 0` (backward compatibility). |
//!
//! All variants include an optional photometric penalty term based on the
//! flux standard deviation ratio between the two seeds.
//!
//! See [`EdgeFeatures::compute_cost`](crate::graph::edge::edge_features::EdgeFeatures::compute_cost)
//! for the full formula.
//!
//! # ML Post-filtering
//!
//! After kinematic edge construction, an optional ONNX-based post-filter can
//! discard low-confidence edges before the solver stage. It is controlled by
//! two fields in [`EdgeConfig`](crate::engine_config::edge_config::EdgeConfig):
//!
//! - `ml_post_filter: bool` — enables the filter.
//! - `ml_post_filter_threshold: f32` — minimum `p(class=1)` to retain an edge.
//!
//! The model is loaded via
//! [`EdgeRankingModelPool`](crate::graph::edge::edge_prediction::EdgeRankingModelPool),
//! which manages one ONNX session per thread using a thread-local pool.
//! Intra-operator parallelism is configured by `onnx_intra_threads`.
//!
//! # Configuration
//!
//! All engine behaviour is controlled by a single
//! [`EngineConfig`](crate::engine_config::EngineConfig) struct, loaded from a
//! YAML file and optionally overridden by environment variables with the
//! `FINK_FAT__` prefix (e.g. `FINK_FAT__MAX_GAP_NIGHTS=4`).
//!
//! Key sub-sections:
//!
//! | Section | Type | Controls |
//! |---------|------|----------|
//! | `pairs` | [`PairConfig`](crate::engine_config::pair_config::PairConfig) | Intra-night pair pre-filter (angular separation, time gap). |
//! | `triplets` | [`TripletConfig`](crate::engine_config::triplet_config::TripletConfig) | Intra-night triplet generation (acceleration term). |
//! | `edges` | [`EdgeConfig`](crate::engine_config::edge_config::EdgeConfig) | Inter-night candidate search, cost variant, ML post-filter, ONNX threads. |
//! | `solver` | [`SolverConfig`](crate::engine_config::solver_config::SolverConfig) | Solver selection, beam width, pruning budgets. |
//!
//! Loading is done via
//! [`load_engine_config_validated`](crate::engine_config::load_engine_config_validated).
//!
//! # Persistence
//!
//! The engine uses a **journal-based** persistence model via
//! [`PersistenceManager`](crate::persistence::PersistenceManager). Each night
//! appends an immutable delta (edge operations, new seeds, new alerts) to an
//! on-disk journal. This allows efficient incremental updates and replay.
//!
//! Serialisation uses [`bitcode`](https://crates.io/crates/bitcode) with
//! optional compression (LZ4 or Zstd). The layout on disk is managed by
//! [`PersistenceLayout`](crate::persistence::layout::PersistenceLayout).
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`alerts`] | Alert data model ([`Alert`], [`AlertKey`]) and per-night store. |
//! | [`seeding`] | Intra-night seed construction, spatial indexing, tangent-plane projection. |
//! | [`graph`] | Inter-night linkage DAG, edge construction, features, and ML post-filter. |
//! | [`solver`] | Solver abstractions, bounded-beam search, connected components. |
//! | [`pipeline`] | Stage-based execution orchestrator and hooks. |
//! | [`persistence`] | Journal-based on-disk persistence, compression, layout, manifest. |
//! | [`engine_config`] | Configuration schema, loading, and validation. |
//! | [`trajectory`] | Trajectory hypothesis and track identifier types. |
//! | [`astro_math`] | Spherical geometry, tangent projection, 2D linear algebra. |
//! | [`error`] | Top-level error type [`EngineError`](crate::error::EngineError). |
//! | [`night_id`] | [`NightId`](crate::night_id::NightId) and pairing mode. |
//! | [`spacetime_bucket`] | Spatio-temporal bucketing used by the spatial binner. |
//! | [`display_format`] | Formatting helpers for human-readable diagnostics. |
//!
//! The following unit type aliases are defined at the crate root:
//! [`MJDTT`], [`Radian`], [`Arcsec`].

pub mod alerts;
pub mod astro_math;
pub mod display_format;
pub mod engine_config;
pub mod error;
pub mod graph;
pub mod night_id;
pub mod persistence;
pub mod pipeline;
pub mod seeding;
pub mod solver;
pub mod spacetime_bucket;
pub mod trajectory;

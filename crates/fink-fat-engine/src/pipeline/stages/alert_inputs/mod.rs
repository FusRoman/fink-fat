//! IngestNights stage: load alerts from an input URI into the runtime `AlertStore`.
//!
//! Overview
//! --------
//! This stage is responsible for ingesting a batch of alerts (typically a Parquet dataset)
//! from an [`InputUri`](crate::pipeline::stages::alert_inputs::input_uri::InputUri) specified
//! in the [`PipelinePlan`](crate::pipeline::PipelinePlan), and materializing them into the
//! runtime [`AlertStore`](crate::alerts::store::AlertStore).
//!
//! In addition to data ingestion, this stage participates in the **hierarchical
//! progress reporting system** of the pipeline through a [`StageProgress`](crate::pipeline::hooks::StageProgress).
//! The stage does not depend on any specific UI or CLI implementation.
//! Instead, it reports structured progress events to an abstract sink,
//! allowing the caller (e.g. CLI) to render progress bars, logs, or metrics.
//!
//! Execution Model
//! ---------------
//! The ingestion pipeline performs the following logical steps:
//!
//! 1. Read the input URI (`ctx.plan.inputs.alerts_uri`).
//! 2. Load alerts synchronously via [`load_alerts_sync`],
//!    which internally relies on DataFusion and the `object_store` abstraction.
//! 3. Normalize the resulting store (sort alerts per night, compute/refresh keys) via
//!    `AlertStore::sort_each_night_and_rekey()`.
//! 4. Derive the runtime `NightWindow` from the ingested alerts.
//! 5. Merge the newly ingested store into `ctx.runtime_state.alert_store`.
//! 6. Return counters (`n_alerts`, `n_nights`) for stage reporting.
//!
//! Progress Reporting
//! ------------------
//! This stage reports its internal progress through the [`StageProgress`](crate::pipeline::hooks::StageProgress) provided
//! by the pipeline runner.
//!
//! The stage defines **four logical units of work** and calls:
//!
//! - `stage_sink.set_total(4)` at the beginning,
//! - `stage_sink.inc(1)` after each completed logical block.
//!
//! These four units correspond to:
//!
//! 1. Loading alerts from the input URI.
//! 2. Sorting and rekeying alerts per night.
//! 3. Updating the runtime `NightWindow`.
//! 4. Merging the new `AlertStore` into runtime state.
//!
//! The exact rendering of this progress (progress bars, logs, metrics)
//! is delegated to the `StageProgress` implementation.
//!
//! This design ensures:
//!
//! - The engine remains **UI-agnostic**.
//! - Stages can define meaningful internal milestones.
//! - Future CLI implementations can expose hierarchical progress bars.
//!
//! Data Sources
//! ------------
//! The actual I/O and decoding is delegated to the `alert_loader` module, which uses:
//!
//! - **DataFusion** as the query/execution engine for Parquet.
//! - **object_store** backends for storage access (local filesystem, HTTP(S), HDFS, ...).
//! - A projection schema (`AlertParquetColumns`) to read only the columns required by the engine.
//!
//! Notes on Ordering and Keys
//! --------------------------
//! Downstream stages (BuildSeeds, BuildEdges, etc.) assume that alerts are stored
//! in a stable order per night and that alert identifiers (keys) are consistent.
//!
//! `sort_each_night_and_rekey()` ensures:
//!
//! - Deterministic per-night ordering.
//! - Stable alert keys suitable for indexing and persistence.
//!
//! Runtime State Mutations
//! -----------------------
//! This stage mutates the pipeline runtime state in two places:
//!
//! - `ctx.runtime_state.window` is set from the ingested alerts.
//! - `ctx.runtime_state.alert_store` is updated via an in-place merge.
//!
//! Expected Invariants After Success
//! ---------------------------------
//! - `ctx.runtime_state.window.is_some()`
//! - `ctx.runtime_state.alert_store` contains the nights and alerts loaded from the URI.
//! - Alert ordering and keying are normalized.
//!
//! Error Handling
//! --------------
//! This stage returns [`EngineError::StageFailed`] if:
//!
//! - Alert loading fails (I/O, schema mismatch, missing columns, decoding errors, etc.).
//!
//! Extension Points
//! ----------------
//! - If multiple input URIs are supported in the future, this stage may iterate
//!   over several stores and merge them sequentially.
//! - If partial loading based on `NightWindow` becomes necessary, filtering
//!   may be performed either via DataFusion predicate pushdown or post-load filtering.
//! - Finer-grained progress reporting (e.g. per-night ingestion) can be implemented
//!   by introducing nested `StageProgress::child()` scopes.

pub mod alert_loader;
pub mod input_uri;
pub mod storage;

use crate::{
    error::EngineError,
    night_id::PairingMode,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{
            PipelineStage,
            alert_inputs::alert_loader::{AlertParquetColumns, LoadAlertsError, load_alerts_sync},
            run_stage,
        },
    },
};

/// Run the `IngestNights` pipeline stage.
///
/// Overview
/// --------
/// This function executes the `IngestNights` stage within the pipeline lifecycle.
/// It loads alerts from the input URI specified in the [`PipelinePlan`](crate::pipeline::PipelinePlan), normalizes
/// per-night ordering and alert keying, updates the runtime `NightWindow`, and
/// merges the resulting [`AlertStore`](crate::alerts::store::AlertStore) into the pipeline [`RuntimeState`](crate::persistence::runtime_state::RuntimeState).
///
/// This stage is invoked through `run_stage`, which:
/// - emits structured lifecycle hooks (`on_stage_start`, `on_stage_end`),
/// - measures execution time,
/// - collects stage-level counters,
/// - and integrates hierarchical progress reporting via [`StageProgress`](crate::pipeline::hooks::StageProgress).
///
/// Synchronous Boundary
/// --------------------
/// The pipeline runner is synchronous at this level. Although the underlying
/// loading implementation (`load_alerts_sync`) may internally rely on asynchronous
/// runtimes (DataFusion + `object_store`), this complexity is encapsulated and
/// does not leak outside the stage boundary.
///
/// Progress Reporting Contract
/// ---------------------------
/// The `stage_sink` argument represents the **stage-level progress scope**,
/// created by the pipeline runner.
///
/// This stage defines a deterministic internal progress structure:
///
/// - `stage_sink.set_total(4)` declares four logical work units.
/// - `stage_sink.inc(1)` is called after each completed milestone.
///
/// The four milestones correspond to:
///
/// 1. Alert loading from the input URI.
/// 2. Sorting and rekeying alerts per night.
/// 3. Updating the runtime `NightWindow`.
/// 4. Merging the new `AlertStore` into runtime state.
///
/// The exact rendering of this progress (progress bars, logs, metrics, etc.)
/// is determined by the concrete implementation of [`StageProgress`](crate::pipeline::hooks::StageProgress).
///
/// If the provided sink is a no-op implementation, progress reporting has
/// zero runtime cost beyond the method calls.
///
/// Arguments
/// ---------
/// * `ctx` – Mutable pipeline context containing:
///   - the execution plan,
///   - runtime state,
///   - persistence layer,
///   - engine configuration.
/// * `hooks` – Stage lifecycle hooks used for structured logging and reporting.
/// * `stage_sink` – Progress scope corresponding to this stage. May create
///   nested scopes via `child()` if finer-grained progress reporting is desired.
///
/// Return
/// ------
/// * `Ok(StageReport)` – Execution time and stage counters (`n_alerts`, `n_nights`).
/// * `Err(EngineError::StageFailed)` – If alert loading fails or an invariant
///   is violated during ingestion.
///
/// Side Effects
/// ------------
/// On success, this function guarantees:
///
/// - `ctx.runtime_state.window` reflects the ingested nights.
/// - `ctx.runtime_state.alert_store` contains normalized alert data.
/// - Alert keys and per-night ordering are deterministic.
/// - Stage counters reflect post-normalization alert statistics.
///
/// Notes
/// -----
/// - Progress units are **logical milestones**, not proportional to alert count.
///   If finer-grained progress (e.g., per-batch or per-night) is required,
//   nested progress scopes may be introduced using `StageProgress::child()`.
///
/// - The stage does not perform partial ingestion or filtering by
///   `NightWindow`. All alerts provided by the input URI are loaded.
pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::IngestNights,
        hooks,
        StageMeta {
            label: PipelineStage::IngestNights.label().to_string(),
            total: Some(4),
        },
        |stage_sink| {
            // -----------------------------------------------------------------
            // 1) Retrieve the input URI from the pipeline plan.
            // -----------------------------------------------------------------
            //
            // The plan defines where alert data is sourced from (local file, HTTP(S), HDFS...).
            // The loader will resolve this URI into an object_store backend and read Parquet
            // via DataFusion.
            let uri = &ctx.plan.inputs.alerts_uri;

            // -----------------------------------------------------------------
            // 2) Load alerts into a fresh AlertStore.
            // -----------------------------------------------------------------
            //
            // `AlertParquetColumns::default()` defines the projected schema: only the columns
            // required by the engine are read, which reduces I/O and memory.
            //
            // `load_alerts_sync` wraps the async execution (object_store + DataFusion)
            // behind a sync API so the pipeline runner stays sync.
            let mut new_alert_store = load_alerts_sync(uri, AlertParquetColumns::default())
                .map_err(|e| match e {
                    LoadAlertsError::NotFound(_) => EngineError::StageFailed {
                        stage: PipelineStage::IngestNights,
                        message: format!("file not found: {}", uri.0),
                    },
                    _ => EngineError::StageFailed {
                        stage: PipelineStage::IngestNights,
                        message: format!("failed to load alerts from {}: {e:?}", uri.0),
                    },
                })?;
            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 3) Normalize per-night ordering and refresh alert keys.
            // -----------------------------------------------------------------
            //
            // Downstream stages frequently rely on stable ordering (determinism) and on keys
            // for indexing and persistence. This step ensures the store is in a canonical form.
            new_alert_store.sort_each_night_and_rekey();
            stage_sink.inc(1);

            // Collect counters *after* normalization, since normalization may drop/transform
            // some internal representation depending on implementation details.
            let n_new_alerts = new_alert_store.n_alerts();
            let n_new_nights = new_alert_store.n_nights();

            // -----------------------------------------------------------------
            // 4) Update the runtime night window.
            // -----------------------------------------------------------------
            //
            // The night window is derived from which nights are present in the ingested store.
            // Later stages use it to iterate deterministically over the active nights.
            // In the case where we ingest a single night, the window is effectively a singleton (window.is_single == true).
            let last_night =
                new_alert_store
                    .last_night()
                    .ok_or_else(|| EngineError::StageFailed {
                        stage: PipelineStage::IngestNights,
                        message: "ingested AlertStore contains no nights".to_string(),
                    })?;
            let max_gap = ctx.engine_config.max_gap_nights();
            // The ok will normally never trigger because the max_gap has already been validated at config level.
            ctx.runtime_state.window = PairingMode::single_night(last_night, max_gap).ok();
            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 5) Merge the newly loaded AlertStore into the runtime state.
            // -----------------------------------------------------------------
            //
            // The chosen semantics here are "merge in place", typically meaning:
            // - replace existing nights with newly loaded data,
            // - or union nights depending on your AlertStore implementation.
            //
            // The key property is that after this call, runtime state contains the alerts
            // needed for subsequent stages.
            ctx.runtime_state
                .alert_store
                .merge_in_place(new_alert_store);
            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 6) Emit stage counters.
            // -----------------------------------------------------------------
            Ok(vec![
                ("n_alerts", n_new_alerts as u64),
                ("n_nights", n_new_nights as u64),
            ])
        },
    )
}

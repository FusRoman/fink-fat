//! BuildSeeds stage: generate intra-night seeds (pairs → triplets → `SeedNodeOwned`) from alerts.
//!
//! Overview
//! --------
//! This stage constructs **intra-night seeds** from the alerts already loaded in the
//! [`AlertStore`](crate::persistence::alert_store::AlertStore) (stored in `ctx.runtime_state.alert_store`).
//!
//! In addition to seed generation, this stage participates in the pipeline's
//! **hierarchical progress reporting** via the [`ProgressSink`](crate::pipeline::progress_sink::ProgressSink)
//! abstraction. The stage does not depend on any concrete UI (CLI progress bars, logs, metrics);
//! it only emits structured progress events to the provided sink.
//!
//! Seed-building pipeline
//! ----------------------
//! The seed-building pipeline is:
//!
//! 1. **Bucketization** of alerts in (space, time) using:
//!    - [`HealpixBinner`](crate::spacetime_bucket::healpix_binner::HealpixBinner) for sky partitioning,
//!    - [`UniformTimeBinner`](crate::spacetime_bucket::uniform_time_binner::UniformTimeBinner) for time partitioning,
//!    - [`build_alert_bucket_index`](crate::spacetime_bucket::bucket::build_alert_bucket_index) to build the index.
//! 2. **Pair generation** using [`pairs::generate_pairs`](crate::seeding::pairs::generate_pairs).
//! 3. **Triplet generation** from those pairs using
//!    [`triplets::generate_triplets_from_pairs`](crate::seeding::triplets::generate_triplets_from_pairs).
//! 4. **Feature extraction** to build `SeedNode<'alert_lf>` that **borrow alerts**
//!    with [`triplets::extract_triplet_features`](crate::seeding::triplets::extract_triplet_features).
//! 5. **Ownership conversion**: immediately convert each borrowed `SeedNode<'_>` into a
//!    [`SeedNodeOwned`](crate::persistence::seed_node::SeedNodeOwned) using
//!    [`SeedNode::to_owned`](crate::seeding::seed_node::SeedNode::to_owned), and store it into
//!    `ctx.runtime_state.seed_store`.
//!
//! Why borrowed → owned?
//! --------------------
//! Most of the seeding code works on `&Alert` references for performance and to avoid
//! repeated allocation/copies while computing geometric and photometric features.
//! This naturally produces `SeedNode<'alert_lf>` that contains `Vec<&Alert>` members.
//!
//! However, `RuntimeState` is **not lifetime-parameterized**, and the pipeline persistence
//! layer needs fully owned, serializable objects. Therefore, `BuildSeeds` performs the
//! borrowed computation locally, then **immediately converts** to an owned representation
//! (`SeedNodeOwned`) containing only stable identifiers (e.g. `AlertKey`) and owned core data.
//!
//! Progress reporting
//! ------------------
//! The `BuildSeeds` stage uses `ProgressSink` in a **two-level model**:
//!
//! - **Stage-level progress**: `1 unit = 1 processed night`.
//!   The stage calls `stage_sink.set_total(n_nights)` once, then `stage_sink.inc(1)`
//!   after finishing each night.
//!
//! - **Night-level sub-scope** (recommended): for each processed night, the stage creates
//!   a nested scope with `stage_sink.child(StageMeta { ... })` and reports milestone-level
//!   progress within that night.
//!
//! Night milestones are intentionally modeled as a small, deterministic set of logical steps:
//!
//! 1. Compute `t0` and initialize time binning.
//! 2. Build the (space, time) bucket index.
//! 3. Generate pairs.
//! 4. Generate triplets and extract borrowed seed features.
//! 5. Convert to owned seeds and insert into `seed_store`.
//!
//! This design provides meaningful UI feedback without tying progress to potentially huge
//! intermediate cardinalities (pairs/triplets) that can vary widely across nights.
//!
//! Inputs
//! ------
//! - `ctx.runtime_state.window`: [`NightWindow`](crate::night_id::NightWindow) defining which nights are processed.
//! - `ctx.runtime_state.alert_store`: per-night alert vectors.
//! - `ctx.engine_config.pairs`: pair generation configuration.
//! - `ctx.engine_config.triplets`: triplet generation configuration.
//! - `ctx.engine_config.healpix_depth`: Healpix resolution for spatial binning.
//! - `ctx.engine_config.time_binner_width`: time bin width for time binning.
//!
//! Outputs
//! -------
//! - `ctx.runtime_state.seed_store`: filled with `SeedNodeOwned` per processed night.
//! - Returns a [`StageReport`](crate::pipeline::hooks::StageReport) with counters:
//!   - `nights`, `alerts`, `pairs`, `triplets`, `seeds`.
//!
//! Error handling
//! --------------
//! This stage fails with [`EngineError::StageFailed`](crate::error::EngineError::StageFailed) if:
//! - no `NightWindow` is present in runtime state,
//! - a processed night contains zero alerts (cannot derive `t0` for time binning).
//!
//! Determinism & performance notes
//! -------------------------------
//! - Spatial binning uses a stable Healpix configuration, reused across nights.
//! - Time binning uses per-night `t0` derived from the night’s alerts via `AlertSlice::get_t0()`,
//!   ensuring consistent bin alignment within each night.
//! - The stage keeps borrowed artifacts (`BucketIndex`, pairs, triplets, borrowed seeds)
//!   **local to the loop iteration** to minimize memory usage and avoid lifetime leakage.
//! - Progress reporting is based on deterministic milestones; it can be disabled by providing
//!   a no-op sink with zero functional impact on the stage.
//!
//! Extension points
//! ----------------
//! - If you later add additional seed families (e.g. higher-order seeds), they should be built
//!   in the same pattern: compute borrowed features → convert to owned → persist in `seed_store`.
//! - If finer-grained progress is required (e.g. per-bucket or per-pair), introduce additional
//!   nested scopes under the per-night sink using `ProgressSink::child()`.
//! - If you need per-night instrumentation, insert timers around the bucketization/pairs/triplets steps.

use crate::{
    alerts::AlertSlice, error::EngineError, pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        progress_sink::ProgressSink,
        stages::{PipelineStage, run_stage},
    }, seeding::{pairs, seed_node::SeedNode, triplets}, spacetime_bucket::{
        bucket::build_alert_bucket_index, healpix_binner::HealpixBinner,
        uniform_time_binner::UniformTimeBinner,
    }
};

/// Run the `BuildSeeds` pipeline stage.
///
/// Overview
/// --------
/// This function executes the `BuildSeeds` stage for the nights specified by the runtime
/// [`NightWindow`]. For each processed night, it:
///
/// - bucketizes alerts in (space, time),
/// - generates candidate pairs,
/// - generates candidate triplets from those pairs,
/// - extracts seed features into borrowed `SeedNode<'_>`,
/// - converts each borrowed seed to an owned representation (`SeedNodeOwned`),
/// - stores owned seeds into `ctx.runtime_state.seed_store`.
///
/// This stage is invoked via [`run_stage`], which integrates lifecycle hooks and
/// stage-level timing/counters.
///
/// Progress reporting contract
/// ---------------------------
/// The `stage_sink` argument is the stage-level progress scope, created by the pipeline runner.
///
/// This stage reports progress using a two-level structure:
///
/// - Stage-level:
///   - `stage_sink.set_total(n_nights)`
///   - `stage_sink.inc(1)` once per completed night.
///
/// - Night-level:
///   - For each processed night, a nested scope is created using
///     `stage_sink.child(StageMeta { label: ..., total: Some(5) })`.
///   - The night scope reports five deterministic milestones, each incrementing by `1`.
///
/// This model yields stable and meaningful progress reporting independent of the potentially
/// large and highly variable number of generated pairs/triplets.
///
/// Arguments
/// ---------
/// * `ctx` – Mutable pipeline context containing the plan, runtime state, persistence, and config.
/// * `hooks` – Stage lifecycle hooks for structured reporting.
/// * `stage_sink` – Progress scope for this stage; nested scopes may be created for per-night
///   progress reporting.
///
/// Return
/// ------
/// * `Ok(StageReport)` – Execution time and counters (`nights`, `alerts`, `pairs`, `triplets`, `seeds`).
/// * `Err(EngineError::StageFailed)` – If preconditions are not met or if a night is invalid.
///
/// Side effects
/// ------------
/// On success:
///
/// - `ctx.runtime_state.seed_store` contains owned seeds for each processed night.
/// - Previously existing seed data for a given night may be replaced depending on
///   `seed_store.insert` semantics.
/// - Stage counters reflect the total work performed across all processed nights.
///
/// Notes
/// -----
/// - Progress units are **logical milestones**, not proportional to alert count.
///   If required, the progress model can be changed to use finer-grained units
///   (e.g. per-night alert count) while retaining deterministic ordering.
/// - All borrowed intermediate structures (bucket index, pairs, triplets, borrowed seeds)
///   remain local to each loop iteration to avoid lifetime leakage into runtime state.
pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
    stage_sink: &dyn ProgressSink,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::BuildSeeds,
        hooks,
        StageMeta {
            label: PipelineStage::BuildSeeds.label().to_string(),
            total: None,
        },
        stage_sink,
        |stage_sink| {
            // -----------------------------------------------------------------
            // 0) Preconditions: a BuildSeeds run requires a NightWindow.
            // -----------------------------------------------------------------
            let window =
                ctx.runtime_state
                    .window
                    .as_ref()
                    .ok_or_else(|| EngineError::StageFailed {
                        stage: PipelineStage::BuildSeeds,
                        message: "no night window found in runtime state".to_string(),
                    })?;

            let pair_cfg = &ctx.engine_config.pairs;
            let triplet_cfg = &ctx.engine_config.triplets;
            let spatial_binner = HealpixBinner::new(ctx.engine_config.healpix_depth);

            // -----------------------------------------------------------------
            // Progress model (stage-level)
            // ----------------------------
            // - 1 unit = 1 processed night.
            //
            // This is deterministic and avoids tying progress to potentially huge
            // intermediate cardinalities (pairs/triplets), which can vary widely.
            // -----------------------------------------------------------------
            let nights_to_process: Vec<_> = ctx.runtime_state.alert_store.nights_sorted();
            stage_sink.set_total(nights_to_process.len() as u64);

            // -----------------------------------------------------------------
            // 3) Global counters (reported via StageReport).
            // -----------------------------------------------------------------
            let mut total_alerts: u64 = 0;
            let mut total_pairs: u64 = 0;
            let mut total_triplets: u64 = 0;
            let mut total_seeds: u64 = 0;

            // -----------------------------------------------------------------
            // 4) Process each night in the requested window.
            // -----------------------------------------------------------------
            for (night_id, alerts) in ctx.runtime_state.alert_store.night_window_iter(*window) {
                total_alerts += alerts.len() as u64;

                // -------------------------------------------------------------
                // Night sub-scope (optional but recommended)
                // -------------------------------------------------------------
                // 5 logical milestones:
                // 1) t0 + time binner
                // 2) bucket index
                // 3) pairs
                // 4) triplets + features
                // 5) to_owned + insert
                let night_sink = stage_sink.child(StageMeta {
                    label: format!("night {night_id}"),
                    total: Some(5),
                });
                night_sink.set_total(5);

                // 4.1) Determine per-night `t0` for time binning.
                let t0 = alerts.get_t0().ok_or_else(|| EngineError::StageFailed {
                    stage: PipelineStage::BuildSeeds,
                    message: format!(
                        "night {night_id} contains no alerts, cannot determine t0 for time binning"
                    ),
                })?;
                let time_binner = UniformTimeBinner::new(t0, ctx.engine_config.time_binner_width);
                night_sink.inc(1);

                // 4.2) Build the (space, time) bucket index.
                let bucket_index = build_alert_bucket_index(alerts, &spatial_binner, &time_binner);
                night_sink.inc(1);

                // 4.3) Generate candidate pairs.
                let ps =
                    pairs::generate_pairs(&bucket_index, &spatial_binner, &time_binner, pair_cfg);
                total_pairs += ps.len() as u64;

                let pair_seeds = pairs::extract_pair_features(&ps, night_id, None);
                night_sink.inc(1);

                // 4.4) Generate triplets + extract features (borrowed).
                let ts = triplets::generate_triplets_from_pairs(
                    &bucket_index,
                    &spatial_binner,
                    &time_binner,
                    triplet_cfg,
                    &ps,
                );
                total_triplets += ts.len() as u64;

                let triplets_seeds: Vec<SeedNode<'_>> =
                    triplets::extract_triplet_features(&ts, night_id);
                night_sink.inc(1);

                // 4.5) Convert to owned + sort + store.
                // sort is important for the edge builder as it relies on the right nodes to be sorted by epoch_mid for efficient edge generation.
                // a dichotomic search is performed to find the relevant right nodes, and if they are not sorted by epoch_mid, we would need to sort them at each iteration of the edge builder, which would be very costly.
                let mut all_seeds: Vec<SeedNode<'_>> =
                    Vec::with_capacity(pair_seeds.len() + triplets_seeds.len());
                all_seeds.extend(pair_seeds);
                all_seeds.extend(triplets_seeds);
                all_seeds.sort();
                let owned_seeds = all_seeds.iter().map(|s| s.to_owned()).collect::<Vec<_>>();
                total_seeds += owned_seeds.len() as u64;

                ctx.runtime_state.seed_store.insert(night_id, owned_seeds);
                night_sink.inc(1);

                night_sink.finish();

                // 1 unit = 1 processed night
                stage_sink.inc(1);
            }

            Ok(vec![
                ("nights", nights_to_process.len() as u64),
                ("alerts", total_alerts),
                ("pairs", total_pairs),
                ("triplets", total_triplets),
                ("seeds", total_seeds),
            ])
        },
    )
}

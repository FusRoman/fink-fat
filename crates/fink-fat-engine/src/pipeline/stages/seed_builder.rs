//! BuildSeeds stage: generate intra-night seeds (pairs → triplets → `SeedNodeOwned`) from alerts.
//!
//! Overview
//! --------
//! This stage constructs **intra-night seeds** from the alerts already loaded in the
//! [`AlertStore`](crate::alerts::store::AlertStore) (stored in `ctx.runtime_state.alert_store`).
//!
//! In addition to seed generation, this stage participates in the pipeline's
//! **hierarchical progress reporting** via the [`StageProgress`]
//! abstraction. The stage does not depend on any concrete UI (CLI progress bars, logs, metrics);
//! it only emits structured progress events to the provided sink.
//!
//! Seed-building pipeline
//! ----------------------
//! The seed-building pipeline runs in two phases:
//!
//! **Parallel phase** — one task per night, executed concurrently via Rayon
//! (`par_iter`). Each task:
//!
//! 1. Bucketizes alerts in (space, time) using:
//!    - [`HealpixBinner`] for sky partitioning,
//!    - [`UniformTimeBinner`] for time partitioning,
//!    - [`build_alert_bucket_index`] to build the index.
//! 2. Streams valid pairs using [`pairs::stream_pairs`].
//! 3. Extends each accepted pair into triplets immediately using
//!    [`triplets::stream_triplets_from_pair`].
//! 4. Converts accepted pairs and triplets into [`SeedNode`]s using a
//!    thread-local [`crate::seeding::store::SeedStore`] for provisional key
//!    allocation.
//! 5. Sorts seeds by `epoch_mid` (required by the edge builder's dichotomic search).
//!
//! **Sequential merge phase** — runs after all parallel tasks have completed.
//! For each night's result, `finalize_night_seeds`:
//!
//! 1. Reserves `n` consecutive globally-unique seed IDs via
//!    [`crate::seeding::store::SeedStore::alloc_ids_batch`].
//! 2. Overwrites provisional keys with real [`SeedKey`]s.
//! 3. Inserts finalized seeds into `ctx.runtime_state.seed_store`.
//!
//! Why a two-phase approach?
//! -------------------------
//! Night-level work (bucketization, pair generation, triplet generation, feature
//! extraction) is CPU-intensive and fully independent across nights — a natural
//! candidate for parallelism. However, key allocation in
//! [`crate::seeding::store::SeedStore`] requires exclusive mutable access and
//! cannot be shared safely across threads.
//!
//! The two-phase design resolves this tension:
//!
//! - The parallel phase uses a **thread-local** temporary `SeedStore` per night,
//!   so key allocation is entirely local and requires no synchronization.
//! - The sequential merge phase assigns **globally-unique** IDs by calling
//!   [`crate::seeding::store::SeedStore::alloc_ids_batch`] (one call per night)
//!   on the real pipeline store, which is safe because only one thread runs at
//!   this point.
//!
//! Progress reporting
//! ------------------
//! The `BuildSeeds` stage uses `StageProgress` in a **two-level model**:
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
//! 3. Stream pairs and extend them to triplets.
//! 4. Convert accepted pairs and triplets into seeds.
//! 5. Sort the seeds and insert them into `seed_store`.
//!
//! This design provides meaningful UI feedback without tying progress to potentially huge
//! intermediate cardinalities (pairs/triplets) that can vary widely across nights.
//!
//! Inputs
//! ------
//! - `ctx.runtime_state.get_new_night_ids()`: list of night IDs ingested in the current run.
//! - `ctx.runtime_state.alert_store`: per-night alert vectors.
//! - `ctx.engine_config.pairs`: pair generation configuration.
//! - `ctx.engine_config.triplets`: triplet generation configuration.
//! - `ctx.engine_config.healpix_depth`: Healpix resolution for spatial binning.
//! - `ctx.engine_config.time_binner_width`: time bin width for time binning.
//!
//! Outputs
//! -------
//! - `ctx.runtime_state.seed_store`: filled with `SeedNodeOwned` per processed night.
//! - Returns a [`StageReport`] with counters:
//!   - `nights`, `alerts`, `pairs`, `triplets`, `seeds`.
//!
//! Error handling
//! --------------
//! This stage fails with [`EngineError::StageFailed`] if:
//! - `get_new_night_ids()` returns `None` (i.e. `IngestNight` has not run yet),
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
//!   nested scopes under the per-night sink using `StageProgress::child()`.
//! - If you need per-night instrumentation, insert timers around the bucketization/pairs/triplets steps.

use rayon::prelude::*;

use crate::{
    Alert,
    alerts::AlertSlice,
    engine_config::{
        pair_config::PairConfig,
        seeding_config::{HoughSeedingConfig, SeedingMethod},
        triplet_config::TripletConfig,
    },
    error::EngineError,
    night_id::NightId,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageProgress, StageReport},
        stages::{PipelineStage, run_stage},
    },
    seeding::{
        SeedKey, SeedNode,
        hough::{self, HoughSeedStats},
        pairs,
        store::{SeedId, SeedStore},
        triplets,
    },
    spacetime_bucket::{
        bucket::build_alert_bucket_index, healpix_binner::HealpixBinner,
        uniform_time_binner::UniformTimeBinner,
    },
};

// ─────────────────────────────────────────────────────────────────
// Per-night helpers
// ─────────────────────────────────────────────────────────────────

/// Per-night seed compilation result produced by [`process_one_night`].
struct NightSeedResult {
    night_id: NightId,
    /// Candidate seeds sorted by `epoch_mid`, carrying provisional keys from
    /// a thread-local [`SeedStore`]. Keys are reassigned in
    /// [`finalize_night_seeds`] before insertion into the pipeline store.
    seeds: Vec<SeedNode>,
    /// Number of alerts in this night.
    n_alerts: u64,
    /// Number of intra-night pairs generated.
    n_pairs: u64,
    /// Number of intra-night triplets generated.
    n_triplets: u64,
}

struct ProcessOneNightParams<'a> {
    spatial_binner: &'a HealpixBinner,
    pair_cfg: &'a PairConfig,
    triplet_cfg: &'a TripletConfig,
    hough_cfg: &'a HoughSeedingConfig,
    seeding_method: SeedingMethod,
    triplet_only: bool,
    time_binner_width: f64,
    night_sink: &'a dyn StageProgress,
}

fn process_one_night_hough(
    night_id: NightId,
    alerts: &[Alert],
    params: &ProcessOneNightParams<'_>,
) -> NightSeedResult {
    tracing::trace!(
        %night_id,
        n_alerts = alerts.len(),
        "processing night with Hough seeding method"
    );

    let (all_seeds, stats): (Vec<SeedNode>, HoughSeedStats) =
        hough::build_hough_seeds_for_night(alerts, night_id, params.hough_cfg, params.triplet_only);

    tracing::debug!(
        %night_id,
        n_velocity_hypotheses = stats.n_velocity_hypotheses,
        n_accumulator_bins = stats.n_accumulator_bins,
        n_peaks = stats.n_peaks,
        n_peaks_after_photometric_filter = stats.n_peaks_after_photometric_filter,
        n_pair_seeds = stats.n_pair_seeds,
        n_triplet_seeds = stats.n_triplet_seeds,
        n_night_seeds = all_seeds.len(),
        "hough seeding complete"
    );

    NightSeedResult {
        night_id,
        seeds: all_seeds,
        n_alerts: alerts.len() as u64,
        n_pairs: stats.n_pair_seeds,
        n_triplets: stats.n_triplet_seeds,
    }
}

fn process_one_night_pair_triplet(
    night_id: NightId,
    alerts: &[Alert],
    params: &ProcessOneNightParams<'_>,
) -> Result<NightSeedResult, EngineError> {
    let n_alerts = alerts.len() as u64;

    // Milestone 1: determine per-night t0 and initialise time binning.
    let t0 = alerts.get_t0().ok_or_else(|| EngineError::StageFailed {
        stage: PipelineStage::BuildSeeds,
        message: format!(
            "night {night_id} contains no alerts, cannot determine t0 for time binning"
        ),
    })?;
    let time_binner = UniformTimeBinner::new(t0, params.time_binner_width);
    tracing::trace!(
        %night_id,
        t0,
        time_binner_width = params.time_binner_width,
        "t0 and time binner initialised"
    );
    params.night_sink.inc(1);

    // Milestone 2: build the (space, time) bucket index.
    let bucket_index = build_alert_bucket_index(alerts, params.spatial_binner, &time_binner);
    tracing::trace!(%night_id, n_buckets = bucket_index.buckets.len(), "bucket index built");
    params.night_sink.inc(1);

    // Milestone 3 + 4: stream pairs directly into triplet generation.
    let mut local_store = SeedStore::new();
    let mut all_seeds: Vec<SeedNode> = Vec::new();

    let mut triplet_stream_state = triplets::TripletPairStreamState::default();

    let mut n_triplets: u64 = 0;
    let mut n_pair_seeds_emitted: u64 = 0;
    let mut n_triplet_seeds_emitted: u64 = 0;
    let mut n_pairs_with_triplet_support: u64 = 0;

    let pair_stats = pairs::stream_pairs(
        &bucket_index,
        params.spatial_binner,
        &time_binner,
        params.pair_cfg,
        |pair| {
            let mut emitted_triplet_from_pair = false;
            let trip_stats = triplets::stream_triplets_from_pair(
                &bucket_index,
                params.spatial_binner,
                &time_binner,
                params.triplet_cfg,
                &mut triplet_stream_state,
                pair,
                |triplet| {
                    emitted_triplet_from_pair = true;
                    all_seeds.push(SeedNode::from_triplet(
                        &mut local_store,
                        night_id,
                        triplet.a,
                        triplet.b,
                        triplet.c,
                    ));
                    n_triplet_seeds_emitted += 1;
                },
            );

            n_triplets += trip_stats.n_triplets;

            if emitted_triplet_from_pair {
                n_pairs_with_triplet_support += 1;
            } else if !params.triplet_only
                && let Some(seed) =
                    SeedNode::from_pair(&mut local_store, night_id, pair.a, pair.b, None)
            {
                all_seeds.push(seed);
                n_pair_seeds_emitted += 1;
            }
        },
    );
    params.night_sink.inc(1);
    params.night_sink.inc(1);

    let n_pairs = pair_stats.n_pairs;

    tracing::debug!(
        %night_id,
        n_pairs,
        n_triplets,
        n_pairs_with_triplet_support,
        n_pair_seeds_emitted,
        n_triplet_seeds_emitted,
        "streamed pair->triplet generation complete"
    );

    // Milestone 5: combine and sort.
    all_seeds.sort();
    tracing::debug!(%night_id, n_night_seeds = all_seeds.len(), "seeds combined and sorted");
    params.night_sink.inc(1);

    Ok(NightSeedResult {
        night_id,
        seeds: all_seeds,
        n_alerts,
        n_pairs,
        n_triplets,
    })
}

/// Process one observation night: bucketize, stream pairs and triplets, and extract seed features.
///
/// Uses a thread-local [`SeedStore`] for provisional key allocation. Resulting
/// seeds carry placeholder keys that must be replaced with real globally-unique
/// keys by [`finalize_night_seeds`] before insertion into the pipeline seed store.
///
/// Arguments
/// ---------
/// * `night_id` – Identifier of the night being processed.
/// * `alerts` – Alert slice for this night.
/// * `spatial_binner` – Spatial partitioner for (space, time) bucket assignment.
/// * `pair_cfg` – Pair generation configuration.
/// * `triplet_cfg` – Triplet generation configuration.
/// * `triplet_only` – If `true`, emit only triplet-derived seeds.
/// * `time_binner_width` – Time bin width in days.
/// * `night_sink` – Progress sink for the per-night sub-scope.
///
/// Return
/// ------
/// * `Ok(NightSeedResult)` – Alert, pair and triplet counts plus sorted seeds
///   with provisional keys.
/// * `Err(EngineError::StageFailed)` – If the alert slice is empty
///   (cannot determine `t0` for time binning).
fn process_one_night(
    night_id: NightId,
    alerts: &[Alert],
    params: &ProcessOneNightParams<'_>,
) -> Result<NightSeedResult, EngineError> {
    if alerts.is_empty() {
        return Err(EngineError::StageFailed {
            stage: PipelineStage::BuildSeeds,
            message: format!(
                "night {night_id} contains no alerts, cannot determine t0 for time binning"
            ),
        });
    }

    let result = match params.seeding_method {
        SeedingMethod::PairTriplet => process_one_night_pair_triplet(night_id, alerts, params)?,
        SeedingMethod::Hough => {
            params.night_sink.inc(5);
            process_one_night_hough(night_id, alerts, params)
        }
    };
    params.night_sink.finish();
    Ok(result)
}

/// Assign real globally-unique keys to seeds and insert them into the pipeline seed store.
///
/// Calls [`SeedStore::alloc_ids_batch`] to atomically reserve exactly `n`
/// consecutive IDs (where `n = result.seeds.len()`), overwrites each provisional
/// key with a real [`SeedKey`], then calls [`SeedStore::insert_vec_seed`].
///
/// Arguments
/// ---------
/// * `result` – Night seed result produced by [`process_one_night`].
/// * `seed_store` – Mutable pipeline seed store receiving the finalized seeds.
///
/// Return
/// ------
/// Number of seeds inserted (`result.seeds.len()`).
fn finalize_night_seeds(result: NightSeedResult, seed_store: &mut SeedStore) -> u64 {
    let n = result.seeds.len();
    if n == 0 {
        return 0;
    }
    let base_id: SeedId = seed_store.alloc_ids_batch(n);
    let mut seeds = result.seeds;
    for (i, seed) in seeds.iter_mut().enumerate() {
        seed.set_key(SeedKey {
            night_id: result.night_id,
            unique_id: base_id + i as SeedId,
        });
    }
    seed_store.insert_vec_seed(result.night_id, seeds);
    n as u64
}

// ─────────────────────────────────────────────────────────────────

/// Run the `BuildSeeds` pipeline stage.
///
/// Overview
/// --------
/// This function executes the `BuildSeeds` stage for the nights specified by the runtime
/// `NightWindow`. For each processed night, it:
///
/// - bucketizes alerts in (space, time),
/// - generates candidate pairs,
/// - generates candidate triplets from those pairs,
/// - extracts seed features into borrowed `SeedNode<'_>`,
/// - converts each borrowed seed to an owned representation (`SeedNodeOwned`),
/// - stores owned seeds into `ctx.runtime_state.seed_store`.
///
/// This stage is invoked via `run_stage`, which integrates lifecycle hooks and
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
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::BuildSeeds,
        hooks,
        StageMeta {
            label: PipelineStage::BuildSeeds.label().to_string(),
            total: None, // dynamic: determined by number of nights at runtime
        },
        |stage_sink| {
            let pair_cfg = &ctx.engine_config.pairs;
            let triplet_cfg = &ctx.engine_config.triplets;
            let hough_cfg = &ctx.engine_config.seeding.hough;
            let seeding_method = ctx.engine_config.seeding.method;
            let triplet_only = ctx.engine_config.seeding.triplet_only;
            let spatial_binner = HealpixBinner::new(ctx.engine_config.healpix_depth);
            let time_binner_width = ctx.engine_config.time_binner_width;

            // -----------------------------------------------------------------
            // Collect night IDs to process.
            // -----------------------------------------------------------------
            let nights_to_process: Vec<NightId> =
                ctx.runtime_state
                    .get_new_night_ids()
                    .ok_or_else(|| EngineError::StageFailed {
                        stage: PipelineStage::BuildSeeds,
                        message: "runtime state does not contain new night IDs\nThe stage IngestNight must be run before BuildSeeds".to_string(),
                    })?
                    .clone();
            stage_sink.set_total(nights_to_process.len() as u64);

            tracing::debug!(
                n_nights = nights_to_process.len(),
                healpix_depth = ctx.engine_config.healpix_depth,
                seeding_method = ?seeding_method,
                triplet_only,
                time_binner_width,
                "BuildSeeds starting",
            );

            // -----------------------------------------------------------------
            // Collect (NightId, &[Alert]) pairs upfront so Rayon can distribute
            // them across threads without holding a mutable borrow on ctx.
            // All night IDs are validated here; missing ones fail immediately.
            // -----------------------------------------------------------------
            let nights_with_alerts: Vec<(NightId, &[Alert])> = {
                let night_iter = ctx
                    .runtime_state
                    .alert_store
                    .night_iter(&nights_to_process)?;
                nights_to_process.iter().copied().zip(night_iter).collect()
            };

            // -----------------------------------------------------------------
            // Parallel phase: process each night independently.
            // Each task uses its own local SeedStore for provisional key
            // allocation — no shared mutable state is accessed.
            // Results are collected in input order (par_iter preserves order).
            // -----------------------------------------------------------------
            let results: Vec<NightSeedResult> = nights_with_alerts
                .par_iter()
                .map(|&(night_id, alerts)| {
                    let night_sink = stage_sink.child(StageMeta {
                        label: format!("night {night_id}"),
                        total: Some(5),
                    });
                    tracing::debug!(%night_id, n_alerts = alerts.len(), "processing night");
                    let params = ProcessOneNightParams {
                        spatial_binner: &spatial_binner,
                        pair_cfg,
                        triplet_cfg,
                        hough_cfg,
                        seeding_method,
                        triplet_only,
                        time_binner_width,
                        night_sink: &*night_sink,
                    };
                    process_one_night(night_id, alerts, &params)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // -----------------------------------------------------------------
            // Sequential merge phase: assign real global keys and insert seeds.
            // Requires exclusive access to seed_store; runs after all parallel
            // tasks have completed.
            // -----------------------------------------------------------------
            let mut total_alerts: u64 = 0;
            let mut total_pairs: u64 = 0;
            let mut total_triplets: u64 = 0;
            let mut total_seeds: u64 = 0;

            for result in results {
                total_alerts += result.n_alerts;
                total_pairs += result.n_pairs;
                total_triplets += result.n_triplets;
                let night_id = result.night_id;
                let n_seeds = finalize_night_seeds(result, &mut ctx.runtime_state.seed_store);
                total_seeds += n_seeds;
                tracing::debug!(%night_id, n_seeds, "seeds finalised and inserted");
                // 1 unit = 1 processed night
                stage_sink.inc(1);
            }

            tracing::debug!(
                total_nights = nights_to_process.len(),
                total_alerts,
                total_pairs,
                total_triplets,
                total_seeds,
                "BuildSeeds complete",
            );

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

//! Crash recovery of `fink-fat-stream`'s [`Driver`].
//!
//! The `tests/data` nights are turned into synthetic Rubin alerts
//! ([`AlertRecord`]s: a `visit` per exposure epoch, fluxes from the
//! magnitudes, arrival three minutes after the exposure) and replayed twice
//! through the driver, each run with its own storage directory:
//!
//! * **reference**: start to finish;
//! * **interrupted**: stopped mid-night — the driver is dropped without
//!   closing, as a killed process would leave it — then restarted from its
//!   storage with a fresh replay of the whole dump.
//!
//! Both must end on the same state: same branches (id-agnostic fingerprint,
//! see `common/`), same id counters, same last closed night, same archive.

mod common;

use common::{
    archived_hash, combine_unordered, load_fixture, night_fingerprint,
    stream::{day_obs_of, engine_config, storage_dir, synthetic_alerts},
};
use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    topocentric_kf::branching::{BranchCollection, read_archived_log},
};
use fink_fat_stream::{AlertRecord, Driver, DriverConfig, DriverStats, ReplaySource, StreamPaths};

/// Alerts per replay batch: one, so the driver can be stopped anywhere.
const BATCH_SIZE: usize = 1;
/// Nights closed before the interrupted run is stopped.
const STOP_AFTER_NIGHTS: usize = 8;

fn driver<'a>(
    alerts: &[AlertRecord],
    ctx: &'a KalmanContext,
    config: &'a EngineConfig,
) -> Driver<'a, 'a, ReplaySource> {
    let source = ReplaySource::from_records(alerts.to_vec(), BATCH_SIZE);
    Driver::new(source, DriverConfig::default(), ctx, config).expect("start driver")
}

/// Everything the two runs must agree on, read back from storage.
fn final_state(ctx: &KalmanContext, config: &EngineConfig) -> (String, String, u64) {
    let paths = StreamPaths::under(config);
    let collection = BranchCollection::load_snapshot_from_disk(&paths.snapshot, ctx, config)
        .expect("load final snapshot");
    let last_closed = collection
        .last_closed_night_id
        .expect("the run closed nights");
    let archived = read_archived_log(&paths.archive).expect("read archive");
    (
        night_fingerprint(last_closed, &collection),
        format!("{:?}", collection.ids),
        combine_unordered(archived.iter().map(archived_hash).collect()),
    )
}

#[test]
fn an_interrupted_replay_resumes_to_the_same_state() {
    let (tmp, obs_dataset, _) = load_fixture();
    let alerts = synthetic_alerts(&obs_dataset);

    // Reference: start to finish.
    let reference_config = engine_config(&storage_dir(&tmp, "reference"));
    let reference_ctx = reference_config.build_context();
    let reference: DriverStats = driver(&alerts, &reference_ctx, &reference_config)
        .run()
        .expect("reference run");
    assert_eq!(reference.n_accepted, alerts.len());
    assert_eq!(
        reference.n_unresolved_visits, 0,
        "the night datasets must resolve the X05 observer"
    );
    assert!(reference.n_nights_closed > STOP_AFTER_NIGHTS);

    // Interrupted: stop mid-night, once a visit of the open night was advanced.
    let config = engine_config(&storage_dir(&tmp, "interrupted"));
    let ctx = config.build_context();
    let mut first = driver(&alerts, &ctx, &config);
    let mut visits_at_last_close = 0;
    loop {
        assert!(
            first.step().expect("step"),
            "stopped before the stream ended"
        );
        let stats = first.stats();
        if stats.n_nights_closed < STOP_AFTER_NIGHTS {
            visits_at_last_close = stats.n_visits;
        } else if first.open_night_id().is_some() && stats.n_visits > visits_at_last_close {
            break;
        }
    }
    let stopped_at = first.open_night_id();
    drop(first);
    let journal = StreamPaths::under(&config).journal;
    assert!(
        std::fs::metadata(&journal).expect("journal exists").len() > 0,
        "the open night must be journaled"
    );

    // Restart from storage, replaying the whole dump again.
    let resumed_driver = driver(&alerts, &ctx, &config);
    assert_eq!(
        resumed_driver.open_night_id(),
        stopped_at,
        "the open night is rebuilt from the journal"
    );
    let resumed = resumed_driver.run().expect("resumed run");
    assert!(
        resumed.n_duplicates > 0,
        "redelivered alerts are deduplicated"
    );
    assert!(resumed.n_late > 0, "alerts of closed nights are skipped");

    assert_eq!(
        final_state(&ctx, &config),
        final_state(&reference_ctx, &reference_config)
    );
}

#[test]
fn day_obs_of_matches_known_dates() {
    assert_eq!(day_obs_of(51_544), 20_000_101);
    assert_eq!(day_obs_of(61_344), 20_261_031);
    assert_eq!(day_obs_of(60_369), 20_240_229);
}

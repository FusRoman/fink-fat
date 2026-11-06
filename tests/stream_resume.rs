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

use std::f64::consts::LN_10;

use camino::{Utf8Path, Utf8PathBuf};
use common::{CONFIG_PATH, archived_hash, combine_unordered, load_fixture, night_fingerprint};
use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    topocentric_kf::branching::{BranchCollection, read_archived_log},
};
use fink_fat_stream::{
    AlertRecord, Driver, DriverConfig, DriverStats, ReplaySource, StreamPaths,
    record::mjd_tai_to_utc,
};
use photom::{
    io::lsst_alert::{night::day_obs_to_night_id, units::TT_MINUS_TAI_DAYS},
    observation_dataset::{ObsDataset, observation::Observation},
    photometry::Filter,
};
use tempfile::TempDir;

/// Simulated delay between an exposure and its alerts' arrival.
const LATENCY_DAYS: f64 = 3.0 / 1_440.0;
/// Alerts per replay batch: one, so the driver can be stopped anywhere.
const BATCH_SIZE: usize = 1;
/// Nights closed before the interrupted run is stopped.
const STOP_AFTER_NIGHTS: usize = 8;

/// `YYYYMMDD` of the civil date whose MJD is `mjd` (Hinnant's
/// `civil_from_days`).
fn day_obs_of(mjd: i64) -> i64 {
    let z = mjd - 40_587 + 719_468; // days since 0000-03-01
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    year * 10_000 + month * 100 + day
}

/// One synthetic Rubin alert per fixture observation (see the module doc).
fn synthetic_alerts(obs_dataset: &ObsDataset) -> Vec<AlertRecord> {
    let mut observations: Vec<&Observation> = obs_dataset.iter_observations().collect();
    observations.sort();

    let mut records = Vec::with_capacity(observations.len());
    let mut visit_of_epoch: Option<(u64, i64)> = None; // (epoch bits, visit)
    let mut seq_num = 0;
    for observation in observations {
        let midpoint_mjd_tai = observation.mjd_tt() - TT_MINUS_TAI_DAYS;
        let night = (midpoint_mjd_tai - 0.5).floor() as i64;
        let day_obs = day_obs_of(night);
        assert_eq!(
            i64::from(day_obs_to_night_id(day_obs as u32).unwrap().0),
            night,
            "day_obs_of must invert photom's night id"
        );

        let epoch_bits = observation.mjd_tt().to_bits();
        let visit = match visit_of_epoch {
            Some((bits, visit)) if bits == epoch_bits => visit,
            _ => {
                let previous_day_obs = visit_of_epoch.map(|(_, v)| v / 100_000);
                seq_num = if previous_day_obs == Some(day_obs) {
                    seq_num + 1
                } else {
                    1
                };
                day_obs * 100_000 + seq_num
            }
        };
        visit_of_epoch = Some((epoch_bits, visit));

        let equ = observation.equ_coord();
        let photometry = observation.photometry();
        let flux = 10f64.powf((31.4 - photometry.magnitude) / 2.5);
        records.push(AlertRecord {
            dia_source_id: *observation.id() as i64,
            visit,
            midpoint_mjd_tai,
            ra_deg: equ.ra.to_degrees(),
            dec_deg: equ.dec.to_degrees(),
            // The fixture's RA error is on the α coordinate; Rubin's is on sky.
            ra_err_deg: (equ.ra_error * equ.dec.cos()).to_degrees() as f32,
            dec_err_deg: equ.dec_error.to_degrees() as f32,
            psf_flux_njy: flux as f32,
            psf_flux_err_njy: (photometry.error * flux * LN_10 / 2.5) as f32,
            band: match photometry.filter {
                Filter::Int(1) => "g",
                _ => "r",
            }
            .to_string(),
            arrival_mjd: mjd_tai_to_utc(midpoint_mjd_tai) + LATENCY_DAYS,
        });
    }
    records
}

/// The test engine configuration, storing under `storage`.
fn engine_config(storage: &Utf8Path) -> EngineConfig {
    let yaml = std::fs::read_to_string(CONFIG_PATH).expect("read test config");
    assert!(yaml.contains("storage_path: ./storage\n"));
    let yaml = yaml.replace(
        "storage_path: ./storage\n",
        &format!("storage_path: {storage}\n"),
    );
    let path = storage.join("engine_config.yaml");
    std::fs::create_dir_all(storage).expect("create storage");
    std::fs::write(&path, yaml).expect("write engine config");
    EngineConfig::load_engine_config_validated(&path).expect("load engine config")
}

fn storage_dir(tmp: &TempDir, name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from_path_buf(tmp.path().join(name)).expect("utf8 tempdir")
}

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

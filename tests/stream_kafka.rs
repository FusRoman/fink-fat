//! The Kafka source of `fink-fat-stream`, end to end against a local broker.
//!
//! The `tests/data` nights become synthetic Rubin alerts (as in
//! `stream_resume.rs`), published to fresh topics the way Fink publishes
//! them (see `common/kafka.rs`), and consumed by the [`Driver`] through
//! [`KafkaSource`]:
//!
//! * **equivalence**: a run fed by Kafka ends on the same state as a replay
//!   of the same alerts, and the source reports itself caught up at the end
//!   — from a single-partition topic, and from a 4-partition one, which
//!   librdkafka reads badly out of step in time: the partition merge must
//!   restore the replay's order (no leftover, no late alert);
//! * **resume**: a run stopped gracefully mid-night and restarted with the
//!   same consumer group resumes at its committed offsets — nothing lost,
//!   nothing delivered twice — and ends on the same state; with one and
//!   with four partitions, where alerts still buffered at the stop were
//!   never committed and must be read again;
//! * **idle partition**: a partition that never receives anything does
//!   not hold the others back;
//! * **poison message**: an undecodable message is skipped and committed,
//!   not redelivered forever;
//! * **real alerts**: the twelve real Fink LSST alerts of
//!   `fink-fat-stream`'s fixture, with Fink's own schema as key and spread
//!   over three partitions, come out decoded and in arrival order.
//!
//! Needs a broker: `docker compose --profile kafka up -d kafka`, then
//! `KAFKA_BOOTSTRAP=localhost:19092 cargo test -p fink-fat --test stream_kafka`.
//! Without `KAFKA_BOOTSTRAP` every test returns at once — unless
//! `FINK_FAT_REQUIRE_KAFKA=1` (set by the CI), which makes them fail instead.

mod common;

use std::{
    fs::File,
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

use apache_avro::{Reader, Schema, writer::datum::GenericDatumWriter};
use common::{
    archived_hash, combine_unordered,
    kafka::{
        FINK_SHAPED_SCHEMA, alert_messages, bootstrap, create_topic, kafka_config, produce,
        produce_alerts, resolve_bootstrap, unique_name, with_millisecond_arrivals,
    },
    load_fixture, night_fingerprint,
    stream::{engine_config, storage_dir, synthetic_alerts},
};
use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    topocentric_kf::branching::{BranchCollection, read_archived_log},
};
use fink_fat_stream::{
    AlertRecord, AlertSource, Driver, DriverConfig, DriverStats, ReplaySource, StreamPaths,
    record::UNIX_EPOCH_MJD,
    source::{
        KafkaConfig, KafkaSource,
        kafka::{AlertLayout, decode::decode_alert},
    },
};

/// Longest a test waits for the broker before failing.
const BROKER_TIMEOUT: Duration = Duration::from_secs(120);
/// Nights closed before the interrupted run is stopped.
const STOP_AFTER_NIGHTS: usize = 8;
/// Partitions of the multi-partition topics (Fink's has 10).
const PARTITIONS: i32 = 4;

/// Everything two runs must agree on, read back from storage (as in
/// `stream_resume.rs`).
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

fn kafka_driver<'a>(
    config: KafkaConfig,
    ctx: &'a KalmanContext,
    engine_config: &'a EngineConfig,
) -> Driver<'a, 'a, KafkaSource> {
    let source = KafkaSource::new(config).expect("create Kafka source");
    Driver::new(source, DriverConfig::default(), ctx, engine_config).expect("start driver")
}

/// Step `driver` until `done`, failing after [`BROKER_TIMEOUT`].
fn step_until(
    driver: &mut Driver<'_, '_, KafkaSource>,
    what: &str,
    mut done: impl FnMut(&Driver<'_, '_, KafkaSource>) -> bool,
) {
    let deadline = Instant::now() + BROKER_TIMEOUT;
    while !done(driver) {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        assert!(driver.step().expect("step"), "a live source never ends");
    }
}

/// The reference: a replay of `alerts`, start to finish.
fn replay_reference(
    alerts: &[fink_fat_stream::AlertRecord],
    ctx: &KalmanContext,
    config: &EngineConfig,
) -> DriverStats {
    let source = ReplaySource::from_records(alerts.to_vec(), 20);
    Driver::new(source, DriverConfig::default(), ctx, config)
        .expect("start driver")
        .run()
        .expect("reference run")
}

/// Run a Kafka-fed driver over the synthetic alerts, produced to a topic of
/// `partitions` partitions, until it caught up; check it ends on the
/// replay's state with the replay's counters.
fn assert_kafka_run_matches_the_replay(partitions: i32) {
    let Some(bootstrap) = bootstrap() else { return };
    let (tmp, obs_dataset, _) = load_fixture();
    let alerts = with_millisecond_arrivals(&synthetic_alerts(&obs_dataset));
    let topic = unique_name("equivalence");
    produce_alerts(&bootstrap, &topic, &alerts, partitions);

    let reference_config = engine_config(&storage_dir(&tmp, "reference"));
    let reference_ctx = reference_config.build_context();
    let reference = replay_reference(&alerts, &reference_ctx, &reference_config);

    let config = engine_config(&storage_dir(&tmp, "kafka"));
    let ctx = config.build_context();
    let mut driver = kafka_driver(
        kafka_config(&bootstrap, &topic, &unique_name("group")),
        &ctx,
        &config,
    );
    step_until(&mut driver, "every alert", |driver| {
        driver.source().stats().n_delivered == alerts.len() as u64
    });
    step_until(&mut driver, "the source to catch up", |driver| {
        driver.source().caught_up()
    });
    assert!(
        driver.source().wall_clock_mjd().is_some(),
        "a live source has a wall clock"
    );
    let source = driver.source().stats();
    assert_eq!(source.n_buffered, 0, "caught up: nothing left in a buffer");
    assert_eq!(source.n_skipped, 0);
    let stats = driver.finish().expect("finish");

    assert_eq!(stats.n_accepted, reference.n_accepted);
    assert_eq!(stats.n_visits, reference.n_visits);
    assert_eq!(
        stats.n_leftovers, reference.n_leftovers,
        "no alert reaches its visit later than in the replay"
    );
    assert_eq!(stats.n_late, 0, "no alert reaches a closed night");
    assert_eq!(stats.n_nights_closed, reference.n_nights_closed);
    assert_eq!(
        final_state(&ctx, &config),
        final_state(&reference_ctx, &reference_config)
    );
}

#[test]
fn a_single_partition_run_matches_a_replay_and_catches_up() {
    assert_kafka_run_matches_the_replay(1);
}

#[test]
fn a_multi_partition_run_is_merged_back_into_the_replay_order() {
    assert_kafka_run_matches_the_replay(PARTITIONS);
}

/// Stop a Kafka-fed run gracefully mid-night, restart it with the same
/// group, and check nothing was lost or delivered twice and the end state
/// is the replay's — from a topic of `partitions` partitions.
fn assert_a_stopped_run_resumes(partitions: i32) {
    let Some(bootstrap) = bootstrap() else { return };
    let (tmp, obs_dataset, _) = load_fixture();
    let alerts = with_millisecond_arrivals(&synthetic_alerts(&obs_dataset));
    let topic = unique_name("resume");
    produce_alerts(&bootstrap, &topic, &alerts, partitions);
    let group = unique_name("group");

    let reference_config = engine_config(&storage_dir(&tmp, "reference"));
    let reference_ctx = reference_config.build_context();
    replay_reference(&alerts, &reference_ctx, &reference_config);

    // First run: stop gracefully mid-night.
    let config = engine_config(&storage_dir(&tmp, "kafka"));
    let ctx = config.build_context();
    let mut first = kafka_driver(kafka_config(&bootstrap, &topic, &group), &ctx, &config);
    step_until(&mut first, "a mid-night stop point", |driver| {
        driver.stats().n_nights_closed >= STOP_AFTER_NIGHTS && driver.open_night_id().is_some()
    });
    let stopped_at = first.open_night_id();
    // Buffered alerts were never delivered, so never committed: the
    // second run must read them again.
    let delivered_first = first.source().stats().n_delivered;
    first.shutdown().expect("graceful stop");
    assert!(
        std::fs::metadata(StreamPaths::under(&config).journal)
            .expect("journal exists")
            .len()
            > 0,
        "the open night stays in the journal"
    );

    // Second run, same group: resumes after the committed offsets.
    let mut second = kafka_driver(kafka_config(&bootstrap, &topic, &group), &ctx, &config);
    assert_eq!(
        second.open_night_id(),
        stopped_at,
        "night rebuilt from the journal"
    );
    step_until(&mut second, "the remaining alerts", |driver| {
        delivered_first + driver.source().stats().n_delivered >= alerts.len() as u64
    });
    let delivered_second = second.source().stats().n_delivered;
    let resumed = second.finish().expect("finish");
    assert_eq!(
        delivered_first + delivered_second,
        alerts.len() as u64,
        "every alert delivered exactly once over the two runs"
    );
    assert_eq!(
        resumed.n_duplicates, 0,
        "committed alerts are not redelivered"
    );
    assert_eq!(
        resumed.n_late, 0,
        "no alert of a closed night is redelivered"
    );
    assert_eq!(
        final_state(&ctx, &config),
        final_state(&reference_ctx, &reference_config)
    );
}

#[test]
fn a_stopped_kafka_run_resumes_at_its_committed_offsets() {
    assert_a_stopped_run_resumes(1);
}

#[test]
fn a_stopped_multi_partition_run_resumes_without_losing_buffered_alerts() {
    assert_a_stopped_run_resumes(PARTITIONS);
}

/// Poll `source` until it has delivered `n` alerts and caught up, failing
/// after [`BROKER_TIMEOUT`]; acknowledges every batch.
fn drain(source: &mut KafkaSource, n: usize) -> Vec<AlertRecord> {
    let mut records = Vec::new();
    let deadline = Instant::now() + BROKER_TIMEOUT;
    while records.len() < n || !source.caught_up() {
        assert!(
            Instant::now() < deadline,
            "timed out: {} of {n} alerts, caught up: {}",
            records.len(),
            source.caught_up()
        );
        records.extend(source.next_batch().unwrap().expect("live source"));
        source.ack().unwrap();
    }
    records
}

#[test]
fn an_idle_partition_does_not_hold_the_others_back() {
    let Some(bootstrap) = bootstrap() else { return };
    let (_tmp, obs_dataset, _) = load_fixture();
    let alerts = with_millisecond_arrivals(&synthetic_alerts(&obs_dataset));
    let topic = unique_name("idle");
    // Three partitions, everything in partition 0: 1 and 2 stay empty.
    create_topic(&bootstrap, &topic, 3);
    produce(&bootstrap, &topic, &alert_messages(&alerts), 1);

    let mut source =
        KafkaSource::new(kafka_config(&bootstrap, &topic, &unique_name("group"))).unwrap();
    let records = drain(&mut source, alerts.len());
    assert_eq!(records.len(), alerts.len());
    assert_eq!(source.stats().n_buffered, 0);
}

#[test]
fn an_undecodable_message_is_skipped_and_committed() {
    let Some(bootstrap) = bootstrap() else { return };
    let (_tmp, obs_dataset, _) = load_fixture();
    let alerts = with_millisecond_arrivals(&synthetic_alerts(&obs_dataset));
    let mut messages = alert_messages(&alerts[..2]);
    // A garbage payload between the two alerts, stamped between them.
    let poison = (
        FINK_SHAPED_SCHEMA.as_bytes().to_vec(),
        vec![0xFF; 3],
        messages[0].2,
    );
    messages.insert(1, poison);
    let topic = unique_name("poison");
    produce(&bootstrap, &topic, &messages, 1);
    let group = unique_name("group");

    let mut source = KafkaSource::new(kafka_config(&bootstrap, &topic, &group)).unwrap();
    let records = drain(&mut source, 2);
    assert_eq!(records.len(), 2);
    assert_eq!(source.stats().n_skipped, 1);
    drop(source);

    // Same group: everything was committed, the poison message included.
    let mut again = KafkaSource::new(kafka_config(&bootstrap, &topic, &group)).unwrap();
    assert!(drain(&mut again, 0).is_empty());
    assert_eq!(again.stats().n_messages, 0, "nothing redelivered");
}

#[test]
fn a_raised_stop_flag_stops_without_closing_the_night() {
    let Some(bootstrap) = bootstrap() else { return };
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let config = engine_config(&storage_dir(&tmp, "kafka"));
    let ctx = config.build_context();
    let topic = unique_name("stop");
    let driver = kafka_driver(
        kafka_config(&bootstrap, &topic, &unique_name("group")),
        &ctx,
        &config,
    );
    let started = Instant::now();
    let stats = driver.run_until(&AtomicBool::new(true)).expect("stop");
    assert!(started.elapsed() < Duration::from_secs(5), "stops at once");
    assert_eq!(stats.n_nights_closed, 0);
}

#[test]
fn real_fink_alerts_are_consumed_in_arrival_order() {
    let Some(bootstrap) = bootstrap() else { return };
    let fixtures = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/crates/fink-fat-stream/tests/fixtures/fink_lsst"
    );
    let key = std::fs::read(format!("{fixtures}/schema.json")).expect("fixture schema");
    let schema = Schema::parse_str(std::str::from_utf8(&key).unwrap()).expect("Fink schema");
    let mut files: Vec<_> = std::fs::read_dir(fixtures)
        .expect("fixture directory")
        .map(|entry| entry.expect("fixture entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "avro"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 12, "the twelve fixture alerts");

    // Each alert stamped with Fink's ingestion time, as the topic does.
    let mut expected = Vec::new();
    let mut messages = Vec::new();
    for path in &files {
        for alert in Reader::new(File::open(path).expect("fixture alert")).expect("container") {
            let alert = alert.expect("alert");
            let record = decode_alert(&alert, &AlertLayout::default(), None, 0.0).unwrap();
            let timestamp = ((record.arrival_mjd - UNIX_EPOCH_MJD) * 86_400_000.0).round() as i64;
            let payload = GenericDatumWriter::builder(&schema)
                .build()
                .unwrap()
                .write_value_to_vec(alert)
                .expect("encode");
            expected.push((timestamp, record.dia_source_id, record.visit));
            messages.push((key.clone(), payload, timestamp));
        }
    }
    // Produced in arrival order over three partitions.
    let mut order: Vec<usize> = (0..messages.len()).collect();
    order.sort_by_key(|&i| (expected[i].0, expected[i].1));
    let messages: Vec<_> = order.iter().map(|&i| messages[i].clone()).collect();
    expected.sort();
    let topic = unique_name("real");
    create_topic(&bootstrap, &topic, 3);
    produce(&bootstrap, &topic, &messages, 3);

    let mut source =
        KafkaSource::new(kafka_config(&bootstrap, &topic, &unique_name("group"))).unwrap();
    let records = drain(&mut source, messages.len());
    let delivered: Vec<(i64, i64)> = records
        .iter()
        .map(|record| {
            let ms = ((record.arrival_mjd - UNIX_EPOCH_MJD) * 86_400_000.0).round() as i64;
            (ms, record.dia_source_id)
        })
        .collect();
    let in_order: Vec<(i64, i64)> = expected.iter().map(|(ms, id, _)| (*ms, *id)).collect();
    assert_eq!(delivered, in_order, "arrival order across partitions");
    assert!(records.iter().all(|record| record.band == "i"));
    let mut visits: Vec<i64> = records.iter().map(|record| record.visit).collect();
    visits.sort_unstable();
    visits.dedup();
    assert_eq!(
        visits,
        [
            2_026_111_100_106,
            2_026_111_100_109,
            2_026_111_100_110,
            2_026_111_100_111,
            2_026_111_100_112
        ]
    );
    assert_eq!(source.stats().n_skipped, 0);
}

/// Runs without a broker: the skip / run / fail decision of [`bootstrap`].
#[test]
fn a_required_broker_cannot_be_skipped() {
    let broker = "localhost:19092";
    // Given a broker, the tests run, required or not.
    assert_eq!(
        resolve_bootstrap(Some(broker), None),
        Ok(Some(broker.into()))
    );
    assert_eq!(
        resolve_bootstrap(Some(broker), Some("1")),
        Ok(Some(broker.into()))
    );
    // Without one, they are skipped by default…
    assert_eq!(resolve_bootstrap(None, None), Ok(None));
    assert_eq!(resolve_bootstrap(Some("  "), Some("0")), Ok(None));
    // …and fail when the broker is required, an empty value included.
    assert!(resolve_bootstrap(None, Some("1")).is_err());
    assert!(resolve_bootstrap(Some(""), Some("1")).is_err());
}

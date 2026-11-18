//! A local Kafka for the Kafka-source tests (`stream_kafka.rs`): alerts
//! encoded the way Fink publishes them — the writer schema as the message
//! key, a schemaless Avro datum as the value — and produced to a fresh topic.
//!
//! The broker is the `kafka` service of the repo's `docker-compose.yml`
//! (Redpanda): `docker compose --profile kafka up -d kafka`, then run the
//! tests with `KAFKA_BOOTSTRAP=localhost:19092`. Without that variable the
//! tests return at once, unless `FINK_FAT_REQUIRE_KAFKA=1` (set by the CI)
//! turns the missing broker into a failure (see [`bootstrap`]).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use apache_avro::{Schema, types::Value, writer::datum::GenericDatumWriter};
use fink_fat_stream::{
    AlertRecord,
    record::{UNIX_EPOCH_MJD, unix_ms_to_mjd_utc},
    source::KafkaConfig,
};
use rdkafka::{
    ClientConfig,
    admin::{AdminClient, AdminOptions, NewTopic, TopicReplication},
    client::DefaultClientContext,
    producer::{BaseProducer, BaseRecord, Producer},
};

/// Environment variable naming the test broker.
pub const KAFKA_BOOTSTRAP_ENV: &str = "KAFKA_BOOTSTRAP";

/// Environment variable that, set to `1`, makes a missing
/// [`KAFKA_BOOTSTRAP_ENV`] fail the Kafka tests instead of skipping them —
/// so a CI job meant to exercise the broker cannot pass without it.
pub const REQUIRE_KAFKA_ENV: &str = "FINK_FAT_REQUIRE_KAFKA";

/// What a Kafka test does, given the raw values of [`KAFKA_BOOTSTRAP_ENV`]
/// and [`REQUIRE_KAFKA_ENV`].
///
/// An empty bootstrap counts as unset (librdkafka would only fail later,
/// less clearly).
///
/// # Returns
/// `Ok(Some(broker))` to run against `broker`, `Ok(None)` to skip, or
/// `Err(message)` when the broker is required (`REQUIRE_KAFKA_ENV == "1"`)
/// but none is given.
pub fn resolve_bootstrap(
    bootstrap: Option<&str>,
    require: Option<&str>,
) -> Result<Option<String>, String> {
    let bootstrap = bootstrap.map(str::trim).filter(|b| !b.is_empty());
    match (bootstrap, require == Some("1")) {
        (Some(broker), _) => Ok(Some(broker.to_owned())),
        (None, true) => Err(format!(
            "{REQUIRE_KAFKA_ENV}=1 but {KAFKA_BOOTSTRAP_ENV} is unset or empty: \
             the Kafka tests would not run"
        )),
        (None, false) => Ok(None),
    }
}

/// The test broker, or `None` (with a note on stderr) when
/// [`KAFKA_BOOTSTRAP_ENV`] is unset: the calling test then returns, so a
/// plain `cargo test` needs no broker.
///
/// # Panics
/// When [`REQUIRE_KAFKA_ENV`] is `1` and no broker is given (see
/// [`resolve_bootstrap`]).
pub fn bootstrap() -> Option<String> {
    let bootstrap = std::env::var(KAFKA_BOOTSTRAP_ENV).ok();
    let require = std::env::var(REQUIRE_KAFKA_ENV).ok();
    let broker = resolve_bootstrap(bootstrap.as_deref(), require.as_deref())
        .unwrap_or_else(|message| panic!("{message}"));
    if broker.is_none() {
        eprintln!("{KAFKA_BOOTSTRAP_ENV} unset: Kafka test skipped");
    }
    broker
}

/// A name no other run uses (topics and consumer groups).
pub fn unique_name(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after 1970")
        .as_nanos();
    format!("{prefix}_{}_{nanos}", std::process::id())
}

/// Fink's alert schema reduced to what the source reads: the top-level
/// record with `brokerIngestMjd` and a `diaSource` record of nullable Rubin
/// fields. The source decodes with whatever schema the key carries, so the
/// 109-field original is not needed here (the real one is covered by
/// `fink-fat-stream`'s `fink_alert_fixture.rs`).
pub const FINK_SHAPED_SCHEMA: &str = r#"{
  "type": "record", "name": "value", "namespace": "topLevelRecord",
  "fields": [
    {"name": "diaSourceId", "type": ["long", "null"]},
    {"name": "brokerIngestMjd", "type": ["double", "null"]},
    {"name": "diaSource", "type": {
      "type": "record", "name": "diaSource", "namespace": "topLevelRecord",
      "fields": [
        {"name": "diaSourceId", "type": ["long", "null"]},
        {"name": "visit", "type": ["long", "null"]},
        {"name": "midpointMjdTai", "type": ["double", "null"]},
        {"name": "ra", "type": ["double", "null"]},
        {"name": "dec", "type": ["double", "null"]},
        {"name": "raErr", "type": ["float", "null"]},
        {"name": "decErr", "type": ["float", "null"]},
        {"name": "psfFlux", "type": ["float", "null"]},
        {"name": "psfFluxErr", "type": ["float", "null"]},
        {"name": "band", "type": ["string", "null"]}
      ]}}
  ]}"#;

/// The arrival of `record` as a Kafka timestamp, milliseconds.
pub fn arrival_ms(record: &AlertRecord) -> i64 {
    ((record.arrival_mjd - UNIX_EPOCH_MJD) * 86_400_000.0).round() as i64
}

/// `alerts` with their arrival rounded to the millisecond, as a Kafka
/// timestamp carries it — so a replay of them and their Kafka delivery see
/// the very same arrival times.
pub fn with_millisecond_arrivals(alerts: &[AlertRecord]) -> Vec<AlertRecord> {
    alerts
        .iter()
        .map(|record| AlertRecord {
            arrival_mjd: unix_ms_to_mjd_utc(arrival_ms(record)),
            ..record.clone()
        })
        .collect()
}

fn some<T: Into<Value>>(value: T) -> Value {
    Value::Union(0, Box::new(value.into()))
}

/// `record` as a [`FINK_SHAPED_SCHEMA`] datum.
pub fn encode(record: &AlertRecord, schema: &Schema) -> Vec<u8> {
    let source = Value::Record(vec![
        ("diaSourceId".into(), some(record.dia_source_id)),
        ("visit".into(), some(record.visit)),
        ("midpointMjdTai".into(), some(record.midpoint_mjd_tai)),
        ("ra".into(), some(record.ra_deg)),
        ("dec".into(), some(record.dec_deg)),
        ("raErr".into(), some(record.ra_err_deg)),
        ("decErr".into(), some(record.dec_err_deg)),
        ("psfFlux".into(), some(record.psf_flux_njy)),
        ("psfFluxErr".into(), some(record.psf_flux_err_njy)),
        ("band".into(), some(record.band.as_str())),
    ]);
    let alert = Value::Record(vec![
        ("diaSourceId".into(), some(record.dia_source_id)),
        ("brokerIngestMjd".into(), some(record.arrival_mjd)),
        ("diaSource".into(), source),
    ]);
    GenericDatumWriter::builder(schema)
        .build()
        .expect("datum writer")
        .write_value_to_vec(alert)
        .expect("encode alert")
}

/// Create `topic` with `partitions` partitions (Redpanda's auto-creation
/// only makes single-partition topics).
pub fn create_topic(bootstrap: &str, topic: &str, partitions: i32) {
    let admin: AdminClient<DefaultClientContext> = ClientConfig::new()
        .set("bootstrap.servers", bootstrap)
        .create()
        .expect("create admin client");
    let new_topic = NewTopic::new(topic, partitions, TopicReplication::Fixed(1));
    let results = futures_executor::block_on(admin.create_topics(
        [&new_topic],
        &AdminOptions::new().operation_timeout(Some(Duration::from_secs(30))),
    ))
    .expect("create topic request");
    for result in results {
        result.expect("topic created");
    }
}

/// Produce `messages` — `(key, payload, timestamp ms)` — to `topic`, in
/// order, message `i` to partition `i % partitions`, and wait until the
/// broker has them all. Each partition thus receives an in-order
/// subsequence, as Fink's partitions do. With `partitions == 1` the topic
/// may not exist yet: Redpanda's dev mode auto-creates it (one partition);
/// otherwise create it first ([`create_topic`]).
pub fn produce(
    bootstrap: &str,
    topic: &str,
    messages: &[(Vec<u8>, Vec<u8>, i64)],
    partitions: i32,
) {
    let producer: BaseProducer = ClientConfig::new()
        .set("bootstrap.servers", bootstrap)
        .set("enable.idempotence", "true")
        .create()
        .expect("create producer");
    for (index, (key, payload, timestamp)) in messages.iter().enumerate() {
        let partition = (index % partitions.max(1) as usize) as i32;
        let mut record = BaseRecord::to(topic)
            .key(key.as_slice())
            .payload(payload.as_slice())
            .timestamp(*timestamp)
            .partition(partition);
        loop {
            match producer.send(record) {
                Ok(()) => break,
                Err((_, unsent)) => {
                    // Local queue full: let deliveries drain, then retry.
                    producer.poll(Duration::from_millis(100));
                    record = unsent;
                }
            }
        }
    }
    producer
        .flush(Duration::from_secs(60))
        .expect("every message delivered");
}

/// `alerts` in replay order — arrival, then `diaSourceId` — as Fink-shaped
/// messages (see the module doc), each stamped with its arrival.
pub fn alert_messages(alerts: &[AlertRecord]) -> Vec<(Vec<u8>, Vec<u8>, i64)> {
    let schema = Schema::parse_str(FINK_SHAPED_SCHEMA).expect("test schema");
    let mut sorted: Vec<&AlertRecord> = alerts.iter().collect();
    sorted.sort_by(|a, b| {
        a.arrival_mjd
            .total_cmp(&b.arrival_mjd)
            .then(a.dia_source_id.cmp(&b.dia_source_id))
    });
    sorted
        .into_iter()
        .map(|record| {
            (
                FINK_SHAPED_SCHEMA.as_bytes().to_vec(),
                encode(record, &schema),
                arrival_ms(record),
            )
        })
        .collect()
}

/// Produce `alerts` to a new `topic` of `partitions` partitions, as Fink
/// would: in arrival order, spread over the partitions (see [`produce`]).
pub fn produce_alerts(bootstrap: &str, topic: &str, alerts: &[AlertRecord], partitions: i32) {
    if partitions > 1 {
        create_topic(bootstrap, topic, partitions);
    }
    produce(bootstrap, topic, &alert_messages(alerts), partitions);
}

/// A Kafka source configuration reading `topic` as consumer group `group`,
/// with small batches, a short poll and a backlog check at every quiet poll.
pub fn kafka_config(bootstrap: &str, topic: &str, group: &str) -> KafkaConfig {
    serde_yaml::from_str(&format!(
        "bootstrap_servers: {bootstrap}\n\
         topics: [{topic}]\n\
         group_id: {group}\n\
         batch_size: 20\n\
         batch_timeout: \"0.5 sec\"\n\
         caught_up_check_interval: \"0 sec\"\n"
    ))
    .expect("test Kafka configuration")
}

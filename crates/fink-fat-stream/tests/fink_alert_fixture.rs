//! **Real** Fink LSST alerts through the Kafka source's pure path —
//! decoding, the partition merge, visit assembly — without a broker.
//!
//! `tests/fixtures/fink_lsst/` holds twelve alerts of five consecutive
//! visits of night 2026-11-11 (2, 2, 3, 3 and 2 alerts), saved by
//! `finkctl stream -survey lsst --save --dump_schema` (Avro container
//! files, the writer schema in their header), and the schema of the topic
//! (`schema.json`, as dumped by fink-client). `expected.tsv` lists each
//! alert's fields as fastavro read them (Python `repr`, which round-trips
//! exactly).
//!
//! What is checked:
//!
//! * every alert decodes to its expected fields, from its container file
//!   and re-encoded the way it travels on Kafka (a schemaless datum, the
//!   schema JSON as the message key);
//! * the arrival time: the Kafka timestamp first, Fink's `brokerIngestMjd`
//!   without one, the exposure mid-point plus the latency without either;
//! * damaged alerts (null or missing fields, a truncated payload, no key)
//!   are reported, never mis-decoded;
//! * spread over partitions that are read out of step, the alerts come out
//!   of the [`PartitionMerger`] in the replay's order, and the assembler
//!   rebuilds the five visits whole.

#![cfg(feature = "kafka")]

use std::{collections::BTreeMap, fs::File, path::PathBuf};

use apache_avro::{Reader, Schema, types::Value, writer::datum::GenericDatumWriter};
use fink_fat_stream::{
    AlertRecord, VisitAssembler,
    assembler::Placement,
    record::mjd_tai_to_utc,
    source::kafka::{
        AlertLayout,
        decode::{DecodeError, SchemaCache, decode_alert, decode_payload},
        progress::TopicPartition,
        watermark::{Fetched, PartitionMerger},
    },
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fink_lsst")
        .join(name)
}

/// The single alert of a fixture container file.
fn read_alert(name: &str) -> Value {
    let reader = Reader::new(File::open(fixture(name)).unwrap()).unwrap();
    let mut alerts: Vec<Value> = reader.map(Result::unwrap).collect();
    assert_eq!(alerts.len(), 1, "{name}: one alert per file");
    alerts.remove(0)
}

/// One row of `expected.tsv`.
#[derive(Debug, Clone)]
struct Expected {
    file: String,
    record: AlertRecord,
}

/// `expected.tsv`, with `brokerIngestMjd` as the arrival (the fixtures
/// carry no Kafka timestamp).
fn expected() -> Vec<Expected> {
    let text = std::fs::read_to_string(fixture("expected.tsv")).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    lines
        .map(|line| {
            let row: BTreeMap<&str, &str> = header.iter().copied().zip(line.split('\t')).collect();
            let f64_of = |name: &str| row[name].parse::<f64>().unwrap();
            // The APDB errors and fluxes are Avro floats: exact in f64.
            let f32_of = |name: &str| f64_of(name) as f32;
            Expected {
                file: row["file"].to_string(),
                record: AlertRecord {
                    dia_source_id: row["diaSourceId"].parse().unwrap(),
                    visit: row["visit"].parse().unwrap(),
                    midpoint_mjd_tai: f64_of("midpointMjdTai"),
                    ra_deg: f64_of("ra"),
                    dec_deg: f64_of("dec"),
                    ra_err_deg: f32_of("raErr"),
                    dec_err_deg: f32_of("decErr"),
                    psf_flux_njy: f32_of("psfFlux"),
                    psf_flux_err_njy: f32_of("psfFluxErr"),
                    band: row["band"].to_string(),
                    arrival_mjd: f64_of("brokerIngestMjd"),
                },
            }
        })
        .collect()
}

/// `alert` as a Kafka message payload: a schemaless datum of `schema`.
fn encode(schema: &Schema, alert: Value) -> Vec<u8> {
    GenericDatumWriter::builder(schema)
        .build()
        .unwrap()
        .write_value_to_vec(alert)
        .unwrap()
}

/// The topic schema, as the message key carries it.
fn key() -> Vec<u8> {
    std::fs::read(fixture("schema.json")).unwrap()
}

/// The fields of a record value (unions unwrapped), mutably.
fn fields_mut(value: &mut Value) -> &mut Vec<(String, Value)> {
    match value {
        Value::Record(fields) => fields,
        Value::Union(_, inner) => fields_mut(inner),
        other => panic!("not a record: {other:?}"),
    }
}

/// Field `name` of record `value`, mutably.
fn field_mut<'a>(value: &'a mut Value, name: &str) -> &'a mut Value {
    fields_mut(value)
        .iter_mut()
        .find(|(field, _)| field == name)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("no field {name}"))
}

/// Avro null in Fink's nullable unions (`["<type>", "null"]`).
fn null() -> Value {
    Value::Union(1, Box::new(Value::Null))
}

#[test]
fn the_fixture_covers_five_visits() {
    let mut per_visit = BTreeMap::new();
    for expected in expected() {
        *per_visit.entry(expected.record.visit).or_insert(0) += 1;
    }
    assert_eq!(
        per_visit.into_iter().collect::<Vec<_>>(),
        [
            (2_026_111_100_106, 2),
            (2_026_111_100_109, 2),
            (2_026_111_100_110, 3),
            (2_026_111_100_111, 3),
            (2_026_111_100_112, 2),
        ]
    );
}

#[test]
fn real_fink_alerts_decode_from_their_container_files() {
    for expected in expected() {
        let alert = read_alert(&expected.file);
        let record = decode_alert(&alert, &AlertLayout::default(), None, 0.0).unwrap();
        assert_eq!(record, expected.record, "{}", expected.file);
    }
}

#[test]
fn real_fink_alerts_decode_from_kafka_messages() {
    let key = key();
    let mut schemas = SchemaCache::default();
    for expected in expected() {
        let schema = schemas.writer_schema(Some(&key)).unwrap();
        let payload = encode(schema, read_alert(&expected.file));
        let decoded = decode_payload(schema, &payload).unwrap();
        let record = decode_alert(&decoded, &AlertLayout::default(), None, 0.0).unwrap();
        assert_eq!(record, expected.record, "{}", expected.file);
    }
}

#[test]
fn the_kafka_timestamp_is_the_arrival_when_there_is_one() {
    let expected = &expected()[0];
    let alert = read_alert(&expected.file);
    let record = decode_alert(&alert, &AlertLayout::default(), Some(61_356.1), 0.0).unwrap();
    assert_eq!(record.arrival_mjd, 61_356.1);
}

#[test]
fn without_ingestion_time_the_arrival_is_the_mid_point_plus_latency() {
    let expected = &expected()[0];
    let mut alert = read_alert(&expected.file);
    *field_mut(&mut alert, "brokerIngestMjd") = null();
    let latency = 3.0 / 1_440.0;
    let record = decode_alert(&alert, &AlertLayout::default(), None, latency).unwrap();
    assert_eq!(
        record.arrival_mjd,
        mjd_tai_to_utc(expected.record.midpoint_mjd_tai) + latency
    );

    // The same with the ingestion-time field disabled in the layout.
    let layout = AlertLayout {
        ingest_mjd: None,
        ..AlertLayout::default()
    };
    let alert = read_alert(&expected.file);
    let record = decode_alert(&alert, &layout, None, latency).unwrap();
    assert_eq!(
        record.arrival_mjd,
        mjd_tai_to_utc(expected.record.midpoint_mjd_tai) + latency
    );
}

#[test]
fn a_null_required_field_is_reported_through_the_kafka_path() {
    let expected = &expected()[0];
    let key = key();
    let mut schemas = SchemaCache::default();
    let schema = schemas.writer_schema(Some(&key)).unwrap();
    for name in ["psfFlux", "visit", "band", "midpointMjdTai"] {
        let mut alert = read_alert(&expected.file);
        *field_mut(field_mut(&mut alert, "diaSource"), name) = null();
        let decoded = decode_payload(schema, &encode(schema, alert)).unwrap();
        let error = decode_alert(&decoded, &AlertLayout::default(), None, 0.0).unwrap_err();
        assert!(
            matches!(&error, DecodeError::NullField(path) if path == &format!("diaSource.{name}")),
            "{name}: {error}"
        );
    }
}

#[test]
fn a_missing_field_or_record_is_reported() {
    let expected = &expected()[0];
    let mut alert = read_alert(&expected.file);
    fields_mut(field_mut(&mut alert, "diaSource")).retain(|(name, _)| name != "ra");
    let error = decode_alert(&alert, &AlertLayout::default(), None, 0.0).unwrap_err();
    assert!(matches!(&error, DecodeError::MissingField(path) if path == "diaSource.ra"));

    // A layout naming a record the alert does not have.
    let layout = AlertLayout {
        record: "diaObject".into(),
        ..AlertLayout::default()
    };
    let alert = read_alert(&expected.file);
    assert!(matches!(
        decode_alert(&alert, &layout, None, 0.0).unwrap_err(),
        DecodeError::MissingField(_) | DecodeError::NullField(_) | DecodeError::NotARecord(_)
    ));
}

#[test]
fn a_truncated_payload_is_reported() {
    let expected = &expected()[0];
    let key = key();
    let mut schemas = SchemaCache::default();
    let schema = schemas.writer_schema(Some(&key)).unwrap();
    let payload = encode(schema, read_alert(&expected.file));
    let error = decode_payload(schema, &payload[..payload.len() / 2]).unwrap_err();
    assert!(matches!(error, DecodeError::Datum(_)), "{error}");
}

#[test]
fn a_message_without_key_needs_the_fallback_schema() {
    let expected = &expected()[0];
    let schema = Schema::parse_str(std::str::from_utf8(&key()).unwrap()).unwrap();
    let payload = encode(&schema, read_alert(&expected.file));

    let mut without = SchemaCache::default();
    assert!(matches!(
        without.writer_schema(None),
        Err(DecodeError::NoSchema)
    ));

    let mut with = SchemaCache::new(Some(schema));
    let fallback = with.writer_schema(None).unwrap();
    let decoded = decode_payload(fallback, &payload).unwrap();
    let record = decode_alert(&decoded, &AlertLayout::default(), None, 0.0).unwrap();
    assert_eq!(record, expected.record);
}

/// The twelve alerts spread over three partitions by `diaSourceId`, each
/// partition in arrival order — as Fink publishes — then fetched badly out
/// of step: all of partition 2 first, then partition 0, then partition 1.
/// The merger must release them in the replay's order (arrival, then
/// `diaSourceId`), and the assembler must rebuild every visit whole.
#[test]
fn partitions_read_out_of_step_merge_back_into_the_replay_order() {
    let records: Vec<AlertRecord> = expected().into_iter().map(|e| e.record).collect();
    let partition = |i: i32| -> TopicPartition { ("fink_is_solar_system_lsst".into(), i) };

    let mut queues: BTreeMap<i32, Vec<AlertRecord>> = BTreeMap::new();
    for record in &records {
        queues
            .entry((record.dia_source_id % 3) as i32)
            .or_default()
            .push(record.clone());
    }
    for queue in queues.values_mut() {
        queue.sort_by(|a, b| {
            a.arrival_mjd
                .total_cmp(&b.arrival_mjd)
                .then(a.dia_source_id.cmp(&b.dia_source_id))
        });
    }
    assert_eq!(queues.len(), 3, "every partition gets alerts");

    let mut merger = PartitionMerger::default();
    merger.assign(&[partition(0), partition(1), partition(2)]);
    let mut released = Vec::new();
    for index in [2, 0, 1] {
        for (offset, record) in queues[&index].iter().enumerate() {
            let fetched = Fetched {
                offset: offset as i64,
                arrival: record.arrival_mjd,
                record: Some(record.clone()),
            };
            merger.push(&partition(index), fetched);
            released.extend(merger.release(usize::MAX));
        }
        // The partition is now read to its end.
        let n = queues[&index].len() as i64;
        merger.set_offsets(&partition(index), 0, n);
        released.extend(merger.release(usize::MAX));
    }
    assert!(merger.is_caught_up());

    let merged: Vec<AlertRecord> = released.into_iter().filter_map(|r| r.record).collect();
    let mut replay_order = records.clone();
    replay_order.sort_by(|a, b| {
        a.arrival_mjd
            .total_cmp(&b.arrival_mjd)
            .then(a.dia_source_id.cmp(&b.dia_source_id))
    });
    assert_eq!(merged, replay_order);

    // In that order, with the default 90 s timeout, no alert is a leftover
    // and every visit comes out whole.
    let mut assembler = VisitAssembler::new(90.0 / 86_400.0);
    let mut visits = Vec::new();
    for record in &merged {
        assert_eq!(
            assembler.push(record.visit, record.arrival_mjd, record.dia_source_id),
            Placement::Buffered,
            "{} arrived after its visit was emitted",
            record.dia_source_id
        );
        visits.extend(assembler.poll(record.arrival_mjd));
    }
    visits.extend(assembler.drain());
    let sizes: Vec<(i64, usize)> = visits.iter().map(|v| (v.visit, v.items.len())).collect();
    assert_eq!(
        sizes,
        [
            (2_026_111_100_106, 2),
            (2_026_111_100_109, 2),
            (2_026_111_100_110, 3),
            (2_026_111_100_111, 3),
            (2_026_111_100_112, 2),
        ]
    );
}

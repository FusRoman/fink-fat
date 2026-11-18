//! Decoding of a Fink Kafka message into an [`AlertRecord`] — pure, no
//! broker involved.
//!
//! # Message format
//!
//! Fink publishes each alert as one Kafka message whose **key** holds the
//! Avro writer schema as JSON text, and whose **value** is the alert as a
//! *schemaless* Avro datum (no container header, no schema-registry
//! prefix) — what `fink-client` decodes with `json.loads(key)` and
//! `fastavro.schemaless_reader`. The schema is the same for every message
//! of a topic, so [`SchemaCache`] parses it once.
//!
//! # Extraction
//!
//! [`decode_alert`] reads the [`AlertLayout`]'s record (`diaSource`) and the
//! ten fields of its [`ColumnMap`](crate::source::ColumnMap). Every Rubin
//! field is a nullable union (`["T", "null"]`): a null in one of the ten is
//! a [`DecodeError::NullField`], and the alert is skipped by the source,
//! like a replay row with a null.

use apache_avro::{Schema, reader::datum::GenericDatumReader, types::Value};
use thiserror::Error;

use crate::{
    record::{AlertRecord, mjd_tai_to_utc},
    source::kafka::AlertLayout,
};

/// Why a message could not become an [`AlertRecord`]. Never fatal: the
/// source counts and skips the message.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// The message key is not text, and no fallback schema is configured.
    #[error("the message key is not UTF-8 text: {0}")]
    KeyNotUtf8(#[source] std::str::Utf8Error),

    /// The message key is not a parsable Avro schema, and no fallback
    /// schema is configured.
    #[error("the message key is not an Avro schema: {0}")]
    KeySchema(#[source] Box<apache_avro::Error>),

    /// The message has no key, and no fallback schema is configured.
    #[error("the message has no key (schema) and no `schema_path` is configured")]
    NoSchema,

    /// The message has no value.
    #[error("the message has no payload")]
    NoPayload,

    /// The value is not a datum of the writer schema.
    #[error("the payload is not a datum of the schema: {0}")]
    Datum(#[source] Box<apache_avro::Error>),

    /// A value expected to be a record is not one.
    #[error("`{0}` is not a record")]
    NotARecord(String),

    /// A field is absent from its record.
    #[error("field `{0}` is missing")]
    MissingField(String),

    /// A required field is null.
    #[error("field `{0}` is null")]
    NullField(String),

    /// A field has a type the extraction cannot read.
    #[error("field `{field}` has an unexpected type ({found})")]
    WrongType {
        /// `record.field` path.
        field: String,
        /// Debug rendering of the value found.
        found: String,
    },
}

/// Parse the writer schema a message key carries.
///
/// # Errors
/// [`DecodeError::KeyNotUtf8`] / [`DecodeError::KeySchema`] when the key is
/// not UTF-8 JSON describing a valid Avro schema.
pub fn parse_key_schema(key: &[u8]) -> Result<Schema, DecodeError> {
    let text = std::str::from_utf8(key).map_err(DecodeError::KeyNotUtf8)?;
    Schema::parse_str(text).map_err(|e| DecodeError::KeySchema(Box::new(e)))
}

/// The writer schemas of the messages, parsed once.
///
/// Holds the schema of the last key seen — one per topic in practice,
/// compared byte for byte so a change of schema is picked up at once — and
/// the configured fallback, used when a key is absent or unparsable (as
/// `fink-client` does with its `schema_path`).
#[derive(Debug, Default)]
pub struct SchemaCache {
    /// The last key and its parsed schema.
    last: Option<(Vec<u8>, Schema)>,
    /// `schema_path`'s schema, if configured.
    fallback: Option<Schema>,
}

impl SchemaCache {
    /// A cache falling back to `fallback` (see the type doc).
    pub fn new(fallback: Option<Schema>) -> Self {
        Self {
            last: None,
            fallback,
        }
    }

    /// The writer schema of a message whose key is `key`.
    ///
    /// # Errors
    /// [`DecodeError::KeyNotUtf8`], [`DecodeError::KeySchema`] or
    /// [`DecodeError::NoSchema`] when the key gives no schema and there is
    /// no fallback.
    pub fn writer_schema(&mut self, key: Option<&[u8]>) -> Result<&Schema, DecodeError> {
        let Some(key) = key.filter(|key| !key.is_empty()) else {
            return self.fallback.as_ref().ok_or(DecodeError::NoSchema);
        };
        let cached = matches!(&self.last, Some((last_key, _)) if last_key.as_slice() == key);
        if !cached {
            match parse_key_schema(key) {
                Ok(schema) => self.last = Some((key.to_vec(), schema)),
                Err(e) => return self.fallback.as_ref().ok_or(e),
            }
        }
        Ok(&self.last.as_ref().expect("cached or just parsed").1)
    }
}

/// Decode a schemaless Avro datum.
///
/// # Errors
/// [`DecodeError::Datum`] when `payload` is not a datum of `schema`.
pub fn decode_payload(schema: &Schema, payload: &[u8]) -> Result<Value, DecodeError> {
    GenericDatumReader::builder(schema)
        .build()
        .and_then(|reader| reader.read_value(&mut &payload[..]))
        .map_err(|e| DecodeError::Datum(Box::new(e)))
}

/// Build the [`AlertRecord`] of a decoded alert.
///
/// # Arrival time
/// The first available of:
/// 1. `message_mjd`, the Kafka message timestamp (MJD UTC) — when the
///    alert reached the topic, which a backlog catch-up keeps (a reception
///    time would squeeze hours of backlog into seconds and mislead the
///    night clock);
/// 2. the layout's ingestion-time field (Fink's `brokerIngestMjd`);
/// 3. the exposure mid-point converted to UTC plus `latency`, as the replay
///    simulates it.
///
/// # Arguments
/// * `alert` – The decoded alert (a top-level record).
/// * `layout` – Where the fields are.
/// * `message_mjd` – The message timestamp, MJD UTC, if the message has one.
/// * `latency` – Arrival delay of the last-resort arrival time, days.
///
/// # Errors
/// [`DecodeError::NotARecord`], [`DecodeError::MissingField`],
/// [`DecodeError::NullField`] or [`DecodeError::WrongType`] when the alert
/// does not have the layout's shape.
pub fn decode_alert(
    alert: &Value,
    layout: &AlertLayout,
    message_mjd: Option<f64>,
    latency: f64,
) -> Result<AlertRecord, DecodeError> {
    let top = record_fields(alert, "<alert>")?;
    let source_path = layout.record.as_str();
    let source = record_fields(field(top, source_path, source_path)?, source_path)?;
    let fields = &layout.fields;
    let get = |name: &str| -> Result<(&Value, String), DecodeError> {
        let path = format!("{source_path}.{name}");
        Ok((field(source, name, &path)?, path))
    };
    let long = |name: &str| get(name).and_then(|(value, path)| as_i64(value, &path));
    let double = |name: &str| get(name).and_then(|(value, path)| as_f64(value, &path));
    let float = |name: &str| double(name).map(|value| value as f32);

    let midpoint_mjd_tai = double(&fields.midpoint_mjd_tai)?;
    let ingest_mjd = match &layout.ingest_mjd {
        // Optional by nature: absent or null falls through to the mid-point.
        Some(name) => field(top, name, name)
            .ok()
            .and_then(|value| as_f64(value, name).ok()),
        None => None,
    };
    let arrival_mjd = message_mjd
        .or(ingest_mjd)
        .unwrap_or_else(|| mjd_tai_to_utc(midpoint_mjd_tai) + latency);

    let (band, band_path) = get(&fields.band)?;
    Ok(AlertRecord {
        dia_source_id: long(&fields.dia_source_id)?,
        visit: long(&fields.visit)?,
        midpoint_mjd_tai,
        ra_deg: double(&fields.ra)?,
        dec_deg: double(&fields.dec)?,
        ra_err_deg: float(&fields.ra_err)?,
        dec_err_deg: float(&fields.dec_err)?,
        psf_flux_njy: float(&fields.psf_flux)?,
        psf_flux_err_njy: float(&fields.psf_flux_err)?,
        band: as_string(band, &band_path)?,
        arrival_mjd,
    })
}

/// The value inside a union branch (Avro nullable fields are unions).
fn unwrap_union(value: &Value) -> &Value {
    match value {
        Value::Union(_, inner) => inner,
        other => other,
    }
}

/// The fields of a record value.
fn record_fields<'a>(value: &'a Value, path: &str) -> Result<&'a [(String, Value)], DecodeError> {
    match unwrap_union(value) {
        Value::Record(fields) => Ok(fields),
        Value::Null => Err(DecodeError::NullField(path.to_string())),
        _ => Err(DecodeError::NotARecord(path.to_string())),
    }
}

/// The non-null value of field `name`, unions unwrapped.
fn field<'a>(
    fields: &'a [(String, Value)],
    name: &str,
    path: &str,
) -> Result<&'a Value, DecodeError> {
    let value = fields
        .iter()
        .find(|(field, _)| field == name)
        .map(|(_, value)| unwrap_union(value))
        .ok_or_else(|| DecodeError::MissingField(path.to_string()))?;
    match value {
        Value::Null => Err(DecodeError::NullField(path.to_string())),
        value => Ok(value),
    }
}

fn wrong_type(value: &Value, path: &str) -> DecodeError {
    DecodeError::WrongType {
        field: path.to_string(),
        found: format!("{value:?}"),
    }
}

/// An Avro `long` (or `int`).
fn as_i64(value: &Value, path: &str) -> Result<i64, DecodeError> {
    match value {
        Value::Long(v) => Ok(*v),
        Value::Int(v) => Ok(i64::from(*v)),
        other => Err(wrong_type(other, path)),
    }
}

/// An Avro `double` or `float` (APDB errors and fluxes are `float`).
fn as_f64(value: &Value, path: &str) -> Result<f64, DecodeError> {
    match value {
        Value::Double(v) => Ok(*v),
        Value::Float(v) => Ok(f64::from(*v)),
        other => Err(wrong_type(other, path)),
    }
}

/// An Avro `string`.
fn as_string(value: &Value, path: &str) -> Result<String, DecodeError> {
    match value {
        Value::String(v) => Ok(v.clone()),
        other => Err(wrong_type(other, path)),
    }
}

#[cfg(test)]
mod tests {
    use apache_avro::writer::datum::GenericDatumWriter;

    use super::*;
    use crate::source::ColumnMap;

    /// A Fink-shaped schema reduced to what the extraction reads (plus one
    /// unread field): Rubin fields are nullable unions, as in Fink's.
    const SCHEMA: &str = r#"{
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
            {"name": "band", "type": ["string", "null"]},
            {"name": "snr", "type": ["float", "null"]}
          ]}}
      ]}"#;

    fn some<T: Into<Value>>(value: T) -> Value {
        Value::Union(0, Box::new(value.into()))
    }

    fn null() -> Value {
        Value::Union(1, Box::new(Value::Null))
    }

    /// An alert of [`SCHEMA`], with `visit` replaced by `visit`.
    fn alert(visit: Value, ingest: Value) -> Value {
        let source = Value::Record(vec![
            ("diaSourceId".into(), some(42_i64)),
            ("visit".into(), visit),
            ("midpointMjdTai".into(), some(61_356.05_f64)),
            ("ra".into(), some(10.5_f64)),
            ("dec".into(), some(-20.25_f64)),
            ("raErr".into(), some(1e-5_f32)),
            ("decErr".into(), some(2e-5_f32)),
            ("psfFlux".into(), some(1_234.5_f32)),
            ("psfFluxErr".into(), some(12.5_f32)),
            ("band".into(), some("i")),
            ("snr".into(), null()),
        ]);
        Value::Record(vec![
            ("diaSourceId".into(), some(42_i64)),
            ("brokerIngestMjd".into(), ingest),
            ("diaSource".into(), source),
        ])
    }

    fn encode(value: Value) -> Vec<u8> {
        let schema = Schema::parse_str(SCHEMA).unwrap();
        GenericDatumWriter::builder(&schema)
            .build()
            .unwrap()
            .write_value_to_vec(value)
            .unwrap()
    }

    fn decode(payload: &[u8], message_mjd: Option<f64>) -> Result<AlertRecord, DecodeError> {
        let mut cache = SchemaCache::default();
        let schema = cache.writer_schema(Some(SCHEMA.as_bytes()))?;
        let value = decode_payload(schema, payload)?;
        decode_alert(&value, &AlertLayout::default(), message_mjd, 0.0)
    }

    #[test]
    fn a_message_decodes_into_a_record() {
        let payload = encode(alert(some(2_026_111_100_106_i64), some(61_356.06_f64)));
        let record = decode(&payload, Some(61_356.07)).unwrap();
        assert_eq!(record.dia_source_id, 42);
        assert_eq!(record.visit, 2_026_111_100_106);
        assert_eq!(record.midpoint_mjd_tai, 61_356.05);
        assert_eq!((record.ra_deg, record.dec_deg), (10.5, -20.25));
        assert_eq!((record.ra_err_deg, record.dec_err_deg), (1e-5, 2e-5));
        assert_eq!(
            (record.psf_flux_njy, record.psf_flux_err_njy),
            (1_234.5, 12.5)
        );
        assert_eq!(record.band, "i");
        assert_eq!(record.arrival_mjd, 61_356.07, "message timestamp first");
    }

    #[test]
    fn arrival_falls_back_to_ingestion_then_to_the_mid_point() {
        let payload = encode(alert(some(2_026_111_100_106_i64), some(61_356.06_f64)));
        assert_eq!(decode(&payload, None).unwrap().arrival_mjd, 61_356.06);

        let payload = encode(alert(some(2_026_111_100_106_i64), null()));
        let arrival = decode(&payload, None).unwrap().arrival_mjd;
        assert!((arrival - mjd_tai_to_utc(61_356.05)).abs() < 1e-12);
    }

    #[test]
    fn a_null_required_field_is_reported() {
        let payload = encode(alert(null(), null()));
        let err = decode(&payload, None).unwrap_err();
        assert!(
            matches!(&err, DecodeError::NullField(path) if path == "diaSource.visit"),
            "{err}"
        );
    }

    #[test]
    fn a_missing_field_or_record_is_reported() {
        let payload = encode(alert(some(1_i64), null()));
        let mut cache = SchemaCache::default();
        let value = decode_payload(
            cache.writer_schema(Some(SCHEMA.as_bytes())).unwrap(),
            &payload,
        )
        .unwrap();
        let layout = AlertLayout {
            fields: ColumnMap {
                visit: "visitId".into(),
                ..ColumnMap::default()
            },
            ..AlertLayout::default()
        };
        assert!(matches!(
            decode_alert(&value, &layout, None, 0.0),
            Err(DecodeError::MissingField(path)) if path == "diaSource.visitId"
        ));
        let layout = AlertLayout {
            record: "diaSourceId".into(),
            ..AlertLayout::default()
        };
        assert!(matches!(
            decode_alert(&value, &layout, None, 0.0),
            Err(DecodeError::NotARecord(_))
        ));
    }

    #[test]
    fn the_schema_is_parsed_once_per_key() {
        let mut cache = SchemaCache::default();
        let first: *const Schema = cache.writer_schema(Some(SCHEMA.as_bytes())).unwrap();
        let second: *const Schema = cache.writer_schema(Some(SCHEMA.as_bytes())).unwrap();
        assert_eq!(first, second, "the cached schema is reused");
    }

    #[test]
    fn a_bad_key_uses_the_fallback_schema_or_fails() {
        let mut cache = SchemaCache::default();
        assert!(matches!(
            cache.writer_schema(Some(b"not json")),
            Err(DecodeError::KeySchema(_))
        ));
        assert!(matches!(
            cache.writer_schema(None),
            Err(DecodeError::NoSchema)
        ));

        let mut cache = SchemaCache::new(Some(Schema::parse_str(SCHEMA).unwrap()));
        assert!(cache.writer_schema(Some(b"not json")).is_ok());
        assert!(cache.writer_schema(None).is_ok());
        assert!(cache.writer_schema(Some(&[0xff, 0xfe])).is_ok());
    }

    #[test]
    fn a_payload_of_another_schema_is_reported() {
        let mut cache = SchemaCache::default();
        let schema = cache.writer_schema(Some(SCHEMA.as_bytes())).unwrap();
        assert!(matches!(
            decode_payload(schema, &[0x02]),
            Err(DecodeError::Datum(_))
        ));
    }
}

//! Load alerts from a Parquet input URI using DataFusion + object_store.
//!
//! Goal
//! ----
//! - The pipeline receives an `InputUri` (file/http/https/hdfs).
//! - This module resolves the URI to an `ObjectStore` backend.
//! - DataFusion reads the Parquet and returns Arrow `RecordBatch`es.
//! - We project only the required columns and build a `Vec<Alert>`.
//! - `Alert.key` is constructed at runtime via a caller-provided closure.
//!
//! Async boundary
//! --------------
//! This function is async because `object_store` is async and DataFusion
//! uses async execution for I/O. The Tokio runtime can stay at the CLI layer;
//! the engine only exposes `async fn` APIs.

use std::sync::Arc;

use ahash::AHashMap;
use arrow_array::{
    Array, RecordBatch, StringArray, StringViewArray,
    cast::AsArray,
    types::{Float64Type, UInt8Type, UInt32Type, UInt64Type},
};
use datafusion::{error::DataFusionError, object_store::ObjectStore, prelude::*};
use tokio::runtime::Runtime;
use url::Url;

use crate::{
    Alert, AlertKey, AlertStore,
    night_id::NightId,
    pipeline::stages::alert_inputs::{input_uri::InputUri, storage::resolve_input_uri},
};

/// Column names expected in the Parquet file.
#[derive(Clone, Debug)]
pub struct AlertParquetColumns {
    pub night_id: &'static str,
    pub dia_source_id: &'static str,
    pub ra: &'static str,
    pub ra_err: &'static str,
    pub dec: &'static str,
    pub dec_err: &'static str,
    pub mjd_tt: &'static str,
    pub flux: &'static str,
    pub flux_err: &'static str,
    pub band: &'static str,
    pub observer_mpc_code: &'static str,
}

impl Default for AlertParquetColumns {
    fn default() -> Self {
        Self {
            night_id: "night_id",
            dia_source_id: "dia_source_id",
            ra: "ra",
            ra_err: "ra_err",
            dec: "dec",
            dec_err: "dec_err",
            mjd_tt: "mjd_tt",
            flux: "flux",
            flux_err: "flux_err",
            band: "band",
            observer_mpc_code: "observer_mpc_code",
        }
    }
}

/// Error type for Parquet alert loading.
#[derive(Debug)]
pub enum LoadAlertsError {
    Resolve(String),
    DataFusion(DataFusionError),
    Arrow(String),
}

impl From<DataFusionError> for LoadAlertsError {
    fn from(e: DataFusionError) -> Self {
        Self::DataFusion(e)
    }
}

pub fn load_alerts_sync(
    input: &InputUri,
    columns: AlertParquetColumns,
) -> Result<AlertStore, LoadAlertsError> {
    let rt = Runtime::new().expect("failed to build tokio runtime");
    rt.block_on(load_alerts_from_parquet_uri(input, columns))
}

/// Load alerts from a Parquet input URI.
///
/// `make_key` is called for each row (in file order) to build `AlertKey`.
///
/// Notes
/// -----
/// - This function uses DataFusion to read Parquet and optionally push down
///   projection (column pruning).
/// - If you later want filtering (quality flags, night window, etc.),
///   add DataFusion `df.filter(...)` before `collect()`.
pub async fn load_alerts_from_parquet_uri(
    input: &InputUri,
    columns: AlertParquetColumns,
) -> Result<AlertStore, LoadAlertsError> {
    let url = input
        .parse()
        .map_err(|e| LoadAlertsError::Resolve(format!("invalid uri: {e}")))?;

    // 1) Resolve URI -> object_store backend + object_store path
    let resolved =
        resolve_input_uri(input).map_err(|e| LoadAlertsError::Resolve(format!("{e:?}")))?;

    // 2) Build a DataFusion context with the store registered for this URL
    let ctx = build_session_context_with_store(&url, resolved.store.clone())?;

    // 3) Read the parquet via DataFusion (using the original URI string)
    // DataFusion will route the I/O through the registered object store.
    let df = ctx
        .read_parquet(input.0.as_str(), ParquetReadOptions::default())
        .await?;

    // 4) Projection: select only columns needed to build Alert
    let df = df.select(vec![
        col(columns.night_id),
        col(columns.dia_source_id),
        col(columns.ra),
        col(columns.ra_err),
        col(columns.dec),
        col(columns.dec_err),
        col(columns.mjd_tt),
        col(columns.flux),
        col(columns.flux_err),
        col(columns.band),
        col(columns.observer_mpc_code),
    ])?;

    // 5) Execute and collect batches
    let batches = df.collect().await?;

    // 6) Convert batches -> Vec<Alert>
    let alerts = build_alerts_from_batches(&batches, &columns)?;

    Ok(alerts)
}

/// Register an object_store backend into a DataFusion `SessionContext` for a given URL.
///
/// DataFusion APIs around store registration changed across versions.
/// The method used here is the one commonly available in modern DataFusion:
/// `ctx.runtime_env().register_object_store(url, store)`.
fn build_session_context_with_store(
    url: &Url,
    store: Arc<dyn ObjectStore>,
) -> Result<SessionContext, LoadAlertsError> {
    let ctx = SessionContext::new();

    // IMPORTANT:
    // DataFusion associates stores to an "ObjectStoreUrl" (scheme + authority).
    // Using the full URL generally works; DataFusion normalizes it internally.
    //
    // If your DataFusion version doesn't expose `register_object_store`,
    // the equivalent is registering in the runtime env's object store registry.
    ctx.runtime_env().register_object_store(url, store);

    Ok(ctx)
}

/// Convert Arrow `RecordBatch`es to a `Vec<Alert>`.
///
/// This is intentionally strict:
/// - missing columns => error
/// - unexpected dtypes => error (via downcast)
fn build_alerts_from_batches(
    batches: &[RecordBatch],
    c: &AlertParquetColumns,
) -> Result<AlertStore, LoadAlertsError> {
    let mut out = AlertStore::new();
    let mut global_row = 0usize;

    // Intern pool: String -> Arc<String>
    let mut observer_pool: AHashMap<String, Arc<String>> = AHashMap::new();

    for batch in batches {
        let night_id = col_u32(batch, c.night_id)?;
        let dia_source_id = col_u64(batch, c.dia_source_id)?;
        let ra = col_f64(batch, c.ra)?;
        let ra_err = col_f64(batch, c.ra_err)?;
        let dec = col_f64(batch, c.dec)?;
        let dec_err = col_f64(batch, c.dec_err)?;
        let mjd_tt = col_f64(batch, c.mjd_tt)?;
        let flux = col_f64(batch, c.flux)?;
        let flux_err = col_f64(batch, c.flux_err)?;
        let band = col_u8(batch, c.band)?;
        let observer_mpc_code = col_string(batch, c.observer_mpc_code)?;

        let n = batch.num_rows();

        for i in 0..n {
            if night_id.is_null(i)
                || dia_source_id.is_null(i)
                || ra.is_null(i)
                || ra_err.is_null(i)
                || dec.is_null(i)
                || dec_err.is_null(i)
                || mjd_tt.is_null(i)
                || flux.is_null(i)
                || flux_err.is_null(i)
                || band.is_null(i)
                || observer_mpc_code.is_null(i)
            {
                return Err(LoadAlertsError::Arrow(format!(
                    "null value in required columns at global row {global_row}"
                )));
            }

            let night_id = NightId(night_id.value(i));
            let vec_night = out.get_or_init_with_capacity(night_id, n);

            let dia = dia_source_id.value(i);
            let mjd = mjd_tt.value(i);

            // -------- Interning --------
            let code_str = observer_mpc_code.value(i);

            let observer_arc = if let Some(existing) = observer_pool.get(code_str) {
                Arc::clone(existing)
            } else {
                let arc = Arc::new(code_str.to_string());
                observer_pool.insert(code_str.to_string(), Arc::clone(&arc));
                arc
            };

            let alert = Alert {
                key: AlertKey {
                    night_id,
                    dia_source_id: dia,
                },
                ra: ra.value(i),
                ra_err: ra_err.value(i),
                dec: dec.value(i),
                dec_err: dec_err.value(i),
                mjd_tt: mjd,
                flux: flux.value(i),
                flux_err: flux_err.value(i),
                band: band.value(i),
                observer_mpc_code: observer_arc,
            };

            vec_night.push(alert);
            global_row += 1;
        }
    }

    Ok(out)
}

// -----------------------------------------------------------------------------
// Column helpers (strict downcasts with clear errors)
// -----------------------------------------------------------------------------

fn col_index(batch: &RecordBatch, name: &str) -> Result<usize, LoadAlertsError> {
    batch
        .schema()
        .index_of(name)
        .map_err(|_| LoadAlertsError::Arrow(format!("missing column '{name}'")))
}

fn col_u64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::PrimitiveArray<UInt64Type>, LoadAlertsError> {
    let idx = col_index(batch, name)?;
    batch
        .column(idx)
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| LoadAlertsError::Arrow(format!("column '{name}' is not UInt64")))
}

fn col_u32<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::PrimitiveArray<UInt32Type>, LoadAlertsError> {
    let idx = col_index(batch, name)?;
    batch
        .column(idx)
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| LoadAlertsError::Arrow(format!("column '{name}' is not UInt32")))
}

fn col_f64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::PrimitiveArray<Float64Type>, LoadAlertsError> {
    let idx = col_index(batch, name)?;
    batch
        .column(idx)
        .as_primitive_opt::<Float64Type>()
        .ok_or_else(|| LoadAlertsError::Arrow(format!("column '{name}' is not Float64")))
}

fn col_u8<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::PrimitiveArray<UInt8Type>, LoadAlertsError> {
    let idx = col_index(batch, name)?;
    batch
        .column(idx)
        .as_primitive_opt::<UInt8Type>()
        .ok_or_else(|| LoadAlertsError::Arrow(format!("column '{name}' is not UInt8")))
}

/// Wrapper for string columns that may be stored as `Utf8` or `Utf8View`.
///
/// DataFusion may return either representation depending on its configuration
/// and the Parquet file layout. This enum abstracts over both so the rest of
/// the loader can use a uniform `.is_null()` / `.value()` interface.
enum StringCol<'a> {
    Utf8(&'a StringArray),
    View(&'a StringViewArray),
}

impl StringCol<'_> {
    #[inline]
    fn is_null(&self, i: usize) -> bool {
        match self {
            StringCol::Utf8(a) => a.is_null(i),
            StringCol::View(a) => a.is_null(i),
        }
    }
    #[inline]
    fn value(&self, i: usize) -> &str {
        match self {
            StringCol::Utf8(a) => a.value(i),
            StringCol::View(a) => a.value(i),
        }
    }
}

fn col_string<'a>(batch: &'a RecordBatch, name: &str) -> Result<StringCol<'a>, LoadAlertsError> {
    let idx = col_index(batch, name)?;
    let col = batch.column(idx);

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(StringCol::Utf8(arr));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        return Ok(StringCol::View(arr));
    }

    Err(LoadAlertsError::Arrow(format!(
        "column '{name}' is not a string type (expected Utf8 or Utf8View)"
    )))
}

#[cfg(test)]
mod alert_loader_tests {
    use super::*;

    use arrow_array::{
        ArrayRef, Float64Array, RecordBatch, StringArray, UInt8Array, UInt32Array, UInt64Array,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn default_cols() -> AlertParquetColumns {
        AlertParquetColumns::default()
    }

    fn make_schema(c: &AlertParquetColumns) -> Arc<Schema> {
        // IMPORTANT: doit contenir toutes les colonnes requises par build_alerts_from_batches()
        // et avec les dtypes attendus (notamment flux/flux_err en Float64).
        Arc::new(Schema::new(vec![
            Field::new(c.night_id, DataType::UInt32, true),
            Field::new(c.dia_source_id, DataType::UInt64, true),
            Field::new(c.ra, DataType::Float64, true),
            Field::new(c.ra_err, DataType::Float64, true),
            Field::new(c.dec, DataType::Float64, true),
            Field::new(c.dec_err, DataType::Float64, true),
            Field::new(c.mjd_tt, DataType::Float64, true),
            Field::new(c.flux, DataType::Float64, true),
            Field::new(c.flux_err, DataType::Float64, true),
            Field::new(c.band, DataType::UInt8, true),
            Field::new(c.observer_mpc_code, DataType::Utf8, true),
        ]))
    }

    fn batch_two_rows_all_valid_same_night(c: &AlertParquetColumns, night: u32) -> RecordBatch {
        let schema = make_schema(c);

        let night_id: ArrayRef = Arc::new(UInt32Array::from(vec![night, night]));
        let dia: ArrayRef = Arc::new(UInt64Array::from(vec![10_u64, 11_u64]));
        let ra: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0_f64]));
        let ra_err: ArrayRef = Arc::new(Float64Array::from(vec![0.1_f64, 0.2_f64]));
        let dec: ArrayRef = Arc::new(Float64Array::from(vec![3.0_f64, 4.0_f64]));
        let dec_err: ArrayRef = Arc::new(Float64Array::from(vec![0.3_f64, 0.4_f64]));
        let mjd: ArrayRef = Arc::new(Float64Array::from(vec![60000.0_f64, 60001.0_f64]));
        let flux: ArrayRef = Arc::new(Float64Array::from(vec![12.0_f64, 13.0_f64]));
        let flux_err: ArrayRef = Arc::new(Float64Array::from(vec![1.2_f64, 1.3_f64]));
        let band: ArrayRef = Arc::new(UInt8Array::from(vec![1_u8, 2_u8]));
        let obs_code: ArrayRef = Arc::new(StringArray::from(vec!["I41", "I41"]));

        RecordBatch::try_new(
            schema,
            vec![
                night_id, dia, ra, ra_err, dec, dec_err, mjd, flux, flux_err, band, obs_code,
            ],
        )
        .unwrap()
    }

    fn batch_one_row_null_dia(c: &AlertParquetColumns, night: u32) -> RecordBatch {
        let schema = make_schema(c);

        let night_id: ArrayRef = Arc::new(UInt32Array::from(vec![night]));
        let dia: ArrayRef = Arc::new(UInt64Array::from(vec![None])); // NULL
        let ra: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64]));
        let ra_err: ArrayRef = Arc::new(Float64Array::from(vec![0.1_f64]));
        let dec: ArrayRef = Arc::new(Float64Array::from(vec![3.0_f64]));
        let dec_err: ArrayRef = Arc::new(Float64Array::from(vec![0.3_f64]));
        let mjd: ArrayRef = Arc::new(Float64Array::from(vec![60000.0_f64]));
        let flux: ArrayRef = Arc::new(Float64Array::from(vec![12.0_f64]));
        let flux_err: ArrayRef = Arc::new(Float64Array::from(vec![1.2_f64]));
        let band: ArrayRef = Arc::new(UInt8Array::from(vec![1_u8]));
        let obs_code: ArrayRef = Arc::new(StringArray::from(vec!["I41"]));

        RecordBatch::try_new(
            schema,
            vec![
                night_id, dia, ra, ra_err, dec, dec_err, mjd, flux, flux_err, band, obs_code,
            ],
        )
        .unwrap()
    }

    #[test]
    fn build_alerts_happy_path_builds_expected_alerts_and_keys() {
        let c = default_cols();
        let b = batch_two_rows_all_valid_same_night(&c, 42);

        let store = build_alerts_from_batches(&[b], &c).unwrap();

        assert_eq!(store.n_nights(), 1);

        let night_vec = store.get(&NightId(42)).expect("night 42 present");
        assert_eq!(night_vec.len(), 2);

        // Ordre et contenu
        assert_eq!(night_vec[0].key.dia_source_id, 10);
        assert_eq!(night_vec[1].key.dia_source_id, 11);

        assert_eq!(night_vec[0].ra, 1.0);
        assert_eq!(night_vec[1].ra, 2.0);

        assert_eq!(night_vec[0].mjd_tt, 60000.0);
        assert_eq!(night_vec[1].mjd_tt, 60001.0);

        assert_eq!(night_vec[0].band, 1);
        assert_eq!(night_vec[1].band, 2);

        assert_eq!(night_vec[0].key.night_id, NightId(42));
        assert_eq!(night_vec[1].key.night_id, NightId(42));
    }

    #[test]
    fn build_alerts_missing_column_is_error() {
        let c = default_cols();

        // Schema avec toutes les colonnes requises sauf `band`
        let schema = Arc::new(Schema::new(vec![
            Field::new(c.night_id, DataType::UInt32, true),
            Field::new(c.dia_source_id, DataType::UInt64, true),
            Field::new(c.ra, DataType::Float64, true),
            Field::new(c.ra_err, DataType::Float64, true),
            Field::new(c.dec, DataType::Float64, true),
            Field::new(c.dec_err, DataType::Float64, true),
            Field::new(c.mjd_tt, DataType::Float64, true),
            Field::new(c.flux, DataType::Float64, true),
            Field::new(c.flux_err, DataType::Float64, true),
            // Field::new(c.band, DataType::UInt8, true), // manquante
        ]));

        let night_id: ArrayRef = Arc::new(UInt32Array::from(vec![1_u32]));
        let dia: ArrayRef = Arc::new(UInt64Array::from(vec![10_u64]));
        let ra: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64]));
        let ra_err: ArrayRef = Arc::new(Float64Array::from(vec![0.1_f64]));
        let dec: ArrayRef = Arc::new(Float64Array::from(vec![3.0_f64]));
        let dec_err: ArrayRef = Arc::new(Float64Array::from(vec![0.3_f64]));
        let mjd: ArrayRef = Arc::new(Float64Array::from(vec![60000.0_f64]));
        let flux: ArrayRef = Arc::new(Float64Array::from(vec![12.0_f64]));
        let flux_err: ArrayRef = Arc::new(Float64Array::from(vec![1.2_f64]));
        // band manquante

        let batch = RecordBatch::try_new(
            schema,
            vec![night_id, dia, ra, ra_err, dec, dec_err, mjd, flux, flux_err],
        )
        .unwrap();

        let err = build_alerts_from_batches(&[batch], &c).unwrap_err();

        match err {
            LoadAlertsError::Arrow(msg) => {
                assert!(msg.contains("missing column"), "msg={msg}");
                assert!(msg.contains("band"), "msg={msg}");
            }
            other => panic!("expected Arrow error, got: {other:?}"),
        }
    }

    #[test]
    fn build_alerts_wrong_dtype_is_error() {
        let c = default_cols();

        // ra_err attendu Float64, on met UInt64
        let schema = Arc::new(Schema::new(vec![
            Field::new(c.night_id, DataType::UInt32, true),
            Field::new(c.dia_source_id, DataType::UInt64, true),
            Field::new(c.ra, DataType::Float64, true),
            Field::new(c.ra_err, DataType::UInt64, true), // WRONG
            Field::new(c.dec, DataType::Float64, true),
            Field::new(c.dec_err, DataType::Float64, true),
            Field::new(c.mjd_tt, DataType::Float64, true),
            Field::new(c.flux, DataType::Float64, true),
            Field::new(c.flux_err, DataType::Float64, true),
            Field::new(c.band, DataType::UInt8, true),
        ]));

        let night_id: ArrayRef = Arc::new(UInt32Array::from(vec![1_u32]));
        let dia: ArrayRef = Arc::new(UInt64Array::from(vec![10_u64]));
        let ra: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64]));
        let ra_err_wrong: ArrayRef = Arc::new(UInt64Array::from(vec![123_u64])); // WRONG
        let dec: ArrayRef = Arc::new(Float64Array::from(vec![3.0_f64]));
        let dec_err: ArrayRef = Arc::new(Float64Array::from(vec![0.3_f64]));
        let mjd: ArrayRef = Arc::new(Float64Array::from(vec![60000.0_f64]));
        let flux: ArrayRef = Arc::new(Float64Array::from(vec![12.0_f64]));
        let flux_err: ArrayRef = Arc::new(Float64Array::from(vec![1.2_f64]));
        let band: ArrayRef = Arc::new(UInt8Array::from(vec![1_u8]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                night_id,
                dia,
                ra,
                ra_err_wrong,
                dec,
                dec_err,
                mjd,
                flux,
                flux_err,
                band,
            ],
        )
        .unwrap();

        let err = build_alerts_from_batches(&[batch], &c).unwrap_err();

        match err {
            LoadAlertsError::Arrow(msg) => {
                assert!(msg.contains("ra_err"), "msg={msg}");
                assert!(msg.contains("Float64"), "msg={msg}");
            }
            other => panic!("expected Arrow error, got: {other:?}"),
        }
    }

    #[test]
    fn build_alerts_null_in_required_column_is_error_and_reports_global_row() {
        let c = default_cols();
        let schema = make_schema(&c);

        // NULL dans dia_source_id à la ligne 1
        let night_id: ArrayRef = Arc::new(UInt32Array::from(vec![1_u32, 1_u32]));
        let dia: ArrayRef = Arc::new(UInt64Array::from(vec![Some(10_u64), None]));
        let ra: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0_f64]));
        let ra_err: ArrayRef = Arc::new(Float64Array::from(vec![0.1_f64, 0.2_f64]));
        let dec: ArrayRef = Arc::new(Float64Array::from(vec![3.0_f64, 4.0_f64]));
        let dec_err: ArrayRef = Arc::new(Float64Array::from(vec![0.3_f64, 0.4_f64]));
        let mjd: ArrayRef = Arc::new(Float64Array::from(vec![60000.0_f64, 60001.0_f64]));
        let flux: ArrayRef = Arc::new(Float64Array::from(vec![12.0_f64, 13.0_f64]));
        let flux_err: ArrayRef = Arc::new(Float64Array::from(vec![1.2_f64, 1.3_f64]));
        let band: ArrayRef = Arc::new(UInt8Array::from(vec![1_u8, 2_u8]));
        let obs_code: ArrayRef = Arc::new(StringArray::from(vec!["I41", "I41"]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                night_id, dia, ra, ra_err, dec, dec_err, mjd, flux, flux_err, band, obs_code,
            ],
        )
        .unwrap();

        let err = build_alerts_from_batches(&[batch], &c).unwrap_err();

        match err {
            LoadAlertsError::Arrow(msg) => {
                assert!(msg.contains("null value"), "msg={msg}");
                assert!(msg.contains("global row 1"), "msg={msg}");
            }
            other => panic!("expected Arrow error, got: {other:?}"),
        }
    }

    #[test]
    fn build_alerts_global_row_increments_across_batches() {
        let c = default_cols();

        let b1 = batch_two_rows_all_valid_same_night(&c, 1);
        let b2 = batch_one_row_null_dia(&c, 1); // 3e ligne globale => index 2

        let err = build_alerts_from_batches(&[b1, b2], &c).unwrap_err();

        match err {
            LoadAlertsError::Arrow(msg) => {
                assert!(msg.contains("null value"), "msg={msg}");
                assert!(
                    msg.contains("global row 2"),
                    "expected global row 2 (third row overall), msg={msg}"
                );
            }
            other => panic!("expected Arrow error, got: {other:?}"),
        }
    }
}

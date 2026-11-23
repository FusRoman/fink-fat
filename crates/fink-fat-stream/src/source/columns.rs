//! [`ColumnMap`]: where each [`AlertRecord`](crate::AlertRecord) field lives
//! in an alert — shared by the replay (column names of a parquet dump) and
//! the Kafka source (field names inside the Avro alert's `diaSource`
//! record).

use serde::Deserialize;

/// Names of the fields holding each [`AlertRecord`](crate::AlertRecord) field:
/// the column names of a replay dump (`source.columns`), or the field names
/// of the Kafka alert's `diaSource` record (`source.layout.fields`). Unset
/// fields keep the Rubin `DiaSource` name, which is also what Fink
/// publishes.
///
/// Names are flat: a replay dump nesting the `DiaSource` in a struct column
/// must be flattened first; the Kafka source looks them up in one record
/// (`source.layout.record`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ColumnMap {
    /// `diaSourceId` (integer).
    pub dia_source_id: String,
    /// `visit` (integer).
    pub visit: String,
    /// `midpointMjdTai` (MJD TAI).
    pub midpoint_mjd_tai: String,
    /// `ra` (degrees).
    pub ra: String,
    /// `dec` (degrees).
    pub dec: String,
    /// `raErr` (on-sky, degrees).
    pub ra_err: String,
    /// `decErr` (degrees).
    pub dec_err: String,
    /// `psfFlux` (nJy).
    pub psf_flux: String,
    /// `psfFluxErr` (nJy).
    pub psf_flux_err: String,
    /// `band` (string).
    pub band: String,
}

impl Default for ColumnMap {
    /// The Rubin `DiaSource` column names.
    fn default() -> Self {
        Self {
            dia_source_id: "diaSourceId".into(),
            visit: "visit".into(),
            midpoint_mjd_tai: "midpointMjdTai".into(),
            ra: "ra".into(),
            dec: "dec".into(),
            ra_err: "raErr".into(),
            dec_err: "decErr".into(),
            psf_flux: "psfFlux".into(),
            psf_flux_err: "psfFluxErr".into(),
            band: "band".into(),
        }
    }
}

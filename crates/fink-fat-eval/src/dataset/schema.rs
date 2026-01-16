//! Parquet schema definitions and column names.

use polars::prelude::*;

/// Column names for the ZTF-like alert dataset.
pub mod cols {
    pub const CANDID: &str = "candid";
    pub const RA: &str = "ra";
    pub const DEC: &str = "dec";
    pub const JD: &str = "jd";
    pub const MAGPSF: &str = "magpsf";
    pub const SIGMAPSF: &str = "sigmapsf";
    pub const FID: &str = "fid";
    pub const NID: &str = "nid";
    pub const SSNAMENR: &str = "ssnamenr";
    pub const TRAJECTORY_ID: &str = "trajectory_id";
    pub const FINK_CLASS: &str = "fink_class";
    pub const NALERTHIST : &str = "nalerthist";
}

/// Expected dtypes (best effort) after casting.
///
/// Notes
/// -----
/// Parquet physical types can differ (Int32 vs Int64, Float32 vs Float64).
/// We normalize them explicitly at load time.
pub fn ztf_alerts_expected_schema() -> Schema {
    Schema::from_iter([
        (cols::CANDID.into(), DataType::Int64),
        (cols::RA.into(), DataType::Float64),
        (cols::DEC.into(), DataType::Float64),
        (cols::JD.into(), DataType::Float64),
        (cols::MAGPSF.into(), DataType::Float32),
        (cols::SIGMAPSF.into(), DataType::Float32),
        (cols::FID.into(), DataType::UInt8),
        (cols::NID.into(), DataType::UInt32),
        (cols::SSNAMENR.into(), DataType::String),
        (cols::TRAJECTORY_ID.into(), DataType::Int64),
        (cols::FINK_CLASS.into(), DataType::String),
        (cols::NALERTHIST.into(), DataType::Int32),
    ])
}

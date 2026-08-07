use camino::Utf8PathBuf;
use fink_fat_engine::{engine_config::EngineConfig, topocentric_kf::branching::BranchCollection};

use crate::{
    converter::{
        parquet::{to_dataframes, write_parquet_tables},
        sql::write_sql_tables,
    },
    error::FinkFatError,
    init_cli::ConvertFormat,
};

pub mod parquet;
pub mod sql;

pub fn convert(
    config_path: Utf8PathBuf,
    requested_format: ConvertFormat,
    database_url: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::load_engine_config_validated(config_path)?;
    let snapshot_path = engine_config.snapshot_path();

    let kalman_context = engine_config.build_context();

    let collection = if snapshot_path.exists() {
        BranchCollection::load_snapshot_from_disk(&snapshot_path, &kalman_context, &engine_config)?
    } else {
        return Err(FinkFatError::NoSnapshot)?;
    };

    println!("Number of branch: {}", collection.branches.len());

    match requested_format {
        ConvertFormat::Parquet => {
            let dataframes = to_dataframes(&collection)?;
            write_parquet_tables(dataframes, &engine_config.storage_path_buf())?;
        }
        ConvertFormat::SQL => {
            let database_url = database_url.ok_or_else(|| {
                FinkFatError::Message("--database-url is required when --format sql".to_string())
            })?;
            write_sql_tables(&collection, &database_url)?;
        }
    }

    Ok(())
}

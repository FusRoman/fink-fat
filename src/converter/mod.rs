use camino::Utf8PathBuf;
use fink_fat_engine::{
    engine_config::EngineConfig,
    topocentric_kf::branching::{BranchCollection, read_archived_log},
};

use crate::{
    converter::{
        observations::read_observation_rows,
        parquet::{to_dataframes, write_parquet_tables},
    },
    error::FinkFatError,
    init_cli::ConvertFormat,
};

pub mod observations;
pub mod parquet;

pub fn convert(
    config_path: Utf8PathBuf,
    requested_format: ConvertFormat,
    database_url: Option<String>,
    path_observation: Option<Utf8PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::load_engine_config_validated(config_path)?;
    let snapshot_path = engine_config.snapshot_path();

    let kalman_context = engine_config.build_context();

    let mut collection = if snapshot_path.exists() {
        BranchCollection::load_snapshot_from_disk(&snapshot_path, &kalman_context, &engine_config)?
    } else {
        return Err(FinkFatError::NoSnapshot)?;
    };

    // Archived trajectories no longer live in the snapshot (see
    // `BranchCollection::archived`'s doc) — load the full run's archive from
    // its own append-only log instead.
    collection.archived = read_archived_log(&engine_config.archive_log_path())?;

    println!("Number of branch: {}", collection.branches.len());
    println!(
        "Number of archived trajectory: {}",
        collection.archived.len()
    );

    match requested_format {
        ConvertFormat::Parquet => {
            let dataframes = to_dataframes(&collection)?;
            write_parquet_tables(dataframes, &engine_config.storage_path_buf())?;
        }
        ConvertFormat::SQL => {
            let database_url = database_url.ok_or_else(|| {
                FinkFatError::Message("--database-url is required when --format sql".to_string())
            })?;
            let path_observation = path_observation.ok_or_else(|| {
                FinkFatError::Message(
                    "--path-observation is required when --format sql".to_string(),
                )
            })?;
            let observation_rows = read_observation_rows(&path_observation)?;
            fink_fat_sql::write_bulk_tables(&collection, &observation_rows, &database_url)?;
        }
    }

    Ok(())
}

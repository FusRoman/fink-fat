//! `fink-fat stream`: track asteroids visit by visit as alerts arrive.
//!
//! The streaming machinery lives in the `fink-fat-stream` library (see its
//! README); this module only wires it to the command line:
//!
//! 1. load the stream configuration, then the engine configuration it names;
//! 2. install logging if `--logs` is given;
//! 3. build the engine context and the replay source;
//! 4. run the [`Driver`] to the end of the source, resuming from whatever
//!    state lies under the engine's `storage_path`, and print its counters.

use fink_fat_engine::engine_config::EngineConfig;
use fink_fat_stream::{Driver, ReplaySource, StreamConfig};

use crate::{init_cli::StreamCliArgs, logging::initialize_logs};

/// Run `fink-fat stream` (see the [module doc](self)).
///
/// # Errors
/// The configuration files cannot be loaded, logging cannot be installed,
/// the replay dump cannot be read, or the driver fails (see
/// [`fink_fat_stream::StreamError`]).
pub fn stream(args: StreamCliArgs) -> Result<(), Box<dyn std::error::Error>> {
    let config = StreamConfig::load(&args.config)?;
    let engine_config = EngineConfig::load_engine_config_validated(&config.engine_config)?;
    std::fs::create_dir_all(engine_config.storage_path())?;

    // Kept alive for the whole run: dropping it flushes the log file.
    let _file_log_guard = args
        .logs
        .then(|| {
            initialize_logs(
                &engine_config,
                "fink_fat_stream",
                &args.log_targets,
                args.log_retention_days,
            )
        })
        .transpose()?;

    let kalman_context = engine_config.build_context();
    let (source, n_null_rows) = ReplaySource::from_parquet(&config.replay)?;
    tracing::info!(
        n_alerts = source.n_remaining(),
        n_null_rows,
        "replay source loaded"
    );

    let stats = Driver::new(
        source,
        config.driver_config(),
        &kalman_context,
        &engine_config,
    )?
    .run()?;
    println!("{stats:#?}");
    Ok(())
}

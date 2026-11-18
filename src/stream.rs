//! `fink-fat stream`: track asteroids visit by visit as alerts arrive.
//!
//! The streaming machinery lives in the `fink-fat-stream` library (see its
//! README); this module only wires it to the command line:
//!
//! 1. load the stream configuration, then the engine configuration it names;
//! 2. install logging if `--logs` is given;
//! 3. build the engine context and the configured source — a replayed
//!    parquet dump, or Fink's Kafka livestream;
//! 4. run the [`Driver`], resuming from whatever state lies under the
//!    engine's `storage_path`, until the source ends (a replay) or SIGTERM /
//!    SIGINT asks for a graceful stop (see [`Driver::run_until`]); then print
//!    its counters.

use std::sync::{Arc, atomic::AtomicBool};

use fink_fat_engine::engine_config::{EngineConfig, kalman_context::KalmanContext};
use fink_fat_stream::{
    AlertSource, Driver, DriverConfig, ReplaySource, StreamConfig, source::KafkaSource,
    source::SourceConfig,
};
use signal_hook::consts::{SIGINT, SIGTERM};

use crate::{init_cli::StreamCliArgs, logging::initialize_logs};

/// Run `fink-fat stream` (see the [module doc](self)).
///
/// # Errors
/// The configuration files cannot be loaded, logging or the signal handlers
/// cannot be installed, the source cannot be created (replay dump
/// unreadable, Kafka consumer refused), or the driver fails (see
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

    // A first signal raises the flag (graceful stop, see
    // `Driver::run_until`); a second one while stopping terminates at once.
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [SIGTERM, SIGINT] {
        signal_hook::flag::register_conditional_shutdown(signal, 1, Arc::clone(&stop))?;
        signal_hook::flag::register(signal, Arc::clone(&stop))?;
    }

    let kalman_context = engine_config.build_context();
    let driver_config = config.driver_config();
    match config.source {
        SourceConfig::Replay(replay) => {
            let (source, n_null_rows) = ReplaySource::from_parquet(&replay)?;
            tracing::info!(
                n_alerts = source.n_remaining(),
                n_null_rows,
                "replay source loaded"
            );
            run_driver(
                source,
                driver_config,
                &kalman_context,
                &engine_config,
                &stop,
            )
        }
        SourceConfig::Kafka(kafka) => run_driver(
            KafkaSource::new(kafka)?,
            driver_config,
            &kalman_context,
            &engine_config,
            &stop,
        ),
    }
}

/// Run the driver on `source` until it ends or `stop` is raised, and print
/// its counters.
///
/// # Errors
/// The driver fails (see [`fink_fat_stream::StreamError`]).
fn run_driver<S: AlertSource>(
    source: S,
    driver_config: DriverConfig,
    kalman_context: &KalmanContext,
    engine_config: &EngineConfig,
    stop: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stats =
        Driver::new(source, driver_config, kalman_context, engine_config)?.run_until(stop)?;
    println!("{stats:#?}");
    Ok(())
}

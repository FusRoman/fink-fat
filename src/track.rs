use std::io::{BufWriter, Write};

use fink_fat_engine::{
    engine_config::EngineConfig,
    topocentric_kf::branching::{BranchCollection, write_archived_batch},
};
use photom::{
    NightId,
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::{ObsDataset, observation::Observation},
    observer::error_model::ObsErrorModel,
};
use polars::lazy::frame::{LazyFrame, ScanArgsParquet};

use crate::{
    init_cli::FinkFatCliArgs,
    logging::{initialize_logs, print_log_targets},
};

pub fn tracking(cli_args: FinkFatCliArgs) -> Result<(), Box<dyn std::error::Error>> {
    if cli_args.list_log_targets {
        print_log_targets();
        return Ok(());
    }

    let alerts = cli_args
        .alerts
        .clone()
        .expect("required unless --list-log-targets");
    let config = cli_args
        .config
        .clone()
        .expect("required unless --list-log-targets");

    let engine_config = EngineConfig::load_engine_config_validated(config)?;

    std::fs::create_dir_all(engine_config.storage_path())?;

    // Kept alive for the whole run: dropping it flushes the file writer's
    // background worker. `None` when `--logs` wasn't passed.
    let mut _file_log_guard: Option<tracing_appender::non_blocking::WorkerGuard> = None;

    if cli_args.logs {
        initialize_logs(&cli_args, &engine_config, &mut _file_log_guard)?;
    }

    let lf = LazyFrame::scan_parquet(alerts.as_str().into(), ScanArgsParquet::default())?;
    let obs_dataset = ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )?;
    let kalman_context = engine_config.build_context();

    let snapshot_path = engine_config.snapshot_path();

    let mut collection = if snapshot_path.exists() {
        BranchCollection::load_snapshot_from_disk(&snapshot_path, &kalman_context, &engine_config)?
    } else {
        BranchCollection::empty()
    };

    // Kept open and appended to for the whole run, independently of
    // `--snapshot-every`: archived trajectories are small (see
    // `ArchivedTrajectory`'s doc) and this is a cheap append, not a full
    // rewrite, so there's no reason to batch it with the (much more
    // expensive) branch-collection snapshot cadence — this also means an
    // archived trajectory survives a crash as soon as it's written, rather
    // than only at the next snapshot as before.
    let archive_log_path = engine_config.archive_log_path();
    let mut archive_log_writer = BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&archive_log_path)?,
    );

    // A dataset tagged with more than one `night_id` is processed one night
    // at a time, in chronological order; anything else (no night index, or
    // a single night) is treated as one logical night, exactly as before.
    // Each batch carries its night id when the dataset has one, so a run
    // resumed from a snapshot skips the nights it already advanced.
    let night_batches: Vec<(Option<NightId>, Vec<&Observation>)> = match obs_dataset.nb_night() {
        Some(n) if n > 1 => {
            let mut night_ids: Vec<_> = obs_dataset
                .iter_night_id()
                .expect("nb_night() > 1 implies a night index exists")
                .copied()
                .collect();
            night_ids.sort_unstable();
            night_ids
                .into_iter()
                .map(|night_id| {
                    let night_obs = obs_dataset
                        .iter_night_observations(&night_id)
                        .expect("night_id came from iter_night_id()")
                        .collect();
                    (Some(night_id), night_obs)
                })
                .collect()
        }
        _ => {
            let single_night = obs_dataset
                .iter_night_id()
                .and_then(|mut ids| ids.next().copied());
            vec![(single_night, obs_dataset.iter_observations().collect())]
        }
    };

    let n_batches = night_batches.len();
    for (i, (night_id, night_obs)) in night_batches.iter().enumerate() {
        if let Some(night_id) = *night_id
            && collection.is_night_closed(night_id)
        {
            tracing::info!(
                %night_id,
                last_closed = ?collection.last_closed_night_id,
                "night already advanced by a previous run, skipped"
            );
            continue;
        }

        let current_step = collection.current_step;
        collection = collection.advance_one_night(
            night_obs,
            &obs_dataset,
            &engine_config,
            &kalman_context,
            current_step,
            *night_id,
        )?;

        // `collection.archived` is only this night's fresh batch (see
        // `BranchCollection::archived`'s doc) — flush it to the on-disk log
        // and drop it immediately rather than let it ride along in RAM.
        write_archived_batch(&mut archive_log_writer, &collection.archived)?;
        archive_log_writer.flush()?;
        collection.archived.clear();

        if should_write_snapshot(i + 1, n_batches, cli_args.snapshot_every) {
            collection.save_snapshot_to_disk(&snapshot_path)?;
        }
    }

    Ok(())
}

/// Whether the snapshot should be written to disk after processing the
/// `nights_done`-th night (1-based) out of `total` in this run.
///
/// Always `true` on the last night, regardless of `snapshot_every`, so a
/// run never finishes without persisting its final state. Otherwise `true`
/// every `snapshot_every` nights, if set.
fn should_write_snapshot(nights_done: usize, total: usize, snapshot_every: Option<usize>) -> bool {
    nights_done == total || snapshot_every.is_some_and(|n| n > 0 && nights_done.is_multiple_of(n))
}

#[cfg(test)]
mod tracking_test {
    use super::*;

    #[test]
    fn should_write_snapshot_always_true_on_last_night() {
        assert!(should_write_snapshot(1, 1, None));
        assert!(should_write_snapshot(5, 5, None));
        assert!(should_write_snapshot(5, 5, Some(1000)));
    }

    #[test]
    fn should_write_snapshot_false_between_intervals_without_flag() {
        assert!(!should_write_snapshot(1, 5, None));
        assert!(!should_write_snapshot(4, 5, None));
    }

    #[test]
    fn should_write_snapshot_true_every_n_nights() {
        assert!(should_write_snapshot(3, 10, Some(3)));
        assert!(!should_write_snapshot(4, 10, Some(3)));
        assert!(should_write_snapshot(6, 10, Some(3)));
    }
}

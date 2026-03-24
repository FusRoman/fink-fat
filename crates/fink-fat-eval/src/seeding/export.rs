//! Parquet export of seeding results.
//!
//! This module exports one row per alert membership in a seed. The main goal is
//! to provide a flat table that includes a `seed_id` column so downstream tools
//! can group alerts by seed.

use std::fs::File;

use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::pipeline::PipelineContext;
use polars::prelude::*;

use crate::truth_sso::TruthSSO;

/// Export seeding memberships as a Parquet file.
///
/// Output schema:
/// - `seed_id` (u64): unique identifier of the seed.
/// - `seed_night_id` (u32): night ID of the seed.
/// - `seed_n_obs` (u32): number of members in the seed (2 for pair, 3 for triplet).
/// - `member_rank` (u32): position of the alert in the seed member list.
/// - `alert_dia_source_id` (u64): source alert identifier.
/// - `alert_night_id` (u32): alert night ID.
/// - `truth_trajectory_id` (u32): truth trajectory ID, `0` when unknown.
///
/// Notes:
/// - A single alert can appear in multiple seeds, therefore multiple rows can
///   share the same `alert_dia_source_id`.
/// - Parent directories are created automatically.
pub fn export_seeding_members_parquet(
    ctx: &PipelineContext,
    truth: &TruthSSO,
    out_path: &Utf8Path,
) -> Result<()> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;

    let mut nights: Vec<_> = seed_store.nights().copied().collect();
    nights.sort();

    let mut n_rows_estimate: usize = 0;
    for night_id in &nights {
        if let Some(seeds) = seed_store.get(night_id) {
            n_rows_estimate += seeds.iter().map(|s| s.members.len()).sum::<usize>();
        }
    }

    let mut col_seed_id: Vec<u64> = Vec::with_capacity(n_rows_estimate);
    let mut col_seed_night_id: Vec<u32> = Vec::with_capacity(n_rows_estimate);
    let mut col_seed_n_obs: Vec<u32> = Vec::with_capacity(n_rows_estimate);
    let mut col_member_rank: Vec<u32> = Vec::with_capacity(n_rows_estimate);
    let mut col_alert_dia_source_id: Vec<u64> = Vec::with_capacity(n_rows_estimate);
    let mut col_alert_night_id: Vec<u32> = Vec::with_capacity(n_rows_estimate);
    let mut col_truth_trajectory_id: Vec<u32> = Vec::with_capacity(n_rows_estimate);

    for night_id in nights {
        let Some(seeds) = seed_store.get(&night_id) else {
            continue;
        };

        for seed in seeds {
            let seed_id = seed.key().unique_id;
            let seed_night_id = seed.night_id().value();
            let seed_n_obs = seed.n_obs as u32;

            let resolved = seed
                .resolve_members(alert_store)
                .context("failed to resolve seed members for seeding parquet export")?;

            for (member_rank, alert) in resolved.iter().enumerate() {
                col_seed_id.push(seed_id);
                col_seed_night_id.push(seed_night_id);
                col_seed_n_obs.push(seed_n_obs);
                col_member_rank.push(member_rank as u32);
                col_alert_dia_source_id.push(alert.key.dia_source_id);
                col_alert_night_id.push(alert.key.night_id.value());
                col_truth_trajectory_id.push(truth.get_truth_traj_id(alert).unwrap_or(0));
            }
        }
    }

    let mut df = DataFrame::new(vec![
        Series::new("seed_id".into(), &col_seed_id).into_column(),
        Series::new("seed_night_id".into(), &col_seed_night_id).into_column(),
        Series::new("seed_n_obs".into(), &col_seed_n_obs).into_column(),
        Series::new("member_rank".into(), &col_member_rank).into_column(),
        Series::new("alert_dia_source_id".into(), &col_alert_dia_source_id).into_column(),
        Series::new("alert_night_id".into(), &col_alert_night_id).into_column(),
        Series::new("truth_trajectory_id".into(), &col_truth_trajectory_id).into_column(),
    ])
    .context("building seeding membership DataFrame")?;

    if let Some(parent) = out_path.parent()
        && !parent.as_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating parent directory '{parent}'"))?;
    }

    let mut file = File::create(out_path.as_std_path())
        .with_context(|| format!("creating output file '{out_path}'"))?;

    ParquetWriter::new(&mut file)
        .finish(&mut df)
        .with_context(|| format!("writing Parquet to '{out_path}'"))?;

    tracing::info!(
        path = %out_path,
        n_rows = col_seed_id.len(),
        "seeding membership dataset written",
    );

    Ok(())
}

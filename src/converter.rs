use std::fs::File;

use camino::Utf8PathBuf;
use fink_fat_engine::{
    engine_config::EngineConfig,
    topocentric_kf::{branching::BranchCollection, single_kalman::KFStateSnapshot},
};
use polars::prelude::*;

use crate::{error::FinkFatError, init_cli::ConvertFormat};

/// Convert one fixed-size (or ragged) value per row into a Polars `List`
/// column: [`Column::new`] only accepts a flat `Vec<T>` directly, so a
/// `Vec<Vec<T>>` needs each row wrapped in its own unnamed [`Series`] first.
fn to_list_series<T>(rows: Vec<Vec<T>>) -> Vec<Series>
where
    Series: NamedFrom<Vec<T>, [T]>,
{
    rows.into_iter()
        .map(|row| Series::new(PlSmallStr::EMPTY, row))
        .collect()
}

/// Like [`to_list_series`], for an optional list per row (e.g. `kalman_gain`,
/// absent before the first real update).
fn to_opt_list_series<T>(rows: Vec<Option<Vec<T>>>) -> Vec<Option<Series>>
where
    Series: NamedFrom<Vec<T>, [T]>,
{
    rows.into_iter()
        .map(|row| row.map(|row| Series::new(PlSmallStr::EMPTY, row)))
        .collect()
}

/// The five tables a [`BranchCollection`] is translated into, joined by
/// `branch_id` (`branches`/`kf_bank`/`branch_observations`) and
/// `hypothesis_id` (`hypotheses`/`kf_state`). `archived_trajectories` stands
/// alone (no live bank to join against), keyed by its own `designation`.
pub struct BranchCollectionDataFrames {
    pub branches: DataFrame,
    pub kf_bank: DataFrame,
    pub branch_observations: DataFrame,
    pub hypotheses: DataFrame,
    pub kf_state: DataFrame,
    pub archived_trajectories: DataFrame,
}

/// Columns shared by the `kf_state` and `archived_trajectories` tables,
/// built from a single [`KFStateSnapshot`] — used both for a live
/// hypothesis's `kf.to_snapshot()` and an archived trajectory's already-
/// snapshotted `map_state`, so the two tables encode a `KFState` identically.
#[derive(Default)]
struct KfStateColumns {
    ra: Vec<f64>,
    dec: Vec<f64>,
    ra_dot: Vec<f64>,
    dec_dot: Vec<f64>,
    rho: Vec<f64>,
    rho_dot: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    epoch: Vec<f64>,
    r_obs_x: Vec<f64>,
    r_obs_y: Vec<f64>,
    r_obs_z: Vec<f64>,
    v_obs_x: Vec<f64>,
    v_obs_y: Vec<f64>,
    v_obs_z: Vec<f64>,
    universal_anomaly: Vec<Option<f64>>,
    kalman_gain: Vec<Option<Vec<f64>>>,
    nis_ema: Vec<Option<f64>>,
}

impl KfStateColumns {
    fn with_capacity(n: usize) -> Self {
        Self {
            ra: Vec::with_capacity(n),
            dec: Vec::with_capacity(n),
            ra_dot: Vec::with_capacity(n),
            dec_dot: Vec::with_capacity(n),
            rho: Vec::with_capacity(n),
            rho_dot: Vec::with_capacity(n),
            covariance: Vec::with_capacity(n),
            epoch: Vec::with_capacity(n),
            r_obs_x: Vec::with_capacity(n),
            r_obs_y: Vec::with_capacity(n),
            r_obs_z: Vec::with_capacity(n),
            v_obs_x: Vec::with_capacity(n),
            v_obs_y: Vec::with_capacity(n),
            v_obs_z: Vec::with_capacity(n),
            universal_anomaly: Vec::with_capacity(n),
            kalman_gain: Vec::with_capacity(n),
            nis_ema: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, kf: &KFStateSnapshot) {
        self.ra.push(kf.state[0]);
        self.dec.push(kf.state[1]);
        self.ra_dot.push(kf.state[2]);
        self.dec_dot.push(kf.state[3]);
        self.rho.push(kf.state[4]);
        self.rho_dot.push(kf.state[5]);
        self.covariance.push(kf.covariance.to_vec());
        self.epoch.push(kf.epoch);
        self.r_obs_x.push(kf.r_obs[0]);
        self.r_obs_y.push(kf.r_obs[1]);
        self.r_obs_z.push(kf.r_obs[2]);
        self.v_obs_x.push(kf.v_obs[0]);
        self.v_obs_y.push(kf.v_obs[1]);
        self.v_obs_z.push(kf.v_obs[2]);
        self.universal_anomaly.push(kf.universal_anomaly);
        self.kalman_gain.push(kf.kalman_gain.map(|g| g.to_vec()));
        self.nis_ema.push(kf.nis_ema);
    }

    /// Consume into the `kf_state`-shaped columns, prefixed with `id_column`
    /// (the join key: `hypothesis_id` for `kf_state`, `designation` isn't
    /// needed here since `archived_trajectories` already carries its own).
    fn into_columns(self) -> Vec<Column> {
        vec![
            Column::new("ra".into(), self.ra),
            Column::new("dec".into(), self.dec),
            Column::new("ra_dot".into(), self.ra_dot),
            Column::new("dec_dot".into(), self.dec_dot),
            Column::new("rho".into(), self.rho),
            Column::new("rho_dot".into(), self.rho_dot),
            Column::new("covariance".into(), to_list_series(self.covariance)),
            Column::new("epoch".into(), self.epoch),
            Column::new("r_obs_x".into(), self.r_obs_x),
            Column::new("r_obs_y".into(), self.r_obs_y),
            Column::new("r_obs_z".into(), self.r_obs_z),
            Column::new("v_obs_x".into(), self.v_obs_x),
            Column::new("v_obs_y".into(), self.v_obs_y),
            Column::new("v_obs_z".into(), self.v_obs_z),
            Column::new("universal_anomaly".into(), self.universal_anomaly),
            Column::new("kalman_gain".into(), to_opt_list_series(self.kalman_gain)),
            Column::new("nis_ema".into(), self.nis_ema),
        ]
    }
}

/// Translate a [`BranchCollection`] into the five joined [`DataFrame`]s
/// described in [`BranchCollectionDataFrames`], ready to be persisted (e.g.
/// to Parquet) independently of `fink-fat-engine`, which deliberately does
/// not depend on Polars.
pub fn to_dataframes(
    branch_collection: &BranchCollection,
) -> Result<BranchCollectionDataFrames, PolarsError> {
    let n_branches = branch_collection.branches.len();
    let n_track_id_rows: usize = branch_collection
        .branches
        .iter()
        .map(|b| b.bank.track_ids().len())
        .sum();
    let n_hypotheses: usize = branch_collection
        .branches
        .iter()
        .map(|b| b.bank.hypotheses().len())
        .sum();
    let n_archived = branch_collection.archived.len();

    // `branches`
    let mut branch_id = Vec::with_capacity(n_branches);
    let mut lineage_id = Vec::with_capacity(n_branches);
    let mut parent_branch_id = Vec::with_capacity(n_branches);
    let mut ancestor_at_scan_horizon = Vec::with_capacity(n_branches);
    let mut ancestor_creation_step = Vec::with_capacity(n_branches);
    let mut last_real_update_step = Vec::with_capacity(n_branches);
    let mut n_real_updates = Vec::with_capacity(n_branches);
    let mut cumulative_llr = Vec::with_capacity(n_branches);
    let mut lineage_designation = Vec::with_capacity(n_branches);
    let mut designation = Vec::with_capacity(n_branches);

    // `kf_bank`
    let mut bank_branch_id = Vec::with_capacity(n_branches);
    let mut n_steps = Vec::with_capacity(n_branches);
    let mut absolute_magnitude_estimate = Vec::with_capacity(n_branches);
    let mut absolute_magnitude_sample_count = Vec::with_capacity(n_branches);

    // `branch_observations`
    let mut obs_branch_id = Vec::with_capacity(n_track_id_rows);
    let mut obs_position = Vec::with_capacity(n_track_id_rows);
    let mut obs_id = Vec::with_capacity(n_track_id_rows);

    // `hypotheses`
    let mut hypothesis_id = Vec::with_capacity(n_hypotheses);
    let mut hyp_branch_id = Vec::with_capacity(n_hypotheses);
    let mut local_hyp_id = Vec::with_capacity(n_hypotheses);
    let mut log_weight = Vec::with_capacity(n_hypotheses);
    let mut recent_log_liks = Vec::with_capacity(n_hypotheses);

    // `kf_state`
    let mut kf_state_hypothesis_id = Vec::with_capacity(n_hypotheses);
    let mut kf_state_columns = KfStateColumns::with_capacity(n_hypotheses);

    let mut next_hypothesis_id: u64 = 0;

    for branch in &branch_collection.branches {
        branch_id.push(branch.branch_id);
        lineage_id.push(branch.lineage_id);
        parent_branch_id.push(branch.parent_branch_id);
        ancestor_at_scan_horizon.push(branch.ancestor_at_scan_horizon);
        ancestor_creation_step.push(branch.ancestor_creation_step as u64);
        last_real_update_step.push(branch.last_real_update_step as u64);
        n_real_updates.push(branch.n_real_updates as u64);
        cumulative_llr.push(branch.cumulative_llr);
        lineage_designation.push(branch.lineage_designation.to_string());
        designation.push(branch.designation().to_string());

        let bank_snapshot = branch.bank.to_snapshot();

        bank_branch_id.push(branch.branch_id);
        n_steps.push(bank_snapshot.n_steps as u64);
        absolute_magnitude_estimate.push(bank_snapshot.absolute_magnitude_estimate);
        absolute_magnitude_sample_count.push(bank_snapshot.absolute_magnitude_sample_count);

        for (position, id) in bank_snapshot.track_ids.iter().enumerate() {
            obs_branch_id.push(branch.branch_id);
            obs_position.push(position as u32);
            obs_id.push(*id);
        }

        for hyp_snapshot in bank_snapshot.hypotheses {
            let this_hypothesis_id = next_hypothesis_id;
            next_hypothesis_id += 1;

            hypothesis_id.push(this_hypothesis_id);
            hyp_branch_id.push(branch.branch_id);
            local_hyp_id.push(hyp_snapshot.id);
            log_weight.push(hyp_snapshot.log_weight);
            recent_log_liks.push(hyp_snapshot.recent_log_liks);

            kf_state_hypothesis_id.push(this_hypothesis_id);
            kf_state_columns.push(&hyp_snapshot.kf);
        }
    }

    let branches = DataFrame::new_infer_height(vec![
        Column::new("branch_id".into(), branch_id),
        Column::new("lineage_id".into(), lineage_id),
        Column::new("parent_branch_id".into(), parent_branch_id),
        Column::new("ancestor_at_scan_horizon".into(), ancestor_at_scan_horizon),
        Column::new("ancestor_creation_step".into(), ancestor_creation_step),
        Column::new("last_real_update_step".into(), last_real_update_step),
        Column::new("n_real_updates".into(), n_real_updates),
        Column::new("cumulative_llr".into(), cumulative_llr),
        Column::new("lineage_designation".into(), lineage_designation),
        Column::new("designation".into(), designation),
    ])?;

    let kf_bank = DataFrame::new_infer_height(vec![
        Column::new("branch_id".into(), bank_branch_id),
        Column::new("n_steps".into(), n_steps),
        Column::new(
            "absolute_magnitude_estimate".into(),
            absolute_magnitude_estimate,
        ),
        Column::new(
            "absolute_magnitude_sample_count".into(),
            absolute_magnitude_sample_count,
        ),
    ])?;

    let branch_observations = DataFrame::new_infer_height(vec![
        Column::new("branch_id".into(), obs_branch_id),
        Column::new("position".into(), obs_position),
        Column::new("obs_id".into(), obs_id),
    ])?;

    let hypotheses = DataFrame::new_infer_height(vec![
        Column::new("hypothesis_id".into(), hypothesis_id),
        Column::new("branch_id".into(), hyp_branch_id),
        Column::new("local_hyp_id".into(), local_hyp_id),
        Column::new("log_weight".into(), log_weight),
        Column::new("recent_log_liks".into(), to_list_series(recent_log_liks)),
    ])?;

    let mut kf_state_column_list =
        vec![Column::new("hypothesis_id".into(), kf_state_hypothesis_id)];
    kf_state_column_list.extend(kf_state_columns.into_columns());
    let kf_state = DataFrame::new_infer_height(kf_state_column_list)?;

    // `archived_trajectories`
    let mut archived_designation = Vec::with_capacity(n_archived);
    let mut archived_lineage_id = Vec::with_capacity(n_archived);
    let mut archived_track_ids = Vec::with_capacity(n_archived);
    let mut archived_cumulative_llr = Vec::with_capacity(n_archived);
    let mut archived_n_real_updates = Vec::with_capacity(n_archived);
    let mut archived_last_real_update_step = Vec::with_capacity(n_archived);
    let mut archived_at_step = Vec::with_capacity(n_archived);
    let mut archived_absolute_magnitude_estimate = Vec::with_capacity(n_archived);
    let mut archived_absolute_magnitude_sample_count = Vec::with_capacity(n_archived);
    let mut archived_kf_state_columns = KfStateColumns::with_capacity(n_archived);

    for trajectory in &branch_collection.archived {
        archived_designation.push(trajectory.designation.to_string());
        archived_lineage_id.push(trajectory.lineage_id);
        archived_track_ids.push(trajectory.track_ids.clone());
        archived_cumulative_llr.push(trajectory.cumulative_llr);
        archived_n_real_updates.push(trajectory.n_real_updates as u64);
        archived_last_real_update_step.push(trajectory.last_real_update_step as u64);
        archived_at_step.push(trajectory.archived_at_step as u64);
        archived_absolute_magnitude_estimate.push(trajectory.absolute_magnitude_estimate);
        archived_absolute_magnitude_sample_count.push(trajectory.absolute_magnitude_sample_count);
        archived_kf_state_columns.push(&trajectory.map_state);
    }

    let mut archived_column_list = vec![
        Column::new("designation".into(), archived_designation),
        Column::new("lineage_id".into(), archived_lineage_id),
        Column::new("track_ids".into(), to_list_series(archived_track_ids)),
        Column::new("cumulative_llr".into(), archived_cumulative_llr),
        Column::new("n_real_updates".into(), archived_n_real_updates),
        Column::new(
            "last_real_update_step".into(),
            archived_last_real_update_step,
        ),
        Column::new("archived_at_step".into(), archived_at_step),
        Column::new(
            "absolute_magnitude_estimate".into(),
            archived_absolute_magnitude_estimate,
        ),
        Column::new(
            "absolute_magnitude_sample_count".into(),
            archived_absolute_magnitude_sample_count,
        ),
    ];
    archived_column_list.extend(archived_kf_state_columns.into_columns());
    let archived_trajectories = DataFrame::new_infer_height(archived_column_list)?;

    Ok(BranchCollectionDataFrames {
        branches,
        kf_bank,
        branch_observations,
        hypotheses,
        kf_state,
        archived_trajectories,
    })
}

/// Write every table in `dataframes` to `<output_dir>/<table_name>.parquet`.
fn write_parquet_tables(
    dataframes: BranchCollectionDataFrames,
    output_dir: &camino::Utf8Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let tables: [(&str, DataFrame); 6] = [
        ("branches", dataframes.branches),
        ("kf_bank", dataframes.kf_bank),
        ("branch_observations", dataframes.branch_observations),
        ("hypotheses", dataframes.hypotheses),
        ("kf_state", dataframes.kf_state),
        ("archived_trajectories", dataframes.archived_trajectories),
    ];

    for (name, mut df) in tables {
        let out_path = output_dir.join(format!("{name}.parquet"));
        let file = File::create(&out_path)?;
        ParquetWriter::new(file).finish(&mut df)?;
    }

    Ok(())
}

pub fn convert(
    config_path: Utf8PathBuf,
    requested_format: ConvertFormat,
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
            Err(FinkFatError::Message(
                "SQL export is not implemented yet".to_string(),
            ))?;
        }
    }

    Ok(())
}

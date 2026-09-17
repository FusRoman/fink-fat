//! Row shapes for every table [`super::write_sql_tables`] bulk-loads.
//!
//! Plain data only — no I/O, no Postgres types. One struct per table, built
//! by [`super::build`] from a [`fink_fat_engine::topocentric_kf::branching::BranchCollection`]
//! and consumed by [`super::copy`]'s `COPY ... FROM STDIN BINARY` writers.

use fink_fat_engine::topocentric_kf::single_kalman::KFStateSnapshot;

/// Columns shared by the `kf_state` and `archived_trajectories` tables,
/// mirroring [`crate::converter::parquet::KfStateColumns`] but as a per-row
/// struct instead of per-column vectors, since `COPY` writes one row at a
/// time.
pub(super) struct KfStateFields {
    pub ra: f64,
    pub dec: f64,
    pub ra_dot: f64,
    pub dec_dot: f64,
    pub rho: f64,
    pub rho_dot: f64,
    pub covariance: Vec<f64>,
    pub epoch: f64,
    pub r_obs_x: f64,
    pub r_obs_y: f64,
    pub r_obs_z: f64,
    pub v_obs_x: f64,
    pub v_obs_y: f64,
    pub v_obs_z: f64,
    pub universal_anomaly: Option<f64>,
    pub kalman_gain: Option<Vec<f64>>,
    pub nis_ema: Option<f64>,
}

impl KfStateFields {
    /// Maps one Kalman-filter state snapshot to the row shape `COPY` writes.
    ///
    /// # Arguments
    ///
    /// * `kf` — the snapshot to convert, as held by a branch's best
    ///   hypothesis or by an archived trajectory's final state.
    ///
    /// # Returns
    ///
    /// The equivalent [`KfStateFields`], a plain field-for-field copy (no
    /// unit conversion or validation — `kf`'s fields are already in the
    /// representation the `kf_state`/`archived_trajectories` columns expect).
    pub(super) fn from_snapshot(kf: &KFStateSnapshot) -> Self {
        Self {
            ra: kf.state[0],
            dec: kf.state[1],
            ra_dot: kf.state[2],
            dec_dot: kf.state[3],
            rho: kf.state[4],
            rho_dot: kf.state[5],
            covariance: kf.covariance.to_vec(),
            epoch: kf.epoch,
            r_obs_x: kf.r_obs[0],
            r_obs_y: kf.r_obs[1],
            r_obs_z: kf.r_obs[2],
            v_obs_x: kf.v_obs[0],
            v_obs_y: kf.v_obs[1],
            v_obs_z: kf.v_obs[2],
            universal_anomaly: kf.universal_anomaly,
            kalman_gain: kf.kalman_gain.map(|g| g.to_vec()),
            nis_ema: kf.nis_ema,
        }
    }
}

/// One row of the `branches` table.
pub(super) struct BranchRow {
    pub branch_id: i64,
    pub lineage_id: i64,
    pub parent_branch_id: i64,
    pub ancestor_at_scan_horizon: i64,
    pub ancestor_creation_step: i64,
    pub last_real_update_step: i64,
    pub n_real_updates: i64,
    pub cumulative_llr: f64,
    pub lineage_designation: String,
    pub designation: String,
    pub arc_length_days: f64,
    pub n_nights: i64,
    pub median_inter_night_dt_days: Option<f64>,
}

/// One row of the `kf_bank` table.
pub(super) struct KfBankRow {
    pub branch_id: i64,
    pub n_steps: i64,
    pub absolute_magnitude_estimate: Option<f64>,
    pub absolute_magnitude_sample_count: i32,
}

/// One row of the `branch_observations` join table.
pub(super) struct BranchObservationRow {
    pub branch_id: i64,
    pub position: i32,
    pub obs_id: i64,
}

/// One row of the `hypotheses` table.
pub(super) struct HypothesisRow {
    pub hypothesis_id: i64,
    pub branch_id: i64,
    pub local_hyp_id: i64,
    pub log_weight: f64,
    pub recent_log_liks: Vec<f64>,
}

/// One row of the `kf_state` table.
pub(super) struct KfStateRow {
    pub hypothesis_id: i64,
    pub fields: KfStateFields,
}

/// One row of the `archived_trajectories` table.
pub(super) struct ArchivedRow {
    pub designation: String,
    pub lineage_id: i64,
    pub track_ids: Vec<i64>,
    pub cumulative_llr: f64,
    pub n_real_updates: i64,
    pub last_real_update_step: i64,
    pub archived_at_step: i64,
    pub absolute_magnitude_estimate: Option<f64>,
    pub absolute_magnitude_sample_count: i32,
    pub kf_state: KfStateFields,
}

/// One row of the `observations` table: a single alert as produced by
/// `test_exp/prep_alert.py`. The ground-truth `traj_id` column present in
/// that parquet is deliberately not read here — it's an evaluation-only
/// artifact with no equivalent on real production data, and this table is
/// meant to also hold production observations.
pub(super) struct ObservationRow {
    pub id: i64,
    pub night_id: i64,
    pub object_id: String,
    pub magnitude: f64,
    pub mag_err: f64,
    pub filter: i16,
    pub mpc_code_obs: String,
    pub ra: f64,
    pub ra_err: f64,
    pub dec: f64,
    pub dec_err: f64,
    pub mjd_tt: f64,
}

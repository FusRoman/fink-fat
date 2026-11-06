//! Row shapes for every table [`crate::write_bulk_tables`] bulk-loads.
//!
//! Plain data only — no I/O, no Postgres types. One struct per table, built
//! by [`crate::build`] from a [`fink_fat_engine::topocentric_kf::branching::BranchCollection`]
//! and consumed by [`crate::copy`]'s `COPY ... FROM STDIN BINARY` writers.

use fink_fat_engine::topocentric_kf::single_kalman::KFStateSnapshot;

/// Columns shared by the `kf_state` and `archived_trajectories` tables,
/// mirroring the `fink-fat` binary's `converter::parquet::KfStateColumns` but as a per-row
/// struct instead of per-column vectors, since `COPY` writes one row at a
/// time.
pub struct KfStateFields {
    /// Right ascension α of the attributable state, rad (equatorial J2000).
    pub ra: f64,
    /// Declination δ, rad.
    pub dec: f64,
    /// Rate of α, rad/day.
    pub ra_dot: f64,
    /// Rate of δ, rad/day.
    pub dec_dot: f64,
    /// Topocentric range ρ, AU.
    pub rho: f64,
    /// Range rate, AU/day.
    pub rho_dot: f64,
    /// 6×6 state covariance in attributable space, flattened (36 values, as in the engine snapshot).
    pub covariance: Vec<f64>,
    /// Epoch of the state, MJD (TT).
    pub epoch: f64,
    /// Heliocentric observer position x at `epoch`, AU (ecliptic J2000).
    pub r_obs_x: f64,
    /// Heliocentric observer position y, AU.
    pub r_obs_y: f64,
    /// Heliocentric observer position z, AU.
    pub r_obs_z: f64,
    /// Heliocentric observer velocity x at `epoch`, AU/day (ecliptic J2000).
    pub v_obs_x: f64,
    /// Heliocentric observer velocity y, AU/day.
    pub v_obs_y: f64,
    /// Heliocentric observer velocity z, AU/day.
    pub v_obs_z: f64,
    /// Previous universal-anomaly estimate used for propagation, if any.
    pub universal_anomaly: Option<f64>,
    /// 6×2 Kalman gain of the last update, flattened (12 values); `None` before any update.
    pub kalman_gain: Option<Vec<f64>>,
    /// Exponential moving average of the NIS driving fading-memory inflation; `None` before any update.
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
    pub fn from_snapshot(kf: &KFStateSnapshot) -> Self {
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
pub struct BranchRow {
    /// Primary key: the engine's branch id.
    pub branch_id: i64,
    /// Lineage the branch belongs to.
    pub lineage_id: i64,
    /// Branch this one was spawned from.
    pub parent_branch_id: i64,
    /// `branch_id` of the ancestor anchoring the current N-scan window.
    pub ancestor_at_scan_horizon: i64,
    /// Night index at which `ancestor_at_scan_horizon` was last reset.
    pub ancestor_creation_step: i64,
    /// Last step with a real (non-null) observation update.
    pub last_real_update_step: i64,
    /// Number of real observation updates.
    pub n_real_updates: i64,
    /// Cumulative log-likelihood ratio of the branch.
    pub cumulative_llr: f64,
    /// Designation of the lineage (shared by its branches).
    pub lineage_designation: String,
    /// Designation of this branch.
    pub designation: String,
    /// See [`crate::build::ObsStats::arc_length_days`].
    pub arc_length_days: f64,
    /// See [`crate::build::ObsStats::n_nights`].
    pub n_nights: i64,
    /// See [`crate::build::ObsStats::median_inter_night_dt_days`].
    pub median_inter_night_dt_days: Option<f64>,
    /// See [`crate::build::ObsStats::n_observations`].
    pub n_observations: i64,
    /// See [`crate::build::ObsStats::well_sampled_nights`].
    pub well_sampled_nights: i64,
}

/// One row of the `branch_best_state` table: which `kf_state` row holds a
/// branch's best hypothesis. The columns copied from it are resolved at
/// `COPY` time (see `copy::copy_branch_best_state`), so the state is never
/// duplicated in memory.
pub struct BranchBestStateRow {
    /// Branch whose best hypothesis this row points to.
    pub branch_id: i64,
    /// Index of the best hypothesis' row in the `kf_state` row vector built
    /// alongside this one.
    pub kf_state_index: usize,
}

/// One row of the `kf_bank` table.
pub struct KfBankRow {
    /// Primary key, references `branches(branch_id)`.
    pub branch_id: i64,
    /// Number of steps the branch's Kalman bank has run.
    pub n_steps: i64,
    /// Running absolute-magnitude estimate H, if any.
    pub absolute_magnitude_estimate: Option<f64>,
    /// Number of samples behind that estimate.
    pub absolute_magnitude_sample_count: i32,
}

/// One row of the `branch_observations` join table.
pub struct BranchObservationRow {
    /// References `branches(branch_id)`.
    pub branch_id: i64,
    /// Rank of the observation in the branch's track (0-based).
    pub position: i32,
    /// References `observations(id)`.
    pub obs_id: i64,
}

/// One row of the `hypotheses` table.
pub struct HypothesisRow {
    /// Primary key, allocated sequentially across every branch of one load.
    pub hypothesis_id: i64,
    /// References `branches(branch_id)`.
    pub branch_id: i64,
    /// Hypothesis id inside its branch's bank.
    pub local_hyp_id: i64,
    /// Log mixture weight of the hypothesis.
    pub log_weight: f64,
    /// Recent per-update log-likelihoods.
    pub recent_log_liks: Vec<f64>,
}

/// One row of the `kf_state` table.
pub struct KfStateRow {
    /// Primary key, references `hypotheses(hypothesis_id)`.
    pub hypothesis_id: i64,
    /// The hypothesis' Kalman state.
    pub fields: KfStateFields,
}

/// One row of the `archived_trajectories` table.
pub struct ArchivedRow {
    /// Primary key: designation of the archived trajectory.
    pub designation: String,
    /// Lineage the trajectory came from.
    pub lineage_id: i64,
    /// Observation ids of the trajectory, in track order.
    pub track_ids: Vec<i64>,
    /// Cumulative log-likelihood ratio at archiving time.
    pub cumulative_llr: f64,
    /// Number of real observation updates.
    pub n_real_updates: i64,
    /// Last step with a real observation update.
    pub last_real_update_step: i64,
    /// Step at which the trajectory was archived.
    pub archived_at_step: i64,
    /// Absolute-magnitude estimate H, if any.
    pub absolute_magnitude_estimate: Option<f64>,
    /// Number of samples behind that estimate.
    pub absolute_magnitude_sample_count: i32,
    /// MAP Kalman state at archiving time.
    pub kf_state: KfStateFields,
}

/// One row of the `observations` table: a single alert as produced by
/// `test_exp/prep_alert.py`. The ground-truth `traj_id` column present in
/// that parquet is deliberately not read here — it's an evaluation-only
/// artifact with no equivalent on real production data, and this table is
/// meant to also hold production observations.
pub struct ObservationRow {
    /// Primary key: the alert id (ZTF candid / LSST `diaSourceId`).
    pub id: i64,
    /// Night the observation belongs to.
    pub night_id: i64,
    /// Survey object identifier (ZTF `objectId`; synthesized from `id` for LSST).
    pub object_id: String,
    /// Apparent magnitude.
    pub magnitude: f64,
    /// 1-σ magnitude uncertainty.
    pub mag_err: f64,
    /// Numeric filter code.
    pub filter: i16,
    /// MPC observatory code (`I41` for ZTF, `X05` for LSST).
    pub mpc_code_obs: String,
    /// Right ascension, rad.
    pub ra: f64,
    /// 1-σ error on the RA coordinate α, rad.
    pub ra_err: f64,
    /// Declination, rad.
    pub dec: f64,
    /// 1-σ declination error, rad.
    pub dec_err: f64,
    /// Epoch, MJD (TT).
    pub mjd_tt: f64,
}

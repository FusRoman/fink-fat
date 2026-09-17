-- Target schema of the 7 tables `write_sql_tables` bulk-loads, as it would
-- be created on a brand-new database. `UNLOGGED`: these tables are fully
-- rebuilt by every `convert` run (TRUNCATE then COPY, see `mod.rs`) and never
-- modified outside of it, so skipping WAL for them costs nothing — see the
-- crate-level doc comment on `super` for the full rationale. A database
-- created by an older version of this binary needs its existing tables
-- migrated instead of (re)created; that migration lives in
-- `migrate_legacy_columns.sql`, not here.

CREATE UNLOGGED TABLE IF NOT EXISTS observations (
    id BIGINT PRIMARY KEY,
    night_id BIGINT NOT NULL,
    object_id TEXT NOT NULL,
    magnitude DOUBLE PRECISION NOT NULL,
    mag_err DOUBLE PRECISION NOT NULL,
    filter SMALLINT NOT NULL,
    mpc_code_obs TEXT NOT NULL,
    ra DOUBLE PRECISION NOT NULL,
    ra_err DOUBLE PRECISION NOT NULL,
    dec DOUBLE PRECISION NOT NULL,
    dec_err DOUBLE PRECISION NOT NULL,
    mjd_tt DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_object_id ON observations (object_id);

CREATE UNLOGGED TABLE IF NOT EXISTS branches (
    branch_id BIGINT PRIMARY KEY,
    lineage_id BIGINT NOT NULL,
    parent_branch_id BIGINT NOT NULL,
    ancestor_at_scan_horizon BIGINT NOT NULL,
    ancestor_creation_step BIGINT NOT NULL,
    last_real_update_step BIGINT NOT NULL,
    n_real_updates BIGINT NOT NULL,
    cumulative_llr DOUBLE PRECISION NOT NULL,
    lineage_designation TEXT NOT NULL,
    designation TEXT NOT NULL,
    arc_length_days DOUBLE PRECISION NOT NULL,
    n_nights BIGINT NOT NULL,
    median_inter_night_dt_days DOUBLE PRECISION
);

CREATE UNLOGGED TABLE IF NOT EXISTS kf_bank (
    branch_id BIGINT PRIMARY KEY REFERENCES branches(branch_id),
    n_steps BIGINT NOT NULL,
    absolute_magnitude_estimate DOUBLE PRECISION,
    absolute_magnitude_sample_count INTEGER NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS branch_observations (
    branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    position INTEGER NOT NULL,
    obs_id BIGINT NOT NULL REFERENCES observations(id),
    PRIMARY KEY (branch_id, position)
);

CREATE UNLOGGED TABLE IF NOT EXISTS hypotheses (
    hypothesis_id BIGINT PRIMARY KEY,
    branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    local_hyp_id BIGINT NOT NULL,
    log_weight DOUBLE PRECISION NOT NULL,
    recent_log_liks DOUBLE PRECISION[] NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS kf_state (
    hypothesis_id BIGINT PRIMARY KEY REFERENCES hypotheses(hypothesis_id),
    ra DOUBLE PRECISION NOT NULL,
    dec DOUBLE PRECISION NOT NULL,
    ra_dot DOUBLE PRECISION NOT NULL,
    dec_dot DOUBLE PRECISION NOT NULL,
    rho DOUBLE PRECISION NOT NULL,
    rho_dot DOUBLE PRECISION NOT NULL,
    covariance DOUBLE PRECISION[] NOT NULL,
    epoch DOUBLE PRECISION NOT NULL,
    r_obs_x DOUBLE PRECISION NOT NULL,
    r_obs_y DOUBLE PRECISION NOT NULL,
    r_obs_z DOUBLE PRECISION NOT NULL,
    v_obs_x DOUBLE PRECISION NOT NULL,
    v_obs_y DOUBLE PRECISION NOT NULL,
    v_obs_z DOUBLE PRECISION NOT NULL,
    universal_anomaly DOUBLE PRECISION,
    kalman_gain DOUBLE PRECISION[],
    nis_ema DOUBLE PRECISION,
    dynamic_family TEXT NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS archived_trajectories (
    designation TEXT PRIMARY KEY,
    lineage_id BIGINT NOT NULL,
    track_ids BIGINT[] NOT NULL,
    cumulative_llr DOUBLE PRECISION NOT NULL,
    n_real_updates BIGINT NOT NULL,
    last_real_update_step BIGINT NOT NULL,
    archived_at_step BIGINT NOT NULL,
    absolute_magnitude_estimate DOUBLE PRECISION,
    absolute_magnitude_sample_count INTEGER NOT NULL,
    ra DOUBLE PRECISION NOT NULL,
    dec DOUBLE PRECISION NOT NULL,
    ra_dot DOUBLE PRECISION NOT NULL,
    dec_dot DOUBLE PRECISION NOT NULL,
    rho DOUBLE PRECISION NOT NULL,
    rho_dot DOUBLE PRECISION NOT NULL,
    covariance DOUBLE PRECISION[] NOT NULL,
    epoch DOUBLE PRECISION NOT NULL,
    r_obs_x DOUBLE PRECISION NOT NULL,
    r_obs_y DOUBLE PRECISION NOT NULL,
    r_obs_z DOUBLE PRECISION NOT NULL,
    v_obs_x DOUBLE PRECISION NOT NULL,
    v_obs_y DOUBLE PRECISION NOT NULL,
    v_obs_z DOUBLE PRECISION NOT NULL,
    universal_anomaly DOUBLE PRECISION,
    kalman_gain DOUBLE PRECISION[],
    nis_ema DOUBLE PRECISION,
    dynamic_family TEXT NOT NULL,
    semi_major_axis DOUBLE PRECISION NOT NULL,
    eccentricity DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_branch_log_weight
    ON hypotheses (branch_id, log_weight DESC);

-- The explorer's lineage page resolves a lineage by its designation
-- (`WHERE lineage_designation = $1`) from four different queries per
-- page load; without this each one is a full scan of `branches`.
CREATE INDEX IF NOT EXISTS idx_branches_lineage_designation
    ON branches (lineage_designation);

CREATE INDEX IF NOT EXISTS idx_branches_lineage_id
    ON branches (lineage_id);

-- Postgres does not index foreign-key columns on its own, and this one
-- backs both `JOIN observations o ON o.id = bo.obs_id` and the FK
-- checks that TRUNCATE ... CASCADE performs.
CREATE INDEX IF NOT EXISTS idx_branch_observations_obs_id
    ON branch_observations (obs_id);

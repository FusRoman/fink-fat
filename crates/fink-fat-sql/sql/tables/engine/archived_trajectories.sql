-- squawk-ignore-file prefer-bigint-over-int, prefer-bigint-over-smallint
-- INTEGER/SMALLINT are deliberate: `filter`, `position` and
-- `absolute_magnitude_sample_count` are written by the binary COPY as
-- INT2/INT4 (src/copy.rs) and read as i16/i32 by the explorer; widening them
-- would break both, and no value comes near the 32-bit limit.

-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- `archived_trajectories`: closed trajectories, keyed by designation.
-- UNLOGGED: `convert` reloads it in full from the archive log; the
-- streaming pipeline only ever inserts into it, idempotently (`ON CONFLICT
-- (designation) DO NOTHING`), so a lost table is rebuilt by the next
-- `convert`.

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

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE archived_trajectories IS
'Closed trajectories: lineages that stopped being propagated (stale purge) but whose '
'association history is kept as a result, with their final MAP Kalman state. Keyed by '
'designation; only ever inserted into between full reloads.';

COMMENT ON COLUMN archived_trajectories.designation IS
'Designation of the branch the trajectory was archived from. Primary key.';

COMMENT ON COLUMN archived_trajectories.lineage_id IS
'Lineage the trajectory came from.';

COMMENT ON COLUMN archived_trajectories.track_ids IS
'Observation ids of the trajectory (observations.id), in track order.';

COMMENT ON COLUMN archived_trajectories.cumulative_llr IS
'Cumulative log-likelihood ratio at archiving time.';

COMMENT ON COLUMN archived_trajectories.n_real_updates IS
'Number of real observations the trajectory associated.';

COMMENT ON COLUMN archived_trajectories.last_real_update_step IS
'Night step of the last real update: where the arc actually ends.';

COMMENT ON COLUMN archived_trajectories.archived_at_step IS
'Night step at which the lineage was archived.';

COMMENT ON COLUMN archived_trajectories.absolute_magnitude_estimate IS
'Absolute magnitude H estimate at archiving time; NULL if never updated.';

COMMENT ON COLUMN archived_trajectories.absolute_magnitude_sample_count IS
'Number of observations behind absolute_magnitude_estimate.';

COMMENT ON COLUMN archived_trajectories.ra IS
'Right ascension α of the attributable state, rad (equatorial J2000).';

COMMENT ON COLUMN archived_trajectories.dec IS
'Declination δ of the attributable state, rad.';

COMMENT ON COLUMN archived_trajectories.ra_dot IS
'Rate of α, rad/day.';

COMMENT ON COLUMN archived_trajectories.dec_dot IS
'Rate of δ, rad/day.';

COMMENT ON COLUMN archived_trajectories.rho IS
'Topocentric range ρ, AU.';

COMMENT ON COLUMN archived_trajectories.rho_dot IS
'Topocentric range rate, AU/day.';

COMMENT ON COLUMN archived_trajectories.covariance IS
'6×6 state covariance in attributable space (α, δ, α̇, δ̇, ρ, ρ̇), row-major, 36 values.';

COMMENT ON COLUMN archived_trajectories.epoch IS
'Epoch of the state, MJD TT.';

COMMENT ON COLUMN archived_trajectories.r_obs_x IS
'Heliocentric observer position x at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN archived_trajectories.r_obs_y IS
'Heliocentric observer position y at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN archived_trajectories.r_obs_z IS
'Heliocentric observer position z at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN archived_trajectories.v_obs_x IS
'Heliocentric observer velocity x at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN archived_trajectories.v_obs_y IS
'Heliocentric observer velocity y at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN archived_trajectories.v_obs_z IS
'Heliocentric observer velocity z at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN archived_trajectories.universal_anomaly IS
'Last universal anomaly of the two-body propagation (warm start of the next solve); NULL '
'before the first propagation.';

COMMENT ON COLUMN archived_trajectories.kalman_gain IS
'6×2 Kalman gain of the last update, row-major, 12 values; NULL before the first update.';

COMMENT ON COLUMN archived_trajectories.nis_ema IS
'Exponential moving average of the normalized innovation squared, driving the fading-memory '
'inflation; NULL before the first update.';

COMMENT ON COLUMN archived_trajectories.dynamic_family IS
'Dynamical family of the two-body orbit implied by the state (e.g. MBA, NEO, Jupiter trojan, '
'Centaur, TNO; Unknown when the state does not convert to a bound orbit).';

COMMENT ON COLUMN archived_trajectories.semi_major_axis IS
'Semi-major axis of that orbit, AU (0 when Unknown).';

COMMENT ON COLUMN archived_trajectories.eccentricity IS
'Eccentricity of that orbit (0 when Unknown).';

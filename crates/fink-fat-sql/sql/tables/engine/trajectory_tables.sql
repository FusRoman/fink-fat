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

-- The 6 trajectory tables (the live branches of one BranchCollection) and
-- their indexes. UNLOGGED: they are fully regenerated from the engine's
-- snapshot by every `convert` run and every nightly sync, never modified in
-- between, so skipping WAL costs nothing (a Postgres crash empties them;
-- the next load restores them).
--
-- Names are deliberately NOT schema-qualified: `convert` runs this script
-- against `public`, the nightly sync against `fink_fat_staging` (with
-- `search_path = fink_fat_staging, public`, see `src/night_sync.rs`).
-- `REFERENCES observations` then resolves to `public.observations`, the
-- only copy of that table, and every index name stays the same in both
-- schemas (index names are unique per schema).

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
    median_inter_night_dt_days DOUBLE PRECISION,
    -- Matched observations (COUNT(*) over branch_observations ⋈
    -- observations) and nights with >= 2 of them: precomputed so the
    -- explorer never aggregates those two tables to rank or filter branches.
    n_observations BIGINT NOT NULL,
    well_sampled_nights BIGINT NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS kf_bank (
    branch_id BIGINT PRIMARY KEY REFERENCES branches (branch_id),
    n_steps BIGINT NOT NULL,
    absolute_magnitude_estimate DOUBLE PRECISION,
    absolute_magnitude_sample_count INTEGER NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS branch_observations (
    branch_id BIGINT NOT NULL REFERENCES branches (branch_id),
    position INTEGER NOT NULL,
    obs_id BIGINT NOT NULL REFERENCES observations (id),
    PRIMARY KEY (branch_id, position)
);

CREATE UNLOGGED TABLE IF NOT EXISTS hypotheses (
    hypothesis_id BIGINT PRIMARY KEY,
    branch_id BIGINT NOT NULL REFERENCES branches (branch_id),
    local_hyp_id BIGINT NOT NULL,
    log_weight DOUBLE PRECISION NOT NULL,
    recent_log_liks DOUBLE PRECISION[] NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS kf_state (
    hypothesis_id BIGINT PRIMARY KEY REFERENCES hypotheses (hypothesis_id),
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

-- Each branch's best hypothesis (largest log_weight): its family, (a, e)
-- and attributable state, copied from its kf_state row. Denormalized on
-- purpose: the explorer's homepage needs exactly this for every branch, and
-- reading it from hypotheses + kf_state means one index probe and one heap
-- probe per branch into tables tens of GB large. Covariance and Kalman gain
-- are left out: nothing reads them across all branches.
CREATE UNLOGGED TABLE IF NOT EXISTS branch_best_state (
    branch_id BIGINT PRIMARY KEY REFERENCES branches (branch_id),
    hypothesis_id BIGINT NOT NULL,
    dynamic_family TEXT NOT NULL,
    semi_major_axis DOUBLE PRECISION NOT NULL,
    eccentricity DOUBLE PRECISION NOT NULL,
    ra DOUBLE PRECISION NOT NULL,
    dec DOUBLE PRECISION NOT NULL,
    ra_dot DOUBLE PRECISION NOT NULL,
    dec_dot DOUBLE PRECISION NOT NULL,
    rho DOUBLE PRECISION NOT NULL,
    rho_dot DOUBLE PRECISION NOT NULL,
    epoch DOUBLE PRECISION NOT NULL,
    r_obs_x DOUBLE PRECISION NOT NULL,
    r_obs_y DOUBLE PRECISION NOT NULL,
    r_obs_z DOUBLE PRECISION NOT NULL,
    v_obs_x DOUBLE PRECISION NOT NULL,
    v_obs_y DOUBLE PRECISION NOT NULL,
    v_obs_z DOUBLE PRECISION NOT NULL
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

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE branches IS
'One row per live branch of the multi-hypothesis tracker: one candidate association history '
'of one tracked object (its lineage), with its score and precomputed observation statistics. '
'Replaced as a whole by every load (convert, or the streaming pipeline''s nightly swap).';

COMMENT ON COLUMN branches.branch_id IS
'Stable identifier of the branch, never reused (engine id counters are persisted).';

COMMENT ON COLUMN branches.lineage_id IS
'Lineage (tracked object) the branch belongs to: sibling branches share it and compete in '
'top-B pruning.';

COMMENT ON COLUMN branches.parent_branch_id IS
'branch_id of the branch this one was spawned from.';

COMMENT ON COLUMN branches.ancestor_at_scan_horizon IS
'branch_id of the ancestor anchoring the current N-scan pruning window.';

COMMENT ON COLUMN branches.ancestor_creation_step IS
'Night step at which that ancestor was created (decides when the N-scan horizon rolls '
'forward).';

COMMENT ON COLUMN branches.last_real_update_step IS
'Night step of the last real (non-null) observation update.';

COMMENT ON COLUMN branches.n_real_updates IS
'Number of real observations the branch associated.';

COMMENT ON COLUMN branches.cumulative_llr IS
'Cumulative log-likelihood ratio of the track against the clutter background: the branch '
'score.';

COMMENT ON COLUMN branches.lineage_designation IS
'Human-readable, stable identifier of the lineage (shared by its branches); the key used by '
'the explorer pages and by the explorer/submit tables.';

COMMENT ON COLUMN branches.designation IS
'Human-readable identifier unique to this branch.';

COMMENT ON COLUMN branches.arc_length_days IS
'Time span between the first and last observation of the track, days.';

COMMENT ON COLUMN branches.n_nights IS
'Number of distinct nights among the track''s observations.';

COMMENT ON COLUMN branches.median_inter_night_dt_days IS
'Median gap between consecutive observed nights (each night at its earliest observation), '
'days; NULL for a single-night track.';

COMMENT ON COLUMN branches.n_observations IS
'Number of observations in the track (as COUNT over branch_observations ⋈ observations).';

COMMENT ON COLUMN branches.well_sampled_nights IS
'Number of nights with at least two observations of the track.';

COMMENT ON TABLE kf_bank IS
'One row per branch: summary of its Kalman bank (the mixture of (ρ, ρ̇) hypotheses).';

COMMENT ON COLUMN kf_bank.branch_id IS
'The branch (FK to branches).';

COMMENT ON COLUMN kf_bank.n_steps IS
'Number of observation steps the bank processed (drives the hypothesis-cap schedule).';

COMMENT ON COLUMN kf_bank.absolute_magnitude_estimate IS
'Running estimate of the absolute magnitude H (with H-G phase correction); NULL before the '
'first real observation.';

COMMENT ON COLUMN kf_bank.absolute_magnitude_sample_count IS
'Number of observations folded into absolute_magnitude_estimate.';

COMMENT ON TABLE branch_observations IS
'Association history of each branch: one row per observation of its track, in track order.';

COMMENT ON COLUMN branch_observations.branch_id IS
'The branch (FK to branches).';

COMMENT ON COLUMN branch_observations.position IS
'Rank of the observation in the track, from 0.';

COMMENT ON COLUMN branch_observations.obs_id IS
'The observation (FK to observations.id).';

COMMENT ON TABLE hypotheses IS
'One row per hypothesis of each branch''s Kalman bank: one (ρ, ρ̇) mode of the Gaussian '
'mixture.';

COMMENT ON COLUMN hypotheses.hypothesis_id IS
'Identifier of the row, allocated sequentially at load time (not stable across loads).';

COMMENT ON COLUMN hypotheses.branch_id IS
'The branch (FK to branches).';

COMMENT ON COLUMN hypotheses.local_hyp_id IS
'Stable identifier of the hypothesis within its bank.';

COMMENT ON COLUMN hypotheses.log_weight IS
'Log of the hypothesis'' posterior weight in the mixture.';

COMMENT ON COLUMN hypotheses.recent_log_liks IS
'Log-likelihoods of the most recent updates (sliding window), used by the smoothed hypothesis '
'pruning.';

COMMENT ON TABLE kf_state IS
'One row per hypothesis: its Kalman state in attributable coordinates (α, δ, α̇, δ̇, ρ, ρ̇), '
'covariance and the dynamical family of the implied orbit.';

COMMENT ON COLUMN kf_state.hypothesis_id IS
'The hypothesis (FK to hypotheses).';

COMMENT ON COLUMN kf_state.ra IS
'Right ascension α of the attributable state, rad (equatorial J2000).';

COMMENT ON COLUMN kf_state.dec IS
'Declination δ of the attributable state, rad.';

COMMENT ON COLUMN kf_state.ra_dot IS
'Rate of α, rad/day.';

COMMENT ON COLUMN kf_state.dec_dot IS
'Rate of δ, rad/day.';

COMMENT ON COLUMN kf_state.rho IS
'Topocentric range ρ, AU.';

COMMENT ON COLUMN kf_state.rho_dot IS
'Topocentric range rate, AU/day.';

COMMENT ON COLUMN kf_state.covariance IS
'6×6 state covariance in attributable space (α, δ, α̇, δ̇, ρ, ρ̇), row-major, 36 values.';

COMMENT ON COLUMN kf_state.epoch IS
'Epoch of the state, MJD TT.';

COMMENT ON COLUMN kf_state.r_obs_x IS
'Heliocentric observer position x at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN kf_state.r_obs_y IS
'Heliocentric observer position y at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN kf_state.r_obs_z IS
'Heliocentric observer position z at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN kf_state.v_obs_x IS
'Heliocentric observer velocity x at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN kf_state.v_obs_y IS
'Heliocentric observer velocity y at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN kf_state.v_obs_z IS
'Heliocentric observer velocity z at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN kf_state.universal_anomaly IS
'Last universal anomaly of the two-body propagation (warm start of the next solve); NULL '
'before the first propagation.';

COMMENT ON COLUMN kf_state.kalman_gain IS
'6×2 Kalman gain of the last update, row-major, 12 values; NULL before the first update.';

COMMENT ON COLUMN kf_state.nis_ema IS
'Exponential moving average of the normalized innovation squared, driving the fading-memory '
'inflation; NULL before the first update.';

COMMENT ON COLUMN kf_state.dynamic_family IS
'Dynamical family of the two-body orbit implied by the state (e.g. MBA, NEO, Jupiter trojan, '
'Centaur, TNO; Unknown when the state does not convert to a bound orbit).';

COMMENT ON COLUMN kf_state.semi_major_axis IS
'Semi-major axis of that orbit, AU (0 when Unknown).';

COMMENT ON COLUMN kf_state.eccentricity IS
'Eccentricity of that orbit (0 when Unknown).';

COMMENT ON TABLE branch_best_state IS
'One row per branch: copy of the state of its best hypothesis (largest log_weight), '
'denormalized so the explorer homepage reads one row per branch instead of probing hypotheses '
'and kf_state. Covariance and Kalman gain are left out.';

COMMENT ON COLUMN branch_best_state.branch_id IS
'The branch (FK to branches).';

COMMENT ON COLUMN branch_best_state.hypothesis_id IS
'Its best hypothesis (hypotheses.hypothesis_id of the same load).';

COMMENT ON COLUMN branch_best_state.dynamic_family IS
'Dynamical family of the two-body orbit implied by the state (e.g. MBA, NEO, Jupiter trojan, '
'Centaur, TNO; Unknown when the state does not convert to a bound orbit).';

COMMENT ON COLUMN branch_best_state.semi_major_axis IS
'Semi-major axis of that orbit, AU (0 when Unknown).';

COMMENT ON COLUMN branch_best_state.eccentricity IS
'Eccentricity of that orbit (0 when Unknown).';

COMMENT ON COLUMN branch_best_state.ra IS
'Right ascension α of the attributable state, rad (equatorial J2000).';

COMMENT ON COLUMN branch_best_state.dec IS
'Declination δ of the attributable state, rad.';

COMMENT ON COLUMN branch_best_state.ra_dot IS
'Rate of α, rad/day.';

COMMENT ON COLUMN branch_best_state.dec_dot IS
'Rate of δ, rad/day.';

COMMENT ON COLUMN branch_best_state.rho IS
'Topocentric range ρ, AU.';

COMMENT ON COLUMN branch_best_state.rho_dot IS
'Topocentric range rate, AU/day.';

COMMENT ON COLUMN branch_best_state.epoch IS
'Epoch of the state, MJD TT.';

COMMENT ON COLUMN branch_best_state.r_obs_x IS
'Heliocentric observer position x at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN branch_best_state.r_obs_y IS
'Heliocentric observer position y at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN branch_best_state.r_obs_z IS
'Heliocentric observer position z at epoch, AU (ecliptic J2000).';

COMMENT ON COLUMN branch_best_state.v_obs_x IS
'Heliocentric observer velocity x at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN branch_best_state.v_obs_y IS
'Heliocentric observer velocity y at epoch, AU/day (ecliptic J2000).';

COMMENT ON COLUMN branch_best_state.v_obs_z IS
'Heliocentric observer velocity z at epoch, AU/day (ecliptic J2000).';

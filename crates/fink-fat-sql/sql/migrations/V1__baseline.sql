-- squawk-ignore-file prefer-bigint-over-int, prefer-bigint-over-smallint
-- INTEGER/SMALLINT are deliberate: `filter`, `position` and
-- `absolute_magnitude_sample_count` are written by the binary COPY as
-- INT2/INT4 (src/copy.rs) and read as i16/i32 by the explorer, and the
-- orbit-fit counters are written and read as i32; widening them would break
-- every reader, and no value comes near the 32-bit limit.

-- V1 — baseline of the persistent tables, managed by refinery (see
-- `schema::migrate` and the crate README, "Schema migrations").
--
-- FROZEN: refinery stores a checksum of this file in
-- `refinery_schema_history` and refuses to run against a database where the
-- file changed since it was applied (divergent migration). Never edit it,
-- not even a comment: write `V2__<description>.sql` instead.
--
-- Unlike the migrations that will follow it, this one is idempotent on
-- purpose: it must bring BOTH a fresh database and a database created by a
-- pre-refinery `fink-fat convert` (no history table, tables already there)
-- to the baseline schema. Hence the `IF [NOT] EXISTS` throughout, and the
-- upgrades of section 1 run before the `CREATE TABLE IF NOT EXISTS` of
-- section 2 (a no-op against an outdated table, whose `COMMENT ON COLUMN`
-- statements need every column to exist already).
--
-- The 6 trajectory tables are NOT here: they are regenerated from
-- tables/engine/trajectory_tables.sql by every `convert` (DROP + CREATE) and
-- every nightly sync (staging + swap), so a migration of them would be
-- overwritten by the next load.

-- Lock/statement limits (SET LOCAL: they hold until the end of refinery's
-- migration transaction). lock_timeout: fail after 1 min instead of queueing
-- behind a long explorer query, which would also block every query queued
-- behind this one. statement_timeout = 0 (no limit) on purpose: switching a
-- large pre-refinery `observations` to LOGGED rewrites it.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- ==========================================================================
-- 1. Upgrades of tables created by a pre-refinery `fink-fat convert`
--    (no-ops on a fresh database: `ALTER TABLE IF EXISTS`)
-- ==========================================================================

-- `observations` was UNLOGGED until the streaming pipeline started
-- appending to it live; `archived_trajectories` stays UNLOGGED. Both
-- directions are no-ops on a table already in the target state.
ALTER TABLE IF EXISTS observations SET LOGGED;
ALTER TABLE IF EXISTS archived_trajectories SET UNLOGGED;

-- Columns added after the initial rollout. The DEFAULT satisfies NOT NULL on
-- any existing rows; the table is fully rewritten by every `convert`, so the
-- default values never actually get read back out.
ALTER TABLE IF EXISTS archived_trajectories
ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
ALTER TABLE IF EXISTS archived_trajectories
ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS archived_trajectories
ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;

-- ==========================================================================
-- 2. Persistent tables, created if absent and documented (COMMENT ON)
-- ==========================================================================

-- ==========================================================================
-- observations
-- ==========================================================================

-- `observations`: one row per alert. LOGGED (unlike every other table this
-- crate writes): the streaming pipeline appends to it live, night after
-- night, and it is the only store of past nights' alerts — it must survive
-- a Postgres crash. `convert` still TRUNCATEs and reloads it in full.

CREATE TABLE IF NOT EXISTS observations (
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

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE observations IS
'One row per alert (detection) fed to the tracker: its astrometry, photometry and epoch. '
'Written by fink-fat convert (full reload from a parquet) and by the streaming pipeline '
'(live, idempotent inserts). The only store of past alerts in the database, hence WAL-logged.';

COMMENT ON COLUMN observations.id IS
'Alert identifier: ZTF candid or LSST diaSourceId. Primary key; referenced by '
'branch_observations.obs_id.';

COMMENT ON COLUMN observations.night_id IS
'Observing night of the alert. Batch input: relative night index of the input parquet; '
'streaming: MJD of the Rubin dayObs. Do not mix both in one database.';

COMMENT ON COLUMN observations.object_id IS
'Survey object identifier: ZTF objectId; for LSST, the diaSourceId as text (no persistent '
'per-object id for unknown moving objects).';

COMMENT ON COLUMN observations.magnitude IS
'Apparent magnitude (AB for LSST).';

COMMENT ON COLUMN observations.mag_err IS
'1-σ magnitude uncertainty.';

COMMENT ON COLUMN observations.filter IS
'Numeric filter code (LSST: u g r i z y = 0 … 5; ZTF: 1 = g, 2 = r, 3 = i).';

COMMENT ON COLUMN observations.mpc_code_obs IS
'MPC observatory code of the observer (I41 for ZTF, X05 for Rubin).';

COMMENT ON COLUMN observations.ra IS
'Right ascension, rad (ICRF).';

COMMENT ON COLUMN observations.ra_err IS
'1-σ error on the right-ascension coordinate α (on-sky error / cos δ), rad.';

COMMENT ON COLUMN observations.dec IS
'Declination, rad (ICRF).';

COMMENT ON COLUMN observations.dec_err IS
'1-σ declination error, rad.';

COMMENT ON COLUMN observations.mjd_tt IS
'Exposure mid-point, MJD TT.';

-- ==========================================================================
-- archived_trajectories
-- ==========================================================================

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

-- ==========================================================================
-- sync_state
-- ==========================================================================

-- `fink_fat_sync_state`: the last night whose trajectories the streaming
-- pipeline swapped in (see `src/night_sync.rs`). A single row, keyed by a
-- constant. UNLOGGED on purpose, like the trajectory tables it describes: a
-- Postgres crash empties both together, and an empty state tells the
-- pipeline the trajectories must be synced again.

CREATE UNLOGGED TABLE IF NOT EXISTS fink_fat_sync_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_synced_night BIGINT NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE fink_fat_sync_state IS
'Single-row record of the last nightly swap of the trajectory tables by the streaming '
'pipeline. Empty when no sync ever completed, or after a Postgres crash (UNLOGGED, like the '
'trajectory tables), which makes the pipeline sync again.';

COMMENT ON COLUMN fink_fat_sync_state.singleton IS
'Always TRUE: makes the table hold at most one row.';

COMMENT ON COLUMN fink_fat_sync_state.last_synced_night IS
'Last night (MJD of its dayObs) whose trajectories were swapped in.';

COMMENT ON COLUMN fink_fat_sync_state.synced_at IS
'When that swap committed.';

-- ==========================================================================
-- orbit_fits
-- ==========================================================================

-- Outfit n-body least-squares orbit fits, requested on demand from
-- fink-fat-explorer's lineage page. Deliberately NOT truncated by
-- write_bulk_tables and given no FK into branches/observations: those tables
-- are TRUNCATEd (with CASCADE) on every `convert` re-run, and a FK here
-- would wipe this fit history along with them. observation_ids is a plain
-- array rather than a join table for the same reason.

CREATE TABLE IF NOT EXISTS orbit_fits (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    lineage_designation TEXT NOT NULL,
    branch_id BIGINT,
    fitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    observation_ids BIGINT[] NOT NULL,
    n_observations_used INTEGER NOT NULL,
    error_model TEXT NOT NULL,
    fit_method TEXT NOT NULL DEFAULT 'differential_correction',
    fit_params JSONB NOT NULL,
    reference_epoch DOUBLE PRECISION NOT NULL,
    semi_major_axis DOUBLE PRECISION NOT NULL,
    eccentricity_sin_lon DOUBLE PRECISION NOT NULL,
    eccentricity_cos_lon DOUBLE PRECISION NOT NULL,
    tan_half_incl_sin_node DOUBLE PRECISION NOT NULL,
    tan_half_incl_cos_node DOUBLE PRECISION NOT NULL,
    mean_longitude DOUBLE PRECISION NOT NULL,
    covariance DOUBLE PRECISION[] NOT NULL,
    normalised_rms DOUBLE PRECISION NOT NULL,
    total_newton_iterations INTEGER NOT NULL,
    num_measurements INTEGER NOT NULL,
    converged BOOLEAN NOT NULL,
    keplerian JSONB NOT NULL,
    delta_vs_previous_fit JSONB,
    residuals JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orbit_fits_lineage
ON orbit_fits (lineage_designation, fitted_at DESC);

-- Added after the initial rollout, when the bulk fit started fitting every
-- eligible branch independently rather than just each lineage's best
-- branch: without this, rows from different branches of the same lineage
-- were indistinguishable.
ALTER TABLE orbit_fits ADD COLUMN IF NOT EXISTS branch_id BIGINT;

-- The bulk fit runs Gauss IOD with no seed and can fall back to the bare IOD
-- solution when the differential correction diverges
-- (`outfit::differential_orbit_correction::differential_correction`'s own
-- fallback) — this column distinguishes that case from a real
-- least-squares convergence. The single-lineage fit always seeds from the
-- Kalman orbit and only ever inserts a converged DC result, hence the
-- default.
ALTER TABLE orbit_fits
ADD COLUMN IF NOT EXISTS fit_method TEXT NOT NULL DEFAULT 'differential_correction';

CREATE INDEX IF NOT EXISTS idx_orbit_fits_branch
ON orbit_fits (branch_id, fitted_at DESC);

-- One row per failed bulk-fit attempt (Gauss IOD found no valid root).
-- `orbit_fits` only ever holds successes, so without this a branch that
-- failed a fit is indistinguishable from one that was simply never
-- submitted to a bulk fit — the homepage's quality-tier badge needs to tell
-- those two apart. Same rationale as `orbit_fits` for being append-only and
-- FK-less: not TRUNCATEd by `write_bulk_tables`, and a FK into `branches`
-- would wipe this history on every `convert` re-run.
CREATE TABLE IF NOT EXISTS orbit_fit_failures (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    branch_id BIGINT NOT NULL,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_orbit_fit_failures_branch
ON orbit_fit_failures (branch_id, attempted_at DESC);

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE orbit_fits IS
'Orbit fits computed by fink-fat-explorer (single lineage on demand, or bulk): Gauss IOD then '
'n-body differential correction (outfit). Append-only history, one row per fit; owned by the '
'explorer, never truncated by a load, no FK to the trajectory tables.';

COMMENT ON COLUMN orbit_fits.id IS
'Identifier of the fit.';

COMMENT ON COLUMN orbit_fits.lineage_designation IS
'Lineage fitted (branches.lineage_designation, by value).';

COMMENT ON COLUMN orbit_fits.branch_id IS
'Branch fitted (branches.branch_id, by value); NULL for rows written before per-branch fits '
'existed.';

COMMENT ON COLUMN orbit_fits.fitted_at IS
'When the fit was stored.';

COMMENT ON COLUMN orbit_fits.observation_ids IS
'Observations submitted to the fit (observations.id), before outlier rejection.';

COMMENT ON COLUMN orbit_fits.n_observations_used IS
'Observations kept by the fit after outlier rejection.';

COMMENT ON COLUMN orbit_fits.error_model IS
'Astrometric error model used to weight the observations.';

COMMENT ON COLUMN orbit_fits.fit_method IS
'differential_correction for a converged least-squares fit; otherwise the IOD-only fallback '
'(the bulk fit keeps the Gauss solution when the correction diverges).';

COMMENT ON COLUMN orbit_fits.fit_params IS
'Full fit parameters, as JSON (to reload a past configuration).';

COMMENT ON COLUMN orbit_fits.reference_epoch IS
'Epoch of the elements, MJD.';

COMMENT ON COLUMN orbit_fits.semi_major_axis IS
'Equinoctial a: semi-major axis, AU.';

COMMENT ON COLUMN orbit_fits.eccentricity_sin_lon IS
'Equinoctial h = e·sin(Ω + ω).';

COMMENT ON COLUMN orbit_fits.eccentricity_cos_lon IS
'Equinoctial k = e·cos(Ω + ω).';

COMMENT ON COLUMN orbit_fits.tan_half_incl_sin_node IS
'Equinoctial p = tan(i/2)·sin Ω.';

COMMENT ON COLUMN orbit_fits.tan_half_incl_cos_node IS
'Equinoctial q = tan(i/2)·cos Ω.';

COMMENT ON COLUMN orbit_fits.mean_longitude IS
'Equinoctial λ = Ω + ω + M, rad.';

COMMENT ON COLUMN orbit_fits.covariance IS
'6×6 covariance of the equinoctial elements, row-major, 36 values.';

COMMENT ON COLUMN orbit_fits.normalised_rms IS
'Normalized RMS of the astrometric residuals.';

COMMENT ON COLUMN orbit_fits.total_newton_iterations IS
'Iterations of the differential correction.';

COMMENT ON COLUMN orbit_fits.num_measurements IS
'Scalar measurements used (two per observation).';

COMMENT ON COLUMN orbit_fits.converged IS
'Whether the differential correction converged.';

COMMENT ON COLUMN orbit_fits.keplerian IS
'Keplerian elements of the fitted orbit, as JSON.';

COMMENT ON COLUMN orbit_fits.delta_vs_previous_fit IS
'Comparison with the previous fit of the lineage, as JSON; NULL for bulk fits.';

COMMENT ON COLUMN orbit_fits.residuals IS
'Per-observation residuals, as JSON.';

COMMENT ON TABLE orbit_fit_failures IS
'One row per failed bulk orbit fit (Gauss IOD found no orbit), so that the homepage quality '
'tier can tell a failed branch from one never submitted. Owned by fink-fat-explorer, '
'append-only.';

COMMENT ON COLUMN orbit_fit_failures.id IS
'Identifier of the attempt.';

COMMENT ON COLUMN orbit_fit_failures.branch_id IS
'Branch whose fit failed (branches.branch_id, by value).';

COMMENT ON COLUMN orbit_fit_failures.attempted_at IS
'When the attempt failed.';

-- ==========================================================================
-- cnd_queries
-- ==========================================================================

-- MPC Check Near-Duplicates (CND) check results, requested on demand (or in
-- bulk) from fink-fat-explorer's lineage page / bulk CND page. Deliberately
-- NOT truncated by write_bulk_tables and given no FK into
-- branches/observations, for the same reason as orbit_fits/skybot_queries
-- (see the orbit_fits / skybot_bulk sections of this file): those
-- tables are TRUNCATEd (with CASCADE) on every `convert` re-run, and a FK
-- here would wipe this history along with them.
--
-- One row per check attempt, including attempts that found nothing — same
-- rationale as skybot_queries: a lineage with no row here has simply never
-- been checked, distinct from "checked and found nothing".

CREATE TABLE IF NOT EXISTS cnd_queries (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    lineage_designation TEXT NOT NULL,
    branch_id BIGINT,
    queried_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    time_separation_s DOUBLE PRECISION NOT NULL,
    angle_separation_arcsec DOUBLE PRECISION NOT NULL,
    -- The full Vec<CndHit> for this attempt, `[]` when nothing was found —
    -- one entry per observation that had at least one MPC near-duplicate
    -- (see cnd_search::CndHit's doc comment), not one entry per observation
    -- regardless of outcome.
    hits JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cnd_queries_lineage
ON cnd_queries (lineage_designation, queried_at DESC);

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE cnd_queries IS
'MPC Check Near-Duplicates (CND) checks of a lineage''s observations, on demand or in bulk: one '
'row per check attempt, including those that found nothing. Owned by fink-fat-explorer, '
'append-only.';

COMMENT ON COLUMN cnd_queries.id IS
'Identifier of the check.';

COMMENT ON COLUMN cnd_queries.lineage_designation IS
'Lineage checked (branches.lineage_designation, by value).';

COMMENT ON COLUMN cnd_queries.branch_id IS
'Branch checked (branches.branch_id, by value); NULL for older rows.';

COMMENT ON COLUMN cnd_queries.queried_at IS
'When the check ran.';

COMMENT ON COLUMN cnd_queries.time_separation_s IS
'Time tolerance of the near-duplicate search, seconds.';

COMMENT ON COLUMN cnd_queries.angle_separation_arcsec IS
'Angular tolerance of the near-duplicate search, arcsec.';

COMMENT ON COLUMN cnd_queries.hits IS
'Observations with at least one MPC near-duplicate, as a JSON array of CndHit ([] when none).';

-- ==========================================================================
-- skybot_bulk
-- ==========================================================================

-- Durable state for Skybot cross-matching (fink-fat-explorer): the bulk
-- sweep's job/semaphore state, and the single per-observation results table
-- shared by both the bulk sweep and the per-lineage search on the lineage
-- page (there is deliberately no separate per-lineage table — see
-- skybot_obs_status's own comment below). Deliberately NOT truncated by
-- write_bulk_tables and given no FK into branches/observations, same
-- rationale as the other explorer-owned tables (the orbit_fits and
-- cnd_queries sections of this file): those tables are TRUNCATEd (with CASCADE) on every
-- `convert` re-run, and a FK here would wipe this history along with them.
--
-- The bulk job specifically can run for HOURS at real dataset sizes (one
-- HTTP request per observation, not batched like CND), so its state has to
-- survive a server restart, not just live in process memory — skybot_bulk_jobs
-- exists specifically to make that possible.

-- One row per bulk job *attempt* (a fresh "start" click inserts a new row,
-- it never updates an old one back to `running`). The partial unique index
-- is the actual concurrency control: at most one row can have
-- status='running' at any time, enforced by Postgres itself rather than an
-- in-process AtomicBool, so the guarantee survives a restart or a second
-- server instance — a plain `UNIQUE` constraint can't express "unique only
-- for this one status value", hence the partial index on a constant
-- expression.
CREATE TABLE IF NOT EXISTS skybot_bulk_jobs (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    status TEXT NOT NULL, -- running | done | failed | killed | interrupted
    radius_arcsec DOUBLE PRECISION NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    -- Population-wide progress (how much of the whole eligible dataset has
    -- ever been checked), not "since this particular click" — a resumed
    -- job's row starts these at whatever skybot_obs_status already shows,
    -- so the progress bar reflects true completion immediately rather than
    -- resetting to 0% on every resume.
    total_observations BIGINT NOT NULL,
    processed_observations BIGINT NOT NULL DEFAULT 0,
    matched_observations BIGINT NOT NULL DEFAULT 0,
    kill_requested BOOLEAN NOT NULL DEFAULT FALSE,
    error TEXT,
    -- Milestone-only log lines (start, resumed-with-N-already-done, kill
    -- requested, finished) — NOT one line per observation, which at this
    -- job's real scale (tens of thousands of points) would make this column
    -- and the UI's log view unusable.
    logs TEXT[] NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_skybot_bulk_jobs_single_running
ON skybot_bulk_jobs ((1)) WHERE status = 'running';

CREATE INDEX IF NOT EXISTS idx_skybot_bulk_jobs_started_at
ON skybot_bulk_jobs (started_at DESC);

-- One row per observation ever checked by Skybot, upserted on obs_id — the
-- single source of truth for Skybot results, written by BOTH the per-lineage
-- search (crate::skybot_search::run) and the bulk sweep
-- (crate::bulk_skybot::run), never by two separate tables: a lineage's
-- individual search and the population-wide sweep are the same underlying
-- fact ("was this observation checked, and what did Skybot say"), so
-- neither flow should be blind to what the other already found. This
-- per-observation granularity is also what makes the bulk job's "resume,
-- prioritizing never-checked observations first, then oldest-checked first"
-- possible at all. Only written on a *successful* request: a network
-- failure leaves an observation's prior row untouched (or absent), so it
-- naturally gets retried by the priority ordering instead of being wrongly
-- recorded as "checked, no match".
CREATE TABLE IF NOT EXISTS skybot_obs_status (
    obs_id BIGINT PRIMARY KEY,
    lineage_designation TEXT NOT NULL,
    branch_id BIGINT NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL,
    radius_arcsec DOUBLE PRECISION NOT NULL,
    hits JSONB NOT NULL -- Vec<SkybotHit> for this one observation, `[]` if none
);

CREATE INDEX IF NOT EXISTS idx_skybot_obs_status_checked_at
ON skybot_obs_status (checked_at);

CREATE INDEX IF NOT EXISTS idx_skybot_obs_status_lineage
ON skybot_obs_status (lineage_designation);

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE skybot_bulk_jobs IS
'One row per attempt of the explorer''s bulk Skybot sweep (cross-match of every eligible '
'observation with known objects). A partial unique index allows at most one running job.';

COMMENT ON COLUMN skybot_bulk_jobs.id IS
'Identifier of the job.';

COMMENT ON COLUMN skybot_bulk_jobs.status IS
'running, done, failed, killed or interrupted.';

COMMENT ON COLUMN skybot_bulk_jobs.radius_arcsec IS
'Search radius around each observation, arcsec.';

COMMENT ON COLUMN skybot_bulk_jobs.started_at IS
'When the job started.';

COMMENT ON COLUMN skybot_bulk_jobs.finished_at IS
'When it ended; NULL while running.';

COMMENT ON COLUMN skybot_bulk_jobs.total_observations IS
'Eligible observations in the whole dataset (population-wide progress).';

COMMENT ON COLUMN skybot_bulk_jobs.processed_observations IS
'Eligible observations ever checked (resumed jobs start from the previous count).';

COMMENT ON COLUMN skybot_bulk_jobs.matched_observations IS
'Of those, observations matched to a known object.';

COMMENT ON COLUMN skybot_bulk_jobs.kill_requested IS
'Set by the UI to ask the running job to stop.';

COMMENT ON COLUMN skybot_bulk_jobs.error IS
'Error message of a failed job.';

COMMENT ON COLUMN skybot_bulk_jobs.logs IS
'Milestone log lines of the job (start, resume, kill, end).';

COMMENT ON TABLE skybot_obs_status IS
'One row per observation ever checked by Skybot (bulk sweep or per-lineage search), upserted '
'on obs_id: the single source of truth of Skybot results. Only written on a successful '
'request.';

COMMENT ON COLUMN skybot_obs_status.obs_id IS
'Observation checked (observations.id, by value). Primary key.';

COMMENT ON COLUMN skybot_obs_status.lineage_designation IS
'Lineage the observation was checked for.';

COMMENT ON COLUMN skybot_obs_status.branch_id IS
'Branch the observation was checked for.';

COMMENT ON COLUMN skybot_obs_status.checked_at IS
'When it was last checked.';

COMMENT ON COLUMN skybot_obs_status.radius_arcsec IS
'Search radius, arcsec.';

COMMENT ON COLUMN skybot_obs_status.hits IS
'Known objects found, as a JSON array of SkybotHit ([] when none).';

-- ==========================================================================
-- mpc_submissions
-- ==========================================================================

-- Durable record of every MPC submission attempt made by `fink-fat submit`
-- (see `src/submit/`), and the single source of truth for "has this lineage
-- (or these specific observations) already been submitted". Deliberately
-- NOT truncated by write_bulk_tables and given no FK into
-- branches/observations, same rationale as orbit_fits/cnd_queries: those
-- tables are TRUNCATEd (with CASCADE) on every `convert` re-run, and a FK
-- here would wipe this submission history along with them.
-- observation_ids is a plain array rather than a join table for the same
-- reason.

CREATE TABLE IF NOT EXISTS mpc_submissions (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    lineage_designation TEXT NOT NULL,
    branch_id BIGINT,
    trk_sub TEXT NOT NULL,
    -- 'test' (submit_xml_test) or 'production' (submit_xml) — see
    -- fink_fat_ades::mpc_submission::SubmitEndpoint.
    endpoint TEXT NOT NULL,
    -- MPC's ack id, NULL if the submission POST itself failed before MPC
    -- ever acknowledged it.
    submission_id TEXT,
    observation_ids BIGINT[] NOT NULL,
    -- Full ADES payload sent, kept for audit and for the observation-level
    -- duplicate-submission check (see idx_mpc_submissions_observation_ids).
    xml TEXT NOT NULL,
    ack_message TEXT NOT NULL,
    ac2_email TEXT NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 'pending' | 'accepted' | 'rejected' | 'error' — refreshed by
    -- fink-fat-explorer's submission dashboard ("Refresh status" action).
    verdict TEXT NOT NULL DEFAULT 'pending',
    verdict_checked_at TIMESTAMPTZ,
    verdict_detail JSONB
);

-- Added after the initial rollout, once "Refresh status" started also
-- querying WAMO (fink_fat_ades::wamo) for production submissions — kept
-- separate from verdict_detail (which holds the coarse accept/reject
-- payload from the Submission Status API / test-tier status page) since
-- WAMO detail is refreshed independently and evolves over time as MPC
-- finishes processing a submission, unrelated to the coarse verdict itself.
-- NULL for test-tier rows: WAMO never indexes submit_xml_test submissions
-- (confirmed by live probing), so it's simply never queried for those.
ALTER TABLE mpc_submissions ADD COLUMN IF NOT EXISTS wamo_detail JSONB;
ALTER TABLE mpc_submissions ADD COLUMN IF NOT EXISTS wamo_checked_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_mpc_submissions_lineage
ON mpc_submissions (lineage_designation, submitted_at DESC);

-- Powers the explorer's "Submission" page: date-range filter and the
-- `ORDER BY submitted_at` sort, independent of the lineage.
CREATE INDEX IF NOT EXISTS idx_mpc_submissions_submitted_at
ON mpc_submissions (submitted_at DESC);

-- Powers the observation-level "already submitted under a different/renamed
-- lineage" check (fink-fat submit's step 0): `WHERE observation_ids && $1`.
CREATE INDEX IF NOT EXISTS idx_mpc_submissions_observation_ids
ON mpc_submissions USING gin (observation_ids);

-- Documentation of the table(s) above, read by `psql \d+`, pgAdmin and the
-- generated schema docs (crates/fink-fat-sql/docs/schema, see the README).

COMMENT ON TABLE mpc_submissions IS
'Every MPC submission attempt made by fink-fat submit (ADES XML), and the source of truth for '
'what was already submitted. Owned by submit and the explorer''s submission dashboard, '
'append-only.';

COMMENT ON COLUMN mpc_submissions.id IS
'Identifier of the attempt.';

COMMENT ON COLUMN mpc_submissions.lineage_designation IS
'Lineage submitted (branches.lineage_designation, by value).';

COMMENT ON COLUMN mpc_submissions.branch_id IS
'Branch submitted (branches.branch_id, by value).';

COMMENT ON COLUMN mpc_submissions.trk_sub IS
'Tracklet identifier (ADES trkSub) sent to the MPC.';

COMMENT ON COLUMN mpc_submissions.endpoint IS
'test (submit_xml_test) or production (submit_xml).';

COMMENT ON COLUMN mpc_submissions.submission_id IS
'Acknowledgement id returned by the MPC; NULL if the POST failed.';

COMMENT ON COLUMN mpc_submissions.observation_ids IS
'Observations submitted (observations.id, by value); GIN-indexed for the duplicate check.';

COMMENT ON COLUMN mpc_submissions.xml IS
'Full ADES payload sent, for audit.';

COMMENT ON COLUMN mpc_submissions.ack_message IS
'Acknowledgement message returned by the MPC.';

COMMENT ON COLUMN mpc_submissions.ac2_email IS
'Contact email the MPC acknowledgement was sent to.';

COMMENT ON COLUMN mpc_submissions.submitted_at IS
'When the submission was sent.';

COMMENT ON COLUMN mpc_submissions.verdict IS
'pending, accepted, rejected or error; refreshed from the explorer''s submission dashboard.';

COMMENT ON COLUMN mpc_submissions.verdict_checked_at IS
'When the verdict was last refreshed.';

COMMENT ON COLUMN mpc_submissions.verdict_detail IS
'Raw status payload behind the verdict, as JSON.';

COMMENT ON COLUMN mpc_submissions.wamo_detail IS
'Per-observation processing status from MPC WAMO, as JSON (production submissions only).';

COMMENT ON COLUMN mpc_submissions.wamo_checked_at IS
'When wamo_detail was last refreshed.';

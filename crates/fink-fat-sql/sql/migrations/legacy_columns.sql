-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow in `convert` legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- Migrations for a database created by an older version of `fink-fat convert`,
-- where the `tables/**/*.sql` scripts' `CREATE TABLE IF NOT EXISTS` is a no-op
-- against tables that already exist but predate a later schema change.
--
-- Runs BEFORE the tables/** scripts (see `schema::create_tables`): their
-- `COMMENT ON COLUMN` statements need every column to exist already. Hence
-- `ALTER TABLE IF EXISTS` throughout: on a fresh database the tables do not
-- exist yet, nothing is migrated, and the tables/** scripts create them in
-- their current shape. Idempotent, safe to run on every load.

-- `observations` was UNLOGGED until the streaming pipeline started
-- appending to it live (see tables/engine/observations.sql); the trajectory
-- and archive tables stay UNLOGGED. Both directions are no-ops on a table
-- already in the target state.
ALTER TABLE IF EXISTS observations SET LOGGED;
ALTER TABLE IF EXISTS branches SET UNLOGGED;
ALTER TABLE IF EXISTS kf_bank SET UNLOGGED;
ALTER TABLE IF EXISTS branch_observations SET UNLOGGED;
ALTER TABLE IF EXISTS hypotheses SET UNLOGGED;
ALTER TABLE IF EXISTS kf_state SET UNLOGGED;
ALTER TABLE IF EXISTS archived_trajectories SET UNLOGGED;

-- Columns added after the initial rollout: `CREATE TABLE IF NOT EXISTS` in
-- the tables/**/*.sql scripts is a no-op against a pre-existing table, so a DB created
-- before these columns existed needs them backfilled explicitly. The
-- DEFAULT satisfies NOT NULL on any existing rows; these tables are fully
-- rewritten by every load (`convert`'s TRUNCATE + COPY, the streaming
-- pipeline's swap), so the default values never actually get read back out.
ALTER TABLE IF EXISTS kf_state
ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
ALTER TABLE IF EXISTS kf_state
ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS kf_state
ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS archived_trajectories
ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
ALTER TABLE IF EXISTS archived_trajectories
ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS archived_trajectories
ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS branches
ADD COLUMN IF NOT EXISTS arc_length_days DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS branches
ADD COLUMN IF NOT EXISTS n_nights BIGINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS branches
ADD COLUMN IF NOT EXISTS median_inter_night_dt_days DOUBLE PRECISION;
ALTER TABLE IF EXISTS branches
ADD COLUMN IF NOT EXISTS n_observations BIGINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS branches
ADD COLUMN IF NOT EXISTS well_sampled_nights BIGINT NOT NULL DEFAULT 0;

-- Same rationale as above: a DB created before `observations` and the FK on
-- branch_observations.obs_id existed needs the constraint backfilled
-- explicitly. Postgres has no `ADD CONSTRAINT IF NOT EXISTS`, hence the
-- manual pg_constraint check; `to_regclass` (NULL for a missing table,
-- where a `::regclass` cast would fail) skips it on a fresh database.
DO $$
BEGIN
    IF to_regclass('branch_observations') IS NOT NULL
        AND to_regclass('observations') IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'branch_observations_obs_id_fkey'
                AND conrelid = to_regclass('branch_observations')
        ) THEN
        ALTER TABLE branch_observations
            ADD CONSTRAINT branch_observations_obs_id_fkey
            FOREIGN KEY (obs_id) REFERENCES observations(id);
    END IF;
END $$;

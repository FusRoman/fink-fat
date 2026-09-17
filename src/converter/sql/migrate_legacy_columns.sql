-- Migrations for a database created by an older version of this binary,
-- where `create_tables.sql`'s `CREATE TABLE IF NOT EXISTS` is a no-op
-- against tables that already exist but predate a later schema change.
-- Idempotent throughout, safe to run on every `convert`.

-- An UNLOGGED table only got that way starting with this binary version; a
-- table created earlier (still LOGGED) needs this explicit migration to pick
-- up the bulk-load optimisation described in `create_tables.sql`.
ALTER TABLE observations SET UNLOGGED;
ALTER TABLE branches SET UNLOGGED;
ALTER TABLE kf_bank SET UNLOGGED;
ALTER TABLE branch_observations SET UNLOGGED;
ALTER TABLE hypotheses SET UNLOGGED;
ALTER TABLE kf_state SET UNLOGGED;
ALTER TABLE archived_trajectories SET UNLOGGED;

-- Columns added after the initial rollout: `CREATE TABLE IF NOT EXISTS` in
-- create_tables.sql is a no-op against a pre-existing table, so a DB created
-- before these columns existed needs them backfilled explicitly. The
-- DEFAULT satisfies NOT NULL on any existing rows; the table is TRUNCATEd
-- right after this script runs (see `mod.rs::write_sql_tables`), so the
-- default values never actually get read back out.
ALTER TABLE kf_state ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
ALTER TABLE kf_state ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE kf_state ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE archived_trajectories ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
ALTER TABLE archived_trajectories ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE archived_trajectories ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE branches ADD COLUMN IF NOT EXISTS arc_length_days DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE branches ADD COLUMN IF NOT EXISTS n_nights BIGINT NOT NULL DEFAULT 0;
ALTER TABLE branches ADD COLUMN IF NOT EXISTS median_inter_night_dt_days DOUBLE PRECISION;

-- Same rationale as above: a DB created before `observations` and the FK on
-- branch_observations.obs_id existed needs the constraint backfilled
-- explicitly. Postgres has no `ADD CONSTRAINT IF NOT EXISTS`, hence the
-- manual pg_constraint check.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'branch_observations_obs_id_fkey'
    ) THEN
        ALTER TABLE branch_observations
            ADD CONSTRAINT branch_observations_obs_id_fkey
            FOREIGN KEY (obs_id) REFERENCES observations(id);
    END IF;
END $$;

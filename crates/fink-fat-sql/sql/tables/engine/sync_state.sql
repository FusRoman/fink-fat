-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

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

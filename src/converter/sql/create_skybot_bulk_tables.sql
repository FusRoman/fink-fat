-- Durable state for Skybot cross-matching (fink-fat-explorer): the bulk
-- sweep's job/semaphore state, and the single per-observation results table
-- shared by both the bulk sweep and the per-lineage search on the lineage
-- page (there is deliberately no separate per-lineage table — see
-- skybot_obs_status's own comment below). Deliberately NOT truncated by
-- write_sql_tables and given no FK into branches/observations, same
-- rationale as the other explorer-owned tables (create_orbit_fits_tables.sql,
-- create_cnd_tables.sql): those tables are TRUNCATEd (with CASCADE) on every
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
    id BIGSERIAL PRIMARY KEY,
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
    kill_requested BOOLEAN NOT NULL DEFAULT false,
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

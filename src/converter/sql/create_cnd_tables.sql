-- MPC Check Near-Duplicates (CND) check results, requested on demand (or in
-- bulk) from fink-fat-explorer's lineage page / bulk CND page. Deliberately
-- NOT truncated by write_sql_tables and given no FK into
-- branches/observations, for the same reason as orbit_fits/skybot_queries
-- (see create_orbit_fits_tables.sql / create_skybot_tables.sql): those
-- tables are TRUNCATEd (with CASCADE) on every `convert` re-run, and a FK
-- here would wipe this history along with them.
--
-- One row per check attempt, including attempts that found nothing — same
-- rationale as skybot_queries: a lineage with no row here has simply never
-- been checked, distinct from "checked and found nothing".

CREATE TABLE IF NOT EXISTS cnd_queries (
    id BIGSERIAL PRIMARY KEY,
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

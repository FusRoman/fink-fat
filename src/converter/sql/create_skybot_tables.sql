-- Skybot conesearch results, requested on demand from fink-fat-explorer's
-- lineage page. Deliberately NOT truncated by write_sql_tables and given no
-- FK into branches/observations, for the same reason as orbit_fits (see
-- create_orbit_fits_tables.sql): those tables are TRUNCATEd (with CASCADE)
-- on every `convert` re-run, and a FK here would wipe this history along
-- with them.
--
-- One row per search attempt, including attempts that found nothing — a
-- lineage with no matching row has simply never been searched, which is a
-- distinct case from "searched and found nothing" that the lineage page's
-- last-checked display needs to tell apart. Skybot's own catalogue is
-- updated over time, so keeping every attempt (append-only, like
-- orbit_fits) rather than overwriting in place lets a future re-search
-- reveal a match that didn't exist yet at an earlier attempt.

CREATE TABLE IF NOT EXISTS skybot_queries (
    id BIGSERIAL PRIMARY KEY,
    lineage_designation TEXT NOT NULL,
    queried_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    radius_arcsec DOUBLE PRECISION NOT NULL,
    -- The full Vec<SkybotHit> for this attempt, `[]` when nothing was
    -- found. A JSONB blob (rather than a child table) for the same reason
    -- orbit_fits.residuals is JSONB: the hit shape is small, always read
    -- back whole, and never queried by individual field.
    hits JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skybot_queries_lineage
    ON skybot_queries (lineage_designation, queried_at DESC);

-- Durable record of every MPC submission attempt made by `fink-fat submit`
-- (see `src/submit/`), and the single source of truth for "has this lineage
-- (or these specific observations) already been submitted". Deliberately
-- NOT truncated by write_sql_tables and given no FK into
-- branches/observations, same rationale as orbit_fits/cnd_queries: those
-- tables are TRUNCATEd (with CASCADE) on every `convert` re-run, and a FK
-- here would wipe this submission history along with them.
-- observation_ids is a plain array rather than a join table for the same
-- reason.

CREATE TABLE IF NOT EXISTS mpc_submissions (
    id BIGSERIAL PRIMARY KEY,
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

CREATE INDEX IF NOT EXISTS idx_mpc_submissions_lineage
    ON mpc_submissions (lineage_designation, submitted_at DESC);

-- Powers the observation-level "already submitted under a different/renamed
-- lineage" check (fink-fat submit's step 0): `WHERE observation_ids && $1`.
CREATE INDEX IF NOT EXISTS idx_mpc_submissions_observation_ids
    ON mpc_submissions USING GIN (observation_ids);

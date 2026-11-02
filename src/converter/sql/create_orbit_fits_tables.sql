-- Outfit n-body least-squares orbit fits, requested on demand from
-- fink-fat-explorer's lineage page. Deliberately NOT truncated by
-- write_sql_tables and given no FK into branches/observations: those tables
-- are TRUNCATEd (with CASCADE) on every `convert` re-run, and a FK here
-- would wipe this fit history along with them. observation_ids is a plain
-- array rather than a join table for the same reason.

CREATE TABLE IF NOT EXISTS orbit_fits (
    id BIGSERIAL PRIMARY KEY,
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
ALTER TABLE orbit_fits ADD COLUMN IF NOT EXISTS fit_method TEXT NOT NULL DEFAULT 'differential_correction';

CREATE INDEX IF NOT EXISTS idx_orbit_fits_branch
    ON orbit_fits (branch_id, fitted_at DESC);

-- One row per failed bulk-fit attempt (Gauss IOD found no valid root).
-- `orbit_fits` only ever holds successes, so without this a branch that
-- failed a fit is indistinguishable from one that was simply never
-- submitted to a bulk fit — the homepage's quality-tier badge needs to tell
-- those two apart. Same rationale as `orbit_fits` for being append-only and
-- FK-less: not TRUNCATEd by `write_sql_tables`, and a FK into `branches`
-- would wipe this history on every `convert` re-run.
CREATE TABLE IF NOT EXISTS orbit_fit_failures (
    id BIGSERIAL PRIMARY KEY,
    branch_id BIGINT NOT NULL,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_orbit_fit_failures_branch
    ON orbit_fit_failures (branch_id, attempted_at DESC);

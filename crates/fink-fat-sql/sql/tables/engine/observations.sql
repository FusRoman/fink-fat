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

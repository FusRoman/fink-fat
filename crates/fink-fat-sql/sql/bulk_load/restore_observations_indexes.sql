-- squawk-ignore-file adding-foreign-key-constraint, constraint-missing-not-valid
-- squawk-ignore-file require-concurrent-index-creation
-- Offline bulk load, not a live migration: these constraints/indexes are
-- rebuilt on freshly COPY'd tables in the same transaction that dropped
-- them, where validating the FKs once (no NOT VALID) is the point, and
-- CREATE INDEX CONCURRENTLY cannot run inside a transaction.

-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- Undoes bulk_load/drop_observations_indexes.sql.

CREATE INDEX idx_observations_object_id ON observations (object_id);

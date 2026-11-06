-- squawk-ignore-file ban-drop-constraint, require-concurrent-index-deletion
-- Offline bulk load, not a live migration: the constraints and indexes
-- dropped here are rebuilt by the matching bulk_load/restore_*.sql script in the same
-- transaction, and DROP INDEX CONCURRENTLY cannot run inside a transaction.

-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- Secondary index of `observations`, dropped before `convert`'s full COPY
-- and rebuilt by bulk_load/restore_observations_indexes.sql right after. Never run by
-- the streaming pipeline, which appends to a live table.

DROP INDEX IF EXISTS idx_observations_object_id;

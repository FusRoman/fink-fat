-- squawk-ignore-file ban-drop-table
-- Dropping is the point: these are the leftovers of an interrupted sync in
-- the private staging schema, never read by anything.

-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- First step of the nightly sync (see `src/night_sync.rs`): a private
-- schema for the next generation of the trajectory tables, emptied of
-- whatever an interrupted sync may have left. One DROP statement for the 6
-- tables, so their foreign keys never get in the way.

CREATE SCHEMA IF NOT EXISTS fink_fat_staging;

DROP TABLE IF EXISTS
fink_fat_staging.branch_best_state,
fink_fat_staging.kf_state,
fink_fat_staging.hypotheses,
fink_fat_staging.branch_observations,
fink_fat_staging.kf_bank,
fink_fat_staging.branches;

-- squawk-ignore-file ban-drop-table
-- Dropping is the point: the previous generation of the trajectory tables,
-- swapped out of `public` and read by nothing.

-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- Last step of the nightly sync (see `src/night_sync.rs`).

DROP TABLE IF EXISTS
fink_fat_retired.branch_best_state,
fink_fat_retired.kf_state,
fink_fat_retired.hypotheses,
fink_fat_retired.branch_observations,
fink_fat_retired.kf_bank,
fink_fat_retired.branches;

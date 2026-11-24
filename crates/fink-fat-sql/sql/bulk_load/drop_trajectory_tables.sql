-- squawk-ignore-file ban-drop-table
-- Dropping is the point: the trajectory tables are regenerated, never
-- migrated (see migrations/V1__baseline.sql), and every `convert` reloads
-- them in full from the engine's snapshot right after.

-- Lock/statement limits (SET LOCAL: they hold until the end of the
-- enclosing transaction, i.e. also for the statements that run after this
-- script in it). lock_timeout: fail after 1 min instead of queueing behind a
-- long explorer query, which would also block every query queued behind
-- this one. statement_timeout = 0 (no limit) on purpose: the bulk COPYs and
-- index rebuilds that follow legitimately run for minutes.
SET LOCAL lock_timeout = '1min';
SET LOCAL statement_timeout = 0;

-- First step of `convert`'s reload of the trajectory tables (see
-- `schema::recreate_trajectory_tables`): tables/engine/trajectory_tables.sql
-- then creates them in their current shape, whatever version of fink-fat
-- created the dropped ones. One statement for the 6 tables, so their foreign
-- keys between each other need no CASCADE; nothing else references them (the
-- explorer and `submit` tables store branch ids by value, without a FK).
-- Acts on the current `search_path`, i.e. `public`.

DROP TABLE IF EXISTS
branch_best_state,
kf_state,
hypotheses,
branch_observations,
kf_bank,
branches;

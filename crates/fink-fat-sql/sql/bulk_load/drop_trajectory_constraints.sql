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

-- Dropped before the trajectory tables' COPYs and rebuilt by
-- bulk_load/restore_trajectory_constraints.sql right after — see
-- `schema::drop_trajectory_constraints`. Unqualified names, for the same
-- reason as in tables/engine/trajectory_tables.sql.

DROP INDEX IF EXISTS idx_hypotheses_branch_log_weight;
DROP INDEX IF EXISTS idx_branches_lineage_designation;
DROP INDEX IF EXISTS idx_branches_lineage_id;
DROP INDEX IF EXISTS idx_branch_observations_obs_id;

ALTER TABLE kf_bank DROP CONSTRAINT IF EXISTS kf_bank_branch_id_fkey;
ALTER TABLE branch_observations DROP CONSTRAINT IF EXISTS branch_observations_branch_id_fkey;
ALTER TABLE branch_observations DROP CONSTRAINT IF EXISTS branch_observations_obs_id_fkey;
ALTER TABLE hypotheses DROP CONSTRAINT IF EXISTS hypotheses_branch_id_fkey;
ALTER TABLE kf_state DROP CONSTRAINT IF EXISTS kf_state_hypothesis_id_fkey;
ALTER TABLE branch_best_state DROP CONSTRAINT IF EXISTS branch_best_state_branch_id_fkey;

-- Dropped before the bulk-load COPYs and rebuilt by
-- restore_bulk_load_constraints.sql immediately after — see
-- `schema::drop_bulk_load_constraints`'s doc comment for why.

DROP INDEX IF EXISTS idx_observations_object_id;
DROP INDEX IF EXISTS idx_hypotheses_branch_log_weight;
DROP INDEX IF EXISTS idx_branches_lineage_designation;
DROP INDEX IF EXISTS idx_branches_lineage_id;
DROP INDEX IF EXISTS idx_branch_observations_obs_id;

ALTER TABLE kf_bank DROP CONSTRAINT IF EXISTS kf_bank_branch_id_fkey;
ALTER TABLE branch_observations DROP CONSTRAINT IF EXISTS branch_observations_branch_id_fkey;
ALTER TABLE branch_observations DROP CONSTRAINT IF EXISTS branch_observations_obs_id_fkey;
ALTER TABLE hypotheses DROP CONSTRAINT IF EXISTS hypotheses_branch_id_fkey;
ALTER TABLE kf_state DROP CONSTRAINT IF EXISTS kf_state_hypothesis_id_fkey;

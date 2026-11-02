-- Undoes drop_bulk_load_constraints.sql now that the tables hold their
-- final data for this run: one index build and one FK validation scan per
-- constraint, instead of one check per row during the COPYs. Constraint
-- names match Postgres's default naming for an unnamed inline `REFERENCES`
-- (`{table}_{column}_fkey`), the same names
-- drop_bulk_load_constraints.sql and migrate_legacy_columns.sql's
-- legacy-DB backfill both rely on.

ALTER TABLE kf_bank
    ADD CONSTRAINT kf_bank_branch_id_fkey
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id);
ALTER TABLE branch_observations
    ADD CONSTRAINT branch_observations_branch_id_fkey
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id);
ALTER TABLE branch_observations
    ADD CONSTRAINT branch_observations_obs_id_fkey
    FOREIGN KEY (obs_id) REFERENCES observations(id);
ALTER TABLE hypotheses
    ADD CONSTRAINT hypotheses_branch_id_fkey
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id);
ALTER TABLE kf_state
    ADD CONSTRAINT kf_state_hypothesis_id_fkey
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id);

CREATE INDEX idx_observations_object_id ON observations (object_id);
CREATE INDEX idx_hypotheses_branch_log_weight
    ON hypotheses (branch_id, log_weight DESC);
CREATE INDEX idx_branches_lineage_designation ON branches (lineage_designation);
CREATE INDEX idx_branches_lineage_id ON branches (lineage_id);
CREATE INDEX idx_branch_observations_obs_id ON branch_observations (obs_id);

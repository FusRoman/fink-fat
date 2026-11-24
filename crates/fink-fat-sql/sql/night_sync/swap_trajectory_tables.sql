-- squawk-ignore-file ban-set-schema
-- Moving tables between schemas is the swap itself (see below).

-- Short lock wait on purpose: the swap needs ACCESS EXCLUSIVE on the 6
-- public tables, and must not queue behind a long explorer SELECT (every
-- query arriving after it would queue too). The caller retries on timeout.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '1min';

-- Atomic swap of the trajectory tables (see `src/night_sync.rs`): the
-- current generation moves out of `public` into `fink_fat_retired`, the one
-- just built in `fink_fat_staging` moves in. Indexes and constraints move
-- with their tables, keeping their names; foreign keys follow table oids.
-- The explorer resolves `branches` by name, so its next change check sees a
-- new relation (new filenode) and rebuilds its homepage snapshot.

CREATE SCHEMA IF NOT EXISTS fink_fat_retired;

ALTER TABLE public.branch_best_state SET SCHEMA fink_fat_retired;
ALTER TABLE public.kf_state SET SCHEMA fink_fat_retired;
ALTER TABLE public.hypotheses SET SCHEMA fink_fat_retired;
ALTER TABLE public.branch_observations SET SCHEMA fink_fat_retired;
ALTER TABLE public.kf_bank SET SCHEMA fink_fat_retired;
ALTER TABLE public.branches SET SCHEMA fink_fat_retired;

ALTER TABLE fink_fat_staging.branches SET SCHEMA public;
ALTER TABLE fink_fat_staging.kf_bank SET SCHEMA public;
ALTER TABLE fink_fat_staging.branch_observations SET SCHEMA public;
ALTER TABLE fink_fat_staging.hypotheses SET SCHEMA public;
ALTER TABLE fink_fat_staging.kf_state SET SCHEMA public;
ALTER TABLE fink_fat_staging.branch_best_state SET SCHEMA public;

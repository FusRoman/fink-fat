# public.cnd_queries

## Description

MPC Check Near-Duplicates (CND) checks of a lineage's observations, on demand or in bulk: one row per check attempt, including those that found nothing. Owned by fink-fat-explorer, append-only.

## Columns

| Name                    | Type                     | Default | Nullable | Children | Parents                               | Comment                                                                                      |
| ----------------------- | ------------------------ | ------- | -------- | -------- | ------------------------------------- | -------------------------------------------------------------------------------------------- |
| id                      | bigint                   |         | false    |          |                                       | Identifier of the check.                                                                     |
| lineage_designation     | text                     |         | false    |          |                                       | Lineage checked (branches.lineage_designation, by value).                                    |
| branch_id               | bigint                   |         | true     |          | [public.branches](public.branches.md) | Branch checked (branches.branch_id, by value); NULL for older rows.                          |
| queried_at              | timestamp with time zone | now()   | false    |          |                                       | When the check ran.                                                                          |
| time_separation_s       | double precision         |         | false    |          |                                       | Time tolerance of the near-duplicate search, seconds.                                        |
| angle_separation_arcsec | double precision         |         | false    |          |                                       | Angular tolerance of the near-duplicate search, arcsec.                                      |
| hits                    | jsonb                    |         | false    |          |                                       | Observations with at least one MPC near-duplicate, as a JSON array of CndHit ([] when none). |

## Constraints

| Name                                         | Type        | Definition                       |
| -------------------------------------------- | ----------- | -------------------------------- |
| cnd_queries_angle_separation_arcsec_not_null | n           | NOT NULL angle_separation_arcsec |
| cnd_queries_hits_not_null                    | n           | NOT NULL hits                    |
| cnd_queries_id_not_null                      | n           | NOT NULL id                      |
| cnd_queries_lineage_designation_not_null     | n           | NOT NULL lineage_designation     |
| cnd_queries_queried_at_not_null              | n           | NOT NULL queried_at              |
| cnd_queries_time_separation_s_not_null       | n           | NOT NULL time_separation_s       |
| cnd_queries_pkey                             | PRIMARY KEY | PRIMARY KEY (id)                 |

## Indexes

| Name                    | Definition                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| cnd_queries_pkey        | CREATE UNIQUE INDEX cnd_queries_pkey ON public.cnd_queries USING btree (id)                                   |
| idx_cnd_queries_lineage | CREATE INDEX idx_cnd_queries_lineage ON public.cnd_queries USING btree (lineage_designation, queried_at DESC) |

## Relations

```mermaid
erDiagram

"public.cnd_queries" }o--o| "public.branches" : "by value, no FK"

"public.cnd_queries" {
  bigint id "Identifier of the check."
  text lineage_designation "Lineage checked (branches.lineage_designation, by value)."
  bigint branch_id "Branch checked (branches.branch_id, by value); NULL for older rows."
  timestamp_with_time_zone queried_at "When the check ran."
  double_precision time_separation_s "Time tolerance of the near-duplicate search, seconds."
  double_precision angle_separation_arcsec "Angular tolerance of the near-duplicate search, arcsec."
  jsonb hits "Observations with at least one MPC near-duplicate, as a JSON array of CndHit ([] when none)."
}
"public.branches" {
  bigint branch_id "Stable identifier of the branch, never reused (engine id counters are persisted)."
  bigint lineage_id "Lineage (tracked object) the branch belongs to: sibling branches share it and compete in top-B pruning."
  bigint parent_branch_id "branch_id of the branch this one was spawned from."
  bigint ancestor_at_scan_horizon "branch_id of the ancestor anchoring the current N-scan pruning window."
  bigint ancestor_creation_step "Night step at which that ancestor was created (decides when the N-scan horizon rolls forward)."
  bigint last_real_update_step "Night step of the last real (non-null) observation update."
  bigint n_real_updates "Number of real observations the branch associated."
  double_precision cumulative_llr "Cumulative log-likelihood ratio of the track against the clutter background: the branch score."
  text lineage_designation "Human-readable, stable identifier of the lineage (shared by its branches); the key used by the explorer pages and by the explorer/submit tables."
  text designation "Human-readable identifier unique to this branch."
  double_precision arc_length_days "Time span between the first and last observation of the track, days."
  bigint n_nights "Number of distinct nights among the track's observations."
  double_precision median_inter_night_dt_days "Median gap between consecutive observed nights (each night at its earliest observation), days; NULL for a single-night track."
  bigint n_observations "Number of observations in the track (as COUNT over branch_observations ⋈ observations)."
  bigint well_sampled_nights "Number of nights with at least two observations of the track."
}
```

---

> Generated by [tbls](https://github.com/k1LoW/tbls)

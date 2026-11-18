# public.branch_observations

## Description

Association history of each branch: one row per observation of its track, in track order.

## Columns

| Name      | Type    | Default | Nullable | Children | Parents                                       | Comment                                       |
| --------- | ------- | ------- | -------- | -------- | --------------------------------------------- | --------------------------------------------- |
| branch_id | bigint  |         | false    |          | [public.branches](public.branches.md)         | The branch (FK to branches).                  |
| position  | integer |         | false    |          |                                               | Rank of the observation in the track, from 0. |
| obs_id    | bigint  |         | false    |          | [public.observations](public.observations.md) | The observation (FK to observations.id).      |

## Constraints

| Name                                   | Type        | Definition                                             |
| -------------------------------------- | ----------- | ------------------------------------------------------ |
| branch_observations_branch_id_not_null | n           | NOT NULL branch_id                                     |
| branch_observations_obs_id_not_null    | n           | NOT NULL obs_id                                        |
| branch_observations_position_not_null  | n           | NOT NULL "position"                                    |
| branch_observations_obs_id_fkey        | FOREIGN KEY | FOREIGN KEY (obs_id) REFERENCES observations(id)       |
| branch_observations_branch_id_fkey     | FOREIGN KEY | FOREIGN KEY (branch_id) REFERENCES branches(branch_id) |
| branch_observations_pkey               | PRIMARY KEY | PRIMARY KEY (branch_id, "position")                    |

## Indexes

| Name                           | Definition                                                                                                     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| branch_observations_pkey       | CREATE UNIQUE INDEX branch_observations_pkey ON public.branch_observations USING btree (branch_id, "position") |
| idx_branch_observations_obs_id | CREATE INDEX idx_branch_observations_obs_id ON public.branch_observations USING btree (obs_id)                 |

## Relations

```mermaid
erDiagram

"public.branch_observations" }o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"
"public.branch_observations" }o--|| "public.observations" : "FOREIGN KEY (obs_id) REFERENCES observations(id)"

"public.branch_observations" {
  bigint branch_id FK "The branch (FK to branches)."
  integer position "Rank of the observation in the track, from 0."
  bigint obs_id FK "The observation (FK to observations.id)."
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
"public.observations" {
  bigint id "Alert identifier: ZTF candid or LSST diaSourceId. Primary key; referenced by branch_observations.obs_id."
  bigint night_id "Observing night of the alert. Batch input: relative night index of the input parquet; streaming: MJD of the Rubin dayObs. Do not mix both in one database."
  text object_id "Survey object identifier: ZTF objectId; for LSST, the diaSourceId as text (no persistent per-object id for unknown moving objects)."
  double_precision magnitude "Apparent magnitude (AB for LSST)."
  double_precision mag_err "1-σ magnitude uncertainty."
  smallint filter "Numeric filter code (LSST: u g r i z y = 0 … 5; ZTF: 1 = g, 2 = r, 3 = i)."
  text mpc_code_obs "MPC observatory code of the observer (I41 for ZTF, X05 for Rubin)."
  double_precision ra "Right ascension, rad (ICRF)."
  double_precision ra_err "1-σ error on the right-ascension coordinate α (on-sky error / cos δ), rad."
  double_precision dec "Declination, rad (ICRF)."
  double_precision dec_err "1-σ declination error, rad."
  double_precision mjd_tt "Exposure mid-point, MJD TT."
}
```

---

> Generated by [tbls](https://github.com/k1LoW/tbls)

# public.skybot_obs_status

## Description

One row per observation ever checked by Skybot (bulk sweep or per-lineage search), upserted on obs_id: the single source of truth of Skybot results. Only written on a successful request.

## Columns

| Name                | Type                     | Default | Nullable | Children | Parents                                       | Comment                                                           |
| ------------------- | ------------------------ | ------- | -------- | -------- | --------------------------------------------- | ----------------------------------------------------------------- |
| obs_id              | bigint                   |         | false    |          | [public.observations](public.observations.md) | Observation checked (observations.id, by value). Primary key.     |
| lineage_designation | text                     |         | false    |          |                                               | Lineage the observation was checked for.                          |
| branch_id           | bigint                   |         | false    |          | [public.branches](public.branches.md)         | Branch the observation was checked for.                           |
| checked_at          | timestamp with time zone |         | false    |          |                                               | When it was last checked.                                         |
| radius_arcsec       | double precision         |         | false    |          |                                               | Search radius, arcsec.                                            |
| hits                | jsonb                    |         | false    |          |                                               | Known objects found, as a JSON array of SkybotHit ([] when none). |

## Constraints

| Name                                           | Type        | Definition                   |
| ---------------------------------------------- | ----------- | ---------------------------- |
| skybot_obs_status_branch_id_not_null           | n           | NOT NULL branch_id           |
| skybot_obs_status_checked_at_not_null          | n           | NOT NULL checked_at          |
| skybot_obs_status_hits_not_null                | n           | NOT NULL hits                |
| skybot_obs_status_lineage_designation_not_null | n           | NOT NULL lineage_designation |
| skybot_obs_status_obs_id_not_null              | n           | NOT NULL obs_id              |
| skybot_obs_status_radius_arcsec_not_null       | n           | NOT NULL radius_arcsec       |
| skybot_obs_status_pkey                         | PRIMARY KEY | PRIMARY KEY (obs_id)         |

## Indexes

| Name                             | Definition                                                                                               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| skybot_obs_status_pkey           | CREATE UNIQUE INDEX skybot_obs_status_pkey ON public.skybot_obs_status USING btree (obs_id)              |
| idx_skybot_obs_status_checked_at | CREATE INDEX idx_skybot_obs_status_checked_at ON public.skybot_obs_status USING btree (checked_at)       |
| idx_skybot_obs_status_lineage    | CREATE INDEX idx_skybot_obs_status_lineage ON public.skybot_obs_status USING btree (lineage_designation) |

## Relations

```mermaid
erDiagram

"public.skybot_obs_status" |o--|| "public.observations" : "by value, no FK"
"public.skybot_obs_status" }o--|| "public.branches" : "by value, no FK"

"public.skybot_obs_status" {
  bigint obs_id "Observation checked (observations.id, by value). Primary key."
  text lineage_designation "Lineage the observation was checked for."
  bigint branch_id "Branch the observation was checked for."
  timestamp_with_time_zone checked_at "When it was last checked."
  double_precision radius_arcsec "Search radius, arcsec."
  jsonb hits "Known objects found, as a JSON array of SkybotHit ([] when none)."
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

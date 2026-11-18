# public.kf_bank

## Description

One row per branch: summary of its Kalman bank (the mixture of (ρ, ρ̇) hypotheses).

## Columns

| Name                            | Type             | Default | Nullable | Children | Parents                               | Comment                                                                                                           |
| ------------------------------- | ---------------- | ------- | -------- | -------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| branch_id                       | bigint           |         | false    |          | [public.branches](public.branches.md) | The branch (FK to branches).                                                                                      |
| n_steps                         | bigint           |         | false    |          |                                       | Number of observation steps the bank processed (drives the hypothesis-cap schedule).                              |
| absolute_magnitude_estimate     | double precision |         | true     |          |                                       | Running estimate of the absolute magnitude H (with H-G phase correction); NULL before the first real observation. |
| absolute_magnitude_sample_count | integer          |         | false    |          |                                       | Number of observations folded into absolute_magnitude_estimate.                                                   |

## Constraints

| Name                                             | Type        | Definition                                             |
| ------------------------------------------------ | ----------- | ------------------------------------------------------ |
| kf_bank_absolute_magnitude_sample_count_not_null | n           | NOT NULL absolute_magnitude_sample_count               |
| kf_bank_branch_id_not_null                       | n           | NOT NULL branch_id                                     |
| kf_bank_n_steps_not_null                         | n           | NOT NULL n_steps                                       |
| kf_bank_branch_id_fkey                           | FOREIGN KEY | FOREIGN KEY (branch_id) REFERENCES branches(branch_id) |
| kf_bank_pkey                                     | PRIMARY KEY | PRIMARY KEY (branch_id)                                |

## Indexes

| Name         | Definition                                                                 |
| ------------ | -------------------------------------------------------------------------- |
| kf_bank_pkey | CREATE UNIQUE INDEX kf_bank_pkey ON public.kf_bank USING btree (branch_id) |

## Relations

```mermaid
erDiagram

"public.kf_bank" |o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"

"public.kf_bank" {
  bigint branch_id FK "The branch (FK to branches)."
  bigint n_steps "Number of observation steps the bank processed (drives the hypothesis-cap schedule)."
  double_precision absolute_magnitude_estimate "Running estimate of the absolute magnitude H (with H-G phase correction); NULL before the first real observation."
  integer absolute_magnitude_sample_count "Number of observations folded into absolute_magnitude_estimate."
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

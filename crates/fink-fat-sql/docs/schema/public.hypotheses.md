# public.hypotheses

## Description

One row per hypothesis of each branch's Kalman bank: one (ρ, ρ̇) mode of the Gaussian mixture.

## Columns

| Name            | Type               | Default | Nullable | Children                              | Parents                               | Comment                                                                                               |
| --------------- | ------------------ | ------- | -------- | ------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| hypothesis_id   | bigint             |         | false    | [public.kf_state](public.kf_state.md) |                                       | Identifier of the row, allocated sequentially at load time (not stable across loads).                 |
| branch_id       | bigint             |         | false    |                                       | [public.branches](public.branches.md) | The branch (FK to branches).                                                                          |
| local_hyp_id    | bigint             |         | false    |                                       |                                       | Stable identifier of the hypothesis within its bank.                                                  |
| log_weight      | double precision   |         | false    |                                       |                                       | Log of the hypothesis' posterior weight in the mixture.                                               |
| recent_log_liks | double precision[] |         | false    |                                       |                                       | Log-likelihoods of the most recent updates (sliding window), used by the smoothed hypothesis pruning. |

## Constraints

| Name                                | Type        | Definition                                             |
| ----------------------------------- | ----------- | ------------------------------------------------------ |
| hypotheses_branch_id_not_null       | n           | NOT NULL branch_id                                     |
| hypotheses_hypothesis_id_not_null   | n           | NOT NULL hypothesis_id                                 |
| hypotheses_local_hyp_id_not_null    | n           | NOT NULL local_hyp_id                                  |
| hypotheses_log_weight_not_null      | n           | NOT NULL log_weight                                    |
| hypotheses_recent_log_liks_not_null | n           | NOT NULL recent_log_liks                               |
| hypotheses_branch_id_fkey           | FOREIGN KEY | FOREIGN KEY (branch_id) REFERENCES branches(branch_id) |
| hypotheses_pkey                     | PRIMARY KEY | PRIMARY KEY (hypothesis_id)                            |

## Indexes

| Name                             | Definition                                                                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| hypotheses_pkey                  | CREATE UNIQUE INDEX hypotheses_pkey ON public.hypotheses USING btree (hypothesis_id)                        |
| idx_hypotheses_branch_log_weight | CREATE INDEX idx_hypotheses_branch_log_weight ON public.hypotheses USING btree (branch_id, log_weight DESC) |

## Relations

```mermaid
erDiagram

"public.kf_state" |o--|| "public.hypotheses" : "FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)"
"public.hypotheses" }o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"

"public.hypotheses" {
  bigint hypothesis_id "Identifier of the row, allocated sequentially at load time (not stable across loads)."
  bigint branch_id FK "The branch (FK to branches)."
  bigint local_hyp_id "Stable identifier of the hypothesis within its bank."
  double_precision log_weight "Log of the hypothesis' posterior weight in the mixture."
  double_precision__ recent_log_liks "Log-likelihoods of the most recent updates (sliding window), used by the smoothed hypothesis pruning."
}
"public.kf_state" {
  bigint hypothesis_id FK "The hypothesis (FK to hypotheses)."
  double_precision ra "Right ascension α of the attributable state, rad (equatorial J2000)."
  double_precision dec "Declination δ of the attributable state, rad."
  double_precision ra_dot "Rate of α, rad/day."
  double_precision dec_dot "Rate of δ, rad/day."
  double_precision rho "Topocentric range ρ, AU."
  double_precision rho_dot "Topocentric range rate, AU/day."
  double_precision__ covariance "6×6 state covariance in attributable space (α, δ, α̇, δ̇, ρ, ρ̇), row-major, 36 values."
  double_precision epoch "Epoch of the state, MJD TT."
  double_precision r_obs_x "Heliocentric observer position x at epoch, AU (ecliptic J2000)."
  double_precision r_obs_y "Heliocentric observer position y at epoch, AU (ecliptic J2000)."
  double_precision r_obs_z "Heliocentric observer position z at epoch, AU (ecliptic J2000)."
  double_precision v_obs_x "Heliocentric observer velocity x at epoch, AU/day (ecliptic J2000)."
  double_precision v_obs_y "Heliocentric observer velocity y at epoch, AU/day (ecliptic J2000)."
  double_precision v_obs_z "Heliocentric observer velocity z at epoch, AU/day (ecliptic J2000)."
  double_precision universal_anomaly "Last universal anomaly of the two-body propagation (warm start of the next solve); NULL before the first propagation."
  double_precision__ kalman_gain "6×2 Kalman gain of the last update, row-major, 12 values; NULL before the first update."
  double_precision nis_ema "Exponential moving average of the normalized innovation squared, driving the fading-memory inflation; NULL before the first update."
  text dynamic_family "Dynamical family of the two-body orbit implied by the state (e.g. MBA, NEO, Jupiter trojan, Centaur, TNO; Unknown when the state does not convert to a bound orbit)."
  double_precision semi_major_axis "Semi-major axis of that orbit, AU (0 when Unknown)."
  double_precision eccentricity "Eccentricity of that orbit (0 when Unknown)."
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

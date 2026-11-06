# fink-fat

## Description

Tables written by the fink-fat engine (convert, streaming), fink-fat-explorer and fink-fat submit.

## Tables

| Name                                                            | Columns | Comment                                                                                                                                                                                                                                                                       | Type       |
| --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| [public.observations](public.observations.md)                   | 12      | One row per alert (detection) fed to the tracker: its astrometry, photometry and epoch. Written by fink-fat convert (full reload from a parquet) and by the streaming pipeline (live, idempotent inserts). The only store of past alerts in the database, hence WAL-logged.   | BASE TABLE |
| [public.branches](public.branches.md)                           | 15      | One row per live branch of the multi-hypothesis tracker: one candidate association history of one tracked object (its lineage), with its score and precomputed observation statistics. Replaced as a whole by every load (convert, or the streaming pipeline's nightly swap). | BASE TABLE |
| [public.kf_bank](public.kf_bank.md)                             | 4       | One row per branch: summary of its Kalman bank (the mixture of (ρ, ρ̇) hypotheses).                                                                                                                                                                                            | BASE TABLE |
| [public.branch_observations](public.branch_observations.md)     | 3       | Association history of each branch: one row per observation of its track, in track order.                                                                                                                                                                                     | BASE TABLE |
| [public.hypotheses](public.hypotheses.md)                       | 5       | One row per hypothesis of each branch's Kalman bank: one (ρ, ρ̇) mode of the Gaussian mixture.                                                                                                                                                                                 | BASE TABLE |
| [public.kf_state](public.kf_state.md)                           | 21      | One row per hypothesis: its Kalman state in attributable coordinates (α, δ, α̇, δ̇, ρ, ρ̇), covariance and the dynamical family of the implied orbit.                                                                                                                            | BASE TABLE |
| [public.branch_best_state](public.branch_best_state.md)         | 18      | One row per branch: copy of the state of its best hypothesis (largest log_weight), denormalized so the explorer homepage reads one row per branch instead of probing hypotheses and kf_state. Covariance and Kalman gain are left out.                                        | BASE TABLE |
| [public.archived_trajectories](public.archived_trajectories.md) | 29      | Closed trajectories: lineages that stopped being propagated (stale purge) but whose association history is kept as a result, with their final MAP Kalman state. Keyed by designation; only ever inserted into between full reloads.                                           | BASE TABLE |
| [public.fink_fat_sync_state](public.fink_fat_sync_state.md)     | 3       | Single-row record of the last nightly swap of the trajectory tables by the streaming pipeline. Empty when no sync ever completed, or after a Postgres crash (UNLOGGED, like the trajectory tables), which makes the pipeline sync again.                                      | BASE TABLE |
| [public.orbit_fits](public.orbit_fits.md)                       | 24      | Orbit fits computed by fink-fat-explorer (single lineage on demand, or bulk): Gauss IOD then n-body differential correction (outfit). Append-only history, one row per fit; owned by the explorer, never truncated by a load, no FK to the trajectory tables.                 | BASE TABLE |
| [public.orbit_fit_failures](public.orbit_fit_failures.md)       | 3       | One row per failed bulk orbit fit (Gauss IOD found no orbit), so that the homepage quality tier can tell a failed branch from one never submitted. Owned by fink-fat-explorer, append-only.                                                                                   | BASE TABLE |
| [public.cnd_queries](public.cnd_queries.md)                     | 7       | MPC Check Near-Duplicates (CND) checks of a lineage's observations, on demand or in bulk: one row per check attempt, including those that found nothing. Owned by fink-fat-explorer, append-only.                                                                             | BASE TABLE |
| [public.skybot_bulk_jobs](public.skybot_bulk_jobs.md)           | 11      | One row per attempt of the explorer's bulk Skybot sweep (cross-match of every eligible observation with known objects). A partial unique index allows at most one running job.                                                                                                | BASE TABLE |
| [public.skybot_obs_status](public.skybot_obs_status.md)         | 6       | One row per observation ever checked by Skybot (bulk sweep or per-lineage search), upserted on obs_id: the single source of truth of Skybot results. Only written on a successful request.                                                                                    | BASE TABLE |
| [public.mpc_submissions](public.mpc_submissions.md)             | 16      | Every MPC submission attempt made by fink-fat submit (ADES XML), and the source of truth for what was already submitted. Owned by submit and the explorer's submission dashboard, append-only.                                                                                | BASE TABLE |

## Relations

```mermaid
erDiagram

"public.kf_bank" |o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"
"public.branch_observations" }o--|| "public.observations" : "FOREIGN KEY (obs_id) REFERENCES observations(id)"
"public.branch_observations" }o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"
"public.hypotheses" }o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"
"public.kf_state" |o--|| "public.hypotheses" : "FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)"
"public.branch_best_state" |o--|| "public.branches" : "FOREIGN KEY (branch_id) REFERENCES branches(branch_id)"
"public.orbit_fits" }o--o| "public.branches" : "by value, no FK"
"public.orbit_fit_failures" }o--|| "public.branches" : "by value, no FK"
"public.cnd_queries" }o--o| "public.branches" : "by value, no FK"
"public.skybot_obs_status" }o--|| "public.branches" : "by value, no FK"
"public.skybot_obs_status" |o--|| "public.observations" : "by value, no FK"
"public.mpc_submissions" }o--o| "public.branches" : "by value, no FK"

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
"public.kf_bank" {
  bigint branch_id FK "The branch (FK to branches)."
  bigint n_steps "Number of observation steps the bank processed (drives the hypothesis-cap schedule)."
  double_precision absolute_magnitude_estimate "Running estimate of the absolute magnitude H (with H-G phase correction); NULL before the first real observation."
  integer absolute_magnitude_sample_count "Number of observations folded into absolute_magnitude_estimate."
}
"public.branch_observations" {
  bigint branch_id FK "The branch (FK to branches)."
  integer position "Rank of the observation in the track, from 0."
  bigint obs_id FK "The observation (FK to observations.id)."
}
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
"public.branch_best_state" {
  bigint branch_id FK "The branch (FK to branches)."
  bigint hypothesis_id "Its best hypothesis (hypotheses.hypothesis_id of the same load)."
  text dynamic_family "Dynamical family of the two-body orbit implied by the state (e.g. MBA, NEO, Jupiter trojan, Centaur, TNO; Unknown when the state does not convert to a bound orbit)."
  double_precision semi_major_axis "Semi-major axis of that orbit, AU (0 when Unknown)."
  double_precision eccentricity "Eccentricity of that orbit (0 when Unknown)."
  double_precision ra "Right ascension α of the attributable state, rad (equatorial J2000)."
  double_precision dec "Declination δ of the attributable state, rad."
  double_precision ra_dot "Rate of α, rad/day."
  double_precision dec_dot "Rate of δ, rad/day."
  double_precision rho "Topocentric range ρ, AU."
  double_precision rho_dot "Topocentric range rate, AU/day."
  double_precision epoch "Epoch of the state, MJD TT."
  double_precision r_obs_x "Heliocentric observer position x at epoch, AU (ecliptic J2000)."
  double_precision r_obs_y "Heliocentric observer position y at epoch, AU (ecliptic J2000)."
  double_precision r_obs_z "Heliocentric observer position z at epoch, AU (ecliptic J2000)."
  double_precision v_obs_x "Heliocentric observer velocity x at epoch, AU/day (ecliptic J2000)."
  double_precision v_obs_y "Heliocentric observer velocity y at epoch, AU/day (ecliptic J2000)."
  double_precision v_obs_z "Heliocentric observer velocity z at epoch, AU/day (ecliptic J2000)."
}
"public.archived_trajectories" {
  text designation "Designation of the branch the trajectory was archived from. Primary key."
  bigint lineage_id "Lineage the trajectory came from."
  bigint__ track_ids "Observation ids of the trajectory (observations.id), in track order."
  double_precision cumulative_llr "Cumulative log-likelihood ratio at archiving time."
  bigint n_real_updates "Number of real observations the trajectory associated."
  bigint last_real_update_step "Night step of the last real update: where the arc actually ends."
  bigint archived_at_step "Night step at which the lineage was archived."
  double_precision absolute_magnitude_estimate "Absolute magnitude H estimate at archiving time; NULL if never updated."
  integer absolute_magnitude_sample_count "Number of observations behind absolute_magnitude_estimate."
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
"public.fink_fat_sync_state" {
  boolean singleton "Always TRUE: makes the table hold at most one row."
  bigint last_synced_night "Last night (MJD of its dayObs) whose trajectories were swapped in."
  timestamp_with_time_zone synced_at "When that swap committed."
}
"public.orbit_fits" {
  bigint id "Identifier of the fit."
  text lineage_designation "Lineage fitted (branches.lineage_designation, by value)."
  bigint branch_id "Branch fitted (branches.branch_id, by value); NULL for rows written before per-branch fits existed."
  timestamp_with_time_zone fitted_at "When the fit was stored."
  bigint__ observation_ids "Observations submitted to the fit (observations.id), before outlier rejection."
  integer n_observations_used "Observations kept by the fit after outlier rejection."
  text error_model "Astrometric error model used to weight the observations."
  text fit_method "differential_correction for a converged least-squares fit; otherwise the IOD-only fallback (the bulk fit keeps the Gauss solution when the correction diverges)."
  jsonb fit_params "Full fit parameters, as JSON (to reload a past configuration)."
  double_precision reference_epoch "Epoch of the elements, MJD."
  double_precision semi_major_axis "Equinoctial a: semi-major axis, AU."
  double_precision eccentricity_sin_lon "Equinoctial h = e·sin(Ω + ω)."
  double_precision eccentricity_cos_lon "Equinoctial k = e·cos(Ω + ω)."
  double_precision tan_half_incl_sin_node "Equinoctial p = tan(i/2)·sin Ω."
  double_precision tan_half_incl_cos_node "Equinoctial q = tan(i/2)·cos Ω."
  double_precision mean_longitude "Equinoctial λ = Ω + ω + M, rad."
  double_precision__ covariance "6×6 covariance of the equinoctial elements, row-major, 36 values."
  double_precision normalised_rms "Normalized RMS of the astrometric residuals."
  integer total_newton_iterations "Iterations of the differential correction."
  integer num_measurements "Scalar measurements used (two per observation)."
  boolean converged "Whether the differential correction converged."
  jsonb keplerian "Keplerian elements of the fitted orbit, as JSON."
  jsonb delta_vs_previous_fit "Comparison with the previous fit of the lineage, as JSON; NULL for bulk fits."
  jsonb residuals "Per-observation residuals, as JSON."
}
"public.orbit_fit_failures" {
  bigint id "Identifier of the attempt."
  bigint branch_id "Branch whose fit failed (branches.branch_id, by value)."
  timestamp_with_time_zone attempted_at "When the attempt failed."
}
"public.cnd_queries" {
  bigint id "Identifier of the check."
  text lineage_designation "Lineage checked (branches.lineage_designation, by value)."
  bigint branch_id "Branch checked (branches.branch_id, by value); NULL for older rows."
  timestamp_with_time_zone queried_at "When the check ran."
  double_precision time_separation_s "Time tolerance of the near-duplicate search, seconds."
  double_precision angle_separation_arcsec "Angular tolerance of the near-duplicate search, arcsec."
  jsonb hits "Observations with at least one MPC near-duplicate, as a JSON array of CndHit ([] when none)."
}
"public.skybot_bulk_jobs" {
  bigint id "Identifier of the job."
  text status "running, done, failed, killed or interrupted."
  double_precision radius_arcsec "Search radius around each observation, arcsec."
  timestamp_with_time_zone started_at "When the job started."
  timestamp_with_time_zone finished_at "When it ended; NULL while running."
  bigint total_observations "Eligible observations in the whole dataset (population-wide progress)."
  bigint processed_observations "Eligible observations ever checked (resumed jobs start from the previous count)."
  bigint matched_observations "Of those, observations matched to a known object."
  boolean kill_requested "Set by the UI to ask the running job to stop."
  text error "Error message of a failed job."
  text__ logs "Milestone log lines of the job (start, resume, kill, end)."
}
"public.skybot_obs_status" {
  bigint obs_id "Observation checked (observations.id, by value). Primary key."
  text lineage_designation "Lineage the observation was checked for."
  bigint branch_id "Branch the observation was checked for."
  timestamp_with_time_zone checked_at "When it was last checked."
  double_precision radius_arcsec "Search radius, arcsec."
  jsonb hits "Known objects found, as a JSON array of SkybotHit ([] when none)."
}
"public.mpc_submissions" {
  bigint id "Identifier of the attempt."
  text lineage_designation "Lineage submitted (branches.lineage_designation, by value)."
  bigint branch_id "Branch submitted (branches.branch_id, by value)."
  text trk_sub "Tracklet identifier (ADES trkSub) sent to the MPC."
  text endpoint "test (submit_xml_test) or production (submit_xml)."
  text submission_id "Acknowledgement id returned by the MPC; NULL if the POST failed."
  bigint__ observation_ids "Observations submitted (observations.id, by value); GIN-indexed for the duplicate check."
  text xml "Full ADES payload sent, for audit."
  text ack_message "Acknowledgement message returned by the MPC."
  text ac2_email "Contact email the MPC acknowledgement was sent to."
  timestamp_with_time_zone submitted_at "When the submission was sent."
  text verdict "pending, accepted, rejected or error; refreshed from the explorer's submission dashboard."
  timestamp_with_time_zone verdict_checked_at "When the verdict was last refreshed."
  jsonb verdict_detail "Raw status payload behind the verdict, as JSON."
  jsonb wamo_detail "Per-observation processing status from MPC WAMO, as JSON (production submissions only)."
  timestamp_with_time_zone wamo_checked_at "When wamo_detail was last refreshed."
}
```

---

> Generated by [tbls](https://github.com/k1LoW/tbls)

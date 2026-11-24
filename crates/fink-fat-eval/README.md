# fink-fat-eval

Evaluation tools for the Fink-FAT tracking engine. They run the engine
(`fink-fat-engine`) on **labelled** alert data, where every detection carries
the identity of the solar-system object it belongs to, and measure where
reconstruction succeeds or fails.

The crate is a library (the shared statistics, reports and plots) plus one
binary per study. It is a development tool: nothing in the production pipeline
depends on it.

## Binaries

| Binary | Purpose | Main outputs |
| --- | --- | --- |
| `tracking_analysis` | Full night-after-night tracking: advances a `BranchCollection` over the run and reports recall, purity, completeness and fragmentation, broken down by population. | `tracking_report.json`, plots (PNG) |
| `seed_to_bank_analysis` | Intra-night seeding in isolation: every night is seeded from an empty collection, as if it were night 0. Reports seed recall, purity and volume (banks, hypotheses). | report on stdout, plots (PNG) |
| `kalman_full_traj_eval` | Runs the topocentric Kalman filter bank on every ground-truth trajectory: dataset-wide statistics, best/worst trajectories by predictive-region coverage, detailed per-step reports. NEES/RMSE with `--ground-truth`. | report on stdout, optional Parquet |
| `kf_calibrate` | Automatic calibration of the search-region, gating and process-noise parameters: progressive-sampling coordinate descent that keeps recall high while shrinking the mean search-region radius. | `kf_calibration_report.json` |
| `mot_analysis` | Contamination study of the predicted search regions: per visit, how many detections fall inside besides the true one. | report on stdout |
| `kalman_single_traj_eval`, `helio_kf_update` | Development probes on a few hard-coded trajectories (per-step Kalman reports, heliocentric update). | stdout |

Every binary is selected with `--bin`:

```bash
cargo run --profile evaluation -p fink-fat-eval --bin tracking_analysis -- \
  --alerts /path/to/labelled_alerts.parquet \
  --config engine_config.yaml \
  --output-result eval_out/
```

## Inputs

- **`--alerts`**: a Parquet file of labelled alerts, loaded with `photom`
  (observation layout, observations grouped by night). The object identity of
  each detection is the ground truth every metric is computed against.
- **`--config`**: an engine configuration file, the same YAML format as the
  `fink-fat` binary (validated on load); see the
  [engine README](../fink-fat-engine/README.md) for its fields.
- **`--ground-truth`** (optional): a Parquet file of true topocentric states.
  Required for the NEES/RMSE metrics of the Kalman tools; without it they are
  skipped.

## Common options

Shared by every binary (`fink_fat_eval::cli::Cli`):

| Option | Meaning |
| --- | --- |
| `-a, --alerts <ALERTS_FILE>` | Labelled alerts (Parquet). Required. |
| `-c, --config <CONFIG_FILE>` | Engine configuration (YAML). Required. |
| `-o, --output-result <OUTPUT_DIR>` | Directory for reports and plots written to disk. |
| `--ground-truth <GROUND_TRUTH_FILE>` | True topocentric states, for NEES/RMSE. |
| `--steps-parquet-out <FILE>` | Per-step Kalman metrics (one row per filter step). |
| `--summary-parquet-out <FILE>` | Per-trajectory summary metrics. |
| `--override-obs-error-arcsec <ARCSEC>` | Replace every observation's `ra_err`/`dec_err` with one value, to sweep astrometric-noise assumptions without regenerating the input. |

### `tracking_analysis`

| Option | Meaning |
| --- | --- |
| `--from-json <JSON_FILE>` | Regenerate the plots from an existing `tracking_report.json`, without running the engine. |
| `--from-snapshot` | Report on the `.rkyv` snapshot written by `fink-fat` under the config's `storage_path`, instead of simulating the nights. |
| `--list-nights` | Print every night id with its observation count, then exit. |
| `--start-night <N>`, `--end-night <N>`, `--n-nights <N>` | Restrict the run to a range of nights. Tracking always starts from an empty collection at `--start-night`. |
| `--completeness-coverage-threshold <F>` | Fraction of a trajectory a pure branch must cover for the relaxed completeness metric (default `0.8`). |

### `kf_calibrate`

| Option | Default | Meaning |
| --- | --- | --- |
| `--seed` | `42` | Seed of the sampling RNG (same seed and inputs, same report). |
| `--initial-sample-size` | `50` | Trajectories sampled in round 0. |
| `--growth-factor` | `4.0` | Sample growth after each round that does not yet cover the dataset. |
| `--target-recall` | `0.98` | Recall above which cost reductions are preferred, and a full-dataset round may stop the run. |
| `--max-rounds` | `20` | Hard cap on the number of rounds. |
| `--max-failing-pool` | `200` | Failing trajectories carried into the next round's sample. |
| `--completion-threshold` | `0.98` | Completion fraction for "trajectory reconstructed". |
| `--within-radius-threshold` | `95.0` | Percentage of observations inside the search radius for "trajectory reconstructed". |
| `--params` | all | Comma-separated subset of parameters to calibrate, e.g. `max_arcsec,gate_chi2`. |
| `--min-traj-points` | `4` | Minimum observations for a trajectory to take part. |

## Logging

Every binary logs through `tracing`, filtered by the `FINKFAT_LOG`
environment variable (default `warn`):

```bash
FINKFAT_LOG=info cargo run --profile evaluation -p fink-fat-eval --bin mot_analysis -- ...
```

## Library modules

| Module | Content |
| --- | --- |
| `cli` | Shared command-line options and the alert loader. |
| `tracking_report` | Night-after-night tracking statistics and plots. |
| `seed_bank_report` | Intra-night seeding statistics. |
| `snapshot_report` | Statistics and plots of a `BranchCollection` snapshot. |
| `trajectory_processing`, `kalman_traj`, `reporting` | Running the Kalman filter bank on ground-truth trajectories, and its stdout reports. |
| `kalman_traj_plots` | Per-trajectory NIS/NEES time series. |
| `kf_calibration` | Parameter search of `kf_calibrate`. |
| `candidate_filters` | Candidate filters evaluated by the contamination study. |
| `ground_truth_state` | Lookup of the true topocentric state, for NEES/RMSE. |
| `population` | Orbital population of a trajectory (main belt, NEO, …), for per-population metrics. |
| `length_bin` | Trajectory-length bins. |
| `parquet_export` | Parquet export of the per-step and per-trajectory metrics. |

## Build profiles

The workspace `Cargo.toml` defines two profiles for these tools:

| Profile | Settings | Use |
| --- | --- | --- |
| `evaluation` | release optimisations, `lto = false`, `codegen-units = 128`, overflow checks and debug assertions on | investigations: fast incremental rebuilds, numerical bugs (NaN, broken invariants) caught early |
| `evaluation-fast` | release optimisations, thin LTO, `codegen-units = 1`, checks off | full-dataset sweeps once correctness is not in question |

The default debug profile is orders of magnitude slower on real datasets.

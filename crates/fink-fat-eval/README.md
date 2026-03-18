# fink-fat-eval

Evaluation binary for the Fink-FAT asteroid detection pipeline.  It runs the
engine against a labelled dataset (ground-truth SSO identities) and reports
purity/recall metrics at each pipeline stage.

Four independent evaluation modes are available:

| Subcommand | Pipeline stages executed | What it measures |
|---|---|---|
| `seeding-eval` | Ingest → Seeds | Seed purity and trajectory recall per night |
| `edge-eval` | Ingest → Seeds → Edges | Edge purity, lower/upper-bound recall |
| `solver-eval` | Ingest → Seeds → Edges → Solver | Full trajectory reconstruction quality |
| `model-eval` | *(offline, no pipeline)* | ONNX classifier ROC-AUC / PR-AUC on a pre-built feature file |

---

## Common arguments

All pipeline evaluation subcommands (`seeding-eval`, `edge-eval`, `solver-eval`) share the same base arguments:

| Argument | Description |
|---|---|
| `--alerts <URI>` | Input alert file URI (e.g. `file:///path/to/alerts.parquet`) |
| `--config <FILE>` | Engine configuration file (`eval_config.yml`) |
| `--plot-dir <DIR>` | *(optional)* Output directory for PNG diagnostic plots |

The **truth table** embedded in the alert file must contain at minimum two columns:
`dia_source_id` (uint64) and `trajectory_id` (int32).

---

## seeding-eval

Runs the ingestion and seeding stages only.  Reports per-night and global
statistics for the produced seeds:

- **Purity** — fraction of classifiable seeds that are true positives (all
  member alerts belong to the same ground-truth trajectory).
- **Recall** — fraction of recoverable ground-truth trajectories for which at
  least one TP seed was produced.

```bash
cargo run --profile evaluation -p fink-fat-eval -- seeding-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml
```

With plots:

```bash
cargo run --profile evaluation -p fink-fat-eval -- seeding-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --plot-dir seeding_plots
```

---

## edge-eval

Runs ingestion, seeding, and edge-building stages.  Reports per night-pair and
global statistics for the produced inter-night edges:

- **Purity** — fraction of classifiable edges that are true positives.
- **Lower-bound recall** — trajectories recovered by a TP edge whose exact
  `(from_night, to_night)` pair is among consecutive seeded nights.
- **Upper-bound recall** — trajectories recovered by any TP edge connecting two
  seeded nights within `max_gap_nights`, including skip edges.

The gap between lower and upper bound quantifies how many trajectories are
recovered only via skip edges.

```bash
cargo run --profile evaluation -p fink-fat-eval -- edge-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml
```

With plots:

```bash
cargo run --profile evaluation -p fink-fat-eval -- edge-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --plot-dir edge_plots
```

Exporting the edge feature dataset for ML training (produces a Parquet file
with 17 features, the truth label `is_true_edge`, and debug columns):

```bash
cargo run --profile evaluation -p fink-fat-eval -- edge-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --export-features crates/fink-fat-eval/src/bin/edge_ml_prediction/edge_features.parquet
```

---

## solver-eval

Runs the full pipeline including the graph solver.  Reports trajectory-level
quality metrics:

- **Purity** — fraction of reconstructed trajectories that are true positives
  (all member alerts from the same ground-truth object).
- **Recall** — fraction of recoverable trajectories for which at least one TP
  track was produced.
- **Partial recall** — fraction of recoverable trajectories for which at least
  one TP track covers ≥ 50 % of their total alerts.

```bash
cargo run --profile evaluation -p fink-fat-eval -- solver-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml
```

With plots:

```bash
cargo run --profile evaluation -p fink-fat-eval -- solver-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --plot-dir solver_plots
```

---

## model-eval

Offline evaluation of a trained ONNX edge classifier against the labelled
feature Parquet file produced by `edge-eval --export-features`.  No pipeline
is executed.

Loads the ONNX model path from the `training.onnx_output` field of
`xgb_params.yml`, runs inference on the full feature file, and reports:

- **ROC-AUC**
- **PR-AUC** (primary metric, more informative on imbalanced data)

and writes three diagnostic plots: `roc_curve.png`, `pr_curve.png`,
`score_distribution.png`.

```bash
cargo run --profile evaluation -p fink-fat-eval -- model-eval \
  -f crates/fink-fat-eval/src/bin/edge_ml_prediction/edge_features.parquet \
  -x crates/fink-fat-eval/src/bin/edge_ml_prediction/xgb_params.yml \
  -p model_eval_plots
```

| Flag | Description |
|---|---|
| `-f` / `--features-parquet` | Parquet file produced by `edge-eval --export-features` |
| `-x` / `--xgb-params` | `xgb_params.yml` — used to locate the ONNX model |
| `-p` / `--plot-dir` | *(optional)* Output directory for PNG plots |

---

## Configuration

The engine configuration file (`eval_config.yml`) controls all pipeline
parameters: spatial search radii, candidate speed bounds, cost function
variant, solver policy, and more.

The reference file is `crates/fink-fat-eval/eval_config.yml`.

---

## ML training pipeline

The full machine-learning workflow for the edge classifier lives in
`src/bin/edge_ml_prediction/`.  See its own
[README](src/bin/edge_ml_prediction/README.md) for the complete walkthrough
(data generation → HPO → training → ONNX export).

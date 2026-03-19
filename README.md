# Fink-FAT

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
![Version](https://img.shields.io/github/v/release/FusRoman/fink-fat)
![Licence](https://img.shields.io/github/license/FusRoman/fink-fat)

Fink-FAT is a Rust workspace for asteroid detection and trajectory reconstruction in alert streams. It ingests photometric alerts from surveys such as ZTF and Vera Rubin, builds intra-night seeds, links them across nights, solves candidate trajectories, and can optionally fit preliminary orbits.

The repository also contains evaluation tools for measuring seed, edge, and solver quality on labelled datasets, as well as utilities for exporting edge features and generating diagnostic plots.

## Highlights

- Ingest nightly or multi-night alert batches from configurable URIs.
- Build intra-night seeds from pairs and triplets of detections.
- Construct inter-night edges with configurable scoring and optional ONNX ranking.
- Solve connected components into candidate trajectories.
- Optionally run orbit fitting on selected tracks.
- Evaluate seeding, edge, solver, and model quality on labelled data.
- Persist intermediate state to disk for incremental processing.
- Provide progress bars and structured logs for long runs.

## Workspace layout

| Path | Role |
| --- | --- |
| `src/` | Binary entrypoint, CLI parsing, logging, progress hooks, and runtime orchestration. |
| `crates/fink-fat-engine/` | Core engine library: alerts, seeding, edges, solver, persistence, and configuration. |
| `crates/fink-fat-eval/` | Evaluation binary: seeding, edge, solver, and model evaluation modes. |
| `docs/` | Design notes and analysis documents. |
| `katex-header.html` | KaTeX header used when generating Rust documentation. |

## Requirements

- A recent stable Rust toolchain.
- Access to the survey alert files you want to process.
- A validated engine configuration file.
- Optional: an ONNX edge ranking model, if you enable model-based edge filtering.

For engine-specific runtime and configuration details, see [crates/fink-fat-engine/README.md](crates/fink-fat-engine/README.md).

## Quick start

Build the workspace:

```bash
cargo build --workspace
```

Run the main binary help:

```bash
cargo run -- --help
```

Run the automated tests:

```bash
cargo test --workspace
```

Generate the documentation:

```bash
RUSTDOCFLAGS="--html-in-header $(pwd)/katex-header.html" cargo doc --workspace --open
```

## Running the main pipeline

The `fink-fat` binary expects two required arguments:

- `--alerts` - URI of the alert file to process.
- `--config` - path to the engine configuration file.

Optional runtime flags:

- `--progress` enables `indicatif` progress bars.
- `--logs` enables terminal and file logging.

Example:

```bash
cargo run -- \
  --alerts file:///path/to/alerts.parquet \
  --config /path/to/config.yml \
  --progress \
  --logs
```

The pipeline runtime typically follows this sequence:

1. Load and validate the engine configuration.
2. Open or create the persistence layout.
3. Configure progress reporting and logging.
4. Ingest alerts.
5. Build seeds.
6. Build inter-night edges.
7. Solve connected components.
8. Fit orbits when enabled.
9. Persist the updated state.

Generated logs are written under the configured storage root, typically in a `logs/` subdirectory.

## Evaluation workflows

The `fink-fat-eval` crate provides four evaluation modes.

### Seeding evaluation

Measures seed purity and recall.

```bash
cargo run --profile evaluation -p fink-fat-eval -- seeding-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml
```

### Edge evaluation

Measures edge purity and recall, and can export the edge-feature dataset used for model training.

```bash
cargo run --profile evaluation -p fink-fat-eval -- edge-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --plot-dir edge_plots
```

Export features for machine learning:

```bash
cargo run --profile evaluation -p fink-fat-eval -- edge-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --export-features crates/fink-fat-eval/src/bin/edge_ml_prediction/edge_features.parquet
```

### Solver evaluation

Measures trajectory-level reconstruction quality.

```bash
cargo run --profile evaluation -p fink-fat-eval -- solver-eval \
  --alerts file:///path/to/sso_dataset_eval.parquet \
  --config crates/fink-fat-eval/eval_config.yml \
  --plot-dir solver_plots
```

### Model evaluation

Evaluates an ONNX classifier against a labelled feature file.

```bash
cargo run --profile evaluation -p fink-fat-eval -- model-eval \
  -f crates/fink-fat-eval/src/bin/edge_ml_prediction/edge_features.parquet \
  -x crates/fink-fat-eval/src/bin/edge_ml_prediction/xgb_params.yml \
  -p model_eval_plots
```

Use `-t` if you want to override the number of ONNX inference threads.

## Configuration

The engine uses a validated YAML configuration file to control:

- pair and triplet generation,
- edge construction and ranking,
- solver policy and routing,
- persistence layout,
- logging level,
- optional ONNX model paths.

The evaluation crate ships with a reference configuration at [crates/fink-fat-eval/eval_config.yml](crates/fink-fat-eval/eval_config.yml).

For a detailed explanation of the engine configuration schema, see [crates/fink-fat-engine/README.md](crates/fink-fat-engine/README.md).

## Documentation and design notes

- Rust API documentation is generated from the crate-level docs in `src/` and `crates/fink-fat-engine/src/`.
- High-level design notes are available in `docs/`.
- The workspace uses `katex-header.html` when generating docs so mathematical expressions render correctly.

## Testing and benchmarks

Common commands:

```bash
cargo test --workspace
cargo bench -p fink-fat-engine
```

The engine crate also provides dedicated Criterion benchmarks under `crates/fink-fat-engine/benches/`.

## Repository structure

```text
fink-fat/
├── src/                       # binary entrypoint and runtime glue
├── crates/
│   ├── fink-fat-engine/       # core engine library
│   └── fink-fat-eval/         # evaluation binary and ML helper code
├── docs/                      # analysis and architecture notes
├── katex-header.html          # rustdoc KaTeX header
└── README.md                  # project overview and usage
```

## License

See [LICENSE](LICENSE) for licensing information.

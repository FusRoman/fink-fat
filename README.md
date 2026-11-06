<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/fink-fat-lockup-white.svg">
    <img src="assets/logo/fink-fat-lockup.svg" alt="Fink-FAT — Fink Asteroid Tracker" width="460">
  </picture>
</h1>

<p align="center">
  <b>Finding new Solar System objects in the alert streams of large sky surveys.</b>
</p>

<div align="center">

[![CI](https://github.com/FusRoman/fink-fat/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FusRoman/fink-fat/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/FusRoman/fink-fat/branch/main/graph/badge.svg)](https://codecov.io/gh/FusRoman/fink-fat) [![Docs](https://img.shields.io/badge/docs-rustdoc%20%2B%20KaTeX-blue?logo=rust)](#documentation) [![Release](https://img.shields.io/github/v/release/FusRoman/fink-fat?include_prereleases&sort=semver)](https://github.com/FusRoman/fink-fat/releases) [![License: CeCILL-C](https://img.shields.io/badge/license-CeCILL--C-blue)](LICENSE) [![Rust 2024](https://img.shields.io/badge/rust-2024%20edition-orange?logo=rust)](https://doc.rust-lang.org/edition-guide/rust-2024/)<br>[![Fink](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastrolabsoftware%2Ffink-badges%2Fmain%2Fendpoints%2Ffink.json)](https://fink-broker.org) [![fink-fat](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastrolabsoftware%2Ffink-badges%2Fmain%2Fendpoints%2Ffink-fat.json)](https://github.com/FusRoman/fink-fat)

<a href="#quick-start">Quick start</a> · <a href="#how-it-works">How it works</a> · <a href="#documentation">Documentation</a> · <a href="#evaluation">Evaluation</a>

</div>

Fink-FAT finds **new Solar System objects** in the photometric alert streams of
large time-domain surveys (ZTF, Vera C. Rubin / LSST). It links
detections night after night into candidate trajectories, keeps several
association hypotheses alive while the data is ambiguous, and turns the
surviving candidates into orbits and MPC-ready ADES submissions.

The project started as a Python package used inside the
[Fink broker](https://fink-broker.org). Version 1.0 is a full rewrite in
Rust (see [History](#history)).

## How it works

Fink-FAT treats linking as a **tracking problem**, not as a combinatorial
search over all possible combinations of detections:

1. **Intra-night seeding.** Within a night, detections of the same moving
   object are linked into tracklets. A tracklet only constrains the sky
   position and angular rate. Range `ρ` and range-rate `ρ̇` are unknown.
2. **Admissible-region hypotheses.** Each tracklet seeds a *Kalman filter bank*:
   a set of topocentric Kalman filters, one per `(ρ, ρ̇)` node of the
   admissible region. Nodes are weighted by dynamical population priors
   (NEOs, main belt, Hildas, Trojans, Centaurs, Kuiper belt, scattered and
   detached objects).
3. **Night-to-night multi-hypothesis tracking.** Every lineage is propagated to
   each new exposure. Detections inside its predicted search ellipse spawn
   **branches**, which are scored by a log-likelihood ratio against the local
   clutter density, then pruned (top-B / N-scan). Range ambiguity collapses as
   the arc grows.
4. **Discovery and archiving.** Detections that no existing lineage claims seed
   new lineages. Lineages that stop receiving detections are archived, not
   lost: their association history is still a result.
5. **Orbit fitting and submission.** Candidate tracks are fitted (via
   [`outfit`](https://crates.io/crates/outfit)), cross-matched against known
   objects (SkyBoT, CND), ranked by a submission-quality tier and exported as
   ADES XML to the Minor Planet Center.

Steps 1–4 run in the engine. The
[engine README](crates/fink-fat-engine/README.md) describes the algorithms in
detail: state vector, filter equations, scoring, pruning and fragment
merging.

A few principles guide the design:

- **Completeness for unusual objects first.** Fink-FAT aims to find NEOs,
  Centaurs and distant objects, not only main-belt asteroids. Tuning choices
  are judged on metrics split by population, not on aggregate recall alone.
- **Incremental by construction.** Night 0 and night N use the same call
  (`BranchCollection::advance_one_night`). State is a single on-disk snapshot,
  so a nightly cron job and a multi-year replay run the same code.
- **Pure logic, thin I/O.** Algorithms are pure functions over plain data
  types. Postgres, HTTP and filesystem access live at the edges (CLI,
  explorer). For example, `fink-fat-ades` holds no I/O at all, so the CLI and
  the web app always agree on what a valid submission is.
- **Validated, unit-aware configuration.** One YAML file drives the whole
  engine. Unknown keys are rejected. Physical quantities are written with units
  (`"35 arcmin/day"`, `"0.02 au"`). Every validation error is reported at once.

## Workspace layout

| Crate | Path | Role |
| --- | --- | --- |
| `fink-fat` | [`src/`](src/) | Main CLI: `track`, `convert`, `submit`. |
| `fink-fat-engine` | [`crates/fink-fat-engine/`](crates/fink-fat-engine/README.md) | Core library: configuration, seeding, space-time bucketing, topocentric Kalman filters, branching / MHT, archive, snapshot. |
| `fink-fat-ades` | [`crates/fink-fat-ades/`](crates/fink-fat-ades/README.md) | Pure ADES / MPC logic: document building, local `submit.xsd` validation, submission and status API parsing, quality tiers. |
| `fink-fat-sql` | [`crates/fink-fat-sql/`](crates/fink-fat-sql/README.md) | Postgres sink: table schema (`sql/*.sql`), row shapes and `COPY` loaders shared by `convert` and the streaming pipeline. |
| `fink-fat-stream` | [`crates/fink-fat-stream/`](crates/fink-fat-stream/README.md) | Streaming driver library behind `fink-fat stream`: advances the engine visit by visit as Rubin alerts arrive (visit assembly, night clock, crash-safe night journal); replays Fink parquet dumps. |
| `fink-fat-eval` | [`crates/fink-fat-eval/`](crates/fink-fat-eval/README.md) | Evaluation binaries run against labelled datasets (tracking recall/purity, seeding, KF calibration, MOT contamination). |
| `fink-fat-explorer` | [`crates/fink-fat-explorer/`](crates/fink-fat-explorer/README.md) | Dioxus fullstack web app on Postgres: population maps, 3D orbits, lineage pages, orbit fitting, cross-matching, MPC submission dashboard. |

Other notable paths:

| Path | Content |
| --- | --- |
| [`tests/`](tests/) | CLI-level integration tests and small ZTF / LSST night samples. |
| [`test_exp/`](test_exp/) | Python scripts for data preparation (alert Parquet files, ground truth, astrometric error calibration). |
| `engine_config.best.yaml`, `engine_config_lsst.best.yaml` | Reference engine configurations for ZTF and LSST cadence. |
| [`dockerfile`](dockerfile), [`docker-compose.yml`](docker-compose.yml) | Postgres, pgAdmin and the explorer (`dev` / `prod` profiles). |
| [`katex-header.html`](katex-header.html) | KaTeX header so rustdoc renders the maths. |
| [`assets/logo/`](assets/logo/README.md) | Fink-FAT logo: SVG files (light, dark, white) and their design notes. |

## Requirements

- Rust stable with 2024 edition support (≥ 1.85).
- `libfontconfig1-dev` (used by the plotting code in `fink-fat-eval`).
- Network access on first run: the Kalman context downloads the JPL ephemeris
  and UT1 data, and the MPC observatory list.
- Optional: PostgreSQL (or `docker compose`) for `convert --format sql`,
  `submit` and the explorer.

## Quick start

```bash
cargo build --release
./target/release/fink-fat --help
```

### Track asteroids

```bash
fink-fat track \
  --alerts /path/to/alerts.parquet \
  --config engine_config.best.yaml \
  --progress --logs
```

`--alerts` is a Parquet file holding one night, or several nights with
distinct `night_id`s. Nights are processed in chronological order. The run
loads the `BranchCollection` snapshot from the config's `storage_path` if there
is one, advances it through every night, then writes it back. Closed
trajectories go to an append-only archive log next to the snapshot:

```text
<storage_path>/
├── branch_collection.rkyv          # live lineages (zero-copy rkyv snapshot)
├── archived_trajectories.rkyvlog   # append-only log of closed trajectories
└── fink_fat.<date>.log             # with --logs; rotated daily
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--snapshot-every N` | In batch mode, also checkpoint the snapshot every N nights. |
| `--log-target TARGET=LEVEL` | Per-target log level (repeatable). `--list-log-targets` lists them. |
| `--log-retention-days N` | Override how many daily log files are kept. |

Any configuration key can be overridden from the environment with the
`FINK_FAT__` prefix and `__` as the nesting separator.

### Export results

```bash
# Flat Parquet tables
fink-fat convert --config engine_config.best.yaml --format parquet

# Postgres (feeds fink-fat-explorer and `submit`)
fink-fat convert --config engine_config.best.yaml --format sql \
  --database-url postgres://finkfat:finkfat@localhost:5432/finkfat_test \
  --path-observation /path/to/alerts.parquet
```

The Parquet export writes one `<table>.parquet` file per table into
`storage_path`.

### Submit to the Minor Planet Center

```bash
fink-fat submit \
  --csv eligible_lineages.csv \
  --database-url postgres://finkfat:finkfat@localhost:5432/finkfat_test \
  --submitter-config submission.yaml \
  --dry-run
```

`submit` skips lineages that were already submitted, checks that each lineage
is eligible (by quality tier), builds the ADES document and validates it
locally. Then it sends the document and records the outcome in the
`mpc_submissions` table. `--endpoint test` is the default and targets MPC's
test endpoint. Only `--endpoint production` sends a real submission.
`--dry-run` stops before anything leaves the machine.

### Explore results in the browser

```bash
docker compose --profile dev up -d            # db + pgAdmin + explorer (hot reload) on :8080
docker compose --profile prod up -d --build   # optimised build on :8081
```

See [`crates/fink-fat-explorer/README.md`](crates/fink-fat-explorer/README.md)
for the build targets, runtime configuration and data loading.

## Evaluation

`fink-fat-eval` runs the engine on labelled data (alerts with their
ground-truth object identity) and reports where reconstruction succeeds or
fails. Each tool is a separate binary:

| Binary | Purpose |
| --- | --- |
| `tracking_analysis` | Full night-after-night tracking: recall, purity and fragmentation, broken down by population. |
| `seed_to_bank_analysis` | Intra-night seeding in isolation (every night treated as night 0). |
| `kalman_full_traj_eval` / `kalman_single_traj_eval` | Kalman bank behaviour on ground-truth trajectories (coverage, NEES/RMSE). |
| `kf_calibrate` | Automatic search for search-region / gating / process-noise parameters. |
| `mot_analysis` | Contamination of the predicted search regions by unrelated detections. |

```bash
cargo run --profile evaluation-fast -p fink-fat-eval --bin tracking_analysis -- \
  --alerts /path/to/sso_dataset_eval.parquet \
  --config engine_config.best.yaml \
  --output-result eval_out/
```

The `evaluation` profile keeps debug assertions and overflow checks to catch
numerical bugs. `evaluation-fast` drops them for full-dataset sweeps.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude fink-fat-explorer --all-targets --all-features -- -D warnings
cargo test --workspace
```

The SQL scripts in `crates/fink-fat-sql/sql/` are linted with
[SQLFluff](https://sqlfluff.com) (style, Postgres dialect) and
[Squawk](https://squawkhq.com) (Postgres DDL safety). Install the pinned
versions once, then run both from the repository root:

```bash
pip install -r crates/fink-fat-sql/sql-lint-requirements.txt
sqlfluff lint crates/fink-fat-sql/sql            # `sqlfluff fix` to reformat
squawk --config crates/fink-fat-sql/.squawk.toml crates/fink-fat-sql/sql/*.sql
```

Tests use synthetic or bundled data only. The `sql_convert` integration test
needs a Postgres instance given by `DATABASE_URL`; `docker compose up -d db`
provides one with the same credentials as CI.

A `husky-rs` pre-commit hook runs the SQL linters, fmt, clippy, check and the
doc build. CI runs the same jobs, plus the Postgres integration test and engine coverage
(Codecov). Contributor conventions (error handling, collections, doc comment
format, tests) are listed in [`AGENTS.md`](AGENTS.md).

## Documentation

For an overview of the algorithms, start with the
[engine README](crates/fink-fat-engine/README.md).

The API reference is generated with rustdoc. Module-level docs explain both
the algorithms and the reasons behind the design choices. KaTeX renders the
equations:

```bash
RUSTDOCFLAGS="--html-in-header $(pwd)/katex-header.html" cargo doc --workspace --no-deps --open
```

Good entry points are `fink_fat_engine::topocentric_kf::kalman_bank` (the
filter bank), `fink_fat_engine::topocentric_kf::branching` (multi-hypothesis
tracking) and `fink_fat_engine::engine_config` (every configuration key and its
units).

## History

[![arXiv: Python version (v0.x)](https://img.shields.io/badge/arXiv-2305.01123%20%C2%B7%20Python%20v0.x-b31b1b?logo=arxiv)](https://arxiv.org/abs/2305.01123)

The first Fink-FAT was a Python package that ran inside the
[Fink broker](https://fink-broker.org). It is described in
[arXiv:2305.01123](https://arxiv.org/abs/2305.01123); its last release is
[v0.17.0](https://github.com/FusRoman/fink-fat/releases).

> [!NOTE]
> The paper describes that Python version, not this one. Version 1.0 is a
> complete rewrite in Rust, with a different architecture and association
> method (multi-hypothesis tracking over banks of topocentric Kalman filters,
> see [How it works](#how-it-works)). Use the paper for the history of the
> project, not as a description of the current code.

## License

Fink-FAT is distributed under the
[CeCILL-C](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html) free
software license, a French LGPL-style license compatible with EU law. See
[LICENSE](LICENSE).

## AI Disclosure

D3 — [daliascale.org](https://daliascale.org)

D3: Human and AI act as creative partners. AI shapes ideas or output in meaningful ways, but humans retain authorial control.
# Changelog

All notable changes to this project will be documented in this file.

This project follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - Unreleased

Version 1.0 is a complete rewrite of Fink-FAT in Rust. It replaces the Python
package of the 0.x series (last release: v0.17.0); nothing of the former code,
configuration or command-line interface is kept.

### Added

#### Tracking engine (`fink-fat-engine`)
- Multi-hypothesis tracking over banks of topocentric Kalman filters: each
  candidate object is a lineage of competing branches, each branch a Gaussian
  mixture over the unknown distance and radial velocity (ρ, ρ̇), seeded from a
  grid of orbital populations (NEO, main belt, Hildas, Trojans, Centaurs,
  KBOs, …).
- Intra-night seeding by a tracklet linker (time-ordered sweep with an
  incremental linear fit), and a HEALPix space-time index of the alerts.
- Night-after-night association with search regions predicted by the filter
  banks, candidate filters (photometric, cross-track, relative likelihood),
  a log-likelihood-ratio score against the local alert density, a detection
  probability from the limiting magnitude, N-scan pruning and stale-lineage
  purging.
- Two-body heliocentric propagation with light-time correction, JPL
  ephemerides and UT1 data through the `outfit` crate.
- Persistence: a zero-copy `rkyv` snapshot of the live lineages and an
  append-only archive of closed trajectories, so a run resumes where the
  previous one stopped.
- Lineage designations (`FF{YYYY}…`) and per-branch track identifiers that do
  not depend on thread scheduling.
- A validated YAML configuration with physical units (`"5 hours"`,
  `"30 arcmin"`, …), overridable from the environment (`FINK_FAT__` prefix),
  and a reference configuration in `config/engine_config.yaml`.
- Fragment linkage of split arcs (`branching::merge`), available but not yet
  called by the night loop.

#### Command-line interface (`fink-fat`)
- `fink-fat track`: batch tracking of a Parquet alert file, night after night,
  with optional periodic snapshots, structured logs per target and daily log
  rotation.
- `fink-fat stream`: real-time tracking, visit by visit, of Fink's Kafka
  stream of Rubin alerts or of a replayed Parquet dump.
- `fink-fat convert`: export of the tracking state to Parquet files or to
  Postgres.
- `fink-fat submit`: submission of eligible lineages to the Minor Planet
  Center as ADES documents, test endpoint by default, with a dry-run mode and
  a record of every attempt.

#### Real-time pipeline (`fink-fat-stream`)
- Kafka consumer for Fink's livestream (Avro alerts, SASL/SSL), with a
  per-partition watermark that restores arrival order across partitions.
- Visit assembly, a night clock closing nights on the next night's alerts, on
  morning twilight or on the `dayObs` rollover, and deduplication of alerts.
- Crash safety: a journal of the open night and a durable outbox of the rows
  to write, both fsynced before the source is acknowledged; a restart resumes
  the open night.
- Postgres publication without ever blocking tracking: idempotent inserts,
  exponential backoff during outages, and an atomic swap of the trajectory
  tables at each night close.
- Graceful stop on SIGTERM / SIGINT.

#### Database (`fink-fat-sql`)
- Postgres schema for observations, branches, Kalman banks, hypotheses,
  archived trajectories, orbit fits, cross-matches and MPC submissions.
- Versioned migrations with refinery, under an advisory lock.
- Bulk loading with binary `COPY`, and generated schema documentation (tbls).
- SQL linting with SQLFluff and Squawk.

#### ADES and MPC submission (`fink-fat-ades`)
- ADES 2022 document building, local validation of the `submit.xsd`
  constraints, removal of singleton nights.
- Request shaping and response parsing for the MPC submission endpoints, the
  Submission Status API and the WAMO per-observation status API.
- Submission quality tiers deciding which lineages are eligible.

#### Web explorer (`fink-fat-explorer`)
- Dioxus fullstack application on the Postgres database: homepage with
  population plots, a lineage table and a 3D view of the orbits, served from
  an in-memory snapshot rebuilt when the data changes.
- Lineage pages with Kalman metrics, orbit fitting, ADES export.
- Bulk orbit fits and cross-matches against CND and Skybot.
- MPC submission dashboard with live status from the MPC, and a generator for
  the submitter configuration.
- A versioned public API (`/api/v1/...`) to find the lineages of an alert.

#### Evaluation (`fink-fat-eval`)
- Evaluation binaries on labelled data: night-after-night tracking
  (`tracking_analysis`), seeding (`seed_to_bank_analysis`), Kalman bank
  behaviour (`kalman_full_traj_eval`), automatic calibration of the Kalman
  parameters (`kf_calibrate`) and contamination of the search regions
  (`mot_analysis`), with JSON, Parquet and plot outputs.

#### Deployment and tooling
- Docker images of the engine and the explorer, published to GHCR by CI once
  CI is green on `main` or on a release tag.
- `ff-ops` (`deploy/`): a command-line tool managing the production instance
  on OpenStack (Heat stacks for network, storage and machines, cloud-init, SSH
  access through a bastion, volume snapshots, images).
- Rust CI (format, clippy, check, documentation, tests with coverage,
  Postgres and Kafka integration tests), a pre-commit hook running the same
  checks, and rustdoc with KaTeX-rendered equations.
- A README per crate and a project logo.

### Removed
- The Python package `fink_fat/`, its `setup.py` packaging, and the Python
  command-line scripts (`fink_fat`, association, orbit and statistics CLIs).
- The DBSCAN seeding and the OrbFit-based orbit determination, replaced by
  the Kalman filter banks and the `outfit` crate.
- The notebooks, analysis scripts and Python CI of the 0.x series.

[1.0.0]: https://github.com/FusRoman/fink-fat/compare/v0.17.0...v1.0.0

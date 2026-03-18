# fink-fat-engine

Core detection-linking library for the Fink-FAT asteroid detection pipeline.
This crate provides the complete computational engine: from raw photometric
alerts to multi-night trajectory hypotheses and preliminary orbit fitting.

---

## Table of contents

1. [Overview](#overview)
2. [Pipeline stages](#pipeline-stages)
3. [Data model](#data-model)
4. [Configuration](#configuration)
5. [Module map](#module-map)
6. [Key types](#key-types)
7. [Persistence](#persistence)
8. [Orbit fitting](#orbit-fitting)
9. [Running benchmarks](#running-benchmarks)
10. [Development notes](#development-notes)

---

## Overview

`fink-fat-engine` implements a multi-night moving-object linking pipeline
designed for large-scale photometric surveys (ZTF, Vera Rubin Observatory).
The pipeline ingests nightly alert batches, builds **seeds** (intra-night
kinematic models from pairs or triplets of detections), connects seeds across
nights into a **directed graph**, solves each connected component for
trajectory hypotheses, and optionally fits a preliminary orbit.

The crate is a pure library. All I/O, CLI logic, and evaluation tooling live
in the adjacent `fink-fat` and `fink-fat-eval` crates.

---

## Pipeline stages

The engine runs as an ordered sequence of stages, orchestrated by
`PipelineRunner`. Each stage is independent and can be individually
enabled or disabled in the `PipelinePlan`.

```
LoadPersistedData
      │
      ▼
IngestNights        ← parse and store alert Parquet batches
      │
      ▼
BuildSeeds          ← form pairs/triplets → SeedNode (intra-night)
      │
      ▼
BuildEdges          ← inter-night kinematic linking → AlertLinkageDAG
      │
      ▼
Solve               ← connected components → TrackHypothesis
      │
      ▼
FitOrbit            ← preliminary orbit determination (outfit / find_orb)
      │
      ▼
SavePersistedData   ← flush alerts, seeds, edge journal, state
```

### Stage details

| Stage | Key input | Key output |
|---|---|---|
| `IngestNights` | Parquet alert files | `AlertStore` |
| `BuildSeeds` | `AlertStore` | `SeedStore`, `SeedSpatialIndex` |
| `BuildEdges` | `SeedStore` + spatial index | `AlertLinkageDAG` |
| `Solve` | `AlertLinkageDAG` components | `Vec<TrackHypothesis>` |
| `FitOrbit` | `TrackHypothesis` list | `FullOrbitResult` |
| `SavePersistedData` | Runtime state | incremental journal on disk |

Progress is reported through the `PipelineHooks` trait, which can be backed
by any progress-bar or logging implementation.

---

## Data model

### Alert

An `Alert` represents a single photometric detection:

| Field | Type | Unit |
|---|---|---|
| `ra`, `dec` | `f64` | radians, ICRS J2000 |
| `ra_err`, `dec_err` | `f64` | radians, 1σ |
| `mjd_tt` | `f64` | MJD TT (days) |
| `flux`, `flux_err` | `f64` | PSF difference flux (upstream-dependent) |
| `band` | `u8` | photometric band code (LSST: u=0 … y=5) |
| `dia_source_id` | `u64` | upstream unique detection identifier |

`Alert` implements total ordering (primary key: `mjd_tt`), bit-exact `Eq`/`Hash`,
and `Serialize`/`Deserialize`.

### SeedNode

A `SeedNode` is an intra-night kinematic model built from two or three alerts:

- **Pair** → linear motion model on a gnomonic tangent plane.
- **Triplet** → quadratic motion model (includes acceleration).

Each seed stores its `NightId`, a `TangentPlaneModel` (position, velocity,
covariance), photometric aggregates (`Photometry`), and member alert keys.

Seeds are the atomic units fed into the inter-night graph.

### Edge

A directed inter-night link from an earlier `SeedNode` (`from`) to a later
one (`to`). Each edge carries:

- `cost` — strictly positive scalar solver weight,
- `dt_days` — time gap between seeds,
- `active` flag — used for deactivation after track selection.

### TrackHypothesis

An ordered chain of `SeedNode`s connected by `Edge`s. The key fields are:

- `nodes: Vec<SeedKey>` — time-ordered seeds,
- `edges: Vec<EdgeKey>` — connecting edges,
- `cost: f64` — additive solver cost (lower is better),
- `night_span: u32` — number of nights spanned.

---

## Configuration

The engine is configured via a single YAML file loaded by
`load_engine_config_validated`. Unknown keys are rejected; missing fields
fall back to Rust defaults.

### Precedence (later overrides earlier)

1. Rust `Default` implementations.
2. YAML file at the specified path.
3. Environment variables with prefix `FINK_FAT` and separator `__`.

```bash
# Example environment overrides
export FINK_FAT__MAX_GAP_NIGHTS=4
export FINK_FAT__EDGES__TOP_K_PER_LEFT=64
```

### Top-level structure

```yaml
version: 1
max_gap_nights: 2          # maximum inter-night gap considered for linking
storage_path: "storage/"   # root for on-disk persistence

pairs:
  max_dt: "86.4 min"
  max_angular_speed: "35 arcmin/day"
  max_flux_difference: 2.5

triplets:
  max_dt_between: "30 min"
  max_pair_sep: "10 arcmin"
  max_predicted_residual: "5 arcmin"
  max_flux_difference: 2.5

edges:
  top_k_per_left: 32
  parallel_left_batches: true
  parallel_left_batch_size: 512
  predictor_config:
    k_sigma: 3.5
    noise: { variance_floor: 1.0e-12, drift_per_day: 0.0, curvature_per_day2: 5.0e-14 }
    pad_cell_radius: true
    time_bin_dt: 0.021
    v_slack: 0.0
  cost:
    variant: singer_cwna
    sigma_q: 1.0e-3

solver:
  policy:
    # routing thresholds (see SolverPolicy)
  bounded_beam:
    beam_width: 8
    max_tracks: 3
    max_tracks_per_source: 2
    max_expansions: 2048
    max_out_per_node: 4
```

### Edge operational modes

Three modes are available, controlled by `top_k_per_left` and `ml_post_filter`:

| Mode | `top_k_per_left` | `ml_post_filter` | Use-case |
|---|---|---|---|
| Emit-all | `~` (null) | `false` | Debug / dataset generation |
| Cost-based Top-K | `Some(k)` | `false` | Production without ONNX |
| Top-K + ML post-filter | `Some(k)` | `true` | Production with ONNX model |

When `ml_post_filter: true`, the engine scores the entire retained edge set
**once** with an ONNX binary classifier and discards edges below
`ml_post_filter_threshold`. This is cheaper than per-seed inference because
ONNX runs on the already-pruned set.

```yaml
edges:
  ml_post_filter: true
  ml_post_filter_threshold: 0.5
  edge_ranking_model_path: "edge_ranker.onnx"
  onnx_batch_size: 128
  onnx_intra_threads: 4   # optional; None = ORT auto-select
  top_k_per_left: 32
```

### Cost function

The scalar edge cost drives both Top-K candidate selection and graph solvers.
It is computed by
[`EdgeFeatures::compute_cost`](https://docs.rs/fink-fat-engine/latest/fink_fat_engine/graph/edge/edge_features/struct.EdgeFeatures.html#method.compute_cost)
in three independent steps:

$$c = c\_{\mathrm{kin}}(\chi^2\_{\mathrm{pos}},\, \chi^2\_{\mathrm{vel}}) + c\_{\mathrm{phot}}$$

#### Step 1 — χ² extraction

The positional innovation uses the **spherical great-circle residual** $d$
(Vincenty formula):

$$\chi^2\_{\mathrm{pos}} = \frac{d^2}{S\_{\mathrm{pos}}}$$

where $S\_{\mathrm{pos}} = \mathrm{tr}(\mathbf{S}\_{\mathrm{pos}}) / 2$ is the
mean diagonal of the innovation covariance. The velocity innovation uses the
full 2×2 Mahalanobis:

$$\chi^2\_{\mathrm{vel}} = \delta\mathbf{v}^{\top} \mathbf{S}\_{\mathrm{vel}}^{-1} \delta\mathbf{v}$$

#### Step 2 — optional CWNA covariance inflation (Singer model)

When `sigma_q > 0`, both covariances are inflated by a **Continuous White
Noise Acceleration (CWNA)** diagonal term before computing χ² (Singer 1970):

$$\mathbf{S}\_{\mathrm{pos}} \leftarrow \mathbf{S}\_{\mathrm{pos}} + \sigma\_q^2 \frac{\Delta t^3}{3}\,\mathbf{I}$$

$$\mathbf{S}\_{\mathrm{vel}} \leftarrow \mathbf{S}\_{\mathrm{vel}} + \sigma\_q^2 \Delta t\,\mathbf{I}$$

This keeps $\chi^2 \sim O(1)$ for all inter-night gaps when $\sigma\_q$ is
well-calibrated (~$10^{-3}$ rad·day$^{-3/2}$), removing the $\Delta t^2$
growth that plagues the pure Gaussian model.

> Singer, R. A. (1970). *Estimating Optimal Tracking Filter Performance for
> Manned Maneuvering Targets.* IEEE Transactions on Aerospace and Electronic
> Systems, **6**(4), 473–483. doi:[10.1109/TAES.1970.310128](https://doi.org/10.1109/TAES.1970.310128)

#### Step 3 — kinematic loss variant

| Variant | $c\_{\mathrm{kin}}$ | CWNA | Notes |
|---|---|---|---|
| `gaussian_chi2` | $\frac{1}{2}(\chi^2\_\mathrm{pos} + \chi^2\_\mathrm{vel})$ | no | Standard NLL for constant-velocity model |
| `kinematic_log_likelihood` | same | no | Backward-compat alias for `gaussian_chi2` |
| `singer_cwna` | $\frac{1}{2}(\chi^2\_\mathrm{pos} + \chi^2\_\mathrm{vel})$ | **yes** | Same Gaussian NLL on CWNA-inflated covariances (**recommended**) |
| `robust_cauchy` | $\ln\!\bigl(1 + \chi^2\_\mathrm{pos}/\sigma\bigr) + \ln\!\bigl(1 + \chi^2\_\mathrm{vel}/\sigma\bigr)$ | optional | Saturates logarithmically; $\sigma$ = `cauchy_scale` |
| `robust_student_t` | $\frac{\nu+1}{2}\Bigl[\ln\!\bigl(1+\chi^2\_\mathrm{pos}/\nu\bigr) + \ln\!\bigl(1+\chi^2\_\mathrm{vel}/\nu\bigr)\Bigr]$ | optional | $\nu=1$: Cauchy; $\nu\!\to\!\infty$: Gaussian; $\nu$ = `student_nu` |

The robust loss functions follow the M-estimator framework
(Huber 1964; Black & Rangarajan 1996):

> Huber, P. J. (1964). *Robust Estimation of a Location Parameter.*
> The Annals of Mathematical Statistics, **35**(1), 73–101.
> doi:[10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)

> Black, M. J., & Rangarajan, A. (1996). *On the Unification of Line Processes,
> Outlier Rejection, and Robust Statistics with Applications in Early Vision.*
> International Journal of Computer Vision, **19**(1), 57–91.
> doi:[10.1007/BF00131148](https://doi.org/10.1007/BF00131148)

#### Step 4 — photometry penalty

Added unconditionally regardless of the kinematic variant:

$$c\_{\mathrm{phot}} = \frac{1}{2} z\_{\mathrm{flux}}^{2} + \frac{1}{2}\bigl[\ln(|r\_{\sigma}| + \varepsilon)\bigr]^{2} + b\_{\mathrm{band}}$$

where $z\_{\mathrm{flux}}$ is the flux z-score between the two seeds,
$r\_{\sigma}$ is the ratio of their flux standard deviations, and
$b\_{\mathrm{band}} = 0$ when both seeds share a photometric band,
$b\_{\mathrm{band}} \approx 6.9$ otherwise.

For full implementation details see
[`edge_features`](https://docs.rs/fink-fat-engine/latest/fink_fat_engine/graph/edge/edge_features/index.html)
and
[`edge_config`](https://docs.rs/fink-fat-engine/latest/fink_fat_engine/engine_config/edge_config/index.html).

---

## Module map

```
fink-fat-engine/src/
├── alerts/             Alert data model, AlertStore (per-night indexed store)
├── astro_math.rs       Spherical geometry, tangent-plane projections, linear algebra
├── display_format.rs   Indented debug formatting utilities
├── engine_config/      Root EngineConfig + sub-configs:
│   ├── pair_config.rs      PairConfig (intra-night pair pre-filter)
│   ├── triplet_config.rs   TripletConfig (intra-night triplet constraints)
│   ├── edge_config.rs      EdgeConfig (inter-night edge building + ML)
│   ├── propagator_config.rs PredictorParams (cone prediction for retrieval)
│   ├── solver_config/      SolverConfig + BoundedBeamConfig + SolverPolicy
│   └── pipeline_policy.rs  PersistPolicy
├── error.rs            Top-level EngineError / FinkFatError
├── graph/
│   ├── mod.rs          AlertLinkageDAG (directed edge store)
│   └── edge/
│       ├── mod.rs          Edge construction, build_edges entry-point
│       ├── edge_features.rs EdgeFeatures (17 kinematic + photometry features)
│       ├── edge_prediction.rs EdgeRankingModel / EdgeRankingModelPool (ONNX)
│       └── ranking_topk.rs rank_topk_edges_for_left_by_cost (cost-based Top-K)
├── night_id.rs         NightId (integer night identifier + PairingMode)
├── persistence/        PersistenceManager, layouts, edge journal, compression
├── pipeline/
│   ├── mod.rs          PipelineRunner, PipelinePlan, PipelineContext
│   ├── hooks.rs        PipelineHooks + StageProgress traits
│   └── stages/         One file per stage (alert_inputs, seed_builder, …)
├── seeding/
│   ├── mod.rs          SeedNode, from_pair / from_triplet constructors
│   ├── pairs.rs        Intra-night pair generation
│   ├── triplets.rs     Intra-night triplet generation
│   ├── tangent_plane.rs TangentPlaneModel (local kinematic model)
│   ├── photometry.rs   Photometry aggregates
│   ├── seed_spatial_index.rs SeedSpatialIndex (HEALPix + time-bin spatial index)
│   └── store.rs        SeedStore (per-night seed storage)
├── solver/
│   ├── mod.rs          Shared solver API: SolverOutput, TrackHypothesis, SolverDiagnostics
│   ├── components/     ConnectedComponents (component-restricted adjacency)
│   ├── bounded_beam.rs BoundedBeamSolver (beam-search path enumeration)
│   ├── min_cost_flow/  Min-cost flow solver (for larger components)
│   └── solver_manager.rs SolverManager (routing + dispatch)
├── spacetime_bucket/   HEALPix spatial binner + time binner (bucket indexing)
├── trajectory/         TrackHypothesis, TrackId, orbit-fitting I/O
└── units.rs            MJDTT, Radian, Arcsec type aliases
```

---

## Key types

| Type | Module | Role |
|---|---|---|
| `Alert` | `alerts` | Single photometric detection |
| `AlertStore` | `alerts::store` | Per-night alert storage |
| `SeedNode` | `seeding` | Intra-night kinematic seed (pair or triplet) |
| `SeedSpatialIndex` | `seeding::seed_spatial_index` | HEALPix + time spatial index over seeds |
| `TangentPlaneModel` | `seeding::tangent_plane` | Local gnomonic kinematic model |
| `Edge` | `graph::edge` | Directed inter-night seed link |
| `EdgeFeatures` | `graph::edge::edge_features` | 17-dimensional feature vector for ML + cost |
| `EdgeRankingModelPool` | `graph::edge::edge_prediction` | Thread-local lazy ONNX session pool |
| `AlertLinkageDAG` | `graph` | Inter-night directed edge store |
| `TrackHypothesis` | `trajectory` | Ordered multi-night seed chain |
| `PipelineRunner` | `pipeline` | Stage orchestrator |
| `EngineConfig` | `engine_config` | Root configuration container |
| `PersistenceManager` | `persistence` | On-disk state orchestrator |

---

## Persistence

State is persisted incrementally using an **edge journal** (delta-based) and
snapshot files for alerts and seeds. The layout is managed by
`PersistenceLayout` under the `storage_path` directory:

```
storage_path/
├── manifest.json
├── alerts/
│   └── <night_id>/alerts.bin.zst
├── seeds/
│   └── <night_id>/seeds.bin.zst
├── edges/
│   └── journal_<seq>.bin.zst   ← EdgeOp deltas
└── state.bin
```

Supported compression codecs: `zstd`, `lz4`, `gzip`, `none`.
Serialisation uses `bitcode` with `serde` for compact binary encoding.

---

## Orbit fitting

The `FitOrbit` stage uses the [`outfit`](https://crates.io/crates/outfit) crate
([source](https://github.com/FusRoman/Outfit)), a Rust library for initial orbit
determination (IOD) that wraps the Horizons ephemeris system (DE440) and the
FCCT14 error model.

The stage:

1. Converts track hypotheses to observation batches (`ObservationBatch`).
2. Initialises an `Outfit` environment (`DE440` + `FCCT14`).
3. Resolves observers from MPC observatory codes.
4. Builds a `TrajectorySet` from all observation groups.
5. Runs `estimate_all_orbits_in_batches_parallel` with configurable IOD
   parameters (noise realisations, triplet sampling, etc.).
6. Stores results in `RuntimeState::orbit_results` (`FullOrbitResult`).
7. Deactivates graph edges belonging to trajectories that received a confirmed
   orbit (persisted as `EdgeOp::Upsert { active: false }` in the edge journal).

Orbit fitting is optional: the stage can be omitted from the `PipelinePlan`
when running linking-only benchmarks.

---

## Running benchmarks

Two Criterion benchmark suites are provided:

```bash
# Cost-based Top-K edge generation (synthetic data)
cargo bench -p fink-fat-engine --bench generate_topk_edges

# Full edge build on real/realistic data
cargo bench -p fink-fat-engine --bench edge_build_real
```

---

## Development notes

### Coordinate conventions

- All angles: **radians**, ICRS J2000.
- All times: **MJD TT** (days).
- Tangent-plane velocities: **rad/day**.
- Costs: strictly positive, dimensionless.

### Thread safety

- `AlertStore`, `SeedStore`, `AlertLinkageDAG`: single-threaded (stage-owned).
- `EdgeRankingModelPool`: thread-safe via `thread_local::ThreadLocal` + `RefCell`.
  Each Rayon thread gets its own lazily-initialized ONNX session.
- `PipelineHooks`: `Send + Sync` required.

### Testing

Unit tests live alongside each module (`#[cfg(test)]` blocks).

```bash
cargo test -p fink-fat-engine
```

### Documentation

API documentation with KaTeX-rendered math:

```bash
RUSTDOCFLAGS="--html-in-header $(pwd)/katex-header.html" cargo doc --workspace --open
```

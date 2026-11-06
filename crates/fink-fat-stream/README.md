# fink-fat-stream

Streaming driver of Fink-FAT, as a library. The command line lives in the
`fink-fat` binary (`fink-fat stream`, see `src/stream.rs` at the workspace
root), which assembles it from this crate. `fink-fat track` advances the tracker one
**whole night** at a time over a file of alerts. `fink-fat-stream` advances it
**visit by visit** as Rubin alerts arrive, and drives the same engine code
(`NightSession`), so both paths give the same result.

This first version is synchronous and fed by a **replay** of a Fink parquet
dump, with a simulated clock. Postgres writing and a Kafka consumer will come
later, as additions to the same driver.

## Pipeline

```text
AlertSource ─► ingest ─► NightJournal ─► VisitAssembler ─► LiveNight
 (replay)     cuts,       append-only,     complete visits,   NightSession +
              convert,    fsync before     in seqNum order    the night's own
              dedup       ack                                 ObsDataset
                                                                  │
                          NightClock decides the close ───────────┘
                          → archive log, snapshot, journal reset
```

| Module | Role |
| --- | --- |
| `record` | `AlertRecord`: an owned alert (Rubin `DiaSource` fields) plus its arrival time. |
| `ingest` | Quality cuts and conversion (`photom::io::lsst_alert`), deduplication by `diaSourceId`. |
| `assembler` | `VisitAssembler`: buffers alerts by `visit`. A visit is complete after an idle timeout (90 s by default) and visits are emitted in `seqNum` order. |
| `clock` | `NightClock`: when a night closes. Includes an analytic Sun for twilight. |
| `night` | `LiveNight`: the engine's `NightSession`, plus a dataset holding only the open night's observations. |
| `journal` | `NightJournal`: append-only log of the open night, used to rebuild it after a crash. |
| `source` | `AlertSource` trait, and `ReplaySource` (parquet dump, column mapping in `ColumnMap`). |
| `driver` | `Driver`: the loop that ties everything together. |

## When a night closes

A night closes on the first of three triggers:

1. an alert of a later `dayObs` arrives;
2. morning twilight (Sun at −12°) plus a grace period has passed, the source
   is caught up and no visit is still being assembled;
3. safety net: the `dayObs` rollover (12:00 UTC) plus a grace period has
   passed.

A night never closes on inactivity: a weather pause leaves it open.

The clock is **event time**, i.e. the latest alert arrival seen. A live
source's wall clock can push it further, but only while the source is caught
up. During a backlog catch-up, the night closes on the backlog's own times.

## Crash safety

All files live under the engine's `storage_path`:

| File | Content |
| --- | --- |
| `branch_collection.rkyv` | Snapshot written at each night close, in the same format as `fink-fat track`. It records the last closed night. |
| `archived_trajectories.rkyvlog` | Archive of closed trajectories, in the same format as `fink-fat track`. |
| `night_journal.rkyvlog` | Journal of the open night: accepted alerts and "visit emitted" markers. |

Each batch from the source is journaled and fsynced **before** it is
acknowledged.

Closing a night writes, in this order:

1. the archive (fsynced);
2. the snapshot (atomic);
3. the journal truncation.

On restart, the driver:

1. loads the snapshot;
2. replays the journal: visits are re-emitted exactly where the markers put
   them;
3. resumes the source. Redelivered alerts are dropped as duplicates; alerts
   of closed nights count as late.

A crash between the archive write and the snapshot appends the night's
archived batch twice, with the same trajectories and designations.

## Usage

```sh
fink-fat stream --config stream.yaml [--logs] [--log-target stream=debug] [--log-retention-days N]
# from the workspace: cargo run --release -- stream --config stream.yaml
```

Minimal `stream.yaml` (every other section has defaults, see the `config`
module):

```yaml
engine_config: engine_config_lsst.best.yaml
replay:
  path: fink_dump.parquet
  latency: "3 min"
```

The dump needs the Rubin `DiaSource` columns `diaSourceId`, `visit`,
`midpointMjdTai`, `ra`, `dec`, `raErr`, `decErr`, `psfFlux`, `psfFluxErr` and
`band`. Other names are mapped under `replay.columns`.

## Tests

- Unit tests cover each pure building block: assembler, clock, twilight
  against astropy, ingestion, journal and replay parsing.
- `tests/night_session_regression.rs` (workspace root) feeds the golden
  fixture nights visit by visit through `LiveNight`, with one dataset per
  night, and checks the result against the batch golden fingerprint.
- `tests/stream_resume.rs` (workspace root) stops a replay mid-night and
  restarts it from storage. It checks that the final state matches an
  uninterrupted run.

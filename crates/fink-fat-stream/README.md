# fink-fat-stream

Streaming driver of Fink-FAT, as a library. The command line lives in the
`fink-fat` binary (`fink-fat stream`, see `src/stream.rs` at the workspace
root), which assembles it from this crate. `fink-fat track` advances the tracker one
**whole night** at a time over a file of alerts. `fink-fat-stream` advances it
**visit by visit** as Rubin alerts arrive, and drives the same engine code
(`NightSession`), so both paths give the same result.

The driver is synchronous. It reads alerts from one of two sources:

- **Kafka**: Fink's livestream of Rubin alerts, the production source;
- **replay**: a Fink parquet dump replayed as if live, with a simulated
  clock, for tests and validation.

It optionally writes to the explorer's Postgres database.

## Pipeline

```text
AlertSource ─► ingest ─► NightJournal ─► VisitAssembler ─► LiveNight
(Kafka,       cuts,       append-only,     complete visits,   NightSession +
replay)       convert,    fsync before     in seqNum order    the night's own
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
| `source` | `AlertSource` trait. `kafka`: `KafkaSource`, Fink's livestream (`kafka` feature). `replay`: `ReplaySource`, a parquet dump. `ColumnMap`: field names, shared by both. |
| `sink` | `SqlSink`: writes to the explorer's Postgres through a durable outbox, without ever blocking tracking. |
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

## Postgres

With a `database:` section in the configuration, the driver also feeds the
explorer's database. The database is a view of the driver's local state, not
its source of truth:

| Data | When | How |
| --- | --- | --- |
| accepted alerts → `observations` | every batch | queued in `sql_outbox.rkyvlog` (fsynced before the batch is acknowledged), then inserted idempotently when Postgres is reachable |
| archived trajectories → `archived_trajectories` | every night close | same queue |
| live branches → the 6 trajectory tables | every night close | atomic swap (see `fink-fat-sql`'s README) |

Postgres failures are logged and counted, never fatal:

- the outbox keeps everything through outages and restarts;
- flush retries are spaced by an exponential backoff;
- a failed swap is caught up at the next night close, or at the next start-up.
  On start-up, the driver publishes its snapshot if the database's
  `fink_fat_sync_state` is behind.

The URL comes from `database.url` or from the `FINK_FAT_DATABASE_URL`
environment variable, which keeps the password out of the YAML. It is never
logged.

The connection is not encrypted (`NoTls`). TLS comes with the move to a remote
database.

## Kafka source

The production source consumes Fink's Kafka livestream of Rubin alerts
(`source::kafka`, behind the `kafka` cargo feature, which the `fink-fat` binary
enables).

**Message format.** Fink publishes each alert as one message:

- the **key** holds the Avro writer schema as JSON;
- the **value** holds the alert as a *schemaless* Avro datum (no container
  header, no schema registry). This is what `fink-client` decodes.

The schema is parsed once and cached. `schema_path` gives a fallback schema
for messages whose key does not carry one.

The fields come from the alert's `diaSource` record, under their Rubin names.
The `layout:` sub-section remaps them (see `AlertLayout`). This was checked on
real alerts (`tests/fixtures/fink_lsst/`): `diaSource` holds `visit` and the
nine other fields, all nullable. An undecodable message is logged and skipped,
and so is one with a null required field. Its offset is still committed.

**Arrival order across partitions.** Fink spreads each visit's alerts over
all the partitions of its topic, and librdkafka reads them in turns balanced
by message count, not by time. Partitions therefore drift apart: on the live
stream, after 1200 messages, some had delivered alerts up to t = 366 s,
others only up to 240 s. Fed straight to the driver, whose clock is the
latest arrival seen, this declared visits complete before the lagging
partitions had delivered their share (a quarter of the alerts became
leftovers on the first live run), and would close a night on the first
alert of the next one while lagging partitions still held alerts of it.

The source therefore merges the partitions back into one stream in arrival
order, with a **watermark per partition** (`source::kafka::watermark`, whose
module doc walks through an example):

- each partition is ordered in time, so the time of the oldest alert it
  still buffers, or of the last alert fetched from it, bounds anything it
  may still deliver;
- fetched alerts wait in a buffer per partition and are released, oldest
  first, only once no partition can still deliver an older one (ties by
  `diaSourceId`, as the replay);
- a partition consumed to its end (the broker's high watermark offset,
  queried every `caught_up_check_interval`) does not hold the others back;
- a partition whose buffer reaches `max_buffered_per_partition` (default
  10 000) is paused until half drained.

The driver, the visit assembler and the night clock are unchanged: they see
what one time-ordered queue would deliver.

**At least once.**

- Auto-commit is off.
- After the driver has journaled a batch, the source commits, for each
  partition it still owns, the offset following its last **released**
  message. Alerts still buffered are not committed: a restart fetches them
  again. The commit is synchronous.
- A failed commit is logged and retried at the next batch. It is never fatal:
  at worst a restart redelivers alerts that the driver drops as duplicates or
  late.
- A restart with the same `group_id` resumes at the committed offsets.
- On a rebalance, a revoked partition is dropped with its buffer (its next
  owner reads those alerts again); a partition assigned back ignores the
  messages it had already fetched.

**Time.**

- An alert's arrival time is the Kafka message timestamp. If the message has
  none, it is Fink's `brokerIngestMjd`. As a last resort, it is the exposure
  mid-point plus `latency`. A backlog catch-up keeps the real arrival times,
  so nights close on them.
- The source counts as *caught up* when a poll returns without filling its
  batch, nothing is buffered, and every assigned partition has been consumed
  up to its high watermark. The watermarks are queried from the broker at
  most once per `caught_up_check_interval`. Each change of state is logged
  (`caught up with the Kafka stream` / `behind the Kafka stream`).
- Only while the source is caught up does the system clock move the night
  clock forward, so the night can close at twilight.

**Polling.**

- A batch ends after `batch_size` released alerts or `batch_timeout`,
  whichever comes first. It may be empty, which lets the clock advance. The
  consumer is synchronous: librdkafka fetches in its own threads.
- A night close keeps the driver away from `poll` for minutes. For this
  reason, `max_poll_interval` (librdkafka's `max.poll.interval.ms`) defaults
  to 30 min.

**Credentials.** SASL is SCRAM-SHA-512 over `sasl_plaintext` by default, as
Fink uses. The password comes from `sasl.password` or from the
`FINK_FAT_KAFKA_PASSWORD` environment variable. It is never logged. Raw
librdkafka properties go under `extra:`, for example `security.protocol:
sasl_ssl`. `extra:` cannot override the offset settings, the identity, or
the credentials.

**Graceful stop.** On SIGTERM or SIGINT, `fink-fat stream` finishes the
current step, flushes the SQL outbox and exits. It does **not** close the
open night: the night stays in the journal and the next start resumes it. A
second signal exits at once.

**Building.** librdkafka is compiled from its bundled sources. This needs a
C toolchain, `make`, and the OpenSSL headers (`libssl-dev`). The crate does
not use `cmake-build`: librdkafka 2.12's CMake build includes
`<curl/curl.h>` when SSL is enabled, even with curl turned off.

## Usage

```sh
fink-fat stream --config stream.yaml [--logs] [--log-target stream=debug] [--log-retention-days N]
# from the workspace: cargo run --release -- stream --config stream.yaml
```

Minimal `stream.yaml` (every other section has defaults, see the `config`
module). The `source:` section selects the source by `kind`. For Fink's
livestream:

```yaml
engine_config: engine_config_lsst.best.yaml
source:
  kind: kafka
  bootstrap_servers: <host:port given by Fink>
  topics: [<fink topic>]
  group_id: <group given by Fink>
  sasl: { username: <user> }       # password: env FINK_FAT_KAFKA_PASSWORD
```

For a replay:

```yaml
engine_config: engine_config_lsst.best.yaml
source:
  kind: replay
  path: fink_dump.parquet
  latency: "3 min"
```

The dump needs the Rubin `DiaSource` columns `diaSourceId`, `visit`,
`midpointMjdTai`, `ra`, `dec`, `raErr`, `decErr`, `psfFlux`, `psfFluxErr` and
`band`. Other names are mapped under `source.columns`.

## Tests

- Unit tests cover each pure building block:
  - assembler, clock, twilight against astropy;
  - ingestion, journal and replay parsing;
  - Kafka message decoding, offset bookkeeping and the backlog check, and the
    Kafka configuration (`cargo test -p fink-fat-stream --features kafka`).
- `tests/fink_alert_fixture.rs` (this crate, `kafka` feature) runs twelve
  **real** Fink LSST alerts of five visits (`tests/fixtures/fink_lsst/`,
  expected values in `expected.tsv`) through the source's pure path:
  decoding from their container files and as Kafka messages, the arrival-time
  fallbacks, damaged alerts (null or missing fields, truncated payload, no
  key), and the partition merge followed by visit assembly.
- `tests/night_session_regression.rs` (workspace root) feeds the golden
  fixture nights visit by visit through `LiveNight`, with one dataset per
  night, and checks the result against the batch golden fingerprint.
- `tests/stream_resume.rs` (workspace root) stops a replay mid-night and
  restarts it from storage. It checks that the final state matches an
  uninterrupted run.
- `tests/stream_sql.rs` (workspace root, needs `docker compose up -d db`) checks
  three things:
  - a replay with a database leaves exactly the tables `convert` would load;
  - a replay against an unreachable server catches up at restart;
  - the inserts are idempotent.
- `tests/stream_kafka.rs` (workspace root) needs a broker:
  `docker compose --profile kafka up -d kafka`, then
  `KAFKA_BOOTSTRAP=localhost:19092`. It checks:
  - a run fed through Kafka ends on the same state as a replay, and the
    source catches up — from one partition, and from four read out of step
    (no leftover, no late alert);
  - a run stopped gracefully mid-night resumes at its committed offsets,
    with nothing lost and nothing delivered twice — with one partition and
    with four, where buffered alerts were never committed;
  - an idle partition does not hold the others back;
  - an undecodable message is skipped and committed;
  - a raised stop flag stops at once;
  - the twelve real Fink alerts, over three partitions, come out in arrival
    order.

  Without `KAFKA_BOOTSTRAP` these tests return at once, unless
  `FINK_FAT_REQUIRE_KAFKA=1` (set by the CI), which makes them fail instead.

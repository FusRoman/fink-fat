//! # fink-fat-stream
//!
//! Streaming driver of the Fink-FAT tracker: advances the engine **visit by
//! visit** as Rubin alerts arrive, instead of one whole night at a time as
//! `fink-fat track` does.
//!
//! ```text
//! AlertSource ──► ingest ──► NightJournal ──► VisitAssembler ──► LiveNight
//! (Kafka, replay) (cuts,       (append-only,     (complete           (NightSession
//!                 convert,      fsync before      visits, in          + night's
//!                 dedup)        ack)              seqNum order)       ObsDataset)
//!                                                                         │
//!                                     NightClock decides the close ───────┘
//!                                     → archive log, snapshot, journal reset
//! ```
//!
//! | Module | Role | Pure |
//! |--------|------|------|
//! | [`record`] | [`AlertRecord`]: an owned alert + its arrival time | yes |
//! | [`ingest`] | Quality cuts, photom conversion, deduplication | yes |
//! | [`assembler`] | [`VisitAssembler`]: alerts → complete visits | yes |
//! | [`clock`] | [`NightClock`]: when a night closes; analytic twilight | yes |
//! | [`night`] | [`LiveNight`]: the engine's `NightSession` + the night's dataset | — |
//! | [`journal`] | [`NightJournal`]: crash recovery of the open night | I/O |
//! | [`source`] | [`AlertSource`] trait; Fink's Kafka livestream (`kafka` feature), [`ReplaySource`] (parquet dump) | I/O |
//! | [`sink`] | [`SqlSink`](sink::SqlSink): Postgres writes through a durable outbox, never blocking | I/O |
//! | [`driver`] | [`Driver`]: the loop tying everything together | I/O |
//! | [`config`] | [`StreamConfig`]: YAML configuration | — |
//!
//! The driver is synchronous, the Kafka source included (a librdkafka
//! consumer polled with a timeout; no async runtime). The Kafka source needs
//! the `kafka` feature (librdkafka is built from source), which the
//! `fink-fat` binary enables; without it the crate builds the replay source
//! only.
//!
//! This crate is a library only: the `fink-fat` binary assembles the
//! command line (`fink-fat stream`) from it.

pub mod assembler;
pub mod clock;
pub mod config;
pub mod driver;
pub mod error;
pub mod events;
pub mod ingest;
pub mod journal;
pub mod night;
pub mod record;
pub mod sink;
pub mod source;

pub use assembler::VisitAssembler;
pub use clock::NightClock;
pub use config::{DatabaseConfig, DriverConfig, StreamConfig};
pub use driver::{Driver, DriverStats, StreamPaths};
pub use error::{Result, StreamError};
pub use journal::NightJournal;
pub use night::LiveNight;
pub use record::AlertRecord;
pub use source::{AlertSource, ReplaySource, SourceConfig};

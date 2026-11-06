//! Postgres sink of fink-fat: the table schema, the row shapes, and the
//! loaders that write a [`BranchCollection`] and its observations into the
//! database read by `fink-fat-explorer`.
//!
//! Shared by the `fink-fat convert --format sql` CLI (full reload, see
//! [`write_bulk_tables`]) and the streaming binary, which reuses the same
//! building blocks so the schema and the row encoding exist only once.
//!
//! # Tables
//!
//! The 8 loaded tables, joined by `branch_id` / `hypothesis_id` / `id`:
//! `observations`, the 6 trajectory tables (`branches`, `kf_bank`,
//! `branch_observations`, `hypotheses`, `kf_state`, `branch_best_state` — a
//! denormalized copy of each branch's best hypothesis state that the
//! explorer's homepage reads instead of probing `hypotheses`/`kf_state`
//! once per branch), and `archived_trajectories`. See [`schema`] for their
//! persistence (only `observations` is WAL-logged) and the tables owned by
//! the explorer and `submit`.
//!
//! # Two writers
//!
//! | | `fink-fat convert` | streaming pipeline |
//! |-|--------------------|--------------------|
//! | `observations` | full reload | idempotent inserts, live ([`incremental`]) |
//! | trajectory tables | full reload (`TRUNCATE` + `COPY`) | nightly atomic swap ([`night_sync`]) |
//! | `archived_trajectories` | full reload | idempotent inserts, with the swap |
//! | entry point | [`write_bulk_tables`] | [`incremental`], [`night_sync`] |
//!
//! Both build their rows with the same pure functions ([`build`]) and write
//! them with the same `COPY` loaders ([`copy`]) and DDL scripts ([`schema`]);
//! the trajectory tables are loaded by the same [`load_trajectory_tables`],
//! into `public` or into the staging schema.
//!
//! # Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`rows`] | Plain row structs, no logic |
//! | [`build`] | Pure functions turning a [`BranchCollection`] into those rows (unit-tested) |
//! | [`family`] | Dynamical-family classification stored in `dynamic_family` columns |
//! | [`schema`] | DDL, backed by the `.sql` files of the crate's `sql/` directory |
//! | [`copy`] | `COPY ... FROM STDIN BINARY` loaders, one per table |
//! | [`load`] | Bulk loads: [`write_bulk_tables`], [`load_trajectory_tables`] |
//! | [`incremental`] | Idempotent inserts into `observations` / `archived_trajectories` |
//! | [`night_sync`] | Staging + atomic swap of the trajectory tables |
//! | [`error`] | [`SqlError`] |
//!
//! # Conventions
//!
//! `u64`/`usize` fields are cast to `i64` throughout — Postgres has no
//! unsigned integer type. No TLS: `Client::connect` is called with
//! `NoTls`, suitable for a local/trusted Postgres instance. Add TLS
//! support before talking to a non-local database.
//!
//! Bulk loads drop the secondary indexes / foreign keys of the tables they
//! fill before their `COPY`s and rebuild them after
//! ([`schema::drop_trajectory_constraints`] /
//! [`schema::restore_trajectory_constraints`]) — safe because the tables
//! are filled from scratch inside one transaction, invisible until commit.
//!
//! [`BranchCollection`]: fink_fat_engine::topocentric_kf::branching::BranchCollection

#![warn(missing_docs)]

pub mod build;
pub mod copy;
pub mod error;
pub mod family;
pub mod incremental;
pub mod load;
pub mod night_sync;
pub mod rows;
pub mod schema;

pub use error::SqlError;
pub use load::{load_trajectory_tables, write_bulk_tables};

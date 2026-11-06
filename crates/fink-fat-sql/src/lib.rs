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
//! The 8 bulk-loaded tables, joined by `branch_id` / `hypothesis_id` / `id`:
//! `observations`, `branches`, `kf_bank`, `branch_observations`,
//! `hypotheses`, `kf_state`, `archived_trajectories`, plus
//! `branch_best_state` — a denormalized copy of each branch's best
//! hypothesis state that the explorer's homepage reads instead of probing
//! `hypotheses`/`kf_state` once per branch.
//!
//! # Modules
//!
//! Organised by concern rather than by table:
//!
//! | Module | Role |
//! |--------|------|
//! | [`rows`] | Plain row structs, no logic |
//! | [`build`] | Pure functions turning a [`BranchCollection`] into those rows (unit-tested) |
//! | [`family`] | Dynamical-family classification stored in `dynamic_family` columns |
//! | [`schema`] | DDL, backed by the `.sql` files of the crate's `sql/` directory |
//! | [`copy`] | `COPY ... FROM STDIN BINARY` bulk loaders, one per table |
//! | [`load`] | [`write_bulk_tables`], the full reload orchestrating all of the above |
//! | [`error`] | [`SqlError`] |
//!
//! # Conventions
//!
//! `u64`/`usize` fields are cast to `i64` throughout — Postgres has no
//! unsigned integer type. No TLS: `Client::connect` is called with
//! `NoTls`, suitable for a local/trusted Postgres instance. Add TLS
//! support before talking to a non-local database.
//!
//! The 8 bulk-loaded tables are `UNLOGGED` (no WAL, reset to empty by
//! Postgres after a crash/unclean shutdown) and have their secondary
//! indexes/foreign keys dropped before the COPYs and rebuilt after
//! ([`schema::drop_bulk_load_constraints`]/[`schema::restore_bulk_load_constraints`])
//! — both are safe specifically because [`write_bulk_tables`] always fully
//! regenerates these tables from source data inside one transaction: there
//! is no window where a crash could lose anything these tables hold that
//! re-running `convert` wouldn't already recreate, and no query reads them
//! while they're temporarily unconstrained. `orbit_fits`/`orbit_fit_failures`
//! are unaffected — see `sql/create_orbit_fits_tables.sql`.
//!
//! [`BranchCollection`]: fink_fat_engine::topocentric_kf::branching::BranchCollection

#![warn(missing_docs)]

pub mod build;
pub mod copy;
pub mod error;
pub mod family;
pub mod load;
pub mod rows;
pub mod schema;

pub use error::SqlError;
pub use load::write_bulk_tables;

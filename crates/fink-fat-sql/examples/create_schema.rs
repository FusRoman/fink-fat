//! Create every fink-fat table, without any data, in a database recreated
//! from scratch — the input of the generated schema documentation.
//!
//! ```text
//! DATABASE_URL=postgres://finkfat:finkfat@localhost:5432/finkfat_test \
//!     cargo run -p fink-fat-sql --example create_schema -- fink_fat_schema_docs
//! ```
//!
//! `DATABASE_URL` only locates the server (any database of it the user can
//! connect to); the argument names the database to drop and recreate.
//!
//! # Why a fresh database
//!
//! tbls lists tables in catalogue order, i.e. in creation order. A database
//! that has lived through loads and nightly swaps has its trajectory tables
//! recreated last, so its docs would differ from the ones CI generates on a
//! fresh server (`tbls diff` fails). Building the docs on a database created
//! by [`fink_fat_sql::schema::create_tables`] alone gives the same order
//! everywhere — and never touches a development database.

use postgres::{Client, NoTls};

/// Default server: the `db` service of the repo's `docker-compose.yml`,
/// reached from the host.
const DEFAULT_DATABASE_URL: &str = "postgres://finkfat:finkfat@localhost:5432/finkfat_test";

/// Database recreated when no name is given.
const DEFAULT_SCHEMA_DATABASE: &str = "fink_fat_schema_docs";

/// Whether `name` is safe to interpolate as a database identifier.
fn is_plain_identifier(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_string());
    let database = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_SCHEMA_DATABASE.to_string());
    if !is_plain_identifier(&database) {
        return Err(format!("invalid database name {database:?}: use [a-z0-9_]").into());
    }

    // 1. Recreate the database (outside any transaction, as Postgres
    //    requires); `FORCE` closes leftover connections to it.
    let mut server = Client::connect(&url, NoTls)?;
    server.batch_execute(&format!("DROP DATABASE IF EXISTS {database} WITH (FORCE)"))?;
    server.batch_execute(&format!("CREATE DATABASE {database}"))?;
    drop(server);

    // 2. Create every table in it, exactly as the loaders do.
    let mut config: postgres::Config = url.parse()?;
    config.dbname(&database);
    let mut client = config.connect(NoTls)?;
    let mut transaction = client.transaction()?;
    fink_fat_sql::schema::create_tables(&mut transaction)?;
    transaction.commit()?;
    Ok(())
}

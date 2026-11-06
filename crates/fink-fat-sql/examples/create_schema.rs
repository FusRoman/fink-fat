//! Create every fink-fat table in the database at `$DATABASE_URL`
//! (idempotent: [`fink_fat_sql::schema::create_tables`]), without loading
//! any data.
//!
//! Used to generate the schema documentation (`docs/generate_schema_docs.sh`,
//! the `schema-docs` CI job): tbls documents a live database, and this builds
//! one from the very scripts the loaders run.
//!
//! ```text
//! DATABASE_URL=postgres://finkfat:finkfat@localhost:5432/finkfat_test \
//!     cargo run -p fink-fat-sql --example create_schema
//! ```

use postgres::{Client, NoTls};

/// Default connection string: the `db` service of the repo's
/// `docker-compose.yml`, reached from the host.
const DEFAULT_DATABASE_URL: &str = "postgres://finkfat:finkfat@localhost:5432/finkfat_test";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_string());
    let mut client = Client::connect(&url, NoTls)?;
    let mut transaction = client.transaction()?;
    fink_fat_sql::schema::create_tables(&mut transaction)?;
    transaction.commit()?;
    Ok(())
}

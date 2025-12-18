pub mod schema;
pub mod ztf_alerts;
pub mod ingest_config;

use anyhow::Result;
use camino::{Utf8Path, Utf8PathBuf};

#[derive(Clone, Debug)]
pub struct ParquetSource {
    pub path: Utf8PathBuf,
}

impl ParquetSource {
    pub fn new(path: impl Into<Utf8PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            anyhow::bail!("Parquet file not found: {}", path);
        }
        Ok(Self { path })
    }

    pub fn as_path(&self) -> &Utf8Path {
        &self.path
    }
}

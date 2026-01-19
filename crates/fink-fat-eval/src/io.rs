use anyhow::{Context, Result};
use camino::Utf8Path;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{BufWriter, Write},
};

/// Write a value to disk using `bitcode` (serde-backed) binary format.
///
/// Notes
/// -----
/// - This writes the entire serialized buffer at once (no streaming API).
/// - The bitcode format is not guaranteed stable across major versions.
pub fn write_bin<T: Serialize + ?Sized>(path: &Utf8Path, value: &T) -> Result<()> {
    let bytes = bitcode::serialize(value)?;
    let file = File::create(path).with_context(|| format!("create {}", path))?;
    let mut w = BufWriter::new(file);
    w.write_all(&bytes).context("write bitcode bytes")?;
    Ok(())
}

/// Read a value from disk using `bitcode` (serde-backed) binary format.
///
/// Notes
/// -----
/// - This loads the whole file into memory before deserializing.
pub fn read_bin<T: DeserializeOwned>(path: &Utf8Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path))?;
    let v = bitcode::deserialize(&bytes).context("bitcode deserialize")?;
    Ok(v)
}

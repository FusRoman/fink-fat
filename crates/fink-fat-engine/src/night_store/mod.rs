// src/night_store/mod.rs

//! Per-night persistence of alerts/pairs/triplets/seeds.
//!
//! # Overview
//! This module defines **lightweight per-night storage** for fink-fat. The
//! engine is invoked once per night and the process exits, so all state that
//! needs to survive across runs must be written to disk.
//!
//! In this first iteration we focus on persisting **seeds**, as they are the
//! primary inputs for the inter-night graph. Alerts, pairs, and triplets can
//! be added later using the same pattern.
//!
//! On disk, the layout is:
//!
//! ```text
//! <state_root>/
//!   nights/
//!     <night_id>/
//!       seeds.bin       # Vec<SeedNode> (bincode)
//!       summary.json    # NightSummary (JSON, human-readable)
//! ```
//!
//! The `NightStore` type owns the root path (as a UTF-8 path via `camino`) and
//! provides helpers to:
//!
//! - write `NightSummary` + `Vec<SeedNode>` for a given night,
//! - reload `NightSeeds` (seeds + spatial index) when building the inter-night graph.

use std::fs;
use std::io::{BufReader, BufWriter};

use camino::{Utf8Path, Utf8PathBuf};
use serde::{Deserialize, Serialize};

use crate::seeding::seed_node::SeedNode;
use crate::seeding::seed_spatial_index::SeedSpatialIndex;
use crate::{
    error::FinkFatError, night_id::NightId, spacetime_bucket::spatial_binner::SpatialBinner,
};

/// File name constants for on-disk artifacts.
///
/// Notes
/// -----
/// These are implementation details of the current persistence format. If you
/// ever need to change them, bump a schema/version number at a higher level.
const SEEDS_FILE: &str = "seeds.bin";
const SUMMARY_FILE: &str = "summary.json";

/// Lightweight summary of a processed night persisted on disk.
///
/// Overview
/// --------
/// `NightSummary` holds enough information to:
/// - know whether a night is fully ingested,
/// - reload its seeds and spatial index on demand,
/// - participate in inter-night linking.
///
/// Heavy arrays (e.g., `Vec<SeedNode>`) live on disk in the `data_path`
/// directory and are loaded lazily.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NightSummary {
    /// Logical night identifier (e.g., MJD or integer run id).
    pub night_id: NightId,
    /// Number of seeds persisted for this night (sanity-check only).
    pub n_seeds: u32,
    /// Optional processing timestamp as Unix seconds since epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_at_unix_secs: Option<i64>,
}

/// In-memory representation of a processed night for inter-night linking.
///
/// Overview
/// --------
/// `NightSeeds` is what the graph-builder actually needs:
/// - all `SeedNode` for the night, and
/// - a `SeedSpatialIndex` built on `SeedNode.ra_mid/dec_mid`.
///
/// Heavy upstream artifacts (alerts, pairs, triplets) are not required here,
/// and can be added later if you decide to persist them.
#[derive(Debug)]
pub struct NightSeeds {
    /// Night identifier, copied from `NightSummary.night_id`.
    pub night_id: NightId,
    /// All intra-night seeds for that night.
    pub seeds: Vec<SeedNode>,
    /// Spatial index built from `seeds` for fast cone queries.
    pub spatial_index: SeedSpatialIndex,
}

/// Per-night storage manager.
///
/// Overview
/// --------
/// `NightStore` encapsulates the on-disk layout used to persist per-night
/// artifacts. It owns a UTF-8 root directory and provides helper methods to:
///
/// - compute the directory for a given `night_id`,
/// - write seeds and a `NightSummary` for that night,
/// - reload seeds and build a `NightSeeds` structure on demand.
///
/// Layout
/// ------
/// For a given `night_id`, the directory is:
///
/// ```text
/// <root>/nights/<night_id>/
/// ```
///
/// where `<night_id>` is typically rendered as an integer or string via
/// `NightId`'s `Display` implementation.
///
/// Invariants
/// ----------
/// - `save_night_seeds` overwrites existing files for that night.
/// - `load_night_seeds` assumes `seeds.bin` is present and consistent with
///   the `n_seeds` field in `NightSummary`.
pub struct NightStore {
    root: Utf8PathBuf,
}

impl NightStore {
    /// Create a new `NightStore` rooted at the given directory.
    ///
    /// Arguments
    /// ---------
    /// * `root` – Base directory for fink-fat state (e.g. `<state_root>`).
    ///
    /// Notes
    /// -----
    /// This function does **not** create any directories yet. They will be
    /// created lazily when `save_night_seeds` is called.
    pub fn new(root: impl AsRef<Utf8Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Return the directory on disk for a given `night_id`.
    ///
    /// Layout
    /// ------
    /// The directory is:
    ///
    /// ```text
    /// <root>/nights/<night_id>/
    /// ```
    ///
    /// where `<night_id>` is rendered using `NightId`'s `Display` impl.
    #[inline]
    pub fn night_dir(&self, night_id: NightId) -> Utf8PathBuf {
        // Adjust formatting of `night_id` via its Display impl if needed.
        let mut dir = self.root.join("nights");
        dir.push(night_id.to_string());
        dir
    }

    pub fn save_night_seeds(
        &self,
        night_id: NightId,
        seeds: &[SeedNode],
    ) -> Result<NightSummary, FinkFatError> {
        let night_dir = self.night_dir(night_id);

        // Ensure directory exists
        fs::create_dir_all(night_dir.as_std_path()).map_err(FinkFatError::from)?;

        // 1) Write seeds.bin (bincode 2.x) 🔧
        let seeds_path = night_dir.join(SEEDS_FILE);
        let file = fs::File::create(seeds_path.as_std_path()).map_err(FinkFatError::from)?;
        let mut writer = BufWriter::new(file);

        // encode_into_std_write<T: Encode, W: Write>
        let _ = bincode::encode_into_std_write(seeds, &mut writer, bincode::config::standard())
            .map_err(FinkFatError::from)?;

        // 2) Build summary
        let now = chrono::Utc::now();
        let summary = NightSummary {
            night_id,
            n_seeds: seeds.len() as u32,
            processed_at_unix_secs: Some(now.timestamp()),
        };

        // 3) Write summary.json (human-readable)
        let summary_path = night_dir.join(SUMMARY_FILE);
        let summary_file =
            fs::File::create(summary_path.as_std_path()).map_err(FinkFatError::from)?;
        let mut summary_writer = BufWriter::new(summary_file);
        serde_json::to_writer_pretty(&mut summary_writer, &summary).map_err(FinkFatError::from)?;

        Ok(summary)
    }

    /// Load the `NightSummary` for a given night, if present.
    ///
    /// Return
    /// ------
    /// * `Ok(Some(NightSummary))` if the summary exists and is readable.
    /// * `Ok(None)` if the directory or summary file is missing.
    /// * `Err(FinkFatError)` on I/O or deserialization errors.
    pub fn load_night_summary(
        &self,
        night_id: NightId,
    ) -> Result<Option<NightSummary>, FinkFatError> {
        let night_dir = self.night_dir(night_id);
        let summary_path = night_dir.join(SUMMARY_FILE);

        // `exists` is on std::path, so use `as_std_path()`.
        if !summary_path.as_std_path().exists() {
            return Ok(None);
        }

        let file = fs::File::open(summary_path.as_std_path()).map_err(FinkFatError::from)?;
        let reader = BufReader::new(file);
        let summary: NightSummary = serde_json::from_reader(reader).map_err(FinkFatError::from)?;
        Ok(Some(summary))
    }

    /// Load seeds + rebuild spatial index for a given night.
    ///
    /// Overview
    /// --------
    /// This helper reconstructs the `NightSeeds` structure expected by the
    /// graph builder:
    ///
    /// 1. Read `seeds.bin` from disk and deserialize into `Vec<SeedNode>`.
    /// 2. Build a `SeedSpatialIndex` using the provided `binner`.
    ///
    /// Arguments
    /// ---------
    /// * `summary` – Summary describing where this night's data is stored.
    /// * `binner`  – Spatial partitioner used to build the index.
    ///
    /// Return
    /// ------
    /// * `NightSeeds` with all seeds and a fresh spatial index.
    pub fn load_night_seeds<Bs: SpatialBinner>(
        &self,
        summary: &NightSummary,
        binner: &Bs,
    ) -> Result<NightSeeds, FinkFatError> {
        // Recompute directory from night_id
        let night_dir = self.night_dir(summary.night_id);
        let seeds_path = night_dir.join(SEEDS_FILE);

        let file = fs::File::open(seeds_path.as_std_path()).map_err(FinkFatError::from)?;
        let mut reader = BufReader::new(file);

        // bincode 2.x decode 🔧
        let seeds: Vec<SeedNode> =
            bincode::decode_from_std_read(&mut reader, bincode::config::standard())
                .map_err(FinkFatError::from)?;

        // Optional sanity check
        if seeds.len() as u32 != summary.n_seeds {
            return Err(FinkFatError::from(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "seed count mismatch between summary and seeds.bin",
            )));
        }

        let spatial_index = SeedSpatialIndex::build(&seeds, binner);

        Ok(NightSeeds {
            night_id: summary.night_id,
            seeds,
            spatial_index,
        })
    }
}

//! Helpers for verifying pipeline reconstruction quality against ground-truth
//! trajectories stored in the fixture parquet files.
//!
//! # Fragment model
//!
//! The bounded-beam solver emits a track as soon as it reaches `min_nodes`
//! nodes, without extending it further.  A long ground-truth trajectory is
//! therefore expected to produce **multiple short fragments** rather than a
//! single continuous track.  The correct way to evaluate reconstruction quality
//! is to:
//!
//! 1. Collect every detected track that shares at least one observation with
//!    the GT trajectory — these are its **fragments**.
//! 2. Check that each fragment is **pure**: all of its observations belong to
//!    the same GT trajectory (`purity = n_shared / n_track_total`).
//! 3. Measure **union coverage**: the fraction of the GT observation set covered
//!    by any fragment (`union_covered / n_gt_alerts`).
//!
//! # Assumptions
//!
//! The solver deactivates edges after emitting a track, so the observation sets
//! of different tracks are disjoint within one night's solve pass.  Union
//! coverage is therefore computed as `Σ n_shared / n_gt_alerts` without
//! explicit set deduplication.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    path::Path,
};

use camino::Utf8Path;
use datafusion::arrow::array::{Array, Int32Array, StringArray, UInt64Array};
use fink_fat_engine::persistence::envelope::load_parquet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single detected track that partially covers a ground-truth trajectory.
#[derive(Debug, Clone)]
pub struct FragmentResult {
    /// Pipeline-assigned track identifier (e.g. `TRK2025…`).
    pub track_id: String,
    /// Number of observations from the GT trajectory present in this track.
    pub n_shared: usize,
    /// Total number of observations in this track (from all GT trajectories).
    pub n_track_total: usize,
    /// `n_shared / n_track_total` — 1.0 means the track is 100 % pure GT.
    pub purity: f64,
    /// `n_shared / n_gt_alerts` — fraction of the GT trajectory covered.
    pub gt_coverage: f64,
}

/// Reconstruction analysis result for one ground-truth trajectory.
#[derive(Debug)]
pub struct ReconstructionResult {
    /// Ground-truth trajectory identifier from the fixture data.
    pub traj_id: i32,
    /// Total number of observations in the ground-truth trajectory.
    pub n_gt_alerts: usize,
    /// All detected tracks that share at least one observation with this GT
    /// trajectory, sorted by descending purity then descending gt_coverage.
    pub fragments: Vec<FragmentResult>,
    /// Total number of distinct tracks present in the pipeline output across
    /// all nights (useful to distinguish "no tracks at all" from "tracks exist
    /// but none match this GT").
    pub total_tracks_in_storage: usize,
}

impl ReconstructionResult {
    /// Number of fragments that partially cover this GT trajectory.
    pub fn n_fragments(&self) -> usize {
        self.fragments.len()
    }

    /// Whether the pipeline produced at least one fragment for this GT trajectory.
    pub fn has_any_fragment(&self) -> bool {
        !self.fragments.is_empty()
    }

    /// Fraction of GT observations covered by any fragment combined.
    ///
    /// This is `Σ n_shared / n_gt_alerts`.  Because the solver produces
    /// disjoint tracks, this equals the true set-union coverage.
    pub fn union_coverage(&self) -> f64 {
        if self.n_gt_alerts == 0 {
            return 0.0;
        }
        let total_shared: usize = self.fragments.iter().map(|f| f.n_shared).sum();
        total_shared as f64 / self.n_gt_alerts as f64
    }

    /// Whether every fragment has a purity of at least `min_purity`.
    pub fn all_pure(&self, min_purity: f64) -> bool {
        self.fragments.iter().all(|f| f.purity >= min_purity)
    }
}

impl fmt::Display for ReconstructionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GT traj {} | {} obs | {} fragments | union_coverage={:.1}% | \
             total_tracks_in_storage={}",
            self.traj_id,
            self.n_gt_alerts,
            self.n_fragments(),
            self.union_coverage() * 100.0,
            self.total_tracks_in_storage,
        )?;
        for (i, frag) in self.fragments.iter().enumerate() {
            writeln!(
                f,
                "  [{i}] {:<30}  shared={:>3}  total={:>3}  \
                 purity={:.0}%  gt_cov={:.1}%",
                frag.track_id,
                frag.n_shared,
                frag.n_track_total,
                frag.purity * 100.0,
                frag.gt_coverage * 100.0,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Collect all `(track_id, dia_source_id)` pairs from every
/// `track_members-nid=*.parquet` file under `orbits_dir`.
///
/// Returns a map `track_id → set-of-dia_source_ids`.
fn collect_all_track_members(orbits_dir: &Path) -> HashMap<String, HashSet<u64>> {
    let mut result: HashMap<String, HashSet<u64>> = HashMap::new();

    if !orbits_dir.exists() {
        return result;
    }

    for entry in std::fs::read_dir(orbits_dir)
        .unwrap_or_else(|e| panic!("cannot read orbits dir {orbits_dir:?}: {e}"))
        .flatten()
    {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with("track_members-nid=") || !name.ends_with(".parquet") {
            continue;
        }

        let utf8_path = Utf8Path::from_path(&path)
            .unwrap_or_else(|| panic!("non-UTF-8 parquet path: {path:?}"));
        let (_, batches) =
            load_parquet(utf8_path).unwrap_or_else(|e| panic!("cannot load {utf8_path}: {e:?}"));

        for batch in &batches {
            let track_ids = batch
                .column_by_name("track_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .expect("track_members must have a 'track_id' Utf8 column");
            let source_ids = batch
                .column_by_name("dia_source_id")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .expect("track_members must have a 'dia_source_id' UInt64 column");

            for i in 0..batch.num_rows() {
                result
                    .entry(track_ids.value(i).to_owned())
                    .or_default()
                    .insert(source_ids.value(i));
            }
        }
    }

    result
}

/// Load all `dia_source_id` values belonging to `traj_id` from the fixture
/// parquet files in `fixture_night_dir`.
fn ground_truth_ids(fixture_night_dir: &Path, traj_id: i32) -> HashSet<u64> {
    let mut ids = HashSet::new();

    for entry in std::fs::read_dir(fixture_night_dir)
        .unwrap_or_else(|e| panic!("cannot read fixture nights dir {fixture_night_dir:?}: {e}"))
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }

        let utf8_path = Utf8Path::from_path(&path)
            .unwrap_or_else(|| panic!("non-UTF-8 fixture path: {path:?}"));
        let (_, batches) = load_parquet(utf8_path)
            .unwrap_or_else(|e| panic!("cannot load fixture {utf8_path}: {e:?}"));

        for batch in &batches {
            let traj_col = batch
                .column_by_name("trajectory_id")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
                .expect("fixture parquet must have a 'trajectory_id' Int32 column");
            let src_col = batch
                .column_by_name("dia_source_id")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .expect("fixture parquet must have a 'dia_source_id' UInt64 column");

            for i in 0..batch.num_rows() {
                if traj_col.value(i) == traj_id {
                    ids.insert(src_col.value(i));
                }
            }
        }
    }

    ids
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyse how the pipeline reconstructed ground-truth trajectory `traj_id`.
///
/// Scans all `track_members-nid=*.parquet` files under `storage_root/orbits/`
/// and collects every track that shares at least one observation with `traj_id`.
/// Returns a [`ReconstructionResult`] with per-fragment purity, coverage, and
/// aggregate union coverage.
///
/// # Arguments
///
/// * `storage_root` — root of the pipeline persistence directory.
/// * `fixture_night_dir` — directory containing fixture parquet files with a
///   `trajectory_id` column for ground-truth lookup.
/// * `traj_id` — ground-truth trajectory identifier to analyse.
pub fn check_reconstruction(
    storage_root: &Path,
    fixture_night_dir: &Path,
    traj_id: i32,
) -> ReconstructionResult {
    let gt_ids = ground_truth_ids(fixture_night_dir, traj_id);
    let n_gt = gt_ids.len();

    if n_gt == 0 {
        panic!("trajectory_id={traj_id} not found in any fixture parquet file");
    }

    let orbits_dir = storage_root.join("orbits");
    let all_tracks = collect_all_track_members(&orbits_dir);
    let total_tracks = all_tracks.len();

    let mut fragments: Vec<FragmentResult> = all_tracks
        .into_iter()
        .filter_map(|(track_id, track_ids)| {
            let n_shared = gt_ids.intersection(&track_ids).count();
            if n_shared == 0 {
                return None;
            }
            let n_track_total = track_ids.len();
            Some(FragmentResult {
                purity: n_shared as f64 / n_track_total as f64,
                gt_coverage: n_shared as f64 / n_gt as f64,
                track_id,
                n_shared,
                n_track_total,
            })
        })
        .collect();

    // Sort: highest purity first, then highest gt_coverage.
    fragments.sort_by(|a, b| {
        b.purity
            .partial_cmp(&a.purity)
            .unwrap()
            .then(b.gt_coverage.partial_cmp(&a.gt_coverage).unwrap())
    });

    ReconstructionResult {
        traj_id,
        n_gt_alerts: n_gt,
        fragments,
        total_tracks_in_storage: total_tracks,
    }
}

use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    tracklet::{
        generate_candidate::materialize_contiguous_night,
        track_storage::{TrackId, TrackStorage},
    },
};
use photom::{
    NightId,
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::{ObsDataset, observation::Observation},
    observer::error_model::ObsErrorModel,
};

use polars::prelude::{DataFrame, LazyFrame, ScanArgsParquet};

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

pub fn load_data(parquet_path: impl AsRef<Utf8Path>) -> (DataFrame, ObsDataset) {
    let path = parquet_path.as_ref().as_str();
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(path.into(), args).expect("scan_parquet must succeed");
    let obs_dataset = ObsDataset::from_lazy(
        lf.clone(),
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )
    .expect("from_lazy must succeed for int file");

    let df = lf.collect().expect("collect must succeed");
    (df, obs_dataset)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// FINK-FAT: Fink Asteroid Tracker
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: Utf8PathBuf,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,
}

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig> {
    load_engine_config_validated(config_path).context("failed to load engine config")
}

// ---------------------------------------------------------------------------
// DataFrame helpers
// ---------------------------------------------------------------------------

fn build_id_to_traj_id(df: &DataFrame) -> HashMap<u64, u32> {
    let ids = df.column("id").unwrap().u64().unwrap();
    let traj_ids = df.column("traj_id").unwrap().u32().unwrap();

    ids.into_iter()
        .zip(traj_ids.into_iter())
        .filter_map(|(id, traj)| Some((id?, traj?)))
        .collect()
}

/// Compute the set of `traj_id`s that have at least `min_alerts` observations
/// in a given night.
///
/// Asteroids below this threshold cannot be seeded and are excluded from
/// recall computation.
///
/// Arguments
/// ---------
/// * `df`         – Full alert DataFrame containing `id`, `traj_id` and
///   `night_id` columns.
/// * `night`      – Night identifier to filter on.
/// * `min_alerts` – Minimum number of alerts required (typically 2).
///
/// Return
/// ------
/// Set of `traj_id`s satisfying the threshold.
fn seedable_traj_ids(df: &DataFrame, night: u32, min_alerts: usize) -> HashSet<u32> {
    let nights = df.column("night_id").unwrap().u32().unwrap();
    let ids = df.column("id").unwrap().u64().unwrap();
    let traj_ids = df.column("traj_id").unwrap().u32().unwrap();

    let mut counts: HashMap<u32, usize> = HashMap::new();

    for ((n, _id), tid) in nights
        .into_iter()
        .zip(ids.into_iter())
        .zip(traj_ids.into_iter())
    {
        if let (Some(n), Some(tid)) = (n, tid) {
            if n == night {
                *counts.entry(tid).or_default() += 1;
            }
        }
    }

    counts
        .into_iter()
        .filter(|(_, count)| *count >= min_alerts)
        .map(|(tid, _)| tid)
        .collect()
}

// ---------------------------------------------------------------------------
// Diagnostics types
// ---------------------------------------------------------------------------

/// Detailed diagnostics for association quality analysis.
struct AssociationDiagnostics {
    precision: f64,
    recall: f64,
    tp: usize,
    fp: usize,
    /// Number of recoverable asteroids:
    /// seedable in the previous night (≥2 alerts) **and** present in the next night.
    n_recoverable: usize,
    /// Number of recovered asteroids (at least one correct association).
    n_recovered: usize,
    /// Recoverable asteroids that were never associated (missed entirely).
    missed_traj_ids: HashSet<u32>,
    /// Recoverable asteroids whose tracklet had no candidates at all after association.
    not_in_any_tracklet: HashSet<u32>,
    /// Associations per tracklet (number of candidate alerts).
    candidates_per_tracklet: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Diagnostics computation
// ---------------------------------------------------------------------------

/// Evaluate association quality between two consecutive nights.
///
/// Computes precision, recall, and auxiliary diagnostics for the candidate
/// associations produced by the Kalman-based linking step.
///
/// Recall definition
/// -----------------
/// Recall is computed over **recoverable** asteroids only:
///
/// $$\text{Recall} = \frac{|\text{recovered}|}{|\text{recoverable}|}$$
///
/// where *recoverable* is defined as:
///
/// $$\text{recoverable} = \text{seedable}_{\text{prev}} \cap \text{present}_{\text{next}}$$
///
/// Arguments
/// ---------
/// * `id_to_traj`          – Global alert-id → traj-id lookup table.
/// * `storage`             – Active tracklet storage (used to resolve
///   `TrackId` → `Tracklet`).
/// * `associations`        – Output of the linking step: pairs of a
///   `TrackId` and its candidate observations in the next night.
/// * `next_night`          – All observations from the next night.
/// * `seedable_prev_night` – Set of `traj_id`s that have ≥2 alerts in the
///   previous night, as produced by [`seedable_traj_ids`].
///
/// Return
/// ------
/// An [`AssociationDiagnostics`] struct populated with precision, recall,
/// TP/FP counts, and per-tracklet candidate distributions.
fn evaluate_associations_detailed(
    id_to_traj: &HashMap<u64, u32>,
    storage: &TrackStorage,
    associations: &[(TrackId, Vec<&Observation>)],
    next_night: &[Observation],
    seedable_prev_night: &HashSet<u32>,
) -> AssociationDiagnostics {
    // Traj-ids present in the next night.
    let next_traj_ids: HashSet<u32> = next_night
        .iter()
        .filter_map(|obs| id_to_traj.get(obs.id()).copied())
        .collect();

    // Recoverable = seedable in prev night AND present in next night.
    let recoverable: HashSet<u32> = seedable_prev_night
        .intersection(&next_traj_ids)
        .copied()
        .collect();

    // Traj-ids that have at least one tracklet in the association output.
    let traj_ids_with_tracklet: HashSet<u32> = associations
        .iter()
        .filter_map(|(id, _)| {
            let tracklet = storage.get(*id)?;
            tracklet
                .obs_keys()
                .iter()
                .find_map(|k| id_to_traj.get(k))
                .copied()
        })
        .collect();

    // Recoverable asteroids for which no tracklet was built at all.
    let not_in_any_tracklet: HashSet<u32> = recoverable
        .difference(&traj_ids_with_tracklet)
        .copied()
        .collect();

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut recovered: HashSet<u32> = HashSet::new();
    let mut candidates_per_tracklet: Vec<usize> = Vec::with_capacity(associations.len());

    for (id, candidates) in associations {
        candidates_per_tracklet.push(candidates.len());

        let tracklet_tid = storage
            .get(*id)
            .and_then(|t| t.obs_keys().iter().find_map(|k| id_to_traj.get(k)).copied());

        let Some(tracklet_tid) = tracklet_tid else {
            fp += candidates.len();
            continue;
        };

        for obs in candidates {
            match id_to_traj.get(obs.id()) {
                Some(&obs_tid) if obs_tid == tracklet_tid => {
                    tp += 1;
                    recovered.insert(tracklet_tid);
                }
                _ => {
                    fp += 1;
                }
            }
        }
    }

    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };

    let recall = if recoverable.is_empty() {
        0.0
    } else {
        recovered.len() as f64 / recoverable.len() as f64
    };

    let missed_traj_ids: HashSet<u32> = recoverable.difference(&recovered).copied().collect();

    AssociationDiagnostics {
        precision,
        recall,
        tp,
        fp,
        n_recoverable: recoverable.len(),
        n_recovered: recovered.len(),
        missed_traj_ids,
        not_in_any_tracklet,
        candidates_per_tracklet,
    }
}

// ---------------------------------------------------------------------------
// Diagnostics printing
// ---------------------------------------------------------------------------

fn print_diagnostics(
    diag: &AssociationDiagnostics,
    first_night: NightId,
    next_night: NightId,
    id_to_traj: &HashMap<u64, u32>,
    storage: &TrackStorage,
    associations: &[(TrackId, Vec<&Observation>)],
    prev_night_obs: &[Observation],
    next_night_obs: &[Observation],
) {
    let total_candidates: usize = diag.candidates_per_tracklet.iter().sum();
    let mut sorted = diag.candidates_per_tracklet.clone();
    sorted.sort_unstable();

    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
    } else {
        sorted[sorted.len() / 2] as f64
    };

    let mean = if sorted.is_empty() {
        0.0
    } else {
        total_candidates as f64 / sorted.len() as f64
    };

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Association diagnostics: {first_night} → {next_night}          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!();
    println!("── Global metrics ──────────────────────────────────────────────");
    println!(
        "  Precision : {:.4}  (TP={}, FP={})",
        diag.precision, diag.tp, diag.fp
    );
    println!(
        "  Recall    : {:.4}  (recovered={}/{} recoverable)",
        diag.recall, diag.n_recovered, diag.n_recoverable
    );

    println!();
    println!("── Candidates per tracklet ─────────────────────────────────────");
    println!("  Tracklets with ≥1 candidate : {}", associations.len());
    println!("  Total candidate pairs        : {total_candidates}");
    println!("  Mean candidates / tracklet   : {mean:.2}");
    println!("  Median                       : {median:.1}");
    println!(
        "  Min / Max                    : {} / {}",
        sorted.first().copied().unwrap_or(0),
        sorted.last().copied().unwrap_or(0)
    );

    let bucket_1 = sorted.iter().filter(|&&x| x == 1).count();
    let bucket_2_5 = sorted.iter().filter(|&&x| (2..=5).contains(&x)).count();
    let bucket_6_20 = sorted.iter().filter(|&&x| (6..=20).contains(&x)).count();
    let bucket_gt20 = sorted.iter().filter(|&&x| x > 20).count();
    println!("  Distribution:");
    println!("    = 1 candidate   : {bucket_1}");
    println!("    2–5 candidates  : {bucket_2_5}");
    println!("    6–20 candidates : {bucket_6_20}");
    println!("    > 20 candidates : {bucket_gt20}");

    println!();
    println!("── Recall breakdown ────────────────────────────────────────────");
    println!(
        "  Recoverable but missed entirely     : {}",
        diag.missed_traj_ids.len()
    );
    println!(
        "  Of which: no tracklet built at all  : {}",
        diag.not_in_any_tracklet.len()
    );
    let missed_with_tracklet = diag.missed_traj_ids.len() - diag.not_in_any_tracklet.len();
    println!("  Of which: tracklet built but no match found : {missed_with_tracklet}");

    // Alerts per recoverable asteroid in the next night.
    let prev_traj_ids: HashSet<u32> = prev_night_obs
        .iter()
        .filter_map(|obs| id_to_traj.get(obs.id()).copied())
        .collect();
    let next_traj_ids: HashSet<u32> = next_night_obs
        .iter()
        .filter_map(|obs| id_to_traj.get(obs.id()).copied())
        .collect();
    let recoverable: HashSet<u32> = prev_traj_ids
        .intersection(&next_traj_ids)
        .copied()
        .collect();

    let mut alerts_per_traj: HashMap<u32, usize> = HashMap::new();
    for obs in next_night_obs {
        if let Some(&tid) = id_to_traj.get(obs.id()) {
            if recoverable.contains(&tid) {
                *alerts_per_traj.entry(tid).or_default() += 1;
            }
        }
    }
    let single_alert_asteroids = alerts_per_traj.values().filter(|&&n| n == 1).count();

    println!();
    println!("── Next-night alert coverage (recoverable asteroids) ───────────");
    println!(
        "  Recoverable asteroids with only 1 alert in night {next_night}: {single_alert_asteroids}"
    );
    println!("  (these are structurally harder to associate due to no redundancy)");

    // Seeding coverage.
    let traj_ids_with_tracklet: HashSet<u32> = storage
        .iter_tracklets()
        .filter_map(|t| t.obs_keys().iter().find_map(|k| id_to_traj.get(k)).copied())
        .collect();

    let seedable_but_missed = recoverable
        .iter()
        .filter(|tid| {
            !traj_ids_with_tracklet.contains(tid)
                && alerts_per_traj.get(tid).copied().unwrap_or(0) >= 2
        })
        .count();

    let structurally_impossible = recoverable
        .iter()
        .filter(|tid| {
            !traj_ids_with_tracklet.contains(tid)
                && alerts_per_traj.get(tid).copied().unwrap_or(0) < 2
        })
        .count();

    println!();
    println!("── Seeding coverage (recoverable asteroids) ────────────────────");
    println!(
        "  With tracklet in night {first_night}              : {}",
        traj_ids_with_tracklet.len()
    );
    println!("  Seedable (≥2 alerts) but no tracklet built : {seedable_but_missed}  ← seeding gap");
    println!("  Structurally impossible (< 2 alerts)       : {structurally_impossible}");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    let (df, obs_dataset) = load_data(&cli.alerts);

    let mut night_ids: Vec<_> = obs_dataset.iter_night_id().unwrap().collect();
    night_ids.sort();

    let engine_config = load_config(&cli.config)?;

    let first_night = NightId(2925);
    let first_night_obs = materialize_contiguous_night(&obs_dataset, &first_night)?;

    let (night_tracklets, nb_new_tracklet) = TrackStorage::new()
        .generate_seeds(&engine_config, first_night_obs)
        .unwrap();

    println!(
        "For night {first_night}, generated {} tracklets",
        nb_new_tracklet,
    );

    // Associate tracklets to the next night's alerts.
    let next_night = NightId(2926);
    let next_night_obs = materialize_contiguous_night(&obs_dataset, &next_night)?;

    let next_night_associations =
        night_tracklets.generate_candidates(next_night_obs, &engine_config)?;

    println!(
        "Association complete: {} tracklets matched to at least one alert in night {next_night}",
        next_night_associations.len()
    );

    let id_to_traj = build_id_to_traj_id(&df);
    let seedable_prev_night = seedable_traj_ids(&df, first_night.0, 2);

    let prev_night_obs = materialize_contiguous_night(&obs_dataset, &first_night)?;

    let diag = evaluate_associations_detailed(
        &id_to_traj,
        &night_tracklets,
        &next_night_associations,
        &next_night_obs,
        &seedable_prev_night,
    );

    print_diagnostics(
        &diag,
        first_night,
        next_night,
        &id_to_traj,
        &night_tracklets,
        &next_night_associations,
        &prev_night_obs,
        &next_night_obs,
    );

    Ok(())
}

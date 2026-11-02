//! Gate **selectivity** of the cross-night association, broken down by
//! nights-since-seed and by the seed object's orbital-class population.
//!
//! Each [`GateRecord`](fink_fat_engine::topocentric_kf::branching::orchestrate::GateRecord)
//! lists the observations that passed one existing lineage's gate on one
//! visit. Against ground truth we split them into RIGHT (belong to the
//! lineage's own object) and WRONG (belong to another object — contamination
//! candidates). Aggregated per `(nights_since_seed, population)`, the
//! wrong-fraction is the contamination rate; `nights_since_seed == 1` is the
//! first cross-night association after seeding — the hotspot the radius
//! decomposition pointed at. Population lets us see whether exotic (non-MBA)
//! objects are contaminated more than the MBA majority.

use ahash::AHashMap;
use fink_fat_engine::topocentric_kf::branching::{Branch, orchestrate::GateRecord};
use photom::TrajId;

use crate::{
    population::Population,
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
};

/// Longest nights-since-seed tracked individually; deeper links fold into the
/// overflow bucket.
///
/// Deep enough to locate the knee rather than hide it. At depth 6 the overflow
/// bucket held 1.56 M records at a 76 % wrong-fraction — which says the rot is
/// somewhere past five nights, but not where, and "where" is exactly what sets
/// a lifetime cap. Contaminating lineages measured a mean staleness of 11 to 24
/// nights, so the interesting range runs well beyond six.
const BUCKET_DEPTH: usize = 20;

/// A lineage counts as "coasting" once it has gone this many nights without a
/// real update — the population `purge_stale_lineages` would consider.
const COASTING_MIN_NIGHTS: usize = 2;

/// Coarse class of a lineage for the LLR-floor diagnostic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LineageClass {
    /// Pure lineage whose object is a rare/exotic population (NEO/Centaur/KBO/SDO).
    Exotic,
    /// Pure lineage whose object is main-belt-ish (MBA / Cybele-Hilda-Trojan).
    Belt,
    /// Mixed, unknown-population, or otherwise not a clean single real object —
    /// the zombie/noise class we want to cull.
    Noise,
}

impl LineageClass {
    fn label(self) -> &'static str {
        match self {
            LineageClass::Exotic => "exotic (NEO/Centaur/KBO/SDO)",
            LineageClass::Belt => "belt (MBA/Cyb/Hild/Troj)",
            LineageClass::Noise => "noise / mixed / unknown",
        }
    }
}

/// Percentiles reported for each LLR distribution — enough to see whether two
/// classes truly separate or just differ in median while their tails overlap
/// (the case that makes a single scalar floor unsafe).
const PERCENTILES: [(f64, &str); 5] = [
    (0.10, "p10"),
    (0.25, "p25"),
    (0.50, "p50"),
    (0.75, "p75"),
    (0.90, "p90"),
];

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_percentile_row(label: &str, xs: &[f64]) {
    if xs.is_empty() {
        println!("    {label:<12} (no finite samples)");
        return;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print!("    {label:<12}");
    for (q, name) in PERCENTILES {
        print!(" {name}={:>10.3}", percentile(&sorted, q));
    }
    println!(" (n={})", xs.len());
}

/// Print the `cumulative_llr` distribution of **coasting** lineages, split by
/// class — the diagnostic for choosing `stale_llr_floor`. Reports BOTH the
/// raw `cumulative_llr` and the **per-real-update normalized** LLR
/// (`cumulative_llr / n_real_updates`).
///
/// The raw sum conflates a lineage's *coasting duration* (more null-branch
/// penalties accumulated) with the *quality* of its evidence — a
/// long-coasting real object and a long-coasting zombie can land at similar
/// raw values purely from arc length. Normalizing by the number of real
/// updates removes that confound, at the cost of being noisier for lineages
/// with very few updates (n=1 bootstrap-only lineages are nearly all raw
/// signal either way). Percentiles (not just min/median/max) matter here: if
/// two classes' medians differ but their p10/p90 overlap heavily, no single
/// `stale_llr_floor` is safe on that metric.
///
/// Computed once on the final collection: each lineage is reduced to its
/// best-LLR branch (class, `cumulative_llr`, `n_real_updates`) and freshest
/// `last_real_update_step` (coasting age = `current_step − freshest`).
pub fn print_coasting_llr_by_class(
    branches: &[Branch<'_, '_>],
    ground_truth: &ObsTrajLookup,
    traj_population: &AHashMap<TrajId, Population>,
    current_step: usize,
) {
    // Group branches by lineage: best-LLR branch (for class/llr/n_updates) and
    // freshest real update (for coasting age).
    struct Agg {
        best_llr: f64,
        best_n_updates: usize,
        best_track: Vec<photom::observation_dataset::ObsId>,
        freshest_update: usize,
    }
    let mut by_lineage: AHashMap<u64, Agg> = AHashMap::default();
    for b in branches {
        let e = by_lineage.entry(b.lineage_id).or_insert_with(|| Agg {
            best_llr: f64::NEG_INFINITY,
            best_n_updates: 1,
            best_track: Vec::new(),
            freshest_update: 0,
        });
        e.freshest_update = e.freshest_update.max(b.last_real_update_step);
        // `>` so a NaN best_llr is replaced by any finite one (NaN comparisons
        // are false), matching how `purge_stale_lineages` treats the group.
        if b.cumulative_llr > e.best_llr || e.best_track.is_empty() {
            e.best_llr = b.cumulative_llr;
            e.best_n_updates = b.n_real_updates;
            e.best_track = b.track_ids().to_vec();
        }
    }

    // Per class: finite raw + normalized LLR samples, plus non-finite count,
    // restricted to coasting lineages.
    let mut raw: AHashMap<u8, Vec<f64>> = AHashMap::default();
    let mut normalized: AHashMap<u8, Vec<f64>> = AHashMap::default();
    let mut nonfinite: AHashMap<u8, usize> = AHashMap::default();
    let mut total: AHashMap<u8, usize> = AHashMap::default();
    for agg in by_lineage.values() {
        if current_step.saturating_sub(agg.freshest_update) < COASTING_MIN_NIGHTS {
            continue; // not coasting — not a purge candidate
        }
        let class = match ground_truth.classify(&agg.best_track) {
            SeedPurity::Pure(traj) => match traj_population.get(&traj).copied() {
                Some(Population::Neo)
                | Some(Population::Centaur)
                | Some(Population::Kbo)
                | Some(Population::Sdo) => LineageClass::Exotic,
                Some(Population::Mba) | Some(Population::MidOuter) => LineageClass::Belt,
                _ => LineageClass::Noise, // Unknown population
            },
            SeedPurity::Mixed | SeedPurity::Unknown => LineageClass::Noise,
        };
        let key = class as u8;
        *total.entry(key).or_insert(0) += 1;
        if agg.best_llr.is_finite() {
            raw.entry(key).or_default().push(agg.best_llr);
            normalized
                .entry(key)
                .or_default()
                .push(agg.best_llr / agg.best_n_updates.max(1) as f64);
        } else {
            *nonfinite.entry(key).or_insert(0) += 1;
        }
    }

    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!(
        "[Coasting-lineage LLR by class] lineages coasting >= {COASTING_MIN_NIGHTS} nights \
         (choose stale_llr_floor where exotic and noise separate)"
    );
    println!("{sep}");
    for class in [
        LineageClass::Exotic,
        LineageClass::Belt,
        LineageClass::Noise,
    ] {
        let key = class as u8;
        let n = total.get(&key).copied().unwrap_or(0);
        if n == 0 {
            continue;
        }
        let nf = nonfinite.get(&key).copied().unwrap_or(0);
        println!(
            "  {} — n={n}, non-finite={nf} ({:.0}%)",
            class.label(),
            100.0 * nf as f64 / n as f64,
        );
        print_percentile_row("raw llr", raw.get(&key).map(Vec::as_slice).unwrap_or(&[]));
        print_percentile_row(
            "llr/update",
            normalized.get(&key).map(Vec::as_slice).unwrap_or(&[]),
        );
    }
    println!(
        "  (raw = cumulative_llr; llr/update = cumulative_llr / n_real_updates — normalizes \
         away arc-length/coasting-duration bias. A safe stale_llr_floor sits above the noise \
         p90 and below the exotic p10 on whichever metric separates the classes.)"
    );
    println!("{sep}");
}

#[derive(Default, Clone, Copy)]
struct Cell {
    n_lineages: usize,
    sum_gated: usize,
    sum_wrong: usize,
    /// Only used by the bank-width breakdown, to show how mature the lineages
    /// in each width band are.
    sum_n_steps: usize,
    sum_right: usize,
}

/// Accumulates gate-selectivity over the whole run, keyed by
/// `(nights_since_seed bucket, population)`.
#[derive(Default)]
pub struct GateSelectivity {
    // (bucket_index 0-based capped at BUCKET_DEPTH, population) -> Cell
    cells: AHashMap<(usize, Population), Cell>,
    /// The same gate records keyed by bank width instead of by age.
    by_bank_width: AHashMap<usize, Cell>,
}

impl GateSelectivity {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold in one night's gate records. `branches` is the *surviving*
    /// collection after the night (used to map each `lineage_id` to its pure
    /// ground-truth object); records for lineages that aren't purely one
    /// object are skipped (nothing to call "wrong" against).
    pub fn observe_night(
        &mut self,
        gate_records: &[GateRecord],
        branches: &[Branch<'_, '_>],
        ground_truth: &ObsTrajLookup,
        traj_population: &AHashMap<TrajId, Population>,
    ) {
        // lineage_id -> its pure seed trajectory, from the surviving branches.
        let mut lineage_traj: AHashMap<u64, TrajId> = AHashMap::default();
        for b in branches {
            if let SeedPurity::Pure(traj) = ground_truth.classify(b.track_ids()) {
                lineage_traj.entry(b.lineage_id).or_insert(traj);
            }
        }

        for rec in gate_records {
            let Some(seed_traj) = lineage_traj.get(&rec.lineage_id) else {
                continue;
            };
            let pop = traj_population
                .get(seed_traj)
                .copied()
                .unwrap_or(Population::Unknown);
            let bucket = rec.nights_since_seed.min(BUCKET_DEPTH);

            let mut right = 0usize;
            let mut wrong = 0usize;
            for &obs in &rec.gated_obs_ids {
                match ground_truth.traj_of(obs) {
                    Some(t) if t == seed_traj => right += 1,
                    Some(_) => wrong += 1,
                    None => {} // no ground truth for this candidate — neither
                }
            }
            let cell = self.cells.entry((bucket, pop)).or_default();
            cell.n_lineages += 1;
            cell.sum_gated += rec.gated_obs_ids.len();
            cell.sum_right += right;
            cell.sum_wrong += wrong;

            // Same records, indexed by how many hypotheses the bank was
            // carrying instead of by age — age is a proxy, bank width is the
            // mechanism.
            let by_width = self
                .by_bank_width
                .entry(bank_width_bucket(rec.n_hypotheses))
                .or_default();
            by_width.n_lineages += 1;
            by_width.sum_gated += rec.gated_obs_ids.len();
            by_width.sum_right += right;
            by_width.sum_wrong += wrong;
            by_width.sum_n_steps += rec.n_steps;
        }
    }

    fn label_bucket(bucket: usize) -> String {
        if bucket >= BUCKET_DEPTH {
            format!("{BUCKET_DEPTH}+")
        } else {
            bucket.to_string()
        }
    }

    /// Print the first-cross-night-link table (per population) plus the
    /// nights-since-seed decay (all populations).
    pub fn print_summary(&self) {
        let sep = "=".repeat(90);
        println!("\n{sep}");
        println!(
            "[Gate selectivity] cross-night association — RIGHT = lineage's own object, \
             WRONG = other object"
        );
        println!("{sep}");
        if self.cells.is_empty() {
            println!(
                "  (no gate records — run in night-by-night mode with ground truth to populate)"
            );
            println!("{sep}");
            return;
        }

        // (1) First cross-night link (nights_since_seed == 1), per population.
        println!("  First cross-night link (nights_since_seed = 1), by population:");
        println!(
            "    {:<20} {:>10} {:>10} {:>10} {:>12}",
            "population", "lineages", "gated/lin", "wrong/lin", "wrong-frac"
        );
        for pop in Population::all() {
            if let Some(c) = self.cells.get(&(1, pop)) {
                if c.n_lineages == 0 {
                    continue;
                }
                println!(
                    "    {:<20} {:>10} {:>10.2} {:>10.2} {:>11.1}%",
                    pop.label(),
                    c.n_lineages,
                    c.sum_gated as f64 / c.n_lineages as f64,
                    c.sum_wrong as f64 / c.n_lineages as f64,
                    100.0 * c.sum_wrong as f64 / c.sum_gated.max(1) as f64,
                );
            }
        }

        // (2) Decay with nights-since-seed, summed over populations.
        println!("  By nights-since-seed (all populations):");
        println!(
            "    {:>6} {:>10} {:>10} {:>10} {:>12}",
            "nights", "lineages", "gated/lin", "wrong/lin", "wrong-frac"
        );
        for bucket in 1..=BUCKET_DEPTH {
            let mut agg = Cell::default();
            for pop in Population::all() {
                if let Some(c) = self.cells.get(&(bucket, pop)) {
                    agg.n_lineages += c.n_lineages;
                    agg.sum_gated += c.sum_gated;
                    agg.sum_wrong += c.sum_wrong;
                    agg.sum_right += c.sum_right;
                }
            }
            if agg.n_lineages == 0 {
                continue;
            }
            println!(
                "    {:>6} {:>10} {:>10.2} {:>10.2} {:>11.1}%",
                Self::label_bucket(bucket),
                agg.n_lineages,
                agg.sum_gated as f64 / agg.n_lineages as f64,
                agg.sum_wrong as f64 / agg.n_lineages as f64,
                100.0 * agg.sum_wrong as f64 / agg.sum_gated.max(1) as f64,
            );
        }

        // (3) What a hard lifetime cap would buy and what it would cost.
        //
        // A cap at N removes every lineage that has gone N nights without a
        // real update, so it forgoes *all* of their associations — the wrong
        // ones and the right ones alike. Reading one column without the other
        // is how a cap gets set too aggressively.
        //
        // The trade is favourable in a way the raw counts understate: a purged
        // lineage is archived, so its arc survives and the object is re-seeded
        // later. Contamination is traded for fragmentation, and a fragmented
        // *pure* trajectory still counts toward `completeness (pure-only)`.
        println!("\n  What a hard staleness cap would prevent (all populations):");
        println!(
            "    {:>6} {:>14} {:>14} {:>12} {:>16}",
            "cap N", "wrong avoided", "right lost", "ratio", "% of all wrong"
        );
        let mut per_bucket: Vec<(usize, usize, usize)> = Vec::new();
        for bucket in 1..=BUCKET_DEPTH {
            let (mut right, mut wrong) = (0usize, 0usize);
            for pop in Population::all() {
                if let Some(c) = self.cells.get(&(bucket, pop)) {
                    right += c.sum_right;
                    wrong += c.sum_wrong;
                }
            }
            per_bucket.push((bucket, right, wrong));
        }
        let total_wrong: usize = per_bucket.iter().map(|(_, _, w)| *w).sum();
        for &(cap, _, _) in &per_bucket {
            let (right_lost, wrong_avoided): (usize, usize) = per_bucket
                .iter()
                .filter(|(b, _, _)| *b >= cap)
                .fold((0, 0), |(r, w), (_, br, bw)| (r + br, w + bw));
            if wrong_avoided == 0 && right_lost == 0 {
                continue;
            }
            println!(
                "    {:>6} {:>14} {:>14} {:>12} {:>15.1}%",
                Self::label_bucket(cap),
                wrong_avoided,
                right_lost,
                if right_lost > 0 {
                    format!("{:.2}", wrong_avoided as f64 / right_lost as f64)
                } else {
                    "inf".to_string()
                },
                100.0 * wrong_avoided as f64 / total_wrong.max(1) as f64,
            );
        }

        // (4) The same records keyed by bank width. A search region is the
        // union of every live hypothesis's predicted position, so a wide bank
        // projects a wide region and a wide region admits other objects. If the
        // wrong-fraction tracks this more tightly than it tracks age, the lever
        // is hypothesis retention — mechanical and fixable — rather than
        // lineage lifetime.
        println!("\n  By hypotheses in the bank at gate time (all populations):");
        println!(
            "    {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
            "hypotheses", "lineages", "gated/lin", "wrong/lin", "wrong-frac", "n_steps"
        );
        let mut widths: Vec<usize> = self.by_bank_width.keys().copied().collect();
        widths.sort_unstable();
        for w in widths {
            let c = &self.by_bank_width[&w];
            if c.n_lineages == 0 {
                continue;
            }
            println!(
                "    {:>10} {:>10} {:>10.2} {:>10.2} {:>11.1}% {:>10.1}",
                label_width_bucket(w),
                c.n_lineages,
                c.sum_gated as f64 / c.n_lineages as f64,
                c.sum_wrong as f64 / c.n_lineages as f64,
                100.0 * c.sum_wrong as f64 / c.sum_gated.max(1) as f64,
                c.sum_n_steps as f64 / c.n_lineages as f64,
            );
        }
        println!("{sep}");
    }
}

/// Lower edge of the bank-width band a hypothesis count falls in.
///
/// Geometric rather than linear: the interesting contrast is between a bank
/// that has collapsed to a handful of hypotheses and one still carrying
/// dozens, not between 41 and 42.
fn bank_width_bucket(n_hypotheses: usize) -> usize {
    match n_hypotheses {
        0..=1 => 1,
        2..=4 => 2,
        5..=9 => 5,
        10..=24 => 10,
        25..=49 => 25,
        50..=99 => 50,
        _ => 100,
    }
}

fn label_width_bucket(lower: usize) -> String {
    match lower {
        1 => "1".to_string(),
        2 => "2-4".to_string(),
        5 => "5-9".to_string(),
        10 => "10-24".to_string(),
        25 => "25-49".to_string(),
        50 => "50-99".to_string(),
        _ => "100+".to_string(),
    }
}

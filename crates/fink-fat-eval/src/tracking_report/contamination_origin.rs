//! Where contaminated branches come from: bad seeds, or bad associations?
//!
//! # The question
//!
//! 68 676 of 201 500 branches end the run holding observations of more than one
//! object. A branch gets there in one of two ways, and they point at completely
//! unrelated parts of the pipeline:
//!
//! * **born mixed** — the seed itself already spanned several objects, so no
//!   association gate could have prevented it; the fault is in tracklet linking
//!   and seeding;
//! * **turned mixed** — the lineage was pure and then associated a foreign
//!   observation; the fault is in the association gate, and the bank's width at
//!   that moment says which regime it happened in.
//!
//! Without this split, "76 % of what old lineages gate is wrong" and "19.8 % of
//! trajectories are contaminated" are two numbers with no established link
//! between them — and acting on one without knowing it drives the other is how
//! an analysis goes into the wrong subsystem.
//!
//! # Why identity is taken *before* the night
//!
//! [`GateSelectivityStats`](super::gate_selectivity) maps `lineage_id` to a
//! ground-truth object using the branches that **survived** the night, so a
//! lineage that just turned mixed is skipped — precisely the event of interest.
//! Comparing the collection before and after the advance sidesteps that: the
//! lineage's pure identity is still available on the "before" side, so the
//! transition can be attributed instead of dropped.

use ahash::{AHashMap, AHashSet};

use fink_fat_engine::topocentric_kf::branching::{BranchCollection, orchestrate::GateRecord};

use crate::seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity};

/// Counts for one bank-width band.
#[derive(Default, Clone)]
struct WidthCell {
    n_transitions: usize,
    sum_n_steps: usize,
    sum_nights_since_seed: usize,
}

#[derive(Default)]
pub struct ContaminationOriginStats {
    /// Lineages that appeared already holding more than one object.
    n_born_mixed: usize,
    /// Lineages that appeared pure.
    n_born_pure: usize,
    /// Lineages that appeared unscorable (no ground truth for their arc).
    n_born_unknown: usize,
    /// Pure → mixed transitions, keyed by the bank width when the gate ran.
    by_width: AHashMap<usize, WidthCell>,
    /// Transitions for which no gate record was found — the lineage turned
    /// mixed without going through `spawn_branches_for_lineage` this night
    /// (e.g. it was rebuilt from a different path). Reported rather than
    /// folded in, so the attribution's own coverage is visible.
    n_transitions_unattributed: usize,
}

impl ContaminationOriginStats {
    /// Fold in one advanced night.
    ///
    /// `gate_records` must be the records produced by the very advance that
    /// turned `prev` into `current`.
    pub fn observe_night(
        &mut self,
        prev: &BranchCollection<'_, '_>,
        current: &BranchCollection<'_, '_>,
        gate_records: &[GateRecord],
        ground_truth: &ObsTrajLookup,
    ) {
        let purity_of = |collection: &BranchCollection<'_, '_>| -> AHashMap<u64, SeedPurity> {
            let mut map = AHashMap::default();
            for b in &collection.branches {
                map.insert(b.lineage_id, ground_truth.classify(b.track_ids()));
            }
            map
        };
        let before = purity_of(prev);
        let after = purity_of(current);

        // Gate state per lineage this night. A lineage can appear more than
        // once; the widest bank is kept, being the one that opened the door.
        let mut gate_by_lineage: AHashMap<u64, &GateRecord> = AHashMap::default();
        for rec in gate_records {
            gate_by_lineage
                .entry(rec.lineage_id)
                .and_modify(|held| {
                    if rec.n_hypotheses > held.n_hypotheses {
                        *held = rec;
                    }
                })
                .or_insert(rec);
        }

        let previously_seen: AHashSet<u64> = before.keys().copied().collect();

        for (&lineage_id, purity) in &after {
            if !previously_seen.contains(&lineage_id) {
                match purity {
                    SeedPurity::Pure(_) => self.n_born_pure += 1,
                    SeedPurity::Mixed => self.n_born_mixed += 1,
                    SeedPurity::Unknown => self.n_born_unknown += 1,
                }
                continue;
            }

            // Only the pure → mixed step is a contamination *event*; a lineage
            // already mixed staying mixed adds nothing new.
            let was_pure = matches!(before.get(&lineage_id), Some(SeedPurity::Pure(_)));
            if !(was_pure && matches!(purity, SeedPurity::Mixed)) {
                continue;
            }

            match gate_by_lineage.get(&lineage_id) {
                Some(rec) => {
                    let cell = self
                        .by_width
                        .entry(width_bucket(rec.n_hypotheses))
                        .or_default();
                    cell.n_transitions += 1;
                    cell.sum_n_steps += rec.n_steps;
                    cell.sum_nights_since_seed += rec.nights_since_seed;
                }
                None => self.n_transitions_unattributed += 1,
            }
        }
    }

    fn n_transitions(&self) -> usize {
        self.by_width
            .values()
            .map(|c| c.n_transitions)
            .sum::<usize>()
            + self.n_transitions_unattributed
    }

    pub fn print_summary(&self) {
        let sep = "=".repeat(90);
        println!("\n{sep}");
        println!("[Contamination origin] was a mixed branch born that way, or did it turn?");
        println!("{sep}");

        let transitions = self.n_transitions();
        let born_total = self.n_born_pure + self.n_born_mixed + self.n_born_unknown;
        let attributable = self.n_born_mixed + transitions;

        println!("  Lineages created");
        let born_pct = |n: usize| 100.0 * n as f64 / born_total.max(1) as f64;
        println!(
            "  {:<34} {:>10}  ({:>5.1}%)",
            "born pure",
            self.n_born_pure,
            born_pct(self.n_born_pure)
        );
        println!(
            "  {:<34} {:>10}  ({:>5.1}%)   <- seeding, not association",
            "born MIXED",
            self.n_born_mixed,
            born_pct(self.n_born_mixed)
        );
        println!(
            "  {:<34} {:>10}  ({:>5.1}%)",
            "born unscorable",
            self.n_born_unknown,
            born_pct(self.n_born_unknown)
        );

        println!("\n  Contamination events");
        let share = |n: usize| 100.0 * n as f64 / attributable.max(1) as f64;
        println!(
            "  {:<34} {:>10}  ({:>5.1}%)",
            "seeded already mixed",
            self.n_born_mixed,
            share(self.n_born_mixed)
        );
        println!(
            "  {:<34} {:>10}  ({:>5.1}%)",
            "pure -> mixed on association",
            transitions,
            share(transitions)
        );
        if self.n_transitions_unattributed > 0 {
            println!(
                "  {:<34} {:>10}   (of the transitions, no gate record found)",
                "  unattributed", self.n_transitions_unattributed
            );
        }

        if transitions == 0 {
            println!("{sep}");
            return;
        }

        // Which regime the association happened in. If transitions concentrate
        // on wide banks, the never-confirmed-seed story holds; if they spread
        // evenly, it does not and the gate itself is at fault.
        println!("\n  Pure -> mixed transitions, by bank width at the gate");
        println!(
            "  {:<14} {:>12} {:>10} {:>12} {:>18}",
            "hypotheses", "transitions", "share", "mean n_steps", "mean nights stale"
        );
        let mut widths: Vec<usize> = self.by_width.keys().copied().collect();
        widths.sort_unstable();
        for w in widths {
            let c = &self.by_width[&w];
            if c.n_transitions == 0 {
                continue;
            }
            let n = c.n_transitions as f64;
            println!(
                "  {:<14} {:>12} {:>9.1}% {:>12.1} {:>18.1}",
                label_width(w),
                c.n_transitions,
                100.0 * n / transitions as f64,
                c.sum_n_steps as f64 / n,
                c.sum_nights_since_seed as f64 / n,
            );
        }
        println!("{sep}");
    }
}

/// Lower edge of the bank-width band, matching
/// [`gate_selectivity`](super::gate_selectivity)'s banding so the two tables
/// can be read against each other.
fn width_bucket(n_hypotheses: usize) -> usize {
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

fn label_width(lower: usize) -> &'static str {
    match lower {
        1 => "1",
        2 => "2-4",
        5 => "5-9",
        10 => "10-24",
        25 => "25-49",
        50 => "50-99",
        _ => "100+",
    }
}

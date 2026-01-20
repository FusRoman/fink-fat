// -----------------------------------------------------------------------------
// Frozen pairs generation
// -----------------------------------------------------------------------------

use std::collections::HashMap;

use camino::Utf8Path;
use fink_fat_engine::{
    engine_config::score_config::ScoreConfig, graph::score::ScoredEdge, night_id::NightId,
};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::io::{read_bin, write_bin};
use crate::night_seeds::{LabeledEdge, NightSeeds, SeedStore};
use crate::scoring::optimization_metrics::edge_item::EdgeItem;
use crate::scoring::optimization_metrics::metrics::Metrics;

/// Represents a frozen candidate edge between two seeds. The pair is identified
/// by the night IDs of the left and right seed and their indices within the
/// corresponding `NightSeeds::seeds` vectors. The `same` field encodes whether
/// the two seeds share the same truth ID (both truth IDs present and equal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenPair {
    /// Night ID of the left seed.
    pub a_nid: NightId,
    /// Night ID of the right seed.
    pub b_nid: NightId,
    /// Index of the left seed within its night.
    pub a_idx: usize,
    /// Index of the right seed within its night.
    pub b_idx: usize,
    /// True if both seeds have a matching truth ID.
    pub same: bool,
    /// Number of night steps between the two revisits (>= 1).
    pub delta: u32,
}

impl FrozenPair {
    pub fn compute_score_edge(
        &self,
        seed_store: &SeedStore,
        scoring_cfg: &ScoreConfig,
    ) -> Option<ScoredEdge> {
        let a_seeds = seed_store.get(&self.a_nid)?;
        let b_seeds = seed_store.get(&self.b_nid)?;

        let a_seed = &a_seeds.seeds[self.a_idx];
        let b_seed = &b_seeds.seeds[self.b_idx];

        let delta_revisit = self.b_nid.0.saturating_sub(self.a_nid.0);

        ScoredEdge::score(a_seed, b_seed, scoring_cfg, delta_revisit)
    }

    pub fn eval_cfg_on_frozen_pairs(
        store: &SeedStore,
        frozen: &[Self],
        cfg: &ScoreConfig,
    ) -> Vec<LabeledEdge> {
        let mut out: Vec<LabeledEdge> = Vec::with_capacity(frozen.len());

        for p in frozen {
            if let Some(edge) = p.compute_score_edge(store, cfg) {
                out.push(LabeledEdge {
                    same: p.same,
                    edge: edge,
                });
            }
        }
        out
    }

    /// Build a sampled set of frozen inter-night seed pairs up to a given horizon.
    ///
    /// Strategy (per (night, night+delta))
    /// -------------------------------
    /// 1) Enumerate **all true pairs**: all `(i, j)` where both seeds have a truth id
    ///    and `tid(i) == tid(j)`.
    /// 2) Sample **false pairs**: random `(i, j)` with `tid(i) != tid(j)` until reaching:
    ///    `target_false = round(false_to_true_ratio * n_true_pairs_for_this_night_pair)`.
    ///
    /// Arguments
    /// ---------
    /// * `horizon` – Maximum night separation to consider (in days / `NightId` units).
    /// * `false_to_true_ratio` – False pairs budget relative to the number of true pairs.
    /// * `seed` – Optional RNG seed for deterministic sampling.
    ///
    /// Return
    /// ------
    /// * `Vec<FrozenPair>` – Sampled frozen pairs across all (night, delta).
    ///
    /// Notes
    /// -----
    /// * Pairs are **directed**: `(n → n+delta)`.
    /// * `same=true` iff both endpoints have a truth id and they match.
    /// * Nights missing from the store for a given `(n, n+delta)` are skipped.
    pub fn frozen_pairs(
        seed_store: &SeedStore,
        cfg: &ScoreConfig,
        horizon: u32,
        false_to_true_ratio: f64,
        seed: Option<u64>,
    ) -> Vec<Self> {
        let mut out: Vec<FrozenPair> = Vec::new();

        // Deterministic iteration order.
        let mut night_ids: Vec<NightId> = seed_store.keys().copied().collect();
        night_ids.sort_unstable();

        // One RNG stream for the whole procedure (deterministic if seed is provided).
        let mut base_rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        for delta in 1..=horizon {
            let t0 = std::time::Instant::now();
            let before_out_len = out.len();

            println!("Generating frozen pairs for delta = {delta} ...");

            // Deterministic RNG split per delta (mirrors labeled_edges_by_delta style).
            let delta_seed =
                base_rng.next_u64() ^ (delta as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = StdRng::seed_from_u64(delta_seed);

            for &a_nid in night_ids.iter() {
                let b_nid = NightId(a_nid.0.saturating_add(delta));

                let Some(src) = seed_store.get(&a_nid) else {
                    continue;
                };
                let Some(dst) = seed_store.get(&b_nid) else {
                    continue;
                };

                // 1) First pass: accept ONLY true pairs that pass gates.
                let n_before_true = out.len();
                push_true_frozen_pairs_only_gated(a_nid, b_nid, src, dst, cfg, delta, &mut out);
                let n_new_true_accepted = out.len() - n_before_true;

                if n_new_true_accepted == 0 {
                    continue;
                }

                // 2) Second pass: sample false pairs that pass gates,
                // sized relative to accepted true pairs for this (night, delta).
                let target_false = ((false_to_true_ratio.max(0.0)) * (n_new_true_accepted as f64))
                    .round() as usize;

                sample_false_frozen_pairs_gated(
                    a_nid,
                    b_nid,
                    src,
                    dst,
                    cfg,
                    delta,
                    target_false,
                    &mut rng,
                    &mut out,
                );
            }

            println!(
                "  - delta = {delta}: generated {} frozen pairs in {:.2?}\n\n",
                out.len() - before_out_len,
                t0.elapsed()
            );
        }

        out
    }
    /// Evaluate a config on frozen pairs and return lightweight `EdgeItem`s only.
    ///
    /// This avoids allocating full `ScoredEdge`s and is intended for fast tuning.
    ///
    /// Parameters
    /// ----------
    /// * `store` – Seed store holding all `NightSeeds`.
    /// * `frozen` – Frozen pairs (indices + label).
    /// * `cfg` – Score config to evaluate.
    /// * `budget` – Optional prefix length to evaluate (deterministic multi-fidelity).
    ///
    /// Return
    /// ------
    /// * `Some((items, stats))` if at least one finite-cost edge was produced.
    /// * `None` if no usable edges were produced (unreachable / degenerate).
    pub fn eval_cfg_fast_items(
        store: &SeedStore,
        frozen: &[FrozenPair],
        cfg: &ScoreConfig,
        budget: Option<usize>,
    ) -> Option<(Vec<EdgeItem>, FastEvalStats)> {
        let n = budget.unwrap_or(frozen.len()).min(frozen.len());

        let good_indices: Vec<usize> = frozen
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.same.then_some(i))
            .collect();
        let mut pairs: Vec<&FrozenPair> = good_indices.iter().take(n/2).map(|&i| &frozen[i]).collect();


        let bad_indices: Vec<usize> = frozen
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (!p.same).then_some(i))
            .collect();
        let bad_pairs: Vec<&FrozenPair> = bad_indices.iter().take(n/2).map(|&i| &frozen[i]).collect();
        pairs.extend(bad_pairs);

        // Parallel map -> local accumulation -> reduce (no mutex).
        let (items, stats) = pairs
            .par_iter()
            .map(|p| {
                // Local counters for this one pair.
                let mut st = FastEvalStats::default();

                if p.same {
                    st.tot_good += 1;
                } else {
                    st.tot_bad += 1;
                }

                let Some(edge) = p.compute_score_edge(store, cfg) else {
                    return (Vec::<EdgeItem>::new(), st);
                };

                if p.same {
                    st.acc_good += 1;
                } else {
                    st.acc_bad += 1;
                }

                let cost = edge.cost;
                if !cost.is_finite() {
                    return (Vec::<EdgeItem>::new(), st);
                }

                if p.same {
                    st.acc_cost_good += 1;
                } else {
                    st.acc_cost_bad += 1;
                }

                let item = EdgeItem {
                    cost,
                    score: -cost,
                    is_good: p.same,
                };

                (vec![item], st)
            })
            .reduce(
                || (Vec::<EdgeItem>::new(), FastEvalStats::default()),
                |(mut items_a, mut st_a), (mut items_b, st_b)| {
                    // Merge vectors efficiently.
                    items_a.append(&mut items_b);

                    // Merge counters.
                    st_a.tot_good += st_b.tot_good;
                    st_a.tot_bad += st_b.tot_bad;
                    st_a.acc_good += st_b.acc_good;
                    st_a.acc_bad += st_b.acc_bad;
                    st_a.acc_cost_good += st_b.acc_cost_good;
                    st_a.acc_cost_bad += st_b.acc_cost_bad;

                    (items_a, st_a)
                },
            );

        if items.is_empty() || stats.acc_cost_good == 0 || stats.acc_cost_bad == 0 {
            return None;
        }

        Some((items, stats))
    }

    /// Write frozen pairs to a binary file.
    ///
    /// Arguments
    /// ---------
    /// * `path` – Output file path.
    /// * `frozen_pairs` – Slice of frozen pairs to write.
    ///
    /// Return
    /// ------
    /// * `anyhow::Result<()>` – I/O result.
    pub fn write(path: &Utf8Path, frozen_pairs: &[Self]) -> anyhow::Result<()> {
        write_bin(path, frozen_pairs)
    }

    /// Read frozen pairs from a binary file.
    ///
    /// Arguments
    /// ---------
    /// * `path` – Input file path.
    ///
    /// Return
    /// ------
    /// * `anyhow::Result<Vec<FrozenPair>>` – Read frozen pairs.
    pub fn read(path: &Utf8Path) -> anyhow::Result<Vec<Self>> {
        read_bin(path)
    }
}

fn push_true_frozen_pairs_only_gated(
    a_nid: NightId,
    b_nid: NightId,
    src: &NightSeeds,
    dst: &NightSeeds,
    cfg: &ScoreConfig,
    delta: u32,
    out: &mut Vec<FrozenPair>,
) {
    // Group src indices by tid
    let mut src_by_tid: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, t) in src.truth.iter().enumerate() {
        if let Some(tid) = t {
            src_by_tid.entry(*tid).or_default().push(idx);
        }
    }

    // Group dst indices by tid
    let mut dst_by_tid: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, t) in dst.truth.iter().enumerate() {
        if let Some(tid) = t {
            dst_by_tid.entry(*tid).or_default().push(idx);
        }
    }

    for (tid, a_list) in src_by_tid.iter() {
        let Some(b_list) = dst_by_tid.get(tid) else {
            continue;
        };

        for &a_idx in a_list.iter() {
            for &b_idx in b_list.iter() {
                // Gate check (same as labeled_edges_by_delta true pass)
                let a_seed = &src.seeds[a_idx];
                let b_seed = &dst.seeds[b_idx];

                if ScoredEdge::score(a_seed, b_seed, cfg, delta).is_none() {
                    continue;
                }

                out.push(FrozenPair {
                    a_nid,
                    b_nid,
                    a_idx,
                    b_idx,
                    same: true,
                    delta,
                });
            }
        }
    }
}

fn sample_false_frozen_pairs_gated(
    a_nid: NightId,
    b_nid: NightId,
    src: &NightSeeds,
    dst: &NightSeeds,
    cfg: &ScoreConfig,
    delta: u32,
    target_false: usize,
    rng: &mut StdRng,
    out: &mut Vec<FrozenPair>,
) {
    if target_false == 0 {
        return;
    }

    // Build pools restricted to Some(tid) to ensure “different asteroid”.
    let src_pool: Vec<(usize, i32)> = src
        .truth
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.map(|tid| (i, tid)))
        .collect();

    let dst_pool: Vec<(usize, i32)> = dst
        .truth
        .iter()
        .enumerate()
        .filter_map(|(j, t)| t.map(|tid| (j, tid)))
        .collect();

    if src_pool.is_empty() || dst_pool.is_empty() {
        return;
    }

    // Avoid duplicates among accepted false pairs (stability).
    let mut used: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::with_capacity(target_false.saturating_mul(2));

    let mut accepted = 0usize;
    let mut attempts = 0usize;

    // Rejection sampling: many candidates may be gated out, keep bounded.
    let max_attempts = target_false.saturating_mul(50).max(10_000);

    while accepted < target_false && attempts < max_attempts {
        attempts += 1;

        let (a_idx, ta) = src_pool[rng.random_range(0..src_pool.len())];
        let (b_idx, tb) = dst_pool[rng.random_range(0..dst_pool.len())];

        if ta == tb {
            continue; // would be true
        }
        if !used.insert((a_idx, b_idx)) {
            continue; // duplicate
        }

        let a_seed = &src.seeds[a_idx];
        let b_seed = &dst.seeds[b_idx];

        // Gate check: must be accepted by score() like labeled_edges_by_delta false pass
        if ScoredEdge::score(a_seed, b_seed, cfg, delta).is_none() {
            continue;
        }

        out.push(FrozenPair {
            a_nid,
            b_nid,
            a_idx,
            b_idx,
            same: false,
            delta,
        });
        accepted += 1;
    }
}

/// Bookkeeping counters for fast evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub struct FastEvalStats {
    pub tot_good: usize,
    pub tot_bad: usize,
    pub acc_good: usize,
    pub acc_bad: usize,
    pub acc_cost_good: usize,
    pub acc_cost_bad: usize,
}

impl FastEvalStats {
    pub fn accept_good_rate(&self) -> f64 {
        if self.tot_good == 0 {
            0.0
        } else {
            self.acc_cost_good as f64 / self.tot_good as f64
        }
    }
    pub fn accept_bad_rate(&self) -> f64 {
        if self.tot_bad == 0 {
            0.0
        } else {
            self.acc_cost_bad as f64 / self.tot_bad as f64
        }
    }
}

/// Compute a fast scalar objective for Optuna using existing `Metrics` implementations.
///
/// Recommended objective
/// ---------------------
/// - Primary: `FPR@TPR=0.99` on cost (lower is better)
/// - Penalty: discourage degenerate configs that reject too many good edges
pub fn compute_fast_objective(
    items: &[EdgeItem],
    stats: FastEvalStats,
    target_tpr: f64,
    min_accept_good_rate: f64,
    reject_penalty_weight: f64,
) -> f64 {
    let n_good = stats.acc_cost_good;
    let n_bad = stats.acc_cost_bad;

    // Main term: FPR at fixed TPR (using your tie-block sweep).
    let fpr = items.fpr_at_tpr_on_cost(n_good, n_bad, target_tpr);

    // Penalty if we accept too few good edges (to avoid "win by rejecting everything").
    let acc_good_rate = stats.accept_good_rate();
    let penalty = if acc_good_rate < min_accept_good_rate {
        reject_penalty_weight * (min_accept_good_rate - acc_good_rate)
    } else {
        0.0
    };

    // Also guard against NaNs just in case.
    let obj = fpr + penalty;
    if obj.is_finite() { obj } else { 1e9 }
}

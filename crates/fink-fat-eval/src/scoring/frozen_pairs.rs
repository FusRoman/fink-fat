// -----------------------------------------------------------------------------
// Frozen pairs generation
// -----------------------------------------------------------------------------

use fink_fat_engine::{graph::score::ScoredEdge, night_id::NightId};
use rand::{SeedableRng, seq::index::sample};

use crate::night_seeds::{NightSeeds, SeedStore};

/// Configuration for balanced inter-night sampling.
#[derive(Debug, Clone)]
pub struct EdgeSampling {
    /// Number of right seeds sampled per left seed.
    ///
    /// This controls the main runtime knob: total score calls per night pair
    /// are approximately `O(|left| * sample_right_per_left)` (before early-stop).
    pub sample_right_per_left: usize,

    /// Target edges per class (same AND diff) **kept per night pair**.
    ///
    /// Total kept per pair is at most `2 * target_per_class`.
    pub target_per_class: usize,

    /// Hard cap on the number of seed-pair score attempts per night pair.
    ///
    /// This bounds runtime even when `ScoredEdge::score` rejects most pairs.
    pub max_tested_pairs_per_night_pair: usize,

    /// Limit night pairs to `j <= i + max_night_jump` in sorted night order.
    ///
    /// - `Some(1)` means only consecutive nights.
    /// - `Some(2)` allows a 2-night jump, etc.
    /// - `None` scores all `(i, j)` with `i < j` (can be huge).
    pub max_night_jump: Option<u32>,

    /// If true, only consider edges where both endpoints have truth ids.
    pub only_truth: bool,

    /// Base RNG seed (deterministic per night pair).
    pub base_seed: u64,
}

impl Default for EdgeSampling {
    fn default() -> Self {
        Self {
            sample_right_per_left: 32,
            target_per_class: 2_000,
            max_tested_pairs_per_night_pair: 200_000,
            max_night_jump: Some(2),
            only_truth: true,
            base_seed: 42,
        }
    }
}

/// Scored edge with a truth label.
///
/// Notes
/// -----
/// `same=true` means both endpoints have a truth id and they match.
#[derive(Clone, Debug)]
pub struct LabeledEdge {
    pub same: bool,
    pub edge: ScoredEdge,
}

/// Represents a frozen candidate edge between two seeds. The pair is identified
/// by the night IDs of the left and right seed and their indices within the
/// corresponding `NightSeeds::seeds` vectors. The `same` field encodes whether
/// the two seeds share the same truth ID (both truth IDs present and equal).
#[derive(Debug, Clone)]
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

/// Helper: return the list of eligible seed indices on one side.
///
/// If `only_truth` is true, keeps only seeds for which a truth ID is present.
fn eligible_seed_indices(truth: &[Option<i32>], only_truth: bool) -> Vec<usize> {
    if !only_truth {
        return (0..truth.len()).collect();
    }
    truth
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.is_some().then_some(i))
        .collect()
}

/// Deterministic RNG seed for a given night pair.
///
/// This replicates the mixing used in `score_edge_gen` to produce per-pair seeds.
fn seed_for_pair(base_seed: u64, nid_a: i32, nid_b: i32) -> [u8; 32] {
    // See `score_edge_gen.rs` for the rationale. We reproduce the mixing here.
    fn mix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    let a = nid_a as i64 as u64;
    let b = nid_b as i64 as u64;
    let s0 = mix64(base_seed ^ a.wrapping_mul(0xD6E8FEB86659FD93) ^ b.rotate_left(7));
    let s1 =
        mix64(base_seed.rotate_left(17) ^ b.wrapping_mul(0xA5A3564E27F9C8D1) ^ a.rotate_left(9));
    let s2 =
        mix64(base_seed.rotate_left(33) ^ a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.rotate_left(13));
    let s3 =
        mix64(base_seed.rotate_left(49) ^ b.wrapping_mul(0xBF58476D1CE4E5B9) ^ a.rotate_left(19));
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&s0.to_le_bytes());
    out[8..16].copy_from_slice(&s1.to_le_bytes());
    out[16..24].copy_from_slice(&s2.to_le_bytes());
    out[24..32].copy_from_slice(&s3.to_le_bytes());
    out
}

/// Build night index pairs `(i, j)` with an optional maximum jump.
///
/// Parameters
/// ----------
/// * `n` – Number of nights (already sorted by nid).
/// * `max_jump` – Optional jump constraint.
///
/// Returns
/// -------
/// * `Vec<(usize, usize)>` – All `(i, j)` with `i < j` and `j < i+1+max_jump` when set.
#[inline]
pub fn night_pairs_with_jump(nights: &[&NightSeeds], max_jump: Option<u32>) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..nights.len() {
        for j in (i + 1)..nights.len() {
            let delta = (nights[j].nid.0) - (nights[i].nid.0);
            if delta <= 0 {
                continue;
            }
            if let Some(m) = max_jump {
                if (delta as u32) > m {
                    break; // nights triés, donc delta ne fera qu'augmenter
                }
            }
            pairs.push((i, j));
        }
    }
    pairs
}

/// Freeze a balanced set of inter-night seed pairs for evaluation.
///
/// This mirrors the sampling strategy used in `generate_balanced_inter_night_edges` but
/// does not call `ScoredEdge::score`. Instead it records the seed indices and
/// their class label (`same` or `diff`). The returned list contains, for each
/// night pair, an equal number of `same` and `diff` pairs up to
/// `sampling.target_per_class`.
pub fn freeze_balanced_edge_pairs(
    seed_store: &SeedStore,
    sampling: &EdgeSampling,
) -> Vec<FrozenPair> {
    // Sort nights deterministically by nid.
    let mut nights: Vec<&NightSeeds> = seed_store.values().collect();
    nights.sort_unstable_by_key(|n| n.nid);
    // Build candidate night pairs.
    let pairs = night_pairs_with_jump(&nights, sampling.max_night_jump);
    let mut out_pairs: Vec<FrozenPair> = Vec::new();
    // Process each night pair sequentially.
    for (i, j) in pairs {
        let a = nights[i];
        let b = nights[j];

        // Number of night steps between the two revisits (>= 1).
        let delta_i64 = (b.nid.0 as i64) - (a.nid.0 as i64);
        if delta_i64 <= 0 {
            continue;
        }
        let delta = delta_i64 as u32;

        // Build eligible indices on both sides.
        let left_idx = eligible_seed_indices(&a.truth, sampling.only_truth);
        let right_idx = eligible_seed_indices(&b.truth, sampling.only_truth);
        if left_idx.is_empty() || right_idx.is_empty() {
            continue;
        }
        let k = sampling.sample_right_per_left.min(right_idx.len());
        // Create per-pair RNG.
        let seed = seed_for_pair(sampling.base_seed, a.nid.0 as i32, b.nid.0 as i32);
        let mut rng = rand::rngs::StdRng::from_seed(seed);
        let mut same_list: Vec<FrozenPair> = Vec::new();
        let mut diff_list: Vec<FrozenPair> = Vec::new();
        let mut tested_pairs = 0usize;
        'outer: for &ia in &left_idx {
            // Early stop when both buckets filled.
            if same_list.len() >= sampling.target_per_class
                && diff_list.len() >= sampling.target_per_class
            {
                break;
            }
            // Hard cap on score attempts.
            if tested_pairs >= sampling.max_tested_pairs_per_night_pair {
                break;
            }
            // Sample k distinct indices on the right side.
            let picked = sample(&mut rng, right_idx.len(), k);
            for pick in picked.iter() {
                if tested_pairs >= sampling.max_tested_pairs_per_night_pair {
                    break 'outer;
                }
                let ib = right_idx[pick];
                let ti = a.truth[ia];
                let tj = b.truth[ib];
                tested_pairs += 1;
                let same = match (ti, tj) {
                    (Some(x), Some(y)) => x == y,
                    _ => false,
                };
                if same {
                    if same_list.len() < sampling.target_per_class {
                        same_list.push(FrozenPair {
                            a_nid: a.nid,
                            b_nid: b.nid,
                            a_idx: ia,
                            b_idx: ib,
                            same: true,
                            delta,
                        });
                    }
                } else if diff_list.len() < sampling.target_per_class {
                    diff_list.push(FrozenPair {
                        a_nid: a.nid,
                        b_nid: b.nid,
                        a_idx: ia,
                        b_idx: ib,
                        same: false,
                        delta,
                    });
                }
                if same_list.len() >= sampling.target_per_class
                    && diff_list.len() >= sampling.target_per_class
                {
                    break 'outer;
                }
            }
        }
        // Truncate to equal lengths.
        let keep = same_list.len().min(diff_list.len());
        for idx in 0..keep {
            out_pairs.push(same_list[idx].clone());
            out_pairs.push(diff_list[idx].clone());
        }
    }
    out_pairs
}

//! Scoring weight optimization utilities (random search).
//!
//! Overview
//! --------
//! This module centralizes the logic used by score-optimization binaries:
//! - A compact edge representation [`EdgeSample`] storing decomposed feature components,
//! - Candidate parameter sets [`Candidate`] (weights + normalizations),
//! - Cost function evaluation (`cost_for_candidate`),
//! - AUC computation (`auc_same_vs_diff`) for same-vs-different separation,
//! - Sampling helpers (subsample + balancing),
//! - Random-search loop (`random_search_best_auc`).
//!
//! Notes
//! -----
//! - This module is intentionally independent from dataset ingestion and seeding.
//! - It expects edges to already be *gated* (i.e., only valid edges are kept).

use rand::{Rng, rngs::StdRng, seq::SliceRandom};

/// Compact representation of an edge as feature components (gates already applied).
#[derive(Clone, Debug)]
pub struct EdgeSample {
    /// True if same asteroid (traj_id match), false otherwise.
    pub same: bool,
    pub d2_pos: f64,
    pub vel_angle_rad: Option<f64>,
    pub vel_speed_diff: Option<f64>,
    pub z_flux: Option<f64>,
    pub gap_penalty: f64,
    pub band_mismatch: bool,
}

#[derive(Clone, Debug)]
pub struct SampleStats {
    pub same: usize,
    pub diff: usize,
}

/// Candidate parameters we optimize.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub w_pos: f64,
    pub w_dir: f64,
    pub w_norm: f64,
    pub w_flux: f64,
    pub w_gap: f64,
    pub w_band: f64,
    pub theta0: f64,
    pub v0: f64,
}

impl Candidate {
    /// Draw a random candidate using log-uniform ranges.
    pub fn random(rng: &mut StdRng) -> Self {
        // Log-uniform helper.
        fn logu(rng: &mut StdRng, lo: f64, hi: f64) -> f64 {
            let a = lo.ln();
            let b = hi.ln();
            (a + rng.random_range(0.0..1.0) * (b - a)).exp()
        }

        Self {
            w_pos: logu(rng, 0.05, 5.0),
            w_dir: logu(rng, 0.01, 5.0),
            w_norm: logu(rng, 0.01, 5.0),
            w_flux: logu(rng, 1e-6, 2.0), // allow near-zero
            w_gap: logu(rng, 1e-3, 2.0),
            w_band: logu(rng, 1e-3, 2.0),
            // theta0 in radians: ~ 0.1 deg .. 20 deg
            theta0: logu(rng, (0.1_f64).to_radians(), (20.0_f64).to_radians()),
            // v0 in rad/day: ~ 0.01 arcmin/day .. 20 deg/day
            v0: logu(rng, (0.01_f64 / 60.0).to_radians(), (20.0_f64).to_radians()),
        }
    }
}

/// Compute cost from components for a candidate parameter set.
/// Lower is better.
#[inline]
pub fn cost_for_candidate(e: &EdgeSample, c: &Candidate) -> f64 {
    let mut cost = 0.0;

    cost += c.w_pos * e.d2_pos;

    if let Some(theta) = e.vel_angle_rad {
        cost += c.w_dir * (theta / c.theta0);
    }
    if let Some(dv) = e.vel_speed_diff {
        cost += c.w_norm * (dv / c.v0);
    }
    if let Some(z) = e.z_flux {
        cost += c.w_flux * z;
    }

    cost += c.w_gap * e.gap_penalty;

    if e.band_mismatch {
        cost += c.w_band;
    }

    cost
}

/// Build AUC for positive class = "same asteroid" using score = -cost.
///
/// Returns
/// -------
/// float
///     AUC in [0,1]. If degenerate, returns 0.5.
pub fn auc_same_vs_diff(cost_same: &[f64], cost_diff: &[f64]) -> f64 {
    let n_pos = cost_same.len();
    let n_neg = cost_diff.len();
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Use Mann–Whitney U via ranking of scores; scores are -cost.
    let mut all: Vec<(f64, bool)> = Vec::with_capacity(n_pos + n_neg);
    all.extend(cost_same.iter().map(|&c| (-c, true)));
    all.extend(cost_diff.iter().map(|&c| (-c, false)));

    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Average ranks for ties.
    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0usize;
    while i < all.len() {
        let j = (i + 1..all.len())
            .find(|&k| all[k].0 != all[i].0)
            .unwrap_or(all.len());

        let rank_lo = (i + 1) as f64;
        let rank_hi = j as f64;
        let rank_avg = 0.5 * (rank_lo + rank_hi);

        for k in i..j {
            if all[k].1 {
                rank_sum_pos += rank_avg;
            }
        }
        i = j;
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;

    let u = rank_sum_pos - n_pos_f * (n_pos_f + 1.0) / 2.0;
    u / (n_pos_f * n_neg_f)
}

/// Randomly subsample a vector in-place if it exceeds `max_n`.
pub fn subsample_in_place<T>(v: &mut Vec<T>, max_n: usize, rng: &mut StdRng) {
    if max_n == 0 || v.len() <= max_n {
        return;
    }
    v.shuffle(rng);
    v.truncate(max_n);
}

/// Balance two vectors by downsampling the larger one to the size of the smaller one.
/// Returns the kept size N.
pub fn balance_same_diff(
    same: &mut Vec<EdgeSample>,
    diff: &mut Vec<EdgeSample>,
    rng: &mut StdRng,
) -> usize {
    same.shuffle(rng);
    diff.shuffle(rng);
    let n = same.len().min(diff.len());
    same.truncate(n);
    diff.truncate(n);
    n
}

/// Result of a random search.
#[derive(Clone, Debug)]
pub struct OptimResult {
    pub best_auc: f64,
    pub best: Candidate,
}

/// Run a random search to maximize AUC (same vs different).
///
/// Parameters
/// ----------
/// edges : &[EdgeSample]
///     Balanced edge set, each edge labeled with `same`.
/// stats : SampleStats
///     Counts for each class (used only for buffer sizing sanity).
/// budget : usize
///     Number of random trials.
/// rng : &mut StdRng
///     Random number generator (reproducibility).
pub fn random_search_best_auc(
    edges: &[EdgeSample],
    stats: &SampleStats,
    budget: usize,
    rng: &mut StdRng,
) -> OptimResult {
    let mut best_auc = -1.0_f64;
    let mut best: Option<Candidate> = None;

    // Pre-allocate cost buffers for speed.
    let mut cost_same: Vec<f64> = Vec::with_capacity(stats.same.max(1));
    let mut cost_diff: Vec<f64> = Vec::with_capacity(stats.diff.max(1));

    for t in 0..budget {
        let cand = Candidate::random(rng);

        cost_same.clear();
        cost_diff.clear();

        for e in edges {
            let c = cost_for_candidate(e, &cand);
            if e.same {
                cost_same.push(c);
            } else {
                cost_diff.push(c);
            }
        }

        let auc = auc_same_vs_diff(&cost_same, &cost_diff);

        if auc > best_auc {
            best_auc = auc;
            best = Some(cand);
            eprintln!("  [best @ {:>5}/{}] AUC={:.6}", t + 1, budget, best_auc);
        }
    }

    OptimResult {
        best_auc,
        best: best.expect("budget >= 1 must yield a best candidate"),
    }
}

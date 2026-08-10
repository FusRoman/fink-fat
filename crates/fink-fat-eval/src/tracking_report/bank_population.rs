//! Why hypothesis banks stay large, and what that costs the association gate.
//!
//! # The contradiction this exists to settle
//!
//! A search region should *shrink* as a lineage converges. Three figures from
//! the same run say it does not:
//!
//! * hypotheses per branch: **84.4** (median 68.3, max 443);
//! * effective sample size: **3.4** — 84 hypotheses of which ~4 carry weight;
//! * bank error box: **940 arcsec** (median 957), the union of those 84.
//!
//! Yet the configured schedule (`Logarithmic { start: 100, end: 1,
//! n_obs_full: 5 }`) evaluates to a cap of **1** as soon as `n_steps >= 5`, and
//! that cap *is* enforced (`cap_to_scheduled_max`), with `n_steps` surviving
//! propagation. The two cannot both be true.
//!
//! An aggregate mean cannot separate the explanations, which is the whole
//! reason for this module. Joining the hypothesis count to `n_steps` does:
//!
//! * if branches at `n_steps >= 5` really hold one hypothesis, the cap works
//!   and the 84.4 mean comes from young branches — the box has another cause;
//! * if they hold dozens, the cap is not reaching them, and that is a defect
//!   with a direct line to the 76 % wrong-fraction of old lineages: a
//!   940 arcsec box admits roughly one candidate per lineage-night, and most
//!   of what it admits belongs to another object.
//!
//! Nothing here changes engine behaviour; it samples what already happens,
//! reusing the propagation `night_stats` performs anyway.

use ahash::AHashMap;

/// One branch's state at the end of a night.
pub struct BankSample {
    pub n_steps: usize,
    pub n_hypotheses: usize,
    /// The cap that applies at this `n_steps`, floor included.
    pub effective_cap: usize,
    pub effective_sample_size: f64,
    /// Predictive box radius for the next night, when it could be computed.
    pub radius_arcsec: Option<f64>,
}

/// Running totals for one `n_steps` bucket.
#[derive(Default, Clone)]
struct Bucket {
    n_branches: usize,
    sum_hypotheses: usize,
    max_hypotheses: usize,
    sum_cap: usize,
    /// Branches holding **more** hypotheses than their own cap allows. Should
    /// be zero: a nonzero count means the cap is not being enforced where it is
    /// measured, which is a defect rather than a tuning question.
    n_over_cap: usize,
    sum_ess: f64,
    sum_radius: f64,
    n_radius: usize,
    max_radius: f64,
}

/// Accumulated across every night of the run.
#[derive(Default)]
pub struct BankPopulationStats {
    buckets: AHashMap<usize, Bucket>,
    n_samples: usize,
}

/// `n_steps` values at or above this are pooled into one bucket: the cap has
/// long since bottomed out, so the individual values stop being informative.
const BUCKET_DEPTH: usize = 10;

impl BankPopulationStats {
    /// Fold in one night's branches.
    pub fn observe_night(&mut self, samples: impl IntoIterator<Item = BankSample>) {
        for s in samples {
            self.n_samples += 1;
            let bucket = self.buckets.entry(s.n_steps.min(BUCKET_DEPTH)).or_default();
            bucket.n_branches += 1;
            bucket.sum_hypotheses += s.n_hypotheses;
            bucket.max_hypotheses = bucket.max_hypotheses.max(s.n_hypotheses);
            bucket.sum_cap += s.effective_cap;
            if s.n_hypotheses > s.effective_cap {
                bucket.n_over_cap += 1;
            }
            bucket.sum_ess += s.effective_sample_size;
            if let Some(r) = s.radius_arcsec.filter(|r| r.is_finite()) {
                bucket.sum_radius += r;
                bucket.n_radius += 1;
                bucket.max_radius = bucket.max_radius.max(r);
            }
        }
    }

    pub fn print_summary(&self) {
        let sep = "=".repeat(90);
        println!("\n{sep}");
        println!(
            "[Bank population] hypothesis retention against observations consumed ({} branch-nights)",
            self.n_samples
        );
        println!("{sep}");
        if self.buckets.is_empty() {
            println!("  (no branch sampled)");
            println!("{sep}");
            return;
        }

        println!(
            "  {:<10} {:>12} {:>10} {:>8} {:>8} {:>10} {:>8} {:>12} {:>12}",
            "n_steps",
            "branches",
            "hyp mean",
            "hyp max",
            "cap",
            "over cap",
            "ESS",
            "radius mean",
            "radius max"
        );

        let mut keys: Vec<usize> = self.buckets.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let b = &self.buckets[&k];
            let n = b.n_branches.max(1) as f64;
            let label = if k >= BUCKET_DEPTH {
                format!("{BUCKET_DEPTH}+")
            } else {
                k.to_string()
            };
            println!(
                "  {:<10} {:>12} {:>10.1} {:>8} {:>8.1} {:>10} {:>8.2} {:>12} {:>12}",
                label,
                b.n_branches,
                b.sum_hypotheses as f64 / n,
                b.max_hypotheses,
                b.sum_cap as f64 / n,
                b.n_over_cap,
                b.sum_ess / n,
                if b.n_radius > 0 {
                    format!("{:.1}", b.sum_radius / b.n_radius as f64)
                } else {
                    "-".to_string()
                },
                if b.n_radius > 0 {
                    format!("{:.1}", b.max_radius)
                } else {
                    "-".to_string()
                },
            );
        }

        // The two readings this table exists to separate, stated rather than
        // left for the reader to reconstruct.
        let mature: Vec<&Bucket> = self
            .buckets
            .iter()
            .filter(|(k, _)| **k >= 5)
            .map(|(_, b)| b)
            .collect();
        let mature_branches: usize = mature.iter().map(|b| b.n_branches).sum();
        let mature_hyp: usize = mature.iter().map(|b| b.sum_hypotheses).sum();
        let mature_over: usize = mature.iter().map(|b| b.n_over_cap).sum();
        println!(
            "\n  Branches at n_steps >= 5           : {mature_branches} ({:.1}% of samples)",
            100.0 * mature_branches as f64 / self.n_samples.max(1) as f64
        );
        if mature_branches > 0 {
            println!(
                "  Their mean hypothesis count        : {:.1}  (the schedule caps them at 1)",
                mature_hyp as f64 / mature_branches as f64
            );
        }
        println!("  Of those, holding more than the cap: {mature_over}");
        println!("{sep}");
    }
}

//! Tracks lineage birth/death across the whole run, so per-night stats can
//! report how many lineages were newly seeded, how many died (were fully
//! pruned), and — aggregated at the end — how many nights a died lineage
//! typically survived.

use ahash::{AHashMap, AHashSet};

/// Per-lineage birth-night bookkeeping, updated once per night via
/// [`LineageTracker::advance`].
#[derive(Default)]
pub struct LineageTracker {
    birth_step: AHashMap<u64, usize>,
}

/// Lineage churn for a single night.
pub struct LineageDiff {
    pub n_new: usize,
    /// Nights survived by each lineage that died this night (`current_step
    /// - birth_step`), one entry per died lineage.
    pub died_survival_nights: Vec<f64>,
}

impl LineageDiff {
    pub fn n_died(&self) -> usize {
        self.died_survival_nights.len()
    }
}

impl LineageTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one night's before/after lineage id sets into the tracker,
    /// registering new births and reaping dead lineages.
    ///
    /// `prev_ids` is the lineage set *before* this night's
    /// `advance_one_night` call, `current_ids` the set *after*.
    pub fn advance(
        &mut self,
        current_step: usize,
        prev_ids: &AHashSet<u64>,
        current_ids: &AHashSet<u64>,
    ) -> LineageDiff {
        let n_new = current_ids
            .iter()
            .filter(|id| !self.birth_step.contains_key(id))
            .count();
        for &id in current_ids {
            self.birth_step.entry(id).or_insert(current_step);
        }

        let died_survival_nights = prev_ids
            .iter()
            .filter(|id| !current_ids.contains(id))
            .filter_map(|id| self.birth_step.remove(id))
            .map(|birth_step| current_step.saturating_sub(birth_step) as f64)
            .collect();

        LineageDiff {
            n_new,
            died_survival_nights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_lineages_are_counted_once() {
        let mut tracker = LineageTracker::new();
        let prev: AHashSet<u64> = AHashSet::default();
        let current: AHashSet<u64> = [1, 2].into_iter().collect();

        let diff = tracker.advance(0, &prev, &current);
        assert_eq!(diff.n_new, 2);
        assert_eq!(diff.n_died(), 0);
    }

    #[test]
    fn survival_nights_is_current_step_minus_birth_step() {
        let mut tracker = LineageTracker::new();
        let step0: AHashSet<u64> = [1].into_iter().collect();
        tracker.advance(0, &AHashSet::default(), &step0);

        let step3_prev = step0.clone();
        let step3_current: AHashSet<u64> = AHashSet::default();
        let diff = tracker.advance(3, &step3_prev, &step3_current);

        assert_eq!(diff.died_survival_nights, vec![3.0]);
    }

    #[test]
    fn a_lineage_seen_again_is_not_recounted_as_new() {
        let mut tracker = LineageTracker::new();
        let ids: AHashSet<u64> = [1].into_iter().collect();
        tracker.advance(0, &AHashSet::default(), &ids);
        let diff = tracker.advance(1, &ids, &ids);
        assert_eq!(diff.n_new, 0);
        assert_eq!(diff.n_died(), 0);
    }
}

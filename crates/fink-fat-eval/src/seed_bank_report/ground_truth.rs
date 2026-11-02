//! Ground-truth trajectory lookup for evaluating intra-night seeding.
//!
//! [`ObsDataset`] already indexes observations by their `traj_id` column
//! when the source Parquet file has one (see
//! [`ObsDataset::iter_full_trajectory`]) — this module just flattens that
//! index into an `ObsId -> TrajId` map for O(1) lookups while scoring a
//! [`Branch`](fink_fat_engine::topocentric_kf::branching::Branch)'s seed
//! pair against ground truth.

use ahash::AHashMap;
use photom::{TrajId, observation_dataset::ObsDataset, observation_dataset::ObsId};

/// `ObsId -> TrajId` reverse lookup, built once per run from
/// [`ObsDataset::iter_full_trajectory`].
pub struct ObsTrajLookup {
    traj_of_obs: AHashMap<ObsId, TrajId>,
}

/// How a seed's member observations relate to ground truth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedPurity {
    /// Every member observation belongs to the same ground-truth trajectory.
    Pure(TrajId),
    /// Member observations belong to at least two distinct trajectories —
    /// a false cross-object merge.
    Mixed,
    /// At least one member observation has no ground-truth trajectory.
    Unknown,
}

impl ObsTrajLookup {
    /// Build the lookup from every trajectory-indexed observation in
    /// `obs_dataset`. Returns an empty lookup (everything classifies as
    /// [`SeedPurity::Unknown`]) if the dataset has no trajectory index —
    /// i.e. the source data had no `traj_id` column.
    pub fn build(obs_dataset: &ObsDataset) -> Self {
        let traj_of_obs = obs_dataset
            .iter_full_trajectory()
            .into_iter()
            .flatten()
            .map(|(traj_id, obs)| (*obs.id(), traj_id))
            .collect();
        Self { traj_of_obs }
    }

    /// Ground-truth trajectory for a single observation id, if known.
    pub fn traj_of(&self, obs_id: ObsId) -> Option<&TrajId> {
        self.traj_of_obs.get(&obs_id)
    }

    /// Whether any observation in the dataset has ground-truth trajectory
    /// data — `false` means the source Parquet had no `traj_id` column, and
    /// any reconstruction-efficacy computation built on top of this lookup
    /// should be skipped.
    pub fn has_ground_truth(&self) -> bool {
        !self.traj_of_obs.is_empty()
    }

    /// Total known-ground-truth observation count per trajectory — the
    /// denominator for reconstruction-coverage ratios (see
    /// `crate::snapshot_report::efficacy`). Built by inverting
    /// `traj_of_obs`.
    pub fn obs_counts_by_traj(&self) -> AHashMap<TrajId, usize> {
        let mut counts = AHashMap::default();
        for traj_id in self.traj_of_obs.values() {
            *counts.entry(traj_id.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Classify a seed's member observation ids against ground truth.
    ///
    /// An empty slice classifies as [`SeedPurity::Unknown`] (nothing to
    /// agree on).
    pub fn classify(&self, member_ids: &[ObsId]) -> SeedPurity {
        let mut members = member_ids.iter();
        let Some(first_traj) = members.next().and_then(|id| self.traj_of(*id)) else {
            return SeedPurity::Unknown;
        };

        for id in members {
            match self.traj_of(*id) {
                Some(traj) if traj == first_traj => {}
                Some(_) => return SeedPurity::Mixed,
                None => return SeedPurity::Unknown,
            }
        }
        SeedPurity::Pure(first_traj.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup_from(pairs: &[(ObsId, u32)]) -> ObsTrajLookup {
        ObsTrajLookup {
            traj_of_obs: pairs
                .iter()
                .map(|&(id, traj)| (id, TrajId::Int(traj)))
                .collect(),
        }
    }

    #[test]
    fn pure_when_all_members_share_one_trajectory() {
        let lookup = lookup_from(&[(1, 10), (2, 10), (3, 10)]);
        assert_eq!(
            lookup.classify(&[1, 2, 3]),
            SeedPurity::Pure(TrajId::Int(10))
        );
    }

    #[test]
    fn mixed_when_members_disagree() {
        let lookup = lookup_from(&[(1, 10), (2, 20)]);
        assert_eq!(lookup.classify(&[1, 2]), SeedPurity::Mixed);
    }

    #[test]
    fn unknown_when_a_member_is_missing_from_ground_truth() {
        let lookup = lookup_from(&[(1, 10)]);
        assert_eq!(lookup.classify(&[1, 2]), SeedPurity::Unknown);
    }

    #[test]
    fn unknown_for_empty_member_list() {
        let lookup = lookup_from(&[]);
        assert_eq!(lookup.classify(&[]), SeedPurity::Unknown);
    }

    #[test]
    fn single_member_is_always_pure() {
        let lookup = lookup_from(&[(1, 10)]);
        assert_eq!(lookup.classify(&[1]), SeedPurity::Pure(TrajId::Int(10)));
    }
}

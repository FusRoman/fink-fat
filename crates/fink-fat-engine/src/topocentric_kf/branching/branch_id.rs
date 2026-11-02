//! Human-readable, deterministic identifiers for [`Branch`](super::Branch)es.
//!
//! Two levels, mirroring how a lineage of candidate association histories
//! relates to the object it is trying to track:
//!
//! - [`lineage_designation`] — `FF{YYYY}{12-letter suffix}`, computed once
//!   from the night-0 seed bank and copied unchanged by every descendant
//!   branch. Groups all sibling candidates for the same tracked object.
//! - [`branch_designation`] — `{lineage}-{8-letter suffix}`, the id actually
//!   safe to use as a `trajectory_id` when persisting `(trajectory_id,
//!   observation_id)` rows: it is unique per distinct association history
//!   (ordered `ObsId` sequence), so two branches that have consumed different
//!   observations never collapse onto the same id. A "null" (missed
//!   detection) branch shares its parent's `branch_designation`, since it
//!   covers exactly the same set of observations.
//!
//! The hashing scheme (fixed-seed `ahash`, base26 encoding) is ported from
//! the older `trajectory::track_id` module — see that module's docs for the
//! full rationale. That module is no longer wired into the crate (it depends
//! on seeding types that no longer exist), so the algorithm is reimplemented
//! here against the flat `ObsId` sequences that `KFBank` actually retains.
//!
//! As with the original, the hash seeds are constants and **must not
//! change** once identifiers are in use, or previously persisted ids will no
//! longer match recomputed ones.

use std::fmt;
use std::hash::BuildHasher;
use std::hash::{Hash, Hasher};

use ahash::RandomState;
use hifitime::{Epoch, TimeScale};
use photom::observation_dataset::ObsId;

/// Deterministic, human-readable identifier for a lineage or a branch.
///
/// Newtype wrapper around a `String` — see the module docs for the two
/// formats this can hold ([`lineage_designation`] vs [`branch_designation`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BranchId(pub String);

impl BranchId {
    /// Return the identifier as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BranchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Compute the lineage-level id `FF{YYYY}{12-letter suffix}` from a night-0
/// seed bank's association history.
///
/// * `track_ids` – The seed bank's `track_ids()` (chronological `ObsId`s),
///   used to derive the suffix.
/// * `epoch_mjd_tt` – Epoch (MJD TT) anchoring the seed, used to derive the
///   `YYYY` prefix (see `mjd_to_year`).
///
/// # Panics
/// Panics if `track_ids` is empty — a seed bank always has at least the two
/// observations of its founding pair.
pub fn lineage_designation(track_ids: &[ObsId], epoch_mjd_tt: f64) -> BranchId {
    assert!(
        !track_ids.is_empty(),
        "Cannot generate a lineage designation from an empty track_ids list"
    );

    const L1: u64 = 0x6A09_E667_F3BC_C909;
    const L2: u64 = 0xBB67_AE85_84CA_A73B;
    const L3: u64 = 0x3C6E_F372_FE94_F82B;
    const L4: u64 = 0xA54F_F53A_5F1D_36F1;

    let suffix = hash_track_ids(track_ids, L1, L2, L3, L4, 12);
    let year = mjd_to_year(epoch_mjd_tt);

    BranchId(format!("FF{}{}", year, suffix))
}

/// Compute the branch-level id `{lineage}-{8-letter suffix}`, unique per
/// distinct association history (ordered `ObsId` sequence).
///
/// Two branches that have consumed different observations get different
/// ids; a "null" branch, which reuses its parent's `track_ids` unchanged,
/// gets the same id as its parent — both cover exactly the same observation
/// set.
pub fn branch_designation(lineage: &BranchId, track_ids: &[ObsId]) -> BranchId {
    assert!(
        !track_ids.is_empty(),
        "Cannot generate a branch designation from an empty track_ids list"
    );

    const B1: u64 = 0x9E37_79B9_7F4A_7C15;
    const B2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const B3: u64 = 0x1656_67B1_9E37_79F9;
    const B4: u64 = 0xFF51_AFD7_ED55_8CCD;

    let suffix = hash_track_ids(track_ids, B1, B2, B3, B4, 8);

    BranchId(format!("{}-{}", lineage.as_str(), suffix))
}

/// Hash an ordered `ObsId` sequence into a fixed-width base26 suffix.
///
/// Mirrors `track_id_from_seed_keys_with_year`'s scheme: hash the length
/// first (to avoid ambiguity across concatenations), then each element in
/// order, using a deterministic `ahash::RandomState` built from fixed seeds.
fn hash_track_ids(track_ids: &[ObsId], k1: u64, k2: u64, k3: u64, k4: u64, width: usize) -> String {
    let build_hasher = RandomState::with_seeds(k1, k2, k3, k4);
    let mut hasher = build_hasher.build_hasher();

    (track_ids.len() as u64).hash(&mut hasher);
    for id in track_ids {
        id.hash(&mut hasher);
    }

    encode_base26_u64(hasher.finish(), width)
}

/// Convert MJD (TT) to Gregorian year, via `hifitime`.
fn mjd_to_year(mjd: f64) -> u32 {
    Epoch::from_mjd_in_time_scale(mjd, TimeScale::TT).year() as u32
}

/// Encode a `u64` into a fixed-width base-26 alphabetic string (`a`–`z`).
///
/// Ported from `trajectory::track_id::encode_base26_u64`.
fn encode_base26_u64(mut value: u64, width: usize) -> String {
    let mut chars = vec!['a'; width];
    for i in (0..width).rev() {
        let digit = (value % 26) as u8;
        chars[i] = (b'a' + digit) as char;
        value /= 26;
    }
    chars.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineage_designation_format_is_ff_year_plus_12_letters() {
        let id = lineage_designation(&[1, 2, 3], 59000.0);
        assert!(id.as_str().starts_with("FF2020"));
        let suffix = &id.as_str()["FF2020".len()..];
        assert_eq!(suffix.len(), 12);
        assert!(suffix.chars().all(|c| ('a'..='z').contains(&c)));
    }

    #[test]
    fn lineage_designation_is_deterministic() {
        let id1 = lineage_designation(&[10, 20], 58000.0);
        let id2 = lineage_designation(&[10, 20], 58000.0);
        assert_eq!(id1, id2);
    }

    #[test]
    fn lineage_designation_changes_when_year_changes() {
        let id1 = lineage_designation(&[10, 20], 58000.0); // 2017
        let id2 = lineage_designation(&[10, 20], 59000.0); // 2020
        assert_ne!(id1, id2);
    }

    #[test]
    fn lineage_designation_changes_when_order_changes() {
        let id_a = lineage_designation(&[10, 20, 30], 58000.0);
        let id_b = lineage_designation(&[10, 30, 20], 58000.0);
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn branch_designation_format_is_lineage_dash_8_letters() {
        let lineage = lineage_designation(&[1, 2], 59000.0);
        let id = branch_designation(&lineage, &[1, 2, 3]);
        let expected_prefix = format!("{}-", lineage.as_str());
        assert!(id.as_str().starts_with(&expected_prefix));
        let suffix = &id.as_str()[expected_prefix.len()..];
        assert_eq!(suffix.len(), 8);
        assert!(suffix.chars().all(|c| ('a'..='z').contains(&c)));
    }

    #[test]
    fn branch_designation_is_deterministic() {
        let lineage = lineage_designation(&[1, 2], 59000.0);
        let id1 = branch_designation(&lineage, &[1, 2, 3]);
        let id2 = branch_designation(&lineage, &[1, 2, 3]);
        assert_eq!(id1, id2);
    }

    #[test]
    fn branch_designation_same_for_identical_track_ids_null_branch_case() {
        let lineage = lineage_designation(&[1, 2], 59000.0);
        let parent = branch_designation(&lineage, &[1, 2, 3]);
        // A null branch reuses the parent's track_ids unchanged.
        let null_child = branch_designation(&lineage, &[1, 2, 3]);
        assert_eq!(parent, null_child);
    }

    #[test]
    fn branch_designation_differs_when_new_observation_added() {
        let lineage = lineage_designation(&[1, 2], 59000.0);
        let parent = branch_designation(&lineage, &[1, 2]);
        let obs_child = branch_designation(&lineage, &[1, 2, 3]);
        assert_ne!(parent, obs_child);
    }

    #[test]
    fn branch_designation_differs_across_different_lineages() {
        let lineage_a = lineage_designation(&[1, 2], 59000.0);
        let lineage_b = lineage_designation(&[3, 4], 59000.0);
        let id_a = branch_designation(&lineage_a, &[1, 2, 5]);
        let id_b = branch_designation(&lineage_b, &[3, 4, 5]);
        assert_ne!(id_a, id_b);
    }
}

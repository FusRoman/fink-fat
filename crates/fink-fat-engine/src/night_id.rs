use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::FinkFatError;

/// Logical identifier for a night of observation.
///
/// Notes
/// -----
/// - By default wraps an `u32`.
/// - Typically represents an MJD day number (e.g., 60312).
/// - Must be stable across runs because it is used as a directory name.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash, Default,
)]
pub struct NightId(pub u32);

impl NightId {
    /// Create a new `NightId` from an integer.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the underlying integer.
    pub fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NightId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for NightId {
    #[inline]
    fn from(v: u32) -> Self {
        NightId(v)
    }
}

impl From<NightId> for u32 {
    #[inline]
    fn from(n: NightId) -> Self {
        n.0
    }
}

/// Pairing mode for night-to-night trajectory linking.
///
/// This enum distinguishes two fundamentally different strategies for
/// selecting candidate night pairs in the linking stage:
///
/// 1. **Single-night anchor**: Process one specific night by linking it with
///    earlier nights within a temporal gap.
/// 2. **Multi-night batch**: Process all nights within an inclusive range.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum PairingMode {
    /// Single-night anchor mode.
    ///
    /// Used for incremental processing: new alerts arrive for a specific night,
    /// and we need to link them with tracklets from earlier nights.
    ///
    /// Fields
    /// ------
    /// * `anchor` – The night to process (the "right" side of pairs).
    /// * `max_gap` – Maximum temporal gap (in nights) to search backwards.
    ///
    /// Pairing logic
    /// -------------
    /// - Right = `anchor` (if present in available nights).
    /// - Left candidates: all nights `n` in available nights such that:
    ///   - `n < anchor`
    ///   - `anchor - n <= max_gap`
    ///
    /// No additional range constraint is applied to left candidates.
    SingleNight { anchor: NightId, max_gap: u8 },

    /// Multi-night batch mode.
    ///
    /// Used for batch processing or reprocessing of a historical time range.
    ///
    /// Fields
    /// ------
    /// * `start` – Inclusive first night of the range.
    /// * `end` – Inclusive last night of the range.
    ///
    /// Pairing logic
    /// -------------
    /// - Right = latest night in available nights within `[start, end]`.
    /// - Left candidates: all nights `n` in available nights such that:
    ///   - `start <= n < right`
    ///
    /// The temporal gap constraint is **not** used in this mode.
    BatchRange { start: NightId, end: NightId },
}

impl PairingMode {
    /// Create a single-night anchor pairing mode.
    ///
    /// Arguments
    /// ---------
    /// * `anchor` – The night to use as the pairing anchor.
    /// * `max_gap` – Maximum temporal gap to search backwards.
    ///
    /// Errors
    /// ------
    /// Returns an error if `max_gap == 0` (no pairs can be formed).
    pub fn single_night(anchor: NightId, max_gap: u8) -> Result<Self, FinkFatError> {
        if max_gap == 0 {
            return Err(FinkFatError::Message(format!(
                "SingleNight mode requires max_gap > 0, got {}",
                max_gap
            )));
        }
        Ok(Self::SingleNight { anchor, max_gap })
    }

    /// Create a multi-night batch pairing mode.
    ///
    /// Arguments
    /// ---------
    /// * `start` – Inclusive first night of the range.
    /// * `end` – Inclusive last night of the range.
    ///
    /// Errors
    /// ------
    /// Returns an error if `start > end`.
    pub fn batch_range(start: NightId, end: NightId) -> Result<Self, FinkFatError> {
        if start > end {
            return Err(FinkFatError::Message(format!(
                "BatchRange mode requires start <= end, got start={} end={}",
                start, end
            )));
        }
        Ok(Self::BatchRange { start, end })
    }

    /// Whether this mode is single-night anchor.
    #[inline]
    pub fn is_single_night(&self) -> bool {
        matches!(self, Self::SingleNight { .. })
    }

    /// Whether this mode is multi-night batch.
    #[inline]
    pub fn is_batch(&self) -> bool {
        matches!(self, Self::BatchRange { .. })
    }

    /// Get the anchor night (only valid in single-night mode).
    ///
    /// Errors
    /// ------
    /// Returns an error if called in batch mode.
    pub fn anchor(&self) -> Result<NightId, FinkFatError> {
        match self {
            Self::SingleNight { anchor, .. } => Ok(*anchor),
            Self::BatchRange { .. } => Err(FinkFatError::Message(
                "anchor() is only valid in SingleNight mode".to_string(),
            )),
        }
    }

    /// Get the maximum gap (only valid in single-night mode).
    ///
    /// Errors
    /// ------
    /// Returns an error if called in batch mode.
    pub fn max_gap(&self) -> Result<u8, FinkFatError> {
        match self {
            Self::SingleNight { max_gap, .. } => Ok(*max_gap),
            Self::BatchRange { .. } => Err(FinkFatError::Message(
                "max_gap() is only valid in SingleNight mode".to_string(),
            )),
        }
    }

    /// Get the start night of the range.
    ///
    /// Behavior
    /// --------
    /// - Single-night mode: returns the anchor night.
    /// - Batch mode: returns the start of the range.
    #[inline]
    pub fn start(&self) -> NightId {
        match self {
            Self::SingleNight { anchor, .. } => *anchor,
            Self::BatchRange { start, .. } => *start,
        }
    }

    /// Get the end night of the range.
    ///
    /// Behavior
    /// --------
    /// - Single-night mode: returns the anchor night.
    /// - Batch mode: returns the end of the range.
    #[inline]
    pub fn end(&self) -> NightId {
        match self {
            Self::SingleNight { anchor, .. } => *anchor,
            Self::BatchRange { end, .. } => *end,
        }
    }

    /// Length in number of nights (inclusive bounds).
    ///
    /// Example:
    /// - Single-night: always returns 1.
    /// - Batch with start=10, end=12: returns 3.
    #[inline]
    pub fn len(&self) -> u32 {
        let (start, end) = match self {
            Self::SingleNight { .. } => return 1,
            Self::BatchRange { start, end } => (start.0, end.0),
        };
        (end - start) + 1
    }

    /// Whether `night` falls within the defined range.
    ///
    /// Behavior
    /// --------
    /// - Single-night mode: checks if `night` falls within `[anchor - max_gap, anchor]`.
    /// - Batch mode: checks if `start <= night <= end`.
    #[inline]
    pub fn contains(&self, night: NightId) -> bool {
        let (start, end) = match self {
            Self::SingleNight { anchor, max_gap } => {
                (NightId(anchor.0.saturating_sub(*max_gap as u32)), *anchor)
            }
            Self::BatchRange { start, end } => (*start, *end),
        };
        start <= night && night <= end
    }

    /// Get the latest night from `available_nights` that falls within this mode's range.
    ///
    /// Behavior
    /// --------
    /// - Filters `available_nights` to keep only those within the range bounds.
    /// - Returns the maximum night ID, if any.
    ///
    /// Returns
    /// -------
    /// - `Some(NightId)` if at least one night is present in the range.
    /// - `None` if no nights fall within the range.
    #[inline]
    fn latest_night_in_range(&self, available_nights: &[NightId]) -> Option<NightId> {
        available_nights
            .iter()
            .copied()
            .filter(|&n| self.contains(n))
            .max()
    }

    /// Collect eligible left nights for pairing with a given right night.
    ///
    /// Overview
    /// --------
    /// Given a `right_night` (typically the latest night in the range) and a list
    /// of `available_nights`, this function returns all nights that are eligible
    /// to be paired as "left" candidates according to the pairing mode.
    ///
    /// Behavior by mode
    /// ----------------
    ///
    /// ### Single-night mode
    ///
    /// Constraints:
    /// 1. `left < right`
    /// 2. `right - left <= max_gap`
    ///
    /// No additional range constraint is applied.
    ///
    /// ### Multi-night batch mode
    ///
    /// Constraints:
    /// 1. `left < right`
    /// 2. `start <= left < right`
    ///
    /// Returns
    /// -------
    /// * `Vec<NightId>` – Sorted list of eligible left nights (increasing order).
    ///
    /// Determinism
    /// -----------
    /// Output is deterministic: the returned vector is sorted increasingly.
    pub fn eligible_left_nights(
        &self,
        available_nights: &[NightId],
        right_night: NightId,
    ) -> Vec<NightId> {
        let mut lefts: Vec<NightId> = match self {
            Self::SingleNight { max_gap, .. } => {
                // Single-night mode: use gap constraint only
                let min_left = right_night.0.saturating_sub(*max_gap as u32);

                available_nights
                    .iter()
                    .copied()
                    .filter(|&n| n < right_night && n.0 >= min_left)
                    .collect()
            }
            Self::BatchRange { start, .. } => {
                // Multi-night batch mode: constrain to range [start, right)
                available_nights
                    .iter()
                    .copied()
                    .filter(|&n| n < right_night && n >= *start)
                    .collect()
            }
        };

        lefts.sort();
        lefts
    }

    /// Generate `(left, right)` night pairs according to this pairing mode.
    ///
    /// Overview
    /// --------
    /// This function:
    /// 1. Finds the latest night present in `available_nights` within the range
    ///    (the "right" anchor).
    /// 2. Collects eligible "left" nights using the mode-specific logic.
    /// 3. Yields `(left, right)` pairs for each eligible left.
    ///
    /// Behavior by mode
    /// ----------------
    ///
    /// ### Single-night mode
    ///
    /// - Right = the anchor night (if present in `available_nights`).
    /// - Left candidates: all nights in `available_nights` within
    ///   `[anchor - max_gap, anchor)`.
    ///
    /// ### Multi-night batch mode
    ///
    /// - Right = latest night in `available_nights` within `[start, end]`.
    /// - Left candidates: all nights in `available_nights` within `[start, right)`.
    ///
    /// Edge cases
    /// ----------
    /// - If `available_nights` is empty: returns empty iterator.
    /// - If no night is present in the range: returns empty iterator.
    ///
    /// Ordering
    /// --------
    /// Output is deterministic:
    /// - `right` is constant (the selected latest night).
    /// - `left` values are emitted in strictly increasing order.
    ///
    /// Arguments
    /// ---------
    /// * `available_nights` – Owned vector of candidate nights (e.g., all nights in a store).
    ///   Ownership is required to avoid lifetime issues in the returned iterator.
    ///
    /// Returns
    /// -------
    /// * `impl Iterator<Item = (NightId, NightId)>` – Iterator yielding `(left, right)` pairs.
    ///
    /// Complexity
    /// ----------
    /// Let `N` be the number of available nights and `P` the number of pairs emitted.
    /// - Time: `O(N + P log P)` due to filtering and sorting.
    /// - Space: `O(P)` for the collected left nights.
    pub fn night_pairs_iter(
        self,
        available_nights: Vec<NightId>,
    ) -> impl Iterator<Item = (NightId, NightId)> {
        // Validate anchor presence in single-night mode and find the latest night in range.
        let right = match self {
            Self::SingleNight { anchor, .. } if !available_nights.contains(&anchor) => {
                return vec![].into_iter();
            }
            _ => match self.latest_night_in_range(&available_nights) {
                Some(night) => night,
                None => return vec![].into_iter(),
            },
        };

        // Collect eligible left nights and emit pairs.
        self.eligible_left_nights(&available_nights, right)
            .into_iter()
            .map(move |l| (l, right))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Eager version of `night_pairs_iter` that returns a `Vec`.
    ///
    /// See [`PairingMode::night_pairs_iter`] for detailed documentation.
    #[inline]
    pub fn night_pairs(self, available_nights: Vec<NightId>) -> Vec<(NightId, NightId)> {
        self.night_pairs_iter(available_nights).collect()
    }
}

impl fmt::Display for PairingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingleNight { anchor, max_gap } => {
                write!(f, "single({}±{})", anchor, max_gap)
            }
            Self::BatchRange { start, end } => {
                write!(f, "batch({}..={})", start, end)
            }
        }
    }
}

impl Default for PairingMode {
    fn default() -> Self {
        // Default to a single-night mode with anchor=0 and max_gap=1
        // (this is somewhat arbitrary but avoids having an invalid state)
        Self::SingleNight {
            anchor: NightId(0),
            max_gap: 1,
        }
    }
}

#[cfg(test)]
mod night_id_tests {
    use super::NightId;

    /// Check that Display prints the underlying integer as-is.
    #[test]
    fn test_display() {
        let nid = NightId::new(60312);
        assert_eq!(nid.to_string(), "60312");

        let other = NightId::new(42);
        assert_eq!(other.to_string(), "42");
    }

    /// Check ordering (Ord / PartialOrd) based on the inner value.
    #[test]
    fn test_ordering() {
        let mut v = vec![NightId::new(10), NightId::new(3), NightId::new(7)];
        v.sort();

        let values: Vec<u32> = v.into_iter().map(|n| n.value()).collect();
        assert_eq!(values, vec![3, 7, 10]);
    }

    /// Check JSON serialization / deserialization round-trip.
    #[test]
    fn test_serde_json_roundtrip() {
        let original = NightId::new(60312);

        let json = serde_json::to_string(&original).expect("serialize NightId to JSON");
        // Should be a bare number, e.g. "60312"
        assert_eq!(json, "60312");

        let decoded: NightId = serde_json::from_str(&json).expect("deserialize NightId from JSON");
        assert_eq!(decoded, original);
    }
}

#[cfg(test)]
mod pairing_mode_tests {
    use super::*;
    use proptest::prelude::*;

    fn nid(v: u32) -> NightId {
        NightId(v)
    }

    // -------------------------------------------------------------------------
    // Construction and validation
    // -------------------------------------------------------------------------

    #[test]
    fn single_night_rejects_zero_gap() {
        let result = PairingMode::single_night(nid(100), 0);
        assert!(result.is_err());
    }

    #[test]
    fn batch_range_rejects_inverted_bounds() {
        let result = PairingMode::batch_range(nid(100), nid(50));
        assert!(result.is_err());
    }

    #[test]
    fn batch_range_accepts_equal_bounds() {
        let result = PairingMode::batch_range(nid(50), nid(50));
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Mode queries
    // -------------------------------------------------------------------------

    #[test]
    fn is_single_night_correct() {
        let single = PairingMode::single_night(nid(100), 10).unwrap();
        assert!(single.is_single_night());
        assert!(!single.is_batch());

        let batch = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        assert!(!batch.is_single_night());
        assert!(batch.is_batch());
    }

    #[test]
    fn anchor_only_valid_in_single_mode() {
        let single = PairingMode::single_night(nid(100), 10).unwrap();
        assert_eq!(single.anchor().unwrap(), nid(100));

        let batch = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        assert!(batch.anchor().is_err());
    }

    #[test]
    fn max_gap_only_valid_in_single_mode() {
        let single = PairingMode::single_night(nid(100), 10).unwrap();
        assert_eq!(single.max_gap().unwrap(), 10);

        let batch = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        assert!(batch.max_gap().is_err());
    }

    // -------------------------------------------------------------------------
    // eligible_left_nights logic
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod eligible_left_nights_tests {
        use super::*;

        fn nid(v: u32) -> NightId {
            NightId(v)
        }

        // -------------------------------------------------------------------------
        // Single-night mode tests
        // -------------------------------------------------------------------------

        #[test]
        fn single_night_empty_available() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let lefts = mode.eligible_left_nights(&[], nid(100));
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn single_night_no_eligible_lefts() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(100), nid(110), nid(120)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn single_night_all_outside_gap() {
            let mode = PairingMode::single_night(nid(100), 5).unwrap();
            let available = vec![nid(80), nid(85), nid(90)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // All nights are more than 5 nights before 100
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn single_night_some_in_gap() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(80), nid(92), nid(95), nid(100)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // 80 is outside gap (100 - 80 = 20 > 10)
            // 92, 95 are within gap
            assert_eq!(lefts, vec![nid(92), nid(95)]);
        }

        #[test]
        fn single_night_exact_gap_boundary() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(89), nid(90), nid(91)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // 89 is outside (100 - 89 = 11 > 10)
            // 90 is exactly at boundary (100 - 90 = 10)
            // 91 is within gap
            assert_eq!(lefts, vec![nid(90), nid(91)]);
        }

        #[test]
        fn single_night_saturating_sub() {
            let mode = PairingMode::single_night(nid(5), 100).unwrap();
            let available = vec![nid(0), nid(1), nid(3), nid(4)];
            let lefts = mode.eligible_left_nights(&available, nid(5));
            // min_left = 5 - 100 saturates to 0
            // All nights < 5 are eligible
            assert_eq!(lefts, vec![nid(0), nid(1), nid(3), nid(4)]);
        }

        #[test]
        fn single_night_unsorted_input() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(95), nid(92), nid(98), nid(91)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // Output should be sorted
            assert_eq!(lefts, vec![nid(91), nid(92), nid(95), nid(98)]);
        }

        // -------------------------------------------------------------------------
        // Multi-night batch mode tests
        // -------------------------------------------------------------------------

        #[test]
        fn batch_range_empty_available() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let lefts = mode.eligible_left_nights(&[], nid(100));
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn batch_range_no_eligible_lefts() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(100), nid(110), nid(120)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn batch_range_all_outside_range() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(10), nid(20), nid(30), nid(40)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // All nights are below start=50
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn batch_range_some_in_range() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(40), nid(60), nid(80), nid(100), nid(110)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // 40 is below start
            // 60, 80 are in range [50, 100)
            // 100 is the right night (excluded)
            // 110 is above right
            assert_eq!(lefts, vec![nid(60), nid(80)]);
        }

        #[test]
        fn batch_range_all_in_range() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(50), nid(60), nid(70), nid(80), nid(90)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // All are in [50, 100) and < 100
            assert_eq!(lefts, vec![nid(50), nid(60), nid(70), nid(80), nid(90)]);
        }

        #[test]
        fn batch_range_exact_boundaries() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(49), nid(50), nid(99), nid(100)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // 49 is below start
            // 50 is at start (included)
            // 99 is within range
            // 100 is the right night (excluded)
            assert_eq!(lefts, vec![nid(50), nid(99)]);
        }

        #[test]
        fn batch_range_unsorted_input() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(80), nid(60), nid(90), nid(70)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // Output should be sorted
            assert_eq!(lefts, vec![nid(60), nid(70), nid(80), nid(90)]);
        }

        #[test]
        fn batch_range_equal_start_end() {
            let mode = PairingMode::batch_range(nid(50), nid(50)).unwrap();
            let available = vec![nid(40), nid(50), nid(60)];
            let lefts = mode.eligible_left_nights(&available, nid(50));
            // start = end = 50, so no lefts possible
            assert_eq!(lefts, vec![]);
        }

        // -------------------------------------------------------------------------
        // Edge cases
        // -------------------------------------------------------------------------

        #[test]
        fn right_not_in_available() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(90), nid(92), nid(95)];
            // Even if right=100 is not in available, we still check eligibility
            let lefts = mode.eligible_left_nights(&available, nid(100));
            assert_eq!(lefts, vec![nid(90), nid(92), nid(95)]);
        }

        #[test]
        fn right_equals_left_candidate() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(100)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // 100 cannot be left of itself
            assert_eq!(lefts, vec![]);
        }

        #[test]
        fn duplicates_in_available() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(90), nid(92), nid(92), nid(95), nid(95)];
            let lefts = mode.eligible_left_nights(&available, nid(100));
            // Duplicates should be preserved (caller's responsibility to dedup if needed)
            assert_eq!(lefts, vec![nid(90), nid(92), nid(92), nid(95), nid(95)]);
        }

        // -------------------------------------------------------------------------
        // Property-based tests
        // -------------------------------------------------------------------------

        proptest! {
            /// All returned lefts must be strictly less than right
            #[test]
            fn prop_all_lefts_less_than_right(
                nights in prop::collection::vec(0u32..1000, 0..100),
                right in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let lefts = mode.eligible_left_nights(&available, nid(right));

                for &left in &lefts {
                    prop_assert!(left < nid(right));
                }
            }

            /// Single-night: all lefts must respect gap constraint
            #[test]
            fn prop_single_night_lefts_respect_gap(
                nights in prop::collection::vec(0u32..1000, 0..100),
                right in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let lefts = mode.eligible_left_nights(&available, nid(right));

                let min_left = right.saturating_sub(gap as u32);

                for &left in &lefts {
                    prop_assert!(left.0 >= min_left);
                    prop_assert!(left.0 < right);
                }
            }

            /// Batch range: all lefts must be in [start, right)
            #[test]
            fn prop_batch_range_lefts_in_range(
                nights in prop::collection::vec(0u32..500, 0..100),
                start in 0u32..400,
                right in 400u32..500,
            ) {
                let mode = PairingMode::batch_range(nid(start), nid(right)).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let lefts = mode.eligible_left_nights(&available, nid(right));

                for &left in &lefts {
                    prop_assert!(left.0 >= start);
                    prop_assert!(left.0 < right);
                }
            }

            /// Output is always sorted
            #[test]
            fn prop_output_sorted(
                nights in prop::collection::vec(0u32..1000, 0..100),
                right in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let lefts = mode.eligible_left_nights(&available, nid(right));

                let mut sorted = lefts.clone();
                sorted.sort();

                prop_assert_eq!(lefts, sorted);
            }

            /// All returned lefts must be present in available
            #[test]
            fn prop_all_lefts_in_available(
                nights in prop::collection::vec(0u32..1000, 0..100),
                right in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let lefts = mode.eligible_left_nights(&available, nid(right));

                let available_set: std::collections::HashSet<_> = available.iter().collect();

                for &left in &lefts {
                    prop_assert!(available_set.contains(&left));
                }
            }

            /// Determinism: repeated calls yield same result
            #[test]
            fn prop_deterministic(
                nights in prop::collection::vec(0u32..1000, 0..100),
                right in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();

                let lefts1 = mode.eligible_left_nights(&available, nid(right));
                let lefts2 = mode.eligible_left_nights(&available, nid(right));

                prop_assert_eq!(lefts1, lefts2);
            }

            /// If right is very small, saturating_sub prevents underflow
            #[test]
            fn prop_saturating_sub_safe(
                nights in prop::collection::vec(0u32..20, 0..20),
                right in 0u32..10,
                gap in 1u8..255,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();

                // Should not panic
                let _lefts = mode.eligible_left_nights(&available, nid(right));
            }

            /// Comparison with brute-force reference
            #[test]
            fn prop_matches_brute_force(
                nights in prop::collection::vec(0u32..500, 0..50).prop_map(|v| {
                    let mut sorted = v;
                    sorted.sort();
                    sorted.dedup();
                    sorted
                }),
                right in 50u32..500,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(right), gap).unwrap();
                let available: Vec<NightId> = nights.iter().map(|&n| nid(n)).collect();

                let result = mode.eligible_left_nights(&available, nid(right));

                // Brute-force reference
                let min_left = right.saturating_sub(gap as u32);
                let mut expected: Vec<NightId> = available
                    .iter()
                    .copied()
                    .filter(|&n| n < nid(right) && n.0 >= min_left)
                    .collect();
                expected.sort();

                prop_assert_eq!(result, expected);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pairing logic
    // -------------------------------------------------------------------------

    mod pairing_tests {
        use super::*;

        /// Helper: brute-force reference implementation
        fn reference_pairs(available: &[NightId], mode: PairingMode) -> Vec<(NightId, NightId)> {
            // Find right
            let Some(&right) = available.iter().filter(|&&n| mode.contains(n)).max() else {
                return vec![];
            };

            let mut lefts: Vec<NightId> = match mode {
                PairingMode::SingleNight { max_gap, .. } => {
                    let min_left = right.0.saturating_sub(max_gap as u32);
                    available
                        .iter()
                        .copied()
                        .filter(|&n| n < right && n.0 >= min_left)
                        .collect()
                }
                PairingMode::BatchRange { start, .. } => available
                    .iter()
                    .copied()
                    .filter(|&n| n < right && n >= start)
                    .collect(),
            };

            lefts.sort();
            lefts.into_iter().map(|l| (l, right)).collect()
        }

        #[test]
        fn single_night_empty_available() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let pairs = mode.night_pairs(vec![]);
            assert_eq!(pairs, vec![]);
        }

        #[test]
        fn single_night_anchor_not_present() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let pairs = mode.night_pairs(vec![nid(50), nid(60)]);
            assert_eq!(pairs, vec![]);
        }

        #[test]
        fn single_night_basic() {
            let mode = PairingMode::single_night(nid(100), 10).unwrap();
            let available = vec![nid(85), nid(92), nid(100)];
            let pairs = mode.night_pairs(available);

            // 85 is outside gap (100 - 85 = 15 > 10)
            // 92 is within gap
            assert_eq!(pairs, vec![(nid(92), nid(100))]);
        }

        #[test]
        fn batch_range_basic() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(40), nid(60), nid(80), nid(100)];
            let pairs = mode.night_pairs(available);

            // right = 100
            // left candidates: 60, 80 (40 is outside range)
            assert_eq!(pairs, vec![(nid(60), nid(100)), (nid(80), nid(100))]);
        }

        #[test]
        fn batch_range_no_nights_in_range() {
            let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
            let available = vec![nid(10), nid(20), nid(30)];
            let pairs = mode.night_pairs(available);
            assert_eq!(pairs, vec![]);
        }

        #[test]
        fn saturating_sub_prevents_underflow() {
            let mode = PairingMode::single_night(nid(2), 250).unwrap();
            let available = vec![nid(0), nid(1), nid(2)];
            let pairs = mode.night_pairs(available);

            // min_left = 2 - 250 saturates to 0
            // eligible: 0, 1
            assert_eq!(pairs, vec![(nid(0), nid(2)), (nid(1), nid(2))]);
        }

        // Property-based tests

        proptest! {
            /// All pairs must respect left < right
            #[test]
            fn prop_left_less_than_right(
                nights in prop::collection::vec(0u32..1000, 0..50),
                anchor in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let pairs = mode.night_pairs(available);

                for (left, right) in pairs {
                    prop_assert!(left < right);
                }
            }

            /// Single-night pairs must respect gap constraint
            #[test]
            fn prop_single_night_gap_constraint(
                nights in prop::collection::vec(0u32..1000, 0..50),
                anchor in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let pairs = mode.night_pairs(available);

                for (left, right) in pairs {
                    let actual_gap = right.0 - left.0;
                    prop_assert!(actual_gap <= gap as u32);
                }
            }

            /// Batch range pairs must have left in range
            #[test]
            fn prop_batch_range_left_in_range(
                nights in prop::collection::vec(0u32..500, 0..50),
                start in 0u32..400,
                end in 400u32..500,
            ) {
                let mode = PairingMode::batch_range(nid(start), nid(end)).unwrap();
                let available: Vec<NightId> = nights.into_iter().map(nid).collect();
                let pairs = mode.night_pairs(available);

                for (left, _) in pairs {
                    prop_assert!(left.0 >= start);
                    prop_assert!(left.0 < end);
                }
            }

            /// Output matches reference implementation
            #[test]
            fn prop_matches_reference(
                nights in prop::collection::vec(0u32..1000, 0..50).prop_map(|v| {
                    let mut sorted = v;
                    sorted.sort();
                    sorted.dedup();
                    sorted
                }),
                anchor in 10u32..1000,
                gap in 1u8..100,
            ) {
                let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
                let available: Vec<NightId> = nights.iter().map(|&n| nid(n)).collect();

                let result = mode.night_pairs(available.clone());
                let expected = reference_pairs(&available, mode);

                if !result.is_empty() {
                    prop_assert_eq!(result, expected);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Display
    // -------------------------------------------------------------------------

    #[test]
    fn display_single_night() {
        let mode = PairingMode::single_night(nid(100), 10).unwrap();
        assert_eq!(mode.to_string(), "single(100±10)");
    }

    #[test]
    fn display_batch_range() {
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        assert_eq!(mode.to_string(), "batch(50..=100)");
    }
}

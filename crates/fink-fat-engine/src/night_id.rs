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

/// Inclusive range of nights processed as a single logical window.
///
/// This type is used to:
/// - scope disk persistence (e.g. journals / manifests) to a time span,
/// - restrict recomputation to a subset of nights,
/// - express sliding-window strategies in the pipeline/orchestrator.
///
/// Semantics
/// ---------
/// - `start` and `end` are **inclusive** bounds.
/// - `start <= end` must hold.
/// - The window does not assume that all intermediate nights exist in the input;
///   it only defines the acceptable ID range.
///
/// Typical usage
/// -------------
/// - `NightWindow { start: 60312, end: 60312 }` for a single-night run.
/// - `NightWindow { start: 60312, end: 60320 }` for a multi-night batch.
///
/// Notes
/// -----
/// - `NightId` is a logical identifier; if you derive it from MJD, be consistent
///   about the day boundary convention (e.g., noon-UTC vs midnight-UTC).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct NightWindow {
    /// Inclusive first night of the window.
    pub start: NightId,
    /// Inclusive last night of the window.
    pub end: NightId,
}

impl NightWindow {
    /// Create a new inclusive night window.
    ///
    /// Panics
    /// ------
    /// Panics if `start > end`.
    #[inline]
    pub fn new(start: NightId, end: NightId) -> Self {
        assert!(start <= end, "NightWindow requires start <= end");
        Self { start, end }
    }

    /// Create a new inclusive night window.
    ///
    /// Errors
    /// ------
    /// Returns an error if `start > end`.
    #[inline]
    pub fn try_new(start: NightId, end: NightId) -> Result<Self, FinkFatError> {
        if start > end {
            return Err(FinkFatError::Message(format!(
                "invalid NightWindow: start ({}) > end ({})",
                start, end
            )));
        }
        Ok(Self { start, end })
    }

    /// Create a single-night window.
    #[inline]
    pub fn single(night: NightId) -> Self {
        Self {
            start: night,
            end: night,
        }
    }

    /// Length in number of nights (inclusive bounds).
    ///
    /// Example: start=10, end=12 => len=3.
    #[inline]
    pub fn len(self) -> u32 {
        (self.end.0 - self.start.0) + 1
    }

    /// Whether this window contains `night`.
    #[inline]
    pub fn contains(self, night: NightId) -> bool {
        self.start <= night && night <= self.end
    }

    /// Whether the window is a single night.
    #[inline]
    pub fn is_single(self) -> bool {
        self.start == self.end
    }
}

impl fmt::Display for NightWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == self.end {
            write!(f, "{}", self.start)
        } else {
            write!(f, "{}..={}", self.start, self.end)
        }
    }
}

impl Default for NightWindow {
    fn default() -> Self {
        Self {
            start: NightId(0),
            end: NightId(0),
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

        let neg = NightId::new(42);
        assert_eq!(neg.to_string(), "42");
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
mod night_window_tests {
    use super::{NightId, NightWindow};

    use proptest::prelude::*;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn nid(v: u32) -> NightId {
        NightId::new(v)
    }

    // Strategy generating (start, end) with start <= end.
    fn night_window_bounds() -> impl Strategy<Value = (NightId, NightId)> {
        // Keep a wide enough range, but avoid overflows in derived computations.
        (0u32..=1_000_000u32, 0u32..=1_000_000u32).prop_map(|(a, b)| {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            (nid(lo), nid(hi))
        })
    }

    // Strategy generating a valid NightWindow directly.
    fn night_window_strategy() -> impl Strategy<Value = NightWindow> {
        night_window_bounds().prop_map(|(start, end)| NightWindow::new(start, end))
    }

    // -------------------------------------------------------------------------
    // Basic constructors and invariants
    // -------------------------------------------------------------------------

    #[test]
    fn new_single_night_is_valid() {
        let w = NightWindow::new(nid(42), nid(42));
        assert_eq!(w.start, nid(42));
        assert_eq!(w.end, nid(42));
        assert!(w.is_single());
        assert_eq!(w.len(), 1);
        assert!(w.contains(nid(42)));
        assert!(!w.contains(nid(41)));
        assert!(!w.contains(nid(43)));
    }

    #[test]
    fn single_constructor_matches_new() {
        let a = NightWindow::single(nid(60312));
        let b = NightWindow::new(nid(60312), nid(60312));
        assert_eq!(a, b);
        assert!(a.is_single());
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn new_multi_night_is_valid() {
        let w = NightWindow::new(nid(10), nid(12));
        assert_eq!(w.len(), 3);
        assert!(w.contains(nid(10)));
        assert!(w.contains(nid(11)));
        assert!(w.contains(nid(12)));
        assert!(!w.contains(nid(9)));
        assert!(!w.contains(nid(13)));
        assert!(!w.is_single());
    }

    #[test]
    fn try_new_rejects_inverted_bounds() {
        let err = NightWindow::try_new(nid(10), nid(3)).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("invalid NightWindow") || msg.contains("start"),
            "error should mention invalid window, got: {msg}"
        );
    }

    #[test]
    #[should_panic(expected = "NightWindow requires start <= end")]
    fn new_panics_on_inverted_bounds() {
        let _ = NightWindow::new(nid(10), nid(3));
    }

    // -------------------------------------------------------------------------
    // Display formatting
    // -------------------------------------------------------------------------

    #[test]
    fn display_single_night() {
        let w = NightWindow::single(nid(60312));
        assert_eq!(w.to_string(), "60312");
    }

    #[test]
    fn display_range() {
        let w = NightWindow::new(nid(60312), nid(60320));
        assert_eq!(w.to_string(), "60312..=60320");
    }

    // -------------------------------------------------------------------------
    // Default semantics
    // -------------------------------------------------------------------------

    #[test]
    fn default_is_zero_single_night() {
        let w = NightWindow::default();
        assert_eq!(w.start, nid(0));
        assert_eq!(w.end, nid(0));
        assert!(w.is_single());
        assert_eq!(w.len(), 1);
        assert!(w.contains(nid(0)));
        assert!(!w.contains(nid(1)));
    }

    // -------------------------------------------------------------------------
    // Serde JSON round-trip
    // -------------------------------------------------------------------------

    #[test]
    fn serde_json_roundtrip() {
        let original = NightWindow::new(nid(60312), nid(60320));

        let json = serde_json::to_string(&original).expect("serialize NightWindow to JSON");
        let decoded: NightWindow =
            serde_json::from_str(&json).expect("deserialize NightWindow from JSON");

        assert_eq!(decoded, original);
    }

    // -------------------------------------------------------------------------
    // Property-based tests (proptest)
    // -------------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_len_matches_inclusive_bounds((start, end) in night_window_bounds()) {
            let w = NightWindow::new(start, end);
            let expected = (end.0 - start.0) + 1;
            prop_assert_eq!(w.len(), expected);
            prop_assert!(w.len() >= 1);
        }

        #[test]
        fn prop_contains_includes_bounds((start, end) in night_window_bounds()) {
            let w = NightWindow::new(start, end);

            prop_assert!(w.contains(start));
            prop_assert!(w.contains(end));

            if start.0 > 0 {
                prop_assert!(!w.contains(NightId(start.0 - 1)));
            }
            // Avoid overflow on +1
            if end.0 < u32::MAX {
                prop_assert!(!w.contains(NightId(end.0 + 1)));
            }
        }

        #[test]
        fn prop_is_single_iff_start_eq_end((start, end) in night_window_bounds()) {
            let w = NightWindow::new(start, end);
            prop_assert_eq!(w.is_single(), start == end);
            prop_assert_eq!(w.is_single(), w.len() == 1);
        }

        #[test]
        fn prop_try_new_agrees_with_new_when_valid((start, end) in night_window_bounds()) {
            let w1 = NightWindow::new(start, end);
            let w2 = NightWindow::try_new(start, end).expect("try_new should succeed for start <= end");
            prop_assert_eq!(w1, w2);
        }

        #[test]
        fn prop_contains_is_monotonic_for_interior_points(w in night_window_strategy(), x in 0u32..=1_000_000u32) {
            // If x is inside, then any y between start..=end that equals x should be inside;
            // and if x is outside, it must violate one side.
            let inside = w.contains(NightId(x));
            if inside {
                prop_assert!(w.start.0 <= x && x <= w.end.0);
            } else {
                prop_assert!(x < w.start.0 || x > w.end.0);
            }
        }

        #[test]
        fn prop_display_format_is_stable(w in night_window_strategy()) {
            let s = w.to_string();
            if w.is_single() {
                prop_assert_eq!(s, w.start.to_string());
            } else {
                prop_assert_eq!(s, format!("{}..={}", w.start, w.end));
            }
        }
    }
}

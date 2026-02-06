use serde::{Deserialize, Serialize};
use std::fmt;

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

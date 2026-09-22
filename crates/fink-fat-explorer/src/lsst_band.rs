//! Single source of truth for fink-fat's LSST photometric band convention:
//! the integer `filter` index stored throughout the pipeline (`observations`
//! table, `ObservationRow::filter`) maps to the standard ugrizy band letters.
//!
//! Inverse of the offline prep scripts' `mapping_band = {"u": 0, "g": 1,
//! "r": 2, "i": 3, "z": 4, "y": 5}` (`test_exp/prep_lsst_alert.py`,
//! `test_exp/prep_lsst_eval_data.py`) — LSST-only, no other survey stores
//! photometry this way.

/// Map a `filter` index to its ugrizy band letter, or `None` if it's outside
/// the known `0..=5` range.
pub fn band_index_to_letter(filter: i16) -> Option<&'static str> {
    match filter {
        0 => Some("u"),
        1 => Some("g"),
        2 => Some("r"),
        3 => Some("i"),
        4 => Some("z"),
        5 => Some("y"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_known_index() {
        assert_eq!(band_index_to_letter(0), Some("u"));
        assert_eq!(band_index_to_letter(5), Some("y"));
    }

    #[test]
    fn rejects_unknown_index() {
        assert_eq!(band_index_to_letter(-1), None);
        assert_eq!(band_index_to_letter(6), None);
    }
}

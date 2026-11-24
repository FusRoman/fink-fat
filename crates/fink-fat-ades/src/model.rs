//! Domain types and pure, dependency-free transformations for ADES export:
//! `trkSub` normalization, band-code mapping, epoch conversion, singleton-night
//! filtering, non-blocking submission-quality advisories, and local
//! `submit.xsd` conformance checks. No XML serialization or network I/O lives
//! here — see `xml.rs` and `mpc_submission.rs`.

use serde::{Deserialize, Serialize};

use crate::error::AdesError;
use crate::format_epoch::iso_utc;

/// Maximum length of an ADES `trkSub` value (`submit.xsd`'s `BaseTrkSubType`).
const TRK_SUB_MAX_LEN: usize = 8;

/// A validated ADES `trkSub` value: at most 8 characters, `[-A-Za-z0-9_]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrkSub(String);

impl TrkSub {
    /// The validated `trkSub` string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Derive a valid ADES `trkSub` (at most 8 characters, `[-A-Za-z0-9_]`) from a
/// lineage's internal designation. Disallowed characters are stripped; if
/// more than 8 remain, the **trailing** 8 are kept, not the leading 8.
///
/// This direction matters: fink-fat's own `lineage_designation` format is
/// `"FF" + <4-digit year> + <12-char base-26 hash of the seeding track>`
/// (`fink-fat-engine::topocentric_kf::branching::branch_id::lineage_designation`).
/// Every lineage discovered in the same year shares the same 6-character
/// `"FF2025"` prefix, so keeping the leading 8 characters would keep that
/// shared prefix plus only the first 2 of the 12 hash characters — 26² ≈ 676
/// possible `trkSub` values shared across every lineage of the year, a
/// collision probability nowhere near the near-zero rate the hash was
/// designed to give. Keeping the trailing 8 characters instead keeps 8 of
/// the 12 hash characters (26⁸ ≈ 2×10¹¹ possible values), matching the
/// designation's own collision resistance far more closely.
///
/// # Arguments
/// * `designation` — the lineage designation to derive a `trkSub` from.
///
/// # Return
/// The validated [`TrkSub`].
///
/// # Errors
/// Returns [`AdesError::InvalidTrkSub`] if `designation` is empty, or if
/// every character is stripped by the allowed-charset filter (nothing valid
/// remains).
pub fn normalize_trk_sub(designation: &str) -> Result<TrkSub, AdesError> {
    if designation.is_empty() {
        return Err(AdesError::InvalidTrkSub {
            designation: designation.to_string(),
            reason: "designation is empty".to_string(),
        });
    }

    let filtered: String = designation
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();

    if filtered.is_empty() {
        return Err(AdesError::InvalidTrkSub {
            designation: designation.to_string(),
            reason: "no character of the designation is valid in a trkSub ([-A-Za-z0-9_])"
                .to_string(),
        });
    }

    // Every character kept by the filter above is a single-byte ASCII
    // character, so byte-slicing from the end is safe (no risk of splitting
    // a multi-byte UTF-8 sequence).
    let truncated = if filtered.len() > TRK_SUB_MAX_LEN {
        filtered[filtered.len() - TRK_SUB_MAX_LEN..].to_string()
    } else {
        filtered
    };

    Ok(TrkSub(truncated))
}

/// Map fink-fat's internal `filter` index to an ADES `band` code
/// (`submit.xsd`'s `BandType`, at most 3 alphanumeric characters), via the
/// shared [`crate::lsst_band::band_index_to_letter`] mapping. Unlike that
/// function's callers in the lineage page UI (which fall back to a display
/// string for an unrecognized index), ADES export needs a hard error.
///
/// # Arguments
/// * `filter` — fink-fat's internal photometric band index.
///
/// # Return
/// The ADES band letter.
///
/// # Errors
/// Returns [`AdesError::UnknownBand`] for any filter index outside `0..=5`.
pub fn band_index_to_ades_band(filter: i16) -> Result<&'static str, AdesError> {
    crate::lsst_band::band_index_to_letter(filter).ok_or(AdesError::UnknownBand { filter })
}

/// Convert an MJD(TT) epoch to the ISO-8601 UTC string ADES expects for
/// `obsTime`. Delegates to [`crate::format_epoch::iso_utc`] (the same TT→UTC
/// conversion also used for human-facing display), kept as a separately-named
/// wrapper so call sites read as "ADES obsTime" and so a future ADES-specific
/// formatting quirk can be special-cased here without touching the shared
/// display helper.
///
/// `iso_utc` (via `hifitime::Epoch::to_isoformat`, which truncates its
/// formatted string to exactly 26 characters) never includes a trailing
/// `Z`, but `submit.xsd`'s `TimeType` requires one
/// (`\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z`) — confirmed the hard
/// way: a real MPC submission was rejected with "Element 'obsTime':
/// '...254790' is not a valid value of the union type 'TimeType'" for
/// exactly this reason. Appended here defensively (checking rather than
/// assuming `iso_utc` never adds one) instead of changing `iso_utc` itself,
/// which is shared with plain human-facing display where the missing `Z` is
/// harmless.
///
/// # Arguments
/// * `mjd_tt` — the observation epoch, Modified Julian Date, Terrestrial Time.
///
/// # Return
/// The ADES `obsTime` string, always ending in `Z`.
///
/// # Errors
/// Returns [`AdesError::ObsTimeConversion`] if `mjd_tt` is not finite.
pub fn mjd_tt_to_ades_obs_time(mjd_tt: f64) -> Result<String, AdesError> {
    if !mjd_tt.is_finite() {
        return Err(AdesError::ObsTimeConversion {
            mjd_tt,
            reason: "MJD(TT) value is not finite".to_string(),
        });
    }
    let formatted = iso_utc(mjd_tt);
    Ok(if formatted.ends_with('Z') {
        formatted
    } else {
        format!("{formatted}Z")
    })
}

/// User-supplied ADES header fields not tracked by the fink-fat pipeline —
/// submitter identity, telescope description, acknowledgement contact —
/// collected once (via the export modal, or a `fink-fat submit`
/// `--submitter-config` file) before every submission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdesHeaderInput {
    pub submitter_name: String,
    pub submitter_institution: Option<String>,
    pub observers: Vec<String>,
    /// Required and non-empty: `submit.xsd`'s `ObsContextType` places
    /// `measurers` without `minOccurs="0"`, unlike `observers`.
    pub measurers: Vec<String>,
    pub telescope_design: String,
    pub telescope_aperture: String,
    pub telescope_detector: String,
    pub ast_cat: String,
    pub mode: String,
    pub funding_source: Option<String>,
    /// Required by MPC's submission form ("Acknowledgment message
    /// (required)").
    pub ack_message: String,
    /// Required by MPC's submission form ("Acknowledgment email
    /// address (required)").
    pub ac2_email: String,
}

/// One real observation belonging to a lineage's best branch, in track order.
/// Deliberately DB-client-agnostic (no `sqlx`/`postgres`-specific derives):
/// `fink-fat-explorer` maps its `sqlx::FromRow` query rows into this type,
/// and the `fink-fat submit` CLI maps its synchronous `postgres::Row`s into
/// the same type, so every pure function downstream of a fetch (this
/// module, `schema_validation`, `xml`) is written once and shared.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationRow {
    pub id: i64,
    pub object_id: String,
    pub position: i32,
    pub mjd_tt: f64,
    pub ra: f64,
    pub ra_err: f64,
    pub dec: f64,
    pub dec_err: f64,
    pub magnitude: f64,
    pub mag_err: f64,
    pub filter: i16,
    pub mpc_code_obs: String,
    /// Observatory-longitude-aware observation night bucket, pre-computed at
    /// ingestion (`observations.night_id`) — used by the ADES export to
    /// detect nights with a single observation ("singleton" nights, which
    /// the MPC rejects an entire batch for containing).
    pub night_id: i64,
}

/// Resolves a lineage's best branch (highest `cumulative_llr`, NaN/Infinity
/// treated as the lowest possible value so a corrupted value can never win).
/// Bind `$1 = lineage_designation`; returns at most one row, `branch_id`.
/// Shared verbatim by `fink-fat-explorer::orbit_fit::run::resolve_best_branch_id`
/// (async `sqlx`) and the `fink-fat submit` CLI (synchronous `postgres`), so
/// both always agree on which branch a `lineage_designation` resolves to.
pub const RESOLVE_BEST_BRANCH_QUERY: &str = "
    SELECT branch_id
    FROM branches
    WHERE lineage_designation = $1
    ORDER BY (
        CASE
            WHEN cumulative_llr = 'NaN'::double precision THEN 0
            WHEN cumulative_llr = 'Infinity'::double precision THEN 0
            WHEN cumulative_llr = '-Infinity'::double precision THEN 0
            ELSE cumulative_llr
        END
    ) DESC
    LIMIT 1
";

/// Fetches one branch's observations in track order, the exact column set
/// needed to build an [`ObservationRow`]. Bind `$1 = branch_id`. Shared
/// verbatim by `fink-fat-explorer::lineage_page::observations_table::fetch_branch_observations`
/// (async `sqlx`) and the `fink-fat submit` CLI (synchronous `postgres`).
pub const FETCH_BRANCH_OBSERVATIONS_QUERY: &str = "
    SELECT o.id, o.object_id, bo.position, o.mjd_tt, o.ra, o.ra_err, o.dec, o.dec_err,
           o.magnitude, o.mag_err, o.filter, o.mpc_code_obs, o.night_id
    FROM branch_observations bo
    JOIN observations o ON o.id = bo.obs_id
    WHERE bo.branch_id = $1
    ORDER BY bo.position
";

/// One observation together with the night bucket it belongs to, from the
/// database's pre-computed, observatory-longitude-aware `night_id` column.
#[derive(Clone, Debug, PartialEq)]
pub struct NightObservation {
    pub night_id: i64,
    pub observation: ObservationRow,
}

/// How many observations/nights were removed from an ADES export because
/// they were the sole observation of their night — the MPC guide to
/// astrometry states that a batch containing such a "singleton" is rejected
/// in its entirety.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingletonNightSummary {
    pub singleton_night_count: usize,
    pub removed_observation_count: usize,
}

/// Partition observations into (kept, singleton summary), removing every
/// observation whose night contains exactly one observation. Pure data
/// filter — the observations kept here are what `build_ades_document` (in
/// `xml.rs`) must be built from, so a singleton observation never appears in
/// the generated XML.
///
/// # Arguments
/// * `observations` — every candidate observation for one lineage's export.
///
/// # Return
/// `(kept, summary)`: the observations that survive, and a count of what was
/// removed.
pub fn remove_singleton_nights(
    observations: &[NightObservation],
) -> (Vec<NightObservation>, SingletonNightSummary) {
    let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for obs in observations {
        *counts.entry(obs.night_id).or_insert(0) += 1;
    }

    let singleton_nights: std::collections::HashSet<i64> = counts
        .iter()
        .filter(|&(_, &count)| count == 1)
        .map(|(&night_id, _)| night_id)
        .collect();

    let kept: Vec<NightObservation> = observations
        .iter()
        .filter(|obs| !singleton_nights.contains(&obs.night_id))
        .cloned()
        .collect();

    let summary = SingletonNightSummary {
        singleton_night_count: singleton_nights.len(),
        removed_observation_count: observations.len() - kept.len(),
    };

    (kept, summary)
}

/// Minimum number of observations per night recommended by the MPC guide to
/// astrometry ("three to five observations of each object from each night
/// should be included").
const RECOMMENDED_MIN_OBSERVATIONS_PER_NIGHT: usize = 3;

/// Minimum number of distinct nights recommended by the MPC guide to
/// astrometry ("every object must be observed on two distinct nights").
const RECOMMENDED_MIN_NIGHTS: usize = 2;

/// Non-blocking submission-quality warnings, evaluated on the observations
/// that survive singleton-night removal.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SubmissionAdvisory {
    pub warnings: Vec<String>,
}

/// Evaluate the MPC guide to astrometry's non-blocking recommendations
/// (distinct from `submit.xsd`'s strict format rules, checked separately in
/// [`crate::schema_validation::check_local_schema_violations`]) against the
/// observations that remain after [`remove_singleton_nights`].
///
/// # Arguments
/// * `kept` — the observations surviving singleton-night removal.
///
/// # Return
/// The advisory. Never an error: this is a report, not a validation gate.
pub fn check_submission_recommendations(kept: &[NightObservation]) -> SubmissionAdvisory {
    let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for obs in kept {
        *counts.entry(obs.night_id).or_insert(0) += 1;
    }

    let mut warnings = Vec::new();

    if counts.len() < RECOMMENDED_MIN_NIGHTS {
        warnings.push(format!(
            "only {} distinct night(s) after removing singleton nights — the MPC \
             recommends observing an object on at least {RECOMMENDED_MIN_NIGHTS} distinct \
             nights, preferably less than a week apart",
            counts.len()
        ));
    }

    let under_recommended_nights = counts
        .values()
        .filter(|&&count| count < RECOMMENDED_MIN_OBSERVATIONS_PER_NIGHT)
        .count();
    if under_recommended_nights > 0 {
        warnings.push(format!(
            "{under_recommended_nights} night(s) have fewer than the \
             {RECOMMENDED_MIN_OBSERVATIONS_PER_NIGHT} observations per night the MPC \
             recommends"
        ));
    }

    SubmissionAdvisory { warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation_row(id: i64) -> ObservationRow {
        ObservationRow {
            id,
            object_id: format!("obj{id}"),
            position: 0,
            mjd_tt: 60000.0 + id as f64,
            ra: 1.0,
            ra_err: 0.0,
            dec: 0.5,
            dec_err: 0.0,
            magnitude: 19.0,
            mag_err: 0.1,
            filter: 2,
            mpc_code_obs: "I41".to_string(),
            night_id: 0,
        }
    }

    fn night_obs(night_id: i64, id: i64) -> NightObservation {
        NightObservation {
            night_id,
            observation: ObservationRow {
                night_id,
                ..observation_row(id)
            },
        }
    }

    #[test]
    fn normalize_trk_sub_rejects_empty_designation() {
        assert!(matches!(
            normalize_trk_sub(""),
            Err(AdesError::InvalidTrkSub { .. })
        ));
    }

    #[test]
    fn normalize_trk_sub_rejects_designation_with_no_valid_char() {
        assert!(matches!(
            normalize_trk_sub("!!!"),
            Err(AdesError::InvalidTrkSub { .. })
        ));
    }

    #[test]
    fn normalize_trk_sub_keeps_short_valid_designation_unchanged() {
        let trk = normalize_trk_sub("FF2024").unwrap();
        assert_eq!(trk.as_str(), "FF2024");
    }

    #[test]
    fn normalize_trk_sub_truncates_long_designation_keeping_the_tail() {
        let trk = normalize_trk_sub("ABCDEFGHIJ").unwrap();
        assert_eq!(trk.as_str(), "CDEFGHIJ");
    }

    #[test]
    fn normalize_trk_sub_keeps_the_high_entropy_hash_suffix_not_the_shared_year_prefix() {
        // Realistic `lineage_designation`: "FF" + year + 12-char base-26
        // hash. Every lineage of the same year shares "FF2025", so the
        // trkSub must come from the hash tail, not that shared prefix.
        let trk = normalize_trk_sub("FF2025abcdefghijkl").unwrap();
        assert_eq!(trk.as_str(), "efghijkl");
    }

    #[test]
    fn normalize_trk_sub_strips_disallowed_characters_then_truncates() {
        let trk = normalize_trk_sub("FF 2024 AB!").unwrap();
        assert_eq!(trk.as_str(), "FF2024AB");
    }

    #[test]
    fn band_index_to_ades_band_maps_known_indices() {
        assert_eq!(band_index_to_ades_band(0).unwrap(), "u");
        assert_eq!(band_index_to_ades_band(5).unwrap(), "y");
    }

    #[test]
    fn band_index_to_ades_band_rejects_unknown_index() {
        assert!(matches!(
            band_index_to_ades_band(-1),
            Err(AdesError::UnknownBand { filter: -1 })
        ));
        assert!(matches!(
            band_index_to_ades_band(6),
            Err(AdesError::UnknownBand { filter: 6 })
        ));
    }

    #[test]
    fn mjd_tt_to_ades_obs_time_rejects_non_finite() {
        assert!(mjd_tt_to_ades_obs_time(f64::NAN).is_err());
        assert!(mjd_tt_to_ades_obs_time(f64::INFINITY).is_err());
    }

    #[test]
    fn mjd_tt_to_ades_obs_time_appends_z_missing_from_iso_utc() {
        let mjd_tt = 60310.5;
        let iso = iso_utc(mjd_tt);
        assert!(!iso.ends_with('Z'), "test assumption broken: {iso}");
        assert_eq!(mjd_tt_to_ades_obs_time(mjd_tt).unwrap(), format!("{iso}Z"));
    }

    #[test]
    fn remove_singleton_nights_keeps_everything_when_no_singleton() {
        let observations = vec![
            night_obs(1, 1),
            night_obs(1, 2),
            night_obs(2, 3),
            night_obs(2, 4),
        ];
        let (kept, summary) = remove_singleton_nights(&observations);
        assert_eq!(kept.len(), 4);
        assert_eq!(summary, SingletonNightSummary::default());
    }

    #[test]
    fn remove_singleton_nights_removes_only_the_singleton_night() {
        let observations = vec![night_obs(1, 1), night_obs(1, 2), night_obs(2, 3)];
        let (kept, summary) = remove_singleton_nights(&observations);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|o| o.night_id == 1));
        assert_eq!(
            summary,
            SingletonNightSummary {
                singleton_night_count: 1,
                removed_observation_count: 1,
            }
        );
    }

    #[test]
    fn remove_singleton_nights_can_remove_everything() {
        let observations = vec![night_obs(1, 1), night_obs(2, 2)];
        let (kept, summary) = remove_singleton_nights(&observations);
        assert!(kept.is_empty());
        assert_eq!(
            summary,
            SingletonNightSummary {
                singleton_night_count: 2,
                removed_observation_count: 2,
            }
        );
    }

    #[test]
    fn remove_singleton_nights_keeps_a_night_with_exactly_two_observations() {
        let observations = vec![night_obs(1, 1), night_obs(1, 2)];
        let (kept, summary) = remove_singleton_nights(&observations);
        assert_eq!(kept.len(), 2);
        assert_eq!(summary, SingletonNightSummary::default());
    }

    #[test]
    fn check_submission_recommendations_warns_on_single_night() {
        let kept = vec![night_obs(1, 1), night_obs(1, 2), night_obs(1, 3)];
        let advisory = check_submission_recommendations(&kept);
        assert_eq!(advisory.warnings.len(), 1);
        assert!(advisory.warnings[0].contains("distinct night"));
    }

    #[test]
    fn check_submission_recommendations_is_silent_when_recommendations_met() {
        let kept = vec![
            night_obs(1, 1),
            night_obs(1, 2),
            night_obs(1, 3),
            night_obs(2, 4),
            night_obs(2, 5),
            night_obs(2, 6),
        ];
        let advisory = check_submission_recommendations(&kept);
        assert!(advisory.warnings.is_empty());
    }

    #[test]
    fn check_submission_recommendations_warns_on_under_recommended_night() {
        let kept = vec![
            night_obs(1, 1),
            night_obs(1, 2),
            night_obs(2, 3),
            night_obs(2, 4),
            night_obs(2, 5),
        ];
        let advisory = check_submission_recommendations(&kept);
        assert!(advisory.warnings.iter().any(|w| w.contains("fewer than")));
    }
}

//! # Track identifiers (`TrackId`)
//!
//! This module provides **deterministic** and **persistent** identifiers for
//! trajectory hypotheses (“tracks”), designed to be:
//!
//! - **Human-readable**.
//! - **Deterministic**: the same track content yields the same identifier across runs.
//! - **Persistent**: no dependence on wall-clock time, RNG, thread scheduling, or counters.
//!
//! The output is intended for:
//! - logging and debugging (stable labels),
//! - persistence on disk (stable file / record identifiers),
//! - cross-run reproducibility (recomputing a run produces the same ids),
//! - downstream tooling (plots, metrics, exports).
//!
//! ## Identifier format
//!
//! The string representation is:
//!
//! ```text
//! TRK{YYYY}{suffix}
//! ```
//!
//! Example:
//!
//! ```text
//! TRK2026abvurssmkq
//! ```
//!
//! Where:
//! - `TRK` is a fixed prefix.
//! - `YYYY` is a 4-digit Gregorian year.
//! - `suffix` is a fixed-length alphabetic token (`a`–`z`) derived from a stable hash.
//!
//! The goal is to remain compact while being easy to copy/paste and visually scan.
//!
//! ## How the year is chosen
//!
//! The year is derived from the **earliest alert epoch** (`Alert.mjd_tt`) among all
//! alerts referenced by the track nodes:
//!
//! - Each [`SeedNode`] stores `members: Vec<&Alert>` (borrowed alert references).
//! - The year prefix is computed from the minimum `mjd_tt` across all those members.
//!
//! This choice ensures that the year does **not** depend on any meaning attached to
//! `NightId` (which may not always be an MJD day in all pipelines).
//!
//! **Why "earliest alert"?**
//! - It is robust to input ordering issues (we take a minimum).
//! - It is stable across runs.
//! - It is consistent with the notion that a track “starts” at the earliest detection.
//!
//! ## How the suffix is computed
//!
//! The suffix is derived from an `ahash` hasher configured with **fixed seeds**:
//!
//! - We hash the ordered list of [`SeedKey`] extracted from the ordered `nodes` slice.
//! - The node order is therefore part of the identity.
//! - A fixed-length base-26 encoding maps the final `u64` hash into `a`–`z` characters.
//!
//! The seeds are constants and **must not be changed** once identifiers are in use,
//! otherwise previously persisted tracks will no longer match recomputed identifiers.
//!
//! ### Stability caveat
//!
//! The construction is deterministic across runs as long as:
//! - the hashing seeds remain unchanged, and
//! - the `ahash` implementation (crate version / algorithm) remains unchanged.
//!
//! If long-term stability across dependency upgrades is required, prefer a fixed,
//! standardized hash algorithm (e.g. BLAKE3) or an internal “frozen” hash (e.g. FNV-1a).
//!
//! ## Expected invariants
//!
//! - `track_id_from_nodes` expects `nodes` to be the **track order** used elsewhere
//!   (typically increasing time). The function does not reorder nodes.
//! - Each `SeedNode` is expected to contain at least one alert member.
//!
//! These invariants are enforced with assertions / panics:
//! - empty input slices are rejected,
//! - tracks without alerts are rejected.
//!
//! ## Performance notes
//!
//! - Year extraction is `O(total_members)` (scans alerts to find a minimum epoch).
//! - Suffix hashing is `O(n_nodes)` (hash each `SeedKey` once).
//! - The encoding step is `O(width)` where `width` is a small constant.
//!
//! This is designed to be negligible compared to solver and scoring costs.
//!
//! ## API overview
//!
//! - [`TrackId`] – newtype wrapper around the string identifier.
//! - [`track_id_from_nodes`] – main entrypoint in most pipelines.
//! - [`track_id_from_seed_keys_with_year`] – lower-level helper when the year is already known.
//!
//! Internal helpers:
//! - `earliest_alert_mjd_tt` – scan members and return the minimum epoch.
//! - `mjd_to_year` – convert MJD(TT) to Gregorian year (civil-year extraction).
//! - `encode_base26_u64` – fixed-width base-26 alphabet encoding.
//!
//! ## Examples
//!
//! Generating an id from a track hypothesis nodes:
//!
//! ```ignore
//! let id = track_id_from_nodes(&track.nodes);
//! println!("{}", id.as_str());
//! ```
//!
//! Generating an id when year is already computed externally:
//!
//! ```ignore
//! let id = track_id_from_seed_keys_with_year(&seed_keys, 2026);
//! ```
//!
//! The string can be stored directly, or wrapped in [`TrackId`] for type safety.

use std::hash::BuildHasher;
use std::hash::{Hash, Hasher};

use ahash::RandomState;

use crate::{persistence::seed_node::SeedKey, seeding::SeedNode};

/// Identifier for a trajectory hypothesis.
///
/// This is a **newtype wrapper** around a `String` to provide:
/// - stronger typing in APIs,
/// - easier refactors (if a compact binary form is introduced later),
/// - optional trait derivations (`Eq`, `Hash`, etc.) without affecting call sites.
///
/// The inner string follows the module format:
///
/// ```text
/// TRK{YYYY}{suffix}
/// ```
///
/// See the module-level documentation for details.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TrackId(pub String);

impl TrackId {
    /// Return the identifier as a string slice.
    ///
    /// This is the preferred way to access the underlying representation for
    /// logging, serialization, or persistence.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Compute a deterministic track ID from ordered seed keys and an explicit year.
///
/// This is a lower-level helper used when:
/// - the caller already computed the year (e.g., from alert metadata),
/// - or the caller wants to force a specific year prefix.
///
/// Determinism
/// -----------
/// Given the same `(year, keys)` pair, the returned identifier is deterministic:
/// - it does not depend on thread scheduling,
/// - it does not depend on wall-clock time,
/// - it does not depend on randomness.
///
/// Canonicalization requirement
/// ----------------------------
/// The slice `keys` must be in a deterministic order. In typical usage this order
/// is the track node order (time-ordered).
///
/// Hash construction
/// -----------------
/// - A deterministic `ahash::RandomState` is built from fixed seeds.
/// - The number of keys is hashed first to avoid ambiguity across concatenations.
/// - Each `SeedKey` is then hashed in order.
/// - The resulting `u64` is encoded as base-26 letters (`a`–`z`) with fixed width.
///
/// Arguments
/// ---------
/// * `keys` – Ordered list of seed identifiers defining the track content.
/// * `year` – 4-digit Gregorian year to embed in the identifier.
///
/// Return
/// ------
/// * `TrackId` – A `TRK{YYYY}{suffix}` identifier.
///
/// Panics
/// ------
/// Panics if `keys` is empty. The empty track case is considered a logic error.
pub fn track_id_from_seed_keys_with_year(keys: &[SeedKey], year: u32) -> TrackId {
    assert!(
        !keys.is_empty(),
        "Cannot generate TrackId from empty key list"
    );

    // -------------------------------------------------------------------------
    // Deterministic hashing configuration
    // -------------------------------------------------------------------------
    //
    // These seeds define the stable hashing behavior within the project.
    // They MUST remain constant once identifiers are used in persisted outputs.
    //
    // If these values change, previously persisted TrackIds will no longer match
    // recomputed identifiers.
    const K1: u64 = 0x6A09_E667_F3BC_C909;
    const K2: u64 = 0xBB67_AE85_84CA_A73B;
    const K3: u64 = 0x3C6E_F372_FE94_F82B;
    const K4: u64 = 0xA54F_F53A_5F1D_36F1;

    let build_hasher = RandomState::with_seeds(K1, K2, K3, K4);
    let mut hasher = build_hasher.build_hasher();

    // Hash length first to avoid ambiguity:
    // e.g. keys=[A,B] should not collide with keys=[A,B,C] truncated in some way.
    (keys.len() as u64).hash(&mut hasher);

    // Hash ordered keys (order-sensitive).
    for k in keys {
        k.hash(&mut hasher);
    }

    let h = hasher.finish();

    // 12 letters encodes ~56 bits worth of information in base-26.
    // This is compact, while keeping collisions unlikely for
    // practical track volumes.
    let suffix = encode_base26_u64(h, 12);

    TrackId(format!("TRK{}{}", year, suffix))
}

/// Compute a deterministic track ID from the ordered track nodes.
///
/// This is the main entrypoint used by solvers / trajectory builders.
///
/// Steps
/// -----
/// 1) Scan all alert members referenced by the nodes and compute:
///    `mjd_min = min(alert.mjd_tt)`.
/// 2) Convert `mjd_min` into a 4-digit Gregorian year using [`mjd_to_year`].
/// 3) Extract the ordered list of `SeedKey` from the ordered nodes slice.
/// 4) Produce the final identifier using [`track_id_from_seed_keys_with_year`].
///
/// Arguments
/// ---------
/// * `nodes` – Ordered list of seed nodes forming the track. The order is part
///   of the identity and should match the solver’s notion of track order.
///
/// Return
/// ------
/// * `TrackId` – A `TRK{YYYY}{suffix}` identifier.
///
/// Panics
/// ------
/// Panics if `nodes` is empty, or if nodes contain no alerts (members list empty).
pub fn track_id_from_nodes<'seed_lf, 'alert_lf>(
    nodes: &[&'seed_lf SeedNode<'alert_lf>],
) -> TrackId {
    // 1) Year prefix from earliest alert epoch
    let mjd_min = earliest_alert_mjd_tt(nodes);
    let year = mjd_to_year(mjd_min);

    // 2) Suffix from ordered seed keys
    let mut keys = Vec::with_capacity(nodes.len());
    for n in nodes {
        keys.push(n.core.key);
    }

    track_id_from_seed_keys_with_year(&keys, year)
}

/// Return the minimum alert epoch (MJD TT) among all alerts referenced by nodes.
///
/// This helper makes the year computation robust to:
/// - nodes not being perfectly ordered,
/// - members not being perfectly ordered inside each node.
///
/// It scans *all* alert members referenced by the nodes and returns the minimum.
/// The result is used to determine the year prefix.
///
/// Arguments
/// ---------
/// * `nodes` – Track nodes whose alerts are scanned.
///
/// Return
/// ------
/// * `f64` – The minimum `mjd_tt` found among all referenced alerts.
///
/// Panics
/// ------
/// Panics if no alert member is present at all (track nodes without alerts).
fn earliest_alert_mjd_tt<'seed_lf, 'alert_lf>(nodes: &[&'seed_lf SeedNode<'alert_lf>]) -> f64 {
    let mut best: Option<f64> = None;

    for node in nodes {
        for &a in &node.members {
            let mjd = a.mjd_tt;
            best = match best {
                None => Some(mjd),
                Some(cur) => Some(cur.min(mjd)),
            };
        }
    }

    best.expect("Track nodes must contain at least one alert member")
}

/// Convert MJD (TT) to Gregorian year.
///
/// This function is intentionally limited in scope:
/// it only extracts the civil year corresponding to the epoch, which is
/// sufficient to build the `YYYY` prefix in track identifiers.
///
/// Algorithm
/// ---------
/// - Convert MJD to JD:
///
///   ```text
///   JD = MJD + 2400000.5
///   ```
///
/// - Convert JD to the Gregorian calendar year using a standard
///   Julian Day Number → Gregorian conversion.
///
/// Accuracy and conventions
/// ------------------------
/// - Uses the common civil-day convention `Z = floor(JD + 0.5)`.
/// - Sufficient for year extraction in survey-like MJD ranges.
/// - Does not attempt to model leap seconds or time scale subtleties:
///   the input is MJD(TT) and we only need the year.
///
/// Arguments
/// ---------
/// * `mjd` – Epoch in MJD TT (days).
///
/// Return
/// ------
/// * `u32` – Gregorian year (e.g. 2026).
fn mjd_to_year(mjd: f64) -> u32 {
    let jd = mjd + 2_400_000.5;

    let z = (jd + 0.5).floor();

    let mut a = z;
    if z >= 2_299_161.0 {
        let alpha = ((z - 1_867_216.25) / 36_524.25).floor();
        a = z + 1.0 + alpha - (alpha / 4.0).floor();
    }

    let b = a + 1524.0;
    let c = ((b - 122.1) / 365.25).floor();
    let d = (365.25 * c).floor();
    let e = ((b - d) / 30.6001).floor();

    let year = if e < 14.0 { c - 4716.0 } else { c - 4715.0 };
    year as u32
}

/// Encode a `u64` into a fixed-width base-26 alphabetic string (`a`–`z`).
///
/// Encoding scheme
/// ---------------
/// - `0` maps to `"aaaa...a"` (all `a`).
/// - The last character varies fastest.
/// - This is equivalent to representing the number in base-26 where digits are
///   mapped to letters (`0 -> 'a'`, `25 -> 'z'`).
///
/// The output is **fixed-width**: it always contains exactly `width` characters.
/// This ensures stable formatting and easy parsing.
///
/// Arguments
/// ---------
/// * `value` – Integer value to encode.
/// * `width` – Number of base-26 digits to emit.
///
/// Return
/// ------
/// * `String` – Alphabetic base-26 representation with length `width`.
///
/// Notes
/// -----
/// - If `width` is too small, higher-order information is truncated (because the
///   repeated division discards remaining digits). This is intentional for compact IDs.
/// - Choose `width` based on the desired collision envelope (tradeoff with length).
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
mod track_id_tests {
    use super::*;
    use crate::night_id::NightId;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // -------------------------------------------------------------------------
    // Helpers: deterministic test data builders
    // -------------------------------------------------------------------------

    /// Build a SeedKey with deterministic content.
    ///
    /// Adapt field names if your SeedKey differs.
    fn make_seed_key(night_id: u32, idx_in_night: u32) -> SeedKey {
        // Assumes: SeedKey { night_id: NightId, idx_in_night: u32 } or similar.
        SeedKey {
            night_id: crate::night_id::NightId(night_id),
            idx_in_night,
        }
    }

    /// Build a minimal but valid fake alert for seeding.
    ///
    /// The seed-building code (`SeedNode::from_pair`) requires plausible values
    /// for position/time; we keep everything simple and deterministic.
    fn make_alert(mjd_tt: f64, ra_rad: f64, dec_rad: f64, band: u8) -> crate::Alert {
        let mut a = crate::Alert::default();

        // Epoch (your Alert uses `mjd_tt: MJDTT`; in your current code it behaves like f64).
        a.mjd_tt = mjd_tt;

        // Angles: use `.into()` to support both `type Radian = f64` and `struct Radian(f64)`.
        a.ra = ra_rad.into();
        a.dec = dec_rad.into();

        // Uncertainties (only required to satisfy invariants if used by modeling)
        a.ra_err = 1.0.into();
        a.dec_err = 1.0.into();

        // Photometry (not used by track_id but seed construction might carry it)
        a.flux = 1000.0;
        a.flux_err = 10.0;
        a.band = band;

        // dia_source_id / key can stay default for these tests
        a
    }

    /// Build one seed from a pair of alerts using the production constructor.
    ///
    /// This avoids having to manually construct `TangentPlaneModel` and `Photometry`.
    fn make_seed_from_pair<'a>(
        night_id: u32,
        idx_in_night: u32,
        a: &'a crate::Alert,
        b: &'a crate::Alert,
    ) -> SeedNode<'a> {
        let key = SeedKey {
            night_id: NightId(night_id),
            idx_in_night,
        };

        SeedNode::from_pair(key, a, b, None)
            .expect("Test alert pair should always generate a valid SeedNode")
    }

    // -------------------------------------------------------------------------
    // Pure/oracle calendar conversion (independent from mjd_to_year implementation)
    // -------------------------------------------------------------------------
    //
    // We implement an independent conversion:
    // MJD -> JD -> Gregorian date (Y/M/D) using a different pathway:
    // - Convert JD to "Rata Die" day count (days since 0001-01-01)
    // - Convert Rata Die to Gregorian year/month/day
    //
    // This acts as an internal oracle for proptests.

    fn is_leap_year_gregorian(y: i32) -> bool {
        (y % 4 == 0) && ((y % 100 != 0) || (y % 400 == 0))
    }

    fn days_before_year(y: i32) -> i64 {
        // Days before Jan 1 of year y (Gregorian proleptic), relative to year 1.
        let y1 = (y - 1) as i64;
        365 * y1 + y1 / 4 - y1 / 100 + y1 / 400
    }

    fn days_in_month(y: i32, m: i32) -> i32 {
        match m {
            1 => 31,
            2 => {
                if is_leap_year_gregorian(y) {
                    29
                } else {
                    28
                }
            }
            3 => 31,
            4 => 30,
            5 => 31,
            6 => 30,
            7 => 31,
            8 => 31,
            9 => 30,
            10 => 31,
            11 => 30,
            12 => 31,
            _ => panic!("invalid month"),
        }
    }

    fn rata_die_to_ymd(rd: i64) -> (i32, i32, i32) {
        // rd: 1-based day count, where rd=1 is 0001-01-01.
        // Algorithm: find year by integer search using day counts.
        // This is deterministic and independent of mjd_to_year.

        // Binary search for year
        let mut lo = 1i32;
        let mut hi = 10000i32; // enough for MJD ranges we use in tests

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let start_mid = days_before_year(mid) + 1; // rd at Jan 1 mid
            if start_mid <= rd {
                let start_next = days_before_year(mid + 1) + 1;
                if rd < start_next {
                    lo = mid;
                    break;
                }
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let year = lo;
        let mut day_of_year = rd - (days_before_year(year) + 1) + 1; // 1-based doy

        let mut month = 1;
        while month <= 12 {
            let dim = days_in_month(year, month) as i64;
            if day_of_year > dim {
                day_of_year -= dim;
                month += 1;
            } else {
                break;
            }
        }

        (year, month, day_of_year as i32)
    }

    fn mjd_to_year_oracle(mjd: f64) -> u32 {
        let jd = mjd + 2_400_000.5;

        // Civil day number
        let z = (jd + 0.5).floor() as i64;

        let rd = z - 1_721_425;

        let (y, _m, _d) = rata_die_to_ymd(rd);
        y as u32
    }

    // -------------------------------------------------------------------------
    // Unit tests: mjd_to_year sanity with known anchor points
    // -------------------------------------------------------------------------

    #[test]
    fn mjd_to_year_known_anchors() {
        // MJD 51544.0 is 2000-01-01 00:00 (commonly used J2000 epoch day)
        assert_eq!(mjd_to_year(51544.0), 2000);

        // MJD 58000 is in 2017.
        assert_eq!(mjd_to_year(58000.0), 2017);

        // MJD 59000 is in 2020.
        assert_eq!(mjd_to_year(59000.0), 2020);

        // Cross-check with oracle for several points.
        for &mjd in &[
            40000.0, 50000.0, 51544.0, 55000.0, 58000.0, 59000.0, 60000.0, 61000.0,
        ] {
            assert_eq!(mjd_to_year(mjd), mjd_to_year_oracle(mjd));
        }
    }

    #[test]
    fn mjd_to_year_matches_oracle_on_dense_window() {
        // Dense check around a region likely to include leap years etc.
        // 51500..51600 spans early 2000.
        for i in 51500..51600 {
            let mjd = i as f64;
            assert_eq!(mjd_to_year(mjd), mjd_to_year_oracle(mjd));
        }
    }

    // -------------------------------------------------------------------------
    // proptest: mjd_to_year properties and oracle agreement
    // -------------------------------------------------------------------------

    proptest! {
        /// mjd_to_year should match the independent oracle conversion for a wide range of MJDs.
        ///
        /// We restrict to a "reasonable" survey range:
        /// - MJD 45000..70000 ~ years ~ 1981..2051
        #[test]
        fn prop_mjd_to_year_agrees_with_oracle(mjd in 45000f64..70000f64) {
            let y1 = mjd_to_year(mjd);
            let y2 = mjd_to_year_oracle(mjd);
            prop_assert_eq!(y1, y2);
        }

        /// Monotonicity: if mjd increases, year should not decrease.
        #[test]
        fn prop_mjd_to_year_monotone(mjd1 in 45000f64..70000f64, delta_days in 0f64..5000f64) {
            let mjd2 = mjd1 + delta_days;
            let y1 = mjd_to_year(mjd1);
            let y2 = mjd_to_year(mjd2);
            prop_assert!(y2 >= y1);
        }

        /// Local stability: within the same civil year, small day changes should keep the same year.
        ///
        /// We take a random mjd, compute its year, then perturb by up to +/- 100 days and check
        /// that the year is either the same or differs by at most 1 (crossing a new year boundary).
        #[test]
        fn prop_mjd_to_year_local_stability(mjd in 45000f64..70000f64, shift in -100f64..100f64) {
            let y = mjd_to_year(mjd) as i32;
            let y2 = mjd_to_year(mjd + shift) as i32;
            prop_assert!((y2 - y).abs() <= 1);
        }
    }

    // -------------------------------------------------------------------------
    // Tests: earliest_alert_mjd_tt
    // -------------------------------------------------------------------------

    #[test]
    fn earliest_alert_mjd_tt_picks_min_even_if_unsorted() {
        // Owned alerts
        let a1 = make_alert(60000.5, 0.10, 0.10, 1);
        let a2 = make_alert(59000.0, 0.11, 0.10, 1);
        let a3 = make_alert(61000.0, 0.12, 0.10, 1);
        let a4 = make_alert(60500.0, 0.13, 0.10, 1);

        // Build seeds via from_pair (members will be [a,b] internally)
        let s1 = make_seed_from_pair(1, 0, &a1, &a3);
        let s2 = make_seed_from_pair(1, 1, &a2, &a4);

        let nodes: Vec<&SeedNode> = vec![&s1, &s2];
        let mjd_min = earliest_alert_mjd_tt(&nodes);

        assert_abs_diff_eq!(mjd_min, 59000.0, epsilon = 0.0);
    }

    // -------------------------------------------------------------------------
    // Tests: track_id_from_seed_keys_with_year format/determinism
    // -------------------------------------------------------------------------

    #[test]
    fn track_id_format_is_trk_year_plus_12_letters() {
        let keys = vec![
            make_seed_key(42, 0),
            make_seed_key(42, 1),
            make_seed_key(43, 0),
        ];
        let id = track_id_from_seed_keys_with_year(&keys, 2026)
            .as_str()
            .to_string();

        assert!(id.starts_with("TRK2026"));
        let suffix = &id["TRK2026".len()..];
        assert_eq!(suffix.len(), 12);
        assert!(suffix.chars().all(|c| ('a'..='z').contains(&c)));
    }

    #[test]
    fn track_id_is_deterministic_for_same_keys_and_year() {
        let keys = vec![
            make_seed_key(10, 0),
            make_seed_key(10, 1),
            make_seed_key(11, 2),
        ];

        let id1 = track_id_from_seed_keys_with_year(&keys, 2026);
        let id2 = track_id_from_seed_keys_with_year(&keys, 2026);

        assert_eq!(id1, id2);
    }

    #[test]
    fn track_id_changes_when_year_changes() {
        let keys = vec![make_seed_key(10, 0), make_seed_key(10, 1)];

        let id1 = track_id_from_seed_keys_with_year(&keys, 2025);
        let id2 = track_id_from_seed_keys_with_year(&keys, 2026);

        assert_ne!(id1, id2);
        assert!(id1.as_str().starts_with("TRK2025"));
        assert!(id2.as_str().starts_with("TRK2026"));
    }

    #[test]
    fn track_id_changes_when_order_changes() {
        let k1 = make_seed_key(10, 0);
        let k2 = make_seed_key(10, 1);
        let k3 = make_seed_key(11, 2);

        let keys_a = vec![k1, k2, k3];
        let keys_b = vec![k1, k3, k2]; // same set, different order

        let id_a = track_id_from_seed_keys_with_year(&keys_a, 2026);
        let id_b = track_id_from_seed_keys_with_year(&keys_b, 2026);

        assert_ne!(id_a, id_b);
    }

    proptest! {
        /// Determinism property for arbitrary key lists (non-empty).
        #[test]
        fn prop_track_id_deterministic_for_same_input(
            year in 1950u32..2100u32,
            nights in prop::collection::vec(0u32..1000u32, 1..30),
            idxs in prop::collection::vec(0u32..100000u32, 1..30),
        ) {
            // Zip nights/idxs to build keys (truncate to min length)
            let n = nights.len().min(idxs.len());
            let mut keys = Vec::with_capacity(n);
            for i in 0..n {
                keys.push(make_seed_key(nights[i], idxs[i]));
            }

            let id1 = track_id_from_seed_keys_with_year(&keys, year);
            let id2 = track_id_from_seed_keys_with_year(&keys, year);

            prop_assert_eq!(id1, id2);
        }

        /// With high probability, changing one element should change the hash suffix.
        /// (Not a strict guarantee, but practical confidence test.)
        #[test]
        fn prop_track_id_changes_on_small_mutation(
            year in 2000u32..2100u32,
            night in 0u32..1000u32,
            idx in 0u32..100000u32,
            delta in 1u32..1000u32
        ) {
            let keys1 = vec![make_seed_key(night, idx), make_seed_key(night+1, idx+1)];
            let keys2 = vec![make_seed_key(night, idx), make_seed_key(night+1, idx+1+delta)];

            let id1 = track_id_from_seed_keys_with_year(&keys1, year);
            let id2 = track_id_from_seed_keys_with_year(&keys2, year);

            prop_assert_ne!(id1, id2);
        }
    }

    // -------------------------------------------------------------------------
    // Tests: track_id_from_nodes year prefix from earliest alert epoch
    // -------------------------------------------------------------------------

    #[test]
    fn track_id_from_nodes_year_is_from_earliest_alert() {
        let a_early = make_alert(61000.0, 0.10, 0.10, 1);
        let a_late = make_alert(62000.0, 0.11, 0.10, 1);
        let a_mid = make_alert(61500.0, 0.12, 0.10, 1);
        let a_mid2 = make_alert(61600.0, 0.13, 0.10, 1);

        let s1 = make_seed_from_pair(1, 0, &a_late, &a_mid2);
        let s2 = make_seed_from_pair(1, 1, &a_early, &a_mid);

        let nodes: Vec<&SeedNode> = vec![&s1, &s2];

        let id = track_id_from_nodes(&nodes);
        let year_expected = mjd_to_year(61000.0);

        assert!(id.as_str().starts_with(&format!("TRK{}", year_expected)));
    }

    #[test]
    fn track_id_from_nodes_is_insensitive_to_where_the_earliest_alert_is() {
        // Four alerts with distinct epochs; the earliest is a2.
        let a1 = make_alert(60000.0, 0.10, 0.10, 1);
        let a2 = make_alert(59000.0, 0.11, 0.10, 1); // earliest
        let a3 = make_alert(61000.0, 0.12, 0.10, 1);
        let a4 = make_alert(60500.0, 0.13, 0.10, 1);

        // Build two tracks with the SAME SeedKeys in the SAME node order,
        // but place the earliest alert (a2) in a different seed.
        //
        // Track A:
        // - seed0: (a2, a1)  contains earliest
        // - seed1: (a4, a3)
        let s0_a = make_seed_from_pair(1, 0, &a2, &a1);
        let s1_a = make_seed_from_pair(1, 1, &a4, &a3);
        let nodes_a: Vec<&SeedNode> = vec![&s0_a, &s1_a];

        // Track B:
        // - seed0: (a1, a3)
        // - seed1: (a2, a4) contains earliest (moved)
        let s0_b = make_seed_from_pair(1, 0, &a1, &a3);
        let s1_b = make_seed_from_pair(1, 1, &a2, &a4);
        let nodes_b: Vec<&SeedNode> = vec![&s0_b, &s1_b];

        // Year prefix must be derived from the *minimum mjd_tt* across all members,
        // so it should be identical for both tracks (same alert set => same min).
        // Suffix must also match because keys and their order are identical (night_id=1, idx=0 then idx=1).
        let id_a = track_id_from_nodes(&nodes_a);
        let id_b = track_id_from_nodes(&nodes_b);

        assert_eq!(id_a, id_b);
    }

    proptest! {
        #[test]
        fn prop_track_id_from_nodes_year_matches_min_mjd_year(
            obs in prop::collection::vec((45000f64..70000f64, 0f64..6.0, -1.4f64..1.4), 2..10),
        ) {
            // Owned alerts
            let mut alerts: Vec<crate::Alert> = Vec::with_capacity(obs.len());
            for (mjd, ra, dec) in &obs {
                alerts.push(make_alert(*mjd, *ra, *dec, 1));
            }

            // Build seeds: chain pairs (0,1), (1,2), ...
            let mut seeds: Vec<SeedNode> = Vec::with_capacity(alerts.len() - 1);
            for i in 0..alerts.len()-1 {
                let s = make_seed_from_pair(1, i as u32, &alerts[i], &alerts[i+1]);
                seeds.push(s);
            }

            let node_refs: Vec<&SeedNode> = seeds.iter().collect();

            let min_mjd = obs.iter().map(|(m, _, _)| *m).fold(f64::INFINITY, f64::min);
            let year_expected = mjd_to_year(min_mjd);

            let id = track_id_from_nodes(&node_refs);
            let expected_prefix = format!("TRK{}", year_expected);
            prop_assert!(id.as_str().starts_with(&expected_prefix));
        }
    }
}

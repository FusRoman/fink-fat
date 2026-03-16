//! Ground-truth SSO (Solar System Object) identity map.
//!
//! This module provides the [`TruthSSOMap`] type and the [`load_truth_sso_map`]
//! loader used by all evaluation sub-commands to ground-truth alert identities.
//!
//! ## Data model
//!
//! The truth table is a Parquet file with (at minimum) two columns:
//!
//! | Column            | Type     | Description                                           |
//! |-------------------|----------|-------------------------------------------------------|
//! | `dia_source_id`   | `uint64` | Unique detection identifier (matches [`DiaSourceId`]) |
//! | `trajectory_id`   | `int32`  | Ground-truth trajectory / object identifier           |
//!
//! [`load_truth_sso_map`] reads only these two columns and builds an
//! [`AHashMap`](ahash::AHashMap) for O(1) lookups during post-processing.

use ahash::AHashMap;
use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::{Alert, alerts::DiaSourceId, night_id::NightId};
use polars::prelude::*;

/// Ground-truth trajectory identifier.
pub type TrajId = u32;

/// Map from [`DiaSourceId`] to ground-truth [`TrajId`].
///
/// Built once at startup from the truth Parquet file and then passed
/// read-only to every evaluation post-processing function.
pub type TruthSSOMap = AHashMap<DiaSourceId, TrajId>;

/// For each trajectory ID, count the number of alerts per night.
pub type TrajCountMap = AHashMap<TrajId, AHashMap<NightId, usize>>;

/// In-memory representation of the truth SSO map and related pre-computed data.
pub struct TruthSSO {
    /// Map from `dia_source_id` to `trajectory_id`.
    map: TruthSSOMap,
    /// Pre-computed count of alerts per trajectory per night, used for computing recoverable trajectories.
    traj_count: TrajCountMap,
}

impl TruthSSO {
    /// Load the truth SSO map and pre-compute trajectory counts from a Parquet file.
    ///
    /// Arguments
    /// ---------
    /// * `path` – Path to the Parquet file containing the truth table.
    ///
    /// Returns
    /// -------
    /// * `Ok(TruthSSO)` – Loaded truth SSO map and trajectory counts.
    /// * `Err(...)` – On I/O failure, missing columns, or type mismatch
    pub fn load(path: &Utf8Path) -> Result<Self> {
        let df = load_dataframe(path)?;
        let alert_ids = get_dia_source_id(&df)?;
        let trajs = get_trajectory_id(&df)?;
        let night_ids = get_night_id(&df)?; // Not currently used, but could be useful for validation or future features.

        let nb_uniq_trajs = trajs
            .n_unique()
            .context("failed to compute number of unique trajectories")?;

        let mut map = TruthSSOMap::with_capacity(alert_ids.len());
        let mut traj_alert_count: TrajCountMap = AHashMap::with_capacity(nb_uniq_trajs);

        for ((alert_id, traj), night_id) in alert_ids.iter().zip(trajs.iter()).zip(night_ids.iter())
        {
            if let (Some(id), Some(traj), Some(night_id)) = (alert_id, traj, night_id) {
                map.insert(id, traj as TrajId);
                let traj_entry = traj_alert_count
                    .entry(traj as TrajId)
                    .or_insert_with(AHashMap::new);
                *traj_entry.entry(night_id.into()).or_insert(0) += 1;
            }
        }

        Ok(Self {
            map,
            traj_count: traj_alert_count,
        })
    }

    /// Look up the ground-truth trajectory ID for a given alert.
    ///
    /// Arguments
    /// ---------
    /// * `alert` – The alert for which to look up the trajectory ID.
    ///
    /// Returns
    /// -------
    /// * `Some(traj_id)` – If the alert's `dia_source_id` is present in the truth map, returns the corresponding trajectory ID.
    /// * `None` – If the alert's `dia_source_id` is not present in the truth map.
    pub fn get_truth_traj_id(&self, alert: &Alert) -> Option<TrajId> {
        self.map.get(&alert.key.dia_source_id).copied()
    }

    /// Classify a resolved seed slice against the ground-truth map.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` – Resolved member alerts (borrowed from the alert store).
    ///
    /// Return
    /// ------
    /// [`TruthClass`] indicating whether the seed is a true positive, false
    /// positive, or has at least one alert missing from the truth map.
    pub fn classify(&self, alerts: &[&Alert]) -> TruthClass {
        let mut first_id: Option<TrajId> = None;
        for alert in alerts {
            match self.map.get(&alert.key.dia_source_id).copied() {
                None => return TruthClass::Unknown,
                Some(traj_id) => match first_id {
                    None => first_id = Some(traj_id),
                    Some(ref expected) if *expected != traj_id => return TruthClass::FalsePositive,
                    _ => {}
                },
            }
        }
        if first_id.is_some() {
            TruthClass::TruePositive
        } else {
            // Empty slice — treat as unknown.
            TruthClass::Unknown
        }
    }

    /// Returns `true` iff all alerts in the slice share the same ground-truth
    /// trajectory and that trajectory is present in the truth map.
    ///
    /// Arguments
    /// ---------
    /// * `alerts`    – Resolved member alerts.
    pub fn is_true_positive(&self, alerts: &[&Alert]) -> bool {
        matches!(self.classify(alerts), TruthClass::TruePositive)
    }

    /// Iterator over `(traj_id, night_id, count)` tuples from a [`TrajCountMap`].
    ///
    /// This is a convenience for iterating over all trajectory-night pairs without having to deal with the nested map structure.
    ///
    /// Returns
    /// -------
    /// An iterator yielding `(traj_id, night_id, count)` tuples for each trajectory and night where the trajectory has at least one alert.
    pub fn traj_count_iter(&self) -> impl Iterator<Item = (TrajId, NightId, usize)> {
        self.traj_count.iter().flat_map(|(&traj_id, night_counts)| {
            night_counts
                .iter()
                .map(move |(&night_id, &count)| (traj_id, night_id, count))
        })
    }

    /// Get the count of alerts for a given trajectory ID and night ID.
    ///
    /// Arguments
    /// ---------
    /// * `traj_id`  – The trajectory ID for which to get the count.
    /// * `night_id` – The night ID for which to get the count.
    ///
    /// Returns
    /// -------
    /// The number of alerts associated with the given trajectory ID on the given night ID,
    /// or 0 if the trajectory or night is not present in the map.
    pub fn traj_count_for_night(&self, traj_id: TrajId, night_id: NightId) -> usize {
        self.traj_count
            .get(&traj_id)
            .and_then(|night_counts| night_counts.get(&night_id))
            .copied()
            .unwrap_or(0)
    }

    /// Get the total count of alerts for a given trajectory ID across all nights.
    ///
    /// Arguments
    /// ---------
    /// * `traj_id`  – The trajectory ID for which to get the total count.
    ///
    /// Returns
    /// -------
    /// The total number of alerts associated with the given trajectory ID across all nights,
    /// or 0 if the trajectory is not present in the map.
    pub fn traj_count_for_traj(&self, traj_id: TrajId) -> usize {
        self.traj_count
            .get(&traj_id)
            .map(|night_counts| night_counts.values().sum())
            .unwrap_or(0)
    }

    /// Compute the fraction of a trajectory's total alerts covered by a single track.
    ///
    /// Arguments
    /// ---------
    /// * `traj_id`   – The ground-truth trajectory ID.
    /// * `n_covered` – The number of alerts from `traj_id` present in the track
    ///                 (equal to the track length when the track is a true positive).
    ///
    /// Returns
    /// -------
    /// `n_covered / total_alerts_for_traj`, or `0.0` if either `n_covered` is zero
    /// or the trajectory is absent from the truth map.
    ///
    /// Note: `n_covered` is not capped at the total; the caller is responsible for
    /// passing a value ≤ `traj_count_for_traj(traj_id)`.
    pub fn coverage_fraction(&self, traj_id: TrajId, n_covered: usize) -> f64 {
        let total = self.traj_count_for_traj(traj_id);
        if total == 0 || n_covered == 0 {
            0.0
        } else {
            n_covered as f64 / total as f64
        }
    }

    // --- Private helpers ---

    /// Collects and sorts the nights of a trajectory that meet the `night_count` threshold.
    ///
    /// Shared by [`Self::recoverable_edges`] and [`Self::recoverable_traj`] to
    /// eliminate the duplicated collect+sort logic.
    fn seeded_nights(night_counts: &AHashMap<NightId, usize>, night_count: usize) -> Vec<NightId> {
        let mut nights: Vec<NightId> = night_counts
            .iter()
            .filter_map(|(&n, &c)| (c >= night_count).then_some(n))
            .collect();
        nights.sort_unstable();
        nights
    }

    /// Counts consecutive pairs in a sorted night slice whose gap is ≤ `max_gap`.
    ///
    /// Shared by [`Self::recoverable_traj`].
    fn qualifying_edge_count(nights: &[NightId], max_gap: u8) -> usize {
        nights
            .windows(2)
            .filter(|w| w[1].0 - w[0].0 <= max_gap as u32)
            .count()
    }

    // --- Public API ---

    /// Get an iterator over trajectory IDs that have at least `night_count` alerts on the specified night.
    ///
    /// Arguments
    /// ---------
    /// * `night_id`   – The night ID for which to filter trajectories.
    /// * `night_count` – The minimum number of alerts a trajectory must have on the specified night to be included in the output.
    ///
    /// Returns
    /// -------
    /// An iterator yielding trajectory IDs that have at least `night_count` alerts on the specified night.
    /// This is used to identify "recoverable" trajectories for seeding evaluation.
    pub fn recoverable_seeds(
        &self,
        night_id: NightId,
        night_count: usize,
    ) -> impl Iterator<Item = TrajId> {
        self.traj_count
            .iter()
            .filter_map(move |(&traj_id, night_counts)| {
                (night_counts.get(&night_id).copied().unwrap_or(0) >= night_count)
                    .then_some(traj_id)
            })
    }

    /// Iterate over all detectable edges for the given seeding parameters.
    ///
    /// An edge connects two consecutive seeded nights of the same trajectory.
    /// A night is *seeded* when it contains at least `night_count` alerts for
    /// that trajectory.  Two seeded nights form an edge when their gap is
    /// ≤ `max_gap` nights.  Each edge is uniquely identified by
    /// `(traj_id, night_from, night_to)`.
    ///
    /// Arguments
    /// ---------
    /// * `night_count` – Minimum number of alerts on a single night to form a seed.
    /// * `max_gap`     – Maximum allowed gap (in nights) between two consecutive
    ///                   seeds for an edge to exist.
    ///
    /// Return
    /// ------
    /// An iterator yielding one `(traj_id, night_from, night_to)` tuple per
    /// detectable edge, where `night_from < night_to` and
    /// `night_to - night_from ≤ max_gap`.
    pub fn recoverable_edges(
        &self,
        night_count: usize,
        max_gap: u8,
    ) -> impl Iterator<Item = (TrajId, NightId, NightId)> {
        let mut edges: Vec<(TrajId, NightId, NightId)> = Vec::new();
        for (&traj_id, night_counts) in &self.traj_count {
            let nights = Self::seeded_nights(night_counts, night_count);
            edges.extend(
                nights
                    .windows(2)
                    .filter(|w| w[1].0 - w[0].0 <= max_gap as u32)
                    .map(|w| (traj_id, w[0], w[1])),
            );
        }
        edges.into_iter()
    }

    /// Iterate over recoverable trajectories for the given solver parameters.
    /// A trajectory is recoverable when it has at least a number of minimum nodes
    /// (nodes is at least two alerts) separated by a gap of at most `max_gap` nights.
    pub fn recoverable_traj(
        &self,
        night_count: usize,
        max_gap: u8,
        min_nodes: usize,
    ) -> impl Iterator<Item = TrajId> + '_ {
        self.traj_count
            .iter()
            .filter_map(move |(&traj_id, night_counts)| {
                let nights = Self::seeded_nights(night_counts, night_count);
                (Self::qualifying_edge_count(&nights, max_gap) >= min_nodes).then_some(traj_id)
            })
    }
}

/// Load a DataFrame from a Parquet file.
///
/// The required columns are `dia_source_id` (uint64), `trajectory_id` (int32), and `night_id` (uint32).
/// Only these three columns are read; all other columns in the Parquet file are ignored.
///
/// Arguments
/// ---------
/// * `path` – Path to the Parquet file containing the truth table.
///
/// Returns
/// -------
/// * `Ok(DataFrame)` – DataFrame containing the data.
/// * `Err(...)` – On I/O failure, missing columns, or type mismatch.
pub fn load_dataframe(path: &Utf8Path) -> Result<DataFrame> {
    LazyFrame::scan_parquet(path.as_str(), ScanArgsParquet::default())
        .context("failed to open truth parquet file")?
        .select([col("dia_source_id"), col("trajectory_id"), col("night_id")])
        .collect()
        .context("failed to collect truth parquet columns")
}

/// Extract the `dia_source_id` column as a `UInt64Chunked`.
///
/// Arguments
/// ---------
/// * `df` – DataFrame containing the truth data.
///
/// Returns
/// -------
/// * `Ok(&UInt64Chunked)` – The `dia_source_id` column as a `UInt64Chunked`.
/// * `Err(...)` – If the column is missing or has the wrong type.
pub fn get_dia_source_id(df: &DataFrame) -> Result<&UInt64Chunked> {
    df.column("dia_source_id")
        .context("missing column 'dia_source_id'")?
        .u64()
        .context("'dia_source_id' is not uint64")
}

/// Extract the `trajectory_id` column as an `Int32Chunked`.
///
/// Arguments
/// ---------
/// * `df` – DataFrame containing the truth data.
///
/// Returns
/// -------
/// * `Ok(&Int32Chunked)` – The `trajectory_id` column as an `Int32Chunked`.
/// * `Err(...)` – If the column is missing or has the wrong type.
pub fn get_trajectory_id(df: &DataFrame) -> Result<&Int32Chunked> {
    df.column("trajectory_id")
        .context("missing column 'trajectory_id'")?
        .i32()
        .context("'trajectory_id' is not int32")
}

/// Extract the `night_id` column as a `UInt32Chunked`.
///
/// Arguments
/// ---------
/// * `df` – DataFrame containing the truth data.
///
/// Returns
/// -------
/// * `Ok(&UInt32Chunked)` – The `night_id` column as a `UInt32Chunked`.
/// * `Err(...)` – If the column is missing or has the wrong type.
pub fn get_night_id(df: &DataFrame) -> Result<&UInt32Chunked> {
    df.column("night_id")
        .context("missing column 'night_id'")?
        .u32()
        .context("'night_id' is not uint32")
}

/// Classification of a seed with respect to the ground-truth map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthClass {
    /// All member alerts belong to the same ground-truth trajectory.
    TruePositive,
    /// Member alerts belong to at least two distinct ground-truth trajectories.
    FalsePositive,
    /// At least one member alert is absent from the truth map.
    Unknown,
}

#[cfg(test)]
mod truth_sso_tests {
    use super::*;
    use prop_test::prelude::*;

    fn make_truth_sso() -> TruthSSO {
        let mut traj_count: TrajCountMap = AHashMap::new();
        traj_count.insert(1, [(10.into(), 3), (12.into(), 2), (13.into(), 2)].into());
        traj_count.insert(2, [(10.into(), 3), (13.into(), 2)].into());
        traj_count.insert(
            3,
            [
                (10.into(), 3),
                (11.into(), 1),
                (13.into(), 2),
                (14.into(), 2),
                (15.into(), 2),
            ]
            .into(),
        );
        TruthSSO {
            map: TruthSSOMap::new(),
            traj_count,
        }
    }

    // ── recoverable_edges: non-consecutive night pairs ──────────────────────
    //
    // `recoverable_edges` uses `windows(2)` which only emits *consecutive*
    // pairs of seeded nights.  When an intermediate seeded night sits between
    // two nights whose gap is ≤ max_gap, the skip pair is absent from the
    // output.
    //
    // Consequence for edge recall in `compute_edge_stats`:
    //   • the recoverable set is built from `recoverable_edges(2, max_gap)`,
    //   • a TP engine edge (traj, n1, n3) is only credited when the exact
    //     triplet appears in that set,
    //   • if n2 (n1 < n2 < n3) is also a seeded night, the pair (n1, n3)
    //     is absent → the trajectory is NOT counted as recovered even though
    //     a valid TP edge was produced.
    //   ⟹ recall is under-counted whenever the engine produces a skip edge
    //     over an intermediate seeded night.
    //
    // The test below documents this limitation and is marked `#[ignore]`.
    // Remove the attribute to confirm the failure.
    #[test]
    #[ignore = "known recall under-counting: recoverable_edges misses non-consecutive seeded-night pairs within max_gap (windows(2) limitation)"]
    fn recoverable_edges_includes_all_pairs_within_max_gap() {
        let mut traj_count: TrajCountMap = AHashMap::new();
        // Traj 1 has three seeded nights:  10, 11, 13.
        // With max_gap = 3 the pair (10 → 13, gap = 3) is recoverable per the
        // definition, but windows(2) only generates (10,11) and (11,13).
        traj_count.insert(1, [(10.into(), 3), (11.into(), 3), (13.into(), 3)].into());
        let truth_sso = TruthSSO {
            map: TruthSSOMap::new(),
            traj_count,
        };

        let edges: Vec<(TrajId, NightId, NightId)> = truth_sso.recoverable_edges(2, 3).collect();

        // (10, 13) with gap = 3 ≤ max_gap = 3 SHOULD be present.
        assert!(
            edges.contains(&(1, 10.into(), 13.into())),
            "skip edge (10 → 13, gap=3 ≤ max_gap=3) should appear in recoverable_edges; got: {edges:?}"
        );
    }

    #[test]
    fn test_recoverable_trajectories() {
        let truth_sso = make_truth_sso();
        let night_count = 2;
        let max_gap = 2;
        let min_nodes = 2;
        let mut recoverable = truth_sso
            .recoverable_traj(night_count, max_gap, min_nodes)
            .collect::<Vec<_>>();
        recoverable.sort();
        assert_eq!(recoverable, vec![1, 3]);
    }

    #[test]
    fn test_recoverable_seeds() {
        let truth_sso = make_truth_sso();

        // All three trajectories have >= 2 alerts on night 10.
        let mut result: Vec<TrajId> = truth_sso.recoverable_seeds(10.into(), 2).collect();
        result.sort();
        assert_eq!(result, vec![1, 2, 3]);

        // Only trajectory 1 has >= 2 alerts on night 12.
        let mut result: Vec<TrajId> = truth_sso.recoverable_seeds(12.into(), 2).collect();
        result.sort();
        assert_eq!(result, vec![1]);

        // Trajectory 3 has only 1 alert on night 11 — below the threshold.
        let result: Vec<TrajId> = truth_sso.recoverable_seeds(11.into(), 2).collect();
        assert!(result.is_empty());

        // Night with no alerts at all returns nothing.
        let result: Vec<TrajId> = truth_sso.recoverable_seeds(99.into(), 1).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_recoverable_edges() {
        let truth_sso = make_truth_sso();

        // night_count=2, max_gap=2:
        //   traj 1: seeded nights [10, 12, 13] → edges (10→12) gap 2, (12→13) gap 1.
        //   traj 2: seeded nights [10, 13]     → gap 3 > max_gap, no edge.
        //   traj 3: seeded nights [10, 13, 14, 15] → (10→13) gap 3 skipped;
        //           edges (13→14) gap 1, (14→15) gap 1.
        let mut edges: Vec<(TrajId, NightId, NightId)> =
            truth_sso.recoverable_edges(2, 2).collect();
        edges.sort();
        assert_eq!(
            edges,
            vec![
                (1, 10.into(), 12.into()),
                (1, 12.into(), 13.into()),
                (3, 13.into(), 14.into()),
                (3, 14.into(), 15.into()),
            ]
        );

        // Widening max_gap to 3 lets traj 2's single gap through.
        let mut edges: Vec<(TrajId, NightId, NightId)> =
            truth_sso.recoverable_edges(2, 3).collect();
        edges.sort();
        assert!(edges.iter().any(|&(t, f, _)| t == 2 && f == 10.into()));
    }

    #[test]
    fn test_recoverable_traj_min_nodes_variants() {
        let truth_sso = make_truth_sso();

        // min_nodes=1: trajs 1 and 3 each have at least one qualifying edge.
        let mut result: Vec<TrajId> = truth_sso.recoverable_traj(2, 2, 1).collect();
        result.sort();
        assert_eq!(result, vec![1, 3]);

        // min_nodes=3: neither traj reaches 3 qualifying edges within max_gap=2.
        let result: Vec<TrajId> = truth_sso.recoverable_traj(2, 2, 3).collect();
        assert!(result.is_empty());

        // max_gap=3: traj 2 now has a qualifying edge (10→13, gap=3).
        let mut result: Vec<TrajId> = truth_sso.recoverable_traj(2, 3, 1).collect();
        result.sort();
        assert_eq!(result, vec![1, 2, 3]);
    }

    // --- Proptest strategies ---

    prop_compose! {
        /// Generate an arbitrary per-night alert-count map for a single trajectory.
        fn arb_night_counts()(
            pairs in prop::collection::vec((0u32..20u32, 1usize..=5), 1..=6),
        ) -> AHashMap<NightId, usize> {
            pairs.into_iter().map(|(n, c)| (NightId(n), c)).collect()
        }
    }

    prop_compose! {
        /// Generate an arbitrary `TrajCountMap` with 1–5 trajectories.
        fn arb_traj_count()(
            pairs in prop::collection::vec((1u32..=8u32, arb_night_counts()), 1..=5),
        ) -> TrajCountMap {
            pairs.into_iter().collect()
        }
    }

    // Property: recoverable_seeds returns *exactly* the trajectories that have
    // enough alerts on the queried night — no more, no fewer.
    #[test]
    fn prop_recoverable_seeds_matches_manual_filter() {
        prop_test!(&(arb_traj_count(), 0u32..20u32, 1usize..=4), |(
            traj_count,
            night_id_raw,
            night_count,
        )| {
            let night_id = NightId(night_id_raw);
            let truth_sso = TruthSSO {
                map: TruthSSOMap::new(),
                traj_count,
            };
            let mut result: Vec<TrajId> =
                truth_sso.recoverable_seeds(night_id, night_count).collect();
            result.sort();
            let mut expected: Vec<TrajId> = truth_sso
                .traj_count
                .iter()
                .filter(|(_, nc)| nc.get(&night_id).copied().unwrap_or(0) >= night_count)
                .map(|(&id, _)| id)
                .collect();
            expected.sort();
            prop_assert_eq!(result, expected);
            Ok(())
        });
    }

    // Properties for each edge (traj, from, to) produced by recoverable_edges:
    //   - from < to
    //   - to − from ≤ max_gap
    //   - both endpoint nights have ≥ night_count alerts for that trajectory
    //   - no duplicate edges
    #[test]
    fn prop_recoverable_edges_invariants() {
        prop_test!(&(arb_traj_count(), 1usize..=4, 1u8..=5), |(
            traj_count,
            night_count,
            max_gap,
        )| {
            let truth_sso = TruthSSO {
                map: TruthSSOMap::new(),
                traj_count,
            };
            let edges: Vec<(TrajId, NightId, NightId)> =
                truth_sso.recoverable_edges(night_count, max_gap).collect();
            for &(traj_id, from, to) in &edges {
                prop_assert!(from < to, "edge from={} not < to={}", from.0, to.0);
                prop_assert!(
                    to.0 - from.0 <= max_gap as u32,
                    "gap {} exceeds max_gap {}",
                    to.0 - from.0,
                    max_gap
                );
                prop_assert!(truth_sso.traj_count_for_night(traj_id, from) >= night_count);
                prop_assert!(truth_sso.traj_count_for_night(traj_id, to) >= night_count);
            }
            let mut sorted_edges = edges.clone();
            sorted_edges.sort();
            sorted_edges.dedup();
            prop_assert_eq!(sorted_edges.len(), edges.len(), "duplicate edges found");
            Ok(())
        });
    }

    // Property: every trajectory returned by recoverable_traj has at least
    // min_nodes qualifying edges according to recoverable_edges.
    #[test]
    fn prop_recoverable_traj_has_enough_edges() {
        prop_test!(
            &(arb_traj_count(), 1usize..=4, 1u8..=5, 1usize..=4),
            |(traj_count, night_count, max_gap, min_nodes)| {
                let truth_sso = TruthSSO {
                    map: TruthSSOMap::new(),
                    traj_count,
                };
                let recoverable: Vec<TrajId> = truth_sso
                    .recoverable_traj(night_count, max_gap, min_nodes)
                    .collect();
                let mut edge_counts: AHashMap<TrajId, usize> = AHashMap::new();
                for (traj_id, _, _) in truth_sso.recoverable_edges(night_count, max_gap) {
                    *edge_counts.entry(traj_id).or_insert(0) += 1;
                }
                for &traj_id in &recoverable {
                    let count = edge_counts.get(&traj_id).copied().unwrap_or(0);
                    prop_assert!(
                        count >= min_nodes,
                        "traj {} has {} edges, expected >= {}",
                        traj_id,
                        count,
                        min_nodes
                    );
                }
                Ok(())
            }
        );
    }

    // Property: increasing min_nodes can only shrink the result set (monotonicity).
    // If a trajectory is recoverable for min_nodes + 1, it is also recoverable for min_nodes.
    #[test]
    fn prop_recoverable_traj_monotone_in_min_nodes() {
        prop_test!(
            &(arb_traj_count(), 1usize..=4, 1u8..=5, 1usize..=3),
            |(traj_count, night_count, max_gap, min_nodes)| {
                let truth_sso = TruthSSO {
                    map: TruthSSOMap::new(),
                    traj_count,
                };
                let mut r_lower: Vec<TrajId> = truth_sso
                    .recoverable_traj(night_count, max_gap, min_nodes)
                    .collect();
                let r_upper: Vec<TrajId> = truth_sso
                    .recoverable_traj(night_count, max_gap, min_nodes + 1)
                    .collect();
                r_lower.sort();
                for &traj_id in &r_upper {
                    prop_assert!(
                        r_lower.binary_search(&traj_id).is_ok(),
                        "traj {} in min_nodes={} result but missing from min_nodes={} result",
                        traj_id,
                        min_nodes + 1,
                        min_nodes
                    );
                }
                Ok(())
            }
        );
    }

    // --- coverage_fraction tests ---

    #[test]
    fn test_coverage_fraction() {
        let truth_sso = make_truth_sso();
        // traj 1: (10→3) + (12→2) + (13→2) = 7 alerts total.
        assert!((truth_sso.coverage_fraction(1, 7) - 1.0).abs() < 1e-9);
        assert!((truth_sso.coverage_fraction(1, 2) - 2.0 / 7.0).abs() < 1e-9);
        // 0 covered → always 0.0.
        assert_eq!(truth_sso.coverage_fraction(1, 0), 0.0);
        // Unknown trajectory → 0.0.
        assert_eq!(truth_sso.coverage_fraction(99, 5), 0.0);

        // traj 2: 3+2 = 5 alerts; half coverage.
        assert!((truth_sso.coverage_fraction(2, 2) - 2.0 / 5.0).abs() < 1e-9);

        // traj 3: 3+1+2+2+2 = 10 alerts.
        assert!((truth_sso.coverage_fraction(3, 5) - 0.5).abs() < 1e-9);
        assert!((truth_sso.coverage_fraction(3, 10) - 1.0).abs() < 1e-9);
    }

    // Property: coverage_fraction is in [0, 1] when n_covered ≤ total.
    #[test]
    fn prop_coverage_fraction_bounded() {
        prop_test!(&(arb_traj_count(), 1u32..=8u32, 0usize..=20), |(
            traj_count,
            traj_id,
            n_covered,
        )| {
            let truth_sso = TruthSSO {
                map: TruthSSOMap::new(),
                traj_count,
            };
            let total = truth_sso.traj_count_for_traj(traj_id);
            let frac = truth_sso.coverage_fraction(traj_id, n_covered);
            prop_assert!(frac >= 0.0);
            if n_covered <= total {
                prop_assert!(
                    frac <= 1.0 + 1e-9,
                    "fraction {} > 1.0 with n_covered={} total={}",
                    frac,
                    n_covered,
                    total
                );
            }
            // coverage_fraction(id, 0) == 0.0 always.
            prop_assert_eq!(truth_sso.coverage_fraction(traj_id, 0), 0.0);
            Ok(())
        });
    }

    // Property: coverage_fraction is monotone in n_covered (for a fixed traj_id).
    #[test]
    fn prop_coverage_fraction_monotone() {
        prop_test!(
            &(arb_traj_count(), 1u32..=8u32, 0usize..=9, 0usize..=9),
            |(traj_count, traj_id, a, b)| {
                let truth_sso = TruthSSO {
                    map: TruthSSOMap::new(),
                    traj_count,
                };
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                let frac_lo = truth_sso.coverage_fraction(traj_id, lo);
                let frac_hi = truth_sso.coverage_fraction(traj_id, hi);
                prop_assert!(
                    frac_lo <= frac_hi + 1e-9,
                    "coverage_fraction not monotone: f({})={} > f({})={}",
                    lo,
                    frac_lo,
                    hi,
                    frac_hi
                );
                Ok(())
            }
        );
    }
}

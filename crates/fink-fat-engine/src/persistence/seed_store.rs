use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    night_id::{NightId, PairingMode},
    persistence::{
        alert_store::AlertStore,
        error::{BorrowError, PersistenceIoError},
        layout::PersistenceLayout,
        manifest::Manifest,
        seed_node::{SeedKey, SeedNodeOwned, SeedNodeOwnedSlice},
    },
    pipeline::seed_store::SeedStore,
    seeding::seed_node::SeedNode,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedStoreOwned(AHashMap<NightId, Vec<SeedNodeOwned>>);

impl SeedStoreOwned {
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    pub fn from_map(map: AHashMap<NightId, Vec<SeedNodeOwned>>) -> Self {
        Self(map)
    }

    pub fn try_get_seed(&self, seed: SeedKey) -> Option<&SeedNodeOwned> {
        self.0
            .get(&seed.night_id)
            .and_then(|seeds| seeds.get(seed.idx_in_night as usize))
    }

    pub fn insert(&mut self, night_id: NightId, seeds: Vec<SeedNodeOwned>) {
        self.0.insert(night_id, seeds);
    }

    /// Shared implementation for borrowing seeds, optionally restricted to a `NightWindow`.
    fn to_borrowed_impl<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
        night_window: Option<PairingMode>,
    ) -> Result<SeedStore<'alert_lf>, BorrowError> {
        let mut map: AHashMap<NightId, Vec<SeedNode<'alert_lf>>> =
            AHashMap::with_capacity(self.0.len());

        for (night_id, seeds_owned) in self.0.iter() {
            if let Some(w) = night_window {
                if !w.contains(*night_id) {
                    continue;
                }
            }

            let mut seeds_borrowed = Vec::with_capacity(seeds_owned.len());
            for s in seeds_owned {
                seeds_borrowed.push(s.to_borrowed(alerts)?);
            }
            map.insert(*night_id, seeds_borrowed);
        }

        Ok(SeedStore::from_map(map))
    }

    pub fn to_borrowed<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
    ) -> Result<SeedStore<'alert_lf>, BorrowError> {
        self.to_borrowed_impl(alerts, None)
    }

    pub fn to_borrowed_window<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
        night_window: PairingMode,
    ) -> Result<SeedStore<'alert_lf>, BorrowError> {
        self.to_borrowed_impl(alerts, Some(night_window))
    }

    pub fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &mut Manifest,
        night_id: NightId,
    ) -> Result<(), PersistenceIoError> {
        if let Some(seeds) = self.0.get(&night_id) {
            seeds
                .as_slice()
                .save_seeds_night(layout, manifest, night_id)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod seed_store_owned_tests {
    use super::*;
    use crate::{
        Alert,
        night_id::{NightId, PairingMode},
        persistence::{
            alert::AlertKey,
            alert_store::AlertStore,
            seed_node::{SeedKey, SeedNodeOwned},
        },
        seeding::seed_node::SeedNodeCore,
    };

    // -------------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------------

    fn nid(v: u32) -> NightId {
        NightId(v)
    }

    /// Create a minimal `SeedNodeOwned` for testing.
    fn make_seed_owned(
        night_id: NightId,
        idx_in_night: u32,
        alert_keys: Vec<AlertKey>,
    ) -> SeedNodeOwned {
        SeedNodeOwned {
            core: SeedNodeCore {
                key: SeedKey {
                    night_id,
                    idx_in_night,
                },
                ..Default::default()
            },
            members: alert_keys,
        }
    }

    /// Create a test `AlertStore` with dummy alerts.
    /// Each alert has ID = index and is properly keyed.
    fn make_alert_store_with_nights(nights: &[(NightId, usize)]) -> AlertStore {
        let mut map = AHashMap::new();

        for &(night_id, n_alerts) in nights {
            let mut alerts = Vec::new();
            for idx in 0..n_alerts {
                let alert = Alert {
                    key: AlertKey {
                        night_id,
                        idx_in_night: idx as u32,
                    },
                    ..Default::default()
                };
                alerts.push(alert);
            }
            map.insert(night_id, alerts);
        }

        AlertStore::from_map(map)
    }

    /// Create a simple alert store with consecutive nights.
    fn make_alert_store(max_night_id: u32, alerts_per_night: usize) -> AlertStore {
        let nights: Vec<_> = (0..=max_night_id)
            .map(|n| (nid(n), alerts_per_night))
            .collect();
        make_alert_store_with_nights(&nights)
    }

    /// Create a store from a list of night IDs with one seed per night.
    fn make_store_from_nights(nights: &[u32]) -> SeedStoreOwned {
        let mut store = SeedStoreOwned::new();
        for &night in nights {
            let night_id = nid(night);
            let alert_keys = vec![
                AlertKey {
                    night_id,
                    idx_in_night: 0,
                },
                AlertKey {
                    night_id,
                    idx_in_night: 1,
                },
            ];
            let seed = make_seed_owned(night_id, 0, alert_keys);
            store.insert(night_id, vec![seed]);
        }
        store
    }

    // -------------------------------------------------------------------------
    // Construction and insertion
    // -------------------------------------------------------------------------

    #[test]
    fn new_creates_empty_store() {
        let store = SeedStoreOwned::new();
        assert!(store.0.is_empty());
    }

    #[test]
    fn insert_adds_seeds_for_night() {
        let mut store = SeedStoreOwned::new();
        let night_id = nid(100);

        let seeds = vec![
            make_seed_owned(
                night_id,
                0,
                vec![
                    AlertKey {
                        night_id,
                        idx_in_night: 0,
                    },
                    AlertKey {
                        night_id,
                        idx_in_night: 1,
                    },
                ],
            ),
            make_seed_owned(
                night_id,
                1,
                vec![
                    AlertKey {
                        night_id,
                        idx_in_night: 2,
                    },
                    AlertKey {
                        night_id,
                        idx_in_night: 3,
                    },
                ],
            ),
        ];

        store.insert(night_id, seeds);

        assert_eq!(store.0.len(), 1);
        assert!(store.0.contains_key(&night_id));
        assert_eq!(store.0[&night_id].len(), 2);
    }

    #[test]
    fn insert_overwrites_existing_night() {
        let mut store = SeedStoreOwned::new();
        let night_id = nid(100);

        let seed1 = make_seed_owned(
            night_id,
            0,
            vec![AlertKey {
                night_id,
                idx_in_night: 0,
            }],
        );
        store.insert(night_id, vec![seed1]);
        assert_eq!(store.0[&night_id].len(), 1);

        let seeds2 = vec![
            make_seed_owned(
                night_id,
                0,
                vec![AlertKey {
                    night_id,
                    idx_in_night: 0,
                }],
            ),
            make_seed_owned(
                night_id,
                1,
                vec![AlertKey {
                    night_id,
                    idx_in_night: 1,
                }],
            ),
        ];
        store.insert(night_id, seeds2);
        assert_eq!(store.0[&night_id].len(), 2);
    }

    #[test]
    fn insert_multiple_nights() {
        let mut store = SeedStoreOwned::new();

        for night_val in [100, 101, 102] {
            let night_id = nid(night_val);
            let seed = make_seed_owned(
                night_id,
                0,
                vec![AlertKey {
                    night_id,
                    idx_in_night: 0,
                }],
            );
            store.insert(night_id, vec![seed]);
        }

        assert_eq!(store.0.len(), 3);
        assert!(store.0.contains_key(&nid(100)));
        assert!(store.0.contains_key(&nid(101)));
        assert!(store.0.contains_key(&nid(102)));
    }

    // -------------------------------------------------------------------------
    // to_borrowed (no window)
    // -------------------------------------------------------------------------

    #[test]
    fn to_borrowed_empty_store() {
        let store = SeedStoreOwned::new();
        let alerts = make_alert_store(0, 0);

        let borrowed = store.to_borrowed(&alerts).expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_single_night() {
        let mut store = SeedStoreOwned::new();
        let night_id = nid(0);

        let seed = make_seed_owned(
            night_id,
            0,
            vec![
                AlertKey {
                    night_id,
                    idx_in_night: 0,
                },
                AlertKey {
                    night_id,
                    idx_in_night: 1,
                },
            ],
        );
        store.insert(night_id, vec![seed]);

        let alerts = make_alert_store(10, 5);
        let borrowed = store.to_borrowed(&alerts).expect("should succeed");

        assert_eq!(borrowed.len(), 1);
        assert!(borrowed.contains_night(night_id));
    }

    #[test]
    fn to_borrowed_multiple_nights() {
        let mut store = SeedStoreOwned::new();

        for night_val in 0..3 {
            let night_id = nid(night_val);
            let seed = make_seed_owned(
                night_id,
                0,
                vec![AlertKey {
                    night_id,
                    idx_in_night: 0,
                }],
            );
            store.insert(night_id, vec![seed]);
        }

        let alerts = make_alert_store(10, 5);
        let borrowed = store.to_borrowed(&alerts).expect("should succeed");

        assert_eq!(borrowed.len(), 3);
        for night_val in 0..3 {
            assert!(borrowed.contains_night(nid(night_val)));
        }
    }

    #[test]
    fn to_borrowed_missing_alert_returns_error() {
        let mut store = SeedStoreOwned::new();
        let night_id = nid(100);

        // Seed references alert that doesn't exist
        let seed = make_seed_owned(
            night_id,
            0,
            vec![AlertKey {
                night_id,
                idx_in_night: 999,
            }],
        );
        store.insert(night_id, vec![seed]);

        let alerts = make_alert_store(10, 5);

        let result = store.to_borrowed(&alerts);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // to_borrowed_window: SingleNight mode
    // -------------------------------------------------------------------------

    #[test]
    fn to_borrowed_window_single_night_exact_anchor() {
        let store = make_store_from_nights(&[100, 101, 102]);
        let alerts = make_alert_store(110, 5);
        let mode = PairingMode::single_night(nid(101), 5).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // SingleNight mode with anchor=101, gap=5 → window [96, 101]
        // Should include nights 100, 101 (102 is after anchor)
        assert_eq!(borrowed.len(), 2);
        assert!(borrowed.contains_night(nid(100)));
        assert!(borrowed.contains_night(nid(101)));
        assert!(!borrowed.contains_night(nid(102)));
    }

    #[test]
    fn to_borrowed_window_single_night_with_gap() {
        let store = make_store_from_nights(&[90, 95, 98, 100, 105]);
        let alerts = make_alert_store(110, 5);

        // Anchor = 100, max_gap = 5
        // Window: [95, 100]
        let mode = PairingMode::single_night(nid(100), 5).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // Expected: 95, 98, 100
        // Not included: 90 (outside gap), 105 (after anchor)
        assert_eq!(borrowed.len(), 3);
        assert!(!borrowed.contains_night(nid(90)));
        assert!(borrowed.contains_night(nid(95)));
        assert!(borrowed.contains_night(nid(98)));
        assert!(borrowed.contains_night(nid(100)));
        assert!(!borrowed.contains_night(nid(105)));
    }

    #[test]
    fn to_borrowed_window_single_night_gap_boundary() {
        let store = make_store_from_nights(&[94, 95, 96, 100]);
        let alerts = make_alert_store(110, 5);

        // Anchor = 100, max_gap = 5
        // min_night = 100 - 5 = 95
        let mode = PairingMode::single_night(nid(100), 5).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        assert_eq!(borrowed.len(), 3);
        assert!(!borrowed.contains_night(nid(94))); // 100 - 94 = 6 > gap
        assert!(borrowed.contains_night(nid(95)));
        assert!(borrowed.contains_night(nid(96)));
        assert!(borrowed.contains_night(nid(100)));
    }

    #[test]
    fn to_borrowed_window_single_night_anchor_not_in_store() {
        let store = make_store_from_nights(&[90, 95]);
        let alerts = make_alert_store(110, 5);

        // Anchor = 100 doesn't exist in store
        // Window: [90, 100]
        let mode = PairingMode::single_night(nid(100), 10).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // Should still include nights within window [90, 100]
        assert_eq!(borrowed.len(), 2);
        assert!(borrowed.contains_night(nid(90)));
        assert!(borrowed.contains_night(nid(95)));
    }

    #[test]
    fn to_borrowed_window_single_night_saturating_sub() {
        let store = make_store_from_nights(&[0, 1, 2, 5]);
        let alerts = make_alert_store(10, 5);

        // Anchor = 2, max_gap = 250 → min = saturating_sub(2, 250) = 0
        let mode = PairingMode::single_night(nid(2), 250).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // Should include all nights <= 2
        assert_eq!(borrowed.len(), 3);
        assert!(borrowed.contains_night(nid(0)));
        assert!(borrowed.contains_night(nid(1)));
        assert!(borrowed.contains_night(nid(2)));
        assert!(!borrowed.contains_night(nid(5)));
    }

    #[test]
    fn to_borrowed_window_single_night_no_nights_in_window() {
        let store = make_store_from_nights(&[50, 60]);
        let alerts = make_alert_store(110, 5);

        // Anchor = 100, max_gap = 5 → window [95, 100]
        // All nights are outside
        let mode = PairingMode::single_night(nid(100), 5).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_window_single_night_excludes_nights_after_anchor() {
        let store = make_store_from_nights(&[95, 100, 105, 110]);
        let alerts = make_alert_store(120, 5);

        // Anchor = 100, max_gap = 10 → window [90, 100]
        let mode = PairingMode::single_night(nid(100), 10).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // Should only include 95, 100 (not 105, 110)
        assert_eq!(borrowed.len(), 2);
        assert!(borrowed.contains_night(nid(95)));
        assert!(borrowed.contains_night(nid(100)));
        assert!(!borrowed.contains_night(nid(105)));
        assert!(!borrowed.contains_night(nid(110)));
    }

    // -------------------------------------------------------------------------
    // to_borrowed_window: BatchRange mode
    // -------------------------------------------------------------------------

    #[test]
    fn to_borrowed_window_batch_range_basic() {
        let store = make_store_from_nights(&[40, 60, 80, 100, 120]);
        let alerts = make_alert_store(130, 5);

        // Range [50, 100]
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // Expected: 60, 80, 100
        // Not included: 40 (before start), 120 (after end)
        assert_eq!(borrowed.len(), 3);
        assert!(!borrowed.contains_night(nid(40)));
        assert!(borrowed.contains_night(nid(60)));
        assert!(borrowed.contains_night(nid(80)));
        assert!(borrowed.contains_night(nid(100)));
        assert!(!borrowed.contains_night(nid(120)));
    }

    #[test]
    fn to_borrowed_window_batch_range_exact_boundaries() {
        let store = make_store_from_nights(&[49, 50, 100, 101]);
        let alerts = make_alert_store(110, 5);

        // Range [50, 100] inclusive
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        assert_eq!(borrowed.len(), 2);
        assert!(!borrowed.contains_night(nid(49)));
        assert!(borrowed.contains_night(nid(50)));
        assert!(borrowed.contains_night(nid(100)));
        assert!(!borrowed.contains_night(nid(101)));
    }

    #[test]
    fn to_borrowed_window_batch_range_empty_range() {
        let store = make_store_from_nights(&[40, 120]);
        let alerts = make_alert_store(130, 5);

        // Range [60, 80] but no nights in this range
        let mode = PairingMode::batch_range(nid(60), nid(80)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_window_batch_range_single_night_range() {
        let store = make_store_from_nights(&[50, 100]);
        let alerts = make_alert_store(110, 5);

        // Range [100, 100] - single night
        let mode = PairingMode::batch_range(nid(100), nid(100)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        assert_eq!(borrowed.len(), 1);
        assert!(!borrowed.contains_night(nid(50)));
        assert!(borrowed.contains_night(nid(100)));
    }

    #[test]
    fn to_borrowed_window_batch_range_all_before_start() {
        let store = make_store_from_nights(&[10, 20, 30]);
        let alerts = make_alert_store(110, 5);

        // Range [50, 100] - all nights are before
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_window_batch_range_all_after_end() {
        let store = make_store_from_nights(&[110, 120]);
        let alerts = make_alert_store(130, 5);

        // Range [50, 100] - all nights are after
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_window_batch_range_sparse_distribution() {
        let store = make_store_from_nights(&[50, 75, 150, 175, 200]);
        let alerts = make_alert_store(210, 5);

        // Range [100, 180]
        let mode = PairingMode::batch_range(nid(100), nid(180)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        assert_eq!(borrowed.len(), 2);
        assert!(!borrowed.contains_night(nid(50)));
        assert!(!borrowed.contains_night(nid(75)));
        assert!(borrowed.contains_night(nid(150)));
        assert!(borrowed.contains_night(nid(175)));
        assert!(!borrowed.contains_night(nid(200)));
    }

    #[test]
    fn to_borrowed_window_batch_range_includes_nights_after_start() {
        let store = make_store_from_nights(&[85, 95, 100, 105]);
        let alerts = make_alert_store(110, 5);

        // Range [85, 105] - all nights in range
        let mode = PairingMode::batch_range(nid(85), nid(105)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");

        // BatchRange includes all nights in range
        assert_eq!(borrowed.len(), 4);
        assert!(borrowed.contains_night(nid(85)));
        assert!(borrowed.contains_night(nid(95)));
        assert!(borrowed.contains_night(nid(100)));
        assert!(borrowed.contains_night(nid(105)));
    }

    // -------------------------------------------------------------------------
    // Comparison: SingleNight vs BatchRange behavior
    // -------------------------------------------------------------------------

    #[test]
    fn compare_modes_overlapping_windows() {
        let store = make_store_from_nights(&[95, 98, 100]);
        let alerts = make_alert_store(110, 5);

        // SingleNight: anchor=100, gap=5 → [95, 100]
        let single_mode = PairingMode::single_night(nid(100), 5).unwrap();
        let borrowed_single = store
            .to_borrowed_window(&alerts, single_mode)
            .expect("should succeed");

        // BatchRange: [95, 100]
        let batch_mode = PairingMode::batch_range(nid(95), nid(100)).unwrap();
        let borrowed_batch = store
            .to_borrowed_window(&alerts, batch_mode)
            .expect("should succeed");

        // Both should contain the same nights
        assert_eq!(borrowed_single.len(), borrowed_batch.len());
        assert_eq!(borrowed_single.len(), 3);

        for night in [nid(95), nid(98), nid(100)] {
            assert!(borrowed_single.contains_night(night));
            assert!(borrowed_batch.contains_night(night));
        }
    }

    #[test]
    fn compare_modes_different_semantics() {
        let store = make_store_from_nights(&[85, 95, 100, 105]);
        let alerts = make_alert_store(110, 5);

        println!("store keys: {:?}", store.0.keys());
        println!("alerts keys: {:?}", alerts.nights_sorted());

        // SingleNight: anchor=105, gap=10 → [95, 105]
        let single_mode = PairingMode::single_night(nid(105), 10).unwrap();
        let borrowed_single = store
            .to_borrowed_window(&alerts, single_mode)
            .expect("should succeed");

        println!(
            "borrowed_single night_ids: {:?}",
            borrowed_single.night_ids()
        );

        // BatchRange: [85, 105] → includes all
        let batch_mode = PairingMode::batch_range(nid(85), nid(105)).unwrap();
        let borrowed_batch = store
            .to_borrowed_window(&alerts, batch_mode)
            .expect("should succeed");

        println!("borrowed_batch night_ids: {:?}", borrowed_batch.night_ids());

        // SingleNight excludes nights after anchor
        assert_eq!(borrowed_single.len(), 3);
        assert!(borrowed_single.contains_night(nid(95)));
        assert!(borrowed_single.contains_night(nid(100)));
        assert!(borrowed_single.contains_night(nid(105)));

        // BatchRange includes all nights in range
        assert_eq!(borrowed_batch.len(), 4);
        assert!(borrowed_batch.contains_night(nid(85)));
        assert!(borrowed_batch.contains_night(nid(95)));
        assert!(borrowed_batch.contains_night(nid(100)));
        assert!(borrowed_batch.contains_night(nid(105)));
    }

    // -------------------------------------------------------------------------
    // Edge cases and error handling
    // -------------------------------------------------------------------------

    #[test]
    fn to_borrowed_window_empty_store_single_night() {
        let store = SeedStoreOwned::new();
        let alerts = make_alert_store(20, 5);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_window_empty_store_batch_range() {
        let store = SeedStoreOwned::new();
        let alerts = make_alert_store(20, 5);
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();

        let borrowed = store
            .to_borrowed_window(&alerts, mode)
            .expect("should succeed");
        assert!(borrowed.is_empty());
    }

    #[test]
    fn to_borrowed_window_missing_alerts_returns_error() {
        let mut store = SeedStoreOwned::new();
        let night_id = nid(100);

        // Seed references non-existent alert
        let seed = make_seed_owned(
            night_id,
            0,
            vec![AlertKey {
                night_id,
                idx_in_night: 999,
            }],
        );
        store.insert(night_id, vec![seed]);

        let alerts = make_alert_store(20, 5);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();

        let result = store.to_borrowed_window(&alerts, mode);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Property-based tests
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod proptest_seed_store_owned {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// SingleNight mode: all returned nights must be within [anchor - gap, anchor]
            #[test]
            fn prop_single_night_respects_gap(
                nights in prop::collection::vec(0u32..500, 1..50),
                anchor in 50u32..500,
                gap in 1u8..100,
            ) {
                let store = make_store_from_nights(&nights);
                let alerts = make_alert_store(500, 5);
                let mode = PairingMode::single_night(nid(anchor), gap).unwrap();

                let borrowed = store.to_borrowed_window(&alerts, mode).unwrap();

                let min_night = anchor.saturating_sub(gap as u32);
                for night_id in borrowed.night_ids() {
                    prop_assert!(night_id.0 >= min_night);
                    prop_assert!(night_id.0 <= anchor);
                }
            }

            /// BatchRange mode: all returned nights must be within [start, end]
            #[test]
            fn prop_batch_range_respects_bounds(
                nights in prop::collection::vec(0u32..500, 1..50),
                start in 0u32..400,
                end in 400u32..500,
            ) {
                let store = make_store_from_nights(&nights);
                let alerts = make_alert_store(500, 5);
                let mode = PairingMode::batch_range(nid(start), nid(end)).unwrap();

                let borrowed = store.to_borrowed_window(&alerts, mode).unwrap();

                for night_id in borrowed.night_ids() {
                    prop_assert!(night_id.0 >= start);
                    prop_assert!(night_id.0 <= end);
                }
            }

            /// Invariant: to_borrowed_window never returns more nights than in store
            #[test]
            fn prop_window_never_expands(
                nights in prop::collection::vec(0u32..200, 1..30),
                anchor in 50u32..200,
                gap in 1u8..50,
            ) {
                let store = make_store_from_nights(&nights);
                let alerts = make_alert_store(200, 5);
                let mode = PairingMode::single_night(nid(anchor), gap).unwrap();

                let borrowed = store.to_borrowed_window(&alerts, mode).unwrap();

                prop_assert!(borrowed.len() <= store.0.len());
            }

            /// Invariant: no borrowed store contains nights outside original store
            #[test]
            fn prop_no_phantom_nights(
                nights in prop::collection::vec(0u32..200, 1..30),
                start in 0u32..150,
                end in 150u32..200,
            ) {
                let store = make_store_from_nights(&nights);
                let alerts = make_alert_store(200, 5);
                let mode = PairingMode::batch_range(nid(start), nid(end)).unwrap();

                let borrowed = store.to_borrowed_window(&alerts, mode).unwrap();

                for night_id in borrowed.night_ids() {
                    prop_assert!(store.0.contains_key(&night_id));
                }
            }
        }
    }
}

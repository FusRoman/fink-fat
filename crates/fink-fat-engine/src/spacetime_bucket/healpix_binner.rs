// src/seeding/healpix_binner.rs

//! HEALPix-backed implementation of [`SpatialBinner`] (NESTED scheme).
//!
//! This module provides [`HealpixBinner`], a spatial binning strategy based
//! on the [HEALPix](https://healpix.sourceforge.io) tessellation of the sphere,
//! using the [`cdshealpix`](https://docs.rs/cdshealpix) crate.
//!
//! ## Conventions
//!
//! - The HEALPix order (or **depth**) is denoted `K`.  
//!   The number of divisions per side is `NSIDE = 2^K`.  
//!   The total number of pixels is `12 × 4^K`.  
//! - The binning scheme is **NESTED**, not RING.  
//! - Pixel centers and boundaries are handled by the underlying
//!   [`cdshealpix::nested`] API.
//!
//! ## Neighbor Search
//!
//! The method [`neighbors`](HealpixBinner::neighbors) chooses the strategy
//! depending on the query radius `ang_radius`:
//!
//! - If `ang_radius ≤ cell_radius`: returns the **central pixel** and its
//!   8 immediate neighbors (when available).  
//! - Otherwise: falls back to a **cone search** with approximate coverage
//!   (`cone_coverage_approx_flat`) centered on the pixel center.
//!
//! ## Cell Radius
//!
//! [`cell_radius`](HealpixBinner::cell_radius) is defined as the maximum
//! angular distance between the pixel center and one of its vertices,
//! evaluated at the equator `(lon=0, lat=0)` for the given `depth`.
//!
//! This serves as a characteristic angular size used to decide between
//! local-neighbor vs. cone-coverage strategies.

use std::cell::RefCell;

use cdshealpix as chpx;
use chpx::nested;
use chpx::nested::Layer;
use photom::{Radians, coordinates::equatorial::EquCoord};

use crate::spacetime_bucket::spatial_binner::{SpatialBinner, SpatialKey};

/// Spatial binner backed by **HEALPix** (NESTED scheme).
///
/// Provides mapping between (RA, DEC) positions in radians and HEALPix
/// pixel identifiers, along with neighbor queries.
///
/// Internally uses [`cdshealpix`] for hashing, neighbor lookup,
/// and cone coverage.
#[derive(Clone, Copy)]
pub struct HealpixBinner {
    /// HEALPix order (0 ≤ depth ≤ 29).  
    /// NSIDE = 2^depth ; total pixels = 12 × 4^depth.
    depth: u8,
    /// Underlying HEALPix layer from `cdshealpix`.
    layer: &'static Layer,
    /// Characteristic cell radius (radians).  
    /// Defined as the maximum center→vertex distance at the equator.
    cell_radius: Radians,
}

thread_local! {
    static HEALPIX_NEIGHBOUR_BUF: RefCell<Vec<u64>> =
        RefCell::new(Vec::with_capacity(8));
}

impl HealpixBinner {
    /// Construct a new HEALPix binner for the given order (depth).
    ///
    /// Parameters
    /// ----------
    /// * `depth` — HEALPix order (0..=29).  
    ///   NSIDE = 2^depth ; number of pixels = 12 × 4^depth.
    ///
    /// Returns
    /// -------
    /// A [`HealpixBinner`] ready for spatial binning at the requested depth.
    pub fn new(depth: u8) -> Self {
        let layer = nested::get(depth);
        let cell_radius = chpx::largest_center_to_vertex_distance(depth, 0.0_f64, 0.0_f64);
        Self {
            depth,
            layer,
            cell_radius,
        }
    }

    /// Return the HEALPix order (depth).
    #[inline]
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Return NSIDE, defined as 2^depth.
    #[inline]
    pub fn nside(&self) -> u64 {
        chpx::nside(self.depth) as u64
    }
}

impl SpatialBinner for HealpixBinner {
    /// Compute the HEALPix key for a sky position.
    ///
    /// Parameters
    /// ----------
    /// * `ra` — Right ascension (radians).
    /// * `dec` — Declination (radians).
    ///
    /// Returns
    /// -------
    /// [`SpatialKey`] wrapping the NESTED HEALPix hash.
    #[inline]
    fn key_for(&self, eq_coord: &EquCoord) -> SpatialKey {
        // cdshealpix expects (lon, lat) in radians
        let h = self.layer.hash(eq_coord.ra, eq_coord.dec);
        SpatialKey(h)
    }

    fn neighbors_into(&self, key: SpatialKey, ang_radius: Radians, out: &mut Vec<SpatialKey>) {
        out.clear();

        let SpatialKey(pixel_hash) = key;

        // Compute local cell radius at pixel center (lat-dependent).
        let (lon_center, lat_center) = self.layer.center(pixel_hash);
        let local_cell_radius =
            chpx::largest_center_to_vertex_distance(self.depth, lon_center, lat_center);

        if ang_radius <= local_cell_radius {
            // Local mode: <= 9 pixels (center + up to 8 neighbours).
            out.reserve(9);
            out.push(SpatialKey(pixel_hash));

            HEALPIX_NEIGHBOUR_BUF.with(|buf_cell| {
                let mut buf = buf_cell.borrow_mut();
                buf.clear();

                self.layer.append_bulk_neighbours(pixel_hash, &mut buf);

                // Copy hashes into output as SpatialKey.
                out.extend(buf.iter().copied().map(SpatialKey));
            });
        } else {
            // Cone mode: may return many pixels.
            let hashes =
                nested::cone_coverage_approx_flat(self.depth, lon_center, lat_center, ang_radius);

            // If `hashes` supports iter/len, this is allocation-free here.
            out.reserve(hashes.len());
            out.extend(hashes.iter().copied().map(SpatialKey));
        }
    }

    /// Return neighboring pixels of a given key within `ang_radius`.
    ///
    /// Parameters
    /// ----------
    /// * `key` — Central pixel as [`SpatialKey`].
    /// * `ang_radius` — Angular search radius (radians).
    ///
    /// Behavior
    /// --------
    /// - If `ang_radius ≤ cell_radius(center)`: returns the central pixel
    ///   and its 8 adjacent neighbors (when available).  
    /// - Otherwise: uses cone coverage (`cone_coverage_approx_flat`) centered
    ///   on the pixel center with radius `ang_radius`.
    ///
    /// Returns
    /// -------
    /// Vector of [`SpatialKey`] covering the neighborhood.
    fn neighbors(&self, key: SpatialKey, ang_radius: Radians) -> Vec<SpatialKey> {
        let SpatialKey(h) = key;

        // Local characteristic radius at this pixel center
        let (lon_c, lat_c) = self.layer.center(h);
        let local_rc = chpx::largest_center_to_vertex_distance(self.depth, lon_c, lat_c);

        if ang_radius <= local_rc {
            let mut out = Vec::with_capacity(1 + 8);
            out.push(SpatialKey(h));
            let mut buf: Vec<u64> = Vec::with_capacity(8);
            self.layer.append_bulk_neighbours(h, &mut buf);
            out.extend(buf.into_iter().map(SpatialKey));
            out
        } else {
            let hashes = nested::cone_coverage_approx_flat(self.depth, lon_c, lat_c, ang_radius);
            hashes.into_vec().into_iter().map(SpatialKey).collect()
        }
    }

    /// Characteristic angular radius of a HEALPix cell (radians).
    ///
    /// Definition
    /// ----------
    /// Maximum angular distance between a pixel center and any vertex,
    /// evaluated at the equator `(lon=0, lat=0)` for the given `depth`.
    ///
    /// Notes
    /// -----
    /// - The true center→vertex distance varies slightly with latitude,
    ///   but this value is used as a **simple threshold** for switching
    ///   between local-neighbor vs cone-based search.
    #[inline]
    fn cell_radius(&self) -> Radians {
        self.cell_radius
    }
}

#[cfg(test)]
mod healpix_binner_tests {
    use super::*;
    use std::collections::HashSet;

    fn uniq_len<T: std::hash::Hash + Eq + Copy>(v: &[T]) -> usize {
        let set: HashSet<T> = v.iter().copied().collect();
        set.len()
    }

    #[test]
    fn test_new_basic_properties() {
        let b = HealpixBinner::new(8); // NSIDE = 256
        assert_eq!(b.depth(), 8);
        assert_eq!(b.nside(), 1u64 << 8);
        let r = b.cell_radius();
        assert!(
            r.is_finite() && r > 0.0,
            "cell_radius must be finite and > 0"
        );
        assert!(r < 0.1, "cell_radius unexpectedly large at depth=8: {r}");
    }

    #[test]
    fn test_key_for_is_stable() {
        let b = HealpixBinner::new(7);
        let coord = EquCoord {
            ra: 1.2345_f64,
            ra_error: 0.0_f64,
            dec: 0.1234_f64,
            dec_error: 0.0_f64,
        };
        let k1 = b.key_for(&coord);
        let k2 = b.key_for(&coord);
        assert_eq!(
            k1, k2,
            "key_for should be deterministic for identical inputs"
        );
    }

    #[test]
    fn test_neighbors_small_radius_local_mode() {
        let b = HealpixBinner::new(6);
        let (ra, dec) = (1.0_f64, 0.3_f64);
        let coord = EquCoord {
            ra,
            ra_error: 0.0_f64,
            dec,
            dec_error: 0.0_f64,
        };
        let key = b.key_for(&coord);

        let r = b.cell_radius();
        let neighs = b.neighbors(key, r);

        assert!(
            !neighs.is_empty(),
            "neighbors(<=cell_radius) must include at least the center"
        );
        assert!(
            neighs.len() <= 9,
            "neighbors(<=cell_radius) should not exceed 9 entries (center + up to 8 neighbors)"
        );

        // Unicité
        let ulen = uniq_len(&neighs);
        assert_eq!(ulen, neighs.len(), "neighbors must be unique");

        assert!(
            neighs.contains(&key),
            "neighbors(<=cell_radius) must include the center key"
        );

        let neighs_small = b.neighbors(key, r * 0.999);
        assert!(
            !neighs_small.is_empty() && neighs_small.len() <= 9,
            "neighbors(<cell_radius) should stay in local mode with ≤ 9 entries"
        );
        assert!(
            neighs_small.contains(&key),
            "neighbors(<cell_radius) must include the center key"
        );
        assert_eq!(
            uniq_len(&neighs_small),
            neighs_small.len(),
            "neighbors must be unique"
        );
    }

    #[test]
    fn test_neighbors_large_radius_cone_mode() {
        let b = HealpixBinner::new(7);
        let coord = EquCoord {
            ra: 2.2_f64,
            ra_error: 0.0_f64,
            dec: 0.1_f64,
            dec_error: 0.0_f64,
        };
        let key = b.key_for(&coord);

        let r_cell = b.cell_radius();
        let local = b.neighbors(key, r_cell);

        let big_r = 3.0 * r_cell;
        let cone = b.neighbors(key, big_r);

        assert!(
            cone.contains(&key),
            "cone coverage must include the center key"
        );

        assert!(
            cone.len() >= local.len(),
            "cone coverage should have ≥ local neighbors"
        );

        // Unicité
        assert_eq!(uniq_len(&cone), cone.len(), "cone neighbors must be unique");
        assert!(
            cone.len() >= 2,
            "cone coverage with 3 * cell_radius should hit ≥ 2 pixels"
        );
    }

    #[test]
    fn test_center_roundtrip() {
        let b = HealpixBinner::new(9);
        let coord = EquCoord {
            ra: 0.7_f64,
            ra_error: 0.0_f64,
            dec: -0.3_f64,
            dec_error: 0.0_f64,
        };
        let key = b.key_for(&coord);

        let (lon_c, lat_c) = b.layer.center(key.0);
        let coord_c = EquCoord {
            ra: lon_c,
            ra_error: 0.0_f64,
            dec: lat_c,
            dec_error: 0.0_f64,
        };
        let key_center = b.key_for(&coord_c);
        assert_eq!(
            key, key_center,
            "hash(center_of_pixel) should return the same key"
        );
    }

    #[test]
    fn test_mode_switch_around_threshold() {
        let b = HealpixBinner::new(5);
        let coord = EquCoord {
            ra: 1.4_f64,
            ra_error: 0.0_f64,
            dec: 0.2_f64,
            dec_error: 0.0_f64,
        };
        let key = b.key_for(&coord);
        let rc = b.cell_radius();

        let v_local = b.neighbors(key, rc);
        let v_cone = b.neighbors(key, rc * 1.001);

        assert!(
            v_cone.len() >= v_local.len(),
            "just above cell_radius, cone mode should yield ≥ local neighbors"
        );

        assert!(v_local.contains(&key));
        assert!(v_cone.contains(&key));
        assert_eq!(uniq_len(&v_local), v_local.len());
        assert_eq!(uniq_len(&v_cone), v_cone.len());
    }

    #[cfg(test)]
    mod healpix_binner_prop_tests {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;
        use std::f64::consts::PI;

        const N_CASES: u32 = 64;

        const LAT_EPS: f64 = 1e-6;

        // Stratégie: depth ∈ [3, 12], ra ∈ [0, 2π), dec ∈ [-π/2+ε, π/2-ε]
        fn depth_strategy() -> impl Strategy<Value = u8> {
            3u8..=12u8
        }
        fn ra_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }
        fn dec_strategy() -> impl Strategy<Value = f64> {
            (-(PI / 2.0 - LAT_EPS))..(PI / 2.0 - LAT_EPS)
        }

        fn uniq_len<T: std::hash::Hash + Eq + Copy>(v: &[T]) -> usize {
            let set: HashSet<T> = v.iter().copied().collect();
            set.len()
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: N_CASES, .. ProptestConfig::default() })]

            #[test]
            fn prop_key_determinism_and_center_roundtrip(depth in depth_strategy(), ra in ra_strategy(), dec in dec_strategy()) {
                let b = HealpixBinner::new(depth);
                let coord = EquCoord { ra, ra_error: 0.0_f64, dec, dec_error: 0.0_f64 };
                let k1 = b.key_for(&coord);
                let k2 = b.key_for(&coord);
                prop_assert_eq!(k1, k2, "key_for must be deterministic");

                let (lon_c, lat_c) = b.layer.center(k1.0);
                let coord_c = EquCoord { ra: lon_c, ra_error: 0.0_f64, dec: lat_c, dec_error: 0.0_f64 };
                let kc = b.key_for(&coord_c);
                prop_assert_eq!(kc, k1, "hash(center_of_pixel) must return the same key");
            }

            #[test]
            fn prop_neighbors_local_mode(depth in depth_strategy(), ra in ra_strategy(), dec in dec_strategy()) {
                let b = HealpixBinner::new(depth);
                let coord = EquCoord { ra, ra_error: 0.0_f64, dec, dec_error: 0.0_f64 };
                let key = b.key_for(&coord);
                let r = b.cell_radius();

                let neighs = b.neighbors(key, r);
                prop_assert!(!neighs.is_empty(), "local neighbors must include at least the center");
                prop_assert!(neighs.len() <= 9, "local neighbors should be ≤ 9 (center + up to 8)");
                prop_assert_eq!(uniq_len(&neighs), neighs.len(), "neighbors must be unique");
                prop_assert!(neighs.contains(&key), "center key must be included");
            }

            #[test]
            fn prop_neighbors_cone_mode(depth in depth_strategy(), ra in ra_strategy(), dec in dec_strategy()) {
                let b = HealpixBinner::new(depth);
                let coord = EquCoord { ra, ra_error: 0.0_f64, dec, dec_error: 0.0_f64 };
                let key = b.key_for(&coord);

                let r_cell = b.cell_radius();
                let local = b.neighbors(key, r_cell);

                let big_r = 3.0 * r_cell;
                let cone = b.neighbors(key, big_r);

                prop_assert!(cone.contains(&key), "cone coverage must include the center");
                prop_assert!(cone.len() >= local.len(), "cone coverage should have ≥ local neighbors");
                prop_assert_eq!(uniq_len(&cone), cone.len(), "cone neighbors must be unique");
            }

            #[test]
            fn prop_mode_switch_local_margin(
                depth in depth_strategy(), ra in ra_strategy(), dec in dec_strategy()
            ) {
                let b = HealpixBinner::new(depth);
                let coord = EquCoord { ra, ra_error: 0.0_f64, dec, dec_error: 0.0_f64 };
                let key = b.key_for(&coord);

                let (lon_c, lat_c) = b.layer.center(key.0);
                let local_rc = chpx::largest_center_to_vertex_distance(depth, lon_c, lat_c);

                let v_local = b.neighbors(key, local_rc);
                let v_wider = b.neighbors(key, local_rc * 1.5);

                // invariants
                prop_assert!(v_local.contains(&key));
                prop_assert!(v_wider.contains(&key));
                prop_assert_eq!(uniq_len(&v_local), v_local.len());
                prop_assert_eq!(uniq_len(&v_wider), v_wider.len());

                prop_assert!(
                    v_wider.len() >= v_local.len(),
                    "cone coverage at 1.5× local_rc should be ≥ local neighbors"
                );
            }
        }
    }

    mod neighbor_into_test {
        use super::super::*;
        use super::*;

        /* ------------------------- tests: neighbors_into ------------------------- */

        fn as_set(v: &[SpatialKey]) -> HashSet<SpatialKey> {
            v.iter().copied().collect()
        }

        #[test]
        fn test_neighbors_into_clears_and_includes_center_local_mode() {
            let b = HealpixBinner::new(6);
            let coord = EquCoord {
                ra: 1.0_f64,
                ra_error: 0.0_f64,
                dec: 0.3_f64,
                dec_error: 0.0_f64,
            };
            let key = b.key_for(&coord);

            let r = b.cell_radius();

            // Pre-fill buffer with junk to ensure clear() is applied.
            let mut out = vec![SpatialKey(999), SpatialKey(1000)];

            b.neighbors_into(key, r, &mut out);

            assert!(
                !out.is_empty(),
                "neighbors_into must write at least the center key"
            );
            assert!(out.len() <= 9, "local mode should not exceed 9 entries");
            assert!(
                out.contains(&key),
                "neighbors_into must include the center key"
            );
            assert_eq!(uniq_len(&out), out.len(), "neighbors_into must be unique");
        }

        #[test]
        fn test_neighbors_into_matches_neighbors_local_mode_as_set() {
            let b = HealpixBinner::new(7);
            let coord = EquCoord {
                ra: 0.9_f64,
                ra_error: 0.0_f64,
                dec: -0.2_f64,
                dec_error: 0.0_f64,
            };
            let key = b.key_for(&coord);

            let r = b.cell_radius();

            let v = b.neighbors(key, r);

            let mut out = Vec::<SpatialKey>::new();
            b.neighbors_into(key, r, &mut out);

            assert_eq!(
                as_set(&v),
                as_set(&out),
                "neighbors_into must match neighbors (set-wise)"
            );
            assert_eq!(uniq_len(&out), out.len(), "neighbors_into must be unique");
            assert!(out.contains(&key));
        }

        #[test]
        fn test_neighbors_into_matches_neighbors_cone_mode_as_set() {
            let b = HealpixBinner::new(7);
            let coord = EquCoord {
                ra: 2.2_f64,
                ra_error: 0.0_f64,
                dec: 0.1_f64,
                dec_error: 0.0_f64,
            };
            let key = b.key_for(&coord);

            let big_r = 3.0 * b.cell_radius();

            let v = b.neighbors(key, big_r);

            let mut out = Vec::<SpatialKey>::new();
            b.neighbors_into(key, big_r, &mut out);

            assert!(out.contains(&key), "cone coverage must include center");
            assert_eq!(
                as_set(&v),
                as_set(&out),
                "neighbors_into must match neighbors (set-wise)"
            );
            assert_eq!(uniq_len(&out), out.len(), "neighbors_into must be unique");
        }

        #[test]
        fn test_neighbors_into_does_not_shrink_preallocated_buffer() {
            let b = HealpixBinner::new(6);
            let coord = EquCoord {
                ra: 1.7_f64,
                ra_error: 0.0_f64,
                dec: 0.25_f64,
                dec_error: 0.0_f64,
            };
            let key = b.key_for(&coord);

            let mut out = Vec::<SpatialKey>::with_capacity(256);
            out.extend([SpatialKey(1), SpatialKey(2), SpatialKey(3)]);

            let cap_before = out.capacity();

            // Run both modes.
            let r_local = b.cell_radius();
            b.neighbors_into(key, r_local, &mut out);
            assert_eq!(out.capacity(), cap_before);

            let r_cone = 4.0 * r_local;
            b.neighbors_into(key, r_cone, &mut out);
            assert_eq!(out.capacity(), cap_before);
        }
    }
}

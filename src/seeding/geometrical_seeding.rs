//! src/seeding/seeds.rs
//! Génération de seeds (paires & triplets) à partir de buckets spatio-temporels.
use ahash::AHashMap;

use pyo3::pyclass;

use crate::alerts::{Alert, AlertId};
use crate::seeding::space_time_bucket::{
    BucketIndex, BucketKey, MjdTt, Radians, SpatialBinner, SpatialKey, TimeBin, TimeBinner,
};

/* --------------------------- Params & Types --------------------------- */

#[derive(Clone, Copy, Debug)]
pub struct PairParams {
    /// Δt max entre a et b (jours)
    pub max_dt: MjdTt,
    /// séparation angulaire max entre a et b (radians)
    pub max_sep: Radians,
    /// Inclure les paires dans le même bin temporel ?
    pub allow_same_timebin: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct TripletParams {
    /// Δt max entre observations consécutives (a→b, b→c) (jours)
    pub max_dt_between: MjdTt,
    /// séparation max entre voisins consécutifs (a↔b et b↔c) (rad)
    pub max_pair_sep: Radians,
    /// Résidu max de la prédiction linéaire a→b extrapolée à l’instant de c (rad)
    pub max_predicted_residual: Radians,
    /// Imposer t(a) < t(b) < t(c)
    pub enforce_time_order: bool,
}

/* --------------------------- Lookup utilitaire ------------------------ */

/// Table de lookup id -> &Alert pour accès O(1).
pub struct AlertLookup<'a> {
    by_id: AHashMap<AlertId, &'a Alert>,
}
impl<'a> AlertLookup<'a> {
    pub fn new(alerts: &'a [Alert]) -> Self {
        let mut by_id = AHashMap::with_capacity(alerts.len());
        for a in alerts {
            by_id.insert(a.id, a);
        }
        Self { by_id }
    }
    #[inline]
    pub fn get(&self, id: AlertId) -> &'a Alert {
        self.by_id.get(&id).expect("unknown AlertId in buckets")
    }
}

/// Bins temporels à parcourir à partir d’un bin k0, bornés par max_dt.
/// Si `include_same` = false, on commence à k0+1.
fn time_targets<Bt: TimeBinner>(
    tb: &Bt,
    k0: TimeBin,
    max_dt: f64,
    include_same: bool,
) -> impl Iterator<Item = TimeBin> {
    let w = tb.bin_width().max(1e-12);
    let max_steps = (max_dt / w).ceil().max(0.0) as i64;
    let start = if include_same { 0 } else { 1 };
    (start..=max_steps).map(move |dk| TimeBin(k0.0 + dk))
}

/* --------------------------- Génération des PAIRS --------------------- */

pub fn generate_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: PairParams,
) -> Vec<(AlertId, AlertId)> {
    #[inline]
    fn unit_vec(ra: f64, dec: f64) -> [f64; 3] {
        let c = dec.cos();
        [c * ra.cos(), c * ra.sin(), dec.sin()]
    }
    #[inline]
    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    /// première position `i` telle que `times_by_id[ids[i]] > key_time`
    #[inline]
    fn lower_bound_gt_ids(ids: &[AlertId], key_time: f64, times_by_id: &[f64]) -> usize {
        let (mut lo, mut hi) = (0usize, ids.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let t = times_by_id[ids[mid] as usize];
            if t > key_time {
                hi = mid
            } else {
                lo = mid + 1
            }
        }
        lo
    }

    // --- tables directes indexées par AlertId (zéro HashMap dans la boucle chaude)
    debug_assert!(
        alerts.iter().enumerate().all(|(i, a)| a.id as usize == i),
        "generate_pairs expects contiguous AlertId (id == index)"
    );
    let n = alerts.len();
    let mut times_by_id = Vec::with_capacity(n);
    let mut vecs_by_id = Vec::with_capacity(n);
    for a in alerts {
        times_by_id.push(a.mjd_tt);
        vecs_by_id.push(unit_vec(a.ra, a.dec));
    }

    // caches légers
    let r_search = params.max_sep + sb.cell_radius();
    let cos_thresh = params.max_sep.cos();
    let mut neigh_cache: AHashMap<SpatialKey, Vec<SpatialKey>> = AHashMap::new();
    let mut ttargets_cache: AHashMap<TimeBin, Vec<TimeBin>> = AHashMap::new();

    let mut out: Vec<(AlertId, AlertId)> = Vec::with_capacity(n / 8);

    for (key0, bucket0) in &index.buckets {
        // voisins spatiaux (cache + dédup au cas où)
        let s_neighs = neigh_cache.entry(key0.space_key).or_insert_with(|| {
            let mut v = sb.neighbors(key0.space_key, r_search);
            v.sort_unstable();
            v.dedup();
            v
        });
        // time-bins cibles (cache)
        let ttargets = ttargets_cache.entry(key0.time_bin).or_insert_with(|| {
            time_targets(tb, key0.time_bin, params.max_dt, params.allow_same_timebin).collect()
        });

        // source déjà trié: on itère directement
        for &a_id in &bucket0.members {
            let t_a = times_by_id[a_id as usize];
            let t_max = t_a + params.max_dt;
            let va = vecs_by_id[a_id as usize];

            for &tbin in ttargets.iter() {
                for &s_key in s_neighs.iter() {
                    let k = BucketKey {
                        space_key: s_key,
                        time_bin: tbin,
                    };
                    let Some(btgt) = index.buckets.get(&k) else {
                        continue;
                    };
                    let ids = btgt.members.as_slice();

                    // bsearch vers le premier t_b > t_a
                    let mut i = lower_bound_gt_ids(ids, t_a, &times_by_id);
                    // scan jusqu'à t_b > t_max
                    while i < ids.len() {
                        let b_id = ids[i];
                        let t_b = times_by_id[b_id as usize];
                        if t_b > t_max {
                            break;
                        }
                        if b_id != a_id {
                            let vb = vecs_by_id[b_id as usize];
                            if dot3(va, vb) >= cos_thresh {
                                // ordre temporel garanti (t_b > t_a)
                                out.push((a_id, b_id));
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
    }

    // dédup éventuelle (si le même (a,b) apparaît via plusieurs chemins)
    out.sort_unstable();
    out.dedup();
    out
}

/* --------------------------- Génération des TRIPLETS ------------------ */

pub fn generate_triplets_from_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: TripletParams,
    pairs: &[(AlertId, AlertId)],
) -> Vec<(AlertId, AlertId, AlertId)> {
    #[inline]
    fn wrap_pm_pi(x: f64) -> f64 {
        let two_pi = std::f64::consts::PI * 2.0;
        let mut y = (x + std::f64::consts::PI) % two_pi;
        if y < 0.0 {
            y += two_pi;
        }
        y - std::f64::consts::PI
    }
    #[inline]
    fn unit_vec(ra: f64, dec: f64) -> [f64; 3] {
        let c = dec.cos();
        [c * ra.cos(), c * ra.sin(), dec.sin()]
    }
    #[inline]
    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    /// première position i telle que `times_by_id[ids[i]] > key_time`
    #[inline]
    fn lower_bound_gt_ids(ids: &[AlertId], key_time: f64, times_by_id: &[f64]) -> usize {
        let (mut lo, mut hi) = (0usize, ids.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let t = times_by_id[ids[mid] as usize];
            if t > key_time {
                hi = mid
            } else {
                lo = mid + 1
            }
        }
        lo
    }
    /// offsets plan tangent autour de (ra0, dec0) avec cos(dec0) pré-calculé
    #[inline]
    fn planar_offset_fast(ra0: f64, dec0: f64, cos_dec0: f64, ra: f64, dec: f64) -> (f64, f64) {
        let dx = wrap_pm_pi(ra - ra0) * cos_dec0;
        let dy = dec - dec0;
        (dx, dy)
    }

    // --- tableaux indexés par AlertId (aucun HashMap chaud) ---
    // Hypothèse: id == index (contigu). Si ce n’est pas garanti, on peut ajouter une table de traduction id->index.
    debug_assert!(
        alerts.iter().enumerate().all(|(i, a)| a.id as usize == i),
        "generate_triplets_from_pairs expects contiguous AlertId (id == index)"
    );
    let n = alerts.len();
    let mut times_by_id = Vec::with_capacity(n);
    let mut ra_by_id = Vec::with_capacity(n);
    let mut dec_by_id = Vec::with_capacity(n);
    let mut cosdec_by_id = Vec::with_capacity(n);
    let mut vecs_by_id = Vec::with_capacity(n);
    for a in alerts {
        times_by_id.push(a.mjd_tt);
        ra_by_id.push(a.ra);
        dec_by_id.push(a.dec);
        cosdec_by_id.push(a.dec.cos());
        vecs_by_id.push(unit_vec(a.ra, a.dec));
    }

    // (optionnel) pré-calcul des BucketKey natifs des ids pour ce (sb,tb).
    // Utile si tu as énormément de paires, ça évite des recomputations key_for/bin_for.
    let mut spacekey_by_id = Vec::with_capacity(n);
    let mut timebin_by_id = Vec::with_capacity(n);
    for a in alerts {
        spacekey_by_id.push(sb.key_for(a.ra, a.dec));
        timebin_by_id.push(tb.bin_for(a.mjd_tt));
    }

    // caches légers
    let r_search = params.max_pair_sep + sb.cell_radius();
    let cos_pair = params.max_pair_sep.cos();
    let mut neigh_cache: AHashMap<SpatialKey, Vec<SpatialKey>> = AHashMap::new();
    let mut ttargets_cache: AHashMap<TimeBin, Vec<TimeBin>> = AHashMap::new();

    let mut out: Vec<(AlertId, AlertId, AlertId)> = Vec::with_capacity(pairs.len() / 2);

    for &(a_id, b_id) in pairs {
        let t_a = times_by_id[a_id as usize];
        let t_b = times_by_id[b_id as usize];

        // On impose dt_ab>0 pour stabilité (même si enforce_time_order=false)
        if t_b <= t_a {
            continue;
        }
        if params.enforce_time_order && !(t_a < t_b) {
            continue;
        }

        // centre “b” : clé spatiale et time bin (pré-calculées)
        let key_b = BucketKey {
            space_key: spacekey_by_id[b_id as usize],
            time_bin: timebin_by_id[b_id as usize],
        };

        // voisins spatiaux (cache + dédup)
        let s_neighs = neigh_cache.entry(key_b.space_key).or_insert_with(|| {
            let mut v = sb.neighbors(key_b.space_key, r_search);
            v.sort_unstable();
            v.dedup();
            v
        });
        // bins temporels cibles (cache): t in (t_b, t_b + dt_max]
        let ttargets = ttargets_cache.entry(key_b.time_bin).or_insert_with(|| {
            time_targets(tb, key_b.time_bin, params.max_dt_between, false).collect()
        });

        // Mouvement a→b (plan tangent autour de a)
        let ra_a = ra_by_id[a_id as usize];
        let dec_a = dec_by_id[a_id as usize];
        let cos_a = cosdec_by_id[a_id as usize];
        let ra_b = ra_by_id[b_id as usize];
        let dec_b = dec_by_id[b_id as usize];

        let (dx_ab, dy_ab) = planar_offset_fast(ra_a, dec_a, cos_a, ra_b, dec_b);
        let dt_ab = (t_b - t_a).max(1e-12);
        let vx = dx_ab / dt_ab;
        let vy = dy_ab / dt_ab;

        let vb = vecs_by_id[b_id as usize]; // pour test angulaire b↔c
        let t_bmax = t_b + params.max_dt_between;

        for &tbin in ttargets.iter() {
            for &s_key in s_neighs.iter() {
                let k = BucketKey {
                    space_key: s_key,
                    time_bin: tbin,
                };
                let Some(bucket_c) = index.buckets.get(&k) else {
                    continue;
                };
                let ids = bucket_c.members.as_slice(); // trié par temps

                // bsearch: première c avec t_c > t_b
                let mut i = lower_bound_gt_ids(ids, t_b, &times_by_id);

                // scan jusqu'à t_c > t_b + Δt_between
                while i < ids.len() {
                    let c_id = ids[i];
                    if c_id == a_id || c_id == b_id {
                        i += 1;
                        continue;
                    }

                    let t_c = times_by_id[c_id as usize];
                    if t_c > t_bmax {
                        break;
                    }

                    // Filtre pairwise b↔c via dot-product
                    let vc = vecs_by_id[c_id as usize];
                    if dot3(vb, vc) >= cos_pair {
                        // Résidu linéaire: projeter a→b à t(c), comparer à c (plan tangent autour de a)
                        let dt_ac = t_c - t_a;
                        if dt_ac > 0.0 {
                            // prédiction
                            let ra_pred = ra_a + vx * dt_ac / cos_a.max(1e-12);
                            let dec_pred = dec_a + vy * dt_ac;

                            // résidu
                            let (dx_pc, dy_pc) = planar_offset_fast(
                                ra_a,
                                dec_a,
                                cos_a,
                                ra_by_id[c_id as usize],
                                dec_by_id[c_id as usize],
                            );
                            let (dx_pp, dy_pp) =
                                planar_offset_fast(ra_a, dec_a, cos_a, ra_pred, dec_pred);
                            let resid = ((dx_pc - dx_pp).powi(2) + (dy_pc - dy_pp).powi(2)).sqrt();

                            if resid <= params.max_predicted_residual {
                                // ordre temporel déjà garanti: (a < b < c) ⇒ pas besoin de trier
                                out.push((a_id, b_id, c_id));
                            }
                        }
                    }
                    i += 1;
                }
            }
        }
    }

    // dédup (au cas où un même triplet apparaisse via 2 chemins de voisinage)
    out.sort_unstable();
    out.dedup();
    out
}

/// Résultat combiné : *toutes* les paires + les triplets générés à partir de ces paires.
/// Les paires sont **conservées même si aucun triplet** n’a été trouvé pour elles.
#[pyclass(module = "fink_fat")]
#[derive(Clone, Debug, Default)]
pub struct SeedSets {
    pub pairs: Vec<(AlertId, AlertId)>,
    pub triplets: Vec<(AlertId, AlertId, AlertId)>,
}

/// Génère d’abord **toutes** les paires (selon `pair_params`), puis les **triplets**
/// dérivés de ces paires. Les paires sont retournées en entier (on ne retire pas celles
/// utilisées dans un triplet).
pub fn generate_pairs_and_triplets<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    pair_params: PairParams,
    triplet_params: TripletParams,
) -> SeedSets {
    println!("Generating pairs...");
    let pairs = generate_pairs(index, alerts, sb, tb, pair_params);

    println!("Generating triplets from {} pairs...", pairs.len());

    let triplets = generate_triplets_from_pairs(index, alerts, sb, tb, triplet_params, &pairs);
    SeedSets { pairs, triplets }
}

pub fn generate_triplets<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: TripletParams,
) -> Vec<(AlertId, AlertId, AlertId)> {
    // Paires « locales » en amont : même logique que précédemment
    let pair_params = PairParams {
        max_dt: params.max_dt_between,
        max_sep: params.max_pair_sep,
        allow_same_timebin: false,
    };
    let pairs = generate_pairs(index, alerts, sb, tb, pair_params);
    generate_triplets_from_pairs(index, alerts, sb, tb, params, &pairs)
}

#[cfg(test)]
mod geom_seeds_tests {
    use super::*;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    use crate::alerts::{Alert, AlertId};
    use crate::seeding::healpix_binners::HealpixBinner;
    use crate::seeding::space_time_bucket::{build_index_from_alerts_precise, BucketKey};
    use crate::seeding::uniform_time_binner::UniformTimeBinner;

    /* ------------------------- helpers ------------------------- */

    fn mk_alert(id: AlertId, ra: f64, dec: f64, mjd_tt: f64, band: u8) -> Alert {
        Alert {
            id,
            dia_source_id: id as u64,
            ra,
            dec,
            mjd_tt,
            flux: 0.0,
            flux_err: 0.0,
            band,
        }
    }

    #[inline]
    fn arcsec_to_rad(x: f64) -> f64 {
        x * PI / (180.0 * 3600.0)
    }

    #[inline]
    fn wrap_pm_pi(x: f64) -> f64 {
        let mut y = (x + PI) % (2.0 * PI);
        if y < 0.0 {
            y += 2.0 * PI;
        }
        y - PI
    }

    #[inline]
    fn ang_sep(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
        let s1 = dec1.sin();
        let c1 = dec1.cos();
        let s2 = dec2.sin();
        let c2 = dec2.cos();
        let dlon = wrap_pm_pi(ra2 - ra1);
        let cos_d = s1 * s2 + c1 * c2 * dlon.cos();
        (1.0 - cos_d.clamp(-1.0, 1.0)).max(0.0).sqrt().asin() * 2.0
    }

    fn find_alert<'a>(alerts: &'a [Alert], id: AlertId) -> &'a Alert {
        alerts
            .iter()
            .find(|a| a.id == id)
            .expect("alert id not found")
    }

    /* --------------------- unit tests (deterministes) --------------------- */

    #[test]
    fn pairs_basic_one_pair() {
        // Binning
        let sb = HealpixBinner::new(10); // NSIDE=1024
        let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

        // Deux alertes séparées de ~5 arcsec et 8 min -> devraient matcher (Δt<=10 min, sep<=10")
        let t0 = 60000.10;
        let a1 = mk_alert(0, 1.0, 0.2, t0, 1);
        let a2 = mk_alert(
            1,
            1.0 + arcsec_to_rad(5.0) / 0.2_f64.cos(),
            0.2,
            t0 + 8.0 / 1440.0,
            1,
        );

        // Un outlier loin (ne doit pas matcher)
        let a3 = mk_alert(2, 2.0, -0.3, t0 + 5.0 / 1440.0, 1);

        let alerts = vec![a1.clone(), a2.clone(), a3.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let pairs = generate_pairs(
            &index,
            &alerts,
            &sb,
            &tb,
            PairParams {
                max_dt: 10.0 / 1440.0,        // 10 min
                max_sep: arcsec_to_rad(10.0), // 10"
                allow_same_timebin: false,
            },
        );

        assert!(pairs.contains(&(0, 1)) || pairs.contains(&(1, 2)));
        assert!(!pairs
            .iter()
            .any(|&(i, j)| (i == 1 && j == 3) || (i == 2 && j == 3)));
        // unicité
        let set: HashSet<_> = pairs.iter().collect();
        assert_eq!(set.len(), pairs.len());
    }

    #[test]
    fn pairs_same_timebin_behavior() {
        let sb = HealpixBinner::new(9);
        let tb = UniformTimeBinner::new(60000.0, 20.0 / 1440.0); // 20 min bins

        let t0 = 60000.25;
        // Deux alertes dans le même bin temporel (Δt = 5 min < 20 min)
        let a1 = mk_alert(0, 1.5, 0.1, t0, 1);
        let a2 = mk_alert(
            1,
            1.5 + arcsec_to_rad(4.0) / 0.1_f64.cos(),
            0.1,
            t0 + 5.0 / 1440.0,
            1,
        );

        let alerts = vec![a1.clone(), a2.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        // Interdit same timebin -> aucune paire
        let pairs_no_same = generate_pairs(
            &index,
            &alerts,
            &sb,
            &tb,
            PairParams {
                max_dt: 30.0 / 1440.0,
                max_sep: arcsec_to_rad(8.0),
                allow_same_timebin: false,
            },
        );
        assert!(pairs_no_same.is_empty());

        // Autorisé -> la paire doit apparaître
        let pairs_same = generate_pairs(
            &index,
            &alerts,
            &sb,
            &tb,
            PairParams {
                max_dt: 30.0 / 1440.0,
                max_sep: arcsec_to_rad(8.0),
                allow_same_timebin: true,
            },
        );
        assert_eq!(pairs_same.len(), 1);
        let (i, j) = pairs_same[0];
        assert!((i == 0 && j == 1) || (i == 1 && j == 0));
    }

    #[test]
    fn triplets_linear_motion_detected() {
        let sb = HealpixBinner::new(10);
        let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

        let t0 = 60000.0;
        // Mouvement linéaire: ~6" toutes les 10 min le long de RA (plan tangent)
        let dec0: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0, 1.0, dec0, t0, 1);
        let b = mk_alert(1, 1.0 + dr, dec0, t0 + 10.0 / 1440.0, 1);
        let c = mk_alert(2, 1.0 + 2.0 * dr, dec0, t0 + 20.0 / 1440.0, 1);

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let triplets = generate_triplets(
            &index,
            &alerts,
            &sb,
            &tb,
            TripletParams {
                max_dt_between: 30.0 / 1440.0,              // 30 min
                max_pair_sep: arcsec_to_rad(15.0),          // 15"
                max_predicted_residual: arcsec_to_rad(3.0), // 3"
                enforce_time_order: true,
            },
        );

        // On s'attend à (21,22,23)
        assert!(triplets.contains(&(0, 1, 2)));
        // unicité
        let set: HashSet<_> = triplets.iter().collect();
        assert_eq!(set.len(), triplets.len());
    }

    #[test]
    fn triplets_large_residual_rejected() {
        let sb = HealpixBinner::new(10);
        let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0);

        let t0 = 60000.0;
        let dec0: f64 = 0.1;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0, 2.0, dec0, t0, 1);
        let b = mk_alert(1, 2.0 + dr, dec0, t0 + 10.0 / 1440.0, 1);
        // 3ème point dévié de ~40" -> résidu devrait dépasser 5"
        let c = mk_alert(
            2,
            2.0 + 2.0 * dr + arcsec_to_rad(40.0) / dec0.cos(),
            dec0,
            t0 + 20.0 / 1440.0,
            1,
        );

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let triplets = generate_triplets(
            &index,
            &alerts,
            &sb,
            &tb,
            TripletParams {
                max_dt_between: 30.0 / 1440.0,
                max_pair_sep: arcsec_to_rad(60.0), // pairwise OK
                max_predicted_residual: arcsec_to_rad(5.0), // mais trop strict pour la déviation
                enforce_time_order: true,
            },
        );

        assert!(!triplets.contains(&(0, 1, 2)));
    }

    #[test]
    fn pairs_are_kept_when_no_triplet_found() {
        let sb = HealpixBinner::new(9);
        let tb = UniformTimeBinner::new(61000.0, 10.0 / 1440.0); // 10 min

        // Deux points compatibles en Δt/Δθ, mais aucun 3e point dans la fenêtre -> pas de triplet.
        let t0 = 61000.20;
        let dec = 0.2;
        let a = mk_alert(0, 1.0, dec, t0, 1);
        let b = mk_alert(
            1,
            1.0 + arcsec_to_rad(6.0) / dec.cos(),
            dec,
            t0 + 8.0 / 1440.0,
            1,
        );

        let alerts = vec![a.clone(), b.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let seeds = generate_pairs_and_triplets(
            &index,
            &alerts,
            &sb,
            &tb,
            PairParams {
                max_dt: 15.0 / 1440.0,
                max_sep: arcsec_to_rad(10.0),
                allow_same_timebin: false,
            },
            TripletParams {
                max_dt_between: 10.0 / 1440.0, // c devrait être <=10 min après b (absent ici)
                max_pair_sep: arcsec_to_rad(12.0),
                max_predicted_residual: arcsec_to_rad(3.0),
                enforce_time_order: true,
            },
        );

        // On conserve la paire même sans triplet
        assert_eq!(seeds.triplets.len(), 0);
        assert_eq!(seeds.pairs.len(), 1);
        let (i, j) = seeds.pairs[0];
        assert!((i == 0 && j == 1) || (i == 1 && j == 0));
    }

    #[test]
    fn both_pairs_and_triplet_returned() {
        let sb = HealpixBinner::new(10);
        let tb = UniformTimeBinner::new(61000.0, 10.0 / 1440.0); // 10 min

        let t0 = 61000.0;
        let dec: f64 = 0.15;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        let a = mk_alert(0, 2.0, dec, t0, 1);
        let b = mk_alert(1, 2.0 + dr, dec, t0 + 10.0 / 1440.0, 1);
        let c = mk_alert(2, 2.0 + 2.0 * dr, dec, t0 + 20.0 / 1440.0, 1);

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let seeds = generate_pairs_and_triplets(
            &index,
            &alerts,
            &sb,
            &tb,
            PairParams {
                max_dt: 25.0 / 1440.0, // autorise (a,b) et (b,c); (a,c) = 20 min aussi
                max_sep: arcsec_to_rad(15.0),
                allow_same_timebin: false,
            },
            TripletParams {
                max_dt_between: 15.0 / 1440.0, // a→b et b→c valides
                max_pair_sep: arcsec_to_rad(15.0),
                max_predicted_residual: arcsec_to_rad(3.0),
                enforce_time_order: true,
            },
        );

        // Triplet détecté
        assert!(seeds.triplets.contains(&(0, 1, 2)));
        // Les paires incluent au moins (a,b) et (b,c) (et possiblement (a,c) selon max_dt)
        let mut pair_set = std::collections::HashSet::new();
        for &(i, j) in &seeds.pairs {
            pair_set.insert(if i < j { (i, j) } else { (j, i) });
        }
        assert!(pair_set.contains(&(0, 1)));
        assert!(pair_set.contains(&(1, 2)));
    }

    #[test]
    fn triplets_from_pairs_is_subset_of_pairs_prefix() {
        // Vérifie que chaque (a,b,c) renvoyé provient d'une paire (a,b) appartenant au set pairs.
        let sb = HealpixBinner::new(9);
        let tb = UniformTimeBinner::new(62000.0, 10.0 / 1440.0);

        let t0 = 62000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(8.0) / dec.cos();

        let a = mk_alert(0, 0.6, dec, t0, 1);
        let b = mk_alert(1, 0.6 + dr, dec, t0 + 10.0 / 1440.0, 1);
        let c = mk_alert(2, 0.6 + 2.0 * dr, dec, t0 + 20.0 / 1440.0, 1);
        let d = mk_alert(3, 2.5, 0.0, t0 + 5.0 / 1440.0, 1); // bruit

        let alerts = vec![a, b, c, d];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let seeds = generate_pairs_and_triplets(
            &index,
            &alerts,
            &sb,
            &tb,
            PairParams {
                max_dt: 30.0 / 1440.0,
                max_sep: arcsec_to_rad(20.0),
                allow_same_timebin: false,
            },
            TripletParams {
                max_dt_between: 20.0 / 1440.0,
                max_pair_sep: arcsec_to_rad(20.0),
                max_predicted_residual: arcsec_to_rad(4.0),
                enforce_time_order: true,
            },
        );

        // Construire un set des paires (ordre canonique i<j)
        let mut pair_set = std::collections::HashSet::new();
        for &(i, j) in &seeds.pairs {
            pair_set.insert(if i < j { (i, j) } else { (j, i) });
        }

        for &(i, j, _) in &seeds.triplets {
            // (i,j) appartient au set pairs (par construction)
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            assert!(
                pair_set.contains(&(a, b)),
                "triplet must originate from an existing pair"
            );
        }
    }

    /* --------------------- property tests (robustes) --------------------- */

    mod geom_seeds_prop {
        use super::*;
        use proptest::prelude::*;

        const LAT_EPS: f64 = 1e-6;

        fn ra_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }
        fn dec_strategy() -> impl Strategy<Value = f64> {
            (-(PI / 2.0 - LAT_EPS))..(PI / 2.0 - LAT_EPS)
        }
        fn t_strategy() -> impl Strategy<Value = f64> {
            // ~ 4 h de fenêtre
            60000.0f64..60000.1667f64 // 0.1667 ~ 4 h
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

            /// Tous les pairs retournés vérifient les contraintes Δt et Δθ,
            /// et appartiennent à des buckets compatibles (voisinage spatio-temporel).
            #[test]
            fn prop_pairs_respect_constraints_and_buckets(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 0..120)
            ) {
                let sb = HealpixBinner::new(8);
                let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min
                let params = PairParams {
                    max_dt: 30.0 / 1440.0,        // 30 min
                    max_sep: arcsec_to_rad(20.0), // 20"
                    allow_same_timebin: false,
                };
                let search_radius = params.max_sep + sb.cell_radius();

                // build alerts
                let alerts: Vec<Alert> = triples.iter().enumerate().map(|(i, (ra, dec, t))| {
                    mk_alert(i as u32, *ra, *dec, *t, 1)
                }).collect();
                let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

                let pairs = generate_pairs(&index, &alerts, &sb, &tb, params);

                // unicité
                let set: HashSet<_> = pairs.iter().collect();
                prop_assert_eq!(set.len(), pairs.len());

                // contraintes
                for (i, j) in pairs {
                    let a = find_alert(&alerts, i);
                    let b = find_alert(&alerts, j);
                    // ordre temporel dans l'impl
                    prop_assert!(b.mjd_tt > a.mjd_tt);
                    prop_assert!((b.mjd_tt - a.mjd_tt) <= params.max_dt);
                    let d = ang_sep(a.ra, a.dec, b.ra, b.dec);
                    prop_assert!(d <= params.max_sep);

                    // buckets compatibles :
                    let key_a = BucketKey { space_key: sb.key_for(a.ra, a.dec), time_bin: tb.bin_for(a.mjd_tt) };
                    let neighs = sb.neighbors(key_a.space_key, search_radius);
                    let allowed_bins: HashSet<i64> = {
                        let w = tb.bin_width().max(1e-12);
                        let max_steps = (params.max_dt / w).ceil().max(0.0) as i64;
                        // allow_same_timebin=false -> commence à +1
                        (1..=max_steps).map(|dk| key_a.time_bin.0 + dk).collect()
                    };
                    let key_b = BucketKey { space_key: sb.key_for(b.ra, b.dec), time_bin: tb.bin_for(b.mjd_tt) };
                    prop_assert!(neighs.into_iter().any(|k| k == key_b.space_key));
                    prop_assert!(allowed_bins.contains(&key_b.time_bin.0));
                }
            }

            /// Tous les triplets respectent (a,b) et (b,c) en Δt/Δθ et le résidu de prédiction,
            /// et appartiennent à des buckets compatibles (voisinages).
            #[test]
            fn prop_triplets_respect_constraints_and_buckets(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 0..100)
            ) {
                let sb = HealpixBinner::new(8);
                let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min
                let params = TripletParams {
                    max_dt_between: 40.0 / 1440.0,        // 40 min
                    max_pair_sep: arcsec_to_rad(30.0),    // 30"
                    max_predicted_residual: arcsec_to_rad(10.0), // 10"
                    enforce_time_order: true,
                };
                let pair_search_radius = params.max_pair_sep + sb.cell_radius();

                let alerts: Vec<Alert> = triples.iter().enumerate().map(|(i, (ra, dec, t))| {
                    mk_alert(i as u32, *ra, *dec, *t, 1)
                }).collect();
                let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

                let triplets = generate_triplets(&index, &alerts, &sb, &tb, params);
                // unicité
                let set: HashSet<_> = triplets.iter().collect();
                prop_assert_eq!(set.len(), triplets.len());

                for (i, j, k) in triplets {
                    let a = find_alert(&alerts, i);
                    let b = find_alert(&alerts, j);
                    let c = find_alert(&alerts, k);

                    // ordre temporel
                    prop_assert!(a.mjd_tt < b.mjd_tt && b.mjd_tt < c.mjd_tt);

                    // contraintes pairwise Δt/Δθ
                    let dt_ab = b.mjd_tt - a.mjd_tt;
                    let dt_bc = c.mjd_tt - b.mjd_tt;
                    prop_assert!(dt_ab <= params.max_dt_between && dt_bc <= params.max_dt_between);

                    let dab = ang_sep(a.ra, a.dec, b.ra, b.dec);
                    let dbc = ang_sep(b.ra, b.dec, c.ra, c.dec);
                    prop_assert!(dab <= params.max_pair_sep && dbc <= params.max_pair_sep);

                    // résidu de prédiction linéaire (recalcule comme dans l'impl)
                    let (dx_ab, dy_ab) = {
                        let dx = wrap_pm_pi(b.ra - a.ra) * a.dec.cos();
                        let dy = b.dec - a.dec;
                        (dx, dy)
                    };
                    let vx = dx_ab / dt_ab.max(1e-12);
                    let vy = dy_ab / dt_ab.max(1e-12);
                    let dt_ac = c.mjd_tt - a.mjd_tt;
                    let ra_pred  = a.ra + vx * dt_ac / a.dec.cos().max(1e-12);
                    let dec_pred = a.dec + vy * dt_ac;
                    let (dx_pc, dy_pc) = {
                        let dx = wrap_pm_pi(c.ra - a.ra) * a.dec.cos();
                        let dy = c.dec - a.dec;
                        (dx, dy)
                    };
                    let (dx_pp, dy_pp) = {
                        let dx = wrap_pm_pi(ra_pred - a.ra) * a.dec.cos();
                        let dy = dec_pred - a.dec;
                        (dx, dy)
                    };
                    let resid = ((dx_pc - dx_pp).powi(2) + (dy_pc - dy_pp).powi(2)).sqrt();
                    prop_assert!(resid <= params.max_predicted_residual);

                    // buckets compatibles:
                    // (a,b) : b dans les voisins et bins autorisés de a
                    let key_a = BucketKey { space_key: sb.key_for(a.ra, a.dec), time_bin: tb.bin_for(a.mjd_tt) };
                    let key_b = BucketKey { space_key: sb.key_for(b.ra, b.dec), time_bin: tb.bin_for(b.mjd_tt) };
                    let neighs_ab = sb.neighbors(key_a.space_key, pair_search_radius);
                    prop_assert!(neighs_ab.into_iter().any(|k| k == key_b.space_key));
                    {
                        let w = tb.bin_width().max(1e-12);
                        let max_steps = (params.max_dt_between / w).ceil().max(0.0) as i64;
                        // allow_same_timebin=false dans generate_pairs en amont
                        let allowed_ab: HashSet<i64> = (1..=max_steps).map(|dk| key_a.time_bin.0 + dk).collect();
                        prop_assert!(allowed_ab.contains(&key_b.time_bin.0));
                    }

                    // (b,c) : c dans les voisins et bins autorisés de b
                    let key_b2 = key_b;
                    let key_c = BucketKey { space_key: sb.key_for(c.ra, c.dec), time_bin: tb.bin_for(c.mjd_tt) };
                    let neighs_bc = sb.neighbors(key_b2.space_key, pair_search_radius);
                    prop_assert!(neighs_bc.into_iter().any(|k| k == key_c.space_key));
                    {
                        let w = tb.bin_width().max(1e-12);
                        let max_steps = (params.max_dt_between / w).ceil().max(0.0) as i64;
                        let allowed_bc: HashSet<i64> = (1..=max_steps).map(|dk| key_b2.time_bin.0 + dk).collect();
                        prop_assert!(allowed_bc.contains(&key_c.time_bin.0));
                    }
                }
            }
        }
    }
}

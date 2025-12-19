use approx::assert_relative_eq;
use fink_fat_engine::seeding::pairs::{Pair, Pairs};
use fink_fat_engine::seeding::triplets::{Triplet, Triplets};
use fink_fat_engine::{Alert, AlertId, alerts::AlertStore};

use fink_fat_eval::dataset::ztf_alerts::AlertStoreWithTruth;
use fink_fat_eval::seeding::metrics::{pair_metrics, triplet_metrics};

fn mk_alert(id: usize, mjd_tt: f64) -> Alert {
    Alert {
        id: AlertId::from(id),
        dia_source_id: id as u64,
        ra: 0.0,
        ra_err: 0.0,
        dec: 0.0,
        dec_err: 0.0,
        mjd_tt,
        flux: 0.0,
        flux_err: 0.0,
        band: 1,
    }
}

fn mk_store_with_truth(tids: Vec<i32>, times: Vec<f64>) -> AlertStoreWithTruth {
    assert_eq!(tids.len(), times.len());
    let alerts: Vec<Alert> = times
        .into_iter()
        .enumerate()
        .map(|(i, t)| mk_alert(i, t))
        .collect();

    let min_mjd = alerts
        .iter()
        .map(|a| a.mjd_tt)
        .fold(f64::INFINITY, f64::min);
    let store = AlertStore::new(min_mjd.floor(), alerts);

    AlertStoreWithTruth {
        store,
        trajectory_id: tids,
    }
}

fn p(a: usize, b: usize) -> Pair {
    Pair {
        a: AlertId::from(a),
        b: AlertId::from(b),
    }
}

fn t(a: usize, b: usize, c: usize) -> Triplet {
    Triplet {
        a: AlertId::from(a),
        b: AlertId::from(b),
        c: AlertId::from(c),
    }
}

#[test]
fn integration_metrics_smoke() {
    let store = mk_store_with_truth(vec![1, 1, 2, 2, 0], vec![10.0, 11.0, 12.0, 13.0, 14.0]);

    let pairs: Pairs = vec![p(0, 1), p(2, 3), p(1, 2)];
    let pm = pair_metrics(&store, &pairs);

    assert_eq!(pm.n_total, 3);
    assert_eq!(pm.n_true, 2); // (0,1) and (2,3)
    assert_eq!(pm.n_contaminated, 1); // (1,2)
    assert_relative_eq!(pm.precision_on_truth, 2.0 / 3.0, epsilon = 1e-12);

    let triplets: Triplets = vec![t(0, 1, 4), t(0, 1, 2), t(2, 3, 1)];
    let tm = triplet_metrics(&store, &triplets);

    assert_eq!(tm.n_total, 3);
    assert_eq!(tm.n_true, 0);
    assert_eq!(tm.n_contaminated, 2); // (0,1,2) and (2,3,1) are all-truth but mixed
}

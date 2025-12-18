use crate::dataset::{
    ParquetSource,
    ztf_alerts::{ZtfAlertScan, alert_store_with_truth_from_lazyframe, scan_ztf_alerts},
};

pub mod dataset;

fn main() {
    let parquet_source = ParquetSource::new("../../test_exp/ztf_alert.parquet").unwrap();
    let scan = ZtfAlertScan {
        only_truth: true,
        ..Default::default()
    };

    let lf = scan_ztf_alerts(&parquet_source, scan).unwrap();

    // println!("LazyFrame schema: {:?}", lf.last().collect());

    let alert_store = alert_store_with_truth_from_lazyframe(lf, Default::default()).unwrap();

    println!("{}", alert_store);

    for alert in alert_store.store.iter().take(10) {
        println!(
            "alert {:?} : {} (trajectory_id: {})",
            alert.id,
            alert,
            alert_store.trajectory_id[alert.id.idx()]
        );
    }

    println!("===============\n\nAlerts for trajectory_id = 33803:");

    let traj_id = 33803;
    for alert in alert_store.alerts_for_trajectory(traj_id) {
        println!(
            "Trajectory {} alert {:?} : {}",
            traj_id,
            alert.id,
            alert,
        );
    }
}

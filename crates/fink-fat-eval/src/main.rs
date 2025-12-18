use crate::dataset::{
    ParquetSource,
    ztf_alerts::{ZtfAlertScan, alert_store_from_lazyframe, scan_ztf_alerts},
};

pub mod dataset;

fn main() {
    let parquet_source = ParquetSource::new("../../test_exp/ztf_alert.parquet").unwrap();
    let scan = ZtfAlertScan::default();

    let lf = scan_ztf_alerts(&parquet_source, scan).unwrap();

    // println!("LazyFrame schema: {:?}", lf.last().collect());

    let alert_store = alert_store_from_lazyframe(lf, Default::default()).unwrap();

    println!("{}", alert_store);

    for alert in alert_store.iter().take(5) {
        println!("alert {:?} : {}", alert.id, alert);
    }
}

//! Multi-night seeding evaluation (pairs & triplets).
//!
//! Overview
//! --------
//! This binary evaluates the **intra-night seeding stage** (pairs + triplets)
//! across **multiple nights** contained in a single ZTF/LSST-like alert Parquet file.
//!
//! For each night (`nid`):
//! 1. Scan + ingest only that night's alerts into an [`AlertStoreWithTruth`],
//! 2. Build the spatio-temporal bucket index,
//! 3. Generate pairs (and optionally triplets),
//! 4. Compute quality metrics against truth association,
//! 5. Record timings and counts.
//!
//! Outputs
//! -------
//! Writes `multinight_seeding_summary.csv` into `--out-dir`.

use std::fs;

use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use polars::prelude::*;

use fink_fat_engine::{
    engine_config::{pair_config::PairConfig, triplet_config::TripletConfig},
    night_id::NightId,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use fink_fat_eval::{
    bin_utils::{fmt_ms, infer_t0_mjd_tt, ingest_one_night, resolve_nids},
    cli::common::{CommonBinningArgs, CommonPairGenArgs, CommonScanArgs},
    dataset::{ParquetSource, ingest_config::AlertIngestConfig, schema::cols},
    seeding::{
        metrics::{pair_metrics, triplet_metrics},
        plotting::{MultiNightPlotConfig, plot_multinight_summary},
        seed_gen::generate_pairs_and_triplets_ids_only,
    },
};

#[derive(Parser, Debug)]
#[command(
    name = "seeding-multinight-eval",
    version,
    about = "Evaluate intra-night seeding (pairs/triplets) over multiple nights and export a CSV summary.",
    disable_help_subcommand = true,
    verbatim_doc_comment
)]
struct Cli {
    #[command(flatten)]
    scan: CommonScanArgs,

    #[command(flatten)]
    pair_gen: CommonPairGenArgs,

    #[command(flatten)]
    binning: CommonBinningArgs,

    /// Optional explicit list of night ids to process (comma-separated).
    ///
    /// Example: `--nids 3122,3145,3156`
    #[arg(long, value_name = "NID1,NID2,...", help_heading = "Scan")]
    nids: Option<String>,

    /// Limit the number of nights processed (after sorting / selection).
    #[arg(long, value_name = "N", help_heading = "Scan")]
    max_nights: Option<usize>,

    /// Disable triplet generation (pairs-only benchmark).
    #[arg(long, default_value_t = false, help_heading = "Seeding")]
    pairs_only: bool,

    /// Override the default output filename.
    #[arg(
        long,
        default_value = "multinight_seeding_summary.csv",
        value_name = "FILE",
        help_heading = "I/O"
    )]
    out_csv: String,

    /// Keep running even if one night fails ingestion/generation.
    ///
    /// Failed nights will be skipped and reported at the end.
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help_heading = "Run control"
    )]
    continue_on_error: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("failed to create out dir: {}", cli.scan.out_dir))?;

    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;

    // Determine which nights to process (shared logic).
    let nids = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    eprintln!(
        "Processing {} night(s). Output dir: {}",
        nids.len(),
        cli.scan.out_dir
    );

    let ingest_cfg = AlertIngestConfig::default();

    let pair_cfg = PairConfig {
        max_dt: cli.pair_gen.max_dt,
        max_angular_speed: cli.pair_gen.gen_max_omega.as_rad_per_day(),
        max_flux_difference: cli.pair_gen.max_flux_difference,
        allow_same_timebin: cli.pair_gen.allow_same_timebin,
    };
    let triplet_cfg = TripletConfig::default();

    // CSV columns
    let mut v_nid: Vec<u32> = Vec::with_capacity(nids.len());
    let mut v_n_alerts: Vec<i64> = Vec::with_capacity(nids.len());

    let mut v_n_pairs: Vec<i64> = Vec::with_capacity(nids.len());
    let mut v_n_triplets: Vec<i64> = Vec::with_capacity(nids.len());

    let mut v_dt_bucket_ms: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_dt_pairs_ms: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_dt_triplets_ms: Vec<f64> = Vec::with_capacity(nids.len());

    // pair metrics
    let mut v_pair_purity_overall: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_pair_precision_on_truth: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_pair_consecutive_recall: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_pair_n_true: Vec<i64> = Vec::with_capacity(nids.len());
    let mut v_pair_n_contaminated: Vec<i64> = Vec::with_capacity(nids.len());

    // triplet metrics
    let mut v_trip_purity_overall: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_trip_precision_on_truth: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_trip_consecutive_recall: Vec<f64> = Vec::with_capacity(nids.len());
    let mut v_trip_n_true: Vec<i64> = Vec::with_capacity(nids.len());
    let mut v_trip_n_contaminated: Vec<i64> = Vec::with_capacity(nids.len());

    let mut failed: Vec<(u32, String)> = Vec::new();

    for (k, &nid) in nids.iter().enumerate() {
        eprintln!("\n[{}/{}] nid={}", k + 1, nids.len(), nid);

        let mut run_one = || -> Result<()> {
            let store = ingest_one_night(
                &source,
                NightId(nid),
                cli.scan.mode.into(),
                cli.scan.minimal,
                &ingest_cfg,
            )?;

            let n_alerts = store.store.alerts.len() as i64;
            eprintln!("  alerts: {}", n_alerts);

            // binners (origin inferred per night: stable and usually desirable for a per-night benchmark)
            let t0 = infer_t0_mjd_tt(&store);
            let spatial = HealpixBinner::new(cli.binning.healpix_depth);
            let time = UniformTimeBinner::new(cli.binning.time_bin_days, t0);

            let (pairs, triplets, timings) = generate_pairs_and_triplets_ids_only(
                &store,
                &spatial,
                &time,
                &pair_cfg,
                &triplet_cfg,
            );

            eprintln!("  bucket-index: {:.3} ms", fmt_ms(timings.bucket_index));
            eprintln!(
                "  pairs:        {} ({:.3} ms)",
                pairs.len(),
                fmt_ms(timings.pairs)
            );
            if !cli.pairs_only {
                eprintln!(
                    "  triplets:     {} ({:.3} ms)",
                    triplets.len(),
                    fmt_ms(timings.triplets)
                );
            }

            let pm = pair_metrics(&store, &pairs);
            eprintln!(
                "  pair purity_overall={:.4} precision_on_truth={:.4} consecutive_recall={:.4}",
                pm.purity_overall, pm.precision_on_truth, pm.consecutive_recall
            );

            let tm = if cli.pairs_only {
                None
            } else {
                let tm = triplet_metrics(&store, &triplets);
                eprintln!(
                    "  trip purity_overall={:.4} precision_on_truth={:.4} consecutive_recall={:.4}",
                    tm.purity_overall, tm.precision_on_truth, tm.consecutive_recall
                );
                Some(tm)
            };

            // record row
            v_nid.push(nid);
            v_n_alerts.push(n_alerts);
            v_n_pairs.push(pairs.len() as i64);
            v_n_triplets.push(if cli.pairs_only {
                0
            } else {
                triplets.len() as i64
            });

            v_dt_bucket_ms.push(fmt_ms(timings.bucket_index));
            v_dt_pairs_ms.push(fmt_ms(timings.pairs));
            v_dt_triplets_ms.push(fmt_ms(timings.triplets));

            v_pair_purity_overall.push(pm.purity_overall);
            v_pair_precision_on_truth.push(pm.precision_on_truth);
            v_pair_consecutive_recall.push(pm.consecutive_recall);
            v_pair_n_true.push(pm.n_true as i64);
            v_pair_n_contaminated.push(pm.n_contaminated as i64);

            if let Some(tm) = tm {
                v_trip_purity_overall.push(tm.purity_overall);
                v_trip_precision_on_truth.push(tm.precision_on_truth);
                v_trip_consecutive_recall.push(tm.consecutive_recall);
                v_trip_n_true.push(tm.n_true as i64);
                v_trip_n_contaminated.push(tm.n_contaminated as i64);
            } else {
                v_trip_purity_overall.push(f64::NAN);
                v_trip_precision_on_truth.push(f64::NAN);
                v_trip_consecutive_recall.push(f64::NAN);
                v_trip_n_true.push(0);
                v_trip_n_contaminated.push(0);
            }

            Ok(())
        };

        if let Err(e) = run_one() {
            if cli.continue_on_error {
                failed.push((nid, format!("{e:#}")));
                eprintln!("  !! failed nid={nid} (continuing): {e:#}");
                continue;
            } else {
                return Err(e);
            }
        }
    }

    let mut df = DataFrame::new(vec![
        Series::new(cols::NID.into(), v_nid).into(),
        Series::new("n_alerts".into(), v_n_alerts).into(),
        Series::new("n_pairs".into(), v_n_pairs).into(),
        Series::new("n_triplets".into(), v_n_triplets).into(),
        Series::new("dt_bucket_ms".into(), v_dt_bucket_ms).into(),
        Series::new("dt_pairs_ms".into(), v_dt_pairs_ms).into(),
        Series::new("dt_triplets_ms".into(), v_dt_triplets_ms).into(),
        Series::new("pair_purity_overall".into(), v_pair_purity_overall).into(),
        Series::new("pair_precision_on_truth".into(), v_pair_precision_on_truth).into(),
        Series::new("pair_consecutive_recall".into(), v_pair_consecutive_recall).into(),
        Series::new("pair_n_true".into(), v_pair_n_true).into(),
        Series::new("pair_n_contaminated".into(), v_pair_n_contaminated).into(),
        Series::new("triplet_purity_overall".into(), v_trip_purity_overall).into(),
        Series::new(
            "triplet_precision_on_truth".into(),
            v_trip_precision_on_truth,
        )
        .into(),
        Series::new(
            "triplet_consecutive_recall".into(),
            v_trip_consecutive_recall,
        )
        .into(),
        Series::new("triplet_n_true".into(), v_trip_n_true).into(),
        Series::new("triplet_n_contaminated".into(), v_trip_n_contaminated).into(),
    ])?;

    let out_csv = cli.scan.out_dir.join(&cli.out_csv);
    let mut file = std::fs::File::create(out_csv.as_std_path())
        .with_context(|| format!("failed to create {}", out_csv))?;

    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut df)
        .context("failed to write CSV summary")?;

    eprintln!("\nWrote {}", out_csv);

    if !failed.is_empty() {
        eprintln!("\nSkipped {} night(s) due to errors:", failed.len());
        for (nid, err) in failed {
            eprintln!("  - nid={nid}: {err}");
        }
    }

    let plot_cfg = MultiNightPlotConfig {
        width: 1400,
        height: 900,
        log_cost: true,
    };

    let outs = plot_multinight_summary(&df, cli.scan.out_dir.as_std_path(), &plot_cfg)?;
    for p in outs {
        eprintln!("Wrote plot {}", p.display());
    }

    Ok(())
}

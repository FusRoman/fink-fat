use std::{fs, time::Instant};

/// Command-line tool to optimize position scoring parameters
/// using balanced inter-night edge samples.
//// The tool performs a random search over position scoring parameters,
/// evaluating each candidate configuration on a set of balanced inter-night edges
/// frozen from the provided dataset. The goal is to minimize the false positive rate (FPR)
/// at a target true positive rate (TPR) by adjusting scoring parameters related to position differences.

/// Example command to run the tool:
/// ```bash
/// clear && cargo run     \
/// --release     \
/// -p fink-fat-eval     \
/// --bin generate-features \
/// ../../test_exp/ztf_dataset_2025.parquet \
/// --engine-config src/bin/scoring/config_engine.yaml \
/// --mode oracle \
/// -vv \
/// --out-dir edge_features_dataset
/// ```
use anyhow::{Context, Result};
use camino::Utf8Path;
use clap::Parser;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::edge::{
        edge_id::EdgeId,
        {Edge, features::EdgeFeatures},
    },
    night_id::NightId,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};
use fink_fat_eval::{
    bin_utils::resolve_nids,
    cli::scoring::{Cli, update_score_config},
    log, log_section, log_timing, log2,
    night_seeds::{NightSeeds, generate_seed_store},
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;

use rayon::prelude::*;

use std::fs::File;

use polars::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Build the list of (left, right, gap) pairs to evaluate.
///
/// - `consecutive_window`: evaluates i -> i+1..i+W
/// - `gap_min..=gap_max`: evaluates i -> i+gap for each gap in the range
///
/// Returned `gap` is the number of nights between `left` and `right` (>= 1).
fn build_night_pairs(
    nids: &[NightId],
    consecutive_window: usize,
    gap_min: usize,
    gap_max: usize,
) -> Vec<(NightId, NightId, usize)> {
    let mut out = Vec::new();
    if nids.len() < 2 {
        return out;
    }

    let w = consecutive_window.max(1);

    // 1) Sliding consecutive window: i -> i+1..i+w
    for i in 0..nids.len().saturating_sub(1) {
        for k in 1..=w {
            let j = i + k;
            if j >= nids.len() {
                break;
            }
            out.push((nids[i], nids[j], k));
        }
    }

    // 2) Explicit gaps: i -> i+gap_min..i+gap_max
    if gap_min >= 1 && gap_max >= gap_min {
        for i in 0..nids.len() {
            for g in gap_min..=gap_max {
                let j = i + g;
                if j >= nids.len() {
                    break;
                }
                out.push((nids[i], nids[j], g));
            }
        }
    }

    out
}

/// Convert resolved i32 night ids into typed NightId.
fn to_night_ids(nids: Vec<u32>) -> Vec<NightId> {
    nids.into_iter().map(NightId).collect()
}

/// Step 1: build the time binner (anchored on `right`) and generate Top-K edges.
fn step_generate_topk_edges<'a>(
    left: &'a NightSeeds,
    right: &'a NightSeeds,
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
) -> (UniformTimeBinner, Vec<Edge<'a>>) {
    let min_time = right
        .seeds
        .iter()
        .map(|s| s.plane.epoch_mid)
        .fold(f64::INFINITY, f64::min);

    let time_binner = UniformTimeBinner::new(min_time, 30.0 / 60.0 / 24.0); // 30 minutes

    // Edge generation
    let edges = Edge::generate_topk_edges(
        EdgeId(0),
        &left.seeds,
        &right.seeds,
        &engine_cfg.edges,
        spatial_binner,
        &time_binner,
    );

    (time_binner, edges)
}

struct PairParquetJob {
    path: camino::Utf8PathBuf,
    df: DataFrame,
    n_edges: u64,
    n_true: u64,
    n_false: u64,
}

fn build_pair_job_from_edges<'a>(
    out_dir: &Utf8Path,
    idx: usize,
    left_nid: NightId,
    right_nid: NightId,
    gap: usize,
    left: &'a NightSeeds,
    right: &'a NightSeeds,
    edges: &[Edge<'a>],
) -> Result<PairParquetJob> {
    let n = edges.len();

    // ---------------------------------------------------------------------
    // Label + counts
    // ---------------------------------------------------------------------
    let mut is_true = Vec::with_capacity(n);
    let mut n_true: u64 = 0;

    // ---------------------------------------------------------------------
    // Debug columns
    // ---------------------------------------------------------------------
    let mut edge_id_col = Vec::with_capacity(n);
    let mut from_seed_id = Vec::with_capacity(n);
    let mut to_seed_id = Vec::with_capacity(n);

    let mut from_night_id = Vec::with_capacity(n);
    let mut to_night_id = Vec::with_capacity(n);

    let mut t_from = Vec::with_capacity(n);
    let mut t_to = Vec::with_capacity(n);

    let mut from_ra_mid = Vec::with_capacity(n);
    let mut from_dec_mid = Vec::with_capacity(n);
    let mut to_ra_mid = Vec::with_capacity(n);
    let mut to_dec_mid = Vec::with_capacity(n);

    let mut from_ra0 = Vec::with_capacity(n);
    let mut from_dec0 = Vec::with_capacity(n);

    let mut from_flux_mean = Vec::with_capacity(n);
    let mut to_flux_mean = Vec::with_capacity(n);
    let mut from_flux_std = Vec::with_capacity(n);
    let mut to_flux_std = Vec::with_capacity(n);

    let mut from_vx = Vec::with_capacity(n);
    let mut from_vy = Vec::with_capacity(n);
    let mut to_vx = Vec::with_capacity(n);
    let mut to_vy = Vec::with_capacity(n);

    let mut from_has_acc = Vec::with_capacity(n);
    let mut from_ax = Vec::with_capacity(n);
    let mut from_ay = Vec::with_capacity(n);
    let mut to_has_acc = Vec::with_capacity(n);
    let mut to_ax = Vec::with_capacity(n);
    let mut to_ay = Vec::with_capacity(n);

    // ---------------------------------------------------------------------
    // Features: generated dynamically from the canonical flat iterator
    // ---------------------------------------------------------------------
    let feature_names: Vec<&'static str> = EdgeFeatures::flat_names().collect();
    let mut feature_cols: Vec<Vec<f64>> = (0..feature_names.len())
        .map(|_| Vec::with_capacity(n))
        .collect();

    // ---------------------------------------------------------------------
    // Fill rows
    // ---------------------------------------------------------------------
    for e in edges {
        let feat: EdgeFeatures = e.compute_features();

        let t = left.edge_truth(right, e).unwrap_or(false);
        if t {
            n_true += 1;
        }
        is_true.push(t);

        // Debug fields
        edge_id_col.push(e.id.0 as i64);
        from_seed_id.push(e.from.seed_id.0 as i64);
        to_seed_id.push(e.to.seed_id.0 as i64);

        from_night_id.push(e.from.night_id.0 as i64);
        to_night_id.push(e.to.night_id.0 as i64);

        t_from.push(e.from.plane.epoch_mid);
        t_to.push(e.to.plane.epoch_mid);

        from_ra_mid.push(e.from.plane.ra_mid);
        from_dec_mid.push(e.from.plane.dec_mid);
        to_ra_mid.push(e.to.plane.ra_mid);
        to_dec_mid.push(e.to.plane.dec_mid);

        from_ra0.push(e.from.plane.center.ra0);
        from_dec0.push(e.from.plane.center.dec0);

        from_flux_mean.push(e.from.photom.flux_mean as f64);
        to_flux_mean.push(e.to.photom.flux_mean as f64);
        from_flux_std.push(e.from.photom.flux_std as f64);
        to_flux_std.push(e.to.photom.flux_std as f64);

        from_vx.push(e.from.plane.vel_xy[0]);
        from_vy.push(e.from.plane.vel_xy[1]);
        to_vx.push(e.to.plane.vel_xy[0]);
        to_vy.push(e.to.plane.vel_xy[1]);

        if let Some(a) = e.from.plane.acc_xy {
            from_has_acc.push(true);
            from_ax.push(a[0]);
            from_ay.push(a[1]);
        } else {
            from_has_acc.push(false);
            from_ax.push(0.0);
            from_ay.push(0.0);
        }

        if let Some(a) = e.to.plane.acc_xy {
            to_has_acc.push(true);
            to_ax.push(a[0]);
            to_ay.push(a[1]);
        } else {
            to_has_acc.push(false);
            to_ax.push(0.0);
            to_ay.push(0.0);
        }

        // Features in canonical order (no string matching, no allocations)
        for (k, v) in feat.iter_flat().enumerate() {
            // Safety: should always match by construction
            if let Some(col) = feature_cols.get_mut(k) {
                col.push(v);
            }
        }
    }

    let n_edges = n as u64;
    let n_false = n_edges - n_true;

    let left_nid_col = vec![left_nid.0 as i64; n];
    let right_nid_col = vec![right_nid.0 as i64; n];
    let gap_col = vec![gap as i64; n];

    // ---------------------------------------------------------------------
    // Build DataFrame columns
    // ---------------------------------------------------------------------
    let mut cols: Vec<Column> = Vec::with_capacity(3 + 1 + 30 + feature_names.len());

    cols.push(Column::new("left_nid".into(), left_nid_col));
    cols.push(Column::new("right_nid".into(), right_nid_col));
    cols.push(Column::new("gap_nights".into(), gap_col));

    cols.push(Column::new("is_true_edge".into(), is_true));

    // debug ids
    cols.push(Column::new("edge_id".into(), edge_id_col));
    cols.push(Column::new("from_seed_id".into(), from_seed_id));
    cols.push(Column::new("to_seed_id".into(), to_seed_id));
    cols.push(Column::new("from_night_id".into(), from_night_id));
    cols.push(Column::new("to_night_id".into(), to_night_id));

    // debug time/coords
    cols.push(Column::new("t_from".into(), t_from));
    cols.push(Column::new("t_to".into(), t_to));
    cols.push(Column::new("from_ra_mid".into(), from_ra_mid));
    cols.push(Column::new("from_dec_mid".into(), from_dec_mid));
    cols.push(Column::new("to_ra_mid".into(), to_ra_mid));
    cols.push(Column::new("to_dec_mid".into(), to_dec_mid));
    cols.push(Column::new("from_ra0".into(), from_ra0));
    cols.push(Column::new("from_dec0".into(), from_dec0));

    // debug photom
    cols.push(Column::new("from_flux_mean".into(), from_flux_mean));
    cols.push(Column::new("to_flux_mean".into(), to_flux_mean));
    cols.push(Column::new("from_flux_std".into(), from_flux_std));
    cols.push(Column::new("to_flux_std".into(), to_flux_std));

    // debug kin
    cols.push(Column::new("from_vx".into(), from_vx));
    cols.push(Column::new("from_vy".into(), from_vy));
    cols.push(Column::new("to_vx".into(), to_vx));
    cols.push(Column::new("to_vy".into(), to_vy));
    cols.push(Column::new("from_has_acc".into(), from_has_acc));
    cols.push(Column::new("from_ax".into(), from_ax));
    cols.push(Column::new("from_ay".into(), from_ay));
    cols.push(Column::new("to_has_acc".into(), to_has_acc));
    cols.push(Column::new("to_ax".into(), to_ax));
    cols.push(Column::new("to_ay".into(), to_ay));

    // feature columns (canonical names like "position.chi2_pos", ...)
    for (name, values) in feature_names.into_iter().zip(feature_cols.into_iter()) {
        cols.push(Column::new(name.into(), values));
    }

    let df = DataFrame::new(cols)?;

    let path = out_dir.join(format!(
        "edge_features_{idx:06}_left{:05}_right{:05}_gap{}.parquet",
        left_nid.0, right_nid.0, gap
    ));

    Ok(PairParquetJob {
        path,
        df,
        n_edges,
        n_true,
        n_false,
    })
}

fn write_job_to_parquet(job: &mut PairParquetJob) -> Result<()> {
    let mut file = File::create(job.path.as_std_path())
        .with_context(|| format!("create parquet file {}", job.path))?;

    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut job.df)
        .map_err(|e| anyhow::anyhow!("parquet write failed: {e}"))?;

    Ok(())
}

fn main() -> Result<()> {
    let t0 = std::time::Instant::now();
    let cli = Cli::parse();

    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("create {}", cli.scan.out_dir))?;

    // -------------------------------------------------------------------------
    // Engine config
    // -------------------------------------------------------------------------
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("load engine config {}", cli.engine_config))?;

    let updated_config = update_score_config(&cli, &engine_cfg);

    let jobs = cli.jobs.unwrap_or_else(num_cpus::get);

    log!(cli, "Using {jobs} parallel jobs (threads)");

    ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .ok();

    log_timing!(cli, "Script initialized", t0.elapsed());

    log_section!(&cli, "Dataset ingest");

    let seed_store = generate_seed_store(&cli, &updated_config)?;

    log!(&cli, "SeedStore nights: {}", seed_store.len());
    log2!(&cli, "{seed_store}");

    let t_resolve = Instant::now();
    let nids: Vec<u32> = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    log_timing!(&cli, "resolve_nids", t_resolve.elapsed());
    anyhow::ensure!(nids.len() >= 2, "need at least 2 nights to evaluate edges");

    let consecutive_window: usize = 10;
    let gap_min: usize = 0;
    let gap_max: usize = 0;

    let t_pairs = Instant::now();
    let pairs = build_night_pairs(&to_night_ids(nids), consecutive_window, gap_min, gap_max);
    log_timing!(&cli, "build_night_pairs", t_pairs.elapsed());
    anyhow::ensure!(!pairs.is_empty(), "no night pairs to evaluate");

    log!(
        &cli,
        "Pairs to evaluate: {} (consecutive_window={}, gap=[{},{}])",
        pairs.len(),
        consecutive_window,
        gap_min,
        gap_max
    );

    let spatial_binner = HealpixBinner::new(cli.binning.healpix_depth);

    log_section!(&cli, "Evaluation loop");

    const PAIR_BATCH_SIZE: usize = 20;

    let nb_pairs = pairs.len();
    let nb_batches = (nb_pairs + PAIR_BATCH_SIZE - 1) / PAIR_BATCH_SIZE;

    let (mp, pb_batches, pb_pairs) = if cli.quiet {
        (None, ProgressBar::hidden(), ProgressBar::hidden())
    } else {
        let mp = MultiProgress::new();

        let style_batches = ProgressStyle::with_template(
        "{spinner:.green} [batches] [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({eta_precise})",
    )
    .unwrap()
    .progress_chars("=>-");

        let style_pairs = ProgressStyle::with_template(
        "{spinner:.green} [pairs  ] [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({eta_precise})",
    )
    .unwrap()
    .progress_chars("=>-");

        let pb_batches = mp.add(ProgressBar::new(nb_batches as u64));
        pb_batches.set_style(style_batches);
        pb_batches.enable_steady_tick(std::time::Duration::from_millis(200));

        let pb_pairs = mp.add(ProgressBar::new(nb_pairs as u64));
        pb_pairs.set_style(style_pairs);
        pb_pairs.enable_steady_tick(std::time::Duration::from_millis(200));

        (Some(mp), pb_batches, pb_pairs)
    };

    // For rayon closures
    let pb_pairs = Arc::new(pb_pairs);

    // -------------------------------------------------------------------------
    // Evaluate (batched parallel compute + sequential write)
    // -------------------------------------------------------------------------

    let total_edges = AtomicU64::new(0);
    let total_true = AtomicU64::new(0);
    let total_false = AtomicU64::new(0);

    for (batch_id, batch) in pairs.chunks(PAIR_BATCH_SIZE).enumerate() {
        println!("Processing pairs :");
        for (left_nid, right_nid, gap) in batch {
            println!(
                "  left_nid: {}, right_nid: {}, gap: {}",
                left_nid.0, right_nid.0, gap
            );
        }
        println!("---\n");

        // Compute jobs in parallel for this batch
        let mut jobs: Vec<PairParquetJob> = batch
        .par_iter()
        .enumerate()
        .filter_map(|(j, (left_nid, right_nid, gap))| {
            let idx = batch_id * PAIR_BATCH_SIZE + j;

            let left = seed_store.get(left_nid).unwrap();
            let right = seed_store.get(right_nid).unwrap();

            if left.seeds.len() + right.seeds.len() >= 50_000 {
                eprintln!(
                    "Skipping pair #{idx:06}: left_nid={} ({} seeds) + right_nid={} ({} seeds) exceeds limit",
                    left_nid.0,
                    left.seeds.len(),
                    right_nid.0,
                    right.seeds.len(),
                );
                // Skip this pair (do not error the whole batch)
                return None;
            }

            println!(
                "Generating edges for pair #{idx:06}: left_nid={} ({} seeds) -> right_nid={} ({} seeds), gap={} nights",
                left_nid.0,
                left.seeds.len(),
                right_nid.0,
                right.seeds.len(),
                gap
            );

            let (_, edges) =
                step_generate_topk_edges(left, right, &updated_config, &spatial_binner);

            // keep returning Result so we can propagate real errors from build_pair_job_from_edges
            let job_res: Result<PairParquetJob> = build_pair_job_from_edges(
                &cli.scan.out_dir,
                idx,
                *left_nid,
                *right_nid,
                *gap,
                left,
                right,
                &edges,
            );

            // Progress only if job succeeded
            if job_res.is_ok() {
                pb_pairs.inc(1);
            }

            Some(job_res)
    })
    .collect::<Result<Vec<_>>>()?; // propagates errors from build_pair_job_from_edges

        println!(
            "Batch #{:03}: generated {} parquet jobs",
            batch_id,
            jobs.len()
        );

        // Write sequentially
        for job in jobs.iter_mut() {
            if job.n_edges == 0 {
                println!("Skipping parquet write for {} (0 edges)", job.path.as_str());
                continue;
            }
            write_job_to_parquet(job)?;

            total_edges.fetch_add(job.n_edges, Ordering::Relaxed);
            total_true.fetch_add(job.n_true, Ordering::Relaxed);
            total_false.fetch_add(job.n_false, Ordering::Relaxed);
        }

        // Batch progress after successful write of the batch
        pb_batches.inc(1);
    }

    pb_batches.finish_with_message("all batches done");
    pb_pairs.finish_with_message("all pairs done");

    if let Some(mp) = mp {
        // Optional: clear UI once finished
        let _ = mp.clear();
    }

    log_section!(&cli, "Results");
    let n_edges = total_edges.load(Ordering::Relaxed) as usize;
    let n_true = total_true.load(Ordering::Relaxed) as usize;
    let n_false = total_false.load(Ordering::Relaxed) as usize;

    log!(cli, "Total edges generated: {}", n_edges);
    if n_edges > 0 {
        log!(
            cli,
            "True edges: {} ({:.2}%)",
            n_true,
            100.0 * n_true as f64 / n_edges as f64
        );
        log!(
            cli,
            "False edges: {} ({:.2}%)",
            n_false,
            100.0 * n_false as f64 / n_edges as f64
        );
    }

    Ok(())
}

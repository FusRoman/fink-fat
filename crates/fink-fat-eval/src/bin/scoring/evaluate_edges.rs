use std::{collections::BTreeMap, fs, time::Instant};

use ahash::AHashMap;
use anyhow::{Context, Result};
use clap::Parser;

use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::{edge::Edge, edge_id::EdgeId},
    night_id::NightId,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use fink_fat_eval::{
    bin_utils::resolve_nids,
    buflog, buflog_section, buflog_timing, buflog2,
    cli::scoring::Cli,
    dataset::{
        ParquetSource,
        ingest_config::AlertIngestConfig,
        ztf_alerts::{NightStore, collect_nights},
    },
    log, log_section, log_timing, log2,
    night_seeds::SeedStore,
    scoring::edges_diagnostics::{
        EdgesStatsDisplay, MissReason, diagnose_missed_true_edges,
        edge_truth_counts_between_nights, edge_truth_diagnostics,
    },
};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

/// Per-pair summary we accumulate.
#[derive(Debug, Clone, Copy, Default)]
struct PairSummary {
    n_true: usize,
    n_false: usize,
    n_true_possible: usize,
    recall: f64,
    missed_top_k: usize,
    missed_generation: usize,
    true_generated: usize,
    false_generated: usize,
}

#[derive(Debug, Default)]
struct Agg {
    n_pairs: usize,

    sum_true: usize,
    sum_false: usize,
    sum_true_possible: usize,

    sum_recall: f64,          // mean of per-pair recall
    sum_recall_weighted: f64, // weighted by n_true_possible

    sum_missed_top_k: usize,
    sum_missed_generation: usize,
}

impl Agg {
    fn push(&mut self, s: PairSummary) {
        self.n_pairs += 1;

        self.sum_true += s.n_true;
        self.sum_false += s.n_false;
        self.sum_true_possible += s.n_true_possible;

        self.sum_recall += s.recall;
        self.sum_recall_weighted += s.recall * (s.n_true_possible as f64);

        self.sum_missed_top_k += s.missed_top_k;
        self.sum_missed_generation += s.missed_generation;
    }

    fn mean_recall(&self) -> f64 {
        if self.n_pairs == 0 {
            return f64::NAN;
        }
        self.sum_recall / (self.n_pairs as f64)
    }

    fn weighted_recall(&self) -> f64 {
        if self.sum_true_possible == 0 {
            return f64::NAN;
        }
        self.sum_recall_weighted / (self.sum_true_possible as f64)
    }
}

/// Convert resolved i32 night ids into typed NightId.
fn to_night_ids(nids: Vec<u32>) -> Vec<NightId> {
    nids.into_iter().map(NightId).collect()
}

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

/// Step 1: build the time binner (anchored on `right`) and generate Top-K edges.
fn step_generate_topk_edges<'a>(
    logbuf: &mut String,
    cli: &Cli,
    left: &'a fink_fat_eval::night_seeds::NightSeeds,
    right: &'a fink_fat_eval::night_seeds::NightSeeds,
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
) -> (UniformTimeBinner, Vec<Edge<'a>>) {
    // Build time binner anchored on the right night (stable).
    let t_timebin = Instant::now();
    let min_time = right
        .seeds
        .iter()
        .map(|s| s.plane.epoch_mid)
        .fold(f64::INFINITY, f64::min);

    let time_binner = UniformTimeBinner::new(min_time, 30.0 / 60.0 / 24.0); // 30 minutes
    buflog_timing!(logbuf, cli, "build time binner", t_timebin.elapsed());

    // Edge generation
    let t_edges = Instant::now();
    let edges = Edge::generate_topk_edges(
        EdgeId(0),
        &left.seeds,
        &right.seeds,
        &engine_cfg.edges,
        spatial_binner,
        &time_binner,
    );
    buflog_timing!(logbuf, cli, "generate_topk_edges", t_edges.elapsed());

    buflog!(
        logbuf,
        cli,
        "Edges: total={}, active={} (left seeds={}, right seeds={})",
        edges.len(),
        edges.iter().filter(|e| e.active).count(),
        left.seeds.len(),
        right.seeds.len()
    );

    // Optional detailed view
    buflog2!(
        logbuf,
        cli,
        "{}",
        EdgesStatsDisplay::new(&edges).top_k(20).only_active(false)
    );

    (time_binner, edges)
}

/// Step 2: run `edge_truth_diagnostics` and log the summary.
fn step_edge_truth_diagnostics<'a>(
    logbuf: &mut String,
    cli: &Cli,
    left: &'a fink_fat_eval::night_seeds::NightSeeds,
    right: &'a fink_fat_eval::night_seeds::NightSeeds,
    edges: &[Edge<'a>],
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
    time_binner: &UniformTimeBinner,
) -> fink_fat_eval::scoring::edges_diagnostics::EdgeTruthSummary {
    let t_diag = Instant::now();
    let diag = edge_truth_diagnostics(
        left,
        right,
        edges,
        &engine_cfg.edges,
        spatial_binner,
        time_binner,
    );
    buflog_timing!(logbuf, cli, "edge_truth_diagnostics", t_diag.elapsed());

    buflog!(
        logbuf,
        cli,
        "Generated true/false edges: {} / {}",
        diag.true_edges_generated,
        diag.false_edges_generated
    );
    buflog!(
        logbuf,
        cli,
        "True edges possible (oracle): {}",
        diag.true_edges_possible
    );
    buflog!(
        logbuf,
        cli,
        "Missed by top-k: {}, missed by prefilter: {}",
        diag.true_edges_missed_top_k,
        diag.true_edges_missed_generation
    );
    buflog!(logbuf, cli, "Recall (diag): {:.4}", diag.recall());

    diag
}

/// Step 3: run `diagnose_missed_true_edges` and log the (buffered) miss report summary.
/// Returns `(counts, details_len)` so you can later format/print if desired.
fn step_diagnose_missed_true_edges(
    logbuf: &mut String,
    cli: &Cli,
    left: &fink_fat_eval::night_seeds::NightSeeds,
    right: &fink_fat_eval::night_seeds::NightSeeds,
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
    time_binner: &UniformTimeBinner,
    gap_days: usize,
) -> (AHashMap<MissReason, usize>, usize) {
    let t_miss = Instant::now();
    let (counts, details) = diagnose_missed_true_edges(
        left,
        right,
        &engine_cfg.edges,
        spatial_binner,
        time_binner,
        gap_days.max(1) as u32,
    );
    buflog_timing!(logbuf, cli, "diagnose_missed_true_edges", t_miss.elapsed());

    if !cli.quiet && cli.verbose >= 2 {
        // Still not printing details here (stdout) to keep parallel logs clean.
        buflog!(
            logbuf,
            cli,
            "Miss report: total_missed={} (use -vv and/or add buffered formatter for details)",
            details.len()
        );
    } else {
        buflog!(
            logbuf,
            cli,
            "Miss report: total_missed={} (use -vv to print details)",
            details.len()
        );
    }

    (counts, details.len())
}

/// Controls which evaluation steps are executed.
#[derive(Debug, Clone, Copy)]
struct EvalSteps {
    pub diagnostics: bool,
    pub miss_diagnosis: bool,
}

impl EvalSteps {
    fn all() -> Self {
        Self {
            diagnostics: true,
            miss_diagnosis: true,
        }
    }
}

fn eval_pair_with_log(
    logbuf: &mut String,
    cli: &Cli,
    left: &fink_fat_eval::night_seeds::NightSeeds,
    right: &fink_fat_eval::night_seeds::NightSeeds,
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
    gap_days: usize,
    steps: EvalSteps,
) -> Result<PairSummary> {
    buflog_section!(
        logbuf,
        cli,
        &format!(
            "Pair {:?} -> {:?} (gap = {} night(s))",
            left.nid, right.nid, gap_days
        )
    );

    // 1) generate edges (+ time binner)
    let (time_binner, edges) =
        step_generate_topk_edges(logbuf, cli, left, right, engine_cfg, spatial_binner);

    // Truth counts (kept as-is: you already had it here)
    let t_truth = Instant::now();
    let (n_true, n_false, n_true_possible, recall) =
        edge_truth_counts_between_nights(left, right, &edges);
    buflog_timing!(
        logbuf,
        cli,
        "edge_truth_counts_between_nights",
        t_truth.elapsed()
    );

    buflog!(logbuf, cli, "true edges in produced set  : {}", n_true);
    buflog!(logbuf, cli, "false edges in produced set : {}", n_false);
    buflog!(
        logbuf,
        cli,
        "true edges possible (oracle): {}",
        n_true_possible
    );
    buflog!(logbuf, cli, "true-edge recall: {:.4}", recall);

    // 2) diagnostics summary
    let diag = if steps.diagnostics {
        Some(step_edge_truth_diagnostics(
            logbuf,
            cli,
            left,
            right,
            &edges,
            engine_cfg,
            spatial_binner,
            &time_binner,
        ))
    } else {
        buflog!(logbuf, cli, "edge_truth_diagnostics: skipped");
        None
    };

    // 3) missed edges diagnosis (kept for logging only)
    if steps.miss_diagnosis {
        let _ = step_diagnose_missed_true_edges(
            logbuf,
            cli,
            left,
            right,
            engine_cfg,
            spatial_binner,
            &time_binner,
            gap_days,
        );
    } else {
        buflog!(logbuf, cli, "diagnose_missed_true_edges: skipped");
    }

    Ok(PairSummary {
        n_true,
        n_false,
        n_true_possible,
        recall,
        missed_top_k: diag.as_ref().map_or(0, |d| d.true_edges_missed_top_k),
        missed_generation: diag.as_ref().map_or(0, |d| d.true_edges_missed_generation),
        true_generated: diag.as_ref().map_or(0, |d| d.true_edges_generated),
        false_generated: diag.as_ref().map_or(0, |d| d.false_edges_generated),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let t_total = Instant::now();

    log_section!(&cli, "Setup");

    let t_fs = Instant::now();
    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("create {}", cli.scan.out_dir))?;
    log_timing!(&cli, "create out_dir", t_fs.elapsed());

    let t_cfg = Instant::now();
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("load engine config {}", cli.engine_config))?;
    log_timing!(&cli, "load engine config", t_cfg.elapsed());

    log_section!(&cli, "Dataset ingest");

    let t_src = Instant::now();
    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;
    log_timing!(&cli, "open parquet source", t_src.elapsed());

    let ingest_cfg = AlertIngestConfig::default();

    let t_nights = Instant::now();
    let night_store: NightStore =
        collect_nights(&source, cli.scan.mode.into(), cli.scan.minimal, &ingest_cfg)?;
    log_timing!(&cli, "collect_nights", t_nights.elapsed());

    log_section!(&cli, "Seed generation");

    let t_seeds = Instant::now();
    let night_seeds =
        SeedStore::generate_nightseed_store(&night_store, &engine_cfg, 8, 1.0, false)?;
    log_timing!(&cli, "generate_nightseed_store", t_seeds.elapsed());

    log!(&cli, "SeedStore nights: {}", night_seeds.len());
    log2!(&cli, "{night_seeds}");

    log_section!(&cli, "Night pair selection");

    let t_resolve = Instant::now();
    let nids: Vec<u32> = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    log_timing!(&cli, "resolve_nids", t_resolve.elapsed());
    anyhow::ensure!(nids.len() >= 2, "need at least 2 nights to evaluate edges");

    let consecutive_window: usize = 1;
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
    log2!(&cli, "Pairs: {pairs:?}");

    let spatial_binner = HealpixBinner::new(cli.binning.healpix_depth);

    log_section!(&cli, "Evaluation loop");

    let mut by_gap: BTreeMap<usize, Agg> = BTreeMap::new();

    let nb_pairs = pairs.len();

    #[derive(Debug)]
    struct PairOut {
        idx: usize,
        gap: usize,
        summary: Option<PairSummary>,
        log: String,
    }

    let pb = ProgressBar::new(nb_pairs as u64);
    pb.set_style(
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise}) {msg}")
        .unwrap()
        .progress_chars("=>-"),
);
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb.set_message("starting…");

    // 1) Compute in parallel (no stdout).
    let outs: Result<Vec<PairOut>> = pairs
        .par_iter()
        .enumerate()
        .map(|(idx, (left_nid, right_nid, gap))| -> Result<PairOut> {
            let mut logbuf = String::new();
            pb.set_message(format!("{:?} -> {:?} (gap={})", left_nid, right_nid, gap));

            buflog!(
                &mut logbuf,
                &cli,
                "[{}/{}] evaluating {:?} -> {:?} (gap={})",
                idx + 1,
                nb_pairs,
                left_nid,
                right_nid,
                gap
            );

            let Some(left) = night_seeds.get(left_nid) else {
                buflog!(
                    &mut logbuf,
                    &cli,
                    "skip {:?}: not present in SeedStore",
                    left_nid
                );
                pb.inc(1);
                return Ok(PairOut {
                    idx,
                    gap: *gap,
                    summary: None,
                    log: logbuf,
                });
            };

            let Some(right) = night_seeds.get(right_nid) else {
                buflog!(
                    &mut logbuf,
                    &cli,
                    "skip {:?}: not present in SeedStore",
                    right_nid
                );
                pb.inc(1);
                return Ok(PairOut {
                    idx,
                    gap: *gap,
                    summary: None,
                    log: logbuf,
                });
            };

            let steps = EvalSteps {
                diagnostics: false,
                miss_diagnosis: false,
            };

            let t_pair = Instant::now();
            let summary = eval_pair_with_log(
                &mut logbuf,
                &cli,
                left,
                right,
                &engine_cfg,
                &spatial_binner,
                *gap,
                steps,
            )?;
            buflog_timing!(&mut logbuf, &cli, "pair total", t_pair.elapsed());

            // nice separator (buffered)
            if !cli.quiet {
                logbuf.push_str(
                    "\n ____________________________________________________________\n\n",
                );
            }

            pb.inc(1);
            Ok(PairOut {
                idx,
                gap: *gap,
                summary: Some(summary),
                log: logbuf,
            })
        })
        .collect();

    // 2) Flush logs in original order + aggregate sequentially.
    let mut outs = outs?;
    pb.finish_with_message("done");

    outs.sort_by_key(|o| o.idx);

    for o in outs {
        if !cli.quiet {
            print!("{}", o.log);
        }
        if let Some(summary) = o.summary {
            by_gap.entry(o.gap).or_default().push(summary);
        }
    }

    log_section!(&cli, "Final report");

    for (gap, agg) in by_gap {
        log!(
            &cli,
            "Gap={} night(s): pairs={}, sum_true={}, sum_false={}, sum_true_possible={}, mean_recall={:.4}, weighted_recall={:.4}, missed_top_k={}, missed_generation={}",
            gap,
            agg.n_pairs,
            agg.sum_true,
            agg.sum_false,
            agg.sum_true_possible,
            agg.mean_recall(),
            agg.weighted_recall(),
            agg.sum_missed_top_k,
            agg.sum_missed_generation,
        );
    }

    log_timing!(&cli, "TOTAL", t_total.elapsed());
    Ok(())
}

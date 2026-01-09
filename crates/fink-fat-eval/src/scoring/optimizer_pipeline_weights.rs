//! Pipeline implementation for the `score-optimize-weights` binary.
//!
//! This module keeps `src/bin/score_optimize_weights.rs` tiny and readable.

use std::fs;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use clap::{ArgAction, Parser, command};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::score::ScoredEdge,
    night_id::NightId,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use crate::{
    bin_utils::{fmt_ms, infer_t0_mjd_tt, ingest_one_night, resolve_nids},
    cli::common::{CommonBinningArgs, CommonScanArgs},
    dataset::{
        ParquetSource,
        ingest_config::AlertIngestConfig,
        ztf_alerts::{AlertLoadMode, AlertStoreWithTruth},
    },
    scoring::{
        optim_writer::write_best_scoring_yamls,
        score_optim::{
            EdgeSample, SampleStats, balance_same_diff, random_search_best_auc, subsample_in_place,
        },
    },
    seeding::seed_gen::generate_pairs_and_triplets,
};

/// Return the truth trajectory_id for an alert id.
#[inline]
fn alert_truth_id(store: &AlertStoreWithTruth, alert_id: fink_fat_engine::AlertId) -> i64 {
    store.truth_for(alert_id) as i64
}

/// Compute a "seed truth id": returns Some(traj_id) iff all members share the same traj_id > 0.
fn seed_truth_id(
    store: &AlertStoreWithTruth,
    seed: &fink_fat_engine::seeding::seed_node::SeedNode,
) -> Option<i64> {
    let mut t0: Option<i64> = None;
    for &mid in &seed.members {
        let tid = alert_truth_id(store, mid);
        if tid <= 0 {
            return None;
        }
        match t0 {
            None => t0 = Some(tid),
            Some(x) if x == tid => {}
            Some(_) => return None,
        }
    }
    t0
}

/// Build a "feature extraction" ScoreConfig that forces all diagnostic components to be computed.
fn force_feature_weights(
    mut cfg: fink_fat_engine::engine_config::score_config::ScoreConfig,
) -> fink_fat_engine::engine_config::score_config::ScoreConfig {
    cfg.position.w_pos = 1.0;
    cfg.velocity.w_dir = 1.0;
    cfg.velocity.w_norm = 1.0;
    cfg.photometry.w_flux = 1.0;
    cfg.gap.w_gap = 1.0;
    cfg.band.w_band_mismatch = 1.0;
    cfg
}

struct NightSeeds {
    nid: i32,
    seeds: Vec<fink_fat_engine::seeding::seed_node::SeedNode>,
    truth: Vec<Option<i64>>,
}

/// Load one night: ingest -> seed (pairs + optional triplets) -> subsample -> compute seed truth.
fn load_night_seeds(
    source: &ParquetSource,
    ingest_cfg: &AlertIngestConfig,
    engine_cfg: &EngineConfig,
    nid: i32,
    mode: AlertLoadMode,
    minimal: bool,
    healpix_depth: u8,
    time_bin_days: f64,
    pairs_only: bool,
    max_seeds_per_night: usize,
    rng: &mut StdRng,
) -> Result<NightSeeds> {
    let store = ingest_one_night(source, nid, mode, minimal, ingest_cfg)?;

    let n_alerts = store.store.alerts.len();
    eprintln!("  alerts: {}", n_alerts);

    let t0 = infer_t0_mjd_tt(&store);
    let spatial = HealpixBinner::new(healpix_depth);
    let time = UniformTimeBinner::new(time_bin_days, t0);
    let night_id = NightId::new(nid as u32);

    let t_gen = std::time::Instant::now();
    let out = generate_pairs_and_triplets(
        &store,
        night_id,
        &spatial,
        &time,
        &engine_cfg.pairs,
        &engine_cfg.triplets,
        None,
    )?;

    eprintln!(
        "  seeding: bucket={:.3}ms pairs={:.3}ms pair_feat={:.3}ms triplets={:.3}ms trip_feat={:.3}ms",
        fmt_ms(out.timings.bucket_index),
        fmt_ms(out.timings.pairs),
        fmt_ms(out.timings.pair_features),
        fmt_ms(out.timings.triplets),
        fmt_ms(out.timings.triplet_features),
    );
    eprintln!(
        "  seeds: pairs={} triplets={} (elapsed {:.3} ms)",
        out.pair_seeds.len(),
        out.triplet_seeds.len(),
        fmt_ms(t_gen.elapsed())
    );

    let mut seeds = out.pair_seeds;
    if !pairs_only {
        seeds.extend(out.triplet_seeds);
    }

    subsample_in_place(&mut seeds, max_seeds_per_night, rng);
    let truth: Vec<Option<i64>> = seeds.iter().map(|s| seed_truth_id(&store, s)).collect();

    Ok(NightSeeds {
        nid,
        seeds,
        truth,
    })
}

/// Sample edges between consecutive nights, filling same/diff buckets.
fn sample_edges_between(
    a: &NightSeeds,
    b: &NightSeeds,
    score_cfg_feat: &fink_fat_engine::engine_config::score_config::ScoreConfig,
    only_truth: bool,
    sample_right_per_left: usize,
    max_edges_per_pair: usize,
    rng: &mut StdRng,
    edges_same: &mut Vec<EdgeSample>,
    edges_diff: &mut Vec<EdgeSample>,
) {
    eprintln!(
        "\nSampling edges: nid {} -> {} ({} x {})",
        a.nid,
        b.nid,
        a.seeds.len(),
        b.seeds.len()
    );

    if a.seeds.is_empty() || b.seeds.is_empty() {
        return;
    }

    let mut right_indices: Vec<usize> = (0..b.seeds.len()).collect();
    let mut n_considered_for_pair: usize = 0;

    for (ia, si) in a.seeds.iter().enumerate() {
        let ti = a.truth[ia];

        right_indices.shuffle(rng);
        let take = sample_right_per_left.min(right_indices.len());

        for &jb in right_indices.iter().take(take) {
            let sj = &b.seeds[jb];
            let tj = b.truth[jb];

            if only_truth && (ti.is_none() || tj.is_none()) {
                continue;
            }

            let Some(edge) = ScoredEdge::score(si, sj, score_cfg_feat, 1) else {
                continue;
            };

            let same = match (ti, tj) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            };

            let sample = EdgeSample {
                same,
                d2_pos: edge.components.d2_pos,
                vel_angle_rad: edge.components.vel_angle_rad,
                vel_speed_diff: edge.components.vel_speed_diff,
                z_flux: edge.components.z_flux,
                gap_penalty: edge.components.gap_penalty,
                band_mismatch: edge.components.band_mismatch,
            };

            if same {
                edges_same.push(sample);
            } else {
                edges_diff.push(sample);
            }

            n_considered_for_pair += 1;
            if n_considered_for_pair >= max_edges_per_pair {
                break;
            }
        }

        if n_considered_for_pair >= max_edges_per_pair {
            break;
        }
    }

    eprintln!(
        "  totals so far: same={} diff={}",
        edges_same.len(),
        edges_diff.len(),
    );
}

#[derive(Parser, Debug)]
#[command(
    name = "score-optimize-weights",
    version,
    about = "Optimize inter-night scoring weights/normalizations to separate same vs different asteroid edges.",
    disable_help_subcommand = true,
    verbatim_doc_comment
)]
pub struct Cli {
    #[command(flatten)]
    scan: CommonScanArgs,

    #[command(flatten)]
    binning: CommonBinningArgs,

    /// Engine configuration YAML (pairs/triplets/predictor/scoring/edges).
    #[arg(
        long,
        value_name = "YAML",
        help_heading = "Config",
        default_value = "engine.yaml"
    )]
    engine_config: Utf8PathBuf,

    /// Optional explicit list of night ids to process (comma-separated).
    #[arg(long, value_name = "NID1,NID2,...", help_heading = "Scan")]
    nids: Option<String>,

    /// Limit number of nights processed (after sorting / selection).
    #[arg(long, value_name = "N", help_heading = "Scan")]
    max_nights: Option<usize>,

    /// Disable triplet generation (pairs-only).
    #[arg(long, default_value_t = false, help_heading = "Seeding")]
    pairs_only: bool,

    /// Keep only edges where both seeds have a valid truth id (`trajectory_id > 0`)
    /// and each seed is "pure" (all members share the same traj_id).
    #[arg(long, default_value_t = true, action = ArgAction::Set, help_heading = "Truth")]
    only_truth: bool,

    /// Maximum number of seeds kept per night (random subsample if exceeded).
    #[arg(long, default_value_t = 8000, help_heading = "Sampling")]
    max_seeds_per_night: usize,

    /// For each left seed, sample this many right seeds (candidate evaluation budget).
    #[arg(long, default_value_t = 512, help_heading = "Sampling")]
    sample_right_per_left: usize,

    /// Optional cap on total edges sampled per night-pair (after filtering).
    #[arg(long, default_value_t = 2_000_000, help_heading = "Sampling")]
    max_edges_per_pair: usize,

    /// Random seed for reproducibility.
    #[arg(long, default_value_t = 42, help_heading = "Sampling")]
    rng_seed: u64,

    /// Optimization budget (# random trials).
    #[arg(long, default_value_t = 2000, help_heading = "Optimize")]
    budget: usize,
}

/// Run the full optimizer pipeline.
pub fn run(cli: &Cli) -> Result<()> {
    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("failed to create out dir: {}", cli.scan.out_dir))?;

    let mut rng = StdRng::seed_from_u64(cli.rng_seed);

    // Engine config
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("failed to load engine config: {}", cli.engine_config))?;

    let score_cfg_base = engine_cfg.scoring.clone();
    let score_cfg_feat = force_feature_weights(score_cfg_base.clone());

    // Dataset source
    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;

    let nids = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    eprintln!("Processing {} night(s).", nids.len());

    let ingest_cfg = AlertIngestConfig::default();

    // Load per-night seeds
    let mut nights: Vec<NightSeeds> = Vec::with_capacity(nids.len());
    for (k, &nid) in nids.iter().enumerate() {
        eprintln!("\n[{}/{}] nid={}", k + 1, nids.len(), nid);
        let night = load_night_seeds(
            &source,
            &ingest_cfg,
            &engine_cfg,
            nid,
            cli.scan.mode.into(),
            cli.scan.minimal,
            cli.binning.healpix_depth,
            cli.binning.time_bin_days,
            cli.pairs_only,
            cli.max_seeds_per_night,
            &mut rng,
        )?;
        nights.push(night);
    }

    // Sample edges between consecutive nights
    let mut edges_same: Vec<EdgeSample> = Vec::new();
    let mut edges_diff: Vec<EdgeSample> = Vec::new();

    for w in nights.windows(2) {
        sample_edges_between(
            &w[0],
            &w[1],
            &score_cfg_feat,
            cli.only_truth,
            cli.sample_right_per_left,
            cli.max_edges_per_pair,
            &mut rng,
            &mut edges_same,
            &mut edges_diff,
        );
    }

    anyhow::ensure!(
        !edges_same.is_empty() && !edges_diff.is_empty(),
        "degenerate sample before balancing: same={} diff={}",
        edges_same.len(),
        edges_diff.len()
    );

    let n_bal = balance_same_diff(&mut edges_same, &mut edges_diff, &mut rng);
    anyhow::ensure!(n_bal > 0, "balanced sample is empty");

    let mut edges: Vec<EdgeSample> = Vec::with_capacity(2 * n_bal);
    edges.extend(edges_same.into_iter());
    edges.extend(edges_diff.into_iter());

    let stats = SampleStats {
        same: n_bal,
        diff: n_bal,
    };

    eprintln!(
        "Edge sample (balanced): same={} diff={} (total={})",
        stats.same,
        stats.diff,
        edges.len()
    );

    // Optimize
    let res = random_search_best_auc(&edges, &stats, cli.budget, &mut rng);

    eprintln!("\nBest AUC: {:.6}", res.best_auc);
    eprintln!(
        "Best params: w_pos={:.4} w_dir={:.4} w_norm={:.4} w_flux={:.4} w_gap={:.4} w_band={:.4} theta0={:.6}rad v0={:.6}rad/day",
        res.best.w_pos,
        res.best.w_dir,
        res.best.w_norm,
        res.best.w_flux,
        res.best.w_gap,
        res.best.w_band,
        res.best.theta0,
        res.best.v0
    );

    // Write YAML outputs
    let (patch_path, full_path) =
        write_best_scoring_yamls(&cli.scan.out_dir, &engine_cfg, &score_cfg_base, &res.best)?;

    eprintln!("\nWrote:");
    eprintln!("  {}", patch_path);
    eprintln!("  {}", full_path);

    Ok(())
}

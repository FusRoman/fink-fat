//! Pipeline implementation for the `inspect_scoring_components` binary.
//!
//! This module separates the heavy lifting (dataset ingestion, seeding,
//! edge sampling and histogram accumulation) from the thin binary in
//! `inspect_scoring_components.rs`.  It exposes a `Cli` struct for
//! command-line parsing via `clap` and a `run` function that executes
//! the full inspection pipeline.  Plotting routines are delegated to
//! `inspect_scoring_components_plots` to further decouple concerns.

use std::fs;

use anyhow::{Context, Result};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use fink_fat_engine::{
    AlertId,
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::score::ScoredEdge,
    night_id::NightId,
    seeding::seed_node::SeedNode,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use crate::{
    bin_utils::{infer_t0_mjd_tt, ingest_one_night, resolve_nids}, cli::scoring::Cli, dataset::{
        ParquetSource,
        ingest_config::AlertIngestConfig,
        ztf_alerts::{AlertLoadMode, AlertStoreWithTruth},
    }, scoring::inspect_scoring_components_plots::{frac_none, plot_hist, plot_overlay_hist}, seeding::seed_gen::generate_pairs_and_triplets
};

/// Lightweight container for per-night seeds and their associated truth labels.
struct NightSeeds {
    nid: NightId,
    seeds: Vec<SeedNode>,
    truth: Vec<Option<i64>>,
}

/// Return the truth trajectory_id for an alert id.
#[inline]
fn alert_truth_id(store: &AlertStoreWithTruth, alert_id: AlertId) -> i64 {
    store.truth_for(alert_id) as i64
}

/// Compute a "seed truth id": returns Some(traj_id) iff all members share the same traj_id > 0.
fn seed_truth_id(store: &AlertStoreWithTruth, seed: &SeedNode) -> Option<i64> {
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

/// Load one night: ingest -> seed (pairs + optional triplets) -> subsample -> compute seed truth.
fn load_night_seeds(
    source: &ParquetSource,
    ingest_cfg: &AlertIngestConfig,
    engine_cfg: &EngineConfig,
    nid: NightId,
    mode: AlertLoadMode,
    minimal: bool,
    healpix_depth: u8,
    time_bin_days: f64,
    pairs_only: bool,
    max_seeds_per_night: usize,
    rng: &mut StdRng,
) -> Result<(AlertStoreWithTruth, NightSeeds)> {
    let store = ingest_one_night(source, nid, mode, minimal, ingest_cfg)?;
    let t0 = infer_t0_mjd_tt(&store);
    let spatial = HealpixBinner::new(healpix_depth);
    let time = UniformTimeBinner::new(time_bin_days, t0);
    let out = generate_pairs_and_triplets(
        &store,
        nid,
        &spatial,
        &time,
        &engine_cfg.pairs,
        &engine_cfg.triplets,
        None,
    )?;
    let mut seeds = out.pair_seeds;
    if !pairs_only {
        seeds.extend(out.triplet_seeds);
    }
    // Subsample to keep the inspection budget bounded.
    if seeds.len() > max_seeds_per_night {
        seeds.shuffle(rng);
        seeds.truncate(max_seeds_per_night);
    }
    let truth: Vec<Option<i64>> = seeds.iter().map(|s| seed_truth_id(&store, s)).collect();
    Ok((store, NightSeeds { nid, seeds, truth }))
}

/// Sample directed edges from A->B (seed cross-night) without full N² explosion.
fn sample_scored_edges_between(
    a: &NightSeeds,
    b: &NightSeeds,
    score_cfg: &fink_fat_engine::engine_config::score_config::ScoreConfig,
    only_truth: bool,
    sample_right_per_left: usize,
    max_edges_per_pair: usize,
    rng: &mut StdRng,
) -> Vec<ScoredEdge> {
    let mut out = Vec::with_capacity(max_edges_per_pair.min(1024));
    if a.seeds.is_empty() || b.seeds.is_empty() {
        return out;
    }
    let mut right_indices: Vec<usize> = (0..b.seeds.len()).collect();
    'outer: for (ia, si) in a.seeds.iter().enumerate() {
        let ti = a.truth[ia];
        right_indices.shuffle(rng);
        let take = sample_right_per_left.min(right_indices.len());
        for &jb in right_indices.iter().take(take) {
            let sj = &b.seeds[jb];
            let tj = b.truth[jb];
            if only_truth && (ti.is_none() || tj.is_none()) {
                continue;
            }
            if let Some(e) = ScoredEdge::score(si, sj, score_cfg, /*delta*/ 1) {
                out.push(e);
                if out.len() >= max_edges_per_pair {
                    break 'outer;
                }
            }
        }
    }
    out
}

/// Accumulate per-seed QA metrics into the provided vectors.
fn accumulate_seed_qa(
    seeds: &[SeedNode],
    pos_norm: &mut Vec<f64>,
    vel_norm: &mut Vec<f64>,
    acc_norm: &mut Vec<f64>,
    acc_none: &mut usize,
    cov_pos_x: &mut Vec<f64>,
    cov_pos_y: &mut Vec<f64>,
    cov_vel_x: &mut Vec<f64>,
    cov_vel_y: &mut Vec<f64>,
) {
    for s in seeds.iter() {
        let px = s.plane.pos_xy[0];
        let py = s.plane.pos_xy[1];
        pos_norm.push((px * px + py * py).sqrt());
        let vx = s.plane.vel_xy[0];
        let vy = s.plane.vel_xy[1];
        vel_norm.push((vx * vx + vy * vy).sqrt());
        match s.plane.acc_xy {
            Some(a) => {
                let ax = a[0];
                let ay = a[1];
                acc_norm.push((ax * ax + ay * ay).sqrt());
            }
            None => *acc_none += 1,
        }
        cov_pos_x.push(s.plane.cov_pos[0][0]);
        cov_pos_y.push(s.plane.cov_pos[1][1]);
        cov_vel_x.push(s.plane.cov_vel[0][0]);
        cov_vel_y.push(s.plane.cov_vel[1][1]);
    }
}

/// Execute the inspection pipeline.
pub fn run(cli: &Cli) -> Result<()> {
    // Ensure output directory exists
    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("create {}", cli.scan.out_dir))?;
    // Initialize RNG
    let mut rng = StdRng::seed_from_u64(cli.rng_seed);
    // Load engine configuration
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("load engine config {}", cli.engine_config))?;
    // Dataset source
    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;
    let ingest_cfg = AlertIngestConfig::default();
    // Determine night IDs
    let nids = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    anyhow::ensure!(
        nids.len() >= 2,
        "need at least 2 nights to inspect inter-night edges"
    );
    eprintln!("Processing {} night(s).", nids.len());
    // Edge component accumulators (same/diff)
    let mut d2_same = Vec::new();
    let mut d2_diff = Vec::new();
    let mut ang_same = Vec::new();
    let mut ang_diff = Vec::new();
    let mut ang_none_same = 0usize;
    let mut ang_none_diff = 0usize;
    let mut dv_same = Vec::new();
    let mut dv_diff = Vec::new();
    let mut dv_none_same = 0usize;
    let mut dv_none_diff = 0usize;
    let mut z_same = Vec::new();
    let mut z_diff = Vec::new();
    let mut z_none_same = 0usize;
    let mut z_none_diff = 0usize;
    let mut gap_same = Vec::new();
    let mut gap_diff = Vec::new();
    // Seed QA accumulators
    let mut seed_pos_norm = Vec::new();
    let mut seed_vel_norm = Vec::new();
    let mut seed_acc_norm = Vec::new();
    let mut seed_acc_none = 0usize;
    let mut seed_cov_pos_x = Vec::new();
    let mut seed_cov_pos_y = Vec::new();
    let mut seed_cov_vel_x = Vec::new();
    let mut seed_cov_vel_y = Vec::new();
    // Load first night and accumulate seed QA
    let (_, mut night_a) = load_night_seeds(
        &source,
        &ingest_cfg,
        &engine_cfg,
        NightId(nids[0] as u32),
        cli.scan.mode.into(),
        cli.scan.minimal,
        cli.binning.healpix_depth,
        cli.binning.time_bin_days,
        cli.pairs_only,
        cli.max_seeds_per_night,
        &mut rng,
    )?;
    accumulate_seed_qa(
        &night_a.seeds,
        &mut seed_pos_norm,
        &mut seed_vel_norm,
        &mut seed_acc_norm,
        &mut seed_acc_none,
        &mut seed_cov_pos_x,
        &mut seed_cov_pos_y,
        &mut seed_cov_vel_x,
        &mut seed_cov_vel_y,
    );
    // Iterate over subsequent nights
    for &nid_b in nids.iter().skip(1) {
        eprintln!("== Night pair: {} -> {}", night_a.nid, nid_b);
        let (_, night_b) = load_night_seeds(
            &source,
            &ingest_cfg,
            &engine_cfg,
            NightId(nid_b as u32),
            cli.scan.mode.into(),
            cli.scan.minimal,
            cli.binning.healpix_depth,
            cli.binning.time_bin_days,
            cli.pairs_only,
            cli.max_seeds_per_night,
            &mut rng,
        )?;
        accumulate_seed_qa(
            &night_b.seeds,
            &mut seed_pos_norm,
            &mut seed_vel_norm,
            &mut seed_acc_norm,
            &mut seed_acc_none,
            &mut seed_cov_pos_x,
            &mut seed_cov_pos_y,
            &mut seed_cov_vel_x,
            &mut seed_cov_vel_y,
        );
        // Sample edges A->B
        let edges = sample_scored_edges_between(
            &night_a,
            &night_b,
            &engine_cfg.edges.score_config,
            cli.only_truth,
            cli.sample_right_per_left,
            cli.max_edges_per_pair,
            &mut rng,
        );
        for e in edges.iter() {
            let ia = e.from.idx() as usize;
            let ib = e.to.idx() as usize;
            let ta = night_a.truth.get(ia).copied().unwrap_or(None);
            let tb = night_b.truth.get(ib).copied().unwrap_or(None);
            let is_same = match (ta, tb) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            };
            if is_same {
                d2_same.push(e.components.d2_pos);
            } else {
                d2_diff.push(e.components.d2_pos);
            }
            match e.components.vel_angle_rad {
                Some(v) => {
                    if is_same {
                        ang_same.push(v);
                    } else {
                        ang_diff.push(v);
                    }
                }
                None => {
                    if is_same {
                        ang_none_same += 1;
                    } else {
                        ang_none_diff += 1;
                    }
                }
            }
            match e.components.vel_speed_diff {
                Some(v) => {
                    if is_same {
                        dv_same.push(v);
                    } else {
                        dv_diff.push(v);
                    }
                }
                None => {
                    if is_same {
                        dv_none_same += 1;
                    } else {
                        dv_none_diff += 1;
                    }
                }
            }
            match e.components.z_flux {
                Some(v) => {
                    if is_same {
                        z_same.push(v);
                    } else {
                        z_diff.push(v);
                    }
                }
                None => {
                    if is_same {
                        z_none_same += 1;
                    } else {
                        z_none_diff += 1;
                    }
                }
            }
            if is_same {
                gap_same.push(e.components.gap_penalty);
            } else {
                gap_diff.push(e.components.gap_penalty);
            }
        }
        eprintln!(
            "  edges: sampled={} | d2_same={} d2_diff={}",
            edges.len(),
            d2_same.len(),
            d2_diff.len()
        );
        // Slide window: B becomes next A
        night_a = night_b;
    }
    // Plot score components
    plot_overlay_hist(
        &cli.scan.out_dir.join("components_d2_pos.png"),
        "Score component: d2_pos (Mahalanobis^2)",
        "d2_pos",
        &d2_same,
        &d2_diff,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_overlay_hist(
        &cli.scan.out_dir.join("components_vel_angle_rad.png"),
        format!(
            "Score component: vel_angle_rad (None same={:.3}, diff={:.3})",
            frac_none(ang_none_same, ang_same.len()),
            frac_none(ang_none_diff, ang_diff.len())
        ),
        "vel_angle_rad [rad]",
        &ang_same,
        &ang_diff,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_overlay_hist(
        &cli.scan.out_dir.join("components_vel_speed_diff.png"),
        format!(
            "Score component: vel_speed_diff (None same={:.3}, diff={:.3})",
            frac_none(dv_none_same, dv_same.len()),
            frac_none(dv_none_diff, dv_diff.len())
        ),
        "vel_speed_diff [rad/day]",
        &dv_same,
        &dv_diff,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_overlay_hist(
        &cli.scan.out_dir.join("components_z_flux.png"),
        format!(
            "Score component: z_flux (None same={:.3}, diff={:.3})",
            frac_none(z_none_same, z_same.len()),
            frac_none(z_none_diff, z_diff.len())
        ),
        "z_flux",
        &z_same,
        &z_diff,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_overlay_hist(
        &cli.scan.out_dir.join("components_gap_penalty.png"),
        "Score component: gap_penalty",
        "gap_penalty",
        &gap_same,
        &gap_diff,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    // Plot seed QA metrics
    plot_hist(
        &cli.scan.out_dir.join("seed_pos_norm.png"),
        "Seed QA: ||pos_xy|| on tangent plane",
        "||pos_xy|| [rad]",
        &seed_pos_norm,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_hist(
        &cli.scan.out_dir.join("seed_vel_norm.png"),
        "Seed QA: ||vel_xy|| on tangent plane",
        "||vel_xy|| [rad/day]",
        &seed_vel_norm,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_hist(
        &cli.scan.out_dir.join("seed_acc_norm.png"),
        format!(
            "Seed QA: ||acc_xy|| on tangent plane (None fraction={:.3})",
            frac_none(seed_acc_none, seed_acc_norm.len())
        ),
        "||acc_xy|| [rad/day^2]",
        &seed_acc_norm,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_hist(
        &cli.scan.out_dir.join("seed_cov_pos_x.png"),
        "Seed QA: cov_pos[0][0]",
        "cov_pos_x [rad^2]",
        &seed_cov_pos_x,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_hist(
        &cli.scan.out_dir.join("seed_cov_pos_y.png"),
        "Seed QA: cov_pos[1][1]",
        "cov_pos_y [rad^2]",
        &seed_cov_pos_y,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_hist(
        &cli.scan.out_dir.join("seed_cov_vel_x.png"),
        "Seed QA: cov_vel[0][0]",
        "cov_vel_x [rad^2/day^2]",
        &seed_cov_vel_x,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    plot_hist(
        &cli.scan.out_dir.join("seed_cov_vel_y.png"),
        "Seed QA: cov_vel[1][1]",
        "cov_vel_y [rad^2/day^2]",
        &seed_cov_vel_y,
        cli.bins,
        cli.clip_xmax,
        cli.log_y,
    )?;
    eprintln!("Done. PNGs written to {}", cli.scan.out_dir);
    Ok(())
}

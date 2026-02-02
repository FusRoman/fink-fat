use std::{fs, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::edge::{
        Edge,
        edge_id::EdgeId,
        features::{EDGE_FEATURE_KEYS, EdgeFeatureKey, EdgeFeatures},
    },
    night_id::NightId,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};
use fink_fat_eval::{
    bin_utils::resolve_nids,
    buflog, buflog_timing, buflog2,
    cli::scoring::{Cli, update_score_config},
    log, log_section, log_timing, log2,
    night_seeds::{NightSeeds, generate_seed_store},
    scoring::{edges_diagnostics::EdgesStatsDisplay, plotting::plot_two_histograms},
};
use indicatif::{ProgressBar, ProgressStyle};
use plotters::{coord::Shift, prelude::*};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

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

/// Convert resolved u32 night ids into typed NightId.
fn to_night_ids(nids: Vec<u32>) -> Vec<NightId> {
    nids.into_iter().map(NightId).collect()
}

/// Step 1: build the time binner (anchored on `right`) and generate Top-K edges.
fn step_generate_topk_edges<'a>(
    logbuf: &mut String,
    cli: &Cli,
    left: &'a NightSeeds,
    right: &'a NightSeeds,
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
) -> (UniformTimeBinner, Vec<Edge<'a>>) {
    let t_timebin = Instant::now();
    let min_time = right
        .seeds
        .iter()
        .map(|s| s.plane.epoch_mid)
        .fold(f64::INFINITY, f64::min);

    let time_binner = UniformTimeBinner::new(min_time, 30.0 / 60.0 / 24.0); // 30 minutes
    buflog_timing!(logbuf, cli, "build time binner", t_timebin.elapsed());

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

    buflog2!(
        logbuf,
        cli,
        "{}",
        EdgesStatsDisplay::new(&edges).top_k(20).only_active(false)
    );

    (time_binner, edges)
}

struct EdgeFeatureTruth {
    pub feature: EdgeFeatures,
    pub is_true_edge: bool,
}

/// A single plottable edge feature (key -> canonical name + getter).
#[derive(Clone, Copy)]
struct FeatureSpec {
    key: EdgeFeatureKey,
}

impl FeatureSpec {
    #[inline]
    fn name(self) -> &'static str {
        self.key.path()
    }

    #[inline]
    fn get(self, f: &EdgeFeatures) -> f64 {
        f.get(self.key)
    }
}

impl EdgeFeatureTruth {
    /// Return the full list of plottable features (stable canonical order).
    pub fn feature_specs() -> Vec<FeatureSpec> {
        EDGE_FEATURE_KEYS
            .into_iter()
            .map(|k| FeatureSpec { key: k })
            .collect()
    }
}

// -----------------------------------------------------------------------------
// Utils
// -----------------------------------------------------------------------------

fn sanitize_filename(s: &str) -> String {
    s.replace('.', "_").replace('/', "_").replace('\\', "_")
}

/// Signed log transform used for histograms:
/// f(x) = sign(x) * log10(1 + |x|)
#[inline]
fn signed_log10_1p(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let y = (1.0 + x.abs()).log10();
    if x.is_sign_negative() { -y } else { y }
}

fn min_max_from_points(points: &[(f64, f64)]) -> Option<(f64, f64, f64, f64)> {
    if points.is_empty() {
        return None;
    }
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    for &(x, y) in points {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }

    if xmin.is_finite() && xmax.is_finite() && ymin.is_finite() && ymax.is_finite() {
        Some((xmin, xmax, ymin, ymax))
    } else {
        None
    }
}

fn pad_range(minv: f64, maxv: f64, frac: f64) -> (f64, f64) {
    if !(minv.is_finite() && maxv.is_finite()) {
        return (0.0, 1.0);
    }
    if (maxv - minv).abs() <= f64::EPSILON {
        let d = if minv.abs() > 0.0 {
            minv.abs() * 0.1
        } else {
            1.0
        };
        return (minv - d, maxv + d);
    }
    let pad = (maxv - minv) * frac;
    (minv - pad, maxv + pad)
}

// -----------------------------------------------------------------------------
// 1D: log-histograms (already)
// -----------------------------------------------------------------------------

fn plot_feature_histogram(
    cli: &Cli,
    outs: &Vec<EdgeFeatureTruth>,
    spec: FeatureSpec,
) -> Result<()> {
    let name = spec.name();

    let mut same: Vec<f64> = Vec::new();
    let mut diff: Vec<f64> = Vec::new();

    for e in outs {
        let v = spec.get(&e.feature);
        if !v.is_finite() {
            continue;
        }
        let lv = signed_log10_1p(v);
        if !lv.is_finite() {
            continue;
        }
        if e.is_true_edge {
            same.push(lv);
        } else {
            diff.push(lv);
        }
    }

    let out_path = cli.scan.out_dir.join(format!(
        "edge_feature_histogram_log_{}.png",
        sanitize_filename(name)
    ));

    plot_two_histograms(
        &out_path,
        1024,
        768,
        &format!("Edge Feature Histogram (log): {}", name),
        "signed log10(1+|x|)",
        &same,
        &diff,
        200,
    )?;

    log!(cli, "Saved log-histogram for '{}' to {}", name, out_path);
    Ok(())
}

// -----------------------------------------------------------------------------
// 2D density plots: true, false, ratio
// -----------------------------------------------------------------------------

/// Build a regular 2D grid histogram over (x, y).
///
/// Returns a (nx * ny) array in row-major order: idx = iy * nx + ix.
fn grid2d_counts(
    points: &[(f64, f64)],
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; nx * ny];

    let dx = xmax - xmin;
    let dy = ymax - ymin;
    if dx <= 0.0 || dy <= 0.0 {
        return out;
    }

    for &(x, y) in points {
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        if x < xmin || x > xmax || y < ymin || y > ymax {
            continue;
        }
        let fx = ((x - xmin) / dx) * (nx as f64);
        let fy = ((y - ymin) / dy) * (ny as f64);

        let mut ix = fx.floor() as isize;
        let mut iy = fy.floor() as isize;

        // Clamp to valid range
        if ix < 0 {
            ix = 0;
        }
        if iy < 0 {
            iy = 0;
        }
        if ix >= nx as isize {
            ix = nx as isize - 1;
        }
        if iy >= ny as isize {
            iy = ny as isize - 1;
        }

        out[(iy as usize) * nx + (ix as usize)] += 1.0;
    }

    out
}

/// Simple grayscale colormap for Plotters.
fn gray(v01: f64) -> RGBColor {
    let t = v01.clamp(0.0, 1.0);
    let g = (255.0 * t) as u8;
    RGBColor(g, g, g)
}

/// Diverging (blue-white-red) colormap for ratio plots.
/// v01 in [0,1] where 0=blue, 0.5=white, 1=red
fn diverging_bwr(v01: f64) -> RGBColor {
    let t = v01.clamp(0.0, 1.0);
    if t <= 0.5 {
        // blue -> white
        let u = t / 0.5;
        let r = (255.0 * u) as u8;
        let g = (255.0 * u) as u8;
        let b = 255u8;
        RGBColor(r, g, b)
    } else {
        // white -> red
        let u = (t - 0.5) / 0.5;
        let r = 255u8;
        let g = (255.0 * (1.0 - u)) as u8;
        let b = (255.0 * (1.0 - u)) as u8;
        RGBColor(r, g, b)
    }
}

/// Draw a grid as a raster heatmap (each cell is a filled rectangle).
fn draw_grid_heatmap(
    root: &DrawingArea<BitMapBackend, Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    grid: &[f64],
    nx: usize,
    ny: usize,
    color_fn: fn(f64) -> RGBColor,
) -> Result<()> {
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 26))
        .margin(18)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .light_line_style(&WHITE.mix(0.10))
        .draw()?;

    // Normalize grid values to [0,1] for color mapping
    let mut gmin = f64::INFINITY;
    let mut gmax = f64::NEG_INFINITY;
    for &v in grid {
        if v.is_finite() {
            gmin = gmin.min(v);
            gmax = gmax.max(v);
        }
    }
    if !gmin.is_finite() || !gmax.is_finite() || (gmax - gmin).abs() <= f64::EPSILON {
        gmin = 0.0;
        gmax = 1.0;
    }

    let dx = (xmax - xmin) / (nx as f64);
    let dy = (ymax - ymin) / (ny as f64);

    // Render cells. Note: (ix, iy) refers to bin in x/y.
    chart.draw_series((0..ny).flat_map(|iy| {
        (0..nx).map(move |ix| {
            let v = grid[iy * nx + ix];
            let v01 = (v - gmin) / (gmax - gmin);
            let color = color_fn(v01).filled();
            let x0 = xmin + (ix as f64) * dx;
            let x1 = x0 + dx;
            let y0 = ymin + (iy as f64) * dy;
            let y1 = y0 + dy;
            Rectangle::new([(x0, y0), (x1, y1)], color)
        })
    }))?;

    Ok(())
}

fn collect_points_2d(
    outs: &[EdgeFeatureTruth],
    x: FeatureSpec,
    y: FeatureSpec,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mut same = Vec::new();
    let mut diff = Vec::new();

    for e in outs {
        let xv = x.get(&e.feature);
        let yv = y.get(&e.feature);
        if !xv.is_finite() || !yv.is_finite() {
            continue;
        }
        if e.is_true_edge {
            same.push((xv, yv));
        } else {
            diff.push((xv, yv));
        }
    }
    (same, diff)
}

/// Plot 3 maps:
/// - density_true (log counts)
/// - density_false (log counts)
/// - log density ratio: log((p_true+eps)/(p_false+eps))
fn plot_density_triplet(
    cli: &Cli,
    outs: &[EdgeFeatureTruth],
    x: FeatureSpec,
    y: FeatureSpec,
) -> Result<()> {
    let x_name = x.name();
    let y_name = y.name();

    let (same_pts, diff_pts) = collect_points_2d(outs, x, y);

    if same_pts.is_empty() || diff_pts.is_empty() {
        return Ok(());
    }

    // Range from all points
    let mut all = Vec::with_capacity(same_pts.len() + diff_pts.len());
    all.extend_from_slice(&same_pts);
    all.extend_from_slice(&diff_pts);

    let (xmin0, xmax0, ymin0, ymax0) =
        min_max_from_points(&all).context("No finite points for 2D density")?;

    let (xmin, xmax) = pad_range(xmin0, xmax0, 0.02);
    let (ymin, ymax) = pad_range(ymin0, ymax0, 0.02);

    let nx = 140usize;
    let ny = 120usize;

    // Raw counts grids
    let g_true = grid2d_counts(&same_pts, xmin, xmax, ymin, ymax, nx, ny);
    let g_false = grid2d_counts(&diff_pts, xmin, xmax, ymin, ymax, nx, ny);

    // Convert to log-counts for nicer visuals
    let eps = 1e-6;
    let g_true_log: Vec<f64> = g_true.iter().map(|&c| (c + 1.0).ln()).collect();
    let g_false_log: Vec<f64> = g_false.iter().map(|&c| (c + 1.0).ln()).collect();

    // Ratio in log-space: ln((p_true+eps)/(p_false+eps))
    // Use normalized frequencies (counts / total) so the ratio is not dominated by class imbalance.
    let s_true = (same_pts.len() as f64).max(1.0);
    let s_false = (diff_pts.len() as f64).max(1.0);

    let mut ratio = vec![0.0f64; nx * ny];
    for i in 0..nx * ny {
        let p_t = g_true[i] / s_true;
        let p_f = g_false[i] / s_false;
        ratio[i] = ((p_t + eps) / (p_f + eps)).ln(); // positive => more true, negative => more false
    }

    // For ratio: clamp extreme values for nicer colors
    // We'll normalize to [-rmax, +rmax] then map to [0,1]
    let mut rmax = 0.0f64;
    for &v in &ratio {
        if v.is_finite() {
            rmax = rmax.max(v.abs());
        }
    }
    if rmax <= 0.0 {
        rmax = 1.0;
    }
    let ratio_clamped: Vec<f64> = ratio.iter().map(|&v| v.clamp(-rmax, rmax)).collect();

    // Write 3 pngs
    let base = format!(
        "edge_2d_density_{}_vs_{}",
        sanitize_filename(x_name),
        sanitize_filename(y_name)
    );

    let out_true = cli.scan.out_dir.join(format!("{base}_true.png"));
    let out_false = cli.scan.out_dir.join(format!("{base}_false.png"));
    let out_ratio = cli.scan.out_dir.join(format!("{base}_ratio.png"));

    // TRUE density
    {
        let root = BitMapBackend::new(out_true.as_std_path(), (1100, 850)).into_drawing_area();
        draw_grid_heatmap(
            &root,
            &format!("2D density (true): {} vs {}", y_name, x_name),
            x_name,
            y_name,
            xmin,
            xmax,
            ymin,
            ymax,
            &g_true_log,
            nx,
            ny,
            gray,
        )?;
        root.present()?;
    }

    // FALSE density
    {
        let root = BitMapBackend::new(out_false.as_std_path(), (1100, 850)).into_drawing_area();
        draw_grid_heatmap(
            &root,
            &format!("2D density (false): {} vs {}", y_name, x_name),
            x_name,
            y_name,
            xmin,
            xmax,
            ymin,
            ymax,
            &g_false_log,
            nx,
            ny,
            gray,
        )?;
        root.present()?;
    }

    // RATIO
    {
        // Map ratio in [-rmax, rmax] to [0,1] via v01 = 0.5 + 0.5*(v/rmax)
        let ratio01: Vec<f64> = ratio_clamped
            .iter()
            .map(|&v| 0.5 + 0.5 * (v / rmax))
            .collect();

        let root = BitMapBackend::new(out_ratio.as_std_path(), (1100, 850)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!(
                    "log density ratio: ln((p_true+eps)/(p_false+eps))  [{} vs {}]",
                    y_name, x_name
                ),
                ("sans-serif", 24),
            )
            .margin(18)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

        chart
            .configure_mesh()
            .x_desc(x_name)
            .y_desc(y_name)
            .light_line_style(&WHITE.mix(0.10))
            .draw()?;

        let dx = (xmax - xmin) / (nx as f64);
        let dy = (ymax - ymin) / (ny as f64);

        let ratio01_ref: &[f64] = &ratio01;

        chart.draw_series((0..ny).flat_map(|iy| {
            (0..nx).map(move |ix| {
                let v01 = ratio01_ref[iy * nx + ix];
                let color = diverging_bwr(v01).filled();
                let x0 = xmin + (ix as f64) * dx;
                let x1 = x0 + dx;
                let y0 = ymin + (iy as f64) * dy;
                let y1 = y0 + dy;
                Rectangle::new([(x0, y0), (x1, y1)], color)
            })
        }))?;

        // Add an annotation of scale
        // (No colorbar widget in plotters; keep it textual.)
        root.draw(&Text::new(
            format!("red=more true, blue=more false, clamp ±{:.2}", rmax),
            (20, 20),
            ("sans-serif", 18).into_font(),
        ))?;

        root.present()?;
    }

    log!(
        cli,
        "Saved 2D density triplet for '{}' vs '{}' to {}, {}, {}",
        x_name,
        y_name,
        out_true,
        out_false,
        out_ratio
    );

    Ok(())
}

// -----------------------------------------------------------------------------
// Score plot (1D) based on a pair (x,y): score = x^2 + y^2
// -----------------------------------------------------------------------------

fn plot_pair_score_histogram(
    cli: &Cli,
    outs: &[EdgeFeatureTruth],
    x: FeatureSpec,
    y: FeatureSpec,
) -> Result<()> {
    let x_name = x.name();
    let y_name = y.name();

    let mut same = Vec::new();
    let mut diff = Vec::new();

    for e in outs {
        let xv = x.get(&e.feature);
        let yv = y.get(&e.feature);
        if !xv.is_finite() || !yv.is_finite() {
            continue;
        }
        let score = xv * xv + yv * yv;
        if !score.is_finite() {
            continue;
        }

        // Use log scale for the score as well (positive)
        let ls = (1.0 + score).log10();
        if !ls.is_finite() {
            continue;
        }

        if e.is_true_edge {
            same.push(ls);
        } else {
            diff.push(ls);
        }
    }

    let out_path = cli.scan.out_dir.join(format!(
        "edge_pair_score_log_{}_plus_{}.png",
        sanitize_filename(x_name),
        sanitize_filename(y_name)
    ));

    plot_two_histograms(
        &out_path,
        1024,
        768,
        &format!(
            "Pair score histogram: log10(1 + {}^2 + {}^2)",
            x_name, y_name
        ),
        "log10(1 + score)",
        &same,
        &diff,
        200,
    )?;

    log!(
        cli,
        "Saved pair-score histogram for '{}' + '{}' to {}",
        x_name,
        y_name,
        out_path
    );

    Ok(())
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

fn main() -> Result<()> {
    let t0 = std::time::Instant::now();
    let cli = Cli::parse();

    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("create {}", cli.scan.out_dir))?;

    // Engine config
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

    let nb_pairs = pairs.len();

    let pb = if cli.quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(nb_pairs as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise}) {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(200));
        pb.set_message("computing features…");
        pb
    };

    // Evaluate
    let pb_par = pb.clone();

    let outs: Vec<EdgeFeatureTruth> = pairs
        .par_iter()
        .enumerate()
        .fold(
            || Vec::new(),
            |mut acc, (idx, (left_nid, right_nid, gap))| -> Vec<EdgeFeatureTruth> {
                let mut logbuf = String::new();

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

                let left = seed_store.get(left_nid).unwrap();
                let right = seed_store.get(right_nid).unwrap();

                if left.seeds.len() + right.seeds.len() >= 40_000 {
                    return Vec::new();
                }

                let (_, edges) = step_generate_topk_edges(
                    &mut logbuf,
                    &cli,
                    left,
                    right,
                    &updated_config,
                    &spatial_binner,
                );

                let edge_features = edges
                    .iter()
                    .filter_map(|e| {
                        let feat = e.compute_features();
                        let is_true = left.edge_truth(right, e).unwrap_or(false);
                        Some(EdgeFeatureTruth {
                            feature: feat,
                            is_true_edge: is_true,
                        })
                    })
                    .collect::<Vec<_>>();

                acc.extend(edge_features);
                pb_par.inc(1);
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, mut b| {
                a.append(&mut b);
                a
            },
        );

    pb.finish_with_message("evaluation complete");

    log_section!(&cli, "Results");
    log!(cli, "Total edges generated: {}", outs.len());

    if outs.is_empty() {
        log!(cli, "No edges; nothing to plot.");
        return Ok(());
    }

    let n_true = outs.iter().filter(|e| e.is_true_edge).count();
    let n_false = outs.len() - n_true;

    log!(
        cli,
        "True edges: {} ({:.2}%)",
        n_true,
        100.0 * n_true as f64 / outs.len() as f64
    );
    log!(
        cli,
        "False edges: {} ({:.2}%)",
        n_false,
        100.0 * n_false as f64 / outs.len() as f64
    );

    // -------------------------------------------------------------------------
    // 1) Plot log-histograms for all canonical features
    // -------------------------------------------------------------------------
    let specs = EdgeFeatureTruth::feature_specs();
    for spec in &specs {
        if let Err(e) = plot_feature_histogram(&cli, &outs, *spec) {
            log!(
                &cli,
                "Error plotting histogram for feature '{}': {}",
                spec.name(),
                e
            );
        }
    }

    // -------------------------------------------------------------------------
    // 2) 2D density + ratio plots for a few *high-value* pairs
    // -------------------------------------------------------------------------
    // These are the ones that usually reveal structure:
    // - Cholesky components (z1 vs z2)
    // - residual norm vs speed mismatch
    // - chi2_pos vs rel speed diff
    //
    // Adjust / add pairs freely.
    let pairs_to_plot: &[(EdgeFeatureKey, EdgeFeatureKey)] = &[
        (
            EdgeFeatureKey::PositionCholZ1,
            EdgeFeatureKey::PositionCholZ2,
        ),
        (
            EdgeFeatureKey::PositionCholZNorm,
            EdgeFeatureKey::VelocityRelSpeedDiff,
        ),
        (
            EdgeFeatureKey::PositionChi2Pos,
            EdgeFeatureKey::VelocityRelSpeedDiff,
        ),
        (
            EdgeFeatureKey::PositionZResidNorm,
            EdgeFeatureKey::VelocityRelSpeedDiff,
        ),
    ];

    for &(kx, ky) in pairs_to_plot {
        let x = FeatureSpec { key: kx };
        let y = FeatureSpec { key: ky };

        if let Err(e) = plot_density_triplet(&cli, &outs, x, y) {
            log!(
                &cli,
                "Error plotting 2D density for '{}' vs '{}': {}",
                x.name(),
                y.name(),
                e
            );
        }

        if let Err(e) = plot_pair_score_histogram(&cli, &outs, x, y) {
            log!(
                &cli,
                "Error plotting pair-score hist for '{}' + '{}': {}",
                x.name(),
                y.name(),
                e
            );
        }
    }

    Ok(())
}

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use camino::Utf8Path;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fink_fat_engine::AlertKey;
use fink_fat_engine::graph::edge::edge_features::EdgeFeatures;
use fink_fat_engine::graph::edge::edge_prediction::EdgeRankingModelPool;
use fink_fat_engine::graph::edge::ranking_topk::rank_topk_edges_for_left_by_cost;
use fink_fat_engine::pipeline::hooks::NoopProgress;
use fink_fat_engine::seeding::SeedNode;
use fink_fat_engine::seeding::seed_spatial_index::SeedSpatialIndex;
use fink_fat_engine::seeding::store::SeedStore;
use fink_fat_engine::spacetime_bucket::healpix_binner::HealpixBinner;
use fink_fat_engine::spacetime_bucket::uniform_time_binner::UniformTimeBinner;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fink_fat_engine::{
    Alert, engine_config::edge_config::EdgeConfig, graph::edge::Edge, night_id::NightId,
};
use smallvec::SmallVec;

// -----------------------------------------------------------------------------
// Helpers: synthetic data generation
// -----------------------------------------------------------------------------

/// Build a minimal [`Alert`] suitable for creating seeds in benchmarks.
///
/// Notes
/// -----
/// - This is **not** intended to be physically accurate: the goal is to generate
///   stable, deterministic inputs that exercise the linking code paths.
/// - RA/Dec errors are fixed to a small constant to keep the seed model stable.
/// - Mag errors are a simple proportional rule-of-thumb (never below 1).
fn make_alert(
    dia_source_id: u64,
    ra_rad: f64,
    dec_rad: f64,
    mjd_tt: f64,
    band: u8,
    mag: f64,
) -> Alert {
    Alert {
        key: AlertKey {
            night_id: NightId(0),
            dia_source_id,
        },
        ra: ra_rad,
        ra_err: 1.0e-6, // ~0.2 arcsec in radians
        dec: dec_rad,
        dec_err: 1.0e-6,
        mjd_tt,
        mag,
        mag_err: (0.1 * mag.abs()).max(1.0),
        band,
        observer_mpc_code: Arc::new("I41".to_string()),
    }
}

/// Build `num_seeds` [`SeedNode`] values for a single night using a pair-based model.
///
/// The intent is to create a *pseudo track* (monotone time, slowly varying RA/Dec)
/// so that the spatial/time prefiltering in `score_edge_candidates` is exercised,
/// but without requiring real alert streams.
///
/// Parameters
/// ----------
/// rng : &mut StdRng
///     RNG used only to add small, deterministic jitter.
/// night_id : NightId
///     Night identifier assigned to all seeds.
/// seed_id_base : u64
///     Base value for generating unique [`SeedId`] values.
/// num_seeds : usize
///     Number of seeds to generate.
/// start_mjd_tt : f64
///     Start time (MJD TT) for the first seed.
/// seed_time_step_days : f64
///     Time step (days) between successive seeds (controls density along time).
/// start_ra_rad : f64
///     RA of the first seed (radians).
/// start_dec_rad : f64
///     Dec of the first seed (radians).
/// ra_drift_rad_per_seed : f64
///     Linear RA drift applied per seed (radians/seed).
/// dec_drift_rad_per_seed : f64
///     Linear Dec drift applied per seed (radians/seed).
/// max_speed_rad_per_day : Option<f64>
///     Optional speed filter passed to [`SeedNode::from_pair`].
///
/// Returns
/// -------
/// Vec<SeedNode>
///     Seeds time-sorted by `plane.epoch_mid` to satisfy the invariant required
///     by `SeedNode::score_edge_candidates`.
fn make_seeds_pair_model(
    rng: &mut StdRng,
    night_id: NightId,
    num_seeds: usize,
    spec: SeedSeriesSpec,
) -> Vec<SeedNode> {
    let mut seed_store: SeedStore = SeedStore::new();

    // NOTE (bench-only):
    // We leak the alerts to obtain &'static Alert references.
    // This avoids lifetime issues because SeedNode::from_pair stores references.
    // Criterion benchmarks build inputs once, so this is a pragmatic solution.
    for seed_index in 0..num_seeds {
        let time_alert_a = spec.start_mjd_tt + (seed_index as f64) * spec.seed_time_step_days;
        let time_alert_b = time_alert_a + (spec.seed_time_step_days * 0.5).max(1e-6);

        let ra_jitter = (rng.random::<f64>() - 0.5) * 1e-4;
        let dec_jitter = (rng.random::<f64>() - 0.5) * 1e-4;

        let ra_a = spec.start_ra_rad + (seed_index as f64) * spec.ra_drift_rad_per_seed + ra_jitter;
        let dec_a =
            spec.start_dec_rad + (seed_index as f64) * spec.dec_drift_rad_per_seed + dec_jitter;

        let ra_b = ra_a + spec.ra_drift_rad_per_seed * 0.5;
        let dec_b = dec_a + spec.dec_drift_rad_per_seed * 0.5;

        let band_a = (seed_index % 2) as u8;
        let band_b = ((seed_index + 1) % 2) as u8;

        let mag_a = 1000.0 + (rng.random::<f64>() - 0.5) * 50.0;
        let mag_b = mag_a + (rng.random::<f64>() - 0.5) * 20.0;

        let dia_source_id = 1_000_000 + seed_index as u64;

        let alert_a: &'static Alert = Box::leak(Box::new(make_alert(
            dia_source_id,
            ra_a,
            dec_a,
            time_alert_a,
            band_a,
            mag_a,
        )));
        let alert_b: &'static Alert = Box::leak(Box::new(make_alert(
            dia_source_id,
            ra_b,
            dec_b,
            time_alert_b,
            band_b,
            mag_b,
        )));

        let seed_node = SeedNode::from_pair(
            &mut seed_store,
            night_id,
            alert_a,
            alert_b,
            spec.max_speed_rad_per_day,
        )
        .expect("SeedNode::from_pair failed (speed filter too strict?)");

        seed_store.insert_seed(night_id, seed_node);
    }

    seed_store.sort_night(night_id);
    seed_store.get(&night_id).unwrap_or_default().to_vec()
}

#[derive(Clone, Copy)]
struct SeedSeriesSpec {
    start_mjd_tt: f64,
    seed_time_step_days: f64,
    start_ra_rad: f64,
    start_dec_rad: f64,
    ra_drift_rad_per_seed: f64,
    dec_drift_rad_per_seed: f64,
    max_speed_rad_per_day: Option<f64>,
}

/// Construct the spatial + time binners used by `SeedNode::score_edge_candidates`.
///
/// Parameters
/// ----------
/// t0 : f64
///     Origin (MJD TT) for uniform time binning. A good choice is the minimum
///     `epoch_mid` of the right-hand seeds so time-bin indices remain small and
///     stable in logs/diagnostics.
///
/// Notes
/// -----
/// - The Healpix resolution (here `10`) and the uniform bin width are key
///   knobs that strongly affect runtime by changing candidate fan-out.
/// - Keep them stable while profiling; sweep them later if needed.
fn make_binners(t0: f64) -> (HealpixBinner, UniformTimeBinner) {
    let spatial_binner = HealpixBinner::new(10);

    // 5 minutes in days. Smaller bins -> more bins (more indices), but tighter
    // time consistency; larger bins -> fewer indices, but potentially more
    // spatial candidates per bin.
    let bin_width_days = 5.0 / 1440.0;

    let time_binner = UniformTimeBinner::new(t0, bin_width_days);
    (spatial_binner, time_binner)
}

// -----------------------------------------------------------------------------
// Benchmarks
// -----------------------------------------------------------------------------

/// End-to-end benchmark for [`Edge::generate_topk_edges`].
///
/// This measures the full pipeline cost:
/// - per-left candidate generation + scoring,
/// - per-left Top-K extraction,
/// - conversion to final `Edge` objects,
/// - optional global truncation.
fn bench_generate_topk_edges_end_to_end(c: &mut Criterion) {
    let model_path = std::env::var("FINK_FAT_EDGE_ONNX")
        .expect("Missing env var FINK_FAT_EDGE_ONNX (path to .onnx model)");
    let model_path = Utf8Path::new(&model_path);

    let mut group = c.benchmark_group("generate_topk_edges/e2e");

    // Typical scaling cases.
    // L = number of left seeds, R = number of right seeds, K = top-k per left.
    let cases = [
        (32usize, 512usize, 8usize),
        (64usize, 1024usize, 8usize),
        (128usize, 2048usize, 8usize),
    ];

    let mut rng = StdRng::seed_from_u64(42);
    let mut rng2 = rng.clone();

    for (num_left_seeds, num_right_seeds, top_k_per_left) in cases {
        let left_night = NightId(100);
        let right_night = NightId(101);

        // Synthetic inputs: coherent pseudo-track + deterministic jitter.
        let left_seeds = make_seeds_pair_model(
            &mut rng,
            left_night,
            num_left_seeds,
            SeedSeriesSpec {
                start_mjd_tt: 60000.0,
                seed_time_step_days: 2.0 / 1440.0, // 2-minute spacing
                start_ra_rad: 1.0,
                start_dec_rad: 0.5,
                ra_drift_rad_per_seed: 5e-5,
                dec_drift_rad_per_seed: 2e-5,
                max_speed_rad_per_day: None,
            },
        );

        let right_seeds = make_seeds_pair_model(
            &mut rng2,
            right_night,
            num_right_seeds,
            SeedSeriesSpec {
                start_mjd_tt: 60001.0,
                seed_time_step_days: 2.0 / 1440.0,
                start_ra_rad: 1.01,
                start_dec_rad: 0.51,
                ra_drift_rad_per_seed: 5e-5,
                dec_drift_rad_per_seed: 2e-5,
                max_speed_rad_per_day: None,
            },
        );

        // Use the minimum right epoch as time origin so bin indices stay small.
        let time_origin = right_seeds
            .first()
            .map(|seed| seed.plane.epoch_mid)
            .unwrap_or(60001.0);

        let (spatial_binner, _) = make_binners(time_origin);

        // Base configuration for the tested function.
        let edge_config = EdgeConfig {
            top_k_per_left: Some(top_k_per_left),
            ..EdgeConfig::default()
        };

        group.throughput(Throughput::Elements(
            (num_left_seeds * top_k_per_left) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new(
                "no_cap",
                format!("L{num_left_seeds}_R{num_right_seeds}_K{top_k_per_left}"),
            ),
            &(num_left_seeds, num_right_seeds, top_k_per_left),
            |b, _| {
                let pool = EdgeRankingModelPool::new(model_path);

                b.iter(|| {
                    let edges = Edge::build_edges(
                        black_box(&left_seeds),
                        black_box(&right_seeds),
                        black_box(&edge_config),
                        black_box(&spatial_binner),
                        black_box(time_origin),
                        black_box(Some(&pool)),
                        black_box(&NoopProgress {}),
                    )
                    .expect("generate_topk_edges failed");

                    black_box(edges.len())
                })
            },
        );

        // Variant with a global edge cap: adds an extra selection + partial sort step.
        let edge_config_capped = edge_config.clone();

        group.bench_with_input(
            BenchmarkId::new(
                "global_cap",
                format!("L{num_left_seeds}_R{num_right_seeds}_K{top_k_per_left}"),
            ),
            &(num_left_seeds, num_right_seeds, top_k_per_left),
            |b, _| {
                let pool = EdgeRankingModelPool::new(model_path);

                b.iter(|| {
                    let edges = Edge::build_edges(
                        black_box(&left_seeds),
                        black_box(&right_seeds),
                        black_box(&edge_config_capped),
                        black_box(&spatial_binner),
                        black_box(time_origin),
                        black_box(Some(&pool)),
                        black_box(&NoopProgress {}),
                    )
                    .expect("generate_topk_edges failed");

                    black_box(edges.len())
                })
            },
        );
    }

    group.finish();
}

/// Component benchmarks to isolate major contributors.
///
/// This helps decide whether optimizations should target:
/// - candidate generation (bin slicing + index build + cone query),
/// - exact scoring,
/// - Top-K selection and sorting,
/// - fixed overheads per call.
fn bench_generate_topk_edges_components(c: &mut Criterion) {
    let model_path = std::env::var("FINK_FAT_EDGE_ONNX")
        .expect("Missing env var FINK_FAT_EDGE_ONNX (path to .onnx model)");
    let model_path = Utf8Path::new(&model_path);

    let mut group = c.benchmark_group("generate_topk_edges/components");

    // -----------------------------------------------------------------------------
    // Component benchmarks should be *cheap enough* to sample.
    //
    // The previous "total" version (scoring all left seeds for a large (L,R))
    // is effectively:
    //   cost ~ O(L * score_edge_candidates(right))
    // which can easily become minutes per sample and makes Criterion unusable.
    //
    // Here we instead benchmark:
    // - score_edge_candidates for a *single* left seed (per_left_1)
    // - score_edge_candidates for a small batch of left seeds (per_left_16)
    // and keep (L,R) at a moderate size by default.
    // -----------------------------------------------------------------------------

    // Moderate size for decomposition (keep it close to e2e cases).
    let num_left_seeds: usize = 1_024;
    let num_right_seeds: usize = 4_096;
    let top_k_per_left: usize = 8;

    let mut rng = StdRng::seed_from_u64(7);

    let left_night = NightId(200);
    let right_night = NightId(201);

    let mut rng1 = rng.clone();
    let mut rng2 = rng.clone();
    let left_seeds = make_seeds_pair_model(
        &mut rng1,
        left_night,
        num_left_seeds,
        SeedSeriesSpec {
            start_mjd_tt: 61000.0,
            seed_time_step_days: 2.0 / 1440.0,
            start_ra_rad: 2.0,
            start_dec_rad: 0.3,
            ra_drift_rad_per_seed: 6e-5,
            dec_drift_rad_per_seed: 3e-5,
            max_speed_rad_per_day: None,
        },
    );

    let right_seeds = make_seeds_pair_model(
        &mut rng,
        right_night,
        num_right_seeds,
        SeedSeriesSpec {
            start_mjd_tt: 61001.0,
            seed_time_step_days: 2.0 / 1440.0,
            start_ra_rad: 2.01,
            start_dec_rad: 0.31,
            ra_drift_rad_per_seed: 6e-5,
            dec_drift_rad_per_seed: 3e-5,
            max_speed_rad_per_day: None,
        },
    );

    // Use the minimum right epoch as time origin so time-bin indices stay small.
    let time_origin = right_seeds
        .first()
        .map(|seed| seed.plane.epoch_mid)
        .unwrap_or(61001.0);

    let (spatial_binner, time_binner) = make_binners(time_origin);
    let right_index = SeedSpatialIndex::build(&right_seeds, &spatial_binner, &time_binner);

    let edge_config = EdgeConfig {
        top_k_per_left: Some(top_k_per_left),
        ..EdgeConfig::default()
    };

    // -----------------------------------------------------------------------------
    // 1) Candidate generation + scoring per single left seed.
    //
    // This is the primary bottleneck in most configurations (bin slicing,
    // per-bin index build, cone queries, and exact scoring).
    // -----------------------------------------------------------------------------
    group.bench_function("seed_edge_candidates/per_left_1", |b| {
        let left_seed = &left_seeds[0];
        b.iter(|| {
            let mut n: usize = 0;
            for _to in
                left_seed.seed_edge_candidates(black_box(&right_index), black_box(&edge_config))
            {
                n += 1;
            }
            black_box(n)
        })
    });

    group.bench_function("seed_edge_candidates_plus_features/per_left_1", |b| {
        let left_seed = &left_seeds[0];
        b.iter(|| {
            let mut n: usize = 0;
            for to in
                left_seed.seed_edge_candidates(black_box(&right_index), black_box(&edge_config))
            {
                // measure the "feature extraction" cost (CPU)
                let f = EdgeFeatures::compute_features(left_seed, to);
                black_box(f);
                n += 1;
            }
            black_box(n)
        })
    });

    group.bench_function("rank_topk_edges_for_left_by_cost/per_left_1", |b| {
        let left_seed = &left_seeds[0];

        // Reusable output buffer
        let mut out: SmallVec<[(&SeedNode, f64); 32]> = SmallVec::new();

        b.iter(|| {
            rank_topk_edges_for_left_by_cost(
                black_box(left_seed),
                black_box(&right_index),
                black_box(&edge_config),
                black_box(top_k_per_left),
                black_box(&mut out),
            );

            black_box(out.len())
        })
    });

    // -----------------------------------------------------------------------------
    // 1b) Candidate generation + scoring for a small batch of left seeds.
    //
    // This reduces measurement noise (amortizes per-call jitter) and gives a
    // number you can extrapolate to e2e time as ~ (L / batch) * cost(batch).
    // -----------------------------------------------------------------------------
    group.bench_function("seed_edge_candidates/per_left_16", |b| {
        let left_batch: &[SeedNode] = &left_seeds[..16.min(left_seeds.len())];
        b.iter(|| {
            let mut total: usize = 0;
            for left_seed in left_batch {
                for _to in
                    left_seed.seed_edge_candidates(black_box(&right_index), black_box(&edge_config))
                {
                    total += 1;
                }
            }
            black_box(total)
        })
    });

    group.bench_function("seed_edge_candidates_plus_features/per_left_16", |b| {
        let left_batch: &[SeedNode] = &left_seeds[..16.min(left_seeds.len())];
        b.iter(|| {
            let mut total: usize = 0;
            for left_seed in left_batch {
                for to in
                    left_seed.seed_edge_candidates(black_box(&right_index), black_box(&edge_config))
                {
                    let f = EdgeFeatures::compute_features(left_seed, to);
                    black_box(f);
                    total += 1;
                }
            }
            black_box(total)
        })
    });
    // -----------------------------------------------------------------------------
    // 2) Top-K selection cost only (pure selection on synthetic scalar costs).
    //
    // This isolates the cost of:
    // - select_nth_unstable
    // - sorting the top-k prefix
    // - truncation
    //
    // It does *not* include scoring or allocations done upstream.
    // -----------------------------------------------------------------------------
    group.bench_function("topk_select_only", |b| {
        b.iter(|| {
            // NOTE: If you want to isolate selection further, pre-generate a vector
            // and clone it here. This version includes RNG cost but is still useful
            // as an order-of-magnitude indicator.
            let mut synthetic_costs: Vec<f64> = (0..2048).map(|_| rng2.random::<f64>()).collect();

            let k = top_k_per_left.min(synthetic_costs.len());
            if k > 0 {
                let _ = synthetic_costs.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
                synthetic_costs[..k].sort_by(|a, b| a.total_cmp(b));
                synthetic_costs.truncate(k);
            }

            black_box(synthetic_costs.len())
        })
    });

    // -----------------------------------------------------------------------------
    // 3) End-to-end with a small left slice to highlight fixed per-call overhead.
    //
    // This includes map construction, per-left loop overhead, etc., but keeps the
    // number of left seeds small to remain sample-friendly.
    // -----------------------------------------------------------------------------
    group.bench_function("e2e_small_left", |b| {
        let left_small: &[SeedNode] = &left_seeds[..32.min(left_seeds.len())];
        let pool = EdgeRankingModelPool::new(model_path);

        b.iter(|| {
            let edges = Edge::build_edges(
                black_box(left_small),
                black_box(&right_seeds),
                black_box(&edge_config),
                black_box(&spatial_binner),
                black_box(time_origin),
                black_box(Some(&pool)),
                black_box(&NoopProgress {}),
            )
            .expect("generate_topk_edges failed");

            black_box(edges.len())
        })
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Criterion configuration
// -----------------------------------------------------------------------------

/// Criterion configuration tuned for "expensive" end-to-end benchmarks.
///
/// The default Criterion configuration targets ~100 samples and can become
/// impractically slow when each iteration is expensive (e.g. many scoring calls).
///
/// We reduce:
/// - sample count,
/// - warmup time,
/// - measurement time,
/// - and slightly increase tolerated noise to get actionable numbers quickly.
fn criterion_config() -> Criterion {
    Criterion::default()
        .with_plots()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
        .noise_threshold(0.05)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        bench_generate_topk_edges_end_to_end,
        bench_generate_topk_edges_components
}

criterion_main!(benches);

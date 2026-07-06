pub mod eval;

use anyhow::Result;
use polars::prelude::*;

use crate::eval::{metrics, plots};

fn main() -> Result<()> {
    let truth = LazyFrame::scan_parquet(
        "../../test_exp/sso_dataset_eval.parquet".into(),
        ScanArgsParquet::default(),
    )?
    .select([col("id"), col("traj_id"), col("mjd_tt")]) // <── mjd needed for arc
    .collect()?;

    let files: Vec<_> = glob::glob("track_storage_v*.parquet")?
        .filter_map(|e| e.ok())
        .collect();

    let mut all_chunks: Vec<DataFrame> = Vec::new();

    for path in &files {
        println!("Processing {:?}", path);

        // ── 1. Per-tracklet metadata (before explode) ─────────────────────────
        // n_points: count "|"-separated tokens in obs_ids
        // arc_days: requires joining MJD from truth – computed after explode+join
        let per_tracklet =
            LazyFrame::scan_parquet(path.to_str().unwrap().into(), ScanArgsParquet::default())?
                .filter(col("fit_type").is_not_null())
                .filter(
                    col("fit_type")
                        .str()
                        .starts_with(lit("IODGauss"))
                        .or(col("fit_type").eq(lit("DifferentialCorrection"))),
                )
                .select([
                    col("obs_ids"),
                    col("fit_type"),
                    col("track_id"),
                    col("iod_rms"),
                    col("chi2"),
                ])
                .with_column(
                    // n_points = number of obs_ids (split count)
                    col("obs_ids")
                        .str()
                        .split(lit("|"))
                        .list()
                        .len()
                        .cast(DataType::UInt32)
                        .alias("n_points"),
                )
                .collect()?;

        // ── 2. Explode + join truth to get traj_id and mjd ────────────────────
        let exploded = per_tracklet
            .clone()
            .lazy()
            .with_column(col("obs_ids").str().split(lit("|")).alias("obs_id_list"))
            .explode(
                cols(["obs_id_list"]),
                ExplodeOptions {
                    empty_as_null: false,
                    keep_nulls: false,
                },
            )
            .with_column(col("obs_id_list").cast(DataType::UInt64).alias("obs_id"))
            .drop(cols(["obs_id_list", "obs_ids"]))
            .join(
                truth.clone().lazy(),
                [col("obs_id")],
                [col("id")],
                JoinArgs::new(JoinType::Left),
            )
            // Explicit select to guarantee column presence
            .select([
                col("track_id"),
                col("fit_type"),
                col("iod_rms"),
                col("chi2"),
                col("n_points"),
                col("traj_id"),
                col("mjd_tt"),
                col("obs_id"),
            ])
            .collect()?;

        // ── 3. Compute arc_days per tracklet (max_mjd - min_mjd) ─────────────
        let arc = exploded
            .clone()
            .lazy()
            .filter(col("mjd_tt").is_not_null())
            .group_by([col("track_id")])
            .agg([(col("mjd_tt").max() - col("mjd_tt").min()).alias("arc_days")])
            .collect()?;

        // ── 4. Join arc_days back onto the exploded rows ──────────────────────
        let chunk = exploded
            .lazy()
            .join(
                arc.lazy(),
                [col("track_id")],
                [col("track_id")],
                JoinArgs::new(JoinType::Left),
            )
            .collect()?;

        all_chunks.push(chunk);
    }

    let df = concat(
        all_chunks
            .iter()
            .map(|d| d.clone().lazy())
            .collect::<Vec<_>>(),
        UnionArgs::default(),
    )?
    .collect()?;

    println!("df columns: {:?}", df.get_column_names());
    println!("df shape: {:?}", df.shape());

    println!("Joined DataFrame: {} rows", df.height());

    let summary = metrics::compute(&df, &truth)?;
    metrics::print_report(&summary);

    // Annotate once for all plots
    let df_annotated = plots::prepare_df_for_plots(&df)?;

    plots::plot_iod_rms_distribution(&df_annotated, "plot_iod_rms.png")?;
    plots::plot_chi2_distribution(&df_annotated, "plot_chi2.png")?;

    // Per-tracklet plots: collapse here to save memory
    let df_per_track = plots::prepare_per_tracklet_df(&df_annotated)?;
    drop(df_annotated); // free 48M rows before plotting

    plots::plot_arc_distribution(&df_per_track, "plot_arc.png")?;
    plots::plot_npoints_distribution(&df_per_track, "plot_npoints.png")?;

    Ok(())
}

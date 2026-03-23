//! Parquet export of edge features for ML training.
//!
//! This module provides [`export_edge_features_parquet`], which iterates over
//! every edge in the built graph, recomputes the cadence-robust feature set,
//! labels each edge with the ground-truth (`is_true_edge`) from [`TruthSSO`],
//! and writes the result as a Parquet file.
//!
//! The Parquet schema matches the column names expected by the Python training
//! pipeline defined in `config.py`:
//!
//! | Column           | Type | Description                                            |
//! |------------------|------|--------------------------------------------------------|
//! | `position.*`     | f64  | 10 position-feature columns                            |
//! | `velocity.*`     | f64  | 3 velocity-feature columns                             |
//! | `uncertainty.*`  | f64  | 1 uncertainty-feature column                           |
//! | `photometry.*`   | f64  | 3 photometry-feature columns                           |
//! | `is_true_edge`   | i8   | 1 = true positive, 0 = false positive / unknown        |
//! | `from_seed_id`   | u64  | Unique ID of the "from" seed (group column for CV)     |
//! | `left_nid`       | u32  | Night ID of the "from" seed                            |
//! | `right_nid`      | u32  | Night ID of the "to" seed                              |
//! | `gap_nights`     | u32  | `right_nid − left_nid`                                 |

use std::fs::File;

use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::graph::edge::edge_features::{EDGE_FEATURE_KEYS, EdgeFeatures};
use fink_fat_engine::pipeline::PipelineContext;
use polars::prelude::*;

use crate::truth_sso::{TruthClass, TruthSSO};

/// Export edge features for all edges in the graph as a Parquet file.
///
/// Features are recomputed from the seed nodes (not cached during the pipeline
/// run).  The `is_true_edge` label is derived from [`TruthSSO::classify`]:
/// `1` for `TruePositive`, `0` for everything else (false positive or unknown).
///
/// Arguments
/// ---------
/// * `ctx`      – Pipeline context owning the graph and seed/alert stores.
/// * `truth`    – Ground-truth oracle used to label each edge.
/// * `out_path` – Destination Parquet path.  Parent directories are created
///   automatically if they do not exist.
///
/// Return
/// ------
/// `Ok(())` on success.
pub fn export_edge_features_parquet(
    ctx: &PipelineContext,
    truth: &TruthSSO,
    out_path: &Utf8Path,
) -> Result<()> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;
    let edges = &ctx.runtime_state.graph.edges;
    let n = edges.len();

    tracing::info!(n_edges = n, path = %out_path, "exporting edge features");

    // ── Pre-allocate column buffers ─────────────────────────────────────────
    let n_feats = EDGE_FEATURE_KEYS.len();
    let mut feat_bufs: Vec<Vec<f64>> = vec![Vec::with_capacity(n); n_feats];
    let mut col_is_true_edge: Vec<i8> = Vec::with_capacity(n);
    let mut col_from_seed_id: Vec<u64> = Vec::with_capacity(n);
    let mut col_left_nid: Vec<u32> = Vec::with_capacity(n);
    let mut col_right_nid: Vec<u32> = Vec::with_capacity(n);
    let mut col_gap_nights: Vec<u32> = Vec::with_capacity(n);

    // ── Collect one row per edge ────────────────────────────────────────────
    for edge in edges.iter() {
        let from_seed = seed_store
            .try_get_seed(edge.from)
            .context("from seed not found in seed store")?;
        let to_seed = seed_store
            .try_get_seed(edge.to)
            .context("to seed not found in seed store")?;

        // Debug / group columns.
        let left_nid = from_seed.night_id().value();
        let right_nid = to_seed.night_id().value();
        col_from_seed_id.push(edge.from.unique_id);
        col_left_nid.push(left_nid);
        col_right_nid.push(right_nid);
        col_gap_nights.push(right_nid.saturating_sub(left_nid));

        // Truth label.
        let alerts: Vec<_> = from_seed
            .resolve_members(alert_store)
            .context("resolving from-seed alerts")?
            .into_iter()
            .chain(
                to_seed
                    .resolve_members(alert_store)
                    .context("resolving to-seed alerts")?
                    .into_iter(),
            )
            .collect();
        let is_tp = matches!(truth.classify(&alerts), TruthClass::TruePositive);
        col_is_true_edge.push(i8::from(is_tp));

        // Recompute cadence-robust features.
        let feats = EdgeFeatures::compute_features(from_seed, to_seed);
        for (i, &key) in EDGE_FEATURE_KEYS.iter().enumerate() {
            feat_bufs[i].push(feats.get(key));
        }
    }

    // ── Build DataFrame ─────────────────────────────────────────────────────
    let mut columns: Vec<Column> = Vec::with_capacity(n_feats + 5);

    // Feature columns (canonical order from EDGE_FEATURE_KEYS).
    for (i, &key) in EDGE_FEATURE_KEYS.iter().enumerate() {
        columns.push(Series::new(key.path().into(), &feat_bufs[i]).into_column());
    }

    // Target column.
    columns.push(Series::new("is_true_edge".into(), &col_is_true_edge).into_column());

    // Group column (for cross-validation splits in the Python training code).
    columns.push(Series::new("from_seed_id".into(), &col_from_seed_id).into_column());

    // Debug columns.
    columns.push(Series::new("left_nid".into(), &col_left_nid).into_column());
    columns.push(Series::new("right_nid".into(), &col_right_nid).into_column());
    columns.push(Series::new("gap_nights".into(), &col_gap_nights).into_column());

    let mut df = DataFrame::new(columns).context("building edge features DataFrame")?;

    // ── Write Parquet ───────────────────────────────────────────────────────
    if let Some(parent) = out_path.parent()
        && !parent.as_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating parent directory '{parent}'"))?;
    }

    let mut file = File::create(out_path.as_std_path())
        .with_context(|| format!("creating output file '{out_path}'"))?;

    ParquetWriter::new(&mut file)
        .finish(&mut df)
        .with_context(|| format!("writing Parquet to '{out_path}'"))?;

    tracing::info!(
        path = %out_path,
        n_rows = n,
        n_tp = col_is_true_edge.iter().filter(|&&v| v == 1).count(),
        "edge feature dataset written",
    );

    Ok(())
}

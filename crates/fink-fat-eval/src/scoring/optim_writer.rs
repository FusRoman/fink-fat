//! YAML writers for optimization outputs.
//!
//! Overview
//! --------
//! This module centralizes YAML output writing for optimization runs:
//! - a minimal patch file (only `scoring:`),
//! - a full engine config with updated scoring.
//!
//! Keeping this in a separate module prevents binaries from mixing I/O concerns
//! with optimization logic.

use std::fs;

use anyhow::{Context, Result};
use camino::Utf8Path;

use serde::Serialize;

use fink_fat_engine::engine_config::{EngineConfig, score_config::ScoreConfig};

/// Write the best scoring configuration to YAML files.
///
/// Files
/// -----
/// - `best_scoring.yaml`: `{ scoring: ... }` patch
/// - `best_scoring_full_engine.yaml`: full `EngineConfig` with updated `scoring`
pub fn write_best_scoring_yamls(
    out_dir: &Utf8Path,
    engine_cfg: &EngineConfig,
    best_scoring: ScoreConfig,
) -> Result<(camino::Utf8PathBuf, camino::Utf8PathBuf)> {
    #[derive(Serialize)]
    struct ScoringPatch<'a> {
        scoring: &'a ScoreConfig,
    }

    // Patch YAML
    let patch_path = out_dir.join("best_scoring.yaml");
    let patch = ScoringPatch {
        scoring: &best_scoring,
    };
    let s = serde_yaml::to_string(&patch).context("failed to serialize scoring patch")?;
    fs::write(patch_path.as_std_path(), s).context("failed to write best_scoring.yaml")?;

    // Full engine YAML
    let mut best_engine = engine_cfg.clone();
    best_engine.scoring = best_scoring;

    let full_path = out_dir.join("best_scoring_full_engine.yaml");
    let s =
        serde_yaml::to_string(&best_engine).context("failed to serialize full engine config")?;
    fs::write(full_path.as_std_path(), s)
        .context("failed to write best_scoring_full_engine.yaml")?;

    Ok((patch_path, full_path))
}

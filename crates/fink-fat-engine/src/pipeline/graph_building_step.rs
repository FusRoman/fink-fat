use photom::NightId;

use crate::{
    engine_config::EngineConfig,
    error::EngineError,
    graph::{AlertLinkageDAG, edge::edge_prediction::EdgeRankingModelPool},
    pipeline::{hooks::StageProgress, stages::PipelineStage},
    seeding::{seed_spatial_index::SeedSpatialIndex, store::SeedStore},
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

pub fn build_graph(
    engine_config: &EngineConfig,
    mut graph: AlertLinkageDAG,
    seed_store: &SeedStore,
    edge_models: &Option<EdgeRankingModelPool>,
    stage_sink: &dyn StageProgress,
) -> Result<AlertLinkageDAG, EngineError> {
    let max_gap = engine_config.max_gap_nights();
    if max_gap == 0 {
        return Ok(graph);
    }

    let edge_config = &engine_config.edges;
    let spatial_binner = HealpixBinner::new(engine_config.healpix_depth);

    let mut all_nights: Vec<NightId> = seed_store.nights().copied().collect();
    all_nights.sort();

    let &last_night = all_nights.last().ok_or_else(|| EngineError::StageFailed {
        stage: PipelineStage::BuildEdges,
        message: "no nights found in seed store; cannot build graph".to_string(),
    })?;

    let new_seeds = seed_store
        .get(&last_night)
        .ok_or_else(|| EngineError::StageFailed {
            stage: PipelineStage::BuildEdges,
            message: format!("no seeds found for night {last_night}; cannot build graph"),
        })?;

    let time_binner = UniformTimeBinner::new(
        new_seeds[0].plane_model.epoch_mid,
        engine_config.time_binner_width,
    );
    let right_index = SeedSpatialIndex::build(new_seeds.as_ref(), &spatial_binner, &time_binner);

    // Build edges between the last night and all previous nights within the max gap.
    for &night in all_nights
        .iter()
        .filter(|n| last_night.value().saturating_sub(n.value()) <= max_gap as u32)
    {
        tracing::info!("will build edges between night {night} and night {last_night}");

        let left_seeds = seed_store
            .get(&night)
            .ok_or_else(|| EngineError::StageFailed {
                stage: PipelineStage::BuildEdges,
                message: format!("no seeds found for night {night}; cannot build graph"),
            })?;

        graph = graph.add_inter_night_edges_with_index(
            left_seeds,
            &right_index,
            edge_config,
            edge_models.as_ref(),
            stage_sink,
        )?;
    }

    Ok(graph.commit_edges_sort())
}

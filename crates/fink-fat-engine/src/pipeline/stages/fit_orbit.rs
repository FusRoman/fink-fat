use hifitime::ut1::Ut1Provider;
use outfit::{DifferentialCorrectionConfig, FitLSQ, FullOrbitResult, IODParams, JPLEphem};
use photom::{TrajId, observer::error_model::ObsErrorModel};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
};

pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::FitOrbit,
        hooks,
        StageMeta {
            label: PipelineStage::FitOrbit.label().to_string(),
            total: Some(3),
        },
        |stage_sink| {
            let track_hypothesis = &ctx.runtime_state.track_hypotheses;

            tracing::debug!(n_hypotheses = track_hypothesis.len(), "FitOrbit starting",);

            // Early return: nothing to fit when there are no hypotheses.
            if track_hypothesis.is_empty() {
                tracing::debug!("FitOrbit: no hypotheses, skipping orbit fitting");
                ctx.runtime_state.orbit_results = FullOrbitResult::default();
                stage_sink.inc(3);
                return Ok(vec![]);
            }

            stage_sink.inc(1);

            tracing::trace!("initialising Outfit environment (DE440 + FCCT14)");

            stage_sink.inc(1);

            let mut rng = StdRng::seed_from_u64(42_u64);

            let ut1_provider = Ut1Provider::download_from_jpl("latest_eop2.long")?;
            let jpl_ephem: JPLEphem = "horizon:DE440".try_into()?;

            let iod_params = IODParams::builder()
                .n_noise_realizations(20)
                .noise_scale(1.1)
                .max_obs_for_triplets(20)
                .max_triplets(30)
                .build()?;

            let default_diff_cor_config = DifferentialCorrectionConfig::default();

            tracing::debug!(
                n_noise_realizations = 20,
                noise_scale = 1.1_f64,
                max_obs_for_triplets = 20,
                max_triplets = 30,
                "running orbit estimation (parallel batches)",
            );

            let orbit_results = ctx
                .runtime_state
                .obs_dataset
                .fit_lsq(
                    &jpl_ephem,
                    &ut1_provider,
                    ObsErrorModel::FCCT14,
                    &iod_params,
                    &default_diff_cor_config,
                    None,
                    &mut rng,
                )
                .unwrap();
            let nb_orbit = orbit_results.len() as u64;

            ctx.runtime_state.orbit_results = orbit_results;
            stage_sink.inc(1);

            let nb_successful_fits = ctx
                .runtime_state
                .orbit_results
                .iter()
                .filter(|(_, res)| res.is_ok())
                .count() as u64;
            let nb_failed_fits = nb_orbit - nb_successful_fits;

            // Deactivate all edges belonging to trajectories that received a
            // confirmed orbit.  Once an orbit exists, the pipeline no longer
            // tracks these trajectories through the graph; future propagation
            // will be handled by the ephemeris system (not yet implemented).
            // The deactivated edges are persisted as `EdgeOp::Upsert { active:
            // false }` entries and flushed to the edge journal by `SaveData`.
            let edges_to_deactivate: Vec<_> = {
                let orbit_results = &ctx.runtime_state.orbit_results;
                let track_hypotheses = &ctx.runtime_state.track_hypotheses;
                orbit_results
                    .iter()
                    .filter(|(_, r)| r.is_ok())
                    .filter_map(|(obj, _)| match obj {
                        TrajId::Int(hyp_id) => track_hypotheses.get(hyp_id),
                        _ => None,
                    })
                    .flat_map(|track| track.edges.iter().copied())
                    .collect()
            };
            let n_deactivated = ctx
                .runtime_state
                .graph
                .deactivate_edges(&edges_to_deactivate);

            tracing::debug!(
                nb_orbit,
                nb_successful_fits,
                nb_failed_fits,
                n_deactivated,
                "FitOrbit complete",
            );

            Ok(vec![
                ("total_hypotheses", track_hypothesis.len() as u64),
                ("total_orbits", nb_orbit),
                ("successful_fits", nb_successful_fits),
                ("failed_fits", nb_failed_fits),
                ("deactivated_edges", n_deactivated),
            ])
        },
    )
}

use outfit::{
    ErrorModel, FullOrbitResult, IODParams, Outfit, TrajectoryFile, TrajectoryFit, TrajectorySet,
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
    solver::to_observation_batch,
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

            let obs_batches_by_obs = to_observation_batch(
                track_hypothesis,
                &ctx.runtime_state.alert_store,
                &ctx.runtime_state.seed_store,
            )?;

            tracing::trace!(
                n_obs_groups = obs_batches_by_obs.len(),
                "observation batches resolved from hypotheses",
            );
            stage_sink.inc(1);

            // No observations could be resolved (all hypotheses reference
            // missing alerts/seeds). Return empty results rather than failing.
            let mut iter_obs_batch = obs_batches_by_obs.iter();
            let Some((mpc_code_first, obs_batch_first)) = iter_obs_batch.next() else {
                stage_sink.inc(2);
                return Err(EngineError::OrbitFitting(
                    "no resolved observations for any hypothesis".to_string(),
                ));
            };

            tracing::trace!("initialising Outfit environment (DE440 + FCCT14)");
            let mut env_state = Outfit::new("horizon:DE440", ErrorModel::FCCT14)?;

            let first_obs = env_state.get_observer_from_mpc_code(mpc_code_first);
            let mut traj_set =
                TrajectorySet::new_from_vec(&mut env_state, obs_batch_first, first_obs)?;

            let mut n_obs_groups_loaded: usize = 1;
            for (mpc_code, obs_batch) in iter_obs_batch {
                let observer = env_state.get_observer_from_mpc_code(mpc_code);
                traj_set.add_from_vec(&mut env_state, obs_batch, observer)?;
                n_obs_groups_loaded += 1;
            }

            tracing::debug!(
                n_obs_groups_loaded,
                "TrajectorySet built from observation batches",
            );
            stage_sink.inc(1);

            let mut rng = StdRng::seed_from_u64(42_u64);

            let default = IODParams::builder()
                .n_noise_realizations(10)
                .noise_scale(1.1)
                .max_obs_for_triplets(10)
                .max_triplets(30)
                .build()?;

            tracing::debug!(
                n_noise_realizations = 10,
                noise_scale = 1.1_f64,
                max_obs_for_triplets = 10,
                max_triplets = 30,
                "running orbit estimation (parallel batches)",
            );

            let orbit_results =
                traj_set.estimate_all_orbits_in_batches_parallel(&env_state, &mut rng, &default);
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

            tracing::debug!(
                nb_orbit,
                nb_successful_fits,
                nb_failed_fits,
                "FitOrbit complete",
            );

            Ok(vec![
                ("total_hypotheses", track_hypothesis.len() as u64),
                ("total_orbits", nb_orbit),
                ("successful_fits", nb_successful_fits),
                ("failed_fits", nb_failed_fits),
            ])
        },
    )
}

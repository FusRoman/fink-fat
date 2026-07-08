use std::borrow::Cow;

use anyhow::Result;
use clap::Parser;

use fink_fat_engine::{
    engine_config::{kalman_context::KalmanContext, single_kalman_config::KalmanConfig},
    error::{EngineError, FinkFatError},
    topocentric_kf::single_kalman::KFState,
};
use fink_fat_eval::cli::{Cli, load_data};
use hifitime::ut1::Ut1Provider;
use outfit::{
    JPLEphem, OutfitError,
    cache::{
        observer_centric_cache::ObserverCentricCache, observer_fixed_cache::ObserverFixedCache,
    },
    kepler::{SolverKind, SolverType},
};
use photom::{
    MJDTT, TrajId,
    observation_dataset::{ObsDataset, iter::MemLayoutObservations, observation::Observation},
    observer::Observer,
};

pub fn get_observer<'a>(obs_dataset: &'a ObsDataset, obs: &Observation) -> Option<&'a Observer> {
    obs_dataset.get_observer(*obs.id())
}

pub fn new_fixed_cache(observer: &Observer) -> Result<ObserverFixedCache, OutfitError> {
    ObserverFixedCache::new(observer)
}

pub fn new_centric_cache(
    observer: &Observer,
    jpl: &JPLEphem,
    ut1_provider: &Ut1Provider,
    obs_time: MJDTT,
) -> Result<ObserverCentricCache, OutfitError> {
    let observer_fixed_cache = new_fixed_cache(observer)?;
    ObserverCentricCache::new(jpl, ut1_provider, obs_time, &observer_fixed_cache, true)
}

pub fn materialize_contiguous_traj<'o>(
    obs_dataset: &'o ObsDataset,
    traj: &TrajId,
) -> Result<Cow<'o, [Observation]>, EngineError> {
    match obs_dataset.materialize_trajectory(traj).ok_or_else(|| {
        EngineError::FinkFat(FinkFatError::Message(format!(
            "failed to metariaze trajectory with id: {}",
            traj
        )))
    })? {
        MemLayoutObservations::Contiguous(slice) => Ok(Cow::Borrowed(slice)),
        MemLayoutObservations::Split(vec_obs) => {
            Ok(Cow::Owned(vec_obs.iter().map(|o| (*o).clone()).collect()))
        }
    }
}

/// Initialise the Kalman filter bank from the first intra-night pair found
/// in the trajectory.
///
/// Scans `traj` for the first consecutive pair of observations separated by
/// less than one day, computes the heliocentric observer state at both
/// epochs, and builds an [`KFBank`] seeded from the admissible-region grid
/// at the pair midpoint epoch.
///
/// Arguments
/// ---------
/// * `traj`         – Ordered slice of observations along the trajectory.
/// * `obs_dataset`  – Dataset used to resolve the observer site.
/// * `jpl_ephem`    – JPL ephemeris for solar-system body positions.
/// * `ut1_provider` – UT1 time-scale provider.
///
/// Return
/// ------
/// * `Some((idx, bank))` – Index of the first observation in the pair and the
///   initialised hypothesis bank.
/// * `None` – No intra-night pair was found; the trajectory cannot be
///   bootstrapped.
fn init_kf_from_first_pair<'state_lf>(
    traj: &[Observation],
    obs_dataset: &ObsDataset,
    ephem_state: &'state_lf KalmanContext,
) -> Option<(usize, KFState<'state_lf>)> {
    println!(
        "[KF Init] Scanning {} observations for first intra-night pair (dt < 1 day)...",
        traj.len()
    );

    let (idx_first_obs, pair_obs) = traj
        .windows(2)
        .enumerate()
        .find(|(_, w)| w[1].mjd_tt() - w[0].mjd_tt() < 1.0)?;

    let (obs1, obs2) = (&pair_obs[0], &pair_obs[1]);

    let kf = KFState::init_kf_state(obs_dataset, obs1, obs2, 3.0, ephem_state).unwrap();

    Some((idx_first_obs, kf))
}

fn main() -> Result<()> {
    // ── Logging ───────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("FINKFAT_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .without_time() // remove timestamp for readability
        .with_ansi(false) // disable ANSI color codes when redirecting to file
        .init();

    let cli = Cli::parse();

    let kalman_config = KalmanConfig {
        solver_type: SolverType {
            kind: SolverKind::NewtonRaphson,
            ..Default::default()
        },
        ..Default::default()
    };

    let kalman_ctx = KalmanContext::new(kalman_config, "horizon:DE440", None);

    let (_, obs_dataset) = load_data(&cli.alerts);

    let traj = materialize_contiguous_traj(&obs_dataset, &TrajId::Int(54013))?;

    println!("size traj: {}", traj.len());

    let (idx_first_obs, mut kf) =
        init_kf_from_first_pair(&traj, &obs_dataset, &kalman_ctx).unwrap();

    kf = KFState {
        state: [
            kf.state[0],
            kf.state[1],
            kf.state[2],
            kf.state[3],
            2.53579,
            0.0000005,
        ]
        .into(),
        ..kf.clone()
    };

    println!("{kf}");
    println!("idx first obs: {}", idx_first_obs);

    let observations_to_process = &traj[idx_first_obs + 2..];

    let nb_obs_proc = observations_to_process.len() - 1;

    for (i, obs) in observations_to_process.iter().enumerate() {
        println!("\n\n");

        println!("Processing obs {i} / {nb_obs_proc}");

        let observer = obs_dataset.get_observer(*obs.id()).unwrap();
        let helio_state = kf
            .shared_ctx
            .get_ephem()
            .helio_observer_state(observer, obs.mjd_tt())
            .ok()
            .unwrap();
        let coord = obs.equ_coord();

        println!("\n\n == Current kalman == \n");
        println!("{kf}");
        println!("\n current orbit : \n{}\n", kf.to_orbit());
        println!("\n\n");

        let predict = kf
            .predict(
                obs.mjd_tt(),
                helio_state.helio_cart_pos,
                helio_state.helio_cart_vel,
            )
            .unwrap();
        let equ_predic = predict.to_equ_coord().unwrap();

        let sep_from_next_pred = equ_predic.angular_separation(coord);

        println!(
            "separation with next prediction : {}",
            sep_from_next_pred.to_degrees() * 3600.
        );

        match kf.propagate(&obs_dataset, obs) {
            Ok(propagated_kf) => {
                println!("\n\n");

                let predict_coord = propagated_kf.to_equ_coord().unwrap();

                println!("next obs coord: {}", obs.equ_coord());

                println!("predicted coord: {}", predict_coord);

                println!(
                    "separation: {} arcsecond",
                    obs.equ_coord()
                        .angular_separation(&predict_coord)
                        .to_degrees()
                        * 3600.
                );

                println!("\n\n");

                kf = propagated_kf.update(&obs).unwrap();

                println!("\n\n");

                println!("\nupdate kf: {:?}", kf);

                println!("\n ========== \n\n")
            }
            Err(err) => {
                println!("propagation error : {}", err);
                continue;
            }
        }
    }

    Ok(())
}

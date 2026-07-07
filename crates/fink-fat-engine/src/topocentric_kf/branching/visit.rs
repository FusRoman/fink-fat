//! Group a night's observations into "visits" — the unit of propagation for
//! the per-night branching loop.
//!
//! At LSST cadence a night holds hundreds of distinct exposure epochs
//! (~30s apart over ~8h), not one instant. Candidate observations spread
//! across a night must never be compared against a bank propagated to a
//! single, shared per-night epoch — see the `orchestrate` module doc for
//! the correctness argument. Grouping into visits first lets the branching
//! loop propagate once per *visit* (genuinely one shared epoch) instead of
//! once per *night* (wrong) or once per *observation* (correct but repeats
//! the two-body solve for observations that share an exposure).

use photom::observation_dataset::observation::Observation;

/// One group of observations sharing (approximately) the same epoch.
#[derive(Debug)]
pub struct Visit<'obs> {
    /// Anchor epoch (MJD TT) of this visit — the first observation's, once
    /// sorted chronologically.
    pub epoch: f64,
    /// Any one observation from this visit, used only to resolve *which*
    /// observer (site/instrument) took it — every observation in a visit
    /// shares one exposure, hence one observer.
    pub representative_obs: &'obs Observation,
    /// Every observation belonging to this visit.
    pub observations: Vec<&'obs Observation>,
}

/// Group observations into chronologically-ordered visits.
///
/// Observations are sorted by `mjd_tt`, then folded into visits greedily:
/// a observation joins the current visit if its epoch is within
/// `epoch_tolerance_days` of that visit's anchor epoch (the first
/// observation folded into it); otherwise it starts a new visit.
///
/// # Arguments
/// * `night_obs` – This night's observations, any order.
/// * `epoch_tolerance_days` – Maximum epoch spread (days) for observations
///   to be considered part of the same exposure/visit.
///
/// # Returns
/// Visits in chronological order, each with its own `observations` sorted
/// by epoch.
pub fn group_observations_into_visits<'obs>(
    night_obs: &[&'obs Observation],
    epoch_tolerance_days: f64,
) -> Vec<Visit<'obs>> {
    let mut sorted_obs: Vec<&'obs Observation> = night_obs.to_vec();
    sorted_obs.sort_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()));

    let mut visits: Vec<Visit<'obs>> = Vec::new();
    for obs in sorted_obs {
        match visits.last_mut() {
            Some(visit) if (obs.mjd_tt() - visit.epoch).abs() <= epoch_tolerance_days => {
                visit.observations.push(obs);
            }
            _ => visits.push(Visit {
                epoch: obs.mjd_tt(),
                representative_obs: obs,
                observations: vec![obs],
            }),
        }
    }
    visits
}

#[cfg(test)]
mod tests {
    use super::*;
    use photom::{
        coordinates::equatorial::EquCoord,
        observation_dataset::{ObsDataset, observation::ObservationInput},
        photometry::{Filter, Photometry},
    };

    fn mk_observation(id: u64, mjd_tt: f64) -> Observation {
        let obs_dataset = ObsDataset::empty();
        let equ = EquCoord::new(0.0, 0.0, 0.0, 0.0);
        let photometry = Photometry {
            magnitude: 20.0,
            error: 0.1,
            filter: Filter::String("r".to_string()),
        };
        let input = ObservationInput::new(id, equ, photometry, mjd_tt, None);
        let (obs_dataset, obs_id) = obs_dataset.push_observation(vec![input]).unwrap();
        obs_dataset
            .get_obs_by_index(*obs_id.get(0).unwrap())
            .unwrap()
            .clone()
    }

    #[test]
    fn empty_input_yields_no_visits() {
        let visits = group_observations_into_visits(&[], 1e-4);
        assert!(visits.is_empty());
    }

    #[test]
    fn observations_within_tolerance_form_a_single_visit() {
        let tolerance = 1.0 / 86_400.0; // 1 second
        let a = mk_observation(0, 60000.0);
        let b = mk_observation(1, 60000.0 + 0.3 / 86_400.0); // 0.3s later
        let obs = [&a, &b];

        let visits = group_observations_into_visits(&obs, tolerance);

        assert_eq!(visits.len(), 1);
        assert_eq!(visits[0].observations.len(), 2);
    }

    #[test]
    fn a_gap_beyond_tolerance_starts_a_new_visit() {
        let tolerance = 1.0 / 86_400.0; // 1 second
        let a = mk_observation(0, 60000.0);
        let b = mk_observation(1, 60000.0 + 30.0 / 86_400.0); // 30s later
        let obs = [&a, &b];

        let visits = group_observations_into_visits(&obs, tolerance);

        assert_eq!(visits.len(), 2);
        assert_eq!(visits[0].observations.len(), 1);
        assert_eq!(visits[1].observations.len(), 1);
    }

    #[test]
    fn visits_are_returned_in_chronological_order() {
        let tolerance = 1.0 / 86_400.0;
        let a = mk_observation(0, 60000.5);
        let b = mk_observation(1, 60000.0);
        let c = mk_observation(2, 60001.0);
        // Deliberately unsorted input.
        let obs = [&a, &b, &c];

        let visits = group_observations_into_visits(&obs, tolerance);

        assert_eq!(visits.len(), 3);
        assert!(visits[0].epoch < visits[1].epoch);
        assert!(visits[1].epoch < visits[2].epoch);
        assert_eq!(*visits[0].representative_obs.id(), 1);
        assert_eq!(*visits[1].representative_obs.id(), 0);
        assert_eq!(*visits[2].representative_obs.id(), 2);
    }

    #[test]
    fn gap_exactly_at_tolerance_joins_the_same_visit() {
        let tolerance = 10.0 / 86_400.0; // 10 seconds
        let a = mk_observation(0, 60000.0);
        let b = mk_observation(1, 60000.0 + tolerance);
        let obs = [&a, &b];

        let visits = group_observations_into_visits(&obs, tolerance);

        assert_eq!(visits.len(), 1);
    }
}

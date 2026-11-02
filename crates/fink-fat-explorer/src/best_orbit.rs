//! Centralized "which orbit is best for this branch/lineage" priority.
//!
//! Two call sites independently need the same answer: prefer a stored
//! n-body (`orbit_fits.fit_method = 'differential_correction'`) fit, else an
//! IOD-only one, else fall back to the Kalman-bank (`kf_state`) estimate.
//! [`crate::homepage::snapshot`] uses it for the homepage's a/e plot and
//! family badge; [`crate::lineage_page::identity_card`] uses it for the
//! lineage detail page's identity card. Both used to hand-roll their own
//! copy of this cascade — this module is the single place it is decided,
//! so the two pages can never silently drift apart again.
//!
//! This is deliberately *not* the same concept as
//! `homepage::snapshot::LATEST_ORBIT_FIT_QUERY` (feeds quality-tier
//! assessment: "what did the most recent fit *attempt* do", plain
//! `fitted_at DESC`, no method preference) or
//! `orbit_fit::latest::get_latest_orbit_fit_result` (the fit-result page:
//! "the fit you just ran"). Neither of those should be merged into this
//! module — they intentionally answer a different question.

use crate::fit_pipeline::fit::FitMethod;
use crate::homepage::family::DynamicalFamily;

/// SQL fragment implementing the n-body-over-IOD-over-nothing preference,
/// for use in an `ORDER BY` clause over `orbit_fits` rows (ties broken by
/// most recent `fitted_at`).
///
/// Each call site interpolates this into its own query rather than sharing
/// one query string, since the two shapes genuinely differ: a single-lineage
/// lookup (`WHERE lineage_designation = $1 ... LIMIT 1`) versus a
/// `DISTINCT ON (branch_id)` batch preload. Do not reuse this for
/// `LATEST_ORBIT_FIT_QUERY` or `get_latest_orbit_fit_result` — see the
/// module doc for why those stay plain `fitted_at DESC`.
pub const PREFER_NBODY_ORDER_BY: &str =
    "(fit_method = 'differential_correction') DESC, fitted_at DESC";

/// A candidate orbit found in `orbit_fits` by a caller's own query, if any
/// row existed for the branch/lineage in question.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCandidate {
    pub fit_method: FitMethod,
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
}

/// Which of the three sources a [`BestOrbit`] ultimately came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BestOrbitSource {
    NBody,
    Iod,
    Kalman,
}

impl BestOrbitSource {
    /// Short label for display next to the orbital elements it produced.
    pub fn label(self) -> &'static str {
        match self {
            Self::NBody => "N-body fit",
            Self::Iod => "IOD fit",
            Self::Kalman => "Kalman estimate",
        }
    }
}

/// The resolved best orbit for a branch/lineage: elements, the family they
/// classify into, and which source won.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BestOrbit {
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
    pub family: DynamicalFamily,
    pub source: BestOrbitSource,
}

/// Resolves the n-body > IOD > Kalman priority into a single best orbit.
///
/// This is the one place the cascade is decided: `orbit_fit`, when present,
/// always wins over `kalman` regardless of its own method (a stored
/// `orbit_fits` row — n-body or IOD-only — is always preferred to the
/// Kalman-bank estimate); `kalman` is used only when `orbit_fit` is `None`.
/// The family badge is derived from whichever (a, e) wins here, so it can
/// never disagree with the displayed/plotted elements.
///
/// # Arguments
///
/// * `orbit_fit` — the best `orbit_fits` row for this branch/lineage, as
///   already selected by the caller's own query ordered with
///   [`PREFER_NBODY_ORDER_BY`]; `None` if no such row exists.
/// * `kalman` — the `(semi_major_axis, eccentricity)` fallback from the
///   Kalman-bank state, used only when `orbit_fit` is `None`.
///
/// # Return
///
/// The resolved orbit, its dynamical family, and which source it came from.
#[cfg(feature = "server")]
pub fn resolve_best_orbit(orbit_fit: Option<OrbitCandidate>, kalman: (f64, f64)) -> BestOrbit {
    let (semi_major_axis_au, eccentricity, source) = match orbit_fit {
        Some(candidate) => {
            let source = match candidate.fit_method {
                FitMethod::DifferentialCorrection => BestOrbitSource::NBody,
                FitMethod::IodOnly => BestOrbitSource::Iod,
            };
            (candidate.semi_major_axis_au, candidate.eccentricity, source)
        }
        None => (kalman.0, kalman.1, BestOrbitSource::Kalman),
    };

    BestOrbit {
        semi_major_axis_au,
        eccentricity,
        family: DynamicalFamily::classify(semi_major_axis_au, eccentricity),
        source,
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;

    fn candidate(fit_method: FitMethod, a: f64, e: f64) -> OrbitCandidate {
        OrbitCandidate {
            fit_method,
            semi_major_axis_au: a,
            eccentricity: e,
        }
    }

    #[test]
    fn prefers_nbody_over_kalman_fallback() {
        let best = resolve_best_orbit(
            Some(candidate(FitMethod::DifferentialCorrection, 2.5, 0.1)),
            (99.0, 0.9),
        );
        assert_eq!(best.source, BestOrbitSource::NBody);
        assert_eq!(best.semi_major_axis_au, 2.5);
        assert_eq!(best.eccentricity, 0.1);
    }

    #[test]
    fn uses_iod_when_it_is_the_only_orbit_fits_row() {
        let best = resolve_best_orbit(Some(candidate(FitMethod::IodOnly, 3.2, 0.2)), (99.0, 0.9));
        assert_eq!(best.source, BestOrbitSource::Iod);
        assert_eq!(best.semi_major_axis_au, 3.2);
        assert_eq!(best.eccentricity, 0.2);
    }

    #[test]
    fn falls_back_to_kalman_when_no_orbit_fit_exists() {
        let best = resolve_best_orbit(None, (4.1, 0.05));
        assert_eq!(best.source, BestOrbitSource::Kalman);
        assert_eq!(best.semi_major_axis_au, 4.1);
        assert_eq!(best.eccentricity, 0.05);
    }

    #[test]
    fn family_is_rederived_from_whichever_source_won_not_always_kalman() {
        // Same orbit_fit, different Kalman fallback: the family must be
        // identical in both cases, proving it tracks the winning source
        // (the orbit_fit) and not the Kalman tuple it ignored.
        let candidate_a = candidate(FitMethod::DifferentialCorrection, 2.5, 0.1);
        let with_low_kalman = resolve_best_orbit(Some(candidate_a), (0.5, 0.01));
        let with_high_kalman = resolve_best_orbit(Some(candidate_a), (80.0, 0.5));
        assert_eq!(with_low_kalman.family, with_high_kalman.family);
        assert_eq!(with_low_kalman.family, DynamicalFamily::classify(2.5, 0.1));
    }

    #[test]
    fn family_reflects_kalman_fallback_when_no_orbit_fit_exists() {
        let best = resolve_best_orbit(None, (45.0, 0.1));
        assert_eq!(best.family, DynamicalFamily::classify(45.0, 0.1));
    }
}

//! The lineage 3D view's orbit uncertainty: a cloud of clone orbits drawn
//! from the N-body fit's covariance.
//!
//! Only the N-body differential-correction fit's covariance is used. (The
//! Kalman filter's covariance was tried as a fallback, but it is far larger
//! — it comes from a few real-time updates rather than a full least-squares
//! fit — and swamped the plot.) A fit that did not converge, is a bare IOD
//! solution, or carries no usable covariance simply has no cloud;
//! [`select_covariance`] says why so the UI can tell the user.
//!
//! Everything here is pure computation on values already loaded (no I/O),
//! server-only because it uses `nalgebra`, `rand` and `outfit`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use nalgebra::{Matrix6, SymmetricEigen, Vector6};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::orbit3d::ephem_provider::keplerian_from_outfit_elements;
use crate::orbit3d::geometry::{self, Keplerian};
use crate::orbit3d::types::{UncertaintyCloud3D, UNCERTAINTY_ORBIT_SAMPLES};

/// Number of clones drawn from the covariance.
pub const N_CLONES: usize = 300;
/// How many of the clones also get their whole orbit drawn.
pub const N_ORBIT_CLONES: usize = 20;
/// Clones farther from the Sun than this at the view's epoch are dropped
/// from the "now" cloud, AU.
const MAX_HELIOCENTRIC_DISTANCE_AU: f64 = 100.0;
/// Clones farther from the cloud's median center than this many times the
/// median distance to it are dropped from the "now" cloud.
const OUTLIER_DISTANCE_FACTOR: f64 = 4.0;
/// A covariance is rejected as not positive semi-definite when its smallest
/// eigenvalue is below minus this fraction of its largest — tiny negative
/// eigenvalues are rounding noise on a near-singular matrix and are simply
/// clipped when sampling.
const NEGATIVE_EIGENVALUE_TOLERANCE: f64 = 1e-6;

/// What the N-body fit stored, as far as the covariance choice needs it.
pub(crate) struct FitCovariance {
    /// The best-fit equinoctial elements `(a, h, k, p, q, λ)`, in the order
    /// the covariance is expressed in.
    pub elements: Vector6<f64>,
    /// The flattened 6×6 covariance (`orbit_fits.covariance`); may be empty.
    pub covariance: Vec<f64>,
    /// Epoch of `elements`, MJD-TT.
    pub reference_epoch: f64,
    /// `orbit_fits.converged`.
    pub converged: bool,
    /// Whether the fit is a differential correction (as opposed to a bare
    /// IOD solution).
    pub differential_correction: bool,
    /// The fit's normalised RMS.
    pub normalised_rms: f64,
}

/// The covariance [`select_covariance`] accepted, with everything needed to
/// draw clones from it.
pub(crate) struct Selected {
    /// The best-fit equinoctial elements the clones are scattered around.
    pub mean: Vector6<f64>,
    pub covariance: Matrix6<f64>,
    /// Epoch of `mean`, MJD-TT.
    pub epoch_mjd_tt: f64,
    /// A short description of the fit, for the caption.
    pub detail: String,
}

/// Validates a flattened 6×6 covariance and returns it symmetrised.
///
/// # Arguments
///
/// * `values` — the 36 stored values.
///
/// # Returns
///
/// The symmetrised matrix `(C + Cᵀ) / 2`.
///
/// # Errors
///
/// A short noun phrase describing what is wrong (missing, wrong length,
/// non-finite entries, a non-positive variance, or not positive
/// semi-definite), meant to follow "the N-body fit's" in a message.
fn covariance_matrix(values: &[f64]) -> Result<Matrix6<f64>, String> {
    if values.is_empty() {
        return Err("covariance is missing".to_string());
    }
    if values.len() != 36 {
        return Err(format!(
            "covariance is malformed ({} values instead of 36)",
            values.len()
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("covariance has non-finite entries".to_string());
    }

    let raw = Matrix6::from_column_slice(values);
    let covariance = (raw + raw.transpose()) * 0.5;
    if (0..6).any(|i| covariance[(i, i)] <= 0.0) {
        return Err("covariance has a non-positive variance".to_string());
    }

    let eigenvalues = SymmetricEigen::new(covariance).eigenvalues;
    let largest = eigenvalues.max();
    let smallest = eigenvalues.min();
    if largest <= 0.0 || smallest < -NEGATIVE_EIGENVALUE_TOLERANCE * largest {
        return Err("covariance is not positive semi-definite".to_string());
    }
    Ok(covariance)
}

/// Accepts the N-body fit's covariance for the uncertainty cloud, or says
/// why it cannot be used.
///
/// Usable means: a converged differential correction whose stored
/// covariance is a finite, positive semi-definite 6×6 matrix.
///
/// # Arguments
///
/// * `fit` — the lineage's latest stored fit.
///
/// # Returns
///
/// The [`Selected`] covariance.
///
/// # Errors
///
/// A sentence explaining why there is no usable covariance (a bare IOD
/// solution, a fit that did not converge, or a missing/malformed/non-finite/
/// non-positive-semi-definite covariance), meant to be shown to the user.
pub(crate) fn select_covariance(fit: &FitCovariance) -> Result<Selected, String> {
    if !fit.differential_correction {
        return Err(
            "the stored fit is a bare IOD solution, not a differential correction, so it \
             carries no covariance"
                .to_string(),
        );
    }
    if !fit.converged {
        return Err("the N-body fit did not converge".to_string());
    }
    let covariance =
        covariance_matrix(&fit.covariance).map_err(|why| format!("the N-body fit's {why}"))?;

    Ok(Selected {
        mean: fit.elements,
        covariance,
        epoch_mjd_tt: fit.reference_epoch,
        detail: format!(
            "differential correction, normalised RMS {:.2}",
            fit.normalised_rms
        ),
    })
}

/// A standard normal deviate from two uniform ones (Box–Muller).
///
/// # Arguments
///
/// * `u1` — uniform in `(0, 1]`.
/// * `u2` — uniform in `[0, 1)`.
fn standard_normal(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Draws samples from the multivariate normal `N(mean, covariance)`.
///
/// The covariance is factored by eigendecomposition rather than Cholesky, so
/// a near-singular matrix (typical of a short arc, where some directions are
/// barely constrained) still works: eigenvalues that rounding pushed
/// slightly negative are clipped to zero.
///
/// # Arguments
///
/// * `mean` — the distribution's mean.
/// * `covariance` — its covariance; symmetric positive semi-definite.
/// * `n` — how many samples to draw.
/// * `seed` — seeds the generator, so the same inputs always give the same
///   samples.
///
/// # Returns
///
/// `n` samples, or none if the factorisation is not finite.
pub fn sample_multivariate_normal(
    mean: &Vector6<f64>,
    covariance: &Matrix6<f64>,
    n: usize,
    seed: u64,
) -> Vec<Vector6<f64>> {
    let eigen = SymmetricEigen::new(*covariance);
    let scales = eigen.eigenvalues.map(|l| l.max(0.0).sqrt());
    let transform = eigen.eigenvectors * Matrix6::from_diagonal(&scales);
    if transform.iter().any(|x| !x.is_finite()) {
        return Vec::new();
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let z = Vector6::from_fn(|_, _| {
                let u1: f64 = 1.0 - rng.random::<f64>();
                let u2: f64 = rng.random::<f64>();
                standard_normal(u1, u2)
            });
            mean + transform * z
        })
        .collect()
}

/// A seed derived from a lineage designation, so a lineage's cloud is the
/// same every time its page is loaded.
///
/// # Arguments
///
/// * `designation` — the lineage designation.
///
/// # Returns
///
/// A deterministic 64-bit seed (`DefaultHasher` uses fixed keys).
pub fn seed_from_designation(designation: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    designation.hash(&mut hasher);
    hasher.finish()
}

/// Whether `k` is a finite, closed ellipse.
fn is_closed_ellipse(k: &Keplerian) -> bool {
    [
        k.semi_major_axis_au,
        k.eccentricity,
        k.inclination_deg,
        k.ascending_node_longitude_deg,
        k.periapsis_argument_deg,
        k.mean_anomaly_deg,
    ]
    .iter()
    .all(|x| x.is_finite())
        && k.semi_major_axis_au > 0.0
        && (0.0..1.0).contains(&k.eccentricity)
}

/// Turns one sampled solution into orbital elements.
///
/// # Arguments
///
/// * `selected` — the covariance the sample was drawn from (gives the
///   epoch).
/// * `sample` — equinoctial elements `(a, h, k, p, q, λ)`, e.g. a draw from
///   `N(selected.mean, selected.covariance)`.
///
/// # Returns
///
/// The orbit, or `None` if the sample is not a closed ellipse.
fn clone_orbit(selected: &Selected, sample: &Vector6<f64>) -> Option<Keplerian> {
    let epoch = selected.epoch_mjd_tt;
    let equinoctial = outfit::EquinoctialElements {
        reference_epoch: epoch,
        semi_major_axis: sample[0],
        eccentricity_sin_lon: sample[1],
        eccentricity_cos_lon: sample[2],
        tan_half_incl_sin_node: sample[3],
        tan_half_incl_cos_node: sample[4],
        mean_longitude: sample[5],
    };
    let keplerian: outfit::KeplerianElements = (&equinoctial).into();
    let orbit = keplerian_from_outfit_elements(&keplerian, epoch);
    is_closed_ellipse(&orbit).then_some(orbit)
}

/// The median of `values` (upper median for an even count); `0` for none.
fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Drops the clones that would blow up the plot's scale.
///
/// A short arc leaves some directions barely constrained, so a few clones
/// propagated months ahead can land hundreds of AU away — and the 3D plot,
/// which keeps every axis at the same scale, would shrink everything else
/// to a dot. This keeps clones within [`MAX_HELIOCENTRIC_DISTANCE_AU`] of the
/// Sun and within `factor` times the median distance of the cloud's
/// component-wise median center.
///
/// # Arguments
///
/// * `points` — clone positions, AU.
/// * `factor` — the distance-to-center multiple beyond which a clone is
///   dropped.
///
/// # Returns
///
/// The retained positions, in input order. Non-finite points are always
/// dropped.
pub fn trim_outliers(points: &[[f64; 3]], factor: f64) -> Vec<[f64; 3]> {
    let finite: Vec<[f64; 3]> = points
        .iter()
        .copied()
        .filter(|p| p.iter().all(|x| x.is_finite()))
        .filter(|p| geometry::distance(*p, [0.0; 3]) <= MAX_HELIOCENTRIC_DISTANCE_AU)
        .collect();
    if finite.is_empty() {
        return finite;
    }

    let center = [0, 1, 2].map(|axis| median(finite.iter().map(|p| p[axis]).collect()));
    let typical = median(
        finite
            .iter()
            .map(|p| geometry::distance(*p, center))
            .collect(),
    );
    if typical <= 0.0 {
        return finite;
    }
    finite
        .into_iter()
        .filter(|p| geometry::distance(*p, center) <= factor * typical)
        .collect()
}

/// Draws the uncertainty cloud for an accepted covariance.
///
/// # Arguments
///
/// * `selected` — the covariance to draw from.
/// * `seed` — seeds the sampling (see [`seed_from_designation`]).
/// * `last_observation_mjd` — the epoch of the lineage's last observation,
///   MJD-TT: the tight cloud is drawn there.
/// * `now_mjd` — the view's epoch, MJD-TT: the (trimmed) spread cloud is
///   drawn there.
///
/// # Returns
///
/// The [`UncertaintyCloud3D`]: clone positions at both epochs, a few clone
/// orbits, and the best solution's own positions and orbit (sampled like the
/// clone orbits) so the client can exaggerate the clones' deviations from
/// it. Clones that are not closed ellipses are discarded and counted in
/// `n_sampled - n_clones`.
///
/// # Errors
///
/// The best solution itself is not a closed ellipse.
pub(crate) fn build_uncertainty_cloud(
    selected: &Selected,
    seed: u64,
    last_observation_mjd: f64,
    now_mjd: f64,
) -> Result<UncertaintyCloud3D, String> {
    let best = clone_orbit(selected, &selected.mean)
        .ok_or_else(|| "the fit's best solution is not a closed ellipse".to_string())?;

    let samples = sample_multivariate_normal(&selected.mean, &selected.covariance, N_CLONES, seed);
    let orbits: Vec<Keplerian> = samples
        .iter()
        .filter_map(|sample| clone_orbit(selected, sample))
        .collect();

    let at_last_observation: Vec<[f64; 3]> = orbits
        .iter()
        .map(|o| geometry::position_at_epoch(o, last_observation_mjd))
        .filter(|p| p.iter().all(|x| x.is_finite()))
        .collect();
    let at_now_all: Vec<[f64; 3]> = orbits
        .iter()
        .map(|o| geometry::position_at_epoch(o, now_mjd))
        .collect();
    let at_now = trim_outliers(&at_now_all, OUTLIER_DISTANCE_FACTOR);

    Ok(UncertaintyCloud3D {
        detail: selected.detail.clone(),
        n_sampled: samples.len(),
        n_clones: orbits.len(),
        n_kept_now: at_now.len(),
        center_last_observation: geometry::position_at_epoch(&best, last_observation_mjd),
        center_now: geometry::position_at_epoch(&best, now_mjd),
        best_orbit: geometry::ellipse_points(&best, UNCERTAINTY_ORBIT_SAMPLES),
        at_last_observation,
        at_now,
        orbits: orbits
            .iter()
            .take(N_ORBIT_CLONES)
            .map(|o| geometry::ellipse_points(o, UNCERTAINTY_ORBIT_SAMPLES))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_orbit() -> Keplerian {
        Keplerian {
            epoch_mjd_tt: 60_000.0,
            semi_major_axis_au: 2.7,
            eccentricity: 0.1,
            inclination_deg: 10.0,
            ascending_node_longitude_deg: 30.0,
            periapsis_argument_deg: 40.0,
            mean_anomaly_deg: 50.0,
        }
    }

    /// The equinoctial `(a, h, k, p, q, λ)` of a Keplerian orbit.
    fn equinoctial_of(k: &Keplerian) -> Vector6<f64> {
        let node = k.ascending_node_longitude_deg.to_radians();
        let peri = k.periapsis_argument_deg.to_radians();
        let tan_half_i = (k.inclination_deg.to_radians() / 2.0).tan();
        Vector6::new(
            k.semi_major_axis_au,
            k.eccentricity * (node + peri).sin(),
            k.eccentricity * (node + peri).cos(),
            tan_half_i * node.sin(),
            tan_half_i * node.cos(),
            node + peri + k.mean_anomaly_deg.to_radians(),
        )
    }

    fn diagonal(variance: f64) -> Vec<f64> {
        Matrix6::<f64>::identity()
            .iter()
            .map(|x| x * variance)
            .collect()
    }

    fn fit(covariance: Vec<f64>) -> FitCovariance {
        FitCovariance {
            elements: equinoctial_of(&test_orbit()),
            covariance,
            reference_epoch: 60_000.0,
            converged: true,
            differential_correction: true,
            normalised_rms: 0.87,
        }
    }

    #[test]
    fn a_converged_fit_with_a_usable_covariance_is_accepted() {
        let selected = select_covariance(&fit(diagonal(1e-8))).unwrap();
        assert_eq!(selected.epoch_mjd_tt, 60_000.0);
        assert!(
            selected.detail.contains("normalised RMS 0.87"),
            "{}",
            selected.detail
        );
    }

    #[test]
    fn a_fit_that_did_not_converge_is_refused_with_a_reason() {
        let mut f = fit(diagonal(1e-8));
        f.converged = false;
        let why = select_covariance(&f).err().unwrap();
        assert!(why.contains("did not converge"), "{why}");
    }

    #[test]
    fn an_iod_only_fit_is_refused_with_a_reason() {
        let mut f = fit(diagonal(1e-8));
        f.differential_correction = false;
        let why = select_covariance(&f).err().unwrap();
        assert!(why.contains("IOD"), "{why}");
    }

    #[test]
    fn a_missing_or_broken_covariance_is_refused_with_a_reason() {
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (Vec::new(), "missing"),
            (vec![1.0; 10], "malformed"),
            (vec![f64::NAN; 36], "non-finite"),
            (diagonal(0.0), "non-positive variance"),
            // Off-diagonal entries larger than the variances: an indefinite
            // matrix.
            (
                {
                    let mut m = diagonal(1.0);
                    m[1] = 5.0;
                    m[6] = 5.0;
                    m
                },
                "not positive semi-definite",
            ),
        ];
        for (covariance, expected) in cases {
            let why = select_covariance(&fit(covariance)).err().unwrap();
            assert!(why.contains(expected), "{why}");
            assert!(why.starts_with("the N-body fit's"), "{why}");
        }
    }

    #[test]
    fn a_near_singular_covariance_is_still_usable() {
        // One direction is ten orders of magnitude less constrained than the
        // rest — typical of a short arc, and a rounding-level negative
        // eigenvalue must not disqualify it.
        let mut m = diagonal(1e-8);
        m[35] = 1e-18;
        assert!(covariance_matrix(&m).is_ok());
    }

    #[test]
    fn samples_reproduce_the_requested_mean_and_covariance() {
        let mean = Vector6::new(1.0, -2.0, 0.5, 3.0, 0.0, 10.0);
        // Correlated covariance: A Aᵀ + diag, positive definite by
        // construction.
        let a = Matrix6::from_fn(|i, j| ((i * 6 + j) as f64 * 0.37).sin() * 0.5);
        let covariance = a * a.transpose() + Matrix6::identity() * 0.1;

        let n = 40_000;
        let samples = sample_multivariate_normal(&mean, &covariance, n, 7);
        assert_eq!(samples.len(), n);

        let empirical_mean = samples.iter().sum::<Vector6<f64>>() / n as f64;
        let empirical_cov = samples
            .iter()
            .map(|s| (s - empirical_mean) * (s - empirical_mean).transpose())
            .sum::<Matrix6<f64>>()
            / (n as f64 - 1.0);

        for i in 0..6 {
            assert!((empirical_mean[i] - mean[i]).abs() < 0.03, "mean {i}");
            for j in 0..6 {
                assert!(
                    (empirical_cov[(i, j)] - covariance[(i, j)]).abs() < 0.03,
                    "cov ({i},{j}): {} vs {}",
                    empirical_cov[(i, j)],
                    covariance[(i, j)]
                );
            }
        }
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed_and_differs_between_seeds() {
        let mean = Vector6::zeros();
        let cov = Matrix6::identity();
        let a = sample_multivariate_normal(&mean, &cov, 5, 42);
        let b = sample_multivariate_normal(&mean, &cov, 5, 42);
        let c = sample_multivariate_normal(&mean, &cov, 5, 43);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(seed_from_designation("FF1"), seed_from_designation("FF1"));
        assert_ne!(seed_from_designation("FF1"), seed_from_designation("FF2"));
    }

    #[test]
    fn a_covariance_with_a_slightly_negative_eigenvalue_still_samples() {
        let mut cov = Matrix6::identity();
        cov[(5, 5)] = -1e-12;
        let samples = sample_multivariate_normal(&Vector6::zeros(), &cov, 10, 1);
        assert_eq!(samples.len(), 10);
        assert!(samples.iter().all(|s| s.iter().all(|x| x.is_finite())));
        // The clipped direction has no spread.
        assert!(samples.iter().all(|s| s[5].abs() < 1e-9));
    }

    #[test]
    fn trim_outliers_drops_far_flung_and_non_finite_points() {
        let mut points: Vec<[f64; 3]> = (0..50)
            .map(|i| [2.0 + 0.01 * (i % 7) as f64, 1.0, 0.0])
            .collect();
        points.push([500.0, 0.0, 0.0]);
        points.push([f64::NAN, 0.0, 0.0]);
        points.push([30.0, 30.0, 0.0]);

        let kept = trim_outliers(&points, 4.0);

        assert_eq!(kept.len(), 50);
        assert!(kept.iter().all(|p| p[0] < 3.0));
    }

    #[test]
    fn trim_outliers_keeps_everything_when_the_cloud_has_no_spread() {
        let points = vec![[1.0, 2.0, 3.0]; 5];
        assert_eq!(trim_outliers(&points, 4.0).len(), 5);
        assert!(trim_outliers(&[], 4.0).is_empty());
    }

    #[test]
    fn a_fit_cloud_with_a_tiny_covariance_stays_on_the_best_orbit() {
        let selected = select_covariance(&fit(diagonal(1e-14))).unwrap();
        let best = test_orbit();
        let (last_obs, now) = (60_020.0, 60_400.0);

        let cloud = build_uncertainty_cloud(&selected, 3, last_obs, now).unwrap();

        assert_eq!(cloud.n_sampled, N_CLONES);
        assert_eq!(cloud.n_clones, N_CLONES);
        assert_eq!(cloud.at_last_observation.len(), N_CLONES);
        assert_eq!(cloud.n_kept_now, cloud.at_now.len());
        assert_eq!(cloud.orbits.len(), N_ORBIT_CLONES);
        assert!(cloud
            .orbits
            .iter()
            .all(|o| o.len() == UNCERTAINTY_ORBIT_SAMPLES));
        assert_eq!(cloud.best_orbit.len(), UNCERTAINTY_ORBIT_SAMPLES);

        let expected = geometry::position_at_epoch(&best, last_obs);
        assert!(geometry::distance(cloud.center_last_observation, expected) < 1e-6);
        assert!(
            geometry::distance(cloud.center_now, geometry::position_at_epoch(&best, now)) < 1e-6
        );
        for p in &cloud.at_last_observation {
            assert!(
                geometry::distance(*p, expected) < 1e-3,
                "{p:?} vs {expected:?}"
            );
        }
    }

    /// A larger covariance spreads the cloud: the clones' scatter at a later
    /// epoch is bigger than at an earlier one.
    #[test]
    fn uncertainty_grows_with_time_from_the_reference_epoch() {
        // Only the semi-major axis is uncertain: a wrong `a` means a wrong
        // mean motion, so the clones drift apart along the orbit.
        let mut covariance = diagonal(1e-14);
        covariance[0] = 1e-6;
        let selected = select_covariance(&fit(covariance)).unwrap();
        let cloud = build_uncertainty_cloud(&selected, 5, 60_005.0, 60_800.0).unwrap();

        let spread = |points: &[[f64; 3]]| {
            let center = [0, 1, 2].map(|a| median(points.iter().map(|p| p[a]).collect()));
            median(
                points
                    .iter()
                    .map(|p| geometry::distance(*p, center))
                    .collect(),
            )
        };
        assert!(spread(&cloud.at_now) > spread(&cloud.at_last_observation));
    }

    #[test]
    fn a_best_solution_that_is_not_an_ellipse_is_an_error() {
        let mut f = fit(diagonal(1e-14));
        // Eccentricity vector of norm 1.5: hyperbolic.
        f.elements[1] = 1.5;
        f.elements[2] = 0.0;
        let selected = select_covariance(&f).unwrap();
        assert!(build_uncertainty_cloud(&selected, 1, 60_010.0, 60_100.0).is_err());
    }
}
